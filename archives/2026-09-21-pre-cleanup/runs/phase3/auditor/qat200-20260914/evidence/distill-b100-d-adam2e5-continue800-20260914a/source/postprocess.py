import argparse
import gc
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch
import torch.nn.functional as functional
from transformers import AutoConfig

from experiments.phase3.common import (
    MODEL_PATH, PROJECT_ROOT, backbone, build_training_model, data_windows, evaluate,
    full_validation, reload_frozen, rotated_linears, save_frozen, wrappers, write_json,
)
from experiments.phase3.quantization import SP2Quantizer, int4_codes, pack_int4, sp2_project, unpack_int4
from experiments.phase3.run import progress, source_record
from utils.quant_utils import RotationStaticActQuantizer, add_actquant


def load_static(path, device="cuda"):
    from eval_utils.modeling_llama import LlamaForCausalLM

    config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(str(MODEL_PATH), config=config,
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    add_actquant(model)
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer = SP2Quantizer(1.0) if name.endswith("down_proj") else RotationStaticActQuantizer()
    model.requires_grad_(False).eval().to(device)
    model.config.use_cache = False
    state = reload_frozen(model, path)
    return model, state["weights"]


@torch.no_grad()
def reference_weights(path):
    model = build_training_model(path).cuda().eval()
    weights = {}
    for name, wrapper, rotation2, transpose in rotated_linears(model):
        weights[name] = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose).cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return weights


@torch.no_grad()
def capture_inputs(model, calibration):
    samples = {name: dict(quantized=[]) for name in wrappers(model)}
    handles = []

    def capture(name, key, use_output=False):
        def hook(module, inputs, output=None):
            values = output if use_output else inputs[0]
            values = values.detach().reshape(-1, values.shape[-1])
            samples[name].setdefault(key, []).append(values[::8].cpu())
        return hook

    for name, wrapper in wrappers(model).items():
        handles.append(wrapper.module.register_forward_pre_hook(capture(name, "quantized")))
        if name.endswith("down_proj"):
            handles.append(wrapper.register_forward_pre_hook(capture(name, "raw")))
        if name.endswith("gate_proj"):
            handles.append(wrapper.register_forward_hook(capture(name, "output", True)))
    try:
        for ids in calibration:
            backbone(model, ids.to(next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return {name: {key: torch.cat(values) for key, values in record.items()}
            for name, record in samples.items()}


def row_mse(actual, reference):
    return (actual.float() - reference.float()).square().mean(dim=1)


def split_mse(errors):
    boundary = errors.numel() * 3 // 4
    return dict(fit_mse=float(errors[:boundary].mean()), heldout_mse=float(errors[boundary:].mean()))


def dequant_record(record, device, dtype=torch.bfloat16):
    codes = unpack_int4(record["packed"], record["shape"]).to(device)
    return (codes.float() * record["scale"].to(device)).to(dtype)


@torch.no_grad()
def apply_records(model, records, alphas=None):
    for name, record in records.items():
        weight = model.get_submodule(name).module.weight
        weight.copy_(dequant_record(record, weight.device, weight.dtype))
    for name, alpha in (alphas or {}).items():
        model.get_submodule(name).quantizer.scale.fill_(alpha / 127)


@torch.no_grad()
def rounding_candidates(model, records, reference, samples, args):
    specification = importlib.util.spec_from_file_location("phase3_existing_rounding",
        PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    device = next(model.parameters()).device
    ranking = []
    for name, wrapper in wrappers(model).items():
        inputs = samples[name]["quantized"].to(device).float()
        target = functional.linear(inputs, reference[name].to(device).float())
        actual = functional.linear(inputs, wrapper.module.weight.float())
        errors = split_mse(row_mse(actual, target))
        ranking.append(dict(name=name, normalized_mse=float(row_mse(actual, target).mean() /
            target.square().mean().clamp_min(1e-20)), **errors))
    ranking.sort(key=lambda record: record["normalized_mse"], reverse=True)
    write_json(args.output / "weight_ranking.json", ranking)
    proposals = {}
    diagnostics = {}
    for index, ranked in enumerate(ranking[:args.top_modules]):
        name = ranked["name"]
        inputs = samples[name]["quantized"].to(device)
        boundary = inputs.shape[0] * 3 // 4
        trials = module.coordinate_round(reference[name].to(device), records[name]["scale"].to(device),
            inputs[:boundary], inputs[boundary:], milestones=(8, 32, 128),
            initial_codes=unpack_int4(records[name]["packed"], records[name]["shape"]).to(device),
            neighbor_search=True)
        best = min(trials, key=lambda result: result["heldout_mse"])
        diagnostics[name] = dict(selected_step=best["step"],
            trials=[{key: value for key, value in trial.items() if key != "codes"} for trial in trials])
        if best["step"] != 0 and best["heldout_mse"] < trials[0]["heldout_mse"]:
            proposals[name] = dict(records[name], packed=pack_int4(best["codes"]).cpu())
        progress(args.output, "rounding", completed=index + 1, total=min(args.top_modules, len(ranking)))
        write_json(args.output / "rounding.json", diagnostics)
    candidates = [("round-top4", {name: proposals[name] for name in [row["name"] for row in ranking[:4]]
                                 if name in proposals}, {})]
    candidates.append(("round-top-selected", proposals, {}))
    return candidates


@torch.no_grad()
def range_candidates(model, records, samples, args):
    proposals = {}
    diagnostics = {}
    device = next(model.parameters()).device
    for name, wrapper in wrappers(model).items():
        if not name.endswith("down_proj"):
            continue
        values = samples[name]["raw"].to(device)
        weight = wrapper.module.weight.float()
        target = functional.linear(values.float(), weight)
        parent = float(wrapper.quantizer.alpha)
        trials = []
        for exponent in torch.linspace(-4, 4, 33).tolist():
            alpha = parent * 2 ** exponent
            projected = sp2_project(values, alpha, wrapper.quantizer.levels)
            actual = functional.linear(projected.float(), weight)
            trials.append(dict(alpha=alpha, exponent=exponent, **split_mse(row_mse(actual, target))))
        best = min(trials, key=lambda record: record["heldout_mse"])
        original = next(record for record in trials if record["exponent"] == 0)
        diagnostics[name] = dict(parent=original, selected=best, trials=trials)
        if best["heldout_mse"] < original["heldout_mse"]:
            proposals[name] = best["alpha"]
    write_json(args.output / "sp2_ranges.json", diagnostics)
    return [("sp2-ranges", {}, proposals)]


def fused_d_records(up_record, down_record, reference_down, diagonal):
    if diagonal.ndim != 1 or diagonal.numel() != up_record["shape"][0]:
        raise ValueError("D must have one positive entry per intermediate channel")
    if not torch.isfinite(diagonal).all() or not (diagonal > 0).all():
        raise ValueError("D must be finite and positive")
    device = reference_down.device
    diagonal = diagonal.to(device)
    changed = diagonal != 1
    up = dict(up_record, scale=up_record["scale"].to(device) / diagonal[:, None])
    codes = unpack_int4(down_record["packed"], down_record["shape"]).to(device)
    codes[:, changed] = int4_codes(reference_down[:, changed] * diagonal[changed],
                                    down_record["scale"].to(device))
    down = dict(down_record, packed=pack_int4(codes).cpu())
    up["scale"] = up["scale"].cpu()
    return up, down


@torch.no_grad()
def pair_range_trials(inputs, gate, up, down, target, parent_alpha, levels):
    values = gate * functional.linear(inputs, dequant_record(up, inputs.device, inputs.dtype))
    weight = dequant_record(down, inputs.device, inputs.dtype)
    trials = []
    for exponent in range(-8, 5):
        alpha = parent_alpha * 2 ** exponent
        actual = functional.linear(sp2_project(values, alpha, levels), weight)
        trials.append(dict(alpha=alpha, range_exponent=exponent, **split_mse(row_mse(actual, target))))
    return trials


@torch.no_grad()
def diagonal_candidates(model, records, reference, samples, args):
    device = next(model.parameters()).device
    rankings = []
    for name, wrapper in wrappers(model).items():
        if not name.endswith("down_proj"):
            continue
        values = samples[name]["raw"].to(device)
        projected = wrapper.quantizer(values)
        weight = reference[name].to(device).float()
        activation_contributions = ((projected.float() - values.float()).square().mean(dim=0)
                                    * weight.square().sum(dim=0))
        weight_contributions = (projected.float().square().mean(dim=0)
                                * (wrapper.module.weight.float() - weight).square().sum(dim=0))
        contributions = activation_contributions + weight_contributions
        order = contributions.argsort(descending=True)
        rankings.append(dict(name=name, score=float(contributions.sum()),
            channels=order.cpu().tolist(), channel_contributions=contributions.cpu().tolist(),
            activation_contributions=activation_contributions.cpu().tolist(),
            weight_contributions=weight_contributions.cpu().tolist()))
    rankings.sort(key=lambda record: record["score"], reverse=True)
    write_json(args.output / "diagonal_ranking.json", rankings)
    proposals = []
    diagnostics = {}
    for ranking in rankings[:4]:
        down_name = ranking["name"]
        up_name = down_name.replace("down_proj", "up_proj")
        gate_name = down_name.replace("down_proj", "gate_proj")
        inputs = samples[up_name]["quantized"].to(device)
        gate = functional.silu(samples[gate_name]["output"].to(device))
        wrapper = model.get_submodule(down_name)
        parent_alpha = float(wrapper.quantizer.alpha)
        reference_down = reference[down_name].to(device)
        target_up = functional.linear(inputs, reference[up_name].to(device))
        target = functional.linear(gate * target_up, reference_down)
        identity_trials = [dict(factor=1.0, channels=[], **trial) for trial in pair_range_trials(
            inputs, gate, records[up_name], records[down_name], target, parent_alpha, wrapper.quantizer.levels)]
        original = next(trial for trial in identity_trials if trial["range_exponent"] == 0)
        best_identity = min(identity_trials, key=lambda trial: trial["heldout_mse"])
        trials = identity_trials.copy()
        selected = best_identity
        best_records = None
        for count in (1, 4, 16):
            channels = ranking["channels"][:count]
            for factor in (0.25, 0.5, 2.0, 4.0, 16.0, 64.0, 256.0):
                diagonal = torch.ones(records[up_name]["shape"][0], device=device, dtype=torch.float32)
                diagonal[channels] = factor
                up, down = fused_d_records(records[up_name], records[down_name],
                    reference_down, diagonal)
                for measured in pair_range_trials(inputs, gate, up, down, target,
                                                   parent_alpha, wrapper.quantizer.levels):
                    trial = dict(factor=factor, channels=channels, **measured)
                    trials.append(trial)
                    if trial["heldout_mse"] < selected["heldout_mse"]:
                        selected = trial
                        best_records = {up_name: up, down_name: down}
        diagnostics[down_name] = dict(parent=original, best_identity=best_identity, selected=selected, trials=trials,
            parent_sp2_alpha=parent_alpha,
            control="D=I keeps parent codes/SW; both identity and nonidentity search the same SP2 range multipliers")
        if best_identity["heldout_mse"] < original["heldout_mse"]:
            proposals.append(("range-only-" + down_name, {}, {down_name: best_identity["alpha"]}))
        if best_records is not None:
            proposals.append(("local-d-" + down_name, best_records, {down_name: selected["alpha"]}))
        write_json(args.output / "diagonal_candidates.json", diagnostics)
        progress(args.output, "diagonal-search", layer=down_name, selected=selected)
    return proposals


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, calibration, probe, validation, data = data_windows(args.output)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    if args.mode == "round":
        shutil.copyfile(PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py",
                        snapshot / "fixed_grid_rounding.py")
    (snapshot / "tracked.diff").write_bytes(subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    write_json(args.output / "settings.json", dict(arguments={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, source=source_record(), data=data,
        selection="train-only local fit/heldout and fixed train probe; no validation calibration",
        reference_state=str(args.reference_state), parent=str(args.parent)))
    reference = reference_weights(args.reference_state) if args.mode != "sp2" else None
    model, records = load_static(args.parent)
    original_scales = {name: wrapper.quantizer.scale.clone() for name, wrapper in wrappers(model).items()
                       if name.endswith("down_proj")}
    parent_probe = evaluate(model, probe)
    write_json(args.output / "parent_probe.json", parent_probe)
    progress(args.output, "capture-train-inputs", mode=args.mode)
    samples = capture_inputs(model, calibration)
    if args.mode == "round":
        candidates = rounding_candidates(model, records, reference, samples, args)
    elif args.mode == "sp2":
        candidates = range_candidates(model, records, samples, args)
    else:
        candidates = diagonal_candidates(model, records, reference, samples, args)
    best_nll = parent_probe["nll"]
    best = None
    measurements = []
    for label, weights, alphas in candidates:
        if not weights and not alphas:
            continue
        try:
            apply_records(model, weights, alphas)
            measured = evaluate(model, probe)
        finally:
            apply_records(model, {name: records[name] for name in weights})
            for name, scale in original_scales.items():
                model.get_submodule(name).quantizer.scale.copy_(scale)
        measurements.append(dict(candidate=label, changed_modules=list(weights),
            changed_alphas=alphas, probe=measured))
        if measured["nll"] < best_nll:
            best_nll = measured["nll"]
            best = (label, weights, alphas)
        write_json(args.output / "candidate_probes.json", measurements)
    result = dict(parent=str(args.parent), parent_probe=parent_probe["nll"], best_probe=best_nll,
                  selected=None, validation_status="NOT TESTED: no candidate improved the fixed train probe")
    if best is not None:
        label, weights, alphas = best
        apply_records(model, weights, alphas)
        final_records = dict(records, **weights)
        package = args.output / "static_w4a8.pt"
        save_frozen(model, final_records, package, dict(parent=str(args.parent), mode=args.mode,
            candidate=label, reference_state=str(args.reference_state), offline_only=True))
        progress(args.output, "full-validation", candidate=label)
        measured = full_validation(model, validation)
        measured.update(format="112 W4 / 96 static INT8 / 16 static SP2; BF16 KV16 prefill",
                        parent=str(args.parent), candidate=label, source_root=str(SOURCE_ROOT))
        write_json(args.output / "validation.json", measured)
        result.update(selected=label, validation_status="completed", validation=measured)
    write_json(args.output / "result.json", result)
    progress(args.output, "completed", result=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--mode", choices=("round", "sp2", "d"), required=True)
    parser.add_argument("--top-modules", type=int, default=16)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "failure.json", dict(type=type(error).__name__,
                message=str(error), time=time.time()))
        raise


if __name__ == "__main__":
    main()
