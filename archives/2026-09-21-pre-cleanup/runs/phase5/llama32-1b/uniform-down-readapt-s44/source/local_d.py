import argparse
import gc
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch
import torch.nn.functional as functional
from transformers import set_seed

from experiments.phase3.common import backbone, data_windows, full_validation, save_frozen, write_json
from experiments.phase3.postprocess import apply_records, dequant_record, load_static, reference_weights
from experiments.phase3.quantization import int4_codes, pack_int4, sp2_project
from experiments.phase3.run import progress, source_record
from experiments.phase3.sequential_postprocess import rounding_helper, score_candidate, selection_score


@torch.no_grad()
def capture_pair(model, layer, windows):
    prefix = f"model.layers.{layer}.mlp."
    captured = {name: [] for name in ("up_input", "gate_output", "down_input")}

    class PairCaptured(Exception):
        pass

    def capture_up(module, inputs):
        captured["up_input"].append(inputs[0].detach().cpu())

    def capture_gate(module, inputs, output):
        captured["gate_output"].append(output.detach().cpu())

    def capture_down(module, inputs):
        captured["down_input"].append(inputs[0].detach().cpu())
        raise PairCaptured()

    model.cuda().eval()
    handles = [model.get_submodule(prefix + "up_proj").module.register_forward_pre_hook(capture_up),
               model.get_submodule(prefix + "gate_proj").module.register_forward_hook(capture_gate),
               model.get_submodule(prefix + "down_proj").register_forward_pre_hook(capture_down)]
    try:
        for ids in windows:
            try:
                backbone(model, ids.cuda())
            except PairCaptured:
                pass
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != len(windows) for values in captured.values()):
        raise RuntimeError("Incomplete full-window local-D capture")
    return captured


def sparse_direction(activation_rms, down_weight):
    activation = activation_rms.double().clamp_min(1e-12)
    columns = down_weight.double().abs().amax(dim=0).clamp_min(1e-12)
    channel = int((activation / columns).argmax())
    direction = 0.5 * ((activation.log() - activation.log().mean())
                       - (columns.log() - columns.log().mean()))
    bound = math.log(4.0)
    lower, upper = direction.min() - bound, direction.max() + bound
    for iteration in range(70):
        center = (lower + upper) / 2
        if float((direction - center).clamp(-bound, bound).mean()) > 0:
            lower = center
        else:
            upper = center
    direction = (direction - (lower + upper) / 2).clamp(-bound, bound).float()
    sparse = torch.zeros_like(direction)
    sparse[channel] = direction[channel].clamp_min(0)
    sparse -= sparse.mean()
    return channel, sparse


def transformed_records(reference_up, reference_down, parent_up, parent_down, diagonal):
    if diagonal.ndim != 1 or diagonal.numel() != reference_up.shape[0] or diagonal.numel() != reference_down.shape[1]:
        raise ValueError("D must match the intermediate channels")
    if not torch.isfinite(diagonal).all() or not (diagonal > 0).all():
        raise ValueError("D must be finite and positive")
    diagonal = diagonal.to(reference_up.device)
    up_weight = (reference_up.float() / diagonal[:, None]).to(reference_up.dtype)
    down_weight = (reference_down.float() * diagonal[None, :]).to(reference_down.dtype)
    up_scale = parent_up["scale"].to(diagonal.device) / diagonal[:, None]
    down_scale = torch.maximum(parent_down["scale"].to(diagonal.device),
                               down_weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7)
    up = dict(parent_up, packed=pack_int4(int4_codes(up_weight, up_scale)).cpu(), scale=up_scale.cpu())
    down = dict(parent_down, packed=pack_int4(int4_codes(down_weight, down_scale)).cpu(), scale=down_scale.cpu())
    return up, down, down_weight


@torch.no_grad()
def candidate_inputs(captured, up_record, alpha, levels):
    samples = []
    weight = dequant_record(up_record, "cuda")
    for inputs, gate in zip(captured["up_input"], captured["gate_output"]):
        inputs = inputs.cuda()
        values = functional.silu(gate.cuda()) * functional.linear(inputs, weight)
        samples.append(sp2_project(values, alpha, levels.to(values.device)).reshape(-1, values.shape[-1]).cpu())
    boundary = len(samples) * 3 // 4
    return torch.cat(samples[:boundary]).cuda(), torch.cat(samples[boundary:]).cuda()


@torch.no_grad()
def search_pair(model, records, reference, calibration, selection, initial_score, args):
    prefix = f"model.layers.{args.layer}.mlp."
    up_name, down_name = prefix + "up_proj", prefix + "down_proj"
    captured = capture_pair(model, args.layer, calibration)
    fit_values = captured.pop("down_input")[:24]
    squared = [values.float().square().sum(dim=(0, 1)) for values in fit_values]
    activation_rms = (torch.stack(squared).sum(dim=0) / sum(values.shape[1] for values in fit_values)).sqrt()
    channel, direction = sparse_direction(activation_rms, reference[down_name])
    write_json(args.output / "channel_selection.json", dict(layer=args.layer, channel=channel,
        ranking="actual pre-SP2 RMS / current-R original FP down column absmax; first24 full train windows",
        activation_rms=activation_rms.tolist(), direction=direction.tolist()))
    del fit_values, squared
    alpha = float(model.get_submodule(down_name).quantizer.alpha)
    candidates = []
    for candidate_index, strength in enumerate(args.strengths):
        diagonal = (strength * direction).exp().cuda()
        up, down, transformed_down = transformed_records(reference[up_name].cuda(), reference[down_name].cuda(),
            records[up_name], records[down_name], diagonal)
        fit, heldout = candidate_inputs(captured, up, alpha, model.get_submodule(down_name).quantizer.levels)
        progress(args.output, "local-d-rounding", strength=strength, channel=channel,
                 fit_rows=fit.shape[0], heldout_rows=heldout.shape[0])
        trials = rounding_helper()(transformed_down, down["scale"].cuda(), fit, heldout,
            milestones=tuple(args.milestones), neighbor_search=True,
            progress=lambda step: progress(args.output, "local-d-coordinate", strength=strength, step=step)
            if step % 128 == 0 else None)
        del fit, heldout, transformed_down
        torch.cuda.empty_cache()
        chosen = None
        for trial in trials:
            weights = {up_name: up, down_name: dict(down, packed=pack_int4(trial["codes"]).cpu())}
            progress(args.output, "local-d-train-nll", strength=strength, step=trial["step"])
            trial["train_nll"] = score_candidate(model, selection, weights=weights)
            if chosen is None or trial["train_nll"] < chosen["train_nll"]:
                chosen = dict(weights=weights, train_nll=trial["train_nll"], step=trial["step"])
        report = dict(strength=strength, channel=channel, layer=args.layer,
            diagonal_min=float(diagonal.min()), diagonal_max=float(diagonal.max()),
            mean_log_diagonal=float(diagonal.double().log().mean()),
            selected_step=chosen["step"], selected_train_nll=chosen["train_nll"], parent_train_nll=initial_score,
            sp2_alpha=alpha, expanded_sw_rows=int((down["scale"] > records[down_name]["scale"]).sum()),
            trials=[{key: value for key, value in trial.items() if key != "codes"} for trial in trials])
        chosen.update(report=report, diagonal=diagonal.cpu())
        torch.save(chosen, args.output / f"candidate-{candidate_index:02d}.pt")
        write_json(args.output / f"candidate-{candidate_index:02d}.json", report)
        candidates.append(chosen)
    return candidates


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    train, _, _, validation, data = data_windows()
    indices = data["calibration_window_indices"]
    calibration = [train[index:index + 1] for index in indices]
    selection = calibration[24:]
    data.update(calibration_length=2048, fit_window_indices=indices[:24], selection_window_indices=indices[24:],
                fit_rows=49152, heldout_rows=16384, selection_predicted_tokens=16376)
    write_json(args.output / "data.json", data)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    from experiments.phase3.common import PROJECT_ROOT
    shutil.copyfile(PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py", snapshot / "fixed_grid_rounding.py")
    (snapshot / "tracked.diff").write_bytes(subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    parent_metadata = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False)["metadata"]
    write_json(args.output / "settings.json", dict(arguments={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, source=source_record(), parent_metadata=parent_metadata, data=data,
        selection="actual8 full train-window NLL across all D/rounding checkpoints, including identity rule",
        sw_up="parent SW / D; independently quantized transformed current-R FP up",
        sw_down="max(parent SW, transformed BF16 row absmax /7); independent RTN then neighbor rounding for each D",
        constraints="parent16 SP2 alphas and96 non-down SA fixed; gate/unrelated weights/HP unchanged; offline-only D",
        identity="D=I with identical SW rule and rounding search; not assumed equal to untouched parent"))
    started = time.monotonic()
    reference = reference_weights(args.reference_state)
    model, records = load_static(args.parent)
    initial_score = selection_score(model, selection)
    write_json(args.output / "initial_train_selection.json", dict(nll=initial_score, predicted_tokens=16376))
    candidates = search_pair(model, records, reference, calibration, selection, initial_score, args)
    identity = candidates[0]
    chosen = min(candidates, key=lambda candidate: candidate["train_nll"])
    result = dict(parent=str(args.parent), parent_metadata=parent_metadata, initial_train_nll=initial_score,
        identity_rule=identity["report"], selected=chosen["report"],
        validation_status="NOT TESTED: no improvement over untouched parent train selection NLL")
    if chosen["train_nll"] < initial_score:
        paths = {}
        for label, candidate in (("identity_rule", identity), ("selected_d", chosen)):
            if label == "selected_d" and chosen is identity:
                continue
            apply_records(model, candidate["weights"])
            merged = dict(records, **candidate["weights"])
            directory = args.output / label
            directory.mkdir()
            path = directory / "static_w4a8.pt"
            save_frozen(model, merged, path, dict(parent=str(args.parent), parent_metadata=parent_metadata,
                mode="local-d", reference_state=str(args.reference_state), offline_only=True,
                diagonal=dict(layer=args.layer, channel=candidate["report"]["channel"],
                    values=candidate["diagonal"].tolist(), strength=candidate["report"]["strength"]),
                selection=candidate["report"]))
            paths[label] = path
            apply_records(model, {name: records[name] for name in candidate["weights"]})
        del model, reference
        gc.collect()
        torch.cuda.empty_cache()
        measured = {}
        for label, path in paths.items():
            progress(args.output, "local-d-full-validation", label=label)
            frozen, _ = load_static(path)
            measured[label] = full_validation(frozen, validation)
            measured[label].update(parent=str(args.parent), parent_metadata=parent_metadata,
                method="offline paired local D with declared SW transport and train-selected discrete W4",
                format="112 W4 /96 static INT8 /16 static SP2; BF16 KV16 prefill")
            write_json(path.parent / "validation.json", measured[label])
            del frozen
            gc.collect()
            torch.cuda.empty_cache()
        if chosen is identity:
            measured["selected_d"] = dict(measured["identity_rule"], reused_identity=True)
        result.update(validation_status="completed", validation=measured)
    write_json(args.output / "result.json", result)
    progress(args.output, "completed", elapsed_seconds=time.monotonic() - started, result=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--milestones", type=int, nargs="+", default=[512, 2048, 8192])
    args = parser.parse_args()
    if args.layer < 0 or args.layer >= 16:
        parser.error("Layer must be within the sixteen-layer backbone")
    if args.strengths != sorted(set(args.strengths)) or args.strengths[0] != 0 or any(
            not math.isfinite(value) or value < 0 or value > 1 for value in args.strengths):
        parser.error("Strengths must start with identity and increase within [0,1]")
    if args.milestones != sorted(set(args.milestones)) or min(args.milestones) < 1:
        parser.error("Rounding milestones must be positive, distinct and increasing")
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error), time=time.time()))
        raise


if __name__ == "__main__":
    main()
