import argparse
import gc
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch
from transformers import set_seed

from experiments.phase3.common import (
    format_description,
    PROJECT_ROOT, backbone, data_windows, full_validation, save_frozen, wrappers, write_json,
)
from experiments.phase3.postprocess import apply_records, load_static, reference_weights
from experiments.phase3.quantization import pack_int4, unpack_int4
from experiments.phase3.run import progress, source_record


@torch.no_grad()
def capture_module_inputs(model, name, windows):
    samples = []

    class InputCaptured(Exception):
        pass

    def capture(module, inputs):
        values = inputs[0].detach()
        samples.append(values.reshape(-1, values.shape[-1]).cpu())
        raise InputCaptured()

    model.cuda().eval()
    handle = model.get_submodule(name).module.register_forward_pre_hook(capture)
    try:
        for ids in windows:
            try:
                backbone(model, ids.cuda())
            except InputCaptured:
                pass
    finally:
        handle.remove()
    if len(samples) != len(windows):
        raise RuntimeError(f"Incomplete quantized input capture: {name}")
    boundary = len(samples) * 3 // 4
    return torch.cat(samples[:boundary]), torch.cat(samples[boundary:])


def selection_score(model, windows):
    result = full_validation(model, windows)
    if not math.isfinite(result["nll"]):
        raise RuntimeError("Non-finite train selection NLL")
    return result["nll"]


@torch.no_grad()
def score_candidate(model, windows, weights=None, alphas=None):
    weights = weights or {}
    alphas = alphas or {}
    previous_weights = {name: model.get_submodule(name).module.weight.detach().cpu().clone()
                        for name in weights}
    previous_scales = {name: model.get_submodule(name).quantizer.scale.detach().cpu().clone()
                       for name in alphas}
    try:
        apply_records(model, weights, alphas)
        return selection_score(model, windows)
    finally:
        for name, value in previous_weights.items():
            target = model.get_submodule(name).module.weight
            target.copy_(value.to(target))
        for name, value in previous_scales.items():
            target = model.get_submodule(name).quantizer.scale
            target.copy_(value.to(target))


def rounding_helper():
    path = PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py"
    specification = importlib.util.spec_from_file_location("phase3_sequential_rounding", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.coordinate_round


@torch.no_grad()
def round_module(model, name, record, reference, calibration, selection, current_score, args):
    fit, heldout = capture_module_inputs(model, name, calibration)
    progress(args.output, "sequential-round-fit", module=name, fit_rows=fit.shape[0], heldout_rows=heldout.shape[0])
    trials = rounding_helper()(reference.cuda(), record["scale"].cuda(), fit.cuda(), heldout.cuda(),
        milestones=tuple(args.milestones), initial_codes=unpack_int4(record["packed"], record["shape"]).cuda(),
        neighbor_search=True, progress=lambda step: progress(args.output, "sequential-round-coordinate", module=name, step=step)
        if step % 128 == 0 else None)
    del fit, heldout
    torch.cuda.empty_cache()
    best_score = current_score
    best_record = None
    best_step = 0
    trials[0]["train_nll"] = current_score
    for trial in trials[1:]:
        if trial["heldout_mse"] >= trials[0]["heldout_mse"]:
            trial["train_nll"] = None
            continue
        candidate = dict(record, packed=pack_int4(trial["codes"]))
        progress(args.output, "sequential-round-train-nll", module=name, step=trial["step"])
        score = score_candidate(model, selection, weights={name: candidate})
        trial["train_nll"] = score
        if score < best_score:
            best_score, best_record, best_step = score, candidate, trial["step"]
    if best_record is not None:
        apply_records(model, {name: best_record})
    return best_record, best_score, dict(module=name, selected_step=best_step,
        previous_train_nll=current_score, selected_train_nll=best_score,
        trials=[{key: value for key, value in trial.items() if key != "codes"} for trial in trials])


@torch.no_grad()
def range_module(model, name, selection, current_score, args):
    wrapper = model.get_submodule(name)
    parent_alpha = float(wrapper.quantizer.scale) * 127
    trials = [dict(factor=1.0, alpha=parent_alpha, train_nll=current_score, reused_prefix=True)]
    best = trials[0]
    for factor in args.alpha_factors:
        if factor == 1:
            continue
        alpha = parent_alpha * factor
        progress(args.output, "sequential-sp2-train-nll", module=name, factor=factor)
        score = score_candidate(model, selection, alphas={name: alpha})
        trial = dict(factor=factor, alpha=alpha, train_nll=score)
        trials.append(trial)
        if score < best["train_nll"]:
            best = trial
    if best["factor"] != 1:
        apply_records(model, {}, {name: best["alpha"]})
    return best["train_nll"], dict(module=name, selected_factor=best["factor"],
        previous_train_nll=current_score, selected_train_nll=best["train_nll"], trials=trials)


def save_prefix(model, updates, completed, targets, initial_score, current_score, args):
    state = dict(parent=str(args.parent), reference_state=str(args.reference_state), mode=args.mode,
        targets=targets, completed_targets=completed, weights=updates,
        activation_scales={name: wrapper.quantizer.scale.detach().cpu().clone()
                           for name, wrapper in wrappers(model).items() if name.endswith("down_proj")},
        initial_train_nll=initial_score, selected_train_nll=current_score)
    temporary = args.output / "prefix.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(args.output / "prefix.pt")


@torch.no_grad()
def restore_prefix(model, records, targets, args):
    state = torch.load(args.resume_prefix, map_location="cpu", weights_only=False)
    for name, expected in dict(parent=str(args.parent), reference_state=str(args.reference_state),
                               mode=args.mode, targets=targets).items():
        if state[name] != expected:
            raise ValueError(f"Sequential prefix changes {name}")
    completed = state["completed_targets"]
    if completed != targets[:len(completed)]:
        raise ValueError("Sequential prefix is not a completed target prefix")
    apply_records(model, state["weights"])
    for name, scale in state["activation_scales"].items():
        target = model.get_submodule(name).quantizer.scale
        target.copy_(scale.to(target))
    records.update(state["weights"])
    return state["weights"], completed, state["initial_train_nll"], state["selected_train_nll"]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    train_windows, _, _, validation, data = data_windows()
    indices = data["calibration_window_indices"]
    calibration = [train_windows[index:index + 1] for index in indices]
    selection = calibration[24:]
    data.update(calibration_length=2048, fit_window_indices=indices[:24], selection_window_indices=indices[24:],
                fit_rows=24 * 2048, heldout_rows=8 * 2048, selection_predicted_tokens=8 * 2047)
    write_json(args.output / "data.json", data)
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    shutil.copyfile(PROJECT_ROOT / "scripts/phase2/fixed_grid_rounding.py", snapshot / "fixed_grid_rounding.py")
    (snapshot / "tracked.diff").write_bytes(subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    parent_metadata = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False)["metadata"]
    write_json(args.output / "settings.json", dict(arguments={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, source=source_record(), data=data,
        parent_metadata=parent_metadata,
        selection="Greedy actual eight full train-window NLL; accept prefix before next module; no validation calibration",
        selection_precision="existing BF16 logits evaluator, log(float32 PPL), 16376 predicted train tokens",
        constraints="all static W4 scales and non-down SA fixed; no gradient training in this stage; parent provenance retained"))
    reference = reference_weights(args.reference_state) if args.mode == "round" else None
    model, records = load_static(args.parent)
    names = list(wrappers(model))
    targets = args.targets or [name for name in names if name.endswith("down_proj") == (args.family == "down")]
    if not targets or len(set(targets)) != len(targets) or any(name not in names for name in targets):
        raise ValueError("Invalid sequential target modules")
    if args.mode in ("sp2", "range") and any(not name.endswith("down_proj") for name in targets):
        raise ValueError("SP2 range search targets down inputs only")
    if args.resume_prefix is None:
        initial_score = selection_score(model, selection)
        current_score, updates, completed = initial_score, {}, []
        write_json(args.output / "initial_train_selection.json", dict(nll=initial_score, predicted_tokens=8 * 2047))
    else:
        updates, completed, initial_score, current_score = restore_prefix(model, records, targets, args)
    results = []
    started = time.monotonic()
    for name in targets[len(completed):]:
        if args.mode == "round":
            selected, current_score, result = round_module(model, name, records[name], reference[name],
                calibration, selection, current_score, args)
            if selected is not None:
                updates[name] = selected
                records[name] = selected
        else:
            current_score, result = range_module(model, name, selection, current_score, args)
        results.append(result)
        completed.append(name)
        write_json(args.output / (name + ".json"), result)
        save_prefix(model, updates, completed, targets, initial_score, current_score, args)
        progress(args.output, "sequential-module-completed", module=name, completed=len(completed),
                 total=len(targets), selected_train_nll=current_score, elapsed_seconds=time.monotonic() - started)
    outcome = dict(parent=str(args.parent), parent_metadata=parent_metadata, mode=args.mode, initial_train_nll=initial_score,
        selected_train_nll=current_score, completed_targets=completed, new_module_results=results,
        validation_status="NOT TESTED: no improvement over initial train selection NLL")
    if current_score < initial_score:
        package = args.output / "static_w4a8.pt"
        save_frozen(model, records, package, dict(parent=str(args.parent), mode="sequential-" + args.mode,
            reference_state=str(args.reference_state), parent_metadata=parent_metadata, offline_only=True))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = load_static(package)
        progress(args.output, "sequential-full-validation")
        measured = full_validation(model, validation)
        measured.update(parent=str(args.parent), parent_metadata=parent_metadata,
            method="sequential frozen-package postprocessing; no gradient training in this stage",
            format=format_description(model))
        write_json(args.output / "validation.json", measured)
        outcome.update(validation_status="completed", validation=measured)
    write_json(args.output / "result.json", outcome)
    progress(args.output, "completed", elapsed_seconds=time.monotonic() - started, result=outcome)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--resume-prefix", type=Path)
    parser.add_argument("--mode", choices=("round", "sp2", "range"), required=True)
    parser.add_argument("--family", choices=("down", "non-down"), default="down")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--milestones", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--alpha-factors", type=float, nargs="+", default=[1, 1.125, 1.25, 1.5, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.mode == "round" and args.reference_state is None:
        parser.error("Current-R FP reference state required for fixed-grid rounding")
    if args.milestones != sorted(set(args.milestones)) or min(args.milestones) < 1:
        parser.error("Rounding milestones must be positive, distinct and increasing")
    if any(not math.isfinite(factor) or factor <= 0 for factor in args.alpha_factors):
        parser.error("SP2 range factors must be positive and finite")
    set_seed(int(__import__("os").environ.get("PHASE5_SEED", "42")))
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
