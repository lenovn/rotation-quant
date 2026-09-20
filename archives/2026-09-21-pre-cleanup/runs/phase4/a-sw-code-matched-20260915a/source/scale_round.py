"""Matched continuation of Phase3 rounding, with optional per-row scale fits."""
import argparse
import gc
from pathlib import Path
import shutil
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch
from transformers import set_seed
from experiments.phase3.common import data_windows, full_validation, save_frozen, write_json
from experiments.phase3.postprocess import apply_records, load_static, reference_weights
from experiments.phase3.quantization import pack_int4, unpack_int4
from experiments.phase3.run import progress, source_record
from experiments.phase3.sequential_postprocess import capture_module_inputs, rounding_helper, selection_score, score_candidate


@torch.no_grad()
def fit_scale(codes, scale, reference, inputs):
    """Unrounded LS solution, accepted rowwise only if BF16-weight MSE improves."""
    if scale.shape != (codes.shape[0], 1):
        raise ValueError("Scale must have one value per output channel")
    q, w, x = codes.float(), reference.float(), inputs.float()
    # Gram avoids materializing all [49152, 2048] outputs for each candidate.
    h = x.T @ x / x.shape[0]
    qh = q @ h
    denominator = (qh * q).sum(1, keepdim=True)
    numerator = (qh * w).sum(1, keepdim=True)
    proposed = numerator / denominator.clamp_min(1e-30)
    valid = (denominator > 0) & torch.isfinite(proposed) & (proposed > 0)
    proposed = torch.where(valid, proposed, scale)
    old_error = (q * scale).to(reference.dtype).float() - w
    new_error = (q * proposed).to(reference.dtype).float() - w
    old_mse = ((old_error @ h) * old_error).sum(1, keepdim=True)
    new_mse = ((new_error @ h) * new_error).sum(1, keepdim=True)
    accepted = valid & (new_mse < old_mse)
    result = torch.where(accepted, proposed, scale)
    return result, dict(accepted_rows=int(accepted.sum()), total_rows=codes.shape[0],
        relative_change_max=float(((result / scale) - 1).abs().max()),
        ideal_scale_relative_change_median=float(((proposed / scale) - 1).abs().median()),
        fit_mse_before=float(old_mse.mean()),
        fit_mse_after=float(torch.where(accepted, new_mse, old_mse).mean()))


@torch.no_grad()
def mse(record, reference, inputs):
    codes = unpack_int4(record["packed"], record["shape"]).to(reference.device)
    weight = (codes.float() * record["scale"].to(reference.device)).to(reference.dtype).float()
    return float(((weight - reference.float()) @ inputs.float().T).square().mean())


@torch.no_grad()
def generate(record, reference, fit, heldout, arm, args, name):
    started = time.monotonic()
    current = record
    trials = []
    scale_only = None
    for cycle in range(2):
        scale_info = None
        if arm == "A2":
            codes = unpack_int4(current["packed"], current["shape"]).cuda()
            scale, scale_info = fit_scale(codes, current["scale"].cuda(), reference, fit)
            current = dict(current, scale=scale.cpu())
            if cycle == 0:
                scale_only = dict(record=current, fit_mse=mse(current, reference, fit),
                                  heldout_mse=mse(current, reference, heldout), scale_fit=scale_info)
        rounded = rounding_helper()(reference, current["scale"].cuda(), fit, heldout,
            milestones=(args.coordinates,),
            initial_codes=unpack_int4(current["packed"], current["shape"]).cuda(),
            neighbor_search=True,
            progress=lambda step: progress(args.output, "coordinate", arm=arm, module=name,
                                           cycle=cycle, step=step) if step % 512 == 0 else None)
        last = rounded[-1]
        current = dict(current, packed=pack_int4(last["codes"]))
        trials.append(dict(record=current, cycle=cycle + 1, scale_fit=scale_info,
            coordinates=last["step"], fit_mse=last["fit_mse"], heldout_mse=last["heldout_mse"],
            changed_codes=last["changed_codes"]))
    torch.cuda.synchronize()
    return trials, scale_only, time.monotonic() - started


def public(trial):
    return {key: value for key, value in trial.items() if key != "record"}


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    snapshot.mkdir()
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    train, _, _, validation, data = data_windows()
    indices = data["calibration_window_indices"]
    calibration = [train[i:i + 1] for i in indices]
    selection = calibration[24:]
    data.update(calibration_length=2048, fit_rows=49152, heldout_rows=16384,
                fit_window_indices=indices[:24], selection_window_indices=indices[24:])
    write_json(args.output / "settings.json", dict(arguments={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, source=source_record(), data=data,
        input_source="Both arms use identical captures from untouched parent, after input SP2",
        target="same-R unquantized BF16 weight times the same quantized input",
        budget="Each arm: 3 modules x 2 rounds x requested coordinates. A2 adds two closed-form scale fits per module. Actual coordinates and seconds recorded.",
        selection="Two candidate endpoints per arm/module, heldout MSE prefilter then eight train-window NLL; fixed parent captures; full validation only final selected packages"))
    reference = reference_weights(args.reference_state)
    model, records = load_static(args.parent)
    proposed, reports = {}, {}
    started = time.monotonic()
    for name in args.targets:
        progress(args.output, "capture-parent-input", module=name)
        fit, heldout = capture_module_inputs(model, name, calibration)
        fit, heldout, weight = fit.cuda(), heldout.cuda(), reference[name].cuda()
        before = dict(fit_mse=mse(records[name], weight, fit), heldout_mse=mse(records[name], weight, heldout))
        proposed[name], reports[name] = {}, dict(parent=before)
        for arm in ("A1", "A2"):
            trials, scale_only, elapsed = generate(records[name], weight, fit, heldout, arm, args, name)
            proposed[name][arm] = trials
            reports[name][arm] = dict(trials=[public(t) for t in trials], optimization_seconds=elapsed,
                                     scale_only=public(scale_only) if scale_only else None)
            if scale_only:
                torch.save(scale_only["record"], args.output / (name + ".scale_only.pt"))
            write_json(args.output / (name + ".json"), reports[name])
        del fit, heldout, weight
        torch.cuda.empty_cache()
    # Parent stays untouched throughout capture/generation. Selection starts clean per arm.
    initial_score = selection_score(model, selection)
    results = dict(parent=str(args.parent), initial_train_nll=initial_score, arms={})
    for arm in ("A1", "A2"):
        apply_records(model, {name: records[name] for name in args.targets})
        score, updates, choices = initial_score, {}, []
        for name in args.targets:
            best, scores = None, []
            for trial in proposed[name][arm]:
                if trial["heldout_mse"] >= reports[name]["parent"]["heldout_mse"]:
                    scores.append(dict(cycle=trial["cycle"], train_nll=None))
                    continue
                candidate_score = score_candidate(model, selection, weights={name: trial["record"]})
                scores.append(dict(cycle=trial["cycle"], train_nll=candidate_score))
                if candidate_score < score:
                    score, best = candidate_score, trial
            if best:
                updates[name] = best["record"]
                apply_records(model, {name: best["record"]})
            choices.append(dict(module=name, selected_cycle=best["cycle"] if best else 0, scores=scores))
        directory = args.output / arm
        directory.mkdir()
        result = dict(train_nll=score, choices=choices, changed_modules=list(updates))
        torch.save(updates, directory / "updates.pt")
        if updates:
            package = directory / "static_w4a8.pt"
            save_frozen(model, dict(records, **updates), package,
                dict(parent=str(args.parent), reference_state=str(args.reference_state), method="phase4-scale-round",
                     arm=arm, offline_only=True, changed_modules=list(updates)))
            # Mature loader/evaluator: evaluate the actual exported codes/scales.
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model, _ = load_static(package)
            progress(args.output, "full-validation", arm=arm)
            result["validation"] = full_validation(model, validation)
            write_json(directory / "validation.json", result["validation"])
        else:
            result["validation"] = dict(reused_parent=True, ppl=16.112577060930427, nll=2.779600150933802,
                                        predicted_tokens=252728)
        results["arms"][arm] = result
        write_json(directory / "result.json", result)
        write_json(args.output / "result.json", results)
    results.update(elapsed_seconds=time.monotonic() - started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    write_json(args.output / "result.json", results)
    progress(args.output, "completed", results=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--coordinates", type=int, default=2048)
    parser.add_argument("--targets", nargs="+", default=[f"model.layers.{i}.mlp.down_proj" for i in (1, 8, 11)])
    args = parser.parse_args()
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
