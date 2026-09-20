"""GuidedQuant g=1 objective adapted to fixed static-SP2 parent code continuation."""
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
from experiments.phase3.common import data_windows, full_validation, save_frozen, token_nll, wrappers, write_json
from experiments.phase3.postprocess import apply_records, load_static, reference_weights
from experiments.phase3.quantization import SP2ScaleSTE
from experiments.phase3.run import progress, source_record
from experiments.phase3.sequential_postprocess import capture_module_inputs, selection_score, score_candidate
from experiments.phase4.scale_round import generate, mse, public


def weighted_inputs(inputs, saliency, normalizer):
    if saliency.shape != (inputs.shape[0],) or not torch.isfinite(saliency).all() or (saliency < 0).any():
        raise ValueError("Expected finite nonnegative per-token saliency")
    if not torch.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Saliency has no positive finite fit mean")
    return inputs.float() * (saliency.to(inputs.device) / normalizer).sqrt()[:, None]


def capture_saliency(model, name, windows, output):
    """Freeze all parameters; differentiate CE through the fixed quantized suffix."""
    leaf = []
    values = []

    def capture(module, inputs, result):
        value = result.detach().requires_grad_(True)
        leaf.append(value)
        return value

    def sp2_ste(module, inputs, result):
        # Same forward values; use the established Phase3 clipped input STE.
        return SP2ScaleSTE.apply(inputs[0], module.scale, module.levels)

    handles = [model.get_submodule(name).module.register_forward_hook(capture)]
    handles += [wrapper.quantizer.register_forward_hook(sp2_ste)
                for n, wrapper in wrappers(model).items() if n.endswith("down_proj")]
    model.cuda().eval().requires_grad_(False)
    try:
        for index, ids in enumerate(windows):
            leaf.clear()
            with torch.enable_grad():
                loss = token_nll(model, ids.cuda())
                if len(leaf) != 1:
                    raise RuntimeError("Expected one target output per full window")
                gradient, = torch.autograd.grad(loss * 1000, leaf[0])
            saliency = gradient.float().square().mean(-1).reshape(-1)
            values.append(saliency.cpu())
            if index % 4 == 0:
                progress(output, "saliency", window=index + 1, windows=len(windows), nll=float(loss))
            del gradient, loss
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(values)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ("guided_round.py", "scale_round.py"):
        shutil.copyfile(Path(__file__).parent / name, source / name)
    train, _, _, validation, data = data_windows()
    indices = data["calibration_window_indices"]
    calibration = [train[i:i+1] for i in indices]
    selection = calibration[24:]
    data.update(calibration_length=2048, fit_rows=49152, heldout_rows=16384)
    write_json(args.output / "settings.json", dict(arguments={k: str(v) if isinstance(v, Path) else v
        for k,v in vars(args).items()}, source=source_record(), data=data,
        paper="GuidedQuant 2505.07004 v2 section 3.2; g=1 as weight-and-activation setting",
        target="same-R unquantized Wref times unchanged actual parent SP2 X",
        gradients="current quantized parent CE, frozen parameters; existing INT8 STE and Phase3 SP2ScaleSTE; multiply loss by 1000 before backward",
        objective="H = X.T diag(mean_output_channels(grad_CE**2)) X; normalize by fit mean only",
        updates="only target signed INT4 codes; fixed all SW/SA/SP2/R and high precision parameters",
        budget="same 2x2048 coordinates as layer8 A1; additional 32 full-window saliency backwards explicitly charged",
        selection="choose one of two endpoints using 8 train-window NLL, then full validation even if parent remains better"))
    started = time.monotonic()
    reference = reference_weights(args.reference_state)[args.target].cuda()
    model, records = load_static(args.parent)
    fit, heldout = capture_module_inputs(model, args.target, calibration)
    capture_started = time.monotonic()
    saliency = capture_saliency(model, args.target, calibration, args.output)
    normalizer = saliency[:fit.shape[0]].mean()
    torch.save(saliency, args.output / "saliency.pt")
    normalized = saliency / normalizer
    write_json(args.output / "saliency.json", dict(seconds=time.monotonic()-capture_started,
        fit_mean=float(normalizer), zero_tokens=int((saliency == 0).sum()),
        normalized_quantiles=torch.quantile(normalized, torch.tensor([0., .5, .9, .99, 1.])).tolist(),
        effective_fit_rows=float(normalized[:fit.shape[0]].sum().square()/normalized[:fit.shape[0]].square().sum())))
    fit, heldout = fit.cuda(), heldout.cuda()
    wf = weighted_inputs(fit, saliency[:fit.shape[0]], normalizer.cuda())
    wh = weighted_inputs(heldout, saliency[fit.shape[0]:], normalizer.cuda())
    before = dict(fit_mse=mse(records[args.target], reference, fit), heldout_mse=mse(records[args.target], reference, heldout),
                  weighted_fit_mse=mse(records[args.target], reference, wf), weighted_heldout_mse=mse(records[args.target], reference, wh))
    trials, _, elapsed = generate(records[args.target], reference, wf, wh, "A1", args, args.target)
    for trial in trials:
        trial["uniform_fit_mse"] = mse(trial["record"], reference, fit)
        trial["uniform_heldout_mse"] = mse(trial["record"], reference, heldout)
    del fit, heldout, wf, wh, reference
    torch.cuda.empty_cache()
    initial = selection_score(model, selection)
    for trial in trials:
        trial["train_nll"] = score_candidate(model, selection, weights={args.target: trial["record"]})
    best = min(trials, key=lambda t: t["train_nll"])
    report = dict(parent=before, initial_train_nll=initial, selected_train_nll=best["train_nll"],
        train_accepted=best["train_nll"] < initial, selected_cycle=best["cycle"],
        optimization_seconds=elapsed, trials=[public(t) for t in trials])
    write_json(args.output / "selection.json", report)
    torch.save([trial["record"] for trial in trials], args.output / "candidate_records.pt")
    apply_records(model, {args.target: best["record"]})
    package = args.output / "static_w4a8.pt"
    save_frozen(model, dict(records, **{args.target: best["record"]}), package,
                dict(parent=str(args.parent), reference_state=str(args.reference_state),
                     method="GuidedQuant-g1-objective-existing-neighbor", target=args.target, offline_only=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = load_static(package)
    progress(args.output, "full-validation")
    measured = full_validation(model, validation)
    write_json(args.output / "validation.json", measured)
    write_json(args.output / "result.json", dict(selection=report, validation=measured,
        elapsed_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    progress(args.output, "completed", validation=measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--target", default="model.layers.8.mlp.down_proj")
    parser.add_argument("--coordinates", type=int, default=2048)
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
