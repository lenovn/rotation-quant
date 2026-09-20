import argparse
import json
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
from transformers import LlamaTokenizerFast, set_seed

from experiments.phase3.common import DATA_PATH, MODEL_PATH, backbone, wrappers, write_json
from experiments.phase3.postprocess import load_static
from experiments.phase3.quantization import SP2Quantizer
from experiments.phase3.run import progress, source_record
from utils.quant_utils import RotationStaticActQuantizer


def load_calibration(metadata_path):
    from datasets import Dataset

    original = json.loads(Path(metadata_path).read_text())
    if (Path(original["model_path"]).resolve() != MODEL_PATH.resolve()
            or Path(original["dataset_cache"]).resolve() != DATA_PATH.resolve()):
        raise ValueError("Calibration model or data path differs from B100")
    from experiments.phase3.architecture import tokenizer as model_tokenizer
    tokenizer = model_tokenizer(MODEL_PATH)
    dataset = Dataset.from_file(str(DATA_PATH / "wikitext-train.arrow"))
    rows = tokenizer(dataset["text"], add_special_tokens=False)["input_ids"]
    ids = torch.tensor([token for row in rows for token in row], dtype=torch.long)
    count = ids.numel() // 2048
    if ids.numel() != original["train_tokens"] or count != original["train_windows"]:
        raise ValueError("Calibration training tokens differ from B100")
    length = original["calibration_length"]
    indices = original["calibration_window_indices"]
    if not 0 < length <= 2048 or not indices or any(not 0 <= index < count for index in indices):
        raise ValueError("Invalid calibration window selection")
    windows = ids[:count * 2048].reshape(count, 2048)
    calibration = [windows[index:index + 1, :length] for index in indices]
    metadata = dict(dataset="Salesforce/wikitext", subset="wikitext-2-raw-v1", split="train",
        dataset_path=str(DATA_PATH / "wikitext-train.arrow"), tokenizer_path=str(MODEL_PATH),
        source_metadata_path=str(Path(metadata_path).resolve()), seed=original["seed"],
        train_tokens=ids.numel(), train_windows=count, calibration_window_indices=indices,
        calibration_length=length, calibration_tokens=sum(window.numel() for window in calibration),
        add_bos_token=False, add_eos_token=False, protocol=original["train_protocol"])
    return calibration, metadata


@torch.no_grad()
def capture_down_inputs(model, calibration):
    device = next(model.parameters()).device
    quantizers = wrappers(model)
    downs = {name: wrapper for name, wrapper in quantizers.items() if name.endswith("down_proj")}
    samples = {name: [] for name in downs}
    maximum = {name: 0.0 for name in downs}
    original_bits = {name: wrapper.quantizer.bits for name, wrapper in downs.items()}
    original_value_weights = {}
    handles = []

    def capture(name):
        def hook(module, inputs):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            maximum[name] = max(maximum[name], float(values.abs().max()))
            samples[name].append(values[::8].cpu())
        return hook

    try:
        for name, wrapper in quantizers.items():
            if name.endswith("self_attn.v_proj"):
                original_value_weights[name] = wrapper.module.weight.data
                wrapper.module.weight.data = wrapper.module.weight.data.t().contiguous().t()
        for name, wrapper in downs.items():
            wrapper.quantizer.bits = 16
            handles.append(wrapper.register_forward_pre_hook(capture(name)))
        for ids in calibration:
            backbone(model, ids.to(device))
    finally:
        for handle in handles:
            handle.remove()
        for name, wrapper in downs.items():
            wrapper.quantizer.bits = original_bits[name]
        for name, weight in original_value_weights.items():
            quantizers[name].module.weight.data = weight
    return {name: torch.cat(values) for name, values in samples.items()}, maximum


@torch.no_grad()
def search_int8_range(values, weight, bound):
    reference = values.float()
    weight = weight.float()
    bound = max(float(bound), 1e-8)
    quantizer = RotationStaticActQuantizer().to(values.device)
    candidates = []

    def measure(log_ratio):
        alpha = bound * 2 ** log_ratio
        quantizer.load_scale(torch.tensor([alpha / 127], device=values.device, dtype=torch.float32))
        difference = quantizer(values).float() - reference
        mse = float(functional.linear(difference, weight).square().mean())
        candidates.append(dict(alpha=alpha, scale=float(quantizer.scale.item()),
                               output_mse=mse, log2_ratio=log_ratio))
        return mse

    coarse = torch.linspace(-16.0, 1.0, 33).tolist()
    losses = [measure(log_ratio) for log_ratio in coarse]
    best = min(range(len(losses)), key=losses.__getitem__)
    refined = torch.linspace(coarse[max(0, best - 1)], coarse[min(32, best + 1)], 17).tolist()
    for log_ratio in refined:
        measure(log_ratio)
    selected = min(candidates, key=lambda record: record["output_mse"])
    return dict(selected=selected, full_absmax=bound, candidates=candidates, sampled_rows=values.shape[0])


def apply_down_int8(model, scale_path, package):
    state = torch.load(scale_path, map_location="cpu", weights_only=True)
    if Path(state["parent"]).resolve() != Path(package).resolve():
        raise ValueError("INT8 overlay parent differs from evaluated package")
    if (state["calibration_metadata"]["dataset"], state["calibration_metadata"]["split"]) != (
            "Salesforce/wikitext", "train"):
        raise ValueError("INT8 overlay must use WikiText training calibration")
    downs = {name: wrapper for name, wrapper in wrappers(model).items() if name.endswith("down_proj")}
    if len(downs) != model.config.num_hidden_layers or set(state["scales"]) != set(downs):
        raise ValueError("INT8 overlay must cover all model down inputs")
    for name, wrapper in downs.items():
        scale = state["scales"][name]
        if (not isinstance(wrapper.quantizer, SP2Quantizer) or scale.shape != (1,)
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError(f"Invalid matched INT8 overlay: {name}")
    for name, wrapper in downs.items():
        quantizer = RotationStaticActQuantizer().to(wrapper.module.weight.device)
        quantizer.load_scale(state["scales"][name])
        wrapper.quantizer = quantizer
    return dict(path=str(Path(scale_path).resolve()), parent=state["parent"], down_format="int8",
                calibration_metadata=state["calibration_metadata"])


@torch.no_grad()
def run(args):
    started = time.monotonic()
    calibration, metadata = load_calibration(args.calibration_data)
    write_json(args.output / "data.json", dict(calibration=True, training=False, c4_used=False, metadata=metadata))
    expected = json.loads(args.sp2_calibration.read_text())
    progress(args.output, "loading-b100-for-train-only-calibration")
    model, records = load_static(args.parent)
    quantizers = wrappers(model)
    before = {name: {key: value.detach().cpu().clone() for key, value in wrapper.quantizer.state_dict().items()}
              for name, wrapper in quantizers.items()}
    torch.cuda.reset_peak_memory_stats()
    samples, maximum = capture_down_inputs(model, calibration)
    if set(samples) != set(expected):
        raise ValueError("Captured down layers differ from B100 calibration")
    write_json(args.output / "capture_metadata.json", {
        name: dict(sampled_rows=values.shape[0], full_absmax=maximum[name],
                   historical_sampled_rows=expected[name]["sampled_rows"],
                   historical_full_absmax=expected[name]["full_absmax"],
                   historical_candidates=len(expected[name]["candidates"]))
        for name, values in samples.items()})
    results, scales = {}, {}
    for name, wrapper in quantizers.items():
        if name not in samples:
            continue
        values = samples.pop(name).to(wrapper.module.weight.device)
        previous = expected[name]
        if (values.shape[0] != previous["sampled_rows"] or len(previous["candidates"]) != 50
                or not math.isclose(maximum[name], previous["full_absmax"], rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError(f"B100 calibration inputs or search budget differ: {name}")
        progress(args.output, "search-uniform-int8", layer=name)
        result = search_int8_range(values, wrapper.module.weight, maximum[name])
        result["historical_sp2_selected"] = previous["selected"]
        result["historical_sp2_absmax"] = previous["full_absmax"]
        results[name] = result
        scales[name] = torch.tensor([result["selected"]["scale"]], dtype=torch.float32)
        write_json(args.output / "range_search.json", results)
    after = {name: {key: value.detach().cpu().clone() for key, value in wrapper.quantizer.state_dict().items()}
             for name, wrapper in quantizers.items()}
    if not all(torch.equal(value, after[name][key]) for name, record in before.items() for key, value in record.items()):
        raise ValueError("Original frozen quantizer state changed during INT8 calibration")
    torch.save(before, args.output / "quantizers_before.pt")
    torch.save(after, args.output / "quantizers_after.pt")
    torch.save(dict(parent=str(args.parent), calibration_metadata=metadata, scales=scales),
               args.output / "down_int8_scales.pt")
    result = dict(parent=str(args.parent), calibration_metadata=metadata, down_layers=len(scales),
        candidates_per_layer=50, calibration=True, training=False, c4_used=False,
        capture_layout="original frozen_model column-major v_proj; original coldload storage restored afterward",
        parent_scales_unchanged=True, historical_capture_matched=True,
        elapsed_seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    if getattr(args, "export_package", False):
        from experiments.phase3.common import save_frozen
        apply_down_int8(model, args.output / "down_int8_scales.pt", args.parent)
        package = args.output / "static_w4a8.pt"
        save_frozen(model, records, package, dict(parent=str(args.parent),
            method="matched Joint100 down INT8 initialization; no postprocessing or QAT yet"))
        result["package"] = str(package)
    write_json(args.output / "result.json", result)
    progress(args.output, "completed", down_layers=len(scales), elapsed_seconds=result["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--sp2-calibration", type=Path, required=True)
    parser.add_argument("--export-package", action="store_true")
    args = parser.parse_args()
    args.parent = args.parent.resolve()
    args.calibration_data = args.calibration_data.resolve()
    args.sp2_calibration = args.sp2_calibration.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    (source / "tracked.diff").write_text(subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "diff", "HEAD"], text=True))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    try:
        write_json(args.output / "settings.json", dict(arguments={name: str(value) for name, value in vars(args).items()},
                                                     source=source_record()))
        run(args)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
