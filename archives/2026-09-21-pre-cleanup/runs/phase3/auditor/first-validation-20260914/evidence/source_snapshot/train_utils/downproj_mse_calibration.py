import json
from pathlib import Path

import torch

from utils.quant_utils import (
    ActQuantWrapper,
    RotationStaticActQuantizer,
    RotationStaticWeightQuantizer,
    STEQuantize,
)


def _copy_scale_state(scale_state):
    return {
        section: {name: value.detach().cpu().clone() for name, value in values.items()}
        for section, values in scale_state.items()
    }


def load_rotation_static_scales(model, scale_state):
    activation_count = 0
    weight_count = 0
    for name, module in model.named_modules():
        if isinstance(module, RotationStaticActQuantizer):
            module.load_scale(scale_state["activation"][name])
            activation_count += 1
        elif isinstance(module, RotationStaticWeightQuantizer):
            module.load_scale(scale_state["weight"][name])
            weight_count += 1
    return activation_count, weight_count


def _candidate_alphas(start, stop, step):
    first = int(round(start * 1000))
    last = int(round(stop * 1000))
    stride = int(round(step * 1000))
    return [value / 1000 for value in range(first, last + 1, stride)]


def _mse(values, scale):
    scale_tensor = torch.as_tensor(scale, device=values.device, dtype=torch.float32)
    quantized = STEQuantize.apply(values, scale_tensor, 127)
    return float(torch.mean(torch.square(values - quantized)).item())


def _best_alpha(values, max_abs):
    if max_abs == 0:
        return 1.0, 0.0
    coarse_alphas = _candidate_alphas(0.10, 1.00, 0.05)
    coarse_results = [
        (alpha, _mse(values, alpha * max_abs / 127)) for alpha in coarse_alphas
    ]
    coarse_alpha = min(coarse_results, key=lambda item: (item[1], -item[0]))[0]

    fine_start = max(0.05, coarse_alpha - 0.05)
    fine_stop = min(1.00, coarse_alpha + 0.05)
    fine_alphas = _candidate_alphas(fine_start, fine_stop, 0.005)
    fine_results = [
        (alpha, _mse(values, alpha * max_abs / 127)) for alpha in fine_alphas
    ]
    return min(fine_results, key=lambda item: (item[1], -item[0]))


class DownProjMSECalibrator:
    def __init__(self, input_batches, logger=None):
        self.input_batches = [batch.detach().cpu() for batch in input_batches]
        self.logger = logger

    def _collect(self, model):
        captured = {}
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, ActQuantWrapper) and name.endswith("mlp.down_proj"):
                captured[name] = []

                def capture_input(_, args, target_name=name):
                    captured[target_name].append(
                        args[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    )

                handles.append(module.register_forward_pre_hook(capture_input))

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        try:
            with torch.no_grad():
                for input_ids in self.input_batches:
                    model(
                        input_ids=input_ids.to(device),
                        use_cache=False,
                        num_logits_to_keep=1,
                    )
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        return captured

    def run(self, model, base_scale_state, calibration_metadata=None):
        activation_count, weight_count = load_rotation_static_scales(
            model, base_scale_state
        )
        captured = self._collect(model)
        mse_state = _copy_scale_state(base_scale_state)
        shadow_minmax_state = _copy_scale_state(base_scale_state)
        layer_stats = []
        device = next(model.parameters()).device

        for wrapper_name in sorted(captured):
            scale_name = f"{wrapper_name}.quantizer"
            values = torch.cat(captured[wrapper_name], dim=0).to(
                device=device, dtype=torch.float32
            )
            max_abs = float(torch.amax(torch.abs(values)).item())
            old_scale = float(base_scale_state["activation"][scale_name].item())
            minmax_scale = max_abs / 127 if max_abs != 0 else 1.0
            best_alpha, new_mse = _best_alpha(values, max_abs)
            clip_threshold = best_alpha * max_abs
            new_scale = clip_threshold / 127 if clip_threshold != 0 else 1.0
            old_mse = _mse(values, old_scale)
            minmax_mse = _mse(values, minmax_scale)
            unclamped = torch.round(values / new_scale)
            saturation_fraction = float(
                torch.mean(((unclamped < -128) | (unclamped > 127)).float()).item()
            )
            range_exceed_fraction = float(
                torch.mean((torch.abs(values) > clip_threshold).float()).item()
            )

            mse_state["activation"][scale_name] = torch.tensor([new_scale])
            shadow_minmax_state["activation"][scale_name] = torch.tensor(
                [minmax_scale]
            )
            reduction = (old_mse - new_mse) / old_mse if old_mse != 0 else 0.0
            stat = {
                "layer": wrapper_name,
                "old_scale": old_scale,
                "max_abs": max_abs,
                "shadow_minmax_scale": minmax_scale,
                "best_alpha": best_alpha,
                "best_clip_threshold": clip_threshold,
                "new_scale": new_scale,
                "old_quant_mse": old_mse,
                "shadow_minmax_quant_mse": minmax_mse,
                "new_quant_mse": new_mse,
                "mse_reduction_ratio": reduction,
                "range_exceed_fraction": range_exceed_fraction,
                "saturation_fraction": saturation_fraction,
            }
            layer_stats.append(stat)
            if self.logger is not None:
                self.logger.info("down_proj MSE calibration: %s", stat)
            del values, unclamped
            torch.cuda.empty_cache()

        old_scales = [item["old_scale"] for item in layer_stats]
        new_scales = [item["new_scale"] for item in layer_stats]
        alphas = [item["best_alpha"] for item in layer_stats]
        range_exceed = [item["range_exceed_fraction"] for item in layer_stats]
        saturation = [item["saturation_fraction"] for item in layer_stats]
        reductions = [item["mse_reduction_ratio"] for item in layer_stats]
        summary = {
            "activation_scale_entries_loaded": activation_count,
            "weight_scale_entries_loaded": weight_count,
            "down_proj_layers": len(layer_stats),
            "mean_old_scale": sum(old_scales) / len(old_scales),
            "mean_new_scale": sum(new_scales) / len(new_scales),
            "max_old_scale": max(old_scales),
            "max_new_scale": max(new_scales),
            "mean_best_alpha": sum(alphas) / len(alphas),
            "min_best_alpha": min(alphas),
            "max_best_alpha": max(alphas),
            "mean_range_exceed_fraction": sum(range_exceed) / len(range_exceed),
            "max_range_exceed_fraction": max(range_exceed),
            "mean_saturation_fraction": sum(saturation) / len(saturation),
            "max_saturation_fraction": max(saturation),
            "mean_mse_reduction_ratio": sum(reductions) / len(reductions),
        }
        return {
            "mse_scale_state": mse_state,
            "shadow_minmax_scale_state": shadow_minmax_state,
            "stats": {
                "calibration": calibration_metadata or {},
                "summary": summary,
                "layers": layer_stats,
            },
        }


def save_downproj_mse_result(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        result["mse_scale_state"], output_dir / "quant_scales_downproj_mse.pt"
    )
    torch.save(
        result["shadow_minmax_scale_state"],
        output_dir / "quant_scales_downproj_shadow_minmax.pt",
    )
    (output_dir / "downproj_mse_clipping_stats.json").write_text(
        json.dumps(result["stats"], indent=2), encoding="utf-8"
    )
