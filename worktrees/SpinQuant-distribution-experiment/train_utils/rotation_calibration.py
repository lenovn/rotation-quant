import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import TrainerCallback

from utils.quant_utils import (
    RotationStaticActQuantizer,
    RotationStaticWeightQuantizer,
)


def _named_quantizers(model):
    activations = []
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, RotationStaticActQuantizer) and module.bits < 16:
            activations.append((name, module))
        elif isinstance(module, RotationStaticWeightQuantizer):
            weights.append((name, module))
    return activations, weights


def _scale_vector(named_quantizers):
    scales = [module.scale.detach().float().reshape(-1).cpu() for _, module in named_quantizers]
    return torch.cat(scales) if scales else torch.empty(0)


def _relative_drift(before: torch.Tensor, after: torch.Tensor):
    if before.numel() == 0 or before.shape != after.shape:
        return None
    denominator = torch.linalg.vector_norm(before).clamp_min(torch.finfo(torch.float32).eps)
    return float((torch.linalg.vector_norm(after - before) / denominator).item())


def enable_non_downproj_scale_learning(model):
    """Learn non-down_proj SA and make down_proj a true A16 bypass."""

    learned = []
    bypassed = []
    for name, module in model.named_modules():
        if not isinstance(module, RotationStaticActQuantizer):
            continue
        if name.endswith("mlp.down_proj.quantizer"):
            module.bits = 16
            bypassed.append(name)
            continue
        module.enable_scale_learning()
        learned.append((name, module.scale))
    if not learned:
        raise RuntimeError("No non-down_proj activation scales were selected")
    return learned, bypassed


class LearnableScaleClampCallback(TrainerCallback):
    """Keep directly optimized LSQ step sizes strictly positive."""

    def __init__(self, minimum: float = 1e-8) -> None:
        self.minimum = float(minimum)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            for module in model.modules():
                if (
                    isinstance(module, RotationStaticActQuantizer)
                    and module.scale_learnable
                ):
                    module.scale.clamp_(min=self.minimum)
        return control


class RotationScaleCalibrator:
    def __init__(self, input_batches: Iterable[torch.Tensor], logger=None) -> None:
        self.input_batches = [batch.detach().cpu() for batch in input_batches]
        self.logger = logger
        self.history = []

    def run(self, model, update_step: int, kind: str):
        activations, weights = _named_quantizers(model)
        before_a = _scale_vector(activations) if all(q.has_scale for _, q in activations) else torch.empty(0)
        before_w = _scale_vector(weights) if all(q.has_scale for _, q in weights) else torch.empty(0)
        was_training = model.training

        for _, quantizer in weights:
            quantizer.begin_calibration()
        for _, quantizer in activations:
            quantizer.begin_calibration()

        model.eval()
        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                for input_ids in self.input_batches:
                    model(
                        input_ids=input_ids.to(device),
                        use_cache=False,
                        num_logits_to_keep=1,
                    )
            for _, quantizer in weights:
                quantizer.finish_calibration()
            for _, quantizer in activations:
                quantizer.finish_calibration()
        except Exception:
            for _, quantizer in weights:
                quantizer.abort_calibration()
            for _, quantizer in activations:
                quantizer.abort_calibration()
            raise
        finally:
            model.train(was_training)

        after_a = _scale_vector(activations)
        after_w = _scale_vector(weights)
        record = {
            "update_step": int(update_step),
            "kind": kind,
            "activation_scale_mean": (
                float(after_a.mean().item()) if after_a.numel() else None
            ),
            "weight_scale_mean": (
                float(after_w.mean().item()) if after_w.numel() else None
            ),
            "activation_relative_drift": _relative_drift(before_a, after_a),
            "weight_relative_drift": _relative_drift(before_w, after_w),
        }
        self.history.append(record)
        if self.logger is not None:
            self.logger.info("rotation-static calibration: %s", record)
        return record

    def save_history(self, path) -> None:
        Path(path).write_text(json.dumps(self.history, indent=2), encoding="utf-8")


class PeriodicRecalibrationCallback(TrainerCallback):
    def __init__(
        self,
        calibrator: RotationScaleCalibrator,
        interval: int,
        final_update: int,
    ) -> None:
        self.calibrator = calibrator
        self.interval = int(interval)
        self.final_update = int(final_update)
        self._completed_steps = set()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if (
            step > 0
            and step < self.final_update
            and step % self.interval == 0
            and step not in self._completed_steps
        ):
            self.calibrator.run(model, update_step=step, kind="periodic")
            self._completed_steps.add(step)
        return control


def export_rotation_scales(model):
    activations, weights = _named_quantizers(model)
    return {
        "activation": {
            name: module.scale.detach().cpu().clone() for name, module in activations
        },
        "weight": {
            name: module.scale.detach().cpu().clone() for name, module in weights
        },
    }
