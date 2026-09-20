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


@torch.no_grad()
def refresh_current_rotation_weight_scales(model):
    """No activation forward/calibration; share rotation math with training."""
    for layer in model.model.layers:
        for wrapper, r2, transpose in (
            (layer.self_attn.q_proj, None, False),
            (layer.self_attn.k_proj, None, False),
            (layer.self_attn.v_proj, layer.self_attn.R2.weight, False),
            (layer.self_attn.o_proj, layer.self_attn.R2.weight, True),
            (layer.mlp.gate_proj, None, False),
            (layer.mlp.up_proj, None, False),
            (layer.mlp.down_proj, None, True),
        ):
            linear = wrapper.module
            linear.quantizer.refresh_current_weight(
                linear.rotated_weight(model.R1.weight, r2, transpose)
            )


class LearnableScaleClampCallback(TrainerCallback):
    """Keep directly optimized LSQ step sizes strictly positive."""

    def __init__(self, minimum: float = 1e-8, w4_aware=False, logger=None) -> None:
        self.minimum = float(minimum)
        self.w4_aware = w4_aware
        self.logger = logger
        self.history = []
        self.before = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Check after Trainer/Accelerate wrapping; never silently train BF16 SA.
        for name, module in _named_quantizers(model)[0]:
            if module.scale_learnable and module.scale.dtype != torch.float32:
                raise RuntimeError(f"Learnable SA must remain FP32 after wrapping: {name}")
        return control

    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.w4_aware:
            refresh_current_rotation_weight_scales(model)
        self.before = {}
        for name, module in _named_quantizers(model)[0]:
            if not module.scale_learnable:
                continue
            scale = module.scale
            if scale.dtype != torch.float32 or not torch.isfinite(scale).all():
                raise RuntimeError(f"Invalid learnable SA before update: {name}, {scale.dtype}")
            group = next(g for g in optimizer.param_groups if any(p is scale for p in g["params"]))
            self.before[name] = (scale.detach().cpu().clone(), float(group["lr"]))
        self.lr_used = [float(group["lr"]) for group in optimizer.param_groups]
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        modules = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if (
                    isinstance(module, RotationStaticActQuantizer)
                    and module.scale_learnable
                ):
                    if module.scale.dtype != torch.float32 or not torch.isfinite(module.scale).all():
                        raise RuntimeError(f"Invalid SA after update: {name}, {module.scale.dtype}")
                    module.scale.clamp_(min=self.minimum)
                    if name in self.before:
                        before, lr_used = self.before[name]
                        after = module.scale.detach().cpu().clone()
                        delta = (after - before).abs()
                        modules[name] = {
                            "before": before.tolist(), "after": after.tolist(),
                            "delta_abs": delta.tolist(),
                            "delta_relative": (delta / before.abs().clamp_min(self.minimum)).tolist(),
                            "lr_used": lr_used, "dtype": str(module.scale.dtype),
                            "finite": True,
                        }
        if modules and state.is_world_process_zero:
            record = {"update_step": int(state.global_step), "lr_used": self.lr_used, "modules": modules}
            self.history.append(record)
            if self.logger is not None:
                self.logger.info("SA update: %s", record)
        return control

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        # This event precedes Trainer's zero_grad and scheduler advancement.
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(f"Non-finite trainable parameter: {name}")
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite training gradient: {name}")
        return control

    def save_history(self, path):
        Path(path).write_text(json.dumps(self.history, indent=2, allow_nan=False), encoding="utf-8")


def enable_weight_scale_learning(model, weight_scales):
    weights = _named_quantizers(model)[1]
    if set(weight_scales) != {name for name, _ in weights}:
        raise ValueError("Initial SW keys do not match model weight quantizers")
    learned = []
    for name, q in weights:
        value = weight_scales[name]
        if value.shape != q.scale.shape or not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError(f"Invalid initial SW: {name}")
        q.load_scale(value.float())
        q.enable_scale_learning()
        learned.append(q.scale)
    return learned


class LearnableWeightScaleCallback(TrainerCallback):
    """Record actual per-channel SW updates, including warmup."""
    def __init__(self):
        self.history = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        for name, q in _named_quantizers(model)[1]:
            if not q.scale_learnable or q.scale.dtype != torch.float32:
                raise RuntimeError(f"Expected persistent FP32 SW: {name}")

    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.before = {}
        for name, q in _named_quantizers(model)[1]:
            group = next(g for g in optimizer.param_groups if any(p is q.scale for p in g["params"]))
            self.before[name] = (q.scale.detach().clone(), float(group["lr"]))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        records = {}
        with torch.no_grad():
            for name, q in _named_quantizers(model)[1]:
                if q.scale.dtype != torch.float32 or not torch.isfinite(q.scale).all():
                    raise RuntimeError(f"Invalid SW after update: {name}")
                clamped = int((q.scale < 1e-8).sum().item())
                q.scale.clamp_(min=1e-8)
                before, lr = self.before[name]
                delta = q.scale - before
                records[name] = dict(lr_used=lr, dtype=str(q.scale.dtype),
                    changed_channels=int((delta != 0).sum().item()), channels=q.scale.numel(),
                    clamped_channels=clamped,
                    relative_l2=float((delta.norm() / before.norm().clamp_min(1e-12)).item()),
                    max_relative_update=float((delta.abs() / before.abs().clamp_min(1e-8)).max().item()),
                    scale_min=float(q.scale.min().item()), scale_max=float(q.scale.max().item()))
        if state.is_world_process_zero:
            self.history.append(dict(update_step=int(state.global_step), modules=records))

    def save_history(self, path):
        Path(path).write_text(json.dumps(self.history, indent=2, allow_nan=False))


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
