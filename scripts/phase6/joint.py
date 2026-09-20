"""Phase6: fresh R / W4 scales / static INT8+INT16 scales, no distillation."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "worktrees/SpinQuant-multimodel"
sys.path.insert(0, str(SOURCE))

import torch
from experiments.phase3 import common as c
from experiments.phase3.run import schedule, gradient_record
from train_utils.optimizer import SGDG
from utils.quant_utils import LSQQuantize


class SignedActivation(torch.nn.Module):
    rotation_static_enabled = True
    static_enabled = True
    sym = True
    groupsize = -1

    def __init__(self, bits, scale, learnable=True):
        super().__init__()
        if bits not in (8, 16):
            raise ValueError("Only signed INT8 and INT16 supported")
        self.bits = bits
        value = torch.as_tensor(scale).detach().float().reshape(1).clone()
        if not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError("Activation scale must be finite and positive")
        if learnable:
            self.scale = torch.nn.Parameter(value)
        else:
            self.register_buffer("scale", value)
        self.register_buffer("maxq", torch.tensor(2 ** (bits - 1) - 1, device=value.device))
        self.calls = 0

    @property
    def alpha(self):
        return self.scale * self.maxq

    def forward(self, x):
        self.calls += 1
        # Division/rounding must happen in FP32, including the INT16 grid.
        return LSQQuantize.apply(x.float(), self.scale, self.maxq).to(x.dtype)


def down_hook(wrapper, inputs):
    # Legacy ActQuantWrapper bypasses bits=16. Explicitly quantize once here.
    if wrapper.online_full_had or wrapper.online_partial_had:
        raise ValueError("Phase6 supports offline R1/R2 only")
    return (wrapper.quantizer(inputs[0]), *inputs[1:])


def install(model, scales=None, learnable=True):
    for name, wrapper in c.wrappers(model).items():
        down = name.endswith("down_proj")
        if scales is None:
            # initialize_scales supplies absmax/127 for both legacy families.
            scale = wrapper.quantizer.scale.detach() * (127 / 32767 if down else 1)
        else:
            scale = scales[name]
        wrapper.quantizer = SignedActivation(16 if down else 8, scale, learnable)
        if down and not getattr(wrapper, "_phase6_down_installed", False):
            wrapper.register_forward_pre_hook(down_hook)
            wrapper._phase6_down_installed = True


def groups(model):
    result = {"R": [], "SA": [], "SW": [], "DOWN_SA": []}
    for name, parameter in model.named_parameters():
        if name == "R1.weight" or name.endswith("self_attn.R2.weight"):
            result["R"].append((name, parameter))
    for name, wrapper in c.wrappers(model).items():
        result["DOWN_SA" if name.endswith("down_proj") else "SA"].append((name + ".quantizer.scale", wrapper.quantizer.scale))
        result["SW"].append((name + ".module.quantizer.scale", wrapper.module.quantizer.scale))
    return result


def parameters(model):
    return {name: p for rows in groups(model).values() for name, p in rows}


def save_parameters(model, path, **metadata):
    torch.save(dict(parameters={n: p.detach().cpu().clone() for n, p in parameters(model).items()}, metadata=metadata), path)


@torch.no_grad()
def load_parameters(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    load_parameter_values(model, state["parameters"])
    return state


@torch.no_grad()
def load_parameter_values(model, values):
    target = parameters(model)
    if target.keys() != values.keys():
        raise ValueError("Learned parameter coverage mismatch")
    for name, value in values.items():
        if value.shape != target[name].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid parameter: " + name)
        if name.endswith(".scale") and not (value > 0).all():
            raise ValueError("Nonpositive scale: " + name)
        target[name].copy_(value.to(target[name]))


@torch.no_grad()
def freeze(model, current_minmax=False):
    # Reuse the established R1/R2 folding and INT4 packing. The temporary
    # legacy activation objects are replaced before any frozen forward.
    frozen, records = c.frozen_model(model, current_minmax=current_minmax)
    install(frozen, {n: w.quantizer.scale.detach().clone() for n, w in c.wrappers(model).items()}, learnable=False)
    frozen.requires_grad_(False).eval()
    return frozen, records


@torch.no_grad()
def save_package(model, records, path, metadata):
    high = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
            if not any(n.startswith(prefix + ".") for prefix in records)}
    activation = {n: dict(bits=w.quantizer.bits, scale=w.quantizer.scale.detach().cpu().clone())
                  for n, w in c.wrappers(model).items()}
    torch.save(dict(weights=records, activation=activation, high_precision=high,
                    metadata=metadata, config=model.config.to_dict(), format="phase6-static-int4-int8-int16-v1"), path)


@torch.no_grad()
def load_package(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != "phase6-static-int4-int8-int16-v1":
        raise ValueError("Not a Phase6 package")
    model = c.load_model(c.MODEL_PATH, untie=True)
    c.add_actquant(model)
    expected = set(c.wrappers(model))
    if set(state["weights"]) != expected or set(state["activation"]) != expected:
        raise ValueError("Frozen coverage mismatch")
    high = {n: v for n, v in model.state_dict().items() if not any(n.startswith(prefix + ".") for prefix in expected)}
    if high.keys() != state["high_precision"].keys():
        raise ValueError("High precision coverage mismatch")
    model.load_state_dict(state["high_precision"], strict=False)
    for name, record in state["weights"].items():
        wrapper = model.get_submodule(name)
        if tuple(wrapper.module.weight.shape) != tuple(record["shape"]):
            raise ValueError("Weight shape mismatch")
        scale = record["scale"]
        if scale.shape != (wrapper.module.out_features, 1) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Invalid W4 scale")
        codes = c.unpack_int4(record["packed"], record["shape"])
        wrapper.module.weight.copy_((codes.float() * scale).to(wrapper.module.weight))
        if state["activation"][name]["bits"] != (16 if name.endswith("down_proj") else 8):
            raise ValueError("Activation precision mismatch")
    install(model, {n: a["scale"] for n, a in state["activation"].items()}, learnable=False)
    model.config.use_cache = False
    return model.requires_grad_(False).eval().cuda(), state


def data(output):
    windows, _, probe, validation, metadata = c.data_windows()
    if len(windows) < 800:
        raise ValueError("Fewer than 800 training windows")
    windows = windows[:800]
    indices = torch.randperm(800, generator=torch.Generator().manual_seed(42))[:32].tolist()
    calibration = [windows[i:i+1] for i in indices]
    metadata.update(training_window_indices=list(range(800)), calibration_window_indices=indices,
                    calibration_length=2048, phase6_training_windows=800)
    c.write_json(output / "data.json", metadata)
    return windows, calibration, probe, validation


@torch.no_grad()
def recalibrate_frozen(model, calibration):
    observed = {n: 0.0 for n in c.wrappers(model)}
    handles = []
    for name, wrapper in c.wrappers(model).items():
        def capture(module, inputs, name=name):
            observed[name] = max(observed[name], float(inputs[0].detach().abs().max()))
        # prepend observes before the down INT16 hook; all use previous scales.
        handles.append(wrapper.register_forward_pre_hook(capture, prepend=True))
    try:
        for ids in calibration:
            c.backbone(model, ids.cuda())
    finally:
        for handle in handles:
            handle.remove()
    for name, wrapper in c.wrappers(model).items():
        wrapper.quantizer.scale.fill_(max(observed[name], 1e-8) / int(wrapper.quantizer.maxq))


def checkpoint_eval(model, args, step, calibration, validation, probe):
    directory = args.output / f"checkpoint-{step:04d}"
    directory.mkdir(exist_ok=True)
    if (directory / "validation.json").exists() and (step != 0 or (directory / "cold_reload.json").exists()):
        done = json.loads((directory / "validation.json").read_text())
        return dict(step=step, ppl=done["ppl"], nll=done["nll"])
    save_parameters(model, directory / "parameters.pt", step=step, arm=args.arm)
    frozen, records = freeze(model, current_minmax=args.arm == "r-only" and step > 0)
    if args.arm == "r-only" and step > 0:
        recalibrate_frozen(frozen, calibration)
    metadata = dict(model_path=str(c.MODEL_PATH), step=step, arm=args.arm,
                    schedule_horizon=512, down="signed INT16", evaluation="BF16 fakequant prefill; KV BF16")
    save_package(frozen, records, directory / "model.pt", metadata)
    old_validation = args.initial.parent / "checkpoint-0000/validation.json" if args.initial else None
    if (directory / "validation.json").exists():
        result = json.loads((directory / "validation.json").read_text())
    elif step == 0 and old_validation is not None and old_validation.exists():
        result = json.loads(old_validation.read_text())
        result["reused_validation_source"] = str(old_validation)
    else:
        result = c.full_validation(frozen, validation)
    result.update(metadata)
    c.write_json(directory / "validation.json", result)
    if step == 0:
        frozen.cuda()
        reference = c.backbone(frozen, probe[0][:, :32].cuda()).cpu()
        del frozen, records
        gc.collect()
        torch.cuda.empty_cache()
        frozen, state = load_package(directory / "model.pt")
        with torch.no_grad():
            reloaded = c.backbone(frozen, probe[0][:, :32].cuda()).cpu()
        if not torch.equal(reference, reloaded):
            raise RuntimeError("Export cold reload differs")
        c.write_json(directory / "cold_reload.json", dict(exact=True))
        del state
    del frozen
    gc.collect()
    torch.cuda.empty_cache()
    return dict(step=step, ppl=result["ppl"], nll=result["nll"])


def progress(args, stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), time=time.time(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 allocated_gib=torch.cuda.memory_allocated()/2**30, peak_gib=torch.cuda.max_memory_allocated()/2**30, **kw)
    c.write_json(args.output / "progress.json", value)
    print(json.dumps(value), flush=True)


def run(args):
    windows, calibration, probe, validation = data(args.output)
    progress(args, "initializing-from-pretrained")
    model = c.build_training_model(args.initial).cuda()
    if args.initial is None:
        c.initialize_scales(model, calibration)
    install(model)
    if args.initial:
        load_parameters(model, args.initial)
    if not args.resume:
        save_parameters(model, args.output / "initial.pt", initialization="fresh pretrained/random Hadamard R/train absmax", seed=42)
    grouped = groups(model)
    for group, rows in grouped.items():
        learn = args.arm != "no-opt" and (group == "R" or args.arm != "r-only")
        if args.arm == "fixed-down" and group == "DOWN_SA":
            learn = False
        for _, parameter in rows:
            parameter.requires_grad_(learn)
    expected = {id(p) for rows in grouped.values() for _, p in rows if p.requires_grad}
    if any(p.requires_grad and id(p) not in expected for p in model.parameters()):
        raise RuntimeError("Unexpected trainable model weight")
    optimizers = []
    if args.arm != "no-opt":
        optimizers.append(SGDG([dict(params=[p for _, p in grouped["R"]], lr=1.5, initial_lr=1.5, stiefel=True)], lr=1.5))
        scale_groups = [dict(params=[p], lr=.001*float(p.detach().mean()), initial_lr=.001*float(p.detach().mean()))
                        for g, rows in grouped.items() if g != "R" for _, p in rows if p.requires_grad]
        if scale_groups:
            optimizers.append(torch.optim.Adam(scale_groups, eps=1e-12, weight_decay=0))
    results = []
    start = 0
    if args.resume:
        saved = torch.load(args.output / "resume.pt", map_location="cpu", weights_only=False)
        if "parameters" in saved:
            load_parameter_values(model, saved["parameters"])
        else:
            # Compatibility with the first live pilot's two-file checkpoint.
            state = torch.load(args.output / "resume_parameters.pt", map_location="cpu", weights_only=True)
            if state["metadata"]["step"] != saved["step"]:
                raise ValueError("Legacy resume parameter/optimizer steps differ")
            load_parameter_values(model, state["parameters"])
        if saved["arm"] != args.arm:
            raise ValueError("Resume arm differs")
        for optimizer, value in zip(optimizers, saved["optimizers"]):
            optimizer.load_state_dict(value)
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        start = saved["step"]
        if args.steps < start:
            raise ValueError("Cannot resume to a step before the saved update")
        results = json.loads((args.output / "results.json").read_text())
        if (start % 64 == 0 or start == args.steps) and not any(row["step"] == start for row in results):
            results.append(checkpoint_eval(model, args, start, calibration, validation, probe))
            c.write_json(args.output / "results.json", results)
    else:
        results.append(checkpoint_eval(model, args, 0, calibration, validation, probe))
        c.write_json(args.output / "results.json", results)
    began = time.monotonic()
    for step in range(start, 0 if args.arm == "no-opt" else args.steps):
        model.train()
        factor = schedule(step, 512, 10)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
        before = {n: p.detach().clone() for n, p in parameters(model).items()}
        losses = []
        indices = [(step*8+i) % 800 for i in range(8)]
        for index in indices:
            loss = c.token_nll(model, windows[index:index+1].cuda())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training NLL")
            losses.append(float(loss.detach()))
            (loss / 8).backward()
            del loss
        trainable = [p for p in parameters(model).values() if p.requires_grad]
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for name, p in parameters(model).items():
                if name.endswith(".scale") and p.requires_grad:
                    p.clamp_(min=1e-8)
        changes = gradient_record(grouped, before)
        del before
        c.append_json(args.output / "training.jsonl", dict(step=step+1, microbatch_losses=losses,
                      minibatch_mean_nll=sum(losses)/8, window_indices=indices, schedule_factor=factor,
                      gradient_norm=norm, updates=changes, cumulative_train_tokens=(step+1)*8*2048))
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        temporary = args.output / "resume.pt.tmp"
        torch.save(dict(step=step+1, arm=args.arm, optimizers=[o.state_dict() for o in optimizers],
                        parameters={n:p.detach().cpu().clone() for n,p in parameters(model).items()},
                        python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temporary)
        temporary.replace(args.output / "resume.pt")
        progress(args, "training", step=step+1, loss=sum(losses)/8, elapsed=time.monotonic()-began)
        if (step+1) % 64 == 0 or step+1 == args.steps:
            results.append(checkpoint_eval(model, args, step+1, calibration, validation, probe))
            c.write_json(args.output / "results.json", results)
    progress(args, "completed", steps=0 if args.arm == "no-opt" else args.steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["joint", "fixed-down", "r-only", "no-opt"], default="joint")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 512:
        parser.error("steps must be between 1 and 512")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and args.initial is None:
        args.initial = args.output / "initial.pt"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Leave driver/display overhead and a margin below the 90% total-card limit.
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PHASE6_MEMORY_FRACTION", ".85")))
    if not args.resume:
        snapshot = args.output / "source"
        snapshot.mkdir()
        shutil.copy2(__file__, snapshot / "joint.py")
        shutil.copytree(SOURCE / "experiments/phase3", snapshot / "phase3", ignore=shutil.ignore_patterns("__pycache__"))
        (snapshot / "tracked.diff").write_bytes(subprocess.check_output(["git", "-C", str(SOURCE), "diff", "HEAD"]))
        c.write_json(args.output / "settings.json", dict(model_path=str(c.MODEL_PATH), arm=args.arm,
                     steps=args.steps, schedule_horizon=512, warmup=10, seed=42, global_batch=8,
                     training_windows=800, calibration="32x2048 train-only absmax", rotation_lr=1.5,
                     scale_optimizer="Adam", relative_scale_lr=.001, adam_eps=1e-12,
                     source=str(SOURCE), torch=torch.__version__, transformers=__import__("transformers").__version__,
                     gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid()))
    try:
        run(args)
    except BaseException as error:
        c.write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
