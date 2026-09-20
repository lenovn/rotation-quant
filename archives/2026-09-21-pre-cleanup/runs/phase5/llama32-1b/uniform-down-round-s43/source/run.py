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

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import torch

from experiments.phase3.common import (
    format_description,
    MODEL_PATH, PROJECT_ROOT, append_json, backbone, build_training_model,
    calibrate_sp2, data_windows, evaluate, frozen_model, full_validation, initialize_scales,
    learned_parameters, load_state, reload_frozen, reset_weight_scales,
    save_frozen, save_state, token_nll, training_mode, wrappers, write_json,
)
from train_utils.optimizer import SGDG


def schedule(update_index, total_updates, warmup):
    if update_index < warmup:
        return (update_index + 1) / max(warmup, 1)
    progress = (update_index - warmup) / max(total_updates - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizers(groups, args):
    learning_rates = dict(R=args.r_lr, SA=args.sa_lr, SW=args.sw_lr, SP2=args.sp2_lr)
    if args.scale_optimizer == "sgd":
        optimizer = SGDG([dict(params=[parameter for _, parameter in values],
            lr=learning_rates[name], initial_lr=learning_rates[name], stiefel=name == "R", name=name)
            for name, values in groups.items()], lr=args.r_lr)
        return [optimizer]
    rotation_optimizer = SGDG([dict(params=[parameter for _, parameter in groups["R"]],
        lr=args.r_lr, initial_lr=args.r_lr, stiefel=True, name="R")], lr=args.r_lr)
    scale_groups = []
    reference_path = getattr(args, "optimizer_scale_reference", None)
    reference = (torch.load(reference_path, map_location="cpu", weights_only=True)["parameters"]
                 if reference_path is not None else None)
    for name in ("SA", "SW", "SP2"):
        for parameter_name, parameter in groups[name]:
            initial_scale = parameter.detach() if reference is None else reference[parameter_name].to(parameter.device)
            rate = args.relative_scale_lr * float(initial_scale.mean())
            scale_groups.append(dict(params=[parameter], lr=rate, initial_lr=rate,
                                     name=name, parameter_name=parameter_name))
    scale_optimizer = torch.optim.Adam(scale_groups, eps=1e-12, weight_decay=0.0)
    return [rotation_optimizer, scale_optimizer]


def save_resume(model, optimizers, output, step, route):
    state = dict(parameters={name: parameter.detach().cpu().clone()
        for values in learned_parameters(model).values() for name, parameter in values},
        metadata=dict(route=route, update_step=step),
        optimizers=[optimizer.state_dict() for optimizer in optimizers],
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state())
    temporary = output / "resume.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "resume.pt")


def progress(output, stage, **details):
    record = dict(stage=stage, pid=os.getpid(), time=time.time(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  allocated_gib=torch.cuda.memory_allocated() / 2 ** 30,
                  reserved_gib=torch.cuda.memory_reserved() / 2 ** 30,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                  gpu_time_budget=None, **details)
    write_json(output / "progress.json", record)
    print(json.dumps(record, allow_nan=False), flush=True)


def source_record():
    import train_utils.quant_linear
    import utils.quant_utils

    return dict(source_root=str(SOURCE_ROOT), project_root=str(PROJECT_ROOT),
        quant_linear=train_utils.quant_linear.__file__, quant_utils=utils.quant_utils.__file__,
        head=subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "branch", "--show-current"], text=True).strip(),
        pytorch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        model_path=str(MODEL_PATH), model_cache_revision=((MODEL_PATH / "phase5_revision.json") if (MODEL_PATH / "phase5_revision.json").exists() else (MODEL_PATH / ".mv")).read_text(),
        transformers=__import__("transformers").__version__, seed=int(os.environ.get("PHASE5_SEED", "42")))


def gradient_record(groups, before):
    records = {}
    for group_name, parameters in groups.items():
        before_square = 0.0
        delta_square = 0.0
        grad_square = 0.0
        changed = 0
        grad_tensors = 0
        max_relative = 0.0
        details = {}
        for name, parameter in parameters:
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"Non-finite parameter: {name}")
            difference = parameter.detach() - before[name]
            before_square += float(before[name].square().sum())
            delta_square += float(difference.square().sum())
            changed += int((difference != 0).sum())
            max_relative = max(max_relative, float((difference.abs() / before[name].abs().clamp_min(1e-8)).max()))
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite gradient: {name}")
                grad_tensors += 1
                grad_square += float(parameter.grad.square().sum())
            if group_name in ("SA", "SP2"):
                details[name] = dict(before=float(before[name]), after=float(parameter),
                                     changed=bool((difference != 0).any()))
        records[group_name] = dict(relative_l2=math.sqrt(delta_square / max(before_square, 1e-30)),
            gradient_l2=math.sqrt(grad_square), gradient_tensors=grad_tensors,
            changed_elements=changed, max_relative_update=max_relative, scalar_details=details)
    return records


@torch.no_grad()
def probe_with_saturation(model, probe):
    counts = {}

    def capture(name):
        def hook(module, inputs):
            values = inputs[0].detach().float()
            quantizer = module.quantizer
            upper = quantizer.alpha if name.endswith("down_proj") else quantizer.scale * 127
            lower = -upper if name.endswith("down_proj") else quantizer.scale * -128
            row = counts.setdefault(name, dict(total=0, saturated=0, bits=quantizer.bits))
            row["total"] += values.numel()
            row["saturated"] += int(((values < lower) | (values > upper)).sum())
        return hook

    handles = [wrapper.register_forward_pre_hook(capture(name)) for name, wrapper in wrappers(model).items()]
    try:
        result = evaluate(model, probe)
    finally:
        for handle in handles:
            handle.remove()
    for row in counts.values():
        row["fraction"] = row["saturated"] / row["total"]
    result["saturation"] = counts
    return result


def initialize(output, calibration, probe):
    model = build_training_model().cuda()
    progress(output, "initial-scale-calibration")
    initialize_scales(model, calibration)
    frozen, records = frozen_model(model)
    results = calibrate_sp2(frozen, calibration, output)
    with torch.no_grad():
        for name, record in results.items():
            model.get_submodule(name).quantizer.scale.copy_(
                model.get_submodule(name).quantizer.scale.new_tensor([record["selected"]["alpha"] / 127]))
    state = save_state(model, output / "initial.pt", seed=int(os.environ.get("PHASE5_SEED", "42")),
                      rotation_initialization="unoptimized randomized Hadamard; no Phase2 learned R or scales",
                      scale_initialization="non-down A8 full range on FP train forward; W4 current-R per-row max/7; SP2 train-only output-MSE 33+17")
    result = evaluate(frozen, probe)
    write_json(output / "initial_static_probe.json", result)
    save_frozen(frozen, records, output / "initial_static.pt", state["metadata"])
    progress(output, "completed", initial=str(output / "initial.pt"), static_probe=result["nll"])


def evaluate_checkpoint(model, args, step, calibration, probe, validation):
    directory = args.output / f"checkpoint-{step:04d}"
    directory.mkdir(exist_ok=False)
    save_state(model, directory / "state.pt", route=args.route, update_step=step,
               total_updates=args.steps, switch_step=args.switch_step,
               initial_state=str(args.initial), training_tokens=step * args.accumulation * 2048)
    fixed_probe = probe_with_saturation(model, probe)
    write_json(directory / "training_probe.json", fixed_probe)
    if step not in args.validation_steps:
        return dict(step=step, training_probe=fixed_probe["nll"])
    progress(args.output, "freeze-and-calibrate", step=step)
    frozen, records = frozen_model(model, current_minmax=args.route == "A" and step <= args.switch_step)
    if args.route != "C":
        calibrate_sp2(frozen, calibration, directory)
    else:
        for name, wrapper in wrappers(frozen).items():
            if name.endswith("down_proj"):
                wrapper.quantizer.bits = 8
    save_frozen(frozen, records, directory / "static_w4a8.pt",
                dict(route=args.route, step=step, initial=str(args.initial)))
    before = evaluate(frozen, probe[:1])
    reload_frozen(frozen, directory / "static_w4a8.pt")
    after = evaluate(frozen, probe[:1])
    if before["nll"] != after["nll"]:
        raise RuntimeError("Frozen pack/reload probe NLL mismatch")
    write_json(directory / "reload_check.json", dict(before=before["nll"], after=after["nll"], exact=True))
    progress(args.output, "full-validation", step=step)
    result = full_validation(frozen, validation)
    result.update(route=args.route, update_step=step, format=f"{7 * model.config.num_hidden_layers} W4 / {6 * model.config.num_hidden_layers} static INT8 / {model.config.num_hidden_layers} static SP2; BF16 KV16 prefill",
                  scale_initialization=str(args.initial), source_root=str(SOURCE_ROOT))
    write_json(directory / "validation.json", result)
    del frozen, records
    gc.collect()
    torch.cuda.empty_cache()
    progress(args.output, "checkpoint-completed", step=step, nll=result["nll"], ppl=result["ppl"])
    return dict(step=step, training_probe=fixed_probe["nll"], validation_nll=result["nll"], validation_ppl=result["ppl"])


def train(args, train_windows, calibration, probe, validation):
    model = build_training_model(args.initial).cuda()
    groups = learned_parameters(model)
    optimizers = make_optimizers(groups, args)
    start_step = 0
    if args.resume is not None:
        resume_path = args.resume / "resume.pt"
        if resume_path.exists():
            state = load_state(model, resume_path)
            saved_optimizer = torch.load(resume_path, map_location="cpu", weights_only=False)
        else:
            state = load_state(model, args.resume / "resume_state.pt")
            saved_optimizer = torch.load(args.resume / "optimizer.pt", map_location="cpu", weights_only=False)
        if state["metadata"]["route"] != args.route:
            raise ValueError("Resume route differs from checkpoint")
        saved_states = saved_optimizer.get("optimizers", [saved_optimizer.get("optimizer")])
        if len(saved_states) != len(optimizers):
            raise ValueError("Cannot resume with a different optimizer method")
        for optimizer, saved_state in zip(optimizers, saved_states):
            optimizer.load_state_dict(saved_state)
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = dict(R=args.r_lr, SA=args.sa_lr, SW=args.sw_lr, SP2=args.sp2_lr)[group["name"]]
        random.setstate(saved_optimizer["python_rng"])
        torch.set_rng_state(saved_optimizer["torch_rng"])
        torch.cuda.set_rng_state(saved_optimizer["cuda_rng"])
        start_step = state["metadata"]["update_step"]
    if args.scale_only_steps:
        for _, parameter in groups["R"]:
            parameter.requires_grad_(False)
    training_mode(model, args.route, start_step, args.switch_step)
    checkpoints = []
    if args.resume is None:
        checkpoints.append(evaluate_checkpoint(model, args, 0, calibration, probe, validation))
    started = time.monotonic()
    for step in range(start_step, args.steps):
        if args.route == "A" and step == args.switch_step:
            reset_weight_scales(model)
            progress(args.output, "A-switch-to-W4", step=step)
        training_mode(model, args.route, step, args.switch_step)
        factor = schedule(step, args.schedule_steps or args.steps, args.warmup)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
        before = {name: parameter.detach().clone() for values in groups.values() for name, parameter in values}
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        window_indices = []
        for microstep in range(args.accumulation):
            index = (step * args.accumulation + microstep) % len(train_windows)
            window_indices.append(index)
            ids = train_windows[index:index + 1].cuda()
            loss = token_nll(model, ids)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step + 1}")
            losses.append(float(loss.detach()))
            (loss / args.accumulation).backward()
            del loss
        for values in groups.values():
            for name, parameter in values:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite training gradient: {name}")
        trainable = [parameter for values in groups.values() for _, parameter in values if parameter.requires_grad]
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for group_name in ("SA", "SW", "SP2"):
                for _, parameter in groups[group_name]:
                    parameter.clamp_(min=1e-8)
        changes = gradient_record(groups, before)
        del before
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        record = dict(step=step + 1, microbatch_losses=losses, minibatch_mean_nll=sum(losses) / len(losses),
            effective_targets=args.accumulation * 2047, train_tokens=args.accumulation * 2048,
            cumulative_train_tokens=(step + 1) * args.accumulation * 2048,
            window_indices=window_indices,
            learning_rates={group.get("parameter_name", group["name"]): group["lr"]
                            for optimizer in optimizers for group in optimizer.param_groups},
            gradient_norm_before_clip=norm, updates=changes, elapsed_seconds=time.monotonic() - started)
        append_json(args.output / "training.jsonl", record)
        save_resume(model, optimizers, args.output, step + 1, args.route)
        progress(args.output, "training", step=step + 1, total=args.steps, loss=record["minibatch_mean_nll"],
                 elapsed_seconds=time.monotonic() - started)
        if step + 1 in args.checkpoints:
            checkpoints.append(evaluate_checkpoint(model, args, step + 1, calibration, probe, validation))
            write_json(args.output / "results.json", checkpoints)
    if args.scale_only_steps:
        save_state(model, args.output / "improved_initial.pt", initialization="scale-only CE preoptimization",
                   updates=args.steps, extra_train_tokens=args.steps * args.accumulation * 2048,
                   parent=str(args.initial))
    write_json(args.output / "results.json", checkpoints)
    progress(args.output, "completed", steps=args.steps, elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--route", choices=("A", "B", "C"), default="C")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--switch-step", type=int, default=50)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--r-lr", type=float, default=1.5)
    parser.add_argument("--sa-lr", type=float, default=1.0)
    parser.add_argument("--sw-lr", type=float, default=0.01)
    parser.add_argument("--sp2-lr", type=float, default=1.0)
    parser.add_argument("--scale-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--relative-scale-lr", type=float, default=0.001)
    parser.add_argument("--optimizer-scale-reference", type=Path)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--validation-steps", type=int, nargs="*", default=[10, 100])
    parser.add_argument("--scale-only-steps", action="store_true")
    args = parser.parse_args()
    if not args.initialize and args.initial is None:
        parser.error("Training requires a shared initial checkpoint")
    if args.steps < 1 or args.accumulation < 1 or args.warmup < 0:
        parser.error("Invalid training budget")
    args.output.mkdir(parents=True, exist_ok=False)
    source_directory = args.output / "source"
    source_directory.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source_directory / path.name)
    (source_directory / "tracked.diff").write_bytes(subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(args.output / "settings.json", dict(arguments={name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()}, source=source_record(), pid=os.getpid()))
    try:
        progress(args.output, "loading-data")
        train_windows, calibration, probe, validation, metadata = data_windows(args.output)
        if args.initialize:
            initialize(args.output, calibration, probe)
        else:
            train(args, train_windows, calibration, probe, validation)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
