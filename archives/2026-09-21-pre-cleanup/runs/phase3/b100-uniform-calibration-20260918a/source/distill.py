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
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, set_seed

from experiments.phase3.common import (
    MODEL_PATH, append_json, backbone, data_windows, evaluate, full_validation,
    save_frozen, wrappers, write_json,
)
from experiments.phase3.postprocess import load_static, reference_weights
from experiments.phase3.quantization import Phase3WeightQuantizer, int4_codes, pack_int4, unpack_int4
from experiments.phase3.run import progress, schedule, source_record


class TrainableQuantLinear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().float().clone())
        self.bias = None
        self.out_features, self.in_features = weight.shape
        self.quantizer = Phase3WeightQuantizer()
        self.quantizer.configure(4, perchannel=True, sym=True, mse=False, weight_groupsize=-1)
        self.quantizer.load_scale(scale)
        self.quantizer.enable_scale_learning()
        self.to(weight.device)

    def forward(self, inputs):
        weight = self.quantizer.quantize(self.weight).to(inputs.dtype)
        return functional.linear(inputs, weight)


def cell_reference_initialization(record, reference):
    if tuple(reference.shape) != tuple(record["shape"]):
        raise ValueError("Master reference shape differs from parent weight")
    device = reference.device
    codes = unpack_int4(record["packed"], record["shape"]).to(device).float()
    scale = record["scale"].to(device)
    normalized = reference.float() / scale
    lower = torch.where(codes == -8, -torch.inf, codes - 0.49)
    upper = torch.where(codes == 7, torch.inf, codes + 0.49)
    return normalized.clamp(min=lower, max=upper) * scale


@torch.no_grad()
def reference_for_parent(reference_state, parent_path):
    weights = reference_weights(reference_state)
    metadata = torch.load(parent_path, map_location="cpu", weights_only=True, mmap=True)["metadata"]
    transform = None
    candidate = metadata.get("candidate", "")
    if metadata.get("mode") == "d" and candidate.startswith("local-d-"):
        name = candidate[len("local-d-"):]
        diagnostic = json.loads((parent_path.parent / "diagonal_candidates.json").read_text())
        transform = dict(name=name, **diagnostic[name]["selected"])
        channels = transform["channels"]
        factor = transform["factor"]
        weights[name.replace("down_proj", "up_proj")][channels] /= factor
        weights[name][:, channels] *= factor
    return weights, transform


def prepare_student(model, records, master_reference=None):
    model.requires_grad_(False)
    for name, wrapper in wrappers(model).items():
        initial = (wrapper.module.weight if master_reference is None else
                   cell_reference_initialization(records[name], master_reference[name].to(wrapper.module.weight.device)))
        replacement = TrainableQuantLinear(initial, records[name]["scale"])
        wrapper.module = replacement
        wrapper.weight = replacement.weight
        wrapper.bias = None
        if name.endswith("down_proj"):
            wrapper.quantizer.scale = torch.nn.Parameter(wrapper.quantizer.scale.detach().clone())
        else:
            wrapper.quantizer.enable_scale_learning()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def parameter_groups(model):
    groups = dict(W=[], SA=[], SW=[], SP2=[])
    for name, wrapper in wrappers(model).items():
        groups["W"].append((name + ".module.weight", wrapper.module.weight))
        groups["SW"].append((name + ".module.quantizer.scale", wrapper.module.quantizer.scale))
        group = "SP2" if name.endswith("down_proj") else "SA"
        groups[group].append((name + ".quantizer.scale", wrapper.quantizer.scale))
    return groups


def teacher_model():
    from eval_utils.modeling_llama import LlamaForCausalLM

    config = AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(str(MODEL_PATH), config=config,
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    model.lm_head.weight.data = model.model.embed_tokens.weight.detach().clone()
    model.config.use_cache = False
    return model.requires_grad_(False).eval()


def chunk_objective(student_logits, teacher_logits, labels, temperature, ce_weight):
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    ce = functional.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]),
                                  labels.reshape(-1), reduction="sum")
    student_logp = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_logp = functional.log_softmax(teacher_logits / temperature, dim=-1)
    divergence = functional.kl_div(student_logp, teacher_logp, log_target=True,
                                  reduction="sum") * temperature ** 2
    return ce_weight * ce + (1 - ce_weight) * divergence, ce, divergence


def distillation_loss(student, teacher, ids, temperature=1.0, ce_weight=0.1, chunk_size=128,
                      offload_teacher_body=False):
    with torch.no_grad():
        if offload_teacher_body:
            teacher.model.to(ids.device)
        teacher_hidden = backbone(teacher, ids)
        if offload_teacher_body:
            teacher.model.cpu()
    student_hidden = backbone(student, ids)
    targets = ids[:, 1:]
    totals = [student_hidden.new_zeros((), dtype=torch.float32) for _ in range(3)]

    def measure(student_values, teacher_values, labels):
        student_logits = student.lm_head(student_values)
        with torch.no_grad():
            teacher_logits = teacher.lm_head(teacher_values)
        return chunk_objective(student_logits, teacher_logits, labels, temperature, ce_weight)

    for start in range(0, targets.shape[1], chunk_size):
        end = min(start + chunk_size, targets.shape[1])
        inputs = (student_hidden[:, start:end], teacher_hidden[:, start:end], targets[:, start:end])
        if torch.is_grad_enabled() and inputs[0].requires_grad:
            values = checkpoint(measure, *inputs, use_reentrant=False)
        else:
            values = measure(*inputs)
        totals = [total + value for total, value in zip(totals, values)]
    return tuple(total / targets.numel() for total in totals)


def make_optimizers(groups, args):
    weight_groups = [dict(params=[parameter for _, parameter in groups["W"]],
                         lr=args.weight_lr, initial_lr=args.weight_lr, name="W")]
    if getattr(args, "weight_optimizer", "sgd") == "adam":
        weight_optimizer = torch.optim.Adam(weight_groups, eps=1e-8, weight_decay=0.0, foreach=False)
    else:
        weight_optimizer = torch.optim.SGD(weight_groups, momentum=args.momentum, foreach=False)
    scale_groups = []
    for group in ("SA", "SW", "SP2"):
        for name, parameter in groups[group]:
            rate = args.relative_scale_lr * float(parameter.detach().mean())
            scale_groups.append(dict(params=[parameter], lr=rate, initial_lr=rate,
                                     name=group, parameter_name=name))
    return [weight_optimizer, torch.optim.Adam(scale_groups, eps=1e-12, weight_decay=0.0)]


def save_resume(model, optimizers, path, step, args):
    state = dict(parameters={name: parameter.detach().cpu().clone()
        for values in parameter_groups(model).values() for name, parameter in values},
        optimizers=[optimizer.state_dict() for optimizer in optimizers],
        metadata=dict(step=step, parent=str(args.parent), method="full-backbone W4A8 quantization-aware distillation",
                      data_start=args.data_start, accumulation=args.accumulation,
                      temperature=args.temperature, ce_weight=args.ce_weight,
                      reference_state=str(getattr(args, "reference_state", None)),
                      weight_optimizer=getattr(args, "weight_optimizer", "sgd")),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_resume(model, optimizers, path, args):
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    if metadata.get("weight_optimizer", "sgd") != getattr(args, "weight_optimizer", "sgd"):
        raise ValueError("Distillation resume changes weight optimizer")
    if len(state["optimizers"]) != len(optimizers):
        raise ValueError("Distillation resume optimizer coverage differs")
    for name in ("parent", "data_start", "accumulation", "temperature", "ce_weight"):
        expected = str(args.parent) if name == "parent" else getattr(args, name)
        if metadata[name] != expected:
            raise ValueError(f"Distillation resume changes {name}")
    parameters = {name: parameter for values in parameter_groups(model).values() for name, parameter in values}
    if parameters.keys() != state["parameters"].keys():
        raise ValueError("Distillation resume parameter coverage differs")
    with torch.no_grad():
        for name, value in state["parameters"].items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid distillation parameter: {name}")
            if name.endswith("scale") and not (value > 0).all():
                raise ValueError(f"Nonpositive scale: {name}")
        for name, value in state["parameters"].items():
            parameters[name].copy_(value.to(parameters[name]))
    for optimizer, saved in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return metadata["step"]


@torch.no_grad()
def export_student(model, parent_records, path, metadata):
    records = {}
    changed = {}
    for name, wrapper in wrappers(model).items():
        scale = wrapper.module.quantizer.scale.detach()
        codes = int4_codes(wrapper.module.weight, scale)
        parent_codes = unpack_int4(parent_records[name]["packed"], parent_records[name]["shape"]).to(codes.device)
        changed[name] = dict(changed_codes=int((codes != parent_codes).sum()), total_codes=codes.numel())
        records[name] = dict(packed=pack_int4(codes).cpu(), shape=tuple(codes.shape), scale=scale.cpu().clone())
    save_frozen(model, records, path, metadata)
    return changed


def evaluate_checkpoint(model, teacher, parent_records, args, step, probe, validation):
    directory = args.output / f"checkpoint-{step:04d}"
    directory.mkdir()
    probe_result = evaluate(model, probe)
    write_json(directory / "training_probe.json", probe_result)
    result = dict(step=step, training_probe=probe_result["nll"])
    if step not in args.validation_steps:
        return result
    progress(args.output, "export-for-validation", step=step)
    package = directory / "static_w4a8.pt"
    changes = export_student(model, parent_records, package,
        dict(parent=str(args.parent), method="quantization-aware distillation; original backbone weights trained",
             step=step, teacher=str(MODEL_PATH), temperature=args.temperature, ce_weight=args.ce_weight))
    write_json(directory / "code_changes.json", changes)
    model.cpu()
    teacher.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        frozen, records = load_static(package)
    progress(args.output, "full-validation", step=step)
    measured = full_validation(frozen, validation)
    measured.update(method="full-backbone quantization-aware distillation, not frozen-weight PTQ",
        update_step=step, parent=str(args.parent),
        format="112 W4 / 96 static INT8 / 16 static SP2; BF16 KV16 prefill")
    write_json(directory / "validation.json", measured)
    result.update(validation_nll=measured["nll"], validation_ppl=measured["ppl"],
                  changed_codes=sum(row["changed_codes"] for row in changes.values()))
    del frozen, records
    gc.collect()
    torch.cuda.empty_cache()
    model.cuda()
    teacher.cuda()
    return result


def train(args):
    train_windows, _, probe, validation, data = data_windows(args.output)
    student, parent_records = load_static(args.parent)
    reference = None
    reference_state = getattr(args, "reference_state", None)
    if reference_state is not None:
        reference, transform = reference_for_parent(reference_state, args.parent)
        write_json(args.output / "master_initialization.json", dict(reference_state=str(reference_state),
            parent=str(args.parent), diagonal_transform=transform,
            method="project current-R FP reference into parent INT4 cells; retain parent initial quantized forward"))
    prepare_student(student, parent_records, reference)
    del reference
    groups = parameter_groups(student)
    optimizers = make_optimizers(groups, args)
    teacher = teacher_model().cuda()
    start_step = load_resume(student, optimizers, args.resume, args) if args.resume is not None else 0
    write_json(args.output / "parameter_coverage.json", {
        group: dict(tensors=len(values), elements=sum(parameter.numel() for _, parameter in values),
                    dtypes=sorted({str(parameter.dtype) for _, parameter in values}))
        for group, values in groups.items()})
    if start_step == 0:
        write_json(args.output / "initial_probe.json", evaluate(student, probe))
    student.train()
    started = time.monotonic()
    completed_step = start_step
    results = []
    target_reached = False
    parameters = [parameter for values in groups.values() for _, parameter in values]
    for step in range(start_step, args.steps):
        multiplier = schedule(step, args.schedule_steps or args.steps, args.warmup)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            optimizer.zero_grad(set_to_none=True)
        before = {name: parameter.detach().reshape(-1)[::max(parameter.numel() // 64, 1)][:64].clone()
                  for name, parameter in groups["W"]}
        batches = []
        for microbatch in range(args.accumulation):
            index = (args.data_start + step * args.accumulation + microbatch) % train_windows.shape[0]
            ids = train_windows[index:index + 1].cuda()
            loss, ce, divergence = distillation_loss(student, teacher, ids, args.temperature, args.ce_weight,
                offload_teacher_body=getattr(args, "offload_teacher_body", False))
            (loss / args.accumulation).backward()
            measured = dict(window=index, objective=float(loss.detach()), ce=float(ce.detach()),
                            kl=float(divergence.detach()), predicted_tokens=ids.numel() - 1)
            if not all(math.isfinite(measured[key]) for key in ("objective", "ce", "kl")):
                raise RuntimeError("Non-finite distillation loss")
            batches.append(measured)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        gradient_tensors = {name: sum(parameter.grad is not None for _, parameter in values)
                            for name, values in groups.items()}
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for name in ("SA", "SW", "SP2"):
                for _, parameter in groups[name]:
                    parameter.clamp_(min=1e-8)
            sampled_changes = {name: int((parameter.detach().reshape(-1)[::max(parameter.numel() // 64, 1)][:64]
                                         != before[name]).sum()) for name, parameter in groups["W"]}
        completed_step = step + 1
        append_json(args.output / "training.jsonl", dict(step=completed_step, microbatches=batches,
            objective=sum(row["objective"] for row in batches) / len(batches),
            ce=sum(row["ce"] for row in batches) / len(batches),
            kl=sum(row["kl"] for row in batches) / len(batches),
            cumulative_train_tokens=completed_step * args.accumulation * 2048,
            gradient_norm_before_clip=norm, gradient_tensors=gradient_tensors,
            sampled_master_weight_changes=sampled_changes,
            learning_rates={group.get("parameter_name", group["name"]): group["lr"]
                            for optimizer in optimizers for group in optimizer.param_groups},
            elapsed_seconds=time.monotonic() - started))
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if completed_step % args.resume_every == 0 or completed_step in args.checkpoints or completed_step == args.steps:
            save_resume(student, optimizers, args.output / "resume.pt", completed_step, args)
        progress(args.output, "distillation-training", step=completed_step, total=args.steps,
                 ce=sum(row["ce"] for row in batches) / len(batches), elapsed_seconds=time.monotonic() - started)
        if completed_step in args.checkpoints or completed_step == args.steps:
            result = evaluate_checkpoint(student, teacher, parent_records, args, completed_step, probe, validation)
            results.append(result)
            write_json(args.output / "checkpoints.json", results)
            student.train()
            if result.get("validation_ppl", math.inf) <= args.target_ppl:
                target_reached = True
                break
    write_json(args.output / "result.json", dict(completed_steps=completed_step, target_reached=target_reached,
        parent=str(args.parent), checkpoints=results, method="quantization-aware distillation, not frozen-weight PTQ"))
    progress(args.output, "completed", steps=completed_step, target_reached=target_reached,
             elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight-lr", type=float, default=0.1)
    parser.add_argument("--weight-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--offload-teacher-body", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--relative-scale-lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--data-start", type=int, default=800)
    parser.add_argument("--resume-every", type=int, default=25)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--validation-steps", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--target-ppl", type=float, default=14.634657725643203)
    args = parser.parse_args()
    if args.steps < 1 or args.accumulation < 1 or args.resume_every < 1 or args.temperature <= 0 or not 0 <= args.ce_weight <= 1:
        parser.error("Invalid distillation optimization or objective settings")
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = args.output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    (snapshot / "tracked.diff").write_bytes(subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "diff", "--binary", "HEAD"]))
    write_json(args.output / "settings.json", dict(arguments={name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()}, source=source_record(), pid=os.getpid(),
        teacher="original unrotated BF16; no norm fusion; frozen; no validation data",
        optimization="112 FP32 master backbone weights with declared SGD/Adam; 224 static scale tensors with Adam; no online R or new inference operations"))
    try:
        train(args)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
