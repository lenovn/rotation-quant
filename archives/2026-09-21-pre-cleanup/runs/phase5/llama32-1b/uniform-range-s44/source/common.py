import json
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, LlamaTokenizerFast, set_seed

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from experiments.phase3.quantization import (
    Phase3WeightQuantizer, SP2Quantizer, int4_codes, pack_int4, sp2_levels,
    sp2_project, unpack_int4,
)
from experiments.phase3.architecture import load_model, tokenizer as model_tokenizer, evaluator_for
from utils.fuse_norm_utils import fuse_layer_norms
from utils.hadamard_utils import random_hadamard_matrix
from utils.quant_phase import QuantPhase
from utils.quant_utils import ActQuantWrapper, RotationStaticActQuantizer, add_actquant

MODEL_PATH = Path(os.environ.get("PHASE5_MODEL_PATH", str(PROJECT_ROOT / "cache/models/llama-3.2-1b-instruct")))
SEED = int(os.environ.get("PHASE5_SEED", "42"))
DATA_PATH = PROJECT_ROOT / "cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")
        output.flush()


class Rotation(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(value.float())


def wrappers(model):
    return {name: module for name, module in model.named_modules()
            if name.startswith("model.layers.") and isinstance(module, ActQuantWrapper)}


def format_description(model):
    from collections import Counter
    counts = Counter("SP2" if isinstance(wrapper.quantizer, SP2Quantizer) else "INT8"
                     for wrapper in wrappers(model).values())
    return f"{len(wrappers(model))} W4 / " + " / ".join(
        f"{count} static {kind}" for kind, count in sorted(counts.items())) + "; BF16 KV16 prefill"


def rotated_linears(model):
    for index, layer in enumerate(model.model.layers):
        for suffix, wrapper, rotation2, transpose in (
            ("self_attn.q_proj", layer.self_attn.q_proj, None, False),
            ("self_attn.k_proj", layer.self_attn.k_proj, None, False),
            ("self_attn.v_proj", layer.self_attn.v_proj, layer.self_attn.R2.weight, False),
            ("self_attn.o_proj", layer.self_attn.o_proj, layer.self_attn.R2.weight, True),
            ("mlp.gate_proj", layer.mlp.gate_proj, None, False),
            ("mlp.up_proj", layer.mlp.up_proj, None, False),
            ("mlp.down_proj", layer.mlp.down_proj, None, True),
        ):
            yield f"model.layers.{index}.{suffix}", wrapper, rotation2, transpose


def backbone(model, ids):
    kwargs = dict(use_cache=False, return_dict=True)
    if model.config.model_type == "qwen3":
        if hasattr(model, "R1"):
            embeddings = model.model.embed_tokens(ids)
            embeddings = (embeddings.double() @ model.R1.weight.double()).to(embeddings.dtype)
            hidden = model.model(inputs_embeds=embeddings, **kwargs)[0]
        else:
            hidden = model.model(ids, **kwargs)[0]
    else:
        kwargs["quant_phase"] = QuantPhase.PREFILL
        if hasattr(model, "R1"):
            kwargs["R1"] = model.R1.weight
        hidden = model.model(ids, **kwargs)[0]
    if hasattr(model, "R1"):
        hidden = (hidden.double() @ model.R1.weight.T.double()).to(hidden.dtype)
    return hidden


def token_nll(model, ids, chunk_size=128):
    hidden = backbone(model, ids)
    target = ids[:, 1:]
    total = hidden.new_zeros((), dtype=torch.float32)

    def chunk_loss(values, labels):
        logits = model.lm_head(values).float()
        return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                        labels.reshape(-1), reduction="sum")

    for start in range(0, target.shape[1], chunk_size):
        values = hidden[:, start:start + chunk_size]
        labels = target[:, start:start + chunk_size]
        if values.shape[1] != labels.shape[1]:
            values = values[:, :labels.shape[1]]
        if torch.is_grad_enabled() and values.requires_grad:
            total = total + checkpoint(chunk_loss, values, labels, use_reentrant=False)
        else:
            total = total + chunk_loss(values, labels)
    return total / target.numel()


@torch.no_grad()
def evaluate(model, windows, progress=None):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    records = []
    try:
        for index, ids in enumerate(windows):
            value = float(token_nll(model, ids.to(device)))
            if not math.isfinite(value):
                raise RuntimeError("Non-finite evaluation NLL")
            records.append(dict(window=index, tokens=ids.numel(), predicted_tokens=ids.numel() - 1, nll=value))
            if progress is not None and (index % 16 == 0 or index + 1 == len(windows)):
                progress(index + 1, len(windows))
    finally:
        model.train(was_training)
    predicted = sum(record["predicted_tokens"] for record in records)
    nll = sum(record["nll"] * record["predicted_tokens"] for record in records) / predicted
    return dict(nll=nll, ppl=math.exp(nll), predicted_tokens=predicted,
                token_count=sum(record["tokens"] for record in records), windows=records)


@torch.no_grad()
def full_validation(model, windows):
    from utils.eval_utils import evaluator

    path = PROJECT_ROOT / "scripts/phase2/validation_acceptance.py"
    specification = importlib.util.spec_from_file_location("phase3_existing_acceptance", path)
    acceptance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(acceptance)
    model.seqlen = 2048
    model.eval()
    encoding = SimpleNamespace(input_ids=torch.cat(windows, dim=1))
    arguments = SimpleNamespace(eval_nsamples=None, bsz=1, capture_layer_io=False)
    result = acceptance.evaluate_full_validation(model, encoding, "cuda", arguments, evaluator_for(model))
    result["evaluation_precision"] = "existing evaluator: BF16 logits CE, per-token loss to FP32; log(float32 PPL), weighted by predicted tokens"
    result["acceptance_source"] = str(path)
    result["evaluator_source"] = str(SOURCE_ROOT / ("experiments/phase3/architecture.py" if model.config.model_type == "qwen3" else "utils/eval_utils.py"))
    return result


def data_windows(output=None):
    from datasets import Dataset

    tokenizer = model_tokenizer(MODEL_PATH)
    train = Dataset.from_file(str(DATA_PATH / "wikitext-train.arrow"))
    validation = Dataset.from_file(str(DATA_PATH / "wikitext-validation.arrow"))
    rows = tokenizer(train["text"], add_special_tokens=False)["input_ids"]
    train_ids = torch.tensor([token for row in rows for token in row], dtype=torch.long)
    count = train_ids.numel() // 2048
    windows = train_ids[:count * 2048].reshape(count, 2048)
    validation_ids = tokenizer("\n\n".join(validation["text"]), return_tensors="pt").input_ids
    validation_windows = [validation_ids[:, start:start + 2048]
                          for start in range(0, validation_ids.numel() - 1, 2048)]
    if MODEL_PATH.name == "llama-3.2-1b-instruct" and (validation_ids.numel() != 252852 or sum(ids.numel() - 1 for ids in validation_windows) != 252728):
        raise ValueError("Validation tokenizer/protocol differs from the accepted source")
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(count - 8, generator=generator)[:32]
    calibration = [windows[int(index):int(index) + 1, :128] for index in indices]
    probe = [windows[index:index + 1] for index in range(count - 8, count - 4)]
    metadata = dict(model_path=str(MODEL_PATH), dataset_cache=str(DATA_PATH),
        train_protocol="tokenize each raw row without BOS/EOS, concatenate IDs, disjoint 2048 windows, drop tail, fixed order",
        validation_protocol="double-newline joined raw validation; independent 2048 windows including tail; token-weighted NLL",
        train_windows=count, train_tokens=int(train_ids.numel()), dropped_train_tail=int(train_ids.numel() % 2048),
        calibration_window_indices=indices.tolist(), calibration_length=128,
        probe_indices=list(range(count - 8, count - 4)),
        validation_tokens=validation_ids.numel(), validation_targets=sum(ids.numel() - 1 for ids in validation_windows), seed=SEED)
    if output is not None:
        write_json(Path(output) / "data.json", metadata)
    return windows[:-8], calibration, probe, validation_windows, metadata


def build_training_model(initial_state=None):
    set_seed(SEED)
    model = load_model(MODEL_PATH, training=True, untie=True)
    config = model.config
    fuse_layer_norms(model)
    add_actquant(model)
    model.requires_grad_(False)
    model.R1 = Rotation(random_hadamard_matrix(config.hidden_size, "cpu"))
    for layer in model.model.layers:
        layer.self_attn.R2 = Rotation(random_hadamard_matrix(config.head_dim, "cpu"))
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer = RotationStaticActQuantizer()
        weight_quantizer = Phase3WeightQuantizer()
        weight_quantizer.configure(4, perchannel=True, sym=True, mse=False, weight_groupsize=-1)
        wrapper.module.quantizer = weight_quantizer
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if initial_state is not None:
        saved = torch.load(initial_state, map_location="cpu", weights_only=True)["parameters"]
        for name, wrapper in wrappers(model).items():
            weight_scale = saved[name + ".module.quantizer.scale"]
            activation_scale = saved[name + ".quantizer.scale"]
            if weight_scale.shape != (wrapper.module.out_features, 1):
                raise ValueError(f"Saved SW must be per-output-channel: {name}")
            if activation_scale.shape != (1,):
                raise ValueError(f"Saved activation scale must be per-tensor: {name}")
            wrapper.module.quantizer.load_scale(weight_scale)
            wrapper.module.quantizer.enable_scale_learning()
            if name.endswith("down_proj"):
                wrapper.quantizer = SP2Quantizer(float(saved[name + ".quantizer.scale"]) * 127, learnable=True)
            else:
                wrapper.quantizer.load_scale(saved[name + ".quantizer.scale"])
                wrapper.quantizer.enable_scale_learning()
        load_state(model, initial_state)
    return model


@torch.no_grad()
def initialize_scales(model, calibration):
    for name, wrapper, rotation2, transpose in rotated_linears(model):
        weight = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose)
        wrapper.module.quantizer.refresh_current_weight(weight)
        wrapper.module.quantizer.bits = 16
        wrapper.quantizer.begin_calibration()
    model.eval()
    for ids in calibration:
        backbone(model, ids.to(next(model.parameters()).device))
    for name, wrapper in wrappers(model).items():
        wrapper.quantizer.finish_calibration()
        if name.endswith("down_proj"):
            alpha = float(wrapper.quantizer.scale) * 127
            wrapper.quantizer = SP2Quantizer(alpha, learnable=True).to(next(model.parameters()).device)
            wrapper.quantizer.bits = 16
        else:
            wrapper.quantizer.enable_scale_learning()
        wrapper.module.quantizer.bits = 4
        wrapper.module.quantizer.enable_scale_learning()


def learned_parameters(model):
    groups = {"R": [], "SA": [], "SW": [], "SP2": []}
    for name, parameter in model.named_parameters():
        if name == "R1.weight" or name.endswith("self_attn.R2.weight"):
            groups["R"].append((name, parameter))
    for name, wrapper in wrappers(model).items():
        group = "SP2" if name.endswith("down_proj") else "SA"
        groups[group].append((name + ".quantizer.scale", wrapper.quantizer.scale))
        groups["SW"].append((name + ".module.quantizer.scale", wrapper.module.quantizer.scale))
    layers = model.config.num_hidden_layers
    expected = dict(R=1 + layers, SA=6 * layers, SW=7 * layers, SP2=layers)
    if {name: len(parameters) for name, parameters in groups.items()} != expected:
        raise ValueError("Incorrect Phase3 parameter coverage")
    return groups


def training_mode(model, route, step, switch_step):
    use_weight = route != "A" or step >= switch_step
    for name, wrapper in wrappers(model).items():
        wrapper.module.quantizer.bits = 4 if use_weight else 16
        wrapper.module.quantizer.scale.requires_grad_(use_weight)
        if name.endswith("down_proj"):
            wrapper.quantizer.bits = 8 if route == "C" else 16
            wrapper.quantizer.scale.requires_grad_(route == "C")
    return use_weight


@torch.no_grad()
def reset_weight_scales(model):
    for name, wrapper, rotation2, transpose in rotated_linears(model):
        weight = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose)
        wrapper.module.quantizer.scale.copy_(weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7)


def save_state(model, path, **metadata):
    state = {"parameters": {name: parameter.detach().cpu().clone()
              for parameters in learned_parameters(model).values() for name, parameter in parameters},
             "metadata": metadata}
    torch.save(state, path)
    return state


@torch.no_grad()
def load_state(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    expected = {name for values in learned_parameters(model).values() for name, _ in values}
    if set(state["parameters"]) != expected:
        raise ValueError("Checkpoint parameter coverage mismatch")
    for name, value in state["parameters"].items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid saved parameter: {name}")
        if name.endswith(".scale") and not (value > 0).all():
            raise ValueError(f"Nonpositive saved scale: {name}")
        parameters[name].copy_(value.to(parameters[name]))
    return state


@torch.no_grad()
def frozen_model(model, current_minmax=False):
    device = next(model.parameters()).device
    frozen = load_model(MODEL_PATH, untie=True)
    frozen.lm_head.weight.data = model.lm_head.module.weight.detach().cpu().clone()
    frozen.model.embed_tokens.weight.data = model.model.embed_tokens.weight.detach().cpu().clone()
    for index, layer in enumerate(frozen.model.layers):
        layer.input_layernorm.weight.data = model.model.layers[index].input_layernorm.weight.detach().cpu().clone()
        layer.post_attention_layernorm.weight.data = model.model.layers[index].post_attention_layernorm.weight.detach().cpu().clone()
    frozen.model.norm.weight.data = model.model.norm.weight.detach().cpu().clone()
    if model.config.model_type == "qwen3":
        for source, target in zip(model.model.layers, frozen.model.layers):
            for name in ("q_norm", "k_norm"):
                getattr(target.self_attn, name).load_state_dict(getattr(source.self_attn, name).state_dict())
    rotation1 = model.R1.weight.detach().double()
    for module in (frozen.model.embed_tokens, frozen.lm_head):
        module.weight.data = (module.weight.to(device).double() @ rotation1).to(torch.bfloat16).cpu()
    add_actquant(frozen)
    records = {}
    for name, wrapper, rotation2, transpose in rotated_linears(model):
        weight = wrapper.module.rotated_weight(model.R1.weight, rotation2, transpose)
        scale = (weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 7
                 if current_minmax else wrapper.module.quantizer.scale.detach())
        codes = int4_codes(weight, scale)
        destination = frozen.get_submodule(name)
        destination.module.weight.data = (codes.float() * scale).to(torch.bfloat16).cpu()
        records[name] = dict(packed=pack_int4(codes).cpu(), shape=tuple(weight.shape), scale=scale.cpu().clone())
        if name.endswith("down_proj"):
            destination.quantizer = SP2Quantizer(float(wrapper.quantizer.alpha))
            destination.quantizer.bits = wrapper.quantizer.bits
        else:
            destination.quantizer = RotationStaticActQuantizer()
            destination.quantizer.load_scale(wrapper.quantizer.scale.detach().cpu())
    frozen.requires_grad_(False).eval()
    frozen.config.use_cache = False
    return frozen.to(device), records


@torch.no_grad()
def calibrate_sp2(model, calibration, directory=None):
    device = next(model.parameters()).device
    downs = {name: wrapper for name, wrapper in wrappers(model).items() if name.endswith("down_proj")}
    samples = {name: [] for name in downs}
    maximum = {name: 0.0 for name in downs}

    def capture(name):
        def hook(module, inputs):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            maximum[name] = max(maximum[name], float(values.abs().max()))
            samples[name].append(values[::8].cpu())
        return hook

    for wrapper in downs.values():
        wrapper.quantizer.bits = 16
    handles = [wrapper.register_forward_pre_hook(capture(name)) for name, wrapper in downs.items()]
    try:
        for ids in calibration:
            backbone(model, ids.to(device))
    finally:
        for handle in handles:
            handle.remove()
    results = {}
    levels = sp2_levels().to(device)
    for name, wrapper in downs.items():
        values = torch.cat(samples.pop(name)).to(device)
        reference = values.float()
        weight = wrapper.module.weight.float()
        bound = max(maximum[name], 1e-8)
        candidates = []

        def measure(log_ratio):
            alpha = bound * 2 ** log_ratio
            difference = sp2_project(values, alpha, levels).float() - reference
            mse = float(functional.linear(difference, weight).square().mean())
            candidates.append(dict(alpha=alpha, output_mse=mse, log2_ratio=log_ratio))
            return mse

        coarse = torch.linspace(-16.0, 1.0, 33).tolist()
        losses = [measure(log_ratio) for log_ratio in coarse]
        best = min(range(len(losses)), key=losses.__getitem__)
        refined = torch.linspace(coarse[max(0, best - 1)], coarse[min(32, best + 1)], 17).tolist()
        for log_ratio in refined:
            measure(log_ratio)
        selected = min(candidates, key=lambda record: record["output_mse"])
        wrapper.quantizer = SP2Quantizer(selected["alpha"]).to(device)
        results[name] = dict(selected=selected, full_absmax=bound, candidates=candidates,
                             sampled_rows=values.shape[0])
    if directory is not None:
        write_json(Path(directory) / "sp2_calibration.json", results)
    return results


@torch.no_grad()
def save_frozen(model, records, path, metadata):
    high_precision = {name: tensor.detach().cpu().clone()
                      for name, tensor in model.state_dict().items()
                      if not any(name.startswith(prefix + ".") for prefix in records)}
    activation = {name: dict(format="sp2", alpha=float(wrapper.quantizer.alpha))
                  if isinstance(wrapper.quantizer, SP2Quantizer) else dict(format="int8", scale=wrapper.quantizer.scale.cpu().clone())
                  for name, wrapper in wrappers(model).items()}
    torch.save(dict(weights=records, activation=activation, high_precision=high_precision,
                    metadata=metadata, config=model.config.to_dict()), path)


@torch.no_grad()
def reload_frozen(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = set(wrappers(model))
    if set(state["weights"]) != expected or set(state["activation"]) != expected:
        raise ValueError("Frozen model coverage mismatch")
    high_precision = {name: tensor for name, tensor in model.state_dict().items()
                      if not any(name.startswith(prefix + ".") for prefix in expected)}
    if set(state["high_precision"]) != set(high_precision):
        raise ValueError("Frozen high-precision boundary coverage mismatch")
    for name, tensor in state["high_precision"].items():
        if tensor.shape != high_precision[name].shape:
            raise ValueError(f"Frozen high-precision shape mismatch: {name}")
    for name, record in state["weights"].items():
        weight = model.get_submodule(name).module.weight
        if tuple(record["shape"]) != tuple(weight.shape):
            raise ValueError(f"Frozen weight shape mismatch: {name}")
        scale = record["scale"]
        if scale.shape != (weight.shape[0], 1) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"Invalid frozen weight scale: {name}")
        if record["packed"].dtype != torch.uint8 or record["packed"].numel() * 2 != weight.numel():
            raise ValueError(f"Invalid packed weight: {name}")
        activation = state["activation"][name]
        if activation["format"] == "sp2":
            if not name.endswith("down_proj") or not math.isfinite(float(activation["alpha"])) or activation["alpha"] <= 0:
                raise ValueError(f"Invalid frozen SP2 range: {name}")
        else:
            if activation["format"] != "int8":
                raise ValueError(f"Invalid frozen activation format: {name}")
            scale = activation["scale"]
            if scale.shape != (1,) or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"Invalid frozen activation scale: {name}")
    device = next(model.parameters()).device
    model.load_state_dict(state["high_precision"], strict=False)
    for name, record in state["weights"].items():
        wrapper = model.get_submodule(name)
        codes = unpack_int4(record["packed"], record["shape"])
        wrapper.module.weight.copy_((codes.float() * record["scale"]).to(wrapper.module.weight))
        activation = state["activation"][name]
        if activation["format"] == "sp2":
            wrapper.quantizer = SP2Quantizer(activation["alpha"]).to(device)
        else:
            wrapper.quantizer = RotationStaticActQuantizer().to(device)
            wrapper.quantizer.load_scale(activation["scale"].to(device))
    return state
