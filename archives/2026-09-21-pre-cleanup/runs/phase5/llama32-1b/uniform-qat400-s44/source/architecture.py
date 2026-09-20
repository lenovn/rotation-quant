"""Model-family entry points; Qwen attention/norm/RoPE stay in Transformers."""
import weakref

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train_utils.quant_linear import QuantizeLinear


def load_model(path, training=False, untie=False):
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if config.model_type == "qwen3":
        # Load the original tied checkpoint first, then clone the head explicitly.
        model = AutoModelForCausalLM.from_pretrained(str(path), config=config,
            torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
        if untie:
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            model.config.tie_word_embeddings = False
        if training:
            install_rotation_linears(model)
        return model
    if config.model_type != "llama":
        raise ValueError(f"Unsupported Phase5 architecture: {config.model_type}")
    if training:
        from train_utils.modeling_llama_quant import LlamaForCausalLM
    else:
        from eval_utils.modeling_llama import LlamaForCausalLM
    tied = config.tie_word_embeddings
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(str(path), config=config,
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    if tied:
        model.lm_head.weight.data = model.model.embed_tokens.weight.detach().clone()
    return model


def install_rotation_linears(model):
    """Pass live rotations to existing QuantizeLinear, including checkpoint recompute."""
    root_ref = weakref.ref(model)
    for layer in model.model.layers:
        attention_ref = weakref.ref(layer.self_attn)
        for parent, name, use_r2, transpose in (
            (layer.self_attn, "q_proj", False, False),
            (layer.self_attn, "k_proj", False, False),
            (layer.self_attn, "v_proj", True, False),
            (layer.self_attn, "o_proj", True, True),
            (layer.mlp, "gate_proj", False, False),
            (layer.mlp, "up_proj", False, False),
            (layer.mlp, "down_proj", False, True),
        ):
            old = getattr(parent, name)
            linear = QuantizeLinear(old.in_features, old.out_features,
                                    bias=old.bias is not None, device="meta")
            linear.weight = old.weight
            linear.bias = old.bias

            def rotate(module, args, root_ref=root_ref, attention_ref=attention_ref,
                       use_r2=use_r2, transpose=transpose):
                root = root_ref()
                if not hasattr(root, "R1"):
                    return args
                r2 = attention_ref().R2.weight if use_r2 else None
                return (args[0], root.R1.weight, r2, transpose)

            linear.register_forward_pre_hook(rotate)
            setattr(parent, name, linear)


def tokenizer(path):
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    tokenizer_class = AutoTokenizer
    if config.model_type == "llama":
        # The historical Llama fast class honours add_bos_token=False; the
        # generic class selected by this checkpoint's metadata adds a BOS.
        from transformers import LlamaTokenizerFast
        tokenizer_class = LlamaTokenizerFast
    return tokenizer_class.from_pretrained(str(path), local_files_only=True,
        model_max_length=2048, padding_side="right", use_fast=True,
        add_eos_token=False, add_bos_token=False)


@torch.no_grad()
def qwen_evaluator(model, enc, device, args):
    """Official full forward, preserving the historical BF16 CE/FP32 PPL protocol."""
    model.eval().to(device)
    ids = enc.input_ids.reshape(-1, model.seqlen)
    losses = []
    for row in ids:
        batch = row.unsqueeze(0).to(device)
        hidden = model.model(batch, use_cache=False, return_dict=True)[0]
        logits = model.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].permute(0, 2, 1),
                                                batch[:, 1:], reduction="none")
        losses.append(loss.float().mean(dim=1))
        del hidden, logits, loss
    return torch.exp(torch.cat(losses).mean()).item()


def evaluator_for(model):
    if model.config.model_type == "qwen3":
        return qwen_evaluator
    from utils.eval_utils import evaluator
    return evaluator
