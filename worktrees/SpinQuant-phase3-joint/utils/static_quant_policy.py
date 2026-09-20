# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared input-activation quantization policy for training and evaluation."""

from typing import Dict, Tuple

from utils import quant_utils, utils


LEGACY_DYNAMIC = "legacy_dynamic"
DUAL_PHASE_STATIC = "dual_phase_static"
ROTATION_STATIC = "rotation_static"
STATIC_ENCODINGS = ("symmetric_int8", "asymmetric_uint8")
STATIC_GRANULARITY = "per_tensor"

_TARGET_PATHS = (
    ("self_attn.q_proj", ("self_attn", "q_proj")),
    ("self_attn.k_proj", ("self_attn", "k_proj")),
    ("self_attn.v_proj", ("self_attn", "v_proj")),
    ("self_attn.o_proj", ("self_attn", "o_proj")),
    ("mlp.gate_proj", ("mlp", "gate_proj")),
    ("mlp.up_proj", ("mlp", "up_proj")),
    ("mlp.down_proj", ("mlp", "down_proj")),
)


def _activation_mode(args) -> str:
    mode = getattr(args, "a_quant_mode", LEGACY_DYNAMIC)
    if mode not in (LEGACY_DYNAMIC, DUAL_PHASE_STATIC, ROTATION_STATIC):
        raise ValueError(
            "a_quant_mode must be 'legacy_dynamic', 'dual_phase_static', "
            "or 'rotation_static'"
        )
    return mode


def validate_static_activation_args(args) -> None:
    """Validate static-A8 arguments without requiring a model or runtime setup."""

    if _activation_mode(args) != DUAL_PHASE_STATIC:
        return

    encoding = getattr(args, "a_static_encoding", None)
    if encoding not in STATIC_ENCODINGS:
        raise ValueError(
            "dual_phase_static requires an explicit a_static_encoding of "
            "'symmetric_int8' or 'asymmetric_uint8'"
        )
    if getattr(args, "a_bits", 16) != 8:
        raise ValueError("dual_phase_static requires a_bits == 8")
    if getattr(args, "a_groupsize", -1) != -1:
        raise ValueError(
            f"dual_phase_static uses policy-defined {STATIC_GRANULARITY} granularity; "
            "legacy a_groupsize must remain at its default -1"
        )

    clip_ratio = getattr(args, "a_clip_ratio", 1.0)
    try:
        valid_clip_ratio = 0 < float(clip_ratio) <= 1
    except (TypeError, ValueError):
        valid_clip_ratio = False
    if not valid_clip_ratio:
        raise ValueError("a_clip_ratio must be in (0, 1]")

    if getattr(args, "int8_down_proj", False):
        raise ValueError("dual_phase_static rejects int8_down_proj")
    if getattr(args, "act_order", False):
        raise ValueError("dual_phase_static rejects act_order")
    if getattr(args, "k_bits", 16) < 16:
        raise ValueError(
            "dual_phase_static rejects k_bits < 16 until static K quantization exists"
        )
    if getattr(args, "v_bits", 16) < 16:
        raise ValueError(
            "dual_phase_static rejects v_bits < 16 until static V/cache quantization exists"
        )
    if getattr(args, "save_qmodel_path", None):
        raise ValueError(
            "dual_phase_static rejects save_qmodel_path until complete static export exists"
        )
    if getattr(args, "export_to_et", False):
        raise ValueError(
            "dual_phase_static rejects export_to_et until complete static export exists"
        )


def validate_static_activation_model_config(args, model_config) -> None:
    """Validate model-config constraints after args-only validation succeeds."""

    if _activation_mode(args) != DUAL_PHASE_STATIC:
        return
    if getattr(model_config, "pretraining_tp", 1) > 1:
        raise ValueError("dual_phase_static rejects pretraining_tp > 1")


def validate_static_activation_request(args, model_config) -> None:
    """Apply both static-A8 validation stages as a defense-in-depth check."""

    validate_static_activation_args(args)
    validate_static_activation_model_config(args, model_config)


def _get_nested_attr(module, path):
    current = module
    for component in path:
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def _collect_static_targets(model) -> Dict[str, quant_utils.ActQuantWrapper]:
    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if (
        not isinstance(num_hidden_layers, int)
        or isinstance(num_hidden_layers, bool)
        or num_hidden_layers < 1
    ):
        raise RuntimeError("Static A8 coverage requires a positive num_hidden_layers")

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != num_hidden_layers:
        raise RuntimeError(
            "Static A8 coverage does not match model.config.num_hidden_layers"
        )

    expected: Dict[str, quant_utils.ActQuantWrapper] = {}
    seen_wrapper_ids = set()
    for layer_index, layer in enumerate(layers):
        for suffix, path in _TARGET_PATHS:
            name = f"model.layers.{layer_index}.{suffix}"
            wrapper = _get_nested_attr(layer, path)
            if not isinstance(wrapper, quant_utils.ActQuantWrapper):
                raise RuntimeError(f"Static A8 target is missing or unwrapped: {name}")
            wrapper_id = id(wrapper)
            if wrapper_id in seen_wrapper_ids:
                raise RuntimeError(f"Static A8 target is duplicated: {name}")
            seen_wrapper_ids.add(wrapper_id)
            expected[name] = wrapper

    expected_count = num_hidden_layers * len(_TARGET_PATHS)
    if len(expected) != expected_count:
        raise RuntimeError(
            f"Static A8 coverage expected {expected_count} targets, found {len(expected)}"
        )

    # Preserve aliases while enumerating so an extra registration of an otherwise
    # valid projection cannot disappear through named_modules' normal de-duplication.
    discovered_entries = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, quant_utils.ActQuantWrapper)
    ]
    discovered_names = [name for name, _ in discovered_entries if name != "lm_head"]
    expected_names = set(expected)
    if (
        len(discovered_names) != expected_count
        or set(discovered_names) != expected_names
    ):
        raise RuntimeError(
            "Static A8 coverage contains missing, extra, or duplicated target projections"
        )

    discovered = dict(discovered_entries)
    if any(discovered[name] is not wrapper for name, wrapper in expected.items()):
        raise RuntimeError("Static A8 coverage target identity is inconsistent")
    lm_head = discovered.get("lm_head")
    if lm_head is not None and id(lm_head) in seen_wrapper_ids:
        raise RuntimeError("Static A8 coverage contains a duplicated lm_head target")
    return expected


def _configure_legacy_inputs(args, model) -> Tuple[str, ...]:
    if getattr(args, "a_bits", 16) >= 16 and getattr(args, "v_bits", 16) >= 16:
        return ()

    qlayers = quant_utils.find_qlayers(
        model, layers=[quant_utils.ActQuantWrapper]
    )
    down_proj_groupsize = -1
    if args.a_groupsize > 0:
        down_proj_groupsize = utils.llama_down_proj_groupsize(
            model, args.a_groupsize
        )

    num_heads = model.config.num_attention_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads

    for name, wrapper in qlayers.items():
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize

        if "o_proj" in name:
            layer_groupsize = head_dim
        if "lm_head" in name:
            layer_input_bits = 16
        if "down_proj" in name:
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        wrapper.quantizer.configure(
            bits=layer_input_bits,
            groupsize=layer_groupsize,
            sym=not args.a_asym,
            clip_ratio=args.a_clip_ratio,
        )
    return tuple(qlayers)


def configure_input_activation_quantizers(args, model) -> Tuple[str, ...]:
    """Configure every input quantizer from the single shared A8 policy."""

    mode = _activation_mode(args)
    if mode == LEGACY_DYNAMIC:
        return _configure_legacy_inputs(args, model)

    if mode == ROTATION_STATIC:
        targets = _collect_static_targets(model)
        for wrapper in targets.values():
            wrapper.quantizer = quant_utils.RotationStaticActQuantizer(
                clip_ratio=getattr(args, "a_clip_ratio", 1.0)
            )
        return tuple(targets)

    validate_static_activation_request(args, model.config)
    targets = _collect_static_targets(model)
    for wrapper in targets.values():
        wrapper.quantizer.configure_static(
            encoding=args.a_static_encoding,
            clip_ratio=args.a_clip_ratio,
        )
    return tuple(targets)


def load_rotation_static_activation_scales(
    model, activation_scales, down_proj_fp16=False
) -> None:
    """Restore the final exported SA values onto the matching linear inputs."""

    for name, wrapper in _collect_static_targets(model).items():
        if down_proj_fp16 and name.endswith("mlp.down_proj"):
            wrapper.quantizer.bits = 16
            continue
        wrapper.quantizer.load_scale(activation_scales[f"{name}.quantizer"])
