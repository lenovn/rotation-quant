# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import transformers

from train_utils import apply_r3_r4, rtn_utils
from utils import (
    fuse_norm_utils,
    hadamard_utils,
    quant_utils,
    static_quant_policy,
    utils,
)


def prepare_model(args, model):
    static_quant_policy.validate_static_activation_request(args, model.config)
    transformers.set_seed(args.seed)
    model.eval()
    r1_r2_only = getattr(args, "rotation_components", "all") == "r1_r2"

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)
    if not r1_r2_only:
        apply_r3_r4.rotate_model(model, args)
    utils.cleanup_memory(verbos=True)

    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name and not r1_r2_only:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had

    if args.w_bits < 16:
        quantizers = rtn_utils.rtn_fwrd(model, "cuda", args)

    # Keep legacy V output quantization here; shared policy owns input quantizers.
    if args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name in qlayers:
            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

    static_quant_policy.configure_input_activation_quantizers(args, model)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                apply_r3_r4.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model
