# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import (
    data_utils,
    fuse_norm_utils,
    hadamard_utils,
    quant_utils,
    static_quant_policy,
    utils,
)
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def ptq_model(args, model, model_args=None):
    static_quant_policy.validate_static_activation_request(args, model.config)
    transformers.set_seed(args.seed)
    model.eval()
    rotation_static_scales = None
    if getattr(args, "a_quant_mode", None) == static_quant_policy.ROTATION_STATIC:
        scale_path = args.static_scale_path
        if scale_path is None and args.optimized_rotation_path is not None:
            scale_path = os.path.join(
                os.path.dirname(args.optimized_rotation_path), "quant_scales.pt"
            )
        rotation_static_scales = torch.load(
            scale_path, map_location="cpu", weights_only=True
        )

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        r1_r2_only = getattr(args, "rotation_components", "all") == "r1_r2"
        for name in qlayers:
            if "down_proj" in name and not r1_r2_only:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    # Load deployment A8 before GPTQ so its input/Hessian observations see the
    # same static activation grid that will be used by the final W4A8 model.
    static_quant_policy.configure_input_activation_quantizers(args, model)
    if rotation_static_scales is not None:
        static_quant_policy.load_rotation_static_activation_scales(
            model,
            rotation_static_scales["activation"],
            down_proj_fp16=args.static_down_proj_fp16,
        )

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(
                model,
                "cuda",
                args,
                static_weight_scales=(
                    rotation_static_scales["weight"]
                    if rotation_static_scales is not None
                    else None
                ),
            )
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

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
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model
