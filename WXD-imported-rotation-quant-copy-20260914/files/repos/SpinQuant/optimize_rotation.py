# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.downproj_mse_calibration import (
    DownProjMSECalibrator,
    save_downproj_mse_result,
)
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from train_utils.rotation_calibration import (
    LearnableScaleClampCallback,
    PeriodicRecalibrationCallback,
    RotationScaleCalibrator,
    enable_non_downproj_scale_learning,
    export_rotation_scales,
)
from utils import static_quant_policy
from utils.data_utils import CustomJsonDataset, get_wikitext2
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    static_quant_policy.validate_static_activation_args(ptq_args)
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    static_quant_policy.validate_static_activation_model_config(ptq_args, config)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    local_rank = get_local_rank()
    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = prepare_model(ptq_args, model)
    for param in model.parameters():
        param.requires_grad = False
    loaded_rotation = None
    if model_args.optimized_rotation_path is not None:
        loaded_rotation = torch.load(
            model_args.optimized_rotation_path,
            map_location="cpu",
            weights_only=True,
        )
    R1 = (
        loaded_rotation["R1"]
        if loaded_rotation is not None
        else random_hadamard_matrix(model.config.hidden_size, "cuda")
    )
    model.R1 = RotateModule(R1)
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        rotation_key = f"model.layers.{i}.self_attn.R2"
        R2 = (
            loaded_rotation[rotation_key]
            if loaded_rotation is not None
            else random_hadamard_matrix(
                model.config.hidden_size // model.config.num_attention_heads, "cuda"
            )
        )
        model.model.layers[i].self_attn.R2 = RotateModule(R2)
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    rotation_static = ptq_args.a_quant_mode == static_quant_policy.ROTATION_STATIC
    calibrator = None
    callbacks = None
    if rotation_static:
        calibration_data = get_wikitext2(
            nsamples=ptq_args.calibration_nsamples,
            seed=ptq_args.calibration_seed,
            seqlen=ptq_args.calibration_seqlen,
            tokenizer=tokenizer,
        )
        calibrator = RotationScaleCalibrator(
            [input_ids for input_ids, _ in calibration_data], logger=log
        )
        if ptq_args.learn_activation_scales:
            callbacks = [LearnableScaleClampCallback()]
        else:
            callbacks = [
                PeriodicRecalibrationCallback(
                    calibrator=calibrator,
                    interval=ptq_args.recalibration_interval,
                    final_update=training_args.max_steps,
                )
            ]

    if ptq_args.downproj_mse_calibration:
        model.cuda()
        base_scale_state = torch.load(
            ptq_args.static_scale_path,
            map_location="cpu",
            weights_only=True,
        )
        mse_calibrator = DownProjMSECalibrator(
            [input_ids for input_ids, _ in calibration_data], logger=log
        )
        mse_result = mse_calibrator.run(
            model,
            base_scale_state,
            calibration_metadata={
                "split": "train",
                "nsamples": ptq_args.calibration_nsamples,
                "seq_len": ptq_args.calibration_seqlen,
                "seed": ptq_args.calibration_seed,
                "forward_quantization": "old_static_w4a8",
            },
        )
        if local_rank == 0:
            save_downproj_mse_result(
                mse_result, model_args.output_rotation_path
            )
            log.info("down_proj MSE summary: %s", mse_result["stats"]["summary"])
        dist.barrier()
        dist.destroy_process_group()
        return

    if ptq_args.calibration_only:
        model.cuda()
        calibrator.run(model, update_step=0, kind="final")
        if local_rank == 0:
            os.makedirs(model_args.output_rotation_path, exist_ok=True)
            rotation_dict = {"R1": model.R1.weight.detach().cpu()}
            rotation_dict.update(
                {
                    f"model.layers.{i}.self_attn.R2": model.model.layers[
                        i
                    ].self_attn.R2.weight.detach().cpu()
                    for i in range(model.config.num_hidden_layers)
                }
            )
            torch.save(
                rotation_dict,
                os.path.join(model_args.output_rotation_path, "R.bin"),
            )
            torch.save(
                export_rotation_scales(model),
                os.path.join(model_args.output_rotation_path, "quant_scales.pt"),
            )
            calibrator.save_history(
                os.path.join(model_args.output_rotation_path, "calibration_history.json")
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    rotation_parameters = [model.R1.weight] + [
        model.model.layers[i].self_attn.R2.weight
        for i in range(model.config.num_hidden_layers)
    ]
    optimizer_groups = [
        {
            "params": rotation_parameters,
            "lr": training_args.learning_rate,
            "stiefel": True,
        }
    ]
    if ptq_args.learn_activation_scales:
        learned_scales, bypassed_downproj_scales = (
            enable_non_downproj_scale_learning(model)
        )
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in learned_scales],
                "lr": ptq_args.activation_scale_learning_rate,
                "stiefel": False,
            }
        )
        if local_rank == 0:
            log.info(
                "Joint R+SA training: %d learned activation scales, "
                "%d A16 down_proj inputs, R lr=%g, SA lr=%g",
                len(learned_scales),
                len(bypassed_downproj_scales),
                training_args.learning_rate,
                ptq_args.activation_scale_learning_rate,
            )
    model.seqlen = training_args.model_max_length
    optimizer = SGDG(optimizer_groups, lr=training_args.learning_rate)
    MyTrainer = Trainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    torch.distributed.barrier()

    if calibrator is not None:
        calibrator.run(trainer.model, update_step=0, kind="initial")
        if ptq_args.learn_activation_scales and local_rank == 0:
            os.makedirs(model_args.output_rotation_path, exist_ok=True)
            torch.save(
                export_rotation_scales(trainer.model),
                os.path.join(
                    model_args.output_rotation_path,
                    "initial_quant_scales.pt",
                ),
            )
    trainer.train()
    if calibrator is not None and not ptq_args.learn_activation_scales:
        calibrator.run(
            trainer.model,
            update_step=trainer.state.global_step,
            kind="final",
        )
    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if "R1.weight" in key or "self_attn.R2" in key
    }
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "R.bin")
        torch.save(
            R_dict,
            path,
        )
        if calibrator is not None:
            torch.save(
                export_rotation_scales(trainer.model),
                os.path.join(model_args.output_rotation_path, "quant_scales.pt"),
            )
            calibrator.save_history(
                os.path.join(model_args.output_rotation_path, "calibration_history.json")
            )
            with open(
                os.path.join(model_args.output_rotation_path, "training_history.json"),
                "w",
                encoding="utf-8",
            ) as history_file:
                json.dump(trainer.state.log_history, history_file, indent=2)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
