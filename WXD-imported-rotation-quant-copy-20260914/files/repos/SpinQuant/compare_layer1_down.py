import copy
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import LlamaTokenizerFast

from eval_utils.main import ptq_model
from eval_utils.gptq_utils import GPTQ
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, quant_utils
from utils.process_args import process_args_ptq


FAMILIES = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}
PROJECTIONS = sum(FAMILIES.values(), ())


def bypass_activations(model):
    # rotation_static creates A8 quantizers regardless of args.a_bits.
    for module in model.modules():
        if isinstance(module, quant_utils.ActQuantWrapper):
            module.quantizer.bits = 16
            module.out_quantizer.bits = 16


@torch.no_grad()
def main():
    model_args, training_args, args = process_args_ptq()
    if not (
        args.rotate
        and args.rotation_components == "r1_r2"
        and args.a_quant_mode == "rotation_static"
        and args.k_bits == args.v_bits == 16
        and args.eval_nsamples == 8
        and args.bsz == 1
        and training_args.model_max_length == 2048
        and training_args.bf16
        and args.load_qmodel_path
        and args.optimized_rotation_path
        and not args.save_qmodel_path
    ):
        raise ValueError("Use the fixed experiment command shown below.")

    transformers.set_seed(args.seed)
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    tied = config.tie_word_embeddings
    config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(
        model_args.input_model,
        config=config,
        torch_dtype=torch.bfloat16,
        token=model_args.access_token,
    )
    if tied:
        model.lm_head.weight.data = (
            model.model.embed_tokens.weight.data.clone()
        )

    # Build the exact rotated W16 donor without invoking GPTQ.
    prep_args = copy.copy(args)
    prep_args.w_bits = 16
    prep_args.load_qmodel_path = None
    model = ptq_model(prep_args, model.cuda(), model_args).cpu()
    model.seqlen = 2048
    model.config.use_cache = False
    model.eval()
    bypass_activations(model)

    targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, quant_utils.ActQuantWrapper)
        and name.startswith("model.layers.")
        and name.rsplit(".", 1)[-1] in PROJECTIONS
    }
    expected = {
        f"model.layers.{layer}.{block}.{proj}"
        for layer in range(config.num_hidden_layers)
        for proj in PROJECTIONS
        for block in [
            "self_attn" if proj in FAMILIES["attention"] else "mlp"
        ]
    }
    if set(targets) != expected:
        raise ValueError("Unexpected projection coverage.")

    # Clone: state_dict()/detach() alone would alias live parameters.
    w16 = {
        name: module.module.weight.detach().cpu().clone()
        for name, module in targets.items()
    }

    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    testenc = data_utils.get_wikitext2(
        seed=args.seed, seqlen=2048,
        tokenizer=tokenizer, eval_mode=True,
    )
    if testenc.input_ids.numel() < 8 * 2048:
        raise ValueError("Insufficient evaluation tokens.")
    testenc.input_ids = testenc.input_ids[:, :8 * 2048]
    args.capture_layer_io = False

    def evaluate():
        bypass_activations(model)
        ppl = float(eval_utils.evaluator(
            model.cuda(), testenc, torch.device("cuda"), args
        ))
        model.cpu()
        if not math.isfinite(ppl) or ppl <= 0:
            raise RuntimeError(f"Invalid PPL: {ppl}")
        return ppl

    # fresh_w16_ppl = evaluate()

    # # This is the project's existing GPTQ artifact, including quantizer objects.
    # artifact = torch.load(
    #     args.load_qmodel_path, map_location="cpu", weights_only=False
    # )
    # model.load_state_dict(artifact["model"], strict=True)
    # del artifact
    # bypass_activations(model)

    # w4 = {
    #     name: module.module.weight.detach().cpu().clone()
    #     for name, module in targets.items()
    # }

    # def select(projections, layer=None):
    #     return {
    #         name for name in targets
    #         if name.rsplit(".", 1)[-1] in projections
    #         and (layer is None or int(name.split(".")[2]) == layer)
    #     }

    # output = Path(training_args.output_dir)
    # output.mkdir(parents=True, exist_ok=True)
    # measured = {}

    # with (output / "results.jsonl").open("w") as stream:
    #     def emit(row):
    #         line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    #         stream.write(line + "\n")
    #         stream.flush()
    #         print(line, flush=True)

    #     emit({
    #         "kind": "protocol",
    #         "model": model_args.input_model,
    #         "rotation": args.optimized_rotation_path,
    #         "w4_checkpoint": args.load_qmodel_path,
    #         "sequence_count": 8,
    #         "sequence_length": 2048,
    #         "predicted_tokens": 8 * 2047,
    #         "activation_bits": 16,
    #         "kv_bits": 16,
    #         "dtype": "bfloat16",
    #         "fresh_w16_ppl": fresh_w16_ppl,
    #     })

    #     def measure(case, selected):
    #         key = frozenset(selected)
    #         if key in measured:
    #             return measured[key]

    #         # Always reconstruct this condition from full W4.
    #         for name, module in targets.items():
    #             source = w16[name] if name in selected else w4[name]
    #             module.module.weight.copy_(source)

    #         ppl = evaluate()
    #         row = {
    #             "case": case,
    #             "ppl": ppl,
    #             "nll": math.log(ppl),
    #             "restored_modules": sorted(selected),
    #             "restored_numel": sum(w16[n].numel() for n in selected),
    #         }
    #         measured[key] = row
    #         emit(row)
    #         return row

    #     full_w4 = measure("full_w4", set())
    #     all_restored = measure("all_restored", set(targets))

    #     families = {
    #         family: measure("family/" + family, select(projections))
    #         for family, projections in FAMILIES.items()
    #     }
    #     measure("module/o_proj", select(("o_proj",)))

    #     winner = min(families, key=lambda family: families[family]["nll"])
    #     positive = families[winner]["nll"] < full_w4["nll"]

    #     # If every family hurts, do not label the least harmful one a winner.
    #     layer_modules = {"o_proj"}
    #     if positive:
    #         layer_modules.update(FAMILIES[winner])
    #         for proj in FAMILIES[winner]:
    #             measure("module/" + proj, select((proj,)))
    #         for layer in range(config.num_hidden_layers):
    #             measure(
    #                 f"family_layer/{winner}/{layer}",
    #                 select(FAMILIES[winner], layer),
    #             )

    #     for proj in sorted(layer_modules):
    #         for layer in range(config.num_hidden_layers):
    #             measure(
    #                 f"module_layer/{proj}/{layer}",
    #                 select((proj,), layer),
    #             )

    #     gap = full_w4["nll"] - math.log(fresh_w16_ppl)
    #     ranking = []
    #     for row in measured.values():
    #         gain = full_w4["nll"] - row["nll"]
    #         ranking.append({
    #             **row,
    #             "delta_nll": gain,
    #             "delta_ppl": full_w4["ppl"] - row["ppl"],
    #             "recovery_fraction": gain / gap if gap > 0 else None,
    #         })
    #     ranking.sort(key=lambda row: row["delta_nll"], reverse=True)

    #     summary = {
    #         "winning_family": winner if positive else None,
    #         "family_delta_nll": {
    #             name: full_w4["nll"] - row["nll"]
    #             for name, row in families.items()
    #         },
    #         "all_restored_minus_fresh_w16_nll": (
    #             all_restored["nll"] - math.log(fresh_w16_ppl)
    #         ),
    #         "ranking": ranking,
    #     }
    #     (output / "summary.json").write_text(
    #         json.dumps(summary, ensure_ascii=False, indent=2,
    #                    allow_nan=False) + "\n"
    #     )
    
    # Only four PPL evaluations: A / B / C / D.
    target_name = "model.layers.1.mlp.down_proj"
    target = targets[target_name].module
    donor = w16[target_name].clone()  # BF16 after norm fusion + final R1/R2
    del w16

    if not args.static_down_proj_fp16:
        raise ValueError("This artifact used static A8 with down inputs A16.")
    if args.w_asym or args.w_clip or args.act_order:
        raise ValueError("Use the original symmetric, no-clip, no-act-order GPTQ.")

    artifact = torch.load(
        args.load_qmodel_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(artifact["model"], strict=True)
    del artifact
    model.cpu()
    bypass_activations(model)
    original_w4 = target.weight.detach().cpu().clone()

    # Reconstruct the original GPTQ calibration context:
    # - preceding weights remain exactly those of the existing W4 artifact;
    # - static A8 is enabled except on down_proj inputs;
    # - the target's input is intercepted before its Linear executes.
    #
    # In the original sequential GPTQ, q/k/v/o and gate/up in layer 1
    # were already quantized when down_proj's Hessian was measured.
    # Thus the stored W4 prefix supplies the appropriate context.
    for name, wrapper in targets.items():
        wrapper.quantizer.bits = (
            16 if name.endswith(".down_proj") else 8
        )
        wrapper.out_quantizer.bits = 16

    # Match the original GPTQ loader, including its tokenizer choice.
    # Do not substitute the evaluation tokenizer here.
    calibration = data_utils.get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        model=model_args.input_model,
        seqlen=2048,
        eval_mode=False,
    )

    model.cuda()
    collector = GPTQ(target)

    class TargetInputCaptured(Exception):
        pass

    def collect_input(module, inputs):
        # GPTQ.add_batch uses only its input argument.
        collector.add_batch(inputs[0].detach(), None)
        raise TargetInputCaptured()

    handle = target.register_forward_pre_hook(collect_input)
    try:
        for tokens, _ in calibration:
            try:
                model(tokens.cuda(), use_cache=False)
            except TargetInputCaptured:
                pass
            else:
                raise RuntimeError("Target down_proj was not reached.")

        if collector.nsamples != args.nsamples:
            raise RuntimeError("Unexpected calibration sample count.")
        hessian = collector.H.detach().cpu().clone()
    finally:
        handle.remove()
        collector.free()
        model.cpu()
        bypass_activations(model)

    del collector, calibration
    torch.cuda.empty_cache()

    def quantize_donor(bits, groupsize):
        # A separate Linear prevents GPTQ from changing the live W4 model.
        # Every invocation starts from the same BF16 donor and fresh H copy.
        linear = torch.nn.Linear(
            donor.shape[1], donor.shape[0], bias=False
        ).to(device="cuda", dtype=donor.dtype)
        linear.weight.copy_(donor)

        solver = GPTQ(linear)
        solver.H.copy_(hessian)
        solver.nsamples = args.nsamples
        solver.quantizer = quant_utils.WeightQuantizer()
        solver.quantizer.configure(
            bits,
            perchannel=True,
            sym=True,
            mse=False,
        )
        try:
            solver.fasterquant(
                blocksize=128,
                percdamp=args.percdamp,
                groupsize=groupsize,
                actorder=False,
                static_groups=False,
                export_to_et=False,
            )
            result = linear.weight.detach().cpu().clone()
            if not torch.isfinite(result).all():
                raise RuntimeError("Non-finite quantized weights.")
            return result
        finally:
            solver.free()
            del solver, linear
            torch.cuda.empty_cache()

    # Reproduction diagnostic, not a fifth PPL condition.
    regenerated_a = quantize_donor(bits=4, groupsize=-1)
    difference = regenerated_a.float() - original_w4.float()
    reproduction = {
        "exact_match": torch.equal(regenerated_a, original_w4),
        "max_abs_difference": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "different_fraction": (
            (regenerated_a != original_w4).float().mean().item()
        ),
    }
    del regenerated_a, difference

    c_weight = quantize_donor(bits=4, groupsize=32)
    d_weight = quantize_donor(bits=8, groupsize=-1)
    del hessian

    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    with (output / "results.jsonl").open("w") as stream:
        def emit(row):
            line = json.dumps(
                row, ensure_ascii=False, allow_nan=False
            )
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit({
            "kind": "protocol",
            "target": target_name,
            "model": model_args.input_model,
            "rotation": args.optimized_rotation_path,
            "w4_checkpoint": args.load_qmodel_path,
            "static_scales": args.static_scale_path,
            "calibration": {
                "dataset": "WikiText-2 train",
                "nsamples": args.nsamples,
                "seqlen": 2048,
                "seed": args.seed,
                "activation": "static A8, down inputs A16",
                "percdamp": args.percdamp,
                "shared_hessian": True,
            },
            "evaluation": {
                "dataset": "WikiText-2 test",
                "nsamples": 8,
                "seqlen": 2048,
                "activation_bits": 16,
                "kv_bits": 16,
                "dtype": "bfloat16",
            },
            "A_weight_reproduction": reproduction,
        })

        # Only the target parameter is replaced.
        # evaluate() returns the model to CPU after every condition.
        for case, bits, groupsize, weight in [
            ("A", 4, -1, original_w4),
            ("B", 16, None, donor),
            ("C", 4, 32, c_weight),
            ("D", 8, -1, d_weight),
        ]:
            target.weight.copy_(weight)
            ppl = evaluate()
            row = {
                "case": case,
                "target_bits": bits,
                "target_groupsize": groupsize,
                "ppl": ppl,
                "nll": math.log(ppl),
            }
            rows.append(row)
            emit(row)

    target.weight.copy_(original_w4)

    a_nll = rows[0]["nll"]
    b_nll = rows[1]["nll"]
    recoverable_gap = a_nll - b_nll
    for row in rows:
        row["delta_nll_vs_A"] = a_nll - row["nll"]
        row["fraction_of_B_gain"] = (
            row["delta_nll_vs_A"] / recoverable_gap
            if recoverable_gap > 0 else None
        )

    summary = {
        "target": target_name,
        "A_weight_reproduction": reproduction,
        "results": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()