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

    fresh_w16_ppl = evaluate()

    # This is the project's existing GPTQ artifact, including quantizer objects.
    artifact = torch.load(
        args.load_qmodel_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(artifact["model"], strict=True)
    del artifact
    bypass_activations(model)

    w4 = {
        name: module.module.weight.detach().cpu().clone()
        for name, module in targets.items()
    }

    def select(projections, layer=None):
        return {
            name for name in targets
            if name.rsplit(".", 1)[-1] in projections
            and (layer is None or int(name.split(".")[2]) == layer)
        }

    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    measured = {}

    with (output / "results.jsonl").open("w") as stream:
        def emit(row):
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit({
            "kind": "protocol",
            "model": model_args.input_model,
            "rotation": args.optimized_rotation_path,
            "w4_checkpoint": args.load_qmodel_path,
            "sequence_count": 8,
            "sequence_length": 2048,
            "predicted_tokens": 8 * 2047,
            "activation_bits": 16,
            "kv_bits": 16,
            "dtype": "bfloat16",
            "fresh_w16_ppl": fresh_w16_ppl,
        })

        def measure(case, selected):
            key = frozenset(selected)
            if key in measured:
                return measured[key]

            # Always reconstruct this condition from full W4.
            for name, module in targets.items():
                source = w16[name] if name in selected else w4[name]
                module.module.weight.copy_(source)

            ppl = evaluate()
            row = {
                "case": case,
                "ppl": ppl,
                "nll": math.log(ppl),
                "restored_modules": sorted(selected),
                "restored_numel": sum(w16[n].numel() for n in selected),
            }
            measured[key] = row
            emit(row)
            return row

        full_w4 = measure("full_w4", set())
        all_restored = measure("all_restored", set(targets))

        families = {
            family: measure("family/" + family, select(projections))
            for family, projections in FAMILIES.items()
        }
        measure("module/o_proj", select(("o_proj",)))

        winner = min(families, key=lambda family: families[family]["nll"])
        positive = families[winner]["nll"] < full_w4["nll"]

        # If every family hurts, do not label the least harmful one a winner.
        layer_modules = {"o_proj"}
        if positive:
            layer_modules.update(FAMILIES[winner])
            for proj in FAMILIES[winner]:
                measure("module/" + proj, select((proj,)))
            for layer in range(config.num_hidden_layers):
                measure(
                    f"family_layer/{winner}/{layer}",
                    select(FAMILIES[winner], layer),
                )

        for proj in sorted(layer_modules):
            for layer in range(config.num_hidden_layers):
                measure(
                    f"module_layer/{proj}/{layer}",
                    select((proj,), layer),
                )

        gap = full_w4["nll"] - math.log(fresh_w16_ppl)
        ranking = []
        for row in measured.values():
            gain = full_w4["nll"] - row["nll"]
            ranking.append({
                **row,
                "delta_nll": gain,
                "delta_ppl": full_w4["ppl"] - row["ppl"],
                "recovery_fraction": gain / gap if gap > 0 else None,
            })
        ranking.sort(key=lambda row: row["delta_nll"], reverse=True)

        summary = {
            "winning_family": winner if positive else None,
            "family_delta_nll": {
                name: full_w4["nll"] - row["nll"]
                for name, row in families.items()
            },
            "all_restored_minus_fresh_w16_nll": (
                all_restored["nll"] - math.log(fresh_w16_ppl)
            ),
            "ranking": ranking,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()