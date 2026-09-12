#!/usr/bin/env python3
"""Reuse PTQ unchanged, capture full-precision PPL, optionally bypass A after load."""

import argparse
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--ab-eval-mode", choices=("a8", "a16"), required=True)
    parser.add_argument("--ab-result-path", type=Path, required=True)
    parser.add_argument("--ab-log-path", type=Path, required=True)
    options, ptq_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *ptq_argv]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos/SpinQuant"))

    import ptq
    from utils import quant_utils

    original_ptq_model = ptq.ptq_model
    original_evaluator = ptq.eval_utils.evaluator
    result = {}

    def prepare(args, model, model_args=None):
        if options.ab_eval_mode == "a16":
            if not args.load_qmodel_path or args.save_qmodel_path or args.w_bits != 4:
                raise ValueError("A16 requires loading the existing W4 artifact without saving")
        # Calls the original downstream validation, scale load and checkpoint load.
        model = original_ptq_model(args, model, model_args)
        if options.ab_eval_mode == "a16":
            count = 0
            for module in model.modules():
                if isinstance(module, quant_utils.ActQuantWrapper):
                    # Checkpoint loading can restore quantizer state: bypass AFTER load.
                    module.quantizer.bits = 16
                    module.out_quantizer.bits = 16
                    count += 1
            print(f"Post-load A16 bypass: {count} input/output quantizer pairs", flush=True)
        result.update(
            mode="w4a16" if options.ab_eval_mode == "a16" else "w4a8-downa16",
            gptq_path=args.load_qmodel_path or args.save_qmodel_path,
            rotation_path=args.optimized_rotation_path,
            scale_path=args.static_scale_path,
            seed=args.seed,
            eval_nsamples=args.eval_nsamples,
            log_path=str(options.ab_log_path.resolve()),
        )
        return model

    def evaluate(model, testenc, dev, args):
        ppl = original_evaluator(model, testenc, dev, args)
        if not math.isfinite(float(ppl)):
            raise ValueError(f"Non-finite PPL: {ppl}")
        result.update(ppl=float(ppl), eval_seqlen=model.seqlen)
        return ppl

    ptq.ptq_model = prepare
    ptq.eval_utils.evaluator = evaluate
    try:
        ptq.train()
    finally:
        ptq.ptq_model = original_ptq_model
        ptq.eval_utils.evaluator = original_evaluator
    if "ppl" not in result:
        raise RuntimeError("PTQ returned without evaluation")
    with options.ab_result_path.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
