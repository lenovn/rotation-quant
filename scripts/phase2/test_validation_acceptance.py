"""Independent token accounting tests; no model or GPU dependency."""
import math
from types import SimpleNamespace
import unittest

import torch

from validation_acceptance import evaluate_full_validation


class ValidationAggregationTests(unittest.TestCase):
    def run_case(self, token_count, expected_segments, expected_unscored=0):
        model = SimpleNamespace(seqlen=4)
        enc = SimpleNamespace(input_ids=torch.arange(token_count).reshape(1, -1))
        args = SimpleNamespace(eval_nsamples=8, bsz=1, sentinel="keep")
        calls = []

        def evaluator(actual_model, segment, device, eval_args):
            self.assertIs(actual_model, model)
            self.assertEqual(device, "cpu")
            self.assertIsNone(eval_args.eval_nsamples)
            self.assertIsNot(eval_args, args)
            calls.append((model.seqlen, segment.input_ids.clone()))
            return math.exp(1. if model.seqlen == 4 else 3.)

        result = evaluate_full_validation(model, enc, "cpu", args, evaluator)
        self.assertEqual(model.seqlen, 4)
        self.assertEqual(vars(args), dict(eval_nsamples=8, bsz=1, sentinel="keep"))
        self.assertEqual(result["token_count"], token_count)
        self.assertEqual(result["predicted_tokens"], sum((s - 1) * n for s, n in expected_segments))
        self.assertEqual(len(calls), len(expected_segments))
        self.assertEqual(len(result["segments"]), len(expected_segments))
        start = 0
        expected_loss = 0.
        for call, segment, (seqlen, windows) in zip(calls, result["segments"], expected_segments):
            self.assertEqual(call[0], seqlen)
            self.assertTrue(torch.equal(call[1], enc.input_ids[:, start:start + seqlen * windows]))
            start += seqlen * windows
            predicted = (seqlen - 1) * windows
            self.assertEqual(segment["seqlen"], seqlen)
            self.assertEqual(segment["windows"], windows)
            self.assertEqual(segment["predicted_tokens"], predicted)
            expected_loss += predicted * (1. if seqlen == 4 else 3.)
        expected_nll = expected_loss / result["predicted_tokens"]
        self.assertAlmostEqual(result["nll"], expected_nll, places=12)
        self.assertAlmostEqual(result["ppl"], math.exp(expected_nll), places=12)
        self.assertEqual(result.get("unscored_tail_tokens", 0), expected_unscored)

    def test_full_windows_and_partial_tail(self):
        self.run_case(10, [(4, 2), (2, 1)])

    def test_full_windows_only(self):
        self.run_case(8, [(4, 2)])

    def test_only_partial_tail(self):
        self.run_case(3, [(3, 1)])

    def test_single_token_tail_is_unscorable(self):
        self.run_case(9, [(4, 2)], expected_unscored=1)

    def test_too_few_tokens(self):
        for count in (0, 1):
            model = SimpleNamespace(seqlen=4)
            args = SimpleNamespace(eval_nsamples=8, bsz=1)
            def evaluator(*unused):
                self.fail("Insufficient tokens must not invoke evaluator")
            with self.assertRaises(ValueError):
                evaluate_full_validation(model, SimpleNamespace(input_ids=torch.zeros((1, count), dtype=torch.long)), "cpu", args, evaluator)
            self.assertEqual(model.seqlen, 4)
            self.assertEqual(args.eval_nsamples, 8)

    def test_exception_restores_sequence_length_and_args(self):
        model = SimpleNamespace(seqlen=4)
        args = SimpleNamespace(eval_nsamples=8, bsz=1)
        calls = []
        def evaluator(*unused):
            calls.append(model.seqlen)
            if model.seqlen == 2:
                raise RuntimeError("tail evaluator failure")
            return math.exp(1.)
        with self.assertRaisesRegex(RuntimeError, "tail evaluator failure"):
            evaluate_full_validation(model, SimpleNamespace(input_ids=torch.arange(10).reshape(1, -1)), "cpu", args, evaluator)
        self.assertEqual(calls, [4, 2])
        self.assertEqual(model.seqlen, 4)
        self.assertEqual(args.eval_nsamples, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
