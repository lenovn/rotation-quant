"""Independent CPU checks for chunked fixed-input PPL evaluation."""
import math
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from validation_acceptance import evaluate_full_validation


class ChunkedAcceptanceTests(unittest.TestCase):
    def evaluate(self, count, chunk_windows, seqlen=4):
        ids = torch.arange(count, dtype=torch.long).reshape(1, -1)
        model = SimpleNamespace(seqlen=seqlen)
        args = SimpleNamespace(eval_nsamples=8, bsz=3, sentinel="unchanged")
        calls = []

        def evaluator(actual_model, segment, device, eval_args):
            self.assertIs(actual_model, model)
            self.assertEqual(device, "cpu")
            self.assertIsNot(eval_args, args)
            self.assertIsNone(eval_args.eval_nsamples)
            self.assertEqual(eval_args.bsz, 1)
            self.assertEqual(eval_args.sentinel, "unchanged")
            calls.append((model.seqlen, segment.input_ids.clone()))
            # Each target has its own loss. Recomputing from segment contents
            # exposes repeated/dropped tokens and accidental cross-window targets.
            targets = segment.input_ids.reshape(-1, model.seqlen)[:, 1:]
            loss = (targets.double() + 1).mean().item() / (count + 1)
            return math.exp(loss)

        result = evaluate_full_validation(
            model, SimpleNamespace(input_ids=ids), "cpu", args, evaluator,
            chunk_windows=chunk_windows,
        )
        self.assertEqual(model.seqlen, seqlen)
        self.assertEqual(vars(args), dict(eval_nsamples=8, bsz=3, sentinel="unchanged"))
        full, tail = divmod(count, seqlen)
        target_positions = [i for i in range(full * seqlen) if i % seqlen]
        if tail >= 2:
            target_positions.extend(range(full * seqlen + 1, count))
        expected_nll = sum(i + 1 for i in target_positions) / len(target_positions) / (count + 1)
        self.assertEqual(result["token_count"], count)
        self.assertEqual(result["predicted_tokens"], len(target_positions))
        self.assertAlmostEqual(result["nll"], expected_nll, places=12)
        self.assertAlmostEqual(result["ppl"], math.exp(expected_nll), places=12)
        self.assertEqual(result["unscored_tail_tokens"], int(tail == 1))
        scored_input = torch.cat([tokens for _, tokens in calls], dim=1)
        self.assertTrue(torch.equal(scored_input, ids[:, :count - int(tail == 1)]))
        if chunk_windows is not None:
            self.assertTrue(all(tokens.numel() <= length * chunk_windows for length, tokens in calls))
        return result, calls

    def test_short_last_chunk_and_partial_tail_are_token_weighted(self):
        result, calls = self.evaluate(30, 3)
        self.assertEqual([(length, tokens.numel() // length) for length, tokens in calls],
                         [(4, 3), (4, 3), (4, 1), (2, 1)])
        self.assertEqual([s["predicted_tokens"] for s in result["segments"]], [9, 9, 3, 1])
        self.assertGreater(abs(result["ppl"] - sum(s["ppl"] for s in result["segments"]) / 4), 0.01)

    def test_chunked_and_unchunked_score_identical_targets(self):
        expected, _ = self.evaluate(43, None)
        for chunk_windows in (1, 2, 3, 10, 100):
            with self.subTest(chunk_windows=chunk_windows):
                actual, _ = self.evaluate(43, chunk_windows)
                self.assertAlmostEqual(actual["nll"], expected["nll"], places=12)
                self.assertEqual(actual["predicted_tokens"], expected["predicted_tokens"])

    def test_single_token_tail_is_not_a_cross_window_target(self):
        self.evaluate(29, 3)

    def test_short_corpus_still_evaluates_its_partial_window(self):
        result, calls = self.evaluate(3, 128)
        self.assertEqual(result["predicted_tokens"], 2)
        self.assertEqual([(length, tokens.numel()) for length, tokens in calls], [(3, 3)])

    def test_actual_c4_budget_limits_each_evaluator_call(self):
        count = 1024 * 2048
        model = SimpleNamespace(seqlen=2048)
        args = SimpleNamespace(eval_nsamples=8, bsz=1)
        ids = torch.arange(count).reshape(1, -1)
        calls = []

        def evaluator(model, segment, device, eval_args):
            calls.append((segment.input_ids[0, 0].item(), segment.input_ids.numel()))
            return math.e

        result = evaluate_full_validation(model, SimpleNamespace(input_ids=ids), "cpu", args,
                                          evaluator, chunk_windows=128)
        self.assertEqual(calls, [(i * 128 * 2048, 128 * 2048) for i in range(8)])
        self.assertEqual(result["predicted_tokens"], 2_096_128)
        self.assertEqual(result["token_count"], 2_097_152)
        self.assertAlmostEqual(result["nll"], 1.0)

    def test_nonpositive_chunk_size_is_rejected(self):
        for chunk_windows in (0, -1):
            with self.subTest(chunk_windows=chunk_windows):
                with self.assertRaises(ValueError):
                    self.evaluate(8, chunk_windows)

    def test_exception_restores_sequence_length_and_original_args(self):
        model = SimpleNamespace(seqlen=4)
        args = SimpleNamespace(eval_nsamples=8, bsz=3)
        calls = []

        def evaluator(*unused):
            calls.append(model.seqlen)
            if model.seqlen == 2:
                raise RuntimeError("tail failed")
            return math.e

        with self.assertRaisesRegex(RuntimeError, "tail failed"):
            evaluate_full_validation(model, SimpleNamespace(input_ids=torch.arange(14).reshape(1, -1)),
                                     "cpu", args, evaluator, chunk_windows=2)
        self.assertEqual(calls, [4, 4, 2])
        self.assertEqual(model.seqlen, 4)
        self.assertEqual(vars(args), dict(eval_nsamples=8, bsz=3))


class SavedC4InputTests(unittest.TestCase):
    def test_preparation_uses_validation_files_and_tokenizes_joined_text(self):
        import prepare_c4_evaluation

        class Documents:
            def __len__(self):
                return 256

            def __getitem__(self, indices):
                return {"text": [f"document {i}: text." for i in indices],
                        "url": [f"https://example.invalid/{i}" for i in indices]}

        class CharacterTokenizer:
            def __call__(self, text, **kwargs):
                self_check.assertFalse(kwargs["add_special_tokens"])
                self_check.assertFalse(kwargs["truncation"])
                if isinstance(text, list):
                    return {"input_ids": [[ord(c) for c in item] for item in text]}
                return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))

        self_check = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare_c4_evaluation, "load_dataset", return_value=Documents()) as load, \
                    patch.object(prepare_c4_evaluation.LlamaTokenizerFast, "from_pretrained",
                                 return_value=CharacterTokenizer()) as tokenizer:
                prepare_c4_evaluation.prepare(root / "data", root / "model", root / "cache",
                                              windows=16, seqlen=32, seed=42)
            self.assertEqual(load.call_args.args, ("json",))
            self.assertEqual(load.call_args.kwargs["split"], "validation")
            files = load.call_args.kwargs["data_files"]
            self.assertEqual(set(files), {"validation"})
            self.assertEqual(len(files["validation"]), 8)
            self.assertTrue(all("/en/c4-validation." in path for path in files["validation"]))
            self.assertFalse(tokenizer.call_args.kwargs["add_bos_token"])
            self.assertFalse(tokenizer.call_args.kwargs["add_eos_token"])
            rows = [json.loads(line) for line in (root / "data/documents.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["row_index"] for row in rows}), len(rows))
            joined = "\n\n".join(row["text"] for row in rows)
            saved = torch.load(root / "data/input_tokens.pt", weights_only=True)
            expected = torch.tensor([[ord(c) for c in joined[:512]]])
            self.assertTrue(torch.equal(saved["input_ids"], expected))
            self.assertEqual(saved["metadata"]["token_count"], 512)
            self.assertEqual(saved["metadata"]["predicted_tokens"], 16 * 31)
            self.assertEqual(saved["metadata"]["split"], "validation")

    def test_saved_input_entrypoint_never_loads_or_calibrates_a_dataset(self):
        import datasets
        import validation_acceptance

        ids = torch.arange(4096).reshape(1, -1)
        observed = []
        model = torch.nn.Linear(1, 1)
        model.seqlen = 2048

        def forbidden(*args, **kwargs):
            self.fail("Saved-input original BF16 evaluation must not prepare PTQ or load a dataset")

        def evaluator(actual_model, enc, device, args):
            self.assertIs(actual_model, model)
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            observed.append(enc.input_ids.clone())
            return math.e

        fake_ptq = SimpleNamespace(ptq_model=forbidden,
                                   eval_utils=SimpleNamespace(evaluator=evaluator),
                                   data_utils=SimpleNamespace(get_wikitext2=forbidden))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(dataset="allenai/c4", subset="en", split="validation",
                            tokenizer_path=str(root / "model"), window_length=2048)
            token_path = root / "tokens.pt"
            torch.save(dict(input_ids=ids, metadata=metadata), token_path)

            def train():
                args = SimpleNamespace(w_bits=16, a_bits=16, rotate=False, seed=42,
                                       bsz=1, eval_nsamples=8)
                actual_model = fake_ptq.ptq_model(args, model, SimpleNamespace(input_model=str(root / "model")))
                with self.assertRaisesRegex(ValueError, "must not recalibrate or train"):
                    fake_ptq.data_utils.get_wikitext2(eval_mode=False)
                enc = fake_ptq.data_utils.get_wikitext2(
                    tokenizer=SimpleNamespace(model_max_length=2048), eval_mode=True)
                return fake_ptq.eval_utils.evaluator(actual_model, enc, "cpu", args)

            fake_ptq.train = train
            argv = ["validation_acceptance.py", "--accept-mode", "w16a16",
                    "--accept-output", str(root / "result"),
                    "--accept-token-file", str(token_path), "--accept-chunk-windows", "1"]
            with patch.dict(sys.modules, {"ptq": fake_ptq}), \
                    patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)), \
                    patch.object(datasets, "load_dataset", side_effect=forbidden):
                validation_acceptance.main()
            self.assertTrue(torch.equal(torch.cat(observed, dim=1), ids))
            result = json.loads((root / "result/result.json").read_text())
            self.assertEqual(result["input_token_path"], str(token_path))
            self.assertEqual(result["dataset"], "allenai/c4")
            self.assertEqual(result["predicted_tokens"], 4094)
            self.assertEqual(result["quantized_linear_count"], 0)
            self.assertIs(fake_ptq.ptq_model, forbidden)
            self.assertIs(fake_ptq.eval_utils.evaluator, evaluator)
            self.assertIs(fake_ptq.data_utils.get_wikitext2, forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
