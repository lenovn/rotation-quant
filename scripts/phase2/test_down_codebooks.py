"""Independent numerical checks against exact rational codebook definitions."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import torch

from down_codebooks import build_magnitude_codebook, quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos/SpinQuant"))
from down_codebook_experiment import collect_calibration, local_damage, calibrate_layer


def reference_levels(fmt):
    if fmt == "pot":
        return [Fraction(0)] + sorted(Fraction(1, 2**k) for k in range(127))
    if fmt == "sp2":
        a = [Fraction(0)] + [Fraction(1, 2**k) for k in range(1, 16)]
        b = [Fraction(0)] + [Fraction(1, 2**k) for k in range(1, 8)]
        return sorted({x + y for x in a for y in b})
    raise ValueError(fmt)


def scalar_reference(x, alpha, levels):
    magnitude = abs(x / alpha)
    # Sorted levels make min choose smaller magnitude for an exact tie.
    chosen = min(levels, key=lambda q: abs(float(q) - magnitude))
    return float(chosen) * alpha * (1 if x >= 0 else -1)


class CodebookTests(unittest.TestCase):
    def test_exact_sets_and_duplicate_encodings(self):
        for fmt, count, minimum in (("pot", 128, 2**-126), ("sp2", 94, 2**-15)):
            expected = torch.tensor([float(x) for x in reference_levels(fmt)])
            actual = build_magnitude_codebook(fmt)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.device.type, "cpu")
            self.assertEqual(actual.numel(), count)
            self.assertTrue(torch.equal(actual, expected), fmt)
            self.assertEqual(actual[1].item(), minimum)
            self.assertEqual(actual[-1].item(), 1)
        sp2 = build_magnitude_codebook("sp2")
        self.assertEqual(sp2[-2].item(), .75)

    def test_projection_against_exhaustive_reference(self):
        generator = torch.Generator().manual_seed(718)
        samples = torch.randn(4096, generator=generator) * 3
        for fmt in ("pot", "sp2"):
            levels = reference_levels(fmt)
            for alpha in (.125, 1., 8.):
                expected = torch.tensor([scalar_reference(v, alpha, levels) for v in samples.tolist()])
                actual = quantize(samples, alpha, fmt)
                self.assertTrue(torch.equal(actual, expected), (fmt, alpha))

    def test_midpoints_zero_sign_and_clipping(self):
        for fmt in ("pot", "sp2"):
            levels = build_magnitude_codebook(fmt)
            # Dyadic levels and their midpoints are exactly representable in FP32.
            midpoints = (levels[:-1] + levels[1:]) / 2
            self.assertTrue(torch.equal(quantize(midpoints, 1., fmt), levels[:-1]), fmt)
            self.assertTrue(torch.equal(quantize(-midpoints, 1., fmt), -levels[:-1]), fmt)
            x = torch.tensor([-8., -1., 0., 1., 8.])
            self.assertTrue(torch.equal(quantize(x, 1., fmt), x.clamp(-1, 1)))
            self.assertTrue(torch.equal(quantize(levels, 1., fmt), levels))

    def test_bfloat16_cast_and_noncontiguous(self):
        x = torch.linspace(-3, 3, 119).reshape(7, 17).t().to(torch.bfloat16)
        for fmt in ("int8", "pot", "sp2"):
            q = quantize(x, 2., fmt)
            self.assertEqual(q.dtype, x.dtype)
            self.assertEqual(q.shape, x.shape)
            self.assertTrue(torch.equal(q, quantize(x.float(), 2., fmt).to(x.dtype)))

    def test_int8_existing_signed_range(self):
        x = torch.tensor([-129., -128., -127., -1.5, -.5, 0., .5, 1.5, 127., 128.])
        expected = x.round().clamp(-128, 127)
        self.assertTrue(torch.equal(quantize(x, 127., "int8"), expected))


class RunnerTests(unittest.TestCase):
    def test_hooks_preserve_trajectory_and_damage_matches_reference(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.module.weight.copy_(torch.tensor([[1., 2.], [-2., 1.]]))

            def forward(self, x):
                return self.module(x)

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first, self.second = Wrapper(), Wrapper()

            def forward(self, ids, **kwargs):
                x = torch.stack([ids.float(), ids.float() + .25], dim=-1)
                return self.second(self.first(x))

        model = torch.nn.Module()
        model.model = Core()
        downs = {"first": model.model.first, "second": model.model.second}
        windows = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[-2, -1, 0, 1]])]
        reference = [model.model(w).detach().clone() for w in windows]
        captured, maxima = collect_calibration(model, downs, windows, 4, 42, "cpu")
        self.assertEqual(captured["first"].shape, (8, 2))
        self.assertEqual(maxima["first"], 4.25)
        scales = {fmt: {name: 1. for name in downs} for fmt in ("int8", "pot", "sp2")}
        damage = local_damage(model, downs, windows, scales, "cpu")
        for i, window in enumerate(windows):
            self.assertTrue(torch.equal(model.model(window), reference[i]))
        first_x = torch.cat([torch.stack([w.float(), w.float() + .25], dim=-1).reshape(-1, 2) for w in windows])
        for name, x in (("first", first_x), ("second", model.model.first(first_x))):
            weight = downs[name].module.weight
            for fmt in scales:
                delta_y = (quantize(x, 1., fmt) - x) @ weight.T
                self.assertAlmostEqual(damage[name][fmt]["output_mse"], delta_y.square().double().mean().item(), places=6)
                self.assertEqual(damage[name][fmt]["input_count"], 16)
            self.assertEqual(len(downs[name]._forward_pre_hooks), 0)

    def test_calibration_budget_and_selected_objective(self):
        x = torch.tensor([[.125, 1.], [2., -3.]], dtype=torch.bfloat16)
        weight = torch.tensor([[1., -2.], [.5, .25]])
        result = calibrate_layer(x, weight, 3., 33, 17)
        for fmt, row in result.items():
            self.assertEqual(len(row["evaluated_candidates"]), 50)
            expected = ((quantize(x, row["alpha"], fmt).float() - x.float()) @ weight.T).square().mean().item()
            self.assertEqual(row["output_mse"], expected)
            self.assertEqual(row["output_mse"], min(candidate["output_mse"] for candidate in row["evaluated_candidates"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
