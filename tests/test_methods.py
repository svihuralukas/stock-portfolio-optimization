"""Testy metód z míľnika 3 (GA a SA+SLSQP) na malej náhodnej inštancii.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portfolio.instance import Instance  # noqa: E402
from portfolio.methods import SharpeGrad, genetic_algorithm, sa_slsqp  # noqa: E402
from portfolio.model import Evaluator, check  # noqa: E402


def random_instance(n=12, days=120, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, size=(days, n)) + rng.normal(0, 0.0005, n)
    p = 100 * np.cumprod(1 + r, axis=0)
    return Instance(name="rand", assets=tuple(f"A{i}" for i in range(n)),
                    train_dates=np.arange(days), train_prices=p,
                    test_dates=np.arange(days), test_prices=p,
                    available=np.ones(n, bool), kmin=3, kmax=5, wmin=0.1, wmax=0.4)


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.inst = random_instance()

    def test_gradient_matches_finite_differences(self):
        sg = SharpeGrad(self.inst)
        w = np.zeros(self.inst.n)
        w[[0, 3, 5, 7]] = [0.4, 0.3, 0.2, 0.1]
        g = sg.grad(w)
        h = 1e-6
        fd = np.array([(sg(w + h * e) - sg(w - h * e)) / (2 * h) for e in np.eye(self.inst.n)])
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)

    def test_results_feasible_and_budget_respected(self):
        for method in (genetic_algorithm, sa_slsqp):
            for seed in range(3):
                res = method(self.inst, 600, np.random.default_rng(seed))
                self.assertEqual(check(res["w"], self.inst), [], method.__name__)
                self.assertLessEqual(res["evals"], 600)
                self.assertAlmostEqual(res["f"], Evaluator(self.inst)(res["w"]), places=10)

    def test_reproducible(self):
        for method in (genetic_algorithm, sa_slsqp):
            a = method(self.inst, 400, np.random.default_rng(7))
            b = method(self.inst, 400, np.random.default_rng(7))
            np.testing.assert_array_equal(a["w"], b["w"])

    def test_trace_is_monotone(self):
        for method in (genetic_algorithm, sa_slsqp):
            tr = np.full(500, np.nan)
            method(self.inst, 500, np.random.default_rng(1), trace=tr)
            t = tr[np.isfinite(tr)]
            self.assertGreater(len(t), 450)
            self.assertTrue(np.all(np.diff(t) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
