"""Overenie modelu na malých ručne spočítateľných príkladoch.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portfolio.baselines import feasible_counts, hill_climb, random_feasible, top_k_sharpe  # noqa: E402
from portfolio.instance import Instance  # noqa: E402
from portfolio.model import (Evaluator, check, evaluate, max_drawdown,  # noqa: E402
                             n_eff, portfolio_values, sharpe_from_values)
from portfolio.repair import (decode_scores, normalize, project_capped_simplex,  # noqa: E402
                              repair, violation_diversity, violation_rules)


def toy(prices, **rules):
    """Malá inštancia: rovnaké ceny pre optimalizačné aj testovacie obdobie."""
    p = np.asarray(prices, dtype=float)
    n = p.shape[1]
    base = dict(kmin=2, kmax=3, wmin=0.2, wmax=0.6)
    base.update(rules)
    return Instance(name="toy", assets=tuple(f"A{i}" for i in range(n)),
                    train_dates=np.arange(len(p)), train_prices=p,
                    test_dates=np.arange(len(p)), test_prices=p,
                    available=np.ones(n, bool), **base)


class TestSimulation(unittest.TestCase):
    def test_shares_and_values_by_hand(self):
        # Aktívum A: 100 -> 110 -> 99, aktívum B: 50 -> 50 -> 60. Portfólio 50/50 z 1000 USD:
        # 5 kusov A a 10 kusov B; hodnoty 1000, 1050, 1095.
        inst = toy([[100, 50], [110, 50], [99, 60]], kmin=1, budget=1000.0)
        v = portfolio_values(np.array([0.5, 0.5]), inst.train_prices, inst.budget)
        np.testing.assert_allclose(v, [1000, 1050, 1095])

    def test_sharpe_by_hand(self):
        # Denné výnosy 0,05 a 0,042857...; Sharpe = priemer / smer. odchýlka * sqrt(252)
        v = np.array([1000.0, 1050.0, 1095.0])
        r = np.array([0.05, 45 / 1050])
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(sharpe_from_values(v), expected)

    def test_evaluator_matches_full_simulation(self):
        rng = np.random.default_rng(1)
        p = np.cumprod(1 + rng.normal(0.0005, 0.01, (300, 6)), axis=0) * 50
        inst = toy(p)
        w = np.array([0.3, 0, 0.5, 0, 0.2, 0])
        self.assertAlmostEqual(Evaluator(inst)(w), evaluate(w, inst)["sharpe"])

    def test_drawdown_and_final_value(self):
        self.assertAlmostEqual(max_drawdown(np.array([100, 120, 90, 130, 117])), 0.25)
        inst = toy([[10, 20], [12, 20], [15, 25]], kmin=1)
        res = evaluate(np.array([0.4, 0.6]), inst)
        # 4000 USD -> 400 kusov po 10, 6000 USD -> 300 kusov po 20; na konci 400*15 + 300*25
        self.assertAlmostEqual(res["final_value"], 13_500)

    def test_two_assets_diversification_raises_sharpe(self):
        # Dve aktíva s rovnakým trendom a opačným šumom: 50/50 má nižšie kolísanie.
        t = np.arange(200)
        a = 100 * np.exp(0.001 * t + 0.02 * np.sin(t))
        b = 100 * np.exp(0.001 * t - 0.02 * np.sin(t))
        inst = toy(np.column_stack([a, b]), kmin=1)
        f = Evaluator(inst)
        self.assertGreater(f(np.array([0.5, 0.5])), f(np.array([1.0, 0.0])))


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.inst = toy(np.ones((5, 6)) * 10)  # 6 aktív, 2–3 vybrané, podiel 20–60 %

    def test_feasible(self):
        self.assertEqual(check(np.array([0.5, 0.5, 0, 0, 0, 0]), self.inst), [])
        self.assertEqual(check(np.array([0.2, 0.2, 0.6, 0, 0, 0]), self.inst), [])

    def test_each_rule_detected(self):
        cases = {
            "O1": [0.5, 0.4, 0, 0, 0, 0],
            "O2": [0.7, 0.5, -0.2, 0, 0, 0],
            "O3": [0.25, 0.25, 0.25, 0.25, 0, 0],
            "O4": [0.7, 0.3, 0, 0, 0, 0],
        }
        for rule, w in cases.items():
            errors = check(np.array(w), self.inst)
            self.assertTrue(any(e.startswith(rule) for e in errors), (rule, errors))

    def test_unavailable_asset(self):
        inst = self.inst.with_rules(available=np.array([True] * 5 + [False]))
        self.assertTrue(check(np.array([0.5, 0, 0, 0, 0, 0.5]), inst)[0].startswith("O5"))

    def test_single_asset_is_infeasible(self):
        # Na otázku "nebude to konvergovať k 100 % v jednom aktíve": také portfólio
        # porušuje O3 aj O4, takže v tejto formulácii nie je prípustné.
        errors = check(np.array([1.0, 0, 0, 0, 0, 0]), self.inst)
        self.assertEqual(sorted(e[:2] for e in errors), ["O3", "O4"])


class TestRepair(unittest.TestCase):
    def setUp(self):
        self.inst = toy(np.ones((5, 8)) * 10, kmin=3, kmax=5, wmin=0.1, wmax=0.4)
        self.rng = np.random.default_rng(0)

    def test_projection(self):
        # tau = -0.15: 0.9+0.15 -> orezané na 0.6; 0.1+0.15 = 0.25; 0+0.15 = 0.15; súčet 1
        w = project_capped_simplex(np.array([0.9, 0.1, 0.0]), 0.1, 0.6)
        np.testing.assert_allclose(w, [0.6, 0.25, 0.15], atol=1e-9)
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_projection_matches_bisection(self):
        # Referencia: tau nájdené bisekciou (súčet clip(v - tau) je v tau nerastúci).
        for _ in range(500):
            k = int(self.rng.integers(4, 11))
            v = self.rng.normal(0, 0.5, k)
            a, b = v.min() - 0.3, v.max() - 0.05
            for _ in range(200):
                tau = (a + b) / 2
                a, b = (tau, b) if np.clip(v - tau, 0.05, 0.3).sum() > 1 else (a, tau)
            np.testing.assert_allclose(project_capped_simplex(v, 0.05, 0.3),
                                       np.clip(v - (a + b) / 2, 0.05, 0.3), atol=1e-12)

    def test_projection_keeps_feasible_point(self):
        w = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(project_capped_simplex(w, 0.1, 0.6), w, atol=1e-12)

    def test_repair_always_feasible(self):
        for _ in range(2000):
            v = self.rng.normal(0, 1, 8) * (self.rng.random(8) < self.rng.random())
            w = repair(v, self.inst, self.rng)
            self.assertEqual(check(w, self.inst), [], v)

    def test_repair_trims_and_fills(self):
        w = repair(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]), self.inst)
        self.assertEqual(np.flatnonzero(w).tolist(), [0, 1, 2, 3, 4])   # 5 najväčších
        w = repair(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), self.inst, priority=np.arange(8))
        self.assertEqual(sorted(np.flatnonzero(w).tolist()), [0, 6, 7])  # doplnené podľa priority

    def test_decoder_always_feasible_and_deterministic(self):
        for _ in range(2000):
            s = self.rng.random(8)
            w = decode_scores(s, self.inst)
            self.assertEqual(check(w, self.inst), [])
            np.testing.assert_array_equal(w, decode_scores(s, self.inst))

    def test_random_feasible(self):
        for _ in range(2000):
            self.assertEqual(check(random_feasible(self.inst, self.rng), self.inst), [])

    def test_random_feasible_tight_and_impossible_rules(self):
        # K * wmax = 1: jediné prípustné riešenie je 5 aktív po 20 % (predtým nekonečný cyklus).
        tight = self.inst.with_rules(wmax=0.2)
        self.assertEqual(feasible_counts(tight), [5])
        for _ in range(200):
            w = random_feasible(tight, self.rng)
            self.assertEqual(check(w, tight), [])
            np.testing.assert_allclose(w[w > 0], 0.2)
        # Pravidlá, ktoré sa nedajú splniť (5 * 15 % < 100 %), vyhodia chybu namiesto zaseknutia.
        with self.assertRaises(ValueError):
            random_feasible(self.inst.with_rules(wmax=0.15), self.rng)

    def test_random_feasible_project_rules(self):
        # Pravidlá z projektu a prísnejší strop 20 % plánovaný pre míľnik 3 (30 aktív).
        inst = toy(np.ones((5, 30)) * 10, kmin=5, kmax=10, wmin=0.05, wmax=0.30)
        for rules in ({}, {"wmax": 0.20}):
            r = inst.with_rules(**rules)
            for _ in range(500):
                self.assertEqual(check(random_feasible(r, self.rng), r), [])

    def test_repair_requires_rng_for_random_fill(self):
        with self.assertRaises(ValueError):
            repair(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), self.inst)

    def test_decoder_fills_by_score(self):
        # Nad prahom 0,5 je len jedno skóre -> doplnia sa aktíva s najvyšším skóre pod prahom.
        s = np.array([0.9, 0.1, 0.45, 0.2, 0.3, 0.4, 0.05, 0.0])
        w = decode_scores(s, self.inst)
        self.assertEqual(sorted(np.flatnonzero(w).tolist()), [0, 2, 5])
        self.assertEqual(check(w, self.inst), [])


class TestPenaltyAndDiversity(unittest.TestCase):
    def setUp(self):
        self.inst = toy(np.ones((5, 8)) * 10, kmin=3, kmax=5, wmin=0.1, wmax=0.4)

    def test_n_eff(self):
        self.assertAlmostEqual(n_eff(np.array([1.0, 0, 0])), 1.0)
        self.assertAlmostEqual(n_eff(np.full(4, 0.25)), 4.0)

    def test_violation_zero_iff_feasible(self):
        rng = np.random.default_rng(3)
        for _ in range(2000):
            v = rng.random(8) * (rng.random(8) < 0.6)
            if v.sum() == 0:
                continue
            w = normalize(v)
            self.assertEqual(violation_rules(w, self.inst) == 0, check(w, self.inst) == [])

    def test_violation_grows_with_concentration(self):
        one = np.array([1.0] + [0] * 7)
        two = np.array([0.5, 0.5] + [0] * 6)
        ok = np.array([0.3, 0.3, 0.4] + [0] * 5)
        self.assertGreater(violation_rules(one, self.inst), violation_rules(two, self.inst))
        self.assertEqual(violation_rules(ok, self.inst), 0)
        self.assertGreater(violation_diversity(one, self.inst, 3), violation_diversity(two, self.inst, 3))
        self.assertEqual(violation_diversity(np.array([0.34, 0.33, 0.33] + [0] * 5), self.inst, 2.9), 0)


class TestTopK(unittest.TestCase):
    def test_picks_best_assets(self):
        t = np.arange(100)[:, None]
        noise = np.random.default_rng(0).normal(0, 0.01, (100, 4)).cumsum(0)
        p = 100 * np.exp(np.array([0.0001, 0.003, 0.002, -0.001]) * t + noise)
        inst = toy(p)
        self.assertEqual(np.flatnonzero(top_k_sharpe(inst, 2)).tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
