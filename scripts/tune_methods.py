"""Ladenie parametrov GA a SA+SLSQP – iba na ladiacej inštancii (2011–2014).

5 000 ohodnotení, 8 behov na nastavenie (seedy 1000–1007). Výstup: results/m3_tuning.csv.

    python scripts/tune_methods.py
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.instance import load_all  # noqa: E402
from portfolio.methods import genetic_algorithm, sa_slsqp  # noqa: E402

BUDGET, RUNS = 5000, 8


def main():
    inst = load_all()["ladiaca"]
    rows = []
    grids = [("GA", genetic_algorithm, {"pop_size": [20, 50, 100], "p_mut": [0.4, 0.7, 1.0]}),
             ("SA+SLSQP", sa_slsqp, {"t0": [0.01, 0.05, 0.2], "maxiter": [20, 50]})]
    for name, fn, grid in grids:
        keys = list(grid)
        for vals in itertools.product(*grid.values()):
            params = dict(zip(keys, vals))
            f, t = [], []
            for s in range(RUNS):
                t0 = time.perf_counter()
                f.append(fn(inst, BUDGET, np.random.default_rng(1000 + s), **params)["f"])
                t.append(time.perf_counter() - t0)
            rows.append({"metóda": name, **params, "sharpe_mean": np.mean(f),
                         "sharpe_sd": np.std(f, ddof=1), "čas_s": np.mean(t)})
            print(rows[-1])
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "m3_tuning.csv", index=False)


if __name__ == "__main__":
    main()
