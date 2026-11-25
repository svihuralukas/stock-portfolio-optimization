"""M3: vplyv hornej hranice podielu (u = 30 % vs 20 %) na výsledok SA+SLSQP.

20 000 ohodnotení, 5 behov (SA+SLSQP je takmer deterministický). Výstup: results/m3_cap.csv.

    python scripts/experiment_cap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.instance import load_all  # noqa: E402
from portfolio.methods import sa_slsqp  # noqa: E402
from portfolio.model import check, evaluate  # noqa: E402

BUDGET, RUNS = 20000, 5


def main():
    rows = []
    for name, base in load_all().items():
        for u in (0.30, 0.20):
            inst = base.with_rules(wmax=u)
            for run in range(RUNS):
                res = sa_slsqp(inst, BUDGET, np.random.default_rng(500 + run))
                w = res["w"]
                assert not check(w, inst)
                tr, te = evaluate(w, inst, "train"), evaluate(w, inst, "test")
                rows.append({"inštancia": name, "u": u, "beh": run, "train_sharpe": tr["sharpe"],
                             "test_sharpe": te["sharpe"], "test_max_drawdown": te["max_drawdown"],
                             "n_assets": tr["n_assets"], "n_eff": tr["n_eff"], "max_podiel": w.max()})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "m3_cap.csv", index=False)
    s = df.groupby(["inštancia", "u"], sort=False).mean(numeric_only=True).drop(columns="beh")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
