"""Baseline a referenčné postupy na všetkých inštanciách.

  * 1/N všetkých 30 aktív (referencia, porušuje O3),
  * náhodné prípustné portfóliá – PRIEMER cez 10 000 kandidátov (hlavný baseline),
  * 10 náhodných aktív po 10 % (baseline z míľnika 1) – priemer cez 10 000,
  * greedy: 10 aktív s najvyšším vlastným Sharpe, po 10 %,
  * náhodné prehľadávanie (najlepší z 5 000) a (1+1) lokálne prehľadávanie, 20 behov.

    python scripts/run_baselines.py          # tabuľky do results/ a report/tables/, grafy do report/figures/
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.baselines import (equal_weight_all, hill_climb, random_equal,  # noqa: E402
                                 random_feasible, top_k_sharpe)
from portfolio.instance import load_all  # noqa: E402
from portfolio.model import Evaluator, check, evaluate  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
N_RANDOM = 10_000
BUDGET = 5000
RUNS = 20
METRICS = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "final_value", "n_eff"]


def metrics_row(method, inst, ws, extra=None):
    """Priemer (a smer. odchýlka) metrík cez zoznam portfólií ws."""
    row = {"inštancia": inst.name, "metóda": method, "počet": len(ws)}
    for period in ("train", "test"):
        ev = pd.DataFrame([evaluate(w, inst, period) for w in ws])
        for m in METRICS:
            row[f"{period}_{m}"] = ev[m].mean()
            row[f"{period}_{m}_sd"] = ev[m].std(ddof=1) if len(ws) > 1 else 0.0
    row.update(extra or {})
    return row


def random_search_trace(inst, rng):
    f = Evaluator(inst)
    best, best_f, trace = None, -np.inf, np.empty(BUDGET)
    for t in range(BUDGET):
        w = random_feasible(inst, rng)
        fw = f(w)
        if fw > best_f:
            best, best_f = w, fw
        trace[t] = best_f
    return best, trace


def hill_climb_trace(inst, rng):
    """Lokálne prehľadávanie s opravou; priebeh najlepšieho Sharpe po ohodnoteniach."""
    trace = np.empty(BUDGET)
    res = hill_climb(inst, BUDGET, rng, handling="repair", trace=trace)
    return res["w"], trace


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, random_tables, traces = [], {}, {}
    for name, inst in load_all().items():
        t0 = time.perf_counter()
        rng = np.random.default_rng(2026)
        rows.append(metrics_row("1/N všetkých 30 (mimo O3)", inst, [equal_weight_all(inst)]))

        rand = [random_feasible(inst, rng) for _ in range(N_RANDOM)]
        assert all(not check(w, inst) for w in rand)
        rows.append(metrics_row("náhodné prípustné (priemer)", inst, rand))
        random_tables[name] = pd.DataFrame({
            "train": [evaluate(w, inst, "train")["sharpe"] for w in rand],
            "test": [evaluate(w, inst, "test")["sharpe"] for w in rand],
        })
        rows.append(metrics_row("10 náhodných po 10 % (priemer)", inst,
                                [random_equal(inst, rng) for _ in range(N_RANDOM)]))
        rows.append(metrics_row("greedy top-10 Sharpe", inst, [top_k_sharpe(inst)]))

        rs, rs_tr, hc, hc_tr = [], [], [], []
        for run in range(RUNS):
            w, tr = random_search_trace(inst, np.random.default_rng(100 + run))
            rs.append(w)
            rs_tr.append(tr)
            w, tr = hill_climb_trace(inst, np.random.default_rng(200 + run))
            assert not check(w, inst)
            hc.append(w)
            hc_tr.append(tr)
        rows.append(metrics_row(f"náhodné prehľadávanie ({BUDGET} ohodn.)", inst, rs))
        rows.append(metrics_row(f"lokálne prehľadávanie ({BUDGET} ohodn.)", inst, hc))
        traces[name] = (np.array(rs_tr), np.array(hc_tr))
        print(f"{name}: {time.perf_counter() - t0:.0f} s")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "baselines.csv", index=False)
    show = df[["inštancia", "metóda", "train_sharpe", "test_sharpe", "test_sharpe_sd",
               "test_ann_return", "test_ann_vol", "test_max_drawdown", "test_final_value", "train_n_eff"]]
    print(show.round(3).to_string(index=False))

    # korelácia Sharpe train vs. test medzi náhodnými portfóliami
    corr = {n: t.corr(method="spearman").loc["train", "test"] for n, t in random_tables.items()}
    better = {n: (t["test"] > df[(df["inštancia"] == n) & df["metóda"].str.startswith("1/N")]["test_sharpe"].iloc[0]).mean()
              for n, t in random_tables.items()}
    summary = pd.DataFrame({"spearman_train_test": corr, "podiel_nahodnych_lepsich_ako_1N_test": better}).round(3)
    summary.to_csv(RES / "random_train_test_corr.csv")
    print(summary)

    write_latex(df)
    plot_random(random_tables, df)
    plot_traces(traces)


def write_latex(df):
    order = ["náhodné prípustné (priemer)", "10 náhodných po 10 % (priemer)", "greedy top-10 Sharpe",
             f"náhodné prehľadávanie ({BUDGET} ohodn.)", f"lokálne prehľadávanie ({BUDGET} ohodn.)",
             "1/N všetkých 30 (mimo O3)"]
    def num(x, d=2):
        return f"{x:.{d}f}".replace("-", "$-$").replace(".", "{,}")

    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Metóda & Sharpe opt. & Sharpe test & $N_\text{eff}$ & Výnos & Kolísanie & Prepad & Suma \\"]
    for name in df["inštancia"].unique():
        part = df[df["inštancia"] == name].set_index("metóda").loc[order]
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{name}}}}}\\")
        for m, r in part.iterrows():
            sd = f"\\,$\\pm$\\,{num(r.test_sharpe_sd)}" if r["počet"] > 1 else ""
            label = m.replace(" %", r"\,\%").replace(f"{BUDGET}", f"{BUDGET // 1000}\\,000")
            final = f"{r.test_final_value:,.0f}".replace(",", r"\,")
            lines.append(
                f"{label} & {num(r.train_sharpe)} & {num(r.test_sharpe)}{sd} & {num(r.train_n_eff)} & "
                f"{num(r.test_ann_return * 100, 1)} & "
                f"{num(r.test_ann_vol * 100, 1)} & {num(r.test_max_drawdown * 100, 1)} & {final}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "tables" / "baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_random(tables, df):
    names = list(tables)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    axes = axes.ravel()
    for ax, n in zip(axes, names):
        t = tables[n]
        ax.scatter(t["train"], t["test"], s=2, alpha=0.25, color="#7f8c8d", label="náhodné prípustné")
        part = df[df["inštancia"] == n]
        for key, color, marker in [("1/N", "#c0392b", "s"), ("greedy", "#8e44ad", "D"),
                                   ("náhodné preh", "#2980b9", "o"), ("lokálne", "#27ae60", "^")]:
            r = part[part["metóda"].str.startswith(key)].iloc[0]
            ax.scatter(r.train_sharpe, r.test_sharpe, color=color, marker=marker, s=55,
                       edgecolor="black", linewidth=0.5, label=r["metóda"].split(" (")[0], zorder=3)
        ax.set_title(f"{n} (ρ = {t.corr(method='spearman').iloc[0, 1]:.2f})")
        ax.grid(alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("Sharpe – optimalizačné obdobie")
    for ax in axes[::2]:
        ax.set_ylabel("Sharpe – testovacie obdobie")
    axes[-1].legend(loc="lower right", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(FIG / "random_train_test.png", dpi=200)
    plt.close(fig)


def plot_traces(traces):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.ravel()
    x = np.arange(1, BUDGET + 1)
    for ax, (n, (rs, hc)) in zip(axes, traces.items()):
        for arr, label, color in [(rs, "náhodné prehľadávanie", "#2980b9"), (hc, "lokálne prehľadávanie", "#27ae60")]:
            m = arr.mean(0)
            ax.plot(x, m, color=color, label=label)
            ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.2)
        ax.set_xscale("log")
        ax.set_title(n)
        ax.grid(alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("počet ohodnotení")
    for ax in axes[::2]:
        ax.set_ylabel("najlepší Sharpe (opt.)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
