"""Míľnik 3: obe navrhnuté metódy spolu s baseline na všetkých inštanciách.

Metódy (všetky s rovnakým Evaluatorom, výsledok vždy prípustný):
  * náhodné prehľadávanie (baseline),
  * (1+1) lokálne prehľadávanie s dekóderom skóre (baseline z M2),
  * genetický algoritmus s priamou reprezentáciou a opravou,
  * hybrid: simulované žíhanie nad množinami aktív + SLSQP pre podiely (analytický gradient).
Rozpočty 5 000 a 20 000 ohodnotení, 20 behov s pevnými seedmi.

    python scripts/run_methods.py            # ~10 min na 2 jadrách: results/m3_*.csv/.npz, tabuľka, grafy
    python scripts/run_methods.py --report   # iba tabuľky a grafy z uložených výsledkov
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.baselines import hill_climb, random_feasible  # noqa: E402
from portfolio.instance import load_all  # noqa: E402
from portfolio.methods import genetic_algorithm, sa_slsqp  # noqa: E402
from portfolio.model import Evaluator, check, evaluate  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
BUDGETS = [5000, 20000]
RUNS = 20
# parametre z ladenia na ladiacej inštancii (scripts/tune_methods.py)
GA_PARAMS = dict(pop_size=20, p_mut=0.4)
SA_PARAMS = dict(t0=0.01, maxiter=50)
METHODS = ["náhodné prehľadávanie", "lokálne prehľadávanie", "GA", "SA+SLSQP"]
SEED0 = {"náhodné prehľadávanie": 100, "lokálne prehľadávanie": 200, "GA": 300, "SA+SLSQP": 400}


def random_search(inst, budget, rng, trace):
    f = Evaluator(inst)
    best, best_f = None, -np.inf
    for t in range(budget):
        w = random_feasible(inst, rng)
        fw = f(w)
        if fw > best_f:
            best, best_f = w, fw
        trace[t] = best_f
    return {"w": best, "f": best_f, "evals": f.count}


def local_search(inst, budget, rng, trace):
    res = hill_climb(inst, budget, rng, handling="repair", trace=trace)
    return {"w": res["w"], "f": res["f"], "evals": res["evals"]}


def run_one(args):
    name, method, budget, run = args
    inst = load_all()[name]
    rng = np.random.default_rng(SEED0[method] + run)
    trace = np.full(budget, np.nan)
    t0 = time.perf_counter()
    if method == "náhodné prehľadávanie":
        res = random_search(inst, budget, rng, trace)
    elif method == "lokálne prehľadávanie":
        res = local_search(inst, budget, rng, trace)
    elif method == "GA":
        res = genetic_algorithm(inst, budget, rng, trace=trace, **GA_PARAMS)
    else:
        res = sa_slsqp(inst, budget, rng, trace=trace, **SA_PARAMS)
    seconds = time.perf_counter() - t0
    w = res["w"]
    errors = check(w, inst)
    tr, te = evaluate(w, inst, "train"), evaluate(w, inst, "test")
    row = {"inštancia": name, "metóda": method, "rozpočet": budget, "beh": run,
           "seed": SEED0[method] + run, "ohodnotenia": res["evals"], "čas_s": seconds,
           "prípustné": not errors, "train_sharpe": tr["sharpe"], "test_sharpe": te["sharpe"],
           "test_ann_return": te["ann_return"], "test_ann_vol": te["ann_vol"],
           "test_max_drawdown": te["max_drawdown"], "test_final_value": te["final_value"],
           "n_assets": tr["n_assets"], "n_eff": tr["n_eff"],
           "aktíva": " ".join(f"{inst.assets[i]}:{w[i]:.3f}" for i in np.flatnonzero(w > 0))}
    # ohodnotenie, po ktorom metóda dosiahla 99,9 % svojej konečnej hodnoty
    trace[~np.isfinite(trace)] = np.nan       # pred prvým zaznamenaným kandidátom
    trace[:] = pd.Series(trace).ffill().to_numpy()
    final = trace[-1]
    row["ohodn_do_99_9"] = int(np.argmax(trace >= final - 0.001 * abs(final)) + 1)
    return row, trace


BKV_TOL_PCT = 0.001   # beh "našiel BKV", ak je od neho bližšie ako 0,001 %


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    names = list(load_all())
    if "--report" not in sys.argv:
        compute(names)
    report(names)


def compute(names):
    jobs = [(n, m, b, r) for n in names for b in BUDGETS for m in METHODS for r in range(RUNS)]
    t0 = time.perf_counter()
    with Pool() as pool:
        out = pool.map(run_one, jobs, chunksize=4)
    print(f"spolu {time.perf_counter() - t0:.0f} s")
    runs = pd.DataFrame([r for r, _ in out])
    assert runs["prípustné"].all(), "niektorý výsledok je neprípustný"
    runs.to_csv(RES / "m3_runs.csv", index=False)
    traces = {(j[0], j[1], j[2]): [] for j in jobs}
    for j, (_, tr) in zip(jobs, out):
        traces[(j[0], j[1], j[2])].append(tr)
    np.savez_compressed(RES / "m3_traces.npz", **{f"{k[0]}|{k[1]}|{k[2]}": np.array(v, dtype=np.float32)
                                                 for k, v in traces.items() if k[2] == BUDGETS[-1]})


def report(names):
    runs = pd.read_csv(RES / "m3_runs.csv")
    z = np.load(RES / "m3_traces.npz")
    traces = {tuple(k.split("|")[:2]) + (int(k.split("|")[2]),): z[k] for k in z.files}
    # najlepšia známa hodnota (BKV) na inštanciu = maximum cez všetky behy všetkých metód
    bkv = runs.groupby("inštancia")["train_sharpe"].max()
    runs["gap_pct"] = 100 * (bkv[runs["inštancia"]].to_numpy() - runs["train_sharpe"]) / bkv[runs["inštancia"]].to_numpy()
    runs["našiel_bkv"] = runs["gap_pct"] < BKV_TOL_PCT
    summary = summarize(runs)
    summary.to_csv(RES / "m3_summary.csv", index=False)
    tests = stat_tests(runs)
    tests.to_csv(RES / "m3_tests.csv", index=False)
    pd.Series(bkv, name="bkv_sharpe").to_csv(RES / "m3_bkv.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4).to_string(index=False))
        print(tests.round(4).to_string(index=False))
    write_latex(summary)
    plot_convergence(traces, bkv, names)
    plot_test(runs, names)


def summarize(runs):
    g = runs.groupby(["inštancia", "rozpočet", "metóda"], sort=False)
    s = g.agg(train_mean=("train_sharpe", "mean"), train_sd=("train_sharpe", "std"),
              train_min=("train_sharpe", "min"), train_max=("train_sharpe", "max"),
              gap_mean=("gap_pct", "mean"), bkv_share=("našiel_bkv", "mean"),
              test_mean=("test_sharpe", "mean"), test_sd=("test_sharpe", "std"),
              test_return=("test_ann_return", "mean"), test_final=("test_final_value", "mean"),
              n_eff=("n_eff", "mean"), distinct=("aktíva", "nunique"),
              time_mean=("čas_s", "mean"), evals99=("ohodn_do_99_9", "median")).reset_index()
    return s


def stat_tests(runs):
    """Mann–Whitney (behy sú nezávislé): GA a SA+SLSQP proti lokálnemu prehľadávaniu a navzájom."""
    rows = []
    for (name, budget), part in runs.groupby(["inštancia", "rozpočet"], sort=False):
        val = {m: part[part["metóda"] == m]["train_sharpe"].to_numpy() for m in METHODS}
        for a, b in [("GA", "lokálne prehľadávanie"), ("SA+SLSQP", "lokálne prehľadávanie"), ("SA+SLSQP", "GA")]:
            if np.ptp(np.concatenate([val[a], val[b]])) < 1e-12:
                p = 1.0
            else:
                p = mannwhitneyu(val[a], val[b], alternative="two-sided").pvalue
            rows.append({"inštancia": name, "rozpočet": budget, "A": a, "B": b,
                         "priemer_A": val[a].mean(), "priemer_B": val[b].mean(), "p": p})
    return pd.DataFrame(rows)


def num(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$").replace(".", "{,}")


def write_latex(s):
    short = {"náhodné prehľadávanie": "náhodné preh.", "lokálne prehľadávanie": "lokálne preh.",
             "GA": "GA", "SA+SLSQP": "SA+SLSQP"}
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Inšt. & Metóda & \multicolumn{2}{c}{Sharpe opt. (5\,000)} & \multicolumn{3}{c}{Sharpe opt. (20\,000)} & Test & Čas \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
             r" & & priemer $\pm$ sd & BKV & priemer $\pm$ sd & BKV & ohodn. & (20\,000) & [s] \\"]
    for name in s["inštancia"].unique():
        lines.append(r"\midrule")
        for i, m in enumerate(METHODS):
            a = s[(s["inštancia"] == name) & (s["rozpočet"] == 5000) & (s["metóda"] == m)].iloc[0]
            b = s[(s["inštancia"] == name) & (s["rozpočet"] == 20000) & (s["metóda"] == m)].iloc[0]
            lines.append(
                f"{name if i == 0 else ''} & {short[m]} & {num(a.train_mean)}\\,$\\pm$\\,{num(a.train_sd)} & "
                f"{a.bkv_share * 100:.0f}\\,\\% & {num(b.train_mean)}\\,$\\pm$\\,{num(b.train_sd)} & "
                f"{b.bkv_share * 100:.0f}\\,\\% & {int(b.evals99)} & {num(b.test_mean, 2)} & {num(b.time_mean, 1)}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "m3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


COLORS = {"náhodné prehľadávanie": "#2980b9", "lokálne prehľadávanie": "#27ae60",
          "GA": "#e67e22", "SA+SLSQP": "#8e44ad"}


def plot_convergence(traces, bkv, names):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, n in zip(axes.ravel(), names):
        x = np.arange(1, BUDGETS[-1] + 1)
        for m in METHODS:
            arr = np.array(traces[(n, m, BUDGETS[-1])])
            gap = 100 * (bkv[n] - arr) / bkv[n]
            ax.plot(x, np.nanmedian(gap, 0), color=COLORS[m], label=m)
            ax.fill_between(x, np.nanmin(gap, 0), np.nanmax(gap, 0), color=COLORS[m], alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=0)
        ax.set_title(n)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("počet ohodnotení")
    for ax in axes[:, 0]:
        ax.set_ylabel("odstup od BKV [%]")
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "m3_convergence.png", dpi=200)
    plt.close(fig)


def plot_test(runs, names):
    base = pd.read_csv(RES / "baselines.csv")
    fig, axes = plt.subplots(1, len(names), figsize=(10, 3.2), sharey=False)
    part = runs[runs["rozpočet"] == BUDGETS[-1]]
    for ax, n in zip(axes, names):
        d = part[part["inštancia"] == n]
        data = [d[d["metóda"] == m]["test_sharpe"].to_numpy() for m in METHODS]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops={"color": "black"})
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m])
        b = base[base["inštancia"] == n].set_index("metóda")
        ax.axhline(b.loc["náhodné prípustné (priemer)", "test_sharpe"], color="grey", ls="--", lw=1,
                   label="náhodné prípustné (priemer)")
        ax.axhline(b.loc["1/N všetkých 30 (mimo O3)", "test_sharpe"], color="#c0392b", ls=":", lw=1.2,
                   label="1/N (mimo O3)")
        ax.set_xticks(range(1, len(METHODS) + 1), ["NP", "LP", "GA", "SA"])
        ax.set_title(n)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Sharpe – test")
    axes[-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "m3_test.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
