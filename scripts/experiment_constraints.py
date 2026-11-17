"""Experimenty k diverzifikácii a práci s obmedzeniami (odpoveď na spätnú väzbu).

E1: Kam smeruje maximalizácia Sharpe bez pravidiel O3/O4 (len súčet 1, bez shortu)?
    Presné riešenie (SLSQP, 20 štartov) na optimalizačnom období každej inštancie.
E2: Striktné obmedzenia (oprava/dekóder) vs. penalizácia pravidiel vs. penalizácia
    nízkej diverzifikácie. Rovnaký algoritmus (1+1 lokálne prehľadávanie),
    rovnaký rozpočet ohodnotení, 20 behov; len na ladiacej inštancii (rozhodnutie
    o formulácii nesmie vidieť dáta inštancií zo záverečného porovnania).

    python scripts/experiment_constraints.py          # ~1 min

V tabuľke pre správu sú N_eff aj Sharpe pre odovzdané (vždy prípustné) portfólio;
surové výsledky sú v results/e2_constraint_handling*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.baselines import hill_climb  # noqa: E402
from portfolio.instance import load_all  # noqa: E402
from portfolio.model import Evaluator, check, evaluate, sharpe_from_values  # noqa: E402
from portfolio.repair import repair, violation_diversity, violation_rules  # noqa: E402

RES = ROOT / "results"
BUDGET = 5000
RUNS = 20


def max_sharpe_unconstrained(inst, starts=20, seed=0):
    f = Evaluator(inst)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        x0 = rng.dirichlet(np.ones(inst.n))
        r = minimize(lambda w: -f(w), x0, method="SLSQP", bounds=[(0, 1)] * inst.n,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                     options={"maxiter": 500, "ftol": 1e-12})
        w = np.where(r.x > 1e-4, r.x, 0)
        w /= w.sum()
        if best is None or f(w) > f(best):
            best = w
    return best


def e1(instances):
    rows = []
    for name, inst in instances.items():
        w = max_sharpe_unconstrained(inst)
        tr = evaluate(w, inst, "train")
        single = np.array([sharpe_from_values(inst.train_prices[:, i]) for i in range(inst.n)])
        top = np.argsort(-w)[:4]
        rows.append({
            "inštancia": name,
            "najlepšie 1 aktívum": f"{inst.assets[single.argmax()]} ({single.max():.2f})",
            "Sharpe max (bez O3/O4)": round(tr["sharpe"], 2),
            "aktív > 0": tr["n_assets"],
            "N_eff": round(tr["n_eff"], 2),
            "najväčší podiel": round(w.max(), 3),
            "hlavné zložky": ", ".join(f"{inst.assets[i]} {w[i]*100:.0f}%" for i in top if w[i] > 0),
            "prípustné": not check(w, inst),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e1_unconstrained.csv", index=False)
    print(df.to_string(index=False))
    return df


VARIANTS = [
    ("striktné (oprava)", dict(handling="repair")),
    ("striktné, max. podiel 20 %", dict(handling="repair", rules=dict(wmax=0.20))),
    ("bez O3/O4", dict(handling="none")),
    ("penalizácia pravidiel λ=0,01", dict(handling="penalty", lam=0.01)),
    ("penalizácia pravidiel λ=0,1", dict(handling="penalty", lam=0.1)),
    ("penalizácia pravidiel λ=1", dict(handling="penalty", lam=1.0)),
    ("penalizácia N_eff<5 λ=0,1", dict(handling="penalty", lam=0.1, violation=violation_diversity)),
    ("penalizácia N_eff<5 λ=1", dict(handling="penalty", lam=1.0, violation=violation_diversity)),
]


def e2(inst):
    rows = []
    for label, kw in VARIANTS:
        for run in range(RUNS):
            rng = np.random.default_rng(1000 + run)
            args = dict(kw)
            run_inst = inst.with_rules(**args.pop("rules", {}))
            res = hill_climb(run_inst, BUDGET, rng, **args)
            w = res["w"]                                  # najlepší podľa vlastného f
            feasible = not check(w, inst)
            # Výsledok, ktorý by metóda odovzdala: najlepší prípustný videný počas behu,
            # a ak žiaden nie je, opravený najlepší kandidát.
            final = res["best_feasible"] if res["best_feasible"] is not None else repair(w, inst, rng)
            raw_tr, raw_te = evaluate(w, inst, "train"), evaluate(w, inst, "test")
            fin_tr, fin_te = evaluate(final, inst, "train"), evaluate(final, inst, "test")
            rows.append({
                "variant": label, "beh": run,
                "výsledok prípustný": feasible,
                "podiel neprípustných ohodnotení": res["infeasible_share"],
                "surový: Sharpe train": raw_tr["sharpe"], "surový: Sharpe test": raw_te["sharpe"],
                "surový: aktív": raw_tr["n_assets"], "surový: N_eff": raw_tr["n_eff"],
                "surový: max podiel": float(w.max()),
                "odovzdaný: Sharpe train": fin_tr["sharpe"], "odovzdaný: Sharpe test": fin_te["sharpe"],
                "odovzdaný: N_eff": fin_tr["n_eff"],
                "porušenie pravidiel": violation_rules(w, inst),
            })
        print(f"  {label}: hotovo")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e2_constraint_handling_runs.csv", index=False)
    agg = df.groupby("variant", sort=False).agg(
        prip=("výsledok prípustný", "mean"),
        neprip_ohodn=("podiel neprípustných ohodnotení", "mean"),
        raw_train=("surový: Sharpe train", "mean"),
        raw_test=("surový: Sharpe test", "mean"),
        raw_k=("surový: aktív", "mean"),
        raw_neff=("surový: N_eff", "mean"),
        raw_max=("surový: max podiel", "mean"),
        fin_neff=("odovzdaný: N_eff", "mean"),
        fin_train=("odovzdaný: Sharpe train", "mean"),
        fin_train_sd=("odovzdaný: Sharpe train", "std"),
        fin_test=("odovzdaný: Sharpe test", "mean"),
        fin_test_sd=("odovzdaný: Sharpe test", "std"),
    ).round(3)
    agg.to_csv(RES / "e2_constraint_handling.csv")
    print(agg.to_string())
    write_e2_latex(agg)
    return agg


def write_e2_latex(agg):
    def num(x, d=2):
        return f"{x:.{d}f}".replace(".", "{,}")

    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Variant & Prípustné & Nepríp. ohodn. & $N_\text{eff}$ & Sharpe opt. & Sharpe test \\",
             r"\midrule"]
    for variant, r in agg.iterrows():
        name = (variant.replace("N_eff<5", r"$N_\text{eff}<5$").replace(" %", r"\,\%")
                .replace("λ=", r"$\lambda=$").replace("0,", "0{,}"))
        lines.append(
            f"{name} & {r.prip * 100:.0f}\\,\\% & {num(r.neprip_ohodn * 100, 1)}\\,\\% & {num(r.fin_neff)} & "
            f"{num(r.fin_train)}\\,$\\pm$\\,{num(r.fin_train_sd, 3)} & {num(r.fin_test)}\\,$\\pm$\\,{num(r.fin_test_sd, 3)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "report" / "tables" / "e2.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RES.mkdir(exist_ok=True)
    instances = load_all()
    print("E1: maximalizácia Sharpe bez O3/O4")
    e1(instances)
    print(f"\nE2: práca s obmedzeniami, ladiaca inštancia, {RUNS} behov x {BUDGET} ohodnotení")
    e2(instances["ladiaca"])


if __name__ == "__main__":
    main()

