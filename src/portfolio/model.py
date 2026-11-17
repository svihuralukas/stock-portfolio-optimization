"""Model úlohy: kandidát, kontrola obmedzení, simulácia a ohodnotenie.

Kandidát = vektor podielov w dĺžky n (n = 30). Aktívum i je vybrané práve vtedy,
keď w[i] > 0; samostatný vektor áno/nie nepotrebujeme (bol by redundantný).
Prípustný kandidát má každé w[i] z množiny {0} ∪ [wmin, wmax] (semi-spojitá premenná).
"""
import numpy as np

TOL = 1e-9
DAYS_PER_YEAR = 252


# ---------------------------------------------------------------- obmedzenia
def selected(w):
    return np.asarray(w) > TOL


def check(w, inst):
    """Zoznam porušených pravidiel (prázdny zoznam = prípustný kandidát)."""
    w = np.asarray(w, dtype=float)
    errors = []
    if w.shape != (inst.n,) or not np.all(np.isfinite(w)):
        return [f"tvar: očakávaný vektor {inst.n} konečných čísel"]
    if abs(w.sum() - 1.0) > 1e-6:
        errors.append(f"O1 súčet podielov {w.sum():.6f} != 1")
    if np.any(w < -TOL):
        errors.append(f"O2 záporné podiely: {np.flatnonzero(w < -TOL).tolist()}")
    sel = selected(w)
    k = int(sel.sum())
    if not inst.kmin <= k <= inst.kmax:
        errors.append(f"O3 počet aktív {k} mimo [{inst.kmin}, {inst.kmax}]")
    bad = np.flatnonzero(sel & ((w < inst.wmin - TOL) | (w > inst.wmax + TOL)))
    if bad.size:
        errors.append(f"O4 podiel mimo [{inst.wmin}, {inst.wmax}]: "
                      + ", ".join(f"{inst.assets[i]}={w[i]:.4f}" for i in bad))
    unavailable = np.flatnonzero(sel & ~inst.available)
    if unavailable.size:
        errors.append(f"O5 aktívum bez cien: {[inst.assets[i] for i in unavailable]}")
    return errors


def is_feasible(w, inst):
    return not check(w, inst)


# ---------------------------------------------------------------- simulácia
def portfolio_values(w, prices, budget):
    """Hodnota portfólia v každý deň obdobia (nákup v prvý deň, potom držba).

    kusy_i = w_i * rozpočet / cena_i(deň nákupu);  V(t) = sum_i kusy_i * cena_i(t).
    """
    w = np.asarray(w, dtype=float)
    sel = selected(w)
    p = prices[:, sel]
    shares = w[sel] * budget / p[0]
    return p @ shares


def sharpe_from_values(values):
    r = values[1:] / values[:-1] - 1.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(DAYS_PER_YEAR))


def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peak))


def n_eff(w):
    """Efektívny počet aktív 1 / sum w_i^2 (inverzný Herfindahlov index).

    Miera diverzifikácie nezávislá od výnosov: k aktív s rovnakým podielom dá k,
    jedno aktívum dá 1."""
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def evaluate(w, inst, period="train"):
    """Úplné ohodnotenie kandidáta na danom období ('train' alebo 'test')."""
    v = portfolio_values(w, inst.prices(period), inst.budget)
    r = v[1:] / v[:-1] - 1.0
    years = len(r) / DAYS_PER_YEAR
    return {
        "sharpe": sharpe_from_values(v),
        "ann_return": float((v[-1] / v[0]) ** (1.0 / years) - 1.0),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)),
        "max_drawdown": max_drawdown(v),
        "final_value": float(v[-1]),
        "n_assets": int(selected(w).sum()),
        "n_eff": n_eff(w),
    }


class Evaluator:
    """Rýchla ohodnocovacia funkcia (Sharpe na optimalizačnom období) s počítadlom.

    Používajú ju všetky metódy; počítadlo slúži na férové porovnanie pri rovnakom
    počte ohodnotení. Vďaka predpočítaným relatívnym cenám je jedno ohodnotenie
    jedno násobenie matice (dni x 30) vektorom.
    """

    def __init__(self, inst, period="train"):
        p = inst.prices(period)
        self.rel = p / p[0]          # cena_i(t) / cena_i(deň nákupu)
        self.count = 0

    def __call__(self, w):
        self.count += 1
        v = self.rel @ np.asarray(w, dtype=float)  # = V(t) / rozpočet
        return sharpe_from_values(v)
