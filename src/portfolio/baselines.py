"""Porovnávacie riešenia (baseline) a jednoduché referenčné prehľadávania."""
import numpy as np

from .model import Evaluator, check, sharpe_from_values
from .repair import decode_scores, normalize, repair, violation_rules


def equal_weight_all(inst):
    """1/N medzi všetky dostupné aktíva. Porušuje O3 (30 > 10) – len referencia."""
    w = inst.available.astype(float)
    return w / w.sum()


def random_feasible(inst, rng):
    """Náhodné prípustné portfólio, rovnomerne rozdelené pre daný výber aktív.

    K ~ U{kmin..kmax}, K aktív náhodne. Podiely: w = wmin + (1 - K*wmin) * d,
    d ~ Dirichlet(1,...,1) (rovnomerne na simplexe); kandidátov s niektorým
    w_i > wmax zamietneme. Afínny obraz rovnomerného rozdelenia so zamietnutím je
    rovnomerné rozdelenie na množine prípustných podielov – žiadna oprava netreba.
    """
    avail = np.flatnonzero(inst.available)
    k = int(rng.integers(inst.kmin, inst.kmax + 1))
    idx = rng.choice(avail, size=k, replace=False)
    while True:
        d = rng.dirichlet(np.ones(k))
        ws = inst.wmin + (1.0 - k * inst.wmin) * d
        if ws.max() <= inst.wmax:
            break
    w = np.zeros(inst.n)
    w[idx] = ws
    return w


def random_equal(inst, rng):
    """kmax náhodných aktív s rovnakými podielmi (z míľnika 1)."""
    w = np.zeros(inst.n)
    w[rng.choice(np.flatnonzero(inst.available), size=inst.kmax, replace=False)] = 1.0 / inst.kmax
    return w


def top_k_sharpe(inst, k=None):
    """Greedy: k aktív s najvyšším vlastným Sharpe na optimalizačnom období, rovnaké podiely.

    Odpovedá na otázku, čo dá 'investovať do najlepších akcií podľa Sharpe'
    bez ohľadu na to, ako sa aktíva navzájom dopĺňajú."""
    k = k or inst.kmax
    p = inst.train_prices
    s = np.array([sharpe_from_values(p[:, i]) if inst.available[i] else -np.inf
                  for i in range(inst.n)])
    w = np.zeros(inst.n)
    w[np.argsort(-s)[:k]] = 1.0 / k
    return w


def random_search(inst, budget, rng):
    """Náhodné prehľadávanie: `budget` náhodných prípustných kandidátov, vráti najlepšieho."""
    f = Evaluator(inst)
    best_w, best_f = None, -np.inf
    for _ in range(budget):
        w = random_feasible(inst, rng)
        fw = f(w)
        if fw > best_f:
            best_w, best_f = w, fw
    return best_w, best_f, f.count


# ----------------------------------------------------- lokálne prehľadávanie
def _mutate(v, rng, sigma=0.1, rate=None):
    """Gaussovská mutácia vektora z [0, 1]^n; v priemere sa zmenia 2 zložky."""
    n = len(v)
    rate = rate or 2.0 / n
    mask = rng.random(n) < rate
    if not mask.any():
        mask[rng.integers(n)] = True
    out = v.copy()
    out[mask] = np.clip(out[mask] + rng.normal(0, sigma, mask.sum()), 0.0, 1.0)
    return out


def hill_climb(inst, budget, rng, handling="repair", lam=1.0, violation=violation_rules, trace=None):
    """(1+1) lokálne prehľadávanie v priestore skóre v z [0, 1]^n.

    Rovnaký algoritmus, rôzna práca s obmedzeniami (handling):
      'repair'  – v sa dekóduje na prípustné portfólio (decode_scores); vždy prípustné.
      'penalty' – w = normalizované skóre nad prahom 0,5 (O1, O2 platia vždy),
                  f = Sharpe(w) - lam * violation(w); O3/O4 môžu byť porušené.
      'none'    – ako 'penalty' s lam = 0 (bez O3/O4, ukáže kam smeruje čistý Sharpe).
    Vracia najlepšie portfólio podľa f, najlepšie prípustné portfólio a štatistiky.
    Ak je zadané pole `trace`, zapíše doň priebeh najlepšieho f po každom ohodnotení.
    """
    f = Evaluator(inst)

    def to_w(v):
        if handling == "repair":
            return decode_scores(v, inst)
        return normalize(np.where(v > 0.5, v, 0.0))

    def fitness(w):
        s = f(w)
        if handling == "penalty":
            return s - lam * violation(w, inst)
        return s

    best_feasible, best_feasible_f = None, -np.inf
    infeasible_evals = 0
    v = w = None
    fv = -np.inf
    while f.count < budget:
        v2 = rng.random(inst.n) if v is None else _mutate(v, rng)
        w2 = to_w(v2)
        f2 = fitness(w2)
        if check(w2, inst):
            infeasible_evals += 1
        elif f2 > best_feasible_f:         # prípustný => penalizácia je 0, f2 = Sharpe
            best_feasible, best_feasible_f = w2, f2
        if f2 >= fv:                       # >= : prechádzame aj po plošinách
            v, w, fv = v2, w2, f2
        if trace is not None:
            trace[f.count - 1] = fv
    return {
        "w": w, "f": fv, "best_feasible": best_feasible,
        "evals": f.count, "infeasible_share": infeasible_evals / f.count,
    }
