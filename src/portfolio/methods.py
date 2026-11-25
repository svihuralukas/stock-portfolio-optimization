"""Metódy pre míľnik 3: genetický algoritmus a hybrid simulovaného žíhania so SLSQP.

Obe metódy používajú ten istý Evaluator (Sharpe na optimalizačnom období s počítadlom)
ako baseline, takže ich porovnávame pri rovnakom počte ohodnotení. Výsledkom je vždy
prípustné portfólio (O1–O5).
"""
import numpy as np
from scipy.optimize import minimize

from .model import DAYS_PER_YEAR, TOL, Evaluator, check, selected
from .repair import project_capped_simplex, repair

ROUND = 10   # rovnaké zaokrúhlenie pri porovnávaní hodnôt ako v hill_climb


def _key(value):
    return round(float(value), ROUND)


class _Budget:
    """Evaluator s priebehom najlepšej prípustnej hodnoty po každom ohodnotení."""

    def __init__(self, inst, budget, trace=None):
        self.f = Evaluator(inst)
        self.budget = budget
        self.trace = trace
        self.best_w, self.best_f = None, -np.inf

    @property
    def count(self):
        return self.f.count

    def left(self):
        return self.budget - self.f.count

    def __call__(self, w, record=True):
        """Ohodnotí w; ako najlepšie si zapamätá len kandidátov s record=True (prípustných)."""
        fw = self.f(w)
        if record and _key(fw) > _key(self.best_f):
            self.best_w, self.best_f = np.array(w, dtype=float), fw
        self._log()
        return fw

    def _log(self):
        if self.trace is not None and self.f.count <= self.budget:
            self.trace[self.f.count - 1] = self.best_f


# ======================================================================= GA
def _crossover(a, b, inst, rng):
    """Zjednotenie aktív oboch rodičov, podiely spriemerované (0 tam, kde aktívum rodič nemá).

    Ak je aktív viac ako kmax, oprava ponechá kmax s najväčším podielom. Aby nevyhrávali
    vždy spoločné aktíva (majú dvojnásobný priemer), k priemeru pridáme malý šum."""
    child = 0.5 * (a + b)
    sel = selected(child)
    child[sel] *= rng.uniform(0.8, 1.2, sel.sum())
    return child


def _mutate(w, inst, rng):
    """Jedna z troch mutácií: výmena aktíva, presun časti podielu, pridanie/odobratie aktíva."""
    w = w.copy()
    sel = np.flatnonzero(selected(w))
    free = np.flatnonzero(inst.available & ~selected(w))
    op = rng.integers(3)
    if op == 0 and free.size:                       # výmena: nové aktívum prevezme podiel starého
        i, j = rng.choice(sel), rng.choice(free)
        w[j], w[i] = w[i], 0.0
    elif op == 1:                                    # presun časti podielu medzi dvoma vybranými
        i, j = rng.choice(sel, size=2, replace=False)
        d = rng.uniform(0, 0.1)
        w[i] -= d
        w[j] += d
    else:                                            # pridanie alebo odobratie aktíva
        can_add = sel.size < inst.kmax and free.size
        can_remove = sel.size > inst.kmin
        if can_add and (not can_remove or rng.random() < 0.5):
            w[rng.choice(free)] = inst.wmin
        elif can_remove:
            w[rng.choice(sel)] = 0.0
    return w


def genetic_algorithm(inst, budget, rng, pop_size=20, tournament=3, p_cross=0.9,
                      p_mut=0.4, elite=2, trace=None):
    """Genetický algoritmus s priamou reprezentáciou (vektor podielov) a opravou.

    Počiatočná populácia: náhodné prípustné portfóliá (rovnaký generátor ako hlavný baseline).
    Generácia: elita sa prenesie, ostatných vytvorí turnajový výber, kríženie, mutácia a oprava.
    Koniec po `budget` ohodnoteniach. Každý jedinec je po oprave prípustný.
    """
    from .baselines import random_feasible
    f = _Budget(inst, budget, trace)
    pop = [random_feasible(inst, rng) for _ in range(min(pop_size, budget))]
    fit = np.array([f(w) for w in pop])
    gens = 0
    while f.left() > 0:
        order = np.argsort(-fit, kind="stable")
        new = [pop[i] for i in order[:elite]]
        new_fit = list(fit[order[:elite]])

        def pick():
            idx = rng.choice(len(pop), size=tournament, replace=False)
            return pop[idx[np.argmax(fit[idx])]]

        while len(new) < pop_size and f.left() > 0:
            a, b = pick(), pick()
            child = _crossover(a, b, inst, rng) if rng.random() < p_cross else a.copy()
            if rng.random() < p_mut:
                child = _mutate(child, inst, rng)
            child = repair(child, inst, rng=rng)
            new.append(child)
            new_fit.append(f(child))
        pop, fit = new, np.array(new_fit)
        gens += 1
    assert not check(f.best_w, inst)
    return {"w": f.best_w, "f": f.best_f, "evals": f.count, "generations": gens}


# ================================================== hybrid SA + SLSQP (white-box)
class SharpeGrad:
    """Sharpe a jeho analytický gradient podľa podielov (white-box).

    v_t = sum_i R[t,i] w_i, r_t = v_{t+1}/v_t - 1, S = sqrt(252) * mean(r) / sd(r).
    dr_t/dw = (R[t+1] - (1 + r_t) R[t]) / v_t;  d sd = sum (r_t - mean) dr_t / ((N-1) sd).
    """

    def __init__(self, inst):
        p = inst.train_prices
        self.R = p / p[0]

    def __call__(self, w):
        v = self.R @ w
        r = v[1:] / v[:-1] - 1.0
        mu, sd = r.mean(), r.std(ddof=1)
        return float(mu / sd * np.sqrt(DAYS_PER_YEAR))

    def grad(self, w):
        R = self.R
        v = R @ w
        r = v[1:] / v[:-1] - 1.0
        n = len(r)
        mu, sd = r.mean(), r.std(ddof=1)
        dr = (R[1:] - (1.0 + r)[:, None] * R[:-1]) / v[:-1, None]     # (n, aktíva)
        dmu = dr.mean(0)
        dsd = ((r - mu)[:, None] * dr).sum(0) / ((n - 1) * sd)
        return np.sqrt(DAYS_PER_YEAR) * (dmu / sd - mu * dsd / sd ** 2)


def best_weights(S, w0, inst, f, sg, maxiter=50):
    """Najlepšie podiely pre pevnú množinu aktív S (SLSQP, analytický gradient).

    Podmienky: súčet 1, každý podiel v [wmin, wmax]. Každé volanie funkcie aj gradientu
    sa započíta do rozpočtu (f). Výsledok premietneme presne na prípustnú množinu
    (SLSQP môže hranice porušiť o ~1e-10) a znovu ohodnotíme."""
    S = np.asarray(S)

    def full(x):
        w = np.zeros(inst.n)
        w[S] = x
        return w

    def obj(x):
        if f.left() <= 0:
            raise _BudgetOut
        return -f(full(x), record=False)    # medzikrok SLSQP nemusí byť presne prípustný

    def jac(x):
        if f.left() <= 0:
            raise _BudgetOut
        f.f.count += 1                    # gradient je približne rovnako drahý ako ohodnotenie
        f._log()
        return -sg.grad(full(x))[S]

    x0 = project_capped_simplex(w0[S], inst.wmin, inst.wmax)
    x = x0
    try:
        res = minimize(obj, x0, jac=jac, method="SLSQP",
                       bounds=[(inst.wmin, inst.wmax)] * len(S),
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                                     "jac": lambda x: np.ones_like(x)}],
                       options={"maxiter": maxiter, "ftol": 1e-10})
        x = res.x
    except _BudgetOut:
        pass
    w = full(project_capped_simplex(x, inst.wmin, inst.wmax))
    if f.left() > 0:
        return w, f(w)
    return w, -np.inf


class _BudgetOut(Exception):
    pass


def _neighbor_set(S, inst, rng):
    """Sused množiny aktív: výmena, pridanie alebo odobratie jedného aktíva."""
    S = list(S)
    free = [i for i in np.flatnonzero(inst.available) if i not in S]
    moves = ["swap"] if free else []
    if len(S) < inst.kmax and free:
        moves.append("add")
    if len(S) > inst.kmin:
        moves.append("remove")
    m = moves[rng.integers(len(moves))]
    if m == "swap":
        k = rng.integers(len(S))
        out = S.copy(); new = rng.choice(free); old = out[k]; out[k] = new
        return sorted(out), (old, new)
    if m == "add":
        new = rng.choice(free)
        return sorted(S + [new]), (None, new)
    old = S[rng.integers(len(S))]
    return sorted(i for i in S if i != old), (old, None)


def sa_slsqp(inst, budget, rng, t0=0.01, t_end=0.001, maxiter=50, stall=300, kick=3, trace=None):
    """Hybrid: simulované žíhanie nad množinami aktív, podiely počíta SLSQP.

    Hodnota množiny = Sharpe po optimalizácii podielov. Teplota klesá geometricky podľa
    minutého rozpočtu (T0 -> T_end). Hodnoty už navštívených množín si pamätáme (bez ďalších
    ohodnotení). SLSQP štartuje z podielov aktuálneho riešenia (nové aktívum dostane podiel
    odobratého, resp. wmin), čo skracuje vnútorný beh.

    Reštart pri stagnácii (doplnené v M3): keď je `stall` návrhov po sebe už v pamäti
    (okolie je celé preskúmané a žíhanie by sa točilo bez minutia rozpočtu), skočíme
    z najlepšej množiny o `kick` náhodných krokov na ešte nenavštívenú množinu.
    """
    from .baselines import random_feasible
    f = _Budget(inst, budget, trace)
    sg = SharpeGrad(inst)
    cache = {}
    w = random_feasible(inst, rng)
    S = sorted(np.flatnonzero(selected(w)).tolist())
    w, fS = best_weights(S, w, inst, f, sg, maxiter)
    cache[tuple(S)] = (w, fS)
    steps = accepted = hits = restarts = 0
    while f.left() > 0:
        S2, (old, new) = _neighbor_set(S, inst, rng)
        key = tuple(S2)
        if key in cache and hits >= stall:
            # skok z najlepšej množiny na nenavštívenú množinu
            for _ in range(1000):
                S2 = sorted(np.flatnonzero(selected(f.best_w)).tolist())
                for _ in range(kick):
                    S2, _moves = _neighbor_set(S2, inst, rng)
                if tuple(S2) not in cache:
                    break
            key, old, new = tuple(S2), None, None
            restarts += 1
            hits = 0
            jumped = True
        else:
            jumped = False
        if key in cache:
            w2, f2 = cache[key]
            hits += 1
        else:
            hits = 0
            start = w.copy()
            if old is None and new is None:          # po skoku: rovnomerné podiely
                start = np.zeros(inst.n)
                start[S2] = 1.0 / len(S2)
            elif old is not None and new is not None:
                start[new], start[old] = start[old], 0.0
            elif new is not None:
                start[new] = inst.wmin
            else:
                start[old] = 0.0
            w2, f2 = best_weights(S2, start, inst, f, sg, maxiter)
            if f2 == -np.inf:
                break
            cache[key] = (w2, f2)
        steps += 1
        T = t0 * (t_end / t0) ** min(1.0, f.count / budget)
        if jumped or f2 >= fS or rng.random() < np.exp((f2 - fS) / T):
            S, w, fS = S2, w2, f2
            accepted += 1
    best = f.best_w
    assert not check(best, inst), check(best, inst)
    return {"w": best, "f": f.best_f, "evals": f.count, "sets": len(cache),
            "steps": steps, "accepted": accepted, "restarts": restarts}
