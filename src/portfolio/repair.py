"""Práca s obmedzeniami: oprava (repair), dekóder skóre a penalizácia."""
import numpy as np

from .model import TOL, n_eff, selected


def project_capped_simplex(v, lo, hi, total=1.0):
    """Najbližší (euklidovsky) bod k v so súčtom `total` a zložkami v [lo, hi].

    Riešenie má tvar w_i = clip(v_i - tau, lo, hi) (podmienky KKT). Súčet je v tau
    nerastúca po častiach lineárna funkcia so zlomami v bodoch v_i - hi a v_i - lo,
    preto stačí nájsť úsek medzi dvoma zlomami a tau dopočítať lineárnou interpoláciou.
    Predpoklad: len(v)*lo <= total <= len(v)*hi.
    """
    v = np.asarray(v, dtype=float)
    k = len(v)
    if not k * lo - 1e-12 <= total <= k * hi + 1e-12:
        raise ValueError(f"{k} aktív s podielmi v [{lo}, {hi}] nedá súčet {total}")
    taus = np.sort(np.concatenate([v - hi, v - lo]))
    sums = np.clip(v[None, :] - taus[:, None], lo, hi).sum(1)    # nerastúce
    j = min(np.searchsorted(-sums, -total, side="right"), len(taus) - 1)  # prvý zlom so súčtom < total
    if j == 0 or sums[j - 1] == sums[j]:
        tau = taus[j]
    else:
        tau = taus[j - 1] + (sums[j - 1] - total) / (sums[j - 1] - sums[j]) * (taus[j] - taus[j - 1])
    return np.clip(v - tau, lo, hi)


def repair(v, inst, rng=None, priority=None):
    """Oprava ľubovoľného vektora na prípustné portfólio.

    1. (O2, O5) záporné podiely a nedostupné aktíva -> 0.
    2. (O3) Ak je vybraných viac ako kmax, ponecháme kmax s najväčším podielom.
       Ak menej ako kmin, doplníme aktíva s najvyššou prioritou (ak je zadaná),
       inak náhodne vybrané dostupné aktíva – vtedy treba zadať generátor `rng`
       (kvôli reprodukovateľnosti nepoužívame generátor bez seedu).
    3. (O1, O4) Podiely vybraných aktív premietneme na najbližší bod so súčtom 1
       a zložkami v [wmin, wmax]; nevybrané majú 0.
    Výsledok vždy spĺňa O1–O5 (pre K v [kmin, kmax] platí K*wmin <= 1 <= K*wmax).
    """
    v = np.where(inst.available, np.maximum(np.asarray(v, dtype=float), 0.0), 0.0)
    sel = np.flatnonzero(v > TOL)
    if sel.size > inst.kmax:
        sel = sel[np.argsort(-v[sel], kind="stable")[: inst.kmax]]
    elif sel.size < inst.kmin:
        free = np.flatnonzero(inst.available & ~np.isin(np.arange(inst.n), sel))
        need = inst.kmin - sel.size
        if priority is not None:
            add = free[np.argsort(-np.asarray(priority)[free], kind="stable")[:need]]
        else:
            if rng is None:
                raise ValueError("repair: na náhodné doplnenie aktív treba zadať rng (alebo priority)")
            add = rng.choice(free, size=need, replace=False)
        sel = np.concatenate([sel, add])
    w = np.zeros(inst.n)
    w[sel] = project_capped_simplex(v[sel], inst.wmin, inst.wmax)
    return w


def decode_scores(s, inst, threshold=0.5):
    """Reprezentácia cez skóre: 30 čísel z [0, 1] -> prípustné portfólio.

    Vyberieme aktíva so skóre nad prahom (počet orežeme na [kmin, kmax] podľa
    skóre), ich skóre premietneme na podiely so súčtom 1 v [wmin, wmax].
    Každý vektor skóre dá prípustné portfólio, takže na ňom môžu pracovať
    algoritmy pre reálne čísla bez ďalšej kontroly obmedzení.
    """
    s = np.asarray(s, dtype=float)
    return repair(np.where(s > threshold, s, 0.0), inst, priority=s)


# ---------------------------------------------------------------- penalizácia
def normalize(v):
    """Splní O1 a O2 bez ohľadu na O3/O4: záporné na 0, súčet na 1."""
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    total = v.sum()
    return v / total if total > 0 else np.full(len(v), 1.0 / len(v))


def violation_rules(w, inst):
    """Miera porušenia pravidiel O3 a O4 (0 = prípustné).

    Počet aktív mimo [kmin, kmax] meriame v počte chýbajúcich/prebytočných aktív,
    podiely mimo [wmin, wmax] v percentuálnych bodoch (0,05 = 5 p. b. = 1 jednotka).
    """
    sel = selected(w)
    k = int(sel.sum())
    count = max(0, inst.kmin - k) + max(0, k - inst.kmax)
    ws = np.asarray(w)[sel]
    share = (np.maximum(0, inst.wmin - ws).sum() + np.maximum(0, ws - inst.wmax).sum()) / 0.05
    unavailable = int((sel & ~inst.available).sum())
    return count + share + unavailable


def violation_diversity(w, inst, min_eff=5.0):
    """Alternatíva z konzultácie: penalizujeme nízku diverzifikáciu, nie počet aktív.

    Porušenie = o koľko je efektívny počet aktív menší ako min_eff, plus prekročenie
    najväčšieho podielu wmax (v jednotkách 5 p. b.). Portfólio s 1–2 aktívami je
    penalizované, rozumne rozložené portfólio nie.
    """
    over = np.maximum(0, np.asarray(w) - inst.wmax).sum() / 0.05
    return max(0.0, min_eff - n_eff(w)) + over
