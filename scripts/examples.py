"""Konkrétne príklady: ohodnotenie kandidátov, kontrola obmedzení a oprava.

    python scripts/examples.py      # výstup aj do results/examples.txt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.instance import load_instance  # noqa: E402
from portfolio.model import check, evaluate  # noqa: E402
from portfolio.repair import decode_scores, repair  # noqa: E402

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def vec(inst, d):
    w = np.zeros(inst.n)
    for a, x in d.items():
        w[inst.assets.index(a)] = x
    return w


def show(inst, w):
    return ", ".join(f"{inst.assets[i]} {w[i]*100:.1f} %" for i in np.flatnonzero(w > 1e-9))


def report(inst, title, w):
    out(f"--- {title}")
    out(f"    portfólio: {show(inst, w)}")
    errors = check(w, inst)
    out(f"    kontrola:  {'prípustné' if not errors else '; '.join(errors)}")
    for period in ("train", "test"):
        r = evaluate(w, inst, period)
        out(f"    {period:5s}: Sharpe {r['sharpe']:.3f}, ročný výnos {r['ann_return']*100:.1f} %, "
            f"kolísanie {r['ann_vol']*100:.1f} %, prepad {r['max_drawdown']*100:.1f} %, "
            f"konečná hodnota {r['final_value']:.0f} USD, N_eff {r['n_eff']:.2f}")
    return errors


def main():
    inst = load_instance("I1")
    out(f"Inštancia {inst.name}: optimalizácia {str(inst.train_dates[0])[:10]}..{str(inst.train_dates[-1])[:10]}, "
        f"test {str(inst.test_dates[0])[:10]}..{str(inst.test_dates[-1])[:10]}")
    out(f"Aktíva: {', '.join(inst.assets)}")
    out()

    report(inst, "A) prípustný kandidát (príklad z míľnika 1)",
           vec(inst, {"SPY": .30, "VGK": .20, "GLD": .15, "MSFT": .20, "JNJ": .15}))
    report(inst, "B) 100 % v jednom aktíve (najlepší Sharpe spomedzi jednotlivých aktív)",
           vec(inst, {"MSFT": 1.0}))
    w = vec(inst, {"MSFT": 0.6, "AAPL": 0.4})
    report(inst, "C) dve akcie 60/40", w)
    out(f"    oprava -> {show(inst, repair(w, inst, np.random.default_rng(0)))}")
    w = vec(inst, {"SPY": .25, "TLT": .25, "GLD": .25, "MSFT": .02, "AAPL": .03, "JNJ": .2})
    report(inst, "D) súčet 100 %, ale dva podiely pod 5 %", w)
    out(f"    oprava -> {show(inst, repair(w, inst))}")
    w = np.full(inst.n, 1 / inst.n)
    report(inst, "E) rovnomerne všetkých 30 aktív (referencia 1/N)", w)
    out(f"    oprava -> {show(inst, repair(w, inst))}  (ponechá 10 aktív – pri zhode podielov prvých 10)")
    s = np.random.default_rng(42).random(inst.n)
    out("--- F) dekóder: náhodné skóre -> portfólio")
    out(f"    skóre nad 0,5: {int((s > 0.5).sum())} aktív -> ponecháme 10 s najvyšším skóre")
    w = decode_scores(s, inst)
    out(f"    portfólio: {show(inst, w)}")
    out(f"    kontrola:  {check(w, inst) or 'prípustné'}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "examples.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
