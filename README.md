# Variant 4A – Výber akcií alebo indexov

Zadanie 2 z predmetu **Moderné prístupy v optimalizácii**, míľnik 3 (prvé experimenty s metódami GA a SA+SLSQP).
Tím: Lukáš Švihura, Dominik Keil, Ján Hrbál.

Na začiatku obdobia rozdelíme 10 000 USD medzi 5 až 10 z 30 aktív (15 ETF + 15 najväčších
amerických akcií), každé aktívum dostane 5–30 %. Portfólio držíme bez zmien. Maximalizujeme
Sharpeho pomer na optimalizačnom období (4 roky) a výsledok overujeme na nasledujúcom
testovacom období (3 roky, resp. 2 roky pri I3).

| inštancia | optimalizácia | test | použitie |
|---|---|---|---|
| ladiaca | 2011–2014 | 2015–2017 | ladenie parametrov a rozhodnutia o formulácii |
| I1 | 2014–2017 | 2018–2020 | záverečné porovnanie |
| I2 | 2017–2020 | 2021–2023 | záverečné porovnanie |
| I3 | 2020–2023 | 2024–2025 | záverečné porovnanie |

## Míľnik 3 – kde čo nájdete

| požiadavka zadania | súbory |
|---|---|
| implementácia oboch metód (GA s opravou, hybrid SA+SLSQP) | `src/portfolio/methods.py`, testy `tests/test_methods.py` |
| ladenie parametrov (len ladiaca inštancia) | `scripts/tune_methods.py`, `results/m3_tuning.csv` |
| porovnanie s baseline, viac behov, čas, štatistické testy | `scripts/run_methods.py`, `results/m3_runs.csv`, `m3_summary.csv`, `m3_tests.csv` |
| strop podielu 30 % vs 20 % | `scripts/experiment_cap.py`, `results/m3_cap.csv` |
| správa: výsledky, problémy, plán do M4 | `report/main.tex`, `report/main.pdf` (časti 7–9) |

## Míľnik 2 – kde čo nájdete

| požiadavka zadania | súbory |
|---|---|
| reprezentácia kandidáta, účelová funkcia, ohodnotenie, kontrola obmedzení | `src/portfolio/model.py`, `src/portfolio/repair.py` |
| príprava inštancií | `src/portfolio/universe.py`, `src/portfolio/instance.py`, `scripts/fetch_data.py`, `data/prices.csv` |
| baseline | `src/portfolio/baselines.py`, `scripts/run_baselines.py`, `results/baselines.csv` |
| overenie na príkladoch | `tests/test_model.py`, `scripts/examples.py`, `results/examples.txt` |
| penalizácia vs. striktné obmedzenia (spätná väzba) | `scripts/experiment_constraints.py`, `results/e1_*.csv`, `results/e2_*.csv` |
| návrh ďalších metód, správa | `report/main.tex`, `report/main.pdf` (časť 7) |

## Spustenie

Python 3.10+ (s `requirements.txt`). Presné verzie z `requirements-lock.txt` vyžadujú Python 3.11+.

```bash
pip install -r requirements.txt
```

Overenie modelu (testy na ručne spočítaných príkladoch):

```bash
python -m unittest discover -s tests -v
```

Príklady ohodnotenia, kontroly a opravy kandidátov na inštancii I1 (výstup aj do `results/examples.txt`):

```bash
python scripts/examples.py
```

Baseline a referenčné prehľadávania na všetkých inštanciách (`results/baselines.csv`, tabuľka `report/tables/baselines.tex`, grafy v `report/figures/`):

```bash
python scripts/run_baselines.py
```

Experimenty k diverzifikácii a práci s obmedzeniami (E1, E2; `results/e1_*.csv`, `results/e2_*.csv`, tabuľka `report/tables/e2.tex`):

```bash
python scripts/experiment_constraints.py
```

Míľnik 3 – ladenie metód na ladiacej inštancii (asi 2 min, `results/m3_tuning.csv`):

```bash
python scripts/tune_methods.py
```

Míľnik 3 – všetky metódy na všetkých inštanciách, 5 000 a 20 000 ohodnotení, 20 behov
(asi 10 min na 2 jadrách; `results/m3_*.csv`, tabuľka `report/tables/m3.tex`, grafy `report/figures/m3_*.png`).
S `--report` len znova vytvorí tabuľky a grafy z uložených výsledkov:

```bash
python scripts/run_methods.py
python scripts/run_methods.py --report
```

Míľnik 3 – strop podielu 30 % vs 20 % (asi 2 min, `results/m3_cap.csv`):

```bash
python scripts/experiment_cap.py
```

Dáta sú uložené v `data/prices.csv` (upravené záverečné ceny z Yahoo Finance, dividendy a splity
zahrnuté, stiahnuté 17. 11. 2026). Znova ich stiahne (Yahoo spätne prepočítava upravené ceny
pri každej dividende, preto nové stiahnutie dá mierne iné hodnoty; výsledky v správe sú
z priloženého `data/prices.csv`):

```bash
python scripts/fetch_data.py
```

Všetky náhodné postupy majú pevné seedy. Výsledky v `results/` a v správe boli vygenerované
s verziami knižníc v `requirements-lock.txt` (Python 3.13); pri iných verziách sa môžu
v posledných desatinných miestach líšiť. Lokálne prehľadávanie porovnáva hodnoty Sharpe
zaokrúhlené na 10 desatinných miest, aby drobné numerické rozdiely medzi počítačmi nemenili
jeho rozhodnutia.
Na Windows v PowerShell pred spustením nastavte `$env:PYTHONIOENCODING="utf-8"` (diakritika vo výpisoch).

## Štruktúra

```
src/portfolio/
  universe.py    zoznamy aktív a definícia inštancií (obdobia)
  instance.py    načítanie inštancie z data/prices.csv
  model.py       kandidát (vektor podielov), kontrola O1–O5, simulácia, Sharpe, metriky, Evaluator
  repair.py      oprava na prípustné portfólio, dekóder skóre, penalizačné funkcie
  baselines.py   1/N, náhodné prípustné portfóliá, greedy top-k, náhodné a lokálne prehľadávanie
  methods.py     genetický algoritmus s opravou, hybrid SA+SLSQP (analytický gradient Sharpe)
scripts/         spúšťacie skripty (dáta, príklady, baseline, experimenty)
tests/           unit testy modelu a metód
results/         výstupy experimentov (CSV, príklady v examples.txt)
report/          správa k míľniku 3 (main.pdf)
```
