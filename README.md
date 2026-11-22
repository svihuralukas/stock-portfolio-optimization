# Variant 4A – Výber akcií alebo indexov

Zadanie 2 z predmetu **Moderné prístupy v optimalizácii**, míľnik 2 (implementácia modelu a baseline).
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

## Míľnik 2 – kde čo nájdete

| požiadavka zadania | súbory |
|---|---|
| reprezentácia kandidáta, účelová funkcia, ohodnotenie, kontrola obmedzení | `src/portfolio/model.py`, `src/portfolio/repair.py` |
| príprava inštancií | `src/portfolio/universe.py`, `src/portfolio/instance.py`, `scripts/fetch_data.py`, `data/prices.csv` |
| baseline | `src/portfolio/baselines.py`, `scripts/run_baselines.py`, `results/baselines.csv` |
| overenie na príkladoch | `tests/test_model.py`, `scripts/examples.py`, `results/examples.txt` |
| penalizácia vs. striktné obmedzenia (spätná väzba) | `scripts/experiment_constraints.py`, `results/e1_*.csv`, `results/e2_*.csv` |
| návrh ďalších metód, správa | `report/main.tex`, `report/main.pdf` (časť 8) |

## Spustenie

Python 3.10+.

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

Baseline a referenčné prehľadávania na všetkých inštanciách (1–4 min podľa počítača; `results/baselines.csv`, tabuľka `report/tables/baselines.tex`, grafy v `report/figures/`):

```bash
python scripts/run_baselines.py
```

Experimenty k diverzifikácii a práci s obmedzeniami (E1, E2; 1–4 min podľa počítača; `results/e1_*.csv`, `results/e2_*.csv`, tabuľka `report/tables/e2.tex`):

```bash
python scripts/experiment_constraints.py
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
scripts/         spúšťacie skripty (dáta, príklady, baseline, experimenty)
tests/           unit testy modelu
results/         výstupy experimentov (CSV, príklady v examples.txt)
report/          správa k míľniku 2 (main.tex, figures/)
```

## Správa

`report/main.tex` (LaTeX, rovnaký štýl ako míľnik 1). Tabuľky sa vkladajú z
`report/tables/baselines.tex` (generuje `run_baselines.py`) a `report/tables/e2.tex`
(generuje `experiment_constraints.py`). Kompiluje sa cez XeLaTeX.
