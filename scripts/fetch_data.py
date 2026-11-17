"""Stiahne denné upravené ceny (dividendy a splity zahrnuté) z Yahoo Finance
a uloží ich do data/prices.csv. Stačí spustiť raz; ostatné skripty čítajú CSV.

    python scripts/fetch_data.py
"""
import sys
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio.universe import all_tickers  # noqa: E402


def main():
    tickers = all_tickers()
    data = yf.download(tickers, start="2010-01-01", end="2026-01-01",
                       auto_adjust=True, progress=False)["Close"]
    data = data[tickers].round(6)
    out = ROOT / "data" / "prices.csv"
    out.parent.mkdir(exist_ok=True)
    data.to_csv(out)
    print(f"{len(tickers)} aktív, {len(data)} dní -> {out}")
    print("chýbajúce hodnoty na aktívum:")
    print(data.isna().sum()[data.isna().sum() > 0])


if __name__ == "__main__":
    main()
