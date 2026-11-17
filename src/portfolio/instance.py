"""Inštancia úlohy: ceny 30 aktív v optimalizačnom a testovacom období + pravidlá portfólia."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import ETFS, INSTANCES, STOCKS

ROOT = Path(__file__).resolve().parents[2]
PRICES_CSV = ROOT / "data" / "prices.csv"


@dataclass(frozen=True)
class Instance:
    name: str
    assets: tuple             # názvy 30 aktív (prvých 15 ETF, potom 15 akcií)
    train_dates: np.ndarray   # dni optimalizačného obdobia
    train_prices: np.ndarray  # tvar (dni, aktíva)
    test_dates: np.ndarray
    test_prices: np.ndarray
    available: np.ndarray     # (O5) aktívum má ceny počas celého optimalizačného obdobia
    tuning: bool = False      # ladiaca inštancia (nejde do záverečného porovnania)
    budget: float = 10_000.0
    kmin: int = 5             # (O3) najmenej aktív
    kmax: int = 10            # (O3) najviac aktív
    wmin: float = 0.05        # (O4) najmenší podiel vybraného aktíva
    wmax: float = 0.30        # (O4) najväčší podiel vybraného aktíva

    @property
    def n(self):
        return len(self.assets)

    def prices(self, period):
        return self.train_prices if period == "train" else self.test_prices

    def with_rules(self, **kw):
        """Kópia inštancie s inými pravidlami (napr. kmax=15, wmax=0.4)."""
        return replace(self, **kw)


def _window(prices, years):
    start, end = years
    return prices.loc[f"{start}-01-01":f"{end}-12-31"]


def load_instance(name, prices_csv=PRICES_CSV):
    """Načíta inštanciu podľa názvu ('ladiaca', 'I1', 'I2', 'I3')."""
    spec = {s[0]: s for s in INSTANCES}[name]
    _, train_years, test_years, rank_year, tuning = spec
    assets = ETFS + STOCKS[rank_year]
    all_prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)[assets]
    train = _window(all_prices, train_years)
    test = _window(all_prices, test_years)
    available = train.notna().all().to_numpy()
    # O5 overujeme len na optimalizačnom období (to investor v čase rozhodnutia pozná).
    # Na testovacom období musí mať ceny každé dostupné aktívum, inak by simulácia nebola možná.
    missing_test = [a for a, ok in zip(assets, available) if ok and test[a].isna().any()]
    if missing_test:
        raise ValueError(f"{name}: chýbajú testovacie ceny pre {missing_test}")
    return Instance(
        name=name,
        assets=tuple(assets),
        train_dates=train.index.to_numpy(),
        train_prices=train.to_numpy(dtype=float),
        test_dates=test.index.to_numpy(),
        test_prices=test.to_numpy(dtype=float),
        available=available,
        tuning=tuning,
    )


def load_all(prices_csv=PRICES_CSV):
    return {s[0]: load_instance(s[0], prices_csv) for s in INSTANCES}
