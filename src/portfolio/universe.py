"""Definícia inštancií: obdobia a zoznamy aktív.

Každá inštancia = (optimalizačné obdobie, testovacie obdobie, 30 aktív).
ETF sú rovnaké vo všetkých inštanciách (všetky obchodované už pred rokom 2010).
Akcie = 15 najväčších amerických firiem podľa trhovej kapitalizácie ku koncu
roka pred začiatkom optimalizačného obdobia (nie podľa dneška – inak by v zozname
boli len firmy, o ktorých už vieme, že uspeli). Poradie je približné, podľa
verejných rebríčkov kapitalizácie ku koncu daného roka.
"""

ETFS = [
    "SPY",   # akcie USA (S&P 500)
    "IWM",   # malé firmy USA (Russell 2000)
    "VGK",   # akcie Európy
    "EWJ",   # akcie Japonska
    "EEM",   # rozvíjajúce sa trhy
    "AGG",   # dlhopisy USA (celý trh)
    "TLT",   # dlhodobé štátne dlhopisy USA
    "LQD",   # firemné dlhopisy
    "GLD",   # zlato
    "VNQ",   # nehnuteľnosti (REIT)
    "XLK",   # sektor technológie
    "XLE",   # sektor energie
    "XLV",   # sektor zdravotníctvo
    "XLF",   # sektor financie
    "XLP",   # sektor spotrebný tovar (defenzívny)
]

# 15 najväčších firiem ku koncu roka pred začiatkom optimalizačného obdobia
STOCKS = {
    2010: ["XOM", "AAPL", "MSFT", "BRK-B", "GE", "WMT", "GOOGL", "CVX",
           "IBM", "PG", "JNJ", "T", "JPM", "WFC", "KO"],
    2013: ["AAPL", "XOM", "GOOGL", "MSFT", "BRK-B", "GE", "JNJ", "WMT",
           "CVX", "WFC", "JPM", "PG", "IBM", "VZ", "PFE"],
    2016: ["AAPL", "GOOGL", "MSFT", "BRK-B", "AMZN", "XOM", "META", "JNJ",
           "JPM", "WFC", "GE", "T", "PG", "BAC", "CVX"],
    2019: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "V",
           "JNJ", "WMT", "BAC", "PG", "MA", "XOM", "T"],
}

# (názov, optimalizačné roky, testovacie roky, rok rebríčka akcií, ladiaca?)
INSTANCES = [
    ("ladiaca", (2011, 2014), (2015, 2017), 2010, True),
    ("I1",      (2014, 2017), (2018, 2020), 2013, False),
    ("I2",      (2017, 2020), (2021, 2023), 2016, False),
    ("I3",      (2020, 2023), (2024, 2025), 2019, False),
]


def all_tickers():
    tickers = list(ETFS)
    for lst in STOCKS.values():
        for t in lst:
            if t not in tickers:
                tickers.append(t)
    return tickers
