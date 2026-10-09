"""
Sector mapping for the `cross_asset` group's "sector" context (SPEC §15, U15).

Each single stock maps to the cached sector ETF that holds it. AMZN (consumer discretionary, XLY) and GOOGL / META
(communication services, XLC since its 2018-06 launch; information technology before the 2018 GICS change) have no
sector ETF in the cached universe covering 2016 →, so they map to QQQ (the Nasdaq-100) as the closest cached proxy.
ETFs (sector ETFs included) have no sector context: their reference is the market (SPY) alone.
"""

MARKET = "SPY"

SECTOR = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "JPM": "XLF",
    "XOM": "XLE",
    "UNH": "XLV",
    "AMZN": "QQQ",
    "GOOGL": "QQQ",
    "META": "QQQ",
}


def sector_of(symbol: str) -> str | None:
    """The mapped sector ETF of `symbol`, or None (ETFs and unmapped symbols)."""
    return SECTOR.get(symbol.upper())
