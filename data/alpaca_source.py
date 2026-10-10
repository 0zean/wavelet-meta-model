"""
Alpaca market data access. This is the only module that reads credentials.

Bars come back stamped with the bar OPEN time in UTC (Alpaca convention): a 5Min
bar stamped 14:30Z covers [14:30, 14:35) and is only known at 14:35.
"""

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from data.timeframes import get_timeframe

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
NY_TZ = "America/New_York"
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
_RAW_KEYS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count"}
FEEDS = ("sip", "iex")
ADJUSTMENTS = ("raw", "split", "dividend", "all")


class MissingCredentialsError(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    """API_KEY / SECRET_KEY from the environment, else from the repo's .env (never put into os.environ)."""
    file_vals = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    key = os.environ.get("API_KEY") or file_vals.get("API_KEY")
    secret = os.environ.get("SECRET_KEY") or file_vals.get("SECRET_KEY")
    missing = [name for name, val in (("API_KEY", key), ("SECRET_KEY", secret)) if not val]
    if missing:
        raise MissingCredentialsError(
            f"Alpaca credentials missing: set {', '.join(missing)} in the environment or {ENV_PATH}"
        )
    return key, secret


def bars_from_raw(raw: list[dict]) -> pd.DataFrame:
    """Alpaca raw bar dicts → UTC-indexed float64 frame with BAR_COLUMNS."""
    if not raw:
        return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"), dtype="float64")
    df = pd.DataFrame(raw)
    idx = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).rename(None)
    out = df.rename(columns=_RAW_KEYS).reindex(columns=BAR_COLUMNS).astype("float64")
    out.index = idx
    return out


class AlpacaSource:
    """Thin wrapper over alpaca-py; tests substitute any object with the same two methods."""

    def __init__(self, feed: str = "sip"):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        if feed not in FEEDS:
            raise ValueError(f"feed must be one of {FEEDS}, got {feed!r}")
        key, secret = _credentials()
        self.feed = feed
        self._data = StockHistoricalDataClient(key, secret, raw_data=True)
        self._trading = TradingClient(key, secret, paper=True, raw_data=True)

    def bars(self, symbol: str, timeframe: str, start: datetime, end: datetime, adjustment: str) -> pd.DataFrame:
        """
        Native Alpaca bars in [start, end), UTC index. Raises on any API error — including
        an unauthorised feed — rather than falling back to another feed.
        """
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = get_timeframe(timeframe)
        if not tf.native:
            raise ValueError(f"{timeframe} is not a native Alpaca timeframe; resample from {tf.base}")
        if adjustment not in ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {ADJUSTMENTS}, got {adjustment!r}")
        # 1DayPrint: Alpaca's daily bar (stamped at NY midnight; its open / close are the official auction prints)
        alpaca_tf = TimeFrame.Day if tf.is_daily else TimeFrame(tf.minutes, TimeFrameUnit.Minute)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=pd.Timestamp(start).tz_convert("UTC").to_pydatetime(),
            end=pd.Timestamp(end).tz_convert("UTC").to_pydatetime(),
            adjustment=Adjustment(adjustment),
            feed=DataFeed(self.feed),
        )
        raw = self._data.get_stock_bars(req)
        df = bars_from_raw(raw.get(symbol, []))
        # Alpaca's `end` is inclusive; keep [start, end)
        return df[df.index < pd.Timestamp(end).tz_convert("UTC")]

    def quotes(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
        """
        Raw SIP NBBO quotes in [start, end) per symbol ({"t", "bp", "ap", "bs", "as", ...}; every page fetched).
        Raises on any API error, as `bars` does.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest

        req = StockQuotesRequest(
            symbol_or_symbols=list(symbols),
            start=pd.Timestamp(start).tz_convert("UTC").to_pydatetime(),
            end=pd.Timestamp(end).tz_convert("UTC").to_pydatetime(),
            feed=DataFeed(self.feed),
        )
        raw = self._data.get_stock_quotes(req)
        stop = pd.Timestamp(end).tz_convert("UTC")
        out = {}
        for s, rows in raw.items():  # Alpaca's `end` is inclusive; keep [start, end) (rows come time-ascending)
            k = len(rows)
            while k and pd.Timestamp(rows[k - 1]["t"]) >= stop:
                k -= 1
            out[s] = rows[:k]
        return out

    def option_chain(self, underlying: str, expiry_lte: date) -> dict[str, dict]:
        """Current option snapshots of `underlying` (indicative feed on the free plan) expiring by `expiry_lte`."""
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest

        key, secret = _credentials()
        client = OptionHistoricalDataClient(key, secret, raw_data=True)
        return client.get_option_chain(OptionChainRequest(underlying_symbol=underlying, expiration_date_lte=expiry_lte))

    def option_contracts(self, underlying: str, expiry_lte: date) -> list[dict]:
        """Active option contracts of `underlying` expiring by `expiry_lte`, every page (open interest as of the
        contract's `open_interest_date`)."""
        from alpaca.trading.requests import GetOptionContractsRequest

        rows, token = [], None
        while True:
            req = GetOptionContractsRequest(underlying_symbols=[underlying], expiration_date_lte=expiry_lte,
                                            limit=10000, page_token=token)  # fmt: skip
            page = self._trading.get_option_contracts(req)
            rows += page.get("option_contracts", [])
            token = page.get("next_page_token")
            if not token:
                return rows

    def calendar(self, start: date, end: date) -> pd.DataFrame:
        """Trading sessions in [start, end] — see calendar_from_raw."""
        from alpaca.trading.requests import GetCalendarRequest

        raw = self._trading.get_calendar(GetCalendarRequest(start=start, end=end))
        return calendar_from_raw(raw)


def calendar_from_raw(raw: list[dict]) -> pd.DataFrame:
    """
    Alpaca calendar rows {"date": "YYYY-MM-DD", "open": "HH:MM", "close": "HH:MM"} (NY local)
    → frame indexed by session date with tz-aware NY `session_open` / `session_close`.
    """
    df = pd.DataFrame(raw)[["date", "open", "close"]]
    day = pd.to_datetime(df["date"])
    out = pd.DataFrame(
        {
            "session_open": pd.to_datetime(df["date"] + " " + df["open"]).dt.tz_localize(NY_TZ),
            "session_close": pd.to_datetime(df["date"] + " " + df["close"]).dt.tz_localize(NY_TZ),
        }
    )
    out.index = pd.DatetimeIndex(day, name="date")
    return out.sort_index()
