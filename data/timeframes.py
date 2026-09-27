from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Timeframe:
    """
    A bar timeframe.

    Native timeframes are fetched from Alpaca as-is. Others are resampled locally
    from RTH 5Min bars anchored at the session open: Alpaca's clock-aligned hourly
    bars would start at 09:00 and include pre-market minutes, and its daily bars
    include extended-hours trades (closes up to ~5% off the RTH close in 2020).
    """

    name: str
    minutes: int | None  # None = one bar per session (1Day)
    native: bool
    base: str | None = None  # timeframe resampled from, if not native

    @property
    def is_daily(self) -> bool:
        return self.minutes is None

    @property
    def delta(self) -> pd.Timedelta:
        if self.minutes is None:
            raise ValueError("1Day bars span a whole session; use the calendar for their length")
        return pd.Timedelta(minutes=self.minutes)


TIMEFRAMES: dict[str, Timeframe] = {
    "1Min": Timeframe("1Min", 1, native=True),
    "5Min": Timeframe("5Min", 5, native=True),
    "15Min": Timeframe("15Min", 15, native=False, base="5Min"),
    "30Min": Timeframe("30Min", 30, native=False, base="5Min"),
    "1Hour": Timeframe("1Hour", 60, native=False, base="5Min"),
    "1Day": Timeframe("1Day", None, native=False, base="5Min"),
}


def get_timeframe(name: str) -> Timeframe:
    try:
        return TIMEFRAMES[name]
    except KeyError:
        raise ValueError(f"Unknown timeframe {name!r}; expected one of {list(TIMEFRAMES)}") from None
