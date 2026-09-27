"""
Regular-trading-hours filtering and session-anchored resampling.

Bar stamps are bar OPEN times (tz-aware, America/New_York). A bar is kept only if it
opens at or after the session open and ends at or before the session close from the
exchange calendar, so early closes (e.g. 13:00) are respected.
"""

import numpy as np
import pandas as pd

from data.alpaca_source import NY_TZ
from data.timeframes import get_timeframe


def _session_bounds(index: pd.DatetimeIndex, calendar: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Tz-aware session open/close for each stamp's NY date (NaT on non-session dates)."""
    cal = calendar.reindex(index.tz_convert(NY_TZ).tz_localize(None).normalize())
    return pd.DatetimeIndex(cal["session_open"]), pd.DatetimeIndex(cal["session_close"])


def to_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a tz-aware (or naive UTC) frame to America/New_York with a nanosecond index."""
    out = df.tz_convert(NY_TZ) if df.index.tz is not None else df.tz_localize("UTC").tz_convert(NY_TZ)
    out.index = out.index.as_unit("ns")
    return out


def rth_filter(df: pd.DataFrame, calendar: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """
    Keep intraday bars whose whole interval [t, t + bar_minutes) lies inside a calendar session.

    Args:
        df (pd.DataFrame): Bars with a tz-aware index (any tz).
        calendar (pd.DataFrame): Sessions from calendar_from_raw (must cover df's dates).
        bar_minutes (int): Bar length in minutes.

    Returns:
        pd.DataFrame: The RTH subset, indexed in America/New_York.
    """
    df = to_ny(df)
    _check_calendar_covers(df.index, calendar)
    opens, closes = _session_bounds(df.index, calendar)
    keep = (df.index >= opens) & (df.index + pd.Timedelta(minutes=bar_minutes) <= closes)  # NaT compares False
    return df[keep]


def _check_calendar_covers(index: pd.DatetimeIndex, calendar: pd.DataFrame) -> None:
    if len(index) == 0:
        return
    first, last = index[0].tz_localize(None).normalize(), index[-1].tz_localize(None).normalize()
    if calendar.empty or calendar.index[0] > first or calendar.index[-1] < last:
        raise ValueError(f"Calendar does not cover {first.date()} → {last.date()}")


def resample_session(df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate RTH bars into `timeframe` bins anchored at each session's open.

    Bin k of a session covers [open + k·m, open + (k+1)·m) and is stamped with its
    start. The last bin of a session may be a stub (e.g. 15:30–16:00 for 1Hour, or
    shorter on early closes); it is kept. OHLC: first/max/min/last; volume and
    trade_count summed; vwap is the volume-weighted mean of the constituent vwaps.

    Args:
        df (pd.DataFrame): RTH bars (NY index) at a finer timeframe.
        calendar (pd.DataFrame): Sessions from calendar_from_raw.
        timeframe (str): Target timeframe, e.g. "1Hour".

    Returns:
        pd.DataFrame: Resampled bars, NY index, same columns as df.
    """
    tf = get_timeframe(timeframe)
    if tf.is_daily:
        raise ValueError("Use native 1Day bars or resample_daily for sessions")
    df = to_ny(df)
    if df.empty:
        return df.copy()
    _check_calendar_covers(df.index, calendar)
    opens, _ = _session_bounds(df.index, calendar)
    if opens.isna().any():
        raise ValueError("resample_session expects RTH bars only (found bars on non-session dates)")
    step = pd.Timedelta(minutes=tf.minutes)
    key = opens + ((df.index - opens) // step) * step
    return _aggregate(df, key)


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RTH intraday bars into one bar per session, stamped at session midnight (NY)."""
    df = to_ny(df)
    return _aggregate(df, df.index.normalize())


def _aggregate(df: pd.DataFrame, key: pd.DatetimeIndex) -> pd.DataFrame:
    g = df.assign(_pv=df["vwap"] * df["volume"]).groupby(key, sort=True)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _pv=("_pv", "sum"),
        trade_count=("trade_count", "sum"),
    )
    out["vwap"] = out["_pv"] / out["volume"].where(out["volume"] > 0)
    out.index = out.index.as_unit("ns").rename(None)
    return out[df.columns.tolist()]


def expected_bars_per_session(calendar: pd.DataFrame, timeframe: str) -> pd.Series:
    """Bars each session should contain: 1 for 1Day, ceil(session length / bar length) intraday (stub included)."""
    tf = get_timeframe(timeframe)
    if tf.is_daily:
        return pd.Series(1, index=calendar.index)
    minutes = (calendar["session_close"] - calendar["session_open"]).dt.total_seconds() / 60
    return np.ceil(minutes / tf.minutes).astype(int)


def session_coverage(df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str) -> pd.Series:
    """Fraction of expected bars present, per session between df's first and last bar (0 for empty sessions)."""
    day = df.index.tz_localize(None).normalize()
    sessions = calendar.loc[day[0] : day[-1]]
    present = pd.Series(1, index=day).groupby(level=0).size().reindex(sessions.index, fill_value=0)
    return present / expected_bars_per_session(sessions, timeframe)


def drop_incomplete_sessions(
    df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str, min_coverage: float, label: str = ""
) -> pd.DataFrame:
    """
    Remove every bar of sessions with fewer than `min_coverage` × expected bars
    (e.g. Alpaca has only the 09:30 bar for Nasdaq-listed names on 2018-05-02/03).
    Each dropped session is logged.
    """
    if df.empty or min_coverage <= 0:
        return df
    cov = session_coverage(df, calendar, timeframe)
    bad = cov[(cov > 0) & (cov < min_coverage)]
    if bad.empty:
        return df
    days = ", ".join(f"{d.date()} ({c:.0%})" for d, c in bad.items())
    print(f"[DATA]  {label}dropping {len(bad)} incomplete session(s) < {min_coverage:.0%} of bars: {days}")
    return df[~df.index.tz_localize(None).normalize().isin(bad.index)]


def sessions_missing_last_bar(df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str) -> pd.DatetimeIndex:
    """Sessions (NY dates) that have bars but whose final expected intraday bar is absent."""
    tf = get_timeframe(timeframe)
    if df.empty or tf.is_daily:
        return pd.DatetimeIndex([])
    day = df.index.tz_localize(None).normalize()
    last_bar = pd.Series(df.index, index=day).groupby(level=0).max()
    close = calendar["session_close"].reindex(last_bar.index)
    return last_bar.index[(last_bar + pd.Timedelta(minutes=tf.minutes) < close).to_numpy()]


def drop_sessions_missing_last_bar(df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str, label: str = ""):
    """Remove (and log) sessions whose last bar is missing — their daily close would not be the session close."""
    bad = sessions_missing_last_bar(df, calendar, timeframe)
    if bad.empty:
        return df
    print(
        f"[DATA]  {label}dropping {len(bad)} session(s) missing their last bar: {', '.join(str(d.date()) for d in bad)}"
    )
    return df[~df.index.tz_localize(None).normalize().isin(bad)]
