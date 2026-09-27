import numpy as np
import pandas as pd

from data.sessions import expected_bars_per_session, sessions_missing_last_bar


def quality_report(
    df: pd.DataFrame, calendar: pd.DataFrame, timeframe: str, base: pd.DataFrame | None = None, base_tf: str = "5Min"
) -> dict:
    """
    Data-quality summary for one (symbol, timeframe) bar set.

    Expected bars come from the exchange calendar between the first and last bar's
    sessions, so early closes are accounted for. Alpaca omits bars with no trades,
    so `missing_bars` > 0 is expected for thin names / 1Min.

    Args:
        df (pd.DataFrame): Bars from load_bars (NY index).
        calendar (pd.DataFrame): Sessions covering df.
        timeframe (str): df's timeframe.
        base (pd.DataFrame | None, optional): For resampled 1Day bars, the unfiltered intraday bars they
            were built from; session completeness is then measured on `base` (a 1-bar session looks
            complete at daily resolution). Defaults to None.
        base_tf (str, optional): Timeframe of `base`. Defaults to "5Min".

    Returns:
        dict: Summary statistics.
    """
    if df.empty:
        return {"n_bars": 0}
    if base is not None:
        rep = quality_report(base, calendar, base_tf)
        completeness = {
            k: rep[k]
            for k in ("sessions_incomplete", "incomplete_dates", "sessions_missing_last_bar", "missing_last_bar_dates")
        }
        return {**quality_report(df, calendar, timeframe), **completeness, "completeness_from": base_tf}
    day = df.index.tz_localize(None).normalize()
    sessions = calendar.loc[day[0] : day[-1]]
    expected = expected_bars_per_session(sessions, timeframe)
    per_session = pd.Series(1, index=day).groupby(level=0).size().reindex(sessions.index, fill_value=0)
    coverage = per_session / expected
    incomplete = coverage[(coverage > 0) & (coverage < 0.5)]
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    log_ret = np.log(c).diff()
    same_session = pd.Series(day, index=df.index).eq(pd.Series(day, index=df.index).shift())
    no_last = sessions_missing_last_bar(df, calendar, timeframe)
    ohlc_bad = (lo > o.combine(c, min)) | (h < o.combine(c, max)) | (lo > h)
    return {
        "first": df.index[0].isoformat(),
        "last": df.index[-1].isoformat(),
        "n_bars": len(df),
        "n_sessions": len(sessions),
        "sessions_missing": int((per_session == 0).sum()),
        "sessions_incomplete": len(incomplete),  # < 50% of expected bars
        "incomplete_dates": " ".join(str(d.date()) for d in incomplete.index),
        "sessions_missing_last_bar": len(no_last),
        "missing_last_bar_dates": " ".join(str(d.date()) for d in no_last[:20]),
        "early_close_sessions": int((expected < expected.max()).sum()) if len(expected) else 0,
        "expected_bars": int(expected.sum()),
        "missing_bars": int((expected - per_session).clip(lower=0).sum()),
        "missing_pct": float((expected - per_session).clip(lower=0).sum() / max(expected.sum(), 1)),
        "extra_bars": int((per_session - expected).clip(lower=0).sum()),
        "bars_off_calendar": int((~day.isin(calendar.index)).sum()),
        "duplicate_stamps": int(df.index.duplicated().sum()),
        "nan_rows": int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum()),
        "zero_volume": int((df["volume"] <= 0).sum()),
        "nonpositive_price": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "ohlc_violations": int(ohlc_bad.sum()),
        "max_abs_ret_intrasession": float(log_ret[same_session].abs().max()),
        "max_abs_ret_session_gap": float(log_ret[~same_session].abs().max()),
    }
