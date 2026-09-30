"""
Per-symbol half-spread estimate for the cost model (SPEC §7, U8): Corwin & Schultz (2012) high-low estimator.

For consecutive bars (t−1, t), with bar t's high/low first shifted by any gap beyond the prior close (CS §IV.A
overnight adjustment):
    β = ln(H_{t−1}/L_{t−1})² + ln(H_t/L_t)²,   γ = ln(max(H_{t−1}, H_t) / min(L_{t−1}, L_t))²
    α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2)),   S = 2(e^α − 1)/(1 + e^α), negative S set to 0.

The estimator attributes part of a bar's range to volatility, and the part it cannot explain grows with the bar
length: on 2016–2025 SPY the mean half-spread is 14 bp from 1Day bars, 4.9 bp from 1Hour and 1.5 bp from 5Min (the
quoted half-spread is ≈ 0.1 bp). The spread is a property of the symbol, not of the traded timeframe, so it is
always estimated from **5Min** bars: per session, the mean S over the session's bar pairs (pairs straddling two
sessions excluded); the estimate used on a traded bar is the trailing mean over the `window_days` sessions **before**
that bar's session (causal for every timeframe), halved and floored.
"""

import numpy as np
import pandas as pd


def corwin_schultz(df: pd.DataFrame) -> pd.Series:
    """Per-pair spread estimate S_t (bars t−1, t; NaN at the first bar), negative estimates set to 0."""
    h, lo, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    h1, l1 = h[1:].copy(), lo[1:].copy()
    prev_c = c[:-1]
    up = l1 > prev_c  # gap up: shift bar t down so its low meets the prior close
    dn = h1 < prev_c
    shift = np.where(up, l1 - prev_c, np.where(dn, h1 - prev_c, 0.0))
    h1, l1 = h1 - shift, l1 - shift
    beta = np.log(h[:-1] / lo[:-1]) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(np.maximum(h[:-1], h1) / np.minimum(lo[:-1], l1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return pd.Series(np.r_[np.nan, np.maximum(s, 0.0)], index=df.index, name="cs_spread")


def session_spread(bars_5min: pd.DataFrame) -> pd.Series:
    """Mean CS spread per session (index = session date, tz-naive midnight) from intraday bars."""
    s = corwin_schultz(bars_5min)
    day = bars_5min.index.normalize()
    s[np.r_[True, day[1:] != day[:-1]]] = np.nan  # first bar of each session: its pair straddles two sessions
    return s.groupby(day.tz_localize(None) if day.tz is not None else day).mean()


def half_spread(index: pd.DatetimeIndex, bars_5min: pd.DataFrame, window_days: int, floor: float) -> pd.Series:
    """
    Half-spread to charge on a fill at each bar of `index`: max(floor, ½ · mean session spread over the
    `window_days` sessions before the bar's session). NaN when fewer than `window_days` earlier sessions exist —
    callers must supply enough 5Min history before the traded span.
    """
    daily = session_spread(bars_5min).rolling(window_days, min_periods=window_days).mean().shift(1) / 2
    day = index.normalize()
    day = day.tz_localize(None) if day.tz is not None else day
    est = daily.reindex(day).to_numpy()
    # A traded session absent from the 5Min data (dropped for coverage): the latest earlier estimate is still causal
    if np.isnan(est).any():
        prior = daily.dropna()
        pos = prior.index.searchsorted(day, side="right") - 1
        # searchsorted includes the bar's own session; its daily value is already lagged by one session
        fill = np.where(pos >= 0, prior.to_numpy()[np.maximum(pos, 0)], np.nan)
        est = np.where(np.isnan(est), fill, est)
    return pd.Series(np.maximum(est, floor), index=index, name="half_spread").where(~np.isnan(est))
