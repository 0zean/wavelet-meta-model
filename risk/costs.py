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


# ── Per-fill costs (SPEC §19, U13) ───────────────────────────────────────────

COST_MODELS = ("slippage", "cs", "quotes")
QUOTES_FLOOR_BP = 0.25
FILL_KINDS = ("open", "intra", "close")  # entries / gap exits / trims at the open; intrabar barriers; closes


def auction_flags(index: pd.DatetimeIndex, bar_minutes: int | None) -> tuple[np.ndarray, np.ndarray]:
    """
    (open fill is the opening auction, close fill is the closing auction) per bar of `index` (NY bar-open stamps).

    Daily bars: both. Intraday: the open of a bar stamped at the 09:30 session open; the close of a bar ending at or
    after 16:00 (a session-anchored stub bar included), or of the session's last bar in `index` when that bar spans
    13:00 (an early close; a regular session whose data stops at 13:00 would be misread as one).
    """
    n = len(index)
    if bar_minutes is None:
        return np.ones(n, bool), np.ones(n, bool)
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    start = (local.hour * 60 + local.minute).to_numpy()
    end = start + bar_minutes
    day = local.normalize()
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    is_open = start == 9 * 60 + 30
    is_close = (end >= 16 * 60) | (last & (start < 13 * 60) & (end >= 13 * 60))
    return is_open, is_close


def _bin_label(minutes: np.ndarray, bin_minutes: int) -> np.ndarray:
    m0 = 9 * 60 + 30
    start = m0 + ((minutes - m0) // bin_minutes) * bin_minutes
    start = np.clip(start, m0, 16 * 60 - bin_minutes)
    return np.array([f"{s // 60:02d}:{s % 60:02d}" for s in start], dtype=object)


def quotes_half_spread(
    years: np.ndarray, bins: np.ndarray, table: pd.DataFrame, floor_bp: float = QUOTES_FLOOR_BP
) -> np.ndarray:
    """
    Half-spread (fraction) for each (year, bin) from one symbol's quotes table (data.quotes; columns year, bin,
    half_spread_bp): the table's year, else the nearest earlier year; floored at `floor_bp`. A year before the
    table's first raises (no later year stands in for an earlier one).
    """
    lookup = {(int(y), b): v for y, b, v in zip(table["year"], table["bin"], table["half_spread_bp"])}
    have = np.array(sorted({int(y) for y in table["year"]}))
    out = np.empty(len(years))
    cache: dict[tuple[int, str], float] = {}
    for i, (y, b) in enumerate(zip(years, bins)):
        key = (int(y), b)
        if key not in cache:
            earlier = have[have <= key[0]]
            if not len(earlier):
                raise ValueError(f"quotes table has no year <= {key[0]} (first: {have[0] if len(have) else None})")
            yy = next((int(e) for e in earlier[::-1] if (int(e), b) in lookup), None)
            if yy is None:
                raise ValueError(f"quotes table has no {b!r} bin in any year <= {key[0]}")
            cache[key] = max(lookup[(yy, b)], floor_bp) * 1e-4
        out[i] = cache[key]
    return out


def fill_costs(index: pd.DatetimeIndex, cfg, cost_data=None) -> pd.DataFrame | None:
    """
    One-way cost fraction of a fill at each bar of `index`, by fill kind (columns `open`, `intra`, `close`), for
    cfg.COST_MODEL; None for "slippage" (SLIPPAGE_PCT on every fill, the pre-U13 backtest).

    - "cs": SLIPPAGE_PCT + the trailing Corwin–Schultz half-spread (`half_spread`, the risk profile's window and
      floor) on every fill; `cost_data` = the symbol's 5Min bars with history before `index`.
    - "quotes": `cost_data` = the symbol's rows of the quotes table. Opening / closing auction fills (auction_flags)
      pay the auction proxy half-spread only; other fills pay SLIPPAGE_PCT + the half-spread of the time-of-day bin
      of the fill (open: the bar's start, close: its end, intrabar: its start; daily bars' intrabar fills: `day`).
    """
    from data.timeframes import get_timeframe
    from risk.profiles import get_profile

    model = cfg.COST_MODEL
    if model == "slippage":
        return None
    if cost_data is None:
        raise ValueError(f"COST_MODEL={model!r} needs cost data for the symbol (5Min bars for 'cs', quotes rows "
                         "for 'quotes')")  # fmt: skip
    slip = cfg.SLIPPAGE_PCT
    if model == "cs":
        prof = get_profile(cfg.RISK_PROFILE)
        c = slip + half_spread(index, cost_data, prof.spread_window_days, prof.spread_floor).to_numpy()
        return pd.DataFrame({k: c for k in FILL_KINDS}, index=index)
    if model != "quotes":
        raise ValueError(f"COST_MODEL must be one of {COST_MODELS}, got {model!r}")
    from data.quotes import BIN_MINUTES

    minutes = get_timeframe(cfg.TIMEFRAME).minutes
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    years = local.year.to_numpy()
    is_open, is_close = auction_flags(index, minutes)
    n = len(index)

    def hs(bins):
        return quotes_half_spread(years, bins, cost_data)

    if minutes is None:
        open_c = hs(np.full(n, "open_auction", dtype=object))
        close_c = hs(np.full(n, "close_auction", dtype=object))
        intra = slip + hs(np.full(n, "day", dtype=object))
    else:
        start = (local.hour * 60 + local.minute).to_numpy()
        reg_open = slip + hs(_bin_label(start, BIN_MINUTES))
        reg_close = slip + hs(_bin_label(start + minutes, BIN_MINUTES))
        intra = reg_open
        open_c = np.where(is_open, hs(np.full(n, "open_auction", dtype=object)), reg_open)
        close_c = np.where(is_close, hs(np.full(n, "close_auction", dtype=object)), reg_close)
    return pd.DataFrame({"open": open_c, "intra": intra, "close": close_c}, index=index)
