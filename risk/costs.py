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

from pathlib import Path

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
# U22 (SPEC §20): a fill's class for the cost curve = its kind, or the auction when it is one (auction_flags)
FILL_CLASSES = ("open_auction", "open", "intra", "close", "close_auction")
MEASURED_SLIPPAGE_CSV = Path(__file__).resolve().parent.parent / "data" / "costs" / "measured_slippage.csv"


def measured_slippage(path: Path = MEASURED_SLIPPAGE_CSV) -> dict[str, float]:
    """Measured slippage per side in bp by fill kind (columns kind, slippage_bp, n; written by U27's reconciliation).
    Raises when the file does not exist yet or lacks a kind."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"SLIPPAGE_BP='measured' needs {path} (written by the forward test's fill reconciliation)"
        )
    t = pd.read_csv(path, dtype={"kind": str})
    out = {str(k): float(v) for k, v in zip(t["kind"], t["slippage_bp"])}
    missing = [k for k in FILL_KINDS if k not in out]
    if missing:
        raise ValueError(f"{path}: no measured slippage for fill kind(s) {missing}")
    return {k: out[k] for k in FILL_KINDS}


def slippage_by_kind(cfg) -> dict[str, float]:
    """One-way slippage fraction per fill kind for the quotes model: cfg.SLIPPAGE_BP (bp) or the measured profile."""
    bp = measured_slippage() if cfg.SLIPPAGE_BP == "measured" else cfg.SLIPPAGE_BP
    return {k: float(bp[k]) * 1e-4 for k in FILL_KINDS}


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


def asof_half_spread(
    days: np.ndarray, bins: np.ndarray, table: pd.DataFrame, floor_bp: float = QUOTES_FLOOR_BP
) -> tuple[np.ndarray, int]:
    """
    Half-spread (fraction) for each (NY date, bin) from one symbol's as-of quotes table (data.quotes
    build_asof_table; columns asof, bin, half_spread_bp): the latest `asof` on or before the date; a date before the
    first table's quarter takes the first table (returned as the count of such fills: the only look-ahead left).
    A bin missing from the chosen table raises.
    """
    asofs = np.array(sorted({str(a) for a in table["asof"]}), dtype="datetime64[D]")
    lookup = {(str(a), b): v for a, b, v in zip(table["asof"], table["bin"], table["half_spread_bp"])}
    day = np.asarray(days).astype("datetime64[D]")
    pos = np.searchsorted(asofs, day, side="right") - 1
    before = len(np.unique(day[pos < 0]))  # sessions priced with the first table
    pos = np.maximum(pos, 0)
    out = np.empty(len(day))
    cache: dict[tuple[int, str], float] = {}
    for i, (p, b) in enumerate(zip(pos, bins)):
        key = (int(p), b)
        if key not in cache:
            a = str(asofs[p])
            if (a, b) not in lookup:
                raise ValueError(f"as-of quotes table has no {b!r} bin as of {a}")
            cache[key] = max(lookup[(a, b)], floor_bp) * 1e-4
        out[i] = cache[key]
    return out, before


def fill_costs(index: pd.DatetimeIndex, cfg, cost_data=None, stress: pd.Series | None = None) -> pd.DataFrame | None:
    """
    One-way cost fraction of a fill at each bar of `index`, by fill kind (columns `open`, `intra`, `close`), for
    cfg.COST_MODEL; None for "slippage" (SLIPPAGE_PCT on every fill, the pre-U13 backtest).

    - "cs": SLIPPAGE_PCT + the trailing Corwin–Schultz half-spread (`half_spread`, the risk profile's window and
      floor) on every fill; `cost_data` = the symbol's 5Min bars with history before `index`.
    - "quotes": `cost_data` = the symbol's rows of a quotes table: the per-year table (column `year`, the U13
      medians) or the as-of table (column `asof`, U22: the sample weeks completed before the fill's quarter). Opening /
      closing auction fills (auction_flags) pay the auction proxy half-spread only; other fills pay the kind's
      slippage (cfg.SLIPPAGE_BP, SPEC §20) + the half-spread of the time-of-day bin of the fill (open: the bar's
      start, close: its end, intrabar: its start; daily bars' intrabar fills: `day`). In a stress session (`stress`:
      True per NY session date, from the previous session's VIX close >= utils.config.STRESS_VIX) every half-spread
      is multiplied by cfg.STRESS_MULT; a STRESS_MULT above 1 needs the flags. The frame's attrs record
      `asof_before_first` (fills priced with the first as-of table) and `stress_fills`.
    """
    from data.timeframes import get_timeframe
    from risk.profiles import get_profile

    model = cfg.COST_MODEL
    if model == "slippage":
        return None
    if cost_data is None:
        raise ValueError(f"COST_MODEL={model!r} needs cost data for the symbol (5Min bars for 'cs', quotes rows "
                         "for 'quotes')")  # fmt: skip
    if model == "cs":
        prof = get_profile(cfg.RISK_PROFILE)
        c = cfg.SLIPPAGE_PCT + half_spread(index, cost_data, prof.spread_window_days, prof.spread_floor).to_numpy()
        return pd.DataFrame({k: c for k in FILL_KINDS}, index=index)
    if model != "quotes":
        raise ValueError(f"COST_MODEL must be one of {COST_MODELS}, got {model!r}")
    from data.quotes import BIN_MINUTES

    slip = slippage_by_kind(cfg)
    minutes = get_timeframe(cfg.TIMEFRAME).minutes
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    years = local.year.to_numpy()
    days = local.normalize().tz_localize(None).to_numpy().astype("datetime64[D]") if local.tz is not None else \
        local.normalize().to_numpy().astype("datetime64[D]")  # fmt: skip
    is_open, is_close = auction_flags(index, minutes)
    n = len(index)
    mult = np.ones(n)
    if cfg.STRESS_MULT != 1.0:
        if stress is None:
            raise ValueError("STRESS_MULT above 1 needs the stress-session flags (previous VIX close >= STRESS_VIX)")
        flags = stress.reindex(pd.DatetimeIndex(days)).fillna(False).to_numpy(dtype=bool)
        mult = np.where(flags, float(cfg.STRESS_MULT), 1.0)
    asof_mode = "asof" in cost_data.columns
    before = 0

    def hs(bins):
        nonlocal before
        if asof_mode:
            v, b = asof_half_spread(days, bins, cost_data)
            before = max(before, b)
        else:
            v = quotes_half_spread(years, bins, cost_data)
        return v * mult

    if minutes is None:
        open_c = hs(np.full(n, "open_auction", dtype=object))
        close_c = hs(np.full(n, "close_auction", dtype=object))
        intra = slip["intra"] + hs(np.full(n, "day", dtype=object))
    else:
        start = (local.hour * 60 + local.minute).to_numpy()
        hs_start = hs(_bin_label(start, BIN_MINUTES))
        reg_open = slip["open"] + hs_start
        reg_close = slip["close"] + hs(_bin_label(start + minutes, BIN_MINUTES))
        intra = slip["intra"] + hs_start
        open_c = np.where(is_open, hs(np.full(n, "open_auction", dtype=object)), reg_open)
        close_c = np.where(is_close, hs(np.full(n, "close_auction", dtype=object)), reg_close)
    out = pd.DataFrame({"open": open_c, "intra": intra, "close": close_c}, index=index)
    out.attrs["asof_before_first"] = before
    out.attrs["stress_fills"] = int((mult != 1.0).sum())
    return out
