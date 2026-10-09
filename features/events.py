"""
Event samplers (SPEC §13, U14): when a primary may trade.

`sample_events(df, cfg)` returns the event frame (index = event bar t, column `width`). Every sampler keeps the §3
execution model: the decision is taken at the close of the event bar t and the fill is the open of the next bar in the
data, entry_pos = t + 1 (the schedule sampler picks t as the bar before its entry bar, so this holds by construction).

- `cusum` (§3): symmetric CUSUM on log returns, threshold CUSUM_MULT · σ_t.
- `dc`: directional change (Guillaume et al. 1997): an event at the bar that CONFIRMS a reversal, i.e. log price is
  δ_t = dc_mult · σ_t below the running maximum of an up run (or above the minimum of a down run). The run's
  overshoot (confirmation → extreme) is only known at the next confirmation (`dc_events`, the session group).
- `schedule`: fixed session times (`entry_times`), a calendar predicate (`days`) and an optional `gate` over the
  session frame (features.session) at the event bar.

σ_t is bar_volatility with the fold's VolProfile (cfg.VOL_PROFILE="tod") or plain (§3). Unless HOLD_OVERNIGHT,
cusum / dc events on a session's last bar are skipped (their entry would be next session); scheduled events are not
(their entry bar is explicit and the exit model decides).
"""

import numba
import numpy as np
import pandas as pd

from features.triple_barrier_labels import barrier_width, cusum_events
from features.vol_profile import (
    OPEN_MINUTE,
    VolProfile,
    bar_volatility,
    base_volatility,
    hold_scale,
    ny_dates,
    ny_minutes,
)
from utils.config import RunConfig

EVENT_SAMPLERS = ("cusum", "dc", "schedule")
DAY_PREDICATES = ("all", "fomc", "cpi_nfp", "tom", "non_macro", "earnings")
DEFAULT_DC_MULT = 2.0
CLOSE_MINUTE = 16 * 60
_DEFAULTS = {"cusum": {}, "dc": {"dc_mult": DEFAULT_DC_MULT}, "schedule": {"entry_times": None, "days": "all",
             "gate": None}}  # fmt: skip


def parse_time(s: str) -> int:
    """Minutes since midnight of "HH:MM" ("open" = 09:30)."""
    if s == "open":
        return OPEN_MINUTE
    try:
        h, m = s.split(":")
        out = int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        raise ValueError(f"time {s!r} is not 'HH:MM' or 'open'") from None
    if not 0 <= out < 24 * 60:
        raise ValueError(f"time {s!r} is out of range")
    return out


def event_params(cfg: RunConfig) -> dict:
    """cfg.EVENT_PARAMS merged over the sampler's defaults, validated (raises ValueError)."""
    sampler = cfg.EVENT_SAMPLER
    if sampler not in EVENT_SAMPLERS:
        raise ValueError(f"EVENT_SAMPLER must be one of {EVENT_SAMPLERS}, got {sampler!r}")
    unknown = set(cfg.EVENT_PARAMS) - set(_DEFAULTS[sampler])
    if unknown:
        raise ValueError(
            f"EVENT_PARAMS for {sampler!r}: unknown keys {sorted(unknown)}; allowed {sorted(_DEFAULTS[sampler])}"
        )
    p = {**_DEFAULTS[sampler], **cfg.EVENT_PARAMS}
    if sampler == "dc" and not (isinstance(p["dc_mult"], (int, float)) and p["dc_mult"] > 0):
        raise ValueError(f"EVENT_PARAMS dc_mult must be > 0, got {p['dc_mult']!r}")
    if sampler == "schedule":
        times = p["entry_times"]
        if not times or isinstance(times, str) or not all(isinstance(t, str) for t in times):
            raise ValueError(f"EVENT_PARAMS entry_times must be a non-empty list of 'HH:MM' strings, got {times!r}")
        from data.timeframes import get_timeframe

        minutes = get_timeframe(cfg.TIMEFRAME).minutes
        for t in times:
            m = parse_time(t)
            if minutes is None:
                if m != OPEN_MINUTE:
                    raise ValueError(f"1Day bars trade only at the open; entry time {t!r} is not 'open' / '09:30'")
            elif not (OPEN_MINUTE <= m < CLOSE_MINUTE and (m - OPEN_MINUTE) % minutes == 0):
                raise ValueError(f"entry time {t!r} is not a {cfg.TIMEFRAME} bar open between 09:30 and 16:00")
        if p["days"] not in DAY_PREDICATES:
            raise ValueError(f"EVENT_PARAMS days must be one of {DAY_PREDICATES}, got {p['days']!r}")
        if p["gate"] is not None and (not isinstance(p["gate"], str) or minutes is None):
            raise ValueError("EVENT_PARAMS gate must be a string expression over session columns (intraday only)")
    return p


def dc_mult(cfg: RunConfig) -> float:
    """The DC threshold multiple: EVENT_PARAMS dc_mult under the dc sampler, else DEFAULT_DC_MULT."""
    return float(event_params(cfg)["dc_mult"]) if cfg.EVENT_SAMPLER == "dc" else DEFAULT_DC_MULT


# ── Directional change ───────────────────────────────────────────────────────


@numba.njit(cache=True)
def _dc_kernel(x: np.ndarray, d: np.ndarray):
    """
    Directional-change recursion on log prices x with per-bar threshold d. Returns (event positions, overshoot known
    at each bar). Before the first confirmation the run direction is unknown: both the running max and min are
    tracked and the first δ move from either confirms a run. An overshoot (extreme − confirmation price, in units of
    the δ at that confirmation; signed by the run) becomes known at the bar confirming the next reversal.
    Bars with a NaN price or threshold are skipped.
    """
    n = len(x)
    ev = np.empty(n, np.int64)
    known = np.full(n, np.nan)
    k = 0
    mode = 0  # 0 = undetermined, +1 = up run (tracking the max), −1 = down run (tracking the min)
    started = False
    hi = lo = ext = conf_px = conf_d = 0.0
    have_conf = False
    last = np.nan
    for t in range(n):
        if np.isnan(x[t]) or np.isnan(d[t]):
            known[t] = last
            continue
        if not started:
            hi = lo = x[t]
            started = True
        elif mode == 0:
            hi = max(hi, x[t])
            lo = min(lo, x[t])
            if x[t] <= hi - d[t]:
                mode, ext = -1, x[t]
                ev[k] = t
                k += 1
                conf_px, conf_d, have_conf = x[t], d[t], True
            elif x[t] >= lo + d[t]:
                mode, ext = 1, x[t]
                ev[k] = t
                k += 1
                conf_px, conf_d, have_conf = x[t], d[t], True
        elif mode == 1:
            if x[t] > ext:
                ext = x[t]
            elif x[t] <= ext - d[t]:
                if have_conf:
                    last = (ext - conf_px) / conf_d
                mode, ext = -1, x[t]
                ev[k] = t
                k += 1
                conf_px, conf_d = x[t], d[t]
        else:
            if x[t] < ext:
                ext = x[t]
            elif x[t] >= ext + d[t]:
                if have_conf:
                    last = (ext - conf_px) / conf_d
                mode, ext = 1, x[t]
                ev[k] = t
                k += 1
                conf_px, conf_d = x[t], d[t]
        known[t] = last
    return ev[:k], known


def dc_events(close: pd.Series, delta: pd.Series) -> tuple[pd.DatetimeIndex, pd.Series]:
    """
    Directional-change confirmation bars and the lagged overshoot (SPEC §13).

    Args:
        close (pd.Series): Close prices.
        delta (pd.Series): Threshold δ_t in log-return units, aligned to close.

    Returns:
        (event timestamps, overshoot of the last completed run known at each bar, in δ units, NaN before it exists)
    """
    x = np.log(close.astype(float)).to_numpy()
    d = delta.reindex(close.index).to_numpy(dtype=float)
    pos, known = _dc_kernel(x, d)
    return close.index[pos], pd.Series(known, index=close.index, name="dc_overshoot")


# ── Schedule ─────────────────────────────────────────────────────────────────

_MACRO = {"fomc": ("FOMC",), "cpi_nfp": ("CPI", "NFP"), "tom": ("TOM",), "non_macro": ("FOMC", "CPI", "NFP")}


def _decision_utc(index: pd.DatetimeIndex, bar_minutes: int | None) -> np.ndarray:
    """UTC ns of each bar's close (the decision time of an event at that bar); 1Day bars close at 16:00 NY."""
    local = index.tz_convert("America/New_York") if index.tz is not None else index.tz_localize("America/New_York")
    if bar_minutes is None:
        end = local.normalize() + pd.Timedelta(hours=16)
    else:
        end = local + pd.Timedelta(minutes=bar_minutes)
    return end.tz_convert("UTC").as_unit("ns").asi8


def _known_days(kinds, decided: np.ndarray, entry_day: np.ndarray, sessions: np.ndarray, symbol: str | None):
    """
    For each event: is its entry session a `kinds` day (calendar kinds, or "earnings" for `symbol`) whose event is
    known (available_at) by the decision time? An after-close (amc) earnings release reacts in the next session of
    `sessions` (the data's session dates); a calendar date that is not a session (a Sunday FOMC) matches nothing.
    """
    from data.events import event_series

    if kinds == "earnings":
        if symbol is None:
            raise ValueError("schedule days='earnings' needs the cell's symbol")
        ea = event_series("earnings", symbol)
        frames = [(ea, ea["timing"].to_numpy() > 0)]
    else:
        frames = [(event_series("calendar", k), None) for k in kinds]
    q = entry_day.astype("M8[D]").astype(np.int64)
    sess = np.unique(sessions.astype("M8[D]").astype(np.int64))
    hit = np.zeros(len(q), bool)
    for ev, amc in frames:
        days = ev.index.to_numpy().astype("M8[D]").astype(np.int64)
        if amc is not None:
            pos = np.searchsorted(sess, days, side="right")
            nxt = sess[np.minimum(pos, len(sess) - 1)]
            days = np.where(amc & (pos < len(sess)), nxt, days)
        avail = ev["available_at"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
        for dd, av in zip(days, avail):  # a handful of rows per year
            m = q == dd
            hit[m] |= av <= decided[m]
    return hit


def _check_calendar_years(entry_day: np.ndarray) -> None:
    from data.events import read_events

    years = pd.to_datetime(read_events()["date"]).dt.year
    lo, hi = int(years.min()), int(years.max())
    ys = entry_day.astype("M8[Y]").astype(int) + 1970
    if len(ys) and (ys.min() < lo or ys.max() > hi):
        raise ValueError(f"schedule days predicate: bars span {ys.min()}–{ys.max()}, the event table {lo}–{hi}")


def schedule_events(
    df: pd.DataFrame, cfg: RunConfig, *, profile: VolProfile | None = None, symbol: str | None = None
) -> pd.DatetimeIndex:
    """
    Scheduled event bars (SPEC §13): for each entry time T the entry bar is the bar stamped T in its session and the
    event bar is the bar before it in the data (T = 09:30: the previous session's last bar; T later: a bar of the same
    session, else no event that day). On an early close (the session's bars end before 16:00) a T at or past the
    close maps to the session's last bar. `days` keeps sessions whose calendar event is known (available_at) by the
    decision time; `gate` is evaluated on the session frame at the event bar (a NaN comparison is False).
    """
    from data.timeframes import get_timeframe

    p = event_params(cfg)
    minutes = get_timeframe(cfg.TIMEFRAME).minutes
    n = len(df)
    day = ny_dates(df.index)
    if minutes is None:
        entry = np.arange(1, n)
    else:
        mins = ny_minutes(df.index)
        first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
        last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
        sess = np.cumsum(first) - 1
        last_pos = np.flatnonzero(last)  # session k's last bar
        early = (mins[last_pos] + minutes) < CLOSE_MINUTE  # the session's bars stop before 16:00
        parts = []
        for t in p["entry_times"]:
            T = parse_time(t)
            at = np.flatnonzero(mins == T)
            if T > OPEN_MINUTE:
                has = np.zeros(len(last_pos), bool)
                has[sess[at]] = True
                mapped = last_pos[early & ~has & (mins[last_pos] + minutes <= T)]
                at = np.union1d(at, mapped)
                at = at[(at >= 1) & (sess[np.maximum(at - 1, 0)] == sess[at])]
            else:
                at = at[(at >= 1) & first[at]]
            parts.append(at)
        entry = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    t = entry - 1
    keep = np.ones(len(t), bool)
    if p["days"] != "all":
        _check_calendar_years(day[entry])
        kinds = "earnings" if p["days"] == "earnings" else _MACRO[p["days"]]
        hit = _known_days(kinds, _decision_utc(df.index[t], minutes), day[entry], day, symbol)
        keep &= ~hit if p["days"] == "non_macro" else hit
    if p["gate"] is not None and len(t):
        from features.session import session_frame

        frame = session_frame(df, cfg, profile).iloc[t]
        g = frame.eval(p["gate"])
        if not (isinstance(g, pd.Series) and g.dtype == bool):
            raise ValueError(f"schedule gate {p['gate']!r} is not a boolean expression over the session columns")
        keep &= g.to_numpy()
    return df.index[t[keep]]


# ── Dispatcher ───────────────────────────────────────────────────────────────


def sample_events(
    df: pd.DataFrame, cfg: RunConfig, *, profile: VolProfile | None = None, symbol: str | None = None
) -> pd.DataFrame:
    """
    Tradeable events (cfg.EVENT_SAMPLER) and their barrier widths, using only data up to each event.

    σ per bar = bar_volatility (with `profile`, the fold's VolProfile, under VOL_PROFILE="tod"); barriers are
    ±BARRIER_MULT · σ_t · √VERTICAL_BARS, or with a profile ±BARRIER_MULT · σ_base_t · √Σ s(b)² over the held slots
    (vol_profile.hold_scale). Unless HOLD_OVERNIGHT, cusum / dc events on a session's last bar are skipped.

    Args:
        df (pd.DataFrame): OHLC data with a DatetimeIndex.
        cfg (RunConfig): Run configuration.
        profile (VolProfile | None): The fold's time-of-day profile (None = plain σ).
        symbol (str | None): The cell's symbol (schedule days="earnings" only).

    Returns:
        pd.DataFrame: Indexed by event time with column `width`.
    """
    vol = bar_volatility(df["close"], cfg.VOL_SPAN, profile)
    if cfg.EVENT_SAMPLER == "schedule":
        events = schedule_events(df, cfg, profile=profile, symbol=symbol)
    else:
        if cfg.EVENT_SAMPLER == "cusum":
            events = cusum_events(df["close"], cfg.CUSUM_MULT * vol)
        else:
            events = dc_events(df["close"], dc_mult(cfg) * vol)[0]
        if not cfg.HOLD_OVERNIGHT:
            session_end = pd.Series(df.index.normalize(), index=df.index).shift(-1) != df.index.normalize()
            events = events[~session_end.loc[events].to_numpy()]
    if profile is None:
        return barrier_width(vol, cfg.BARRIER_MULT, cfg.VERTICAL_BARS).loc[events].to_frame()
    # SPEC §14 as built: under a profile the barrier is sized by σ over the slots the position holds (σ_base_t ·
    # √Σ_held s(b)²), not by the event slot's σ_t · √VERTICAL_BARS: the 09:30 slot carries the overnight gap
    # (s ≈ 9 on SPY), so σ_t there is far above the move of the hour that follows it
    base = base_volatility(df["close"], cfg.VOL_SPAN, profile).loc[events].to_numpy()
    held = hold_scale(events, profile, cfg.VERTICAL_BARS, cfg.HOLD_OVERNIGHT)
    return pd.DataFrame({"width": cfg.BARRIER_MULT * base * held}, index=events)
