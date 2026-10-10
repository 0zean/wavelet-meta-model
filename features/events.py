"""
Event samplers (SPEC §13, U14): when a primary may trade.

`sample_events(df, cfg)` returns the event frame (index = event bar t, column `width`). Every sampler keeps the §3
execution model: the decision is taken at the close of the event bar t and the fill is the open of the next bar in the
data, entry_pos = t + 1 (the schedule sampler picks t as the bar before its entry bar, so this holds by construction).

- `cusum` (§3): symmetric CUSUM on log returns, threshold CUSUM_MULT · σ_t.
- `dc`: directional change (Guillaume et al. 1997): an event at the bar that CONFIRMS a reversal, i.e. log price is
  δ_t = dc_mult · σ_t below the running maximum of an up run (or above the minimum of a down run). The run's
  overshoot (confirmation → extreme) is only known at the next confirmation (`dc_events`, the session group).
- `schedule`: fixed session times (`entry_times`), a calendar predicate (`days`, tested `day_offset` sessions after the
  entry session), a period (`every`: every session, or only the first session of each week / month in the data) and
  an optional `gate` over the session frame (features.session) at the event bar. `entry_times: ["close"]` (SPEC §16,
  U16) is a market-on-close entry: the entry bar is the session's last bar and the fill is its CLOSE (the closing
  auction), decided at the close of the bar before it (`entry_at_close`).

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
DAY_PREDICATES = ("all", "fomc", "cpi_nfp", "macro", "tom", "month_end", "opex", "non_macro", "earnings")
PERIODS = ("session", "week", "month")
DEFAULT_DC_MULT = 2.0
CLOSE_MINUTE = 16 * 60
EARLY_CLOSE_MINUTE = 13 * 60
AMC_NEXT_MAX_DAYS = 4
_DEFAULTS = {"cusum": {}, "dc": {"dc_mult": DEFAULT_DC_MULT}, "schedule": {"entry_times": None, "days": "all",
             "gate": None, "every": "session", "day_offset": 0, "windows": None}}  # fmt: skip
WINDOW_KEYS = ("days", "day_offset", "hold")
WINDOW_DAYS = ("fomc", "cpi_nfp", "macro", "tom", "month_end", "opex")  # calendar predicates a window may use


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
        if "close" in times and len(times) > 1:
            raise ValueError(
                f"EVENT_PARAMS entry_times: 'close' (a market-on-close entry) cannot be mixed, got {times!r}"
            )
        for t in times:
            if t == "close":
                continue
            m = parse_time(t)
            if minutes is None:
                if m != OPEN_MINUTE:
                    raise ValueError(f"1Day bars trade at the open or the close; entry time {t!r} is not 'open' / "
                                     "'09:30' / 'close'")  # fmt: skip
            elif not (OPEN_MINUTE <= m < CLOSE_MINUTE and (m - OPEN_MINUTE) % minutes == 0):
                raise ValueError(f"entry time {t!r} is not a {cfg.TIMEFRAME} bar open between 09:30 and 16:00")
        if p["days"] not in DAY_PREDICATES:
            raise ValueError(f"EVENT_PARAMS days must be one of {DAY_PREDICATES}, got {p['days']!r}")
        if p["every"] not in PERIODS:
            raise ValueError(f"EVENT_PARAMS every must be one of {PERIODS}, got {p['every']!r}")
        off = p["day_offset"]
        if not (isinstance(off, int) and not isinstance(off, bool) and off >= 0):
            raise ValueError(f"EVENT_PARAMS day_offset must be an integer >= 0, got {off!r}")
        if p["gate"] is not None and (not isinstance(p["gate"], str) or minutes is None):
            raise ValueError("EVENT_PARAMS gate must be a string expression over session columns (intraday only)")
        if p["windows"] is not None:
            _check_windows(p)
    return p


def _check_windows(p: dict) -> None:
    """`windows` (U18): a daily MOC schedule held over the union of calendar windows (see schedule_events)."""
    ws = p["windows"]
    if p["entry_times"] != ["close"] or p["every"] != "session" or p["days"] != "all" or p["day_offset"] != 0:
        raise ValueError("EVENT_PARAMS windows needs entry_times ['close'], every 'session', days 'all', day_offset 0")
    if not isinstance(ws, list) or len(ws) < 2:
        raise ValueError(f"EVENT_PARAMS windows must be a list of at least two windows, got {ws!r}")
    seen = set()
    for w in ws:
        if not isinstance(w, dict) or set(w) != set(WINDOW_KEYS):
            raise ValueError(f"a window is {{days, day_offset, hold}}, got {w!r}")
        if w["days"] not in WINDOW_DAYS:
            raise ValueError(f"window days must be one of {WINDOW_DAYS}, got {w['days']!r}")
        for k, lo in (("day_offset", 0), ("hold", 1)):
            v = w[k]
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= lo):
                raise ValueError(f"window {k} must be an integer >= {lo}, got {v!r}")
        key = (w["days"], w["day_offset"], w["hold"])
        if key in seen:
            raise ValueError(f"duplicate window {w!r}")
        seen.add(key)


def entry_at_close(cfg: RunConfig) -> bool:
    """True when the run's entries fill at the CLOSE of the entry bar (schedule entry_times ["close"], MOC)."""
    return cfg.EVENT_SAMPLER == "schedule" and event_params(cfg)["entry_times"] == ["close"]


def period_starts(day: np.ndarray, every: str) -> np.ndarray:
    """Per session date (sorted, unique): is it the first session of its `every` period (session / ISO week / month)?"""
    d = np.asarray(day).astype("M8[D]")
    if every == "session":
        return np.ones(len(d), bool)
    if every == "week":
        key = (d.astype(np.int64) + 3) // 7  # 1970-01-01 was a Thursday: (days + 3) // 7 changes on Mondays
    else:
        key = d.astype("M8[M]").astype(np.int64)
    return np.r_[True, key[1:] != key[:-1]] if len(d) else np.zeros(0, bool)


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

_MACRO = {"fomc": ("FOMC",), "cpi_nfp": ("CPI", "NFP"), "macro": ("FOMC", "CPI", "NFP"), "tom": ("TOM",),
          "month_end": ("MONTH_END",), "opex": ("OPEX",), "non_macro": ("FOMC", "CPI", "NFP")}  # fmt: skip
_NO_DAY = np.iinfo(np.int64).min + 1  # a target session past the data: matches no calendar day


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
            # amc: the next session of the data, if it is the next exchange session (within AMC_NEXT_MAX_DAYS: a
            # weekend plus a holiday). Otherwise (the data ends there, or starts after it) it matches nothing.
            ok = (pos < len(sess)) & (nxt - days <= AMC_NEXT_MAX_DAYS)
            days = np.where(amc, np.where(ok, nxt, np.iinfo(np.int64).min), days)
        avail = ev["available_at"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
        for dd, av in zip(days, avail):  # a handful of rows per year
            m = q == dd
            hit[m] |= av <= decided[m]
    return hit


def _check_calendar_years(entry_day: np.ndarray) -> None:
    from data.events import read_events

    years = pd.to_datetime(read_events()["date"]).dt.year
    lo, hi = int(years.min()), int(years.max())
    entry_day = entry_day[entry_day.astype("M8[D]").astype(np.int64) != _NO_DAY]
    ys = entry_day.astype("M8[Y]").astype(int) + 1970
    if len(ys) and (ys.min() < lo or ys.max() > hi):
        raise ValueError(f"schedule days predicate: bars span {ys.min()}–{ys.max()}, the event table {lo}–{hi}")


def schedule_events(
    df: pd.DataFrame, cfg: RunConfig, *, profile: VolProfile | None = None, symbol: str | None = None
) -> pd.DatetimeIndex:
    """
    Scheduled event bars (SPEC §13): for each entry time T the entry bar is the bar stamped T in its session and the
    event bar is the bar before it in the data (T = 09:30: the previous session's last bar; T later: a bar of the same
    session, else no event that day). On an early close (the session's last bar spans 13:00) a T at or past the
    close maps to the session's last bar. T = "close" (MOC): the entry bar is the session's last bar, filled at its
    close, when it is the closing-auction bar (risk.costs.auction_flags), the event bar the bar before it in the same
    session (1Day: the previous session). `every` keeps entry
    sessions that are the first of their week / month in the data. `days` keeps sessions whose session `day_offset`
    sessions later (in the data) is a calendar day known (available_at) by the decision time; `gate` is evaluated on
    the session frame at the event bar (a NaN comparison is False). `windows` (U18; MOC entries every session): a list
    of {days, day_offset, hold}; the MOC entry of session x is kept iff some window's entry session e (selected as
    `days` / `day_offset` would select it, known at e's decision) has e <= x < e + hold, so a time exit at the next
    session's close holds the union of the windows (consecutive entries roll into one position).
    """
    from data.timeframes import get_timeframe

    p = event_params(cfg)
    minutes = get_timeframe(cfg.TIMEFRAME).minutes
    n = len(df)
    day = ny_dates(df.index)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    sess = np.cumsum(first) - 1
    first_pos, last_pos = np.flatnonzero(first), np.flatnonzero(last)  # session k's first / last bar
    if minutes is None:
        entry = np.arange(1, n)
    elif p["entry_times"] == ["close"]:
        from risk.costs import auction_flags

        # MOC: the session's last bar must be its closing-auction bar (a session missing it gives no event)
        is_close = auction_flags(df.index, minutes)[1]
        ok = (last_pos >= 1) & (sess[np.maximum(last_pos - 1, 0)] == sess[last_pos]) & is_close[last_pos]
        entry = last_pos[ok]
    else:
        mins = ny_minutes(df.index)
        # an early close: the session's last bar spans 13:00 (risk.costs.auction_flags' rule), so a hole at the end
        # of a regular session is not read as one
        early = (mins[last_pos] < EARLY_CLOSE_MINUTE) & (mins[last_pos] + minutes >= EARLY_CLOSE_MINUTE)
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
    entry = entry[period_starts(day[first_pos], p["every"])[sess[entry]]]
    t = entry - 1
    keep = np.ones(len(t), bool)

    def days_hit(days: str, offset: int) -> np.ndarray:
        target = sess[entry] + offset
        ok = target < len(first_pos)
        tday = np.where(ok, day[first_pos[np.minimum(target, len(first_pos) - 1)]].astype(np.int64), _NO_DAY)
        tday = tday.astype("M8[D]")
        _check_calendar_years(tday)
        kinds = "earnings" if days == "earnings" else _MACRO[days]
        return _known_days(kinds, _decision_utc(df.index[t], minutes), tday, day, symbol)

    if p["days"] != "all":
        hit = days_hit(p["days"], p["day_offset"])
        keep &= ~hit if p["days"] == "non_macro" else hit
    if p["windows"] is not None:
        # a window's entry session e (its predicate tested day_offset sessions later, known at e's decision) holds
        # e + 1 … e + hold; the daily MOC entry at session x (held to x + 1's close) is kept iff x + 1 is held by some
        # window, i.e. x ∈ [e, e + hold). Consecutive kept entries roll (one position, no cost).
        held = np.zeros(len(first_pos) + max(w["hold"] for w in p["windows"]), bool)
        for w in p["windows"]:
            for e in sess[entry][days_hit(w["days"], w["day_offset"])]:
                held[e : e + w["hold"]] = True
        keep &= held[sess[entry]]
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
