"""
Exit models (SPEC §13, U14): how a position opened at open[t+1] for an event at bar t is closed.

Every model returns the barrier_exits frame (t1, entry_pos, exit_pos, entry_px, exit_px, ret, label, barrier, width),
so labels, meta-labels, the backtests and average_uniqueness are unchanged downstream. `barrier` names the fill:
- "upper" / "lower" / "tie": triple-barrier touches (§3);
- "vertical": a scheduled fill at the close of the exit bar (vertical barrier, `time` exit_time "close" (MOC) or
  hold_bars, hysteresis max hold / session close);
- "time" / "hysteresis": a fill at the OPEN of the exit bar (`time` exit_time "HH:MM" / "open" (MOO); a hysteresis
  exit decided at the previous bar's close).
OPEN_FILL lists the open-fill names: the backtests book those exits at the bar's open (phase 0, cost kind "open").

Models (cfg.EXIT_MODEL, cfg.EXIT_PARAMS):
- `triple_barrier`: barrier_exits (§3), unchanged.
- `time`: exactly one of `exit_time` or `hold_bars` (close of the k-th held bar, cut at the session close unless
  HOLD_OVERNIGHT). `exit_time`, in the session `exit_session` sessions after the entry session (default 0, "open" 1):
  "close" = that session's last bar close, modelled MOC; "open" = its first bar open, MOO; "HH:MM" = the open of its
  first bar stamped at or after it, its MOC close when the session ends first; "next" (SPEC §16, U16: hold to the next
  rebalance) = the first bar open (MOO) of the first session of the next schedule `every` period (EVENT_PARAMS: the
  next session, week or month in the data). An auction fill must be an auction bar (risk.costs.auction_flags): a
  session whose last bar (MOC) or first bar (MOO) is missing gives no fill and the event is dropped. An exit not
  after the entry drops the event.
- Entries fill at the OPEN of the entry bar, or at its CLOSE under a market-on-close schedule (events.entry_at_close;
  `time` exits only, strictly after that close).
- `hysteresis` (Toulson & Toulson's STTS exit): on the primary's bar-level score (exit_signal), a long exits when the
  score is <= −beta at a bar's close (a short: >= +beta) and fills at the next bar's open, so the decision bar
  precedes the fill; else at the close of the `max_bars`-th held bar or the session close (as the vertical barrier).
  Without a side (labels) the side is the sign of the score at the event (>= 0 long, as rule_frame).
Events whose exit would fall past the end of the data are dropped.
"""

import numpy as np
import pandas as pd

from features.events import CLOSE_MINUTE, parse_time
from features.triple_barrier_labels import barrier_exits
from features.vol_profile import OPEN_MINUTE, ny_dates, ny_minutes
from utils.config import RunConfig

EXIT_MODELS = ("triple_barrier", "time", "hysteresis")
OPEN_FILL = ("time", "hysteresis")
_DEFAULTS = {"triple_barrier": {}, "time": {"exit_time": None, "hold_bars": None, "exit_session": None},
             "hysteresis": {"beta": None, "max_bars": None}}  # fmt: skip


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def exit_params(cfg: RunConfig) -> dict:
    """cfg.EXIT_PARAMS merged over the model's defaults, validated (raises ValueError)."""
    model = cfg.EXIT_MODEL
    if model not in EXIT_MODELS:
        raise ValueError(f"EXIT_MODEL must be one of {EXIT_MODELS}, got {model!r}")
    unknown = set(cfg.EXIT_PARAMS) - set(_DEFAULTS[model])
    if unknown:
        raise ValueError(
            f"EXIT_PARAMS for {model!r}: unknown keys {sorted(unknown)}; allowed {sorted(_DEFAULTS[model])}"
        )
    p = {**_DEFAULTS[model], **cfg.EXIT_PARAMS}
    if model == "time":
        from data.timeframes import get_timeframe

        minutes = get_timeframe(cfg.TIMEFRAME).minutes
        et, hb, k = p["exit_time"], p["hold_bars"], p["exit_session"]
        if (et is None) == (hb is None):
            raise ValueError("EXIT_PARAMS for 'time' must set exactly one of exit_time and hold_bars")
        if hb is not None and not _pos_int(hb):
            raise ValueError(f"EXIT_PARAMS hold_bars must be an integer >= 1, got {hb!r}")
        if k is not None:
            if et is None or et == "next":
                raise ValueError("EXIT_PARAMS exit_session needs exit_time 'close', 'open' or 'HH:MM'")
            lo = 1 if et == "open" else 0
            if not (isinstance(k, int) and not isinstance(k, bool) and k >= lo):
                raise ValueError(f"EXIT_PARAMS exit_session must be an integer >= {lo}, got {k!r}")
        if et == "next" and cfg.EVENT_SAMPLER != "schedule":
            raise ValueError("EXIT_PARAMS exit_time 'next' (the next rebalance) needs EVENT_SAMPLER='schedule'")
        if et is not None and et not in ("close", "open", "next"):
            if not isinstance(et, str) or minutes is None:
                raise ValueError(f"EXIT_PARAMS exit_time must be 'close', 'open' or (intraday) 'HH:MM', got {et!r}")
            m = parse_time(et)
            if not (OPEN_MINUTE < m < CLOSE_MINUTE and (m - OPEN_MINUTE) % minutes == 0):
                raise ValueError(f"exit_time {et!r} is not a {cfg.TIMEFRAME} bar open after 09:30 and before 16:00")
    if model == "hysteresis":
        if not (isinstance(p["beta"], (int, float)) and not isinstance(p["beta"], bool) and p["beta"] >= 0):
            raise ValueError(f"EXIT_PARAMS beta must be a number >= 0, got {p['beta']!r}")
        if not _pos_int(p["max_bars"]):
            raise ValueError(f"EXIT_PARAMS max_bars must be an integer >= 1, got {p['max_bars']!r}")
    return p


def _frame(df, t, e, x, exit_px, barrier, w, entry_close: bool = False) -> pd.DataFrame:
    entry = df["close" if entry_close else "open"].to_numpy()[e]
    ret = exit_px / entry - 1.0
    return pd.DataFrame(
        {
            "t1": df.index[x],
            "entry_pos": e,
            "exit_pos": x,
            "entry_px": entry,
            "exit_px": exit_px,
            "ret": ret,
            "label": np.sign(ret).astype(int),
            "barrier": barrier,
            "width": w,
        },
        index=df.index[t],
    )


def _sessions(df: pd.DataFrame):
    """Per bar: session number, and per session: first / last bar position."""
    day = ny_dates(df.index)
    n = len(df)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    return np.cumsum(first) - 1, np.flatnonzero(first), np.flatnonzero(last)


def _entries(df: pd.DataFrame, events: pd.DatetimeIndex, width: pd.Series):
    n = len(df)
    t = df.index.get_indexer(events)
    w = width.reindex(events).to_numpy(dtype=float)
    keep = (t >= 0) & (t + 1 < n) & ~np.isnan(w)
    return t[keep], t[keep] + 1, w[keep], keep


def time_exits(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    width: pd.Series,
    *,
    exit_time: str | None,
    hold_bars: int | None,
    hold_overnight: bool,
    bar_minutes: int | None,
    exit_session: int | None = None,
    every: str = "session",
    entry_close: bool = False,
) -> pd.DataFrame:
    """Scheduled exits (see the module docstring); `bar_minutes` None = 1Day bars; `every` = the schedule's period
    (exit_time "next"); `entry_close`: entries fill at the entry bar's close (MOC)."""
    from features.events import period_starts
    from risk.costs import auction_flags

    n = len(df)
    close, open_ = df["close"].to_numpy(), df["open"].to_numpy()
    t, e, w, _ = _entries(df, events, width)
    sess, s_first, s_last = _sessions(df)
    is_open, is_close = auction_flags(df.index, bar_minutes)
    n_sess = len(s_first)
    if entry_close and (hold_bars is not None or (exit_time not in ("open", "next") and not exit_session)):
        raise ValueError("a market-on-close entry exits in a later session: set exit_session >= 1 (or 'open' / 'next')")
    if hold_bars is not None:
        if hold_overnight:
            last = e + hold_bars - 1
            ok = last <= n - 1
        else:
            last = np.minimum(e + hold_bars - 1, s_last[sess[e]])
            ok = (last < n - 1) | (e + hold_bars - 1 <= n - 1)
        x = np.minimum(last, n - 1)
        return _frame(df, t[ok], e[ok], x[ok], close[x[ok]], "vertical", w[ok])
    if exit_time == "next":
        starts = np.flatnonzero(period_starts(ny_dates(df.index)[s_first], every))
        j = np.searchsorted(starts, sess[e], side="right")
        ok = j < len(starts)
        x = s_first[starts[np.minimum(j, len(starts) - 1)]] if len(starts) else np.zeros(len(e), np.int64)
        ok &= is_open[x]  # MOO at the next period's first session
        return _frame(df, t[ok], e[ok], x[ok], open_[x[ok]], "time", w[ok], entry_close)
    k = (1 if exit_time == "open" else 0) if exit_session is None else exit_session
    tgt = sess[e] + k
    ok = tgt < n_sess
    tgt = np.minimum(tgt, n_sess - 1)
    if exit_time == "open":
        x = s_first[tgt]
        ok &= is_open[x]  # MOO: the session's first bar must be its opening bar
        return _frame(df, t[ok], e[ok], x[ok], open_[x[ok]], "time", w[ok], entry_close)
    x_close = s_last[tgt]
    if exit_time == "close":
        ok &= is_close[x_close]
        return _frame(df, t[ok], e[ok], x_close[ok], close[x_close[ok]], "vertical", w[ok], entry_close)
    T = parse_time(exit_time)
    mins = ny_minutes(df.index)
    at = np.where((mins >= T), np.arange(n), n)  # first bar of the session stamped >= T
    first_at = pd.Series(at).groupby(sess).transform("min").to_numpy()
    xo = first_at[s_first[tgt]]
    later = (k > 0) | (mins[e] < T)  # the exit session's T is after the entry bar
    by_open = ok & later & (xo < n)
    by_close = ok & later & (xo >= n) & is_close[x_close]
    x = np.where(by_open, np.minimum(xo, n - 1), x_close)
    px = np.where(by_open, open_[x], close[x])
    ok = by_open | by_close
    barrier = np.where(by_open, "time", "vertical")[ok]
    return _frame(df, t[ok], e[ok], x[ok], px[ok], barrier, w[ok], entry_close)


def hysteresis_exits(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    width: pd.Series,
    signal: pd.Series,
    side: pd.Series | None = None,
    *,
    beta: float,
    max_bars: int,
    hold_overnight: bool,
) -> pd.DataFrame:
    """Hysteresis exits on a bar-level signal (see the module docstring)."""
    n = len(df)
    close, open_ = df["close"].to_numpy(), df["open"].to_numpy()
    sig = signal.reindex(df.index).to_numpy(dtype=float)
    t, e, w, keep = _entries(df, events, width)
    if side is None:
        sd = np.where(sig[t] >= 0, 1.0, -1.0)
        sd[np.isnan(sig[t])] = np.nan
    else:
        sd = side.reindex(events).to_numpy(dtype=float)[keep]
    ok = ~np.isnan(sd) & (sd != 0)
    t, e, w, sd = t[ok], e[ok], w[ok], sd[ok]
    sess, _, s_last = _sessions(df)
    session_last = np.full(n, n - 1) if hold_overnight else s_last[sess]
    last = np.minimum(e + max_bars - 1, session_last[e])
    ok = (session_last[t] == session_last[e]) & ((last < n - 1) | (e + max_bars - 1 <= n - 1))
    t, e, w, sd, last = t[ok], e[ok], w[ok], sd[ok], last[ok]

    # (event, k) → bar e + k; a trigger at bar k < last fills at open[k + 1] (a trigger on the last bar is moot)
    idx = np.minimum(e[:, None] + np.arange(max_bars), n - 1)
    live = idx < last[:, None]
    s = sig[idx]
    trig = live & np.where(sd[:, None] > 0, s <= -beta, s >= beta)  # NaN compares False
    hit = trig.any(axis=1)
    k = np.where(hit, trig.argmax(axis=1), 0)
    x = np.where(hit, e + k + 1, last)
    px = np.where(hit, open_[np.minimum(x, n - 1)], close[np.minimum(x, n - 1)])
    return _frame(df, t, e, x, px, np.where(hit, "hysteresis", "vertical"), w)


def exit_signal(df: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """The bar-level score of cfg.PRIMARY that the hysteresis exit reads (rule primaries only)."""
    from primaries import make_primary

    p = make_primary(cfg)
    if not hasattr(p, "score"):
        raise ValueError(f"EXIT_MODEL='hysteresis' needs a rule primary with a bar-level score, not {cfg.PRIMARY!r}")
    return pd.Series(np.asarray(p.score(df, cfg), dtype=float), index=df.index)


def exit_frame(
    df: pd.DataFrame, events: pd.DatetimeIndex, width: pd.Series, cfg: RunConfig, side: pd.Series | None = None
) -> pd.DataFrame:
    """The exit of each event under cfg.EXIT_MODEL (the barrier_exits frame; `side` as there)."""
    if cfg.EXIT_MODEL == "triple_barrier":
        return barrier_exits(
            df, events, width, side, vertical_bars=cfg.VERTICAL_BARS, hold_overnight=cfg.HOLD_OVERNIGHT
        )
    p = exit_params(cfg)
    if cfg.EXIT_MODEL == "time":
        from data.timeframes import get_timeframe
        from features.events import entry_at_close, event_params

        return time_exits(
            df,
            events,
            width,
            exit_time=p["exit_time"],
            hold_bars=p["hold_bars"],
            hold_overnight=cfg.HOLD_OVERNIGHT,
            bar_minutes=get_timeframe(cfg.TIMEFRAME).minutes,
            exit_session=p["exit_session"],
            every=event_params(cfg)["every"] if cfg.EVENT_SAMPLER == "schedule" else "session",
            entry_close=entry_at_close(cfg),
        )
    return hysteresis_exits(
        df,
        events,
        width,
        exit_signal(df, cfg),
        side,
        beta=float(p["beta"]),
        max_bars=p["max_bars"],
        hold_overnight=cfg.HOLD_OVERNIGHT,
    )


def exit_phase(barrier: np.ndarray, exit_px: np.ndarray, open_at_exit: np.ndarray) -> np.ndarray:
    """Backtest fill phase of each exit: 0 = the bar's open (open fills, gaps through a barrier), 1 = intrabar barrier
    touch, 2 = the bar's close (vertical / scheduled close fills)."""
    barrier = np.asarray(barrier)
    vertical = barrier == "vertical"
    at_open = np.isin(barrier, OPEN_FILL) | (exit_px == open_at_exit)
    return np.where(vertical, 2, np.where(at_open, 0, 1))
