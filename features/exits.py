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
- `time`: exactly one of `exit_time` ("close" = the entry session's last bar close, modelled MOC; "open" = the next
  session's first bar open, MOO; "HH:MM" = the open of the first bar of the entry session stamped at or after it, the
  session's MOC close when the session ends first) or `hold_bars` (close of the k-th held bar, cut at the session
  close unless HOLD_OVERNIGHT). An auction fill must be an auction bar (risk.costs.auction_flags): a session whose
  last bar (MOC) or next session whose first bar (MOO) is missing gives no fill and the event is dropped. An "HH:MM"
  not after the entry bar drops the event.
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
_DEFAULTS = {"triple_barrier": {}, "time": {"exit_time": None, "hold_bars": None},
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
        et, hb = p["exit_time"], p["hold_bars"]
        if (et is None) == (hb is None):
            raise ValueError("EXIT_PARAMS for 'time' must set exactly one of exit_time and hold_bars")
        if hb is not None and not _pos_int(hb):
            raise ValueError(f"EXIT_PARAMS hold_bars must be an integer >= 1, got {hb!r}")
        if et is not None and et not in ("close", "open"):
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


def _frame(df, t, e, x, exit_px, barrier, w) -> pd.DataFrame:
    entry = df["open"].to_numpy()[e]
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
) -> pd.DataFrame:
    """Scheduled exits (see the module docstring); `bar_minutes` None = 1Day bars."""
    from risk.costs import auction_flags

    n = len(df)
    close, open_ = df["close"].to_numpy(), df["open"].to_numpy()
    t, e, w, _ = _entries(df, events, width)
    sess, s_first, s_last = _sessions(df)
    is_open, is_close = auction_flags(df.index, bar_minutes)
    if hold_bars is not None:
        if hold_overnight:
            last = e + hold_bars - 1
            ok = last <= n - 1
        else:
            last = np.minimum(e + hold_bars - 1, s_last[sess[e]])
            ok = (last < n - 1) | (e + hold_bars - 1 <= n - 1)
        x = np.minimum(last, n - 1)
        return _frame(df, t[ok], e[ok], x[ok], close[x[ok]], "vertical", w[ok])
    if exit_time == "open":
        nxt = sess[e] + 1
        ok = nxt < len(s_first)
        x = s_first[np.minimum(nxt, len(s_first) - 1)]
        ok &= is_open[x]  # MOO: the next session's first bar must be its opening bar
        return _frame(df, t[ok], e[ok], x[ok], open_[x[ok]], "time", w[ok])
    x_close = s_last[sess[e]]
    if exit_time == "close":
        ok = is_close[x_close]
        return _frame(df, t[ok], e[ok], x_close[ok], close[x_close[ok]], "vertical", w[ok])
    T = parse_time(exit_time)
    mins = ny_minutes(df.index)
    at = np.where((mins >= T), np.arange(n), n)  # first bar of the session stamped >= T
    first_at = pd.Series(at).groupby(sess).transform("min").to_numpy()
    xo = first_at[e]
    by_open = (mins[e] < T) & (xo < n)
    by_close = (mins[e] < T) & (xo >= n) & is_close[x_close]
    x = np.where(by_open, np.minimum(xo, n - 1), x_close)
    px = np.where(by_open, open_[x], close[x])
    ok = by_open | by_close
    barrier = np.where(by_open, "time", "vertical")[ok]
    return _frame(df, t[ok], e[ok], x[ok], px[ok], barrier, w[ok])


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

        return time_exits(
            df,
            events,
            width,
            exit_time=p["exit_time"],
            hold_bars=p["hold_bars"],
            hold_overnight=cfg.HOLD_OVERNIGHT,
            bar_minutes=get_timeframe(cfg.TIMEFRAME).minutes,
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
