"""
Position backtest (SPEC §23, U23): the daily P&L of one symbol's target-position series on a decision schedule, many
series at once, without the event machinery — the specification curve and the cost curve of a region family.

A target series gives, at every scheduled decision bar t_k (features.events.schedule_events; a decision missing from
the series is flat), a fraction of equity in [−1, +1] held from the open of the bar after t_k to the open of the next
decision's entry bar in the same session, else to the session's closing auction: exactly the `next_event` time exit
(SPEC §21) and the portfolio simulator's roll path (risk.portfolio; SIZE_STEP = 0). Accounting is the simulator's,
share for share, on one symbol:
- at a decision's fill (bar b = t_k + 1, reference price px = its open; decisions come after 09:30, so no entry
  fills at an opening auction), with E = the equity at close[b − 1], f = SIZE · |target| and side = sign(target):
  enter (flat before): q = f · E / (px · (1 + side · c)), fill px · (1 + side · c);
  roll (held before, target ≠ 0): q as above, or the old shares when the side and f are unchanged; the old position
  is realized at px without cost and the traded shares |side · q − side_old · q_old| pay c;
  exit (held before, target 0): fill px · (1 − side · c);
- the last position of a session exits at its last bar's close (the closing print under "print") paying that bar's
  `close` cost; a session whose last bar is not the closing-auction bar has no MOC fill, so its last decision is flat
  (the time exit drops that event and the previous position exits at its entry bar);
- c = the bar's one-way cost for the fill kind (risk.costs.fill_costs columns `open` / `close`), SLIPPAGE_PCT without
  a cost frame, or a scalar one-way cost on every fill (the cost curve);
- the daily loss gate of a profile (`daily_loss`): at a close with equity ≤ (1 − daily_loss) × the session's start,
  the position is flattened at the next bar's open (`open` cost) and the session takes no further entry;
- the cash yield: at a session's first bar, the free cash at the previous close (positions are flat overnight, so
  all of max(0, equity)) earns the previous session's rate × calendar days / 360.
The only profile field supported is `daily_loss` (no vol target, caps, drawdown tiers or borrow): those make the
simulator path-dependent across symbols, and a region family registers none of them. Parity with simulate_portfolio
through the rule pass is a test (tests/test_u23.py).

    grid = decision_grid(df, cfg)
    res = position_backtest(df, targets, cfg, costs=..., prints=..., cash_yield=..., profile=...)
    res.ret            # daily returns, sessions × series
"""

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd

from features.vol_profile import ny_dates
from utils.config import RunConfig


@dataclass(frozen=True)
class DecisionGrid:
    """The schedule of one symbol's bars: decision bar positions (time order), the session of every bar, every
    session's first / last bar, and whether the session's last bar is its closing-auction bar."""

    index: pd.DatetimeIndex  # decision bars (timestamps)
    dec: np.ndarray  # decision bar positions
    first: np.ndarray  # per session: first bar position
    last: np.ndarray  # per session: last bar position
    moc: np.ndarray  # per session: the last bar is the closing-auction bar
    days: pd.DatetimeIndex  # per session: NY date (tz-naive)


def decision_grid(df: pd.DataFrame, cfg: RunConfig, sessions=None) -> DecisionGrid:
    """The scheduled decisions of cfg (schedule sampler, `next_event` exit) on df's bars."""
    from data.timeframes import get_timeframe
    from features.events import schedule_events
    from features.exits import exit_params
    from risk.costs import auction_flags

    if cfg.EVENT_SAMPLER != "schedule" or exit_params(cfg).get("exit_time") != "next_event":
        raise ValueError("position_backtest needs the schedule sampler with the 'next_event' time exit (SPEC §21)")
    ev = schedule_events(df, cfg, sessions=sessions)
    dec = df.index.get_indexer(ev)
    day = ny_dates(df.index)
    n = len(df)
    first_b = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last_b = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    is_close = auction_flags(df.index, get_timeframe(cfg.TIMEFRAME).minutes)[1]
    last = np.flatnonzero(last_b)
    return DecisionGrid(ev, dec, np.flatnonzero(first_b), last, is_close[last], pd.DatetimeIndex(day[first_b]))


@numba.njit(cache=True)
def _kernel(first, last, opn, cls, c_open, act, tgt, moc_px, c_moc, rate, gap, daily_loss, init_cash, size,
            eq_end, cost, notional, fills):  # fmt: skip
    """One pass per series (column of tgt) over every bar; see the module docstring. act[b] = the decision whose fill
    is at bar b's open (−1: none). Outputs per (session, series)."""
    n_sess = first.shape[0]
    for col in range(tgt.shape[1]):
        cash = init_cash
        prev_eq = init_cash
        held = False
        side = 0.0
        qty = 0.0
        efill = 0.0
        frac = 0.0
        for s in range(n_sess):
            if s > 0 and rate[s] == rate[s]:  # interest on the free cash over the gap (NaN rate: none)
                free = max(0.0, min(cash, prev_eq))
                interest = free * rate[s] * gap[s] / 360.0
                cash += interest
                prev_eq += interest
            sess_start = prev_eq
            flatten = False
            blocked = False
            c_s, n_s, f_s = 0.0, 0.0, 0
            for b in range(first[s], last[s] + 1):
                e_dec = prev_eq
                if flatten and held:  # the loss gate's flattening at this open
                    cc = c_open[b]
                    px = opn[b]
                    fill = px * (1.0 - side * cc)
                    cash += side * qty * (fill - efill)
                    c_s += qty * px * cc
                    n_s += qty * px
                    f_s += 1
                    held = False
                flatten = False
                k = act[b]
                if k >= 0:
                    t = 0.0 if blocked else tgt[k, col]
                    px = opn[b]
                    cc = c_open[b]
                    if t != 0.0:
                        nside = 1.0 if t > 0.0 else -1.0
                        f = size * abs(t)
                        q = f * e_dec / (px * (1.0 + nside * cc))
                        if held:
                            if nside == side and abs(f - frac) <= 1e-12:
                                q = qty  # the same committed fraction: held as it is (no order)
                            traded = abs(nside * q - side * qty)
                            cash += side * qty * (px - efill)  # the old position, realized at px without cost
                            ci = cc * traded / q
                            c_s += traded * px * cc
                            n_s += traded * px
                            if traded > 0.0:
                                f_s += 1
                        else:
                            ci = cc
                            c_s += q * px * cc
                            n_s += q * px
                            f_s += 1
                        efill = px * (1.0 + nside * ci)
                        side, qty, frac, held = nside, q, f, True
                    elif held:
                        fill = px * (1.0 - side * cc)
                        cash += side * qty * (fill - efill)
                        c_s += qty * px * cc
                        n_s += qty * px
                        f_s += 1
                        held = False
                if b == last[s] and held:  # the session's last position exits at the closing auction
                    px = moc_px[s]
                    cc = c_moc[s]
                    fill = px * (1.0 - side * cc)
                    cash += side * qty * (fill - efill)
                    c_s += qty * px * cc
                    n_s += qty * px
                    f_s += 1
                    held = False
                eq = cash + side * qty * (cls[b] - efill) if held else cash
                if daily_loss > 0.0 and eq / sess_start - 1.0 <= -daily_loss:
                    flatten = True
                    blocked = True
                prev_eq = eq
            eq_end[s, col] = prev_eq
            cost[s, col] = c_s
            notional[s, col] = n_s
            fills[s, col] = f_s


@dataclass(frozen=True)
class PositionResult:
    """Per session (rows, NY dates) × series (columns): end-of-session equity, daily return (equity over the
    previous session's end; the first over INIT_CASH), cost paid and traded notional in cash, fills."""

    equity: pd.DataFrame
    ret: pd.DataFrame
    cost: pd.DataFrame
    notional: pd.DataFrame
    fills: pd.DataFrame
    missing_prints: int  # MOC sessions without a closing print (filled at the bar's close)


def _profile_loss(profile) -> float:
    if profile is None:
        return 0.0
    from risk.profiles import RiskProfile

    if profile != RiskProfile(profile.name, daily_loss=profile.daily_loss):
        raise ValueError(f"position_backtest supports a profile's daily_loss only; {profile.name!r} sets more (caps, "
                         "a vol target, drawdown tiers or borrow): run the portfolio simulator")  # fmt: skip
    return 0.0 if profile.daily_loss is None else float(profile.daily_loss)


def position_backtest(
    df: pd.DataFrame,
    targets: pd.Series | pd.DataFrame,
    cfg: RunConfig,
    *,
    costs: pd.DataFrame | float | None = None,
    prints: pd.DataFrame | None = None,
    cash_yield: pd.Series | None = None,
    profile=None,
    sessions=None,
    grid: DecisionGrid | None = None,
) -> PositionResult:
    """
    Daily P&L of target-position series (module docstring).

    Args:
        df: one symbol's bars (5Min OHLC).
        targets: target fractions in [−1, 1] indexed by decision bars (a Series, or a DataFrame with one series per
            column); a scheduled decision not in the index is flat.
        cfg: the run configuration (schedule, SIZE, INIT_CASH, SLIPPAGE_PCT, FILL_AUCTION).
        costs: risk.costs.fill_costs frame for df's bars, a scalar one-way cost on every fill, or None (SLIPPAGE_PCT).
        prints: the symbol's auction prints (data.bars.load_prints), read under FILL_AUCTION "print".
        cash_yield: annual rate per NY session date (the simulator's `cash_yield`); None = no yield.
        profile: a risk profile with at most `daily_loss` set (None = no gate).
        sessions: the exchange calendar's session dates (the schedule's session clock).
        grid: a precomputed decision_grid (reused across calls).
    """
    grid = decision_grid(df, cfg, sessions) if grid is None else grid
    frame = targets.to_frame() if isinstance(targets, pd.Series) else targets
    extra = frame.index.difference(grid.index)
    if len(extra):
        raise ValueError(f"{len(extra)} targets are not at scheduled decision bars (first {extra[0]})")
    if frame.isna().to_numpy().any():
        raise ValueError("targets contain NaN: a flat decision is 0 (or absent from the index)")
    tgt = frame.reindex(grid.index).fillna(0.0).to_numpy(dtype=float, copy=True)
    if tgt.size and (np.abs(tgt) > 1.0 + 1e-12).any():
        raise ValueError("targets must lie in [−1, 1]")
    n, n_sess = len(df), len(grid.first)
    opn = df["open"].to_numpy(dtype=float)
    cls = df["close"].to_numpy(dtype=float)
    # a session without its closing-auction bar has no MOC fill: its last decision is flat
    dsess = np.searchsorted(grid.first, grid.dec, side="right") - 1
    nxt_same = np.r_[dsess[1:] == dsess[:-1], False] if len(dsess) else np.zeros(0, bool)
    tgt[~nxt_same & ~grid.moc[dsess]] = 0.0
    act = np.full(n, -1, np.int64)
    act[grid.dec + 1] = np.arange(len(grid.dec))
    # the MOC reference price: the official closing print under FILL_AUCTION "print" (a session without one fills at
    # the bar's close and is counted). Entries never fill at an opening-auction bar: decisions are after 09:30.
    moc_px = cls[grid.last].copy()
    missing_prints = 0
    if cfg.FILL_AUCTION == "print":
        pc = np.full(n_sess, np.nan)
        if prints is not None and len(prints):
            pidx = pd.DatetimeIndex(prints.index)
            pday = (pidx.tz_convert("America/New_York").tz_localize(None) if pidx.tz is not None else pidx).normalize()
            pc = pd.Series(prints["close"].to_numpy(dtype=float), index=pday).reindex(grid.days).to_numpy()
        ok = ~np.isnan(pc) & grid.moc
        moc_px = np.where(ok, pc, moc_px)
        missing_prints = int((grid.moc & np.isnan(pc)).sum())
    if costs is None or np.isscalar(costs):
        c = cfg.SLIPPAGE_PCT if costs is None else float(costs)
        c_open, c_close = np.full(n, c), np.full(n, c)
    else:
        cf = costs.reindex(df.index)
        c_open, c_close = cf["open"].to_numpy(dtype=float), cf["close"].to_numpy(dtype=float)
        if np.isnan(c_open[grid.dec + 1]).any() or np.isnan(c_close[grid.last]).any():
            raise ValueError("no cost estimate at some fill bars (supply more 5Min history / quotes rows)")
    rate, gap = np.full(n_sess, np.nan), np.zeros(n_sess)
    if cash_yield is not None:
        cy = pd.Series(cash_yield, dtype=float)
        cidx = pd.DatetimeIndex(cy.index)
        cidx = (cidx.tz_convert("America/New_York").tz_localize(None) if cidx.tz is not None else cidx).normalize()
        r = pd.Series(cy.to_numpy(), index=cidx).reindex(grid.days).to_numpy()  # the rate as of each session
        rate[1:] = r[:-1]  # session s earns the previous session's rate over the calendar days of the gap
        gap[1:] = np.diff(grid.days.to_numpy().astype("datetime64[D]").astype(np.int64))
    shape = (n_sess, tgt.shape[1])
    eq_end, cost, notional, fills = np.empty(shape), np.empty(shape), np.empty(shape), np.zeros(shape, np.int64)
    _kernel(grid.first.astype(np.int64), grid.last.astype(np.int64), opn, cls, c_open, act, np.ascontiguousarray(tgt),
            moc_px, c_close[grid.last], rate, gap, _profile_loss(profile), float(cfg.INIT_CASH), float(cfg.SIZE),
            eq_end, cost, notional, fills)  # fmt: skip
    idx = pd.DatetimeIndex(grid.days, name="session")
    eq = pd.DataFrame(eq_end, index=idx, columns=frame.columns)
    prev = np.vstack([np.full((1, shape[1]), float(cfg.INIT_CASH)), eq_end[:-1]])
    return PositionResult(
        equity=eq,
        ret=pd.DataFrame(eq_end / prev - 1.0, index=idx, columns=frame.columns),
        cost=pd.DataFrame(cost, index=idx, columns=frame.columns),
        notional=pd.DataFrame(notional, index=idx, columns=frame.columns),
        fills=pd.DataFrame(fills, index=idx, columns=frame.columns),
        missing_prints=missing_prints,
    )
