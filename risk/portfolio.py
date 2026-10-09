"""
Event-driven portfolio backtest with the risk layer (SPEC §7, U8).

Every symbol's approved events become bets with the backtest's triple-barrier exits (barrier_exits, side-aware), one
position per symbol at a time as in POSITION_MODE="single" (an event arriving while the symbol holds a position is
skipped; a bet whose final size is 0 does not block later events). All symbols share one union timeline and one
equity curve.

Timing. Every decision that fills at open[b] (all of them computed before any fill at that open) is taken at the close of the previous union bar b−1 with information up to
that close only: the equity E, drawdown and session P&L, the marks (each symbol's last close) of held positions, the
barrier width at the event and the cost estimate. Per union bar b:
  open   1. barrier exits that gap through a barrier at the open        (scheduled, not decisions)
         2. daily-loss-gate flattening                                  (decided at close[b−1])
         3. drift trims back to the caps                                (decided at close[b−1])
         4. entries, notional = f · E(close[b−1]), shares = notional / fill
  intra  5. intrabar barrier exits in bet order
  close  6. vertical-barrier exits; mark every position at its last close → equity[b]; drawdown; gate check.

Risk layer on a new entry (profile = risk.profiles.RiskProfile), fraction of equity f:
  f = min(SIZE · m · min(1, vol_target / σ_hold), position cap) · dd_multiplier(DD at close[b−1])
    m = discretized bet size (as in the single-mode backtest); σ_hold = width / BARRIER_MULT = σ_bar·√VERTICAL_BARS at
    the event; position cap = min(per-position, per-symbol) (one position per symbol);
  entries at one bar ranked by |f| (then symbol) fill the free slots up to max_concurrent, then are scaled down
  pro-rata by one factor k ∈ [0, 1] so gross and each side's gross (held positions at their marks + new) stay within
  max_gross and max_net. Capping each side (long total, short total) at max_net bounds |net| whichever positions
  exit — a cap on the net alone breaks as soon as the offsetting side exits. Held positions are not resized, except a
  drift trim when marks take one position past cap·(1 + drift_tol) (back to the cap) or gross / a side's gross past
  theirs (tradeable positions scaled pro-rata back to the cap). Only a symbol with a bar
  at b can trade: a held symbol without one is trimmed at its next bar, so its stale-marked exposure may sit outside the
  band until then. The caps therefore hold exactly on decision-time marks at every entry, and within drift_tol after
  every open for the positions whose symbol has a bar there. Positions that gap through a barrier at open[b] are not
  known at close[b−1]: they still count toward slots and headroom for the entries filled at that open.
Daily loss gate: session P&L = equity[close b] / equity at the previous session's last close − 1. At ≤ −daily_loss
every position is flattened at its symbol's next open and entries are blocked for the rest of that session (for
daily bars, the next session: its open is the next decision). Drawdown tiers use the close-marked equity high-water mark.
Costs: every fill (entry, exit, trim, flatten) pays its one-way cost c adversely on the fill price: SLIPPAGE_PCT
without `costs`, else the symbol's per-bar cost for the kind of fill (risk.costs.fill_costs, cfg.COST_MODEL): `open`
for entries, gap exits, gate flattening and trims, `intra` for intrabar barrier exits, `close` for vertical exits.
Short borrow at profile.borrow_bps (annual) is charged at exit on the entry notional for the union bars held.

With profile "none" on one symbol this is bit-identical to wfo.backtest.simulate_trades + equity_curve.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk.profiles import RiskProfile
from utils.config import RunConfig
from wfo.backtest import _candidates, _exits


@dataclass
class _Lot:
    sym: str
    bet: int  # row in the bet table
    side: int
    qty: float
    entry_fill: float
    entry_b: int  # union bar of the entry


def _bets(bars, signals, cfg, side_col, size_col) -> pd.DataFrame:
    """All candidate bets across symbols with their local and union positions, in (union entry, symbol) order."""
    frames = []
    for sym, df in bars.items():
        sig = signals[sym]
        sig = sig[(sig.index >= df.index[0]) & (sig.index <= df.index[-1])]
        cand = _candidates(sig, side_col, size_col, cfg.SIZE_STEP)
        ex = _exits(df, cand, cfg)
        ex["side"] = cand.loc[ex.index, "side"].astype(int)
        ex["size"] = cand.loc[ex.index, "size"]
        ex["sym"] = sym
        open_ = df["open"].to_numpy()
        vertical = (ex["barrier"] == "vertical").to_numpy()
        gap = ex["exit_px"].to_numpy() == open_[ex["exit_pos"].to_numpy()]
        ex["exit_phase"] = np.where(vertical, 2, np.where(gap, 0, 1))
        frames.append(ex.rename_axis("event").reset_index())
    cols = ["event", "sym", "side", "size", "width", "entry_pos", "exit_pos", "entry_px", "exit_px", "barrier"]
    if not frames:
        return pd.DataFrame(columns=[*cols, "exit_phase"])
    return pd.concat(frames, ignore_index=True)


def simulate_portfolio(
    bars: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    cfg: RunConfig,
    profile: RiskProfile,
    *,
    side_col: str = "trade_signal",
    size_col: str | None = "bet_size",
    costs: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Portfolio backtest (see the module docstring).

    Args:
        bars: symbol → OHLC over the span to simulate.
        signals: symbol → WFO output (`width`, side and size columns).
        cfg: Run configuration (barrier rules, SIZE, SIZE_STEP, SLIPPAGE_PCT, INIT_CASH, BARRIER_MULT).
        profile: Risk profile.
        side_col / size_col: Side {-1, 0, 1} and raw bet-size columns (size_col None → m = 1).
        costs: symbol → one-way cost fraction per bar of bars[symbol] by fill kind (columns open / intra / close,
            risk.costs.fill_costs); None = SLIPPAGE_PCT on every fill.

    Returns:
        (equity, trades, log): close-marked equity on the union timeline (attrs "turnover", "exposure" = share of
        bars with a position at any point, "avg_position" = mean Σ committed fraction of the held positions over the
        closes with a position — m·SIZE per position without the risk layer, as the single-mode backtest); one row per executed position (entry / exit union positions, fills, one-way `entry_cost_bp` /
        `exit_cost_bp` and round-trip `cost_bp` (the final exit's cost for a trimmed position), `size` = m,
        `frac` = committed fraction of equity, `pnl` in cash, per-unit `pnl_pct`, `exit_reason`); a per-bar log
        (gross / net / count / max |position| on decision-time marks after the open's decisions, the drawdown
        multiplier, gate state).
    """
    if cfg.POSITION_MODE != "single":
        raise ValueError("the portfolio simulator holds one position per symbol (POSITION_MODE='single')")
    if set(bars) != set(signals):
        raise ValueError("bars and signals must have the same symbols")
    if costs is not None and set(costs) != set(bars):
        raise ValueError("costs must hold a cost frame for every symbol")

    syms = sorted(bars)
    union = bars[syms[0]].index
    for s in syms[1:]:
        union = union.union(bars[s].index)
    n = len(union)
    session = union.normalize()
    new_session = np.r_[True, session[1:] != session[:-1]]
    daily_bars = cfg.BARS_PER_DAY == 1

    # Per-symbol arrays on the union timeline (NaN where the symbol has no bar)
    loc = {s: union.get_indexer(bars[s].index) for s in syms}
    has = {s: np.zeros(n, dtype=bool) for s in syms}
    opn, cls, fc = {}, {}, {}
    for s in syms:
        has[s][loc[s]] = True
        opn[s], cls[s] = np.full(n, np.nan), np.full(n, np.nan)
        opn[s][loc[s]] = bars[s]["open"].to_numpy()
        cls[s][loc[s]] = bars[s]["close"].to_numpy()
        if costs is not None:
            c = costs[s].reindex(bars[s].index)
            fc[s] = {k: np.full(n, np.nan) for k in ("open", "intra", "close")}
            for k in fc[s]:
                fc[s][k][loc[s]] = c[k].to_numpy()
    slip = cfg.SLIPPAGE_PCT

    bets = _bets(bars, signals, cfg, side_col, size_col)
    if len(bets):
        bets["entry_b"] = [loc[s][p] for s, p in zip(bets["sym"], bets["entry_pos"])]
        bets["exit_b"] = [loc[s][p] for s, p in zip(bets["sym"], bets["exit_pos"])]
        bets = bets.sort_values(["entry_b", "sym"], kind="stable").reset_index(drop=True)
    entries_at: dict[int, list[int]] = {}
    for i, b in enumerate(bets["entry_b"].to_numpy() if len(bets) else []):
        entries_at.setdefault(int(b), []).append(i)
    if costs is not None and len(bets):
        for s in syms:
            need = bets.loc[bets["sym"] == s, ["entry_b", "exit_b"]].to_numpy().ravel()
            if len(need) and any(np.isnan(fc[s][k][need]).any() for k in fc[s]):
                raise ValueError(f"{s}: no cost estimate at some fill bars (supply more 5Min history / quotes rows)")

    def cost(s: str, b: int, kind: str) -> float:
        return slip if costs is None else fc[s][kind][b]

    b_side = bets["side"].to_numpy() if len(bets) else np.array([], int)
    b_size = bets["size"].to_numpy(dtype=float) if len(bets) else np.array([])
    b_width = bets["width"].to_numpy(dtype=float) if len(bets) else np.array([])
    b_exit_b = bets["exit_b"].to_numpy() if len(bets) else np.array([], int)
    b_exit_pos = bets["exit_pos"].to_numpy() if len(bets) else np.array([], int)
    b_entry_pos = bets["entry_pos"].to_numpy() if len(bets) else np.array([], int)
    b_exit_px = bets["exit_px"].to_numpy(dtype=float) if len(bets) else np.array([])
    b_phase = bets["exit_phase"].to_numpy() if len(bets) else np.array([], int)
    b_sym = bets["sym"].to_numpy() if len(bets) else np.array([], object)

    cash = float(cfg.INIT_CASH)
    lots: dict[str, _Lot] = {}
    busy_until = dict.fromkeys(syms, -1)  # local position of the last bar the symbol's position occupies
    mark = dict.fromkeys(syms, np.nan)  # last close per symbol
    equity = np.full(n, np.nan)
    prev_eq, hwm, sess_start = cash, cash, cash
    flatten: set[str] = set()
    blocked_session = None
    turnover, pos_bars, pos_sum, held_bars = 0.0, 0, 0.0, 0
    trades: list[dict] = []
    log = {k: np.zeros(n) for k in ("gross", "net", "long", "short", "count", "max_pos", "dd_mult", "gate")}
    open_info: dict[str, dict] = {}  # per open lot: entry details for the trade record
    cap = profile.position_cap
    tol = 1.0 + profile.drift_tol

    def unrealized(sym_px: dict[str, float]) -> float:
        return sum(lt.side * lt.qty * (sym_px[s] - lt.entry_fill) for s, lt in lots.items())

    def close_lot(s: str, b: int, px: float, reason: str, kind: str, frac_of_lot: float = 1.0) -> None:
        """Sell (part of) a lot at reference price px with adverse costs (fill kind `kind`); realize P&L into cash."""
        nonlocal cash, turnover
        lt = lots[s]
        q = lt.qty if frac_of_lot == 1.0 else lt.qty * frac_of_lot
        # equity with this lot marked at px (the others at their marks), before the fill's cost
        marks = {k: (px if k == s else mark[k]) for k in lots}
        E = cash + unrealized(marks)
        c = cost(s, b, kind)
        fill = px * (1.0 - lt.side * c)
        pnl = lt.side * q * (fill - lt.entry_fill)
        if profile.borrow_bps and lt.side < 0:
            pnl -= q * lt.entry_fill * profile.borrow_bps * 1e-4 * (b - lt.entry_b + 1) / cfg.bars_per_year
        turnover += q * fill / E
        cash += pnl
        info = open_info[s]
        info["pnl"] += pnl
        info["exit_notional"] += q * fill
        if frac_of_lot == 1.0:
            del lots[s]
            del open_info[s]
            i = info["bet"]
            trades.append(
                {
                    **{k: v for k, v in info.items() if k not in ("bet", "exit_notional")},
                    "exit_b": b,
                    "exit_px": px,
                    "exit_fill": fill,
                    "exit_cost_bp": c * 1e4,
                    "exit_reason": reason if reason != "barrier" else bets.at[i, "barrier"],
                }
            )
        else:
            lt.qty -= q
            info["frac"] *= 1.0 - frac_of_lot
            info["trimmed"] = True

    for b in range(n):
        here = [s for s in syms if has[s][b]]
        # ── decisions at close[b−1]: every input is known at that close (prev_eq, hwm, marks, gate flags). Positions
        # that will gap through a barrier at open[b] are unknown then: they keep their slot and exposure below.
        dd = 1.0 - prev_eq / hwm
        dd_mult = profile.dd_multiplier(dd)
        log["dd_mult"][b] = dd_mult
        E_dec = prev_eq
        in_bar = bool(lots)
        held_dec = set(lots)
        gated = {s for s in here if s in flatten and s in lots}
        expo = {s: lt.side * lt.qty * mark[s] / E_dec for s, lt in lots.items() if s not in gated}
        scale = dict.fromkeys(expo, 1.0)
        if expo and profile.active:
            # drift trims (only symbols with a bar at b can trade; the others are trimmed at their next bar)
            for s, e in expo.items():
                if s in here and abs(e) > cap * tol:
                    scale[s] = cap / abs(e)
            gross = sum(abs(expo[s]) * scale[s] for s in expo)
            if gross > profile.max_gross * tol:
                fixed = sum(abs(expo[s]) * scale[s] for s in expo if s not in here)
                movable = gross - fixed
                k = max(0.0, (profile.max_gross - fixed) / movable) if movable > 0 else 1.0
                for s in expo:
                    if s in here:
                        scale[s] *= min(1.0, k)
            for sgn in (1.0, -1.0):  # each side's gross ≤ max_net, so |net| ≤ max_net whichever positions exit
                side_tot = sum(abs(expo[s]) * scale[s] for s in expo if np.sign(expo[s]) == sgn)
                if side_tot > profile.max_net * tol:
                    fixed = sum(abs(expo[s]) * scale[s] for s in expo if np.sign(expo[s]) == sgn and s not in here)
                    movable = side_tot - fixed
                    k = max(0.0, (profile.max_net - fixed) / movable) if movable > 0 else 1.0
                    for s, e in expo.items():
                        if s in here and np.sign(e) == sgn:
                            scale[s] *= min(1.0, k)
        held = [expo[s] * scale[s] for s in expo if scale[s] > 0]  # positions after the open, as decided
        cand = []
        blocked = blocked_session is not None and session[b] == blocked_session
        for i in entries_at.get(b, ()):
            s = b_sym[i]
            if s in held_dec or b_entry_pos[i] <= busy_until[s] or blocked:
                continue
            f = cfg.SIZE * b_size[i]
            if profile.active:
                if profile.vol_target is not None:
                    f *= min(1.0, profile.vol_target / (b_width[i] / cfg.BARRIER_MULT))
                f = min(f, cap) * dd_mult  # SPEC order: vol target → position cap → drawdown throttle
            if f > 0:
                cand.append((f, s, i))
        if cand and profile.active:
            cand.sort(key=lambda c: (-c[0], c[1]))
            if profile.max_concurrent is not None:
                cand = cand[: max(0, profile.max_concurrent - len(held))]
            k = 1.0
            g_new = sum(f for f, _, _ in cand)
            k = min(k, (profile.max_gross - sum(abs(v) for v in held)) / g_new)
            for sgn in (1, -1):  # per-side gross ≤ max_net bounds |net| even if the other side exits at this open
                new = sum(f for f, _, i in cand if b_side[i] == sgn)
                if new > 0:
                    k = min(k, (profile.max_net - sum(abs(v) for v in held if np.sign(v) == sgn)) / new)
            k = max(0.0, k)
            cand = [(f * k, s, i) for f, s, i in cand] if k < 1.0 else cand
        # ── execution at open[b] ──────────────────────────────────────────────────────────────────
        # (1) gap exits at the open
        for s in here:
            lt = lots.get(s)
            if lt is not None and b_exit_b[lt.bet] == b and b_phase[lt.bet] == 0:
                close_lot(s, b, b_exit_px[lt.bet], "barrier", "open")
        # (2) gate flattening at this symbol's next open
        for s in here:
            if s in flatten:
                flatten.discard(s)
                if s in lots:
                    close_lot(s, b, opn[s][b], "gate", "open")
                    busy_until[s] = int(np.searchsorted(loc[s], b))
        # (3) drift trims (a position that already gapped out at this open has nothing left to trim)
        for s in [s for s in scale if scale[s] < 1.0 and s in lots]:
            if scale[s] <= 0:
                close_lot(s, b, opn[s][b], "trim", "open")
                busy_until[s] = int(np.searchsorted(loc[s], b))
            else:
                close_lot(s, b, opn[s][b], "trim", "open", frac_of_lot=1.0 - scale[s])
        # (4) entries
        if cand:
            for f, s, i in cand:
                if f <= 0:
                    continue
                side = int(b_side[i])
                c_in = cost(s, b, "open")
                fill = opn[s][b] * (1.0 + side * c_in)
                qty = f * E_dec / fill
                turnover += qty * fill / E_dec
                lots[s] = _Lot(s, i, side, qty, fill, b)
                busy_until[s] = int(b_exit_pos[i])
                open_info[s] = {
                    "bet": i,
                    "event": bets.at[i, "event"],
                    "sym": s,
                    "side": side,
                    "size": b_size[i],
                    "frac": f,
                    "entry_b": b,
                    "entry_px": opn[s][b],
                    "entry_fill": fill,
                    "entry_cost_bp": c_in * 1e4,
                    "qty": qty,
                    "pnl": 0.0,
                    "exit_notional": 0.0,
                    "trimmed": False,
                }
        # post-open exposure on decision-time marks (entries at their committed fraction)
        if lots:
            ex = [
                (lt.side * open_info[s]["frac"]) if lt.entry_b == b else lt.side * lt.qty * mark[s] / E_dec
                for s, lt in lots.items()
            ]
            log["gross"][b], log["net"][b] = sum(abs(v) for v in ex), sum(ex)
            log["long"][b], log["short"][b] = sum(v for v in ex if v > 0), -sum(v for v in ex if v < 0)
            log["count"][b], log["max_pos"][b] = len(ex), max(abs(v) for v in ex)
        in_bar |= bool(lots)
        # (5) intrabar barrier exits, bet order; (6) vertical exits at the close
        for phase in (1, 2):
            for s in sorted((s for s in here if s in lots), key=lambda s: lots[s].bet):
                lt = lots[s]
                if b_exit_b[lt.bet] == b and b_phase[lt.bet] == phase:
                    close_lot(s, b, b_exit_px[lt.bet], "barrier", "intra" if phase == 1 else "close")
        # ── close: marks, equity, drawdown, gate ────────────────────────────────────────────────────
        for s in here:
            mark[s] = cls[s][b]
        equity[b] = cash + unrealized(mark) if lots else cash
        if lots:
            pos_bars += 1
            pos_sum += sum(open_info[s]["frac"] for s in lots)
        held_bars += in_bar
        if new_session[b] and b > 0:
            sess_start = prev_eq
        hwm = max(hwm, equity[b])
        if profile.daily_loss is not None and equity[b] / sess_start - 1.0 <= -profile.daily_loss:
            log["gate"][b] = 1
            flatten |= set(lots)
            if daily_bars:
                blocked_session = session[b + 1] if b + 1 < n else None
            else:
                blocked_session = session[b]
        prev_eq = equity[b]

    if lots:  # barrier_exits drops bets whose window runs past the data, so every lot has exited
        raise RuntimeError(f"positions left open at the end of the span: {sorted(lots)}")
    out = pd.Series(equity, index=union, name="equity")
    out.attrs["turnover"] = turnover
    out.attrs["exposure"] = held_bars / n
    out.attrs["avg_position"] = pos_sum / pos_bars if pos_bars else np.nan
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["pnl_pct"] = tr["side"] * (tr["exit_fill"] / tr["entry_fill"] - 1.0)
        # trimmed positions and borrow: the per-unit return of the whole position = cash P&L / entry notional
        whole = tr["trimmed"] | ((tr["side"] < 0) & (profile.borrow_bps > 0))
        tr.loc[whole, "pnl_pct"] = tr["pnl"] / (tr["qty"] * tr["entry_fill"])
        tr["bars_held"] = tr["exit_b"] - tr["entry_b"] + 1
        tr["cost_bp"] = tr["entry_cost_bp"] + tr["exit_cost_bp"]  # round trip, one-way costs at the two fills
        tr = tr.set_index("event").sort_values(["entry_b", "sym"], kind="stable")
    else:
        tr = pd.DataFrame(columns=["sym", "side", "size", "frac", "pnl", "pnl_pct", "bars_held", "cost_bp"])
    return out, tr, pd.DataFrame(log, index=union)
