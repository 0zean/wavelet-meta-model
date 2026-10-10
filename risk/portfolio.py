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

Scheduled entries (SPEC §16, U16):
- Under a market-on-close schedule (features.events.entry_at_close) every entry fills at the CLOSE of its entry bar
  (cost kind `close`), after that bar's exits; it is decided at close[b−1] like every other entry.
- A held position whose exit at bar b is known at the decision time (a `time` / `hysteresis` fill at the open, decided
  at close[b−1] at the latest, or a scheduled `vertical` close fill; never a triple-barrier touch) and comes at or
  before the new entry's fill does not block the symbol. When it exits at the same fill as the new entry (a
  rebalance: hold to the next scheduled entry), the two are one order: the old position closes at the reference price
  without cost, and the new one pays the fill's cost on the traded shares |side·q_new − side_old·q_old| only (booked in
  its entry fill). A new bet on the same side with the same committed fraction keeps the old shares (q_new = q_old):
  an unchanged position is held through the rebalance, untraded and at no cost.

U22 (SPEC §20):
- Auction prints. With cfg.FILL_AUCTION "print" and `prints` (symbol → data.bars.load_prints frame) every fill at an
  opening / closing auction bar (risk.costs.auction_flags: MOO entries, exits and flattening at a 09:30 bar's open;
  MOC entries and `vertical` exits at a closing bar) is priced at the session's official print instead of the bar's
  open / close; a session without a print falls back to the bar price and is counted (`attrs["auction_fallbacks"]`).
  Gap fills through a barrier and intrabar touches are never auction fills. Marks (equity) stay at bar closes.
- A market-on-close entry sizes its shares from the last close known at the decision (the previous bar's close), not
  from the fill price, which is unknown until the auction; `frac` records the intended fraction.
- Cost ledger: `attrs["daily_costs"]` = per session (NY date): the session's starting equity and, per fill class
  (risk.costs.FILL_CLASSES: open_auction, open, intra, close, close_auction), the traded notional and the one-way cost
  paid, so a stream can be re-priced at another cost per class (the family report's cost curve, SPEC §22).
- Cash yield: `cash_yield` (annual rate per NY session date, the 3-month T-bill as of that session) credits, at each
  session's first bar, the free cash (max(0, min(cash, equity)) at the previous close: short proceeds beyond the
  equity earn nothing) × rate of the previous session × calendar days / 360 (`attrs["cash_interest"]`).
- `sessions` (symbol → the exchange calendar's session dates) is the time model's session clock (features.exits).

With profile "none" on one symbol this is bit-identical to wfo.backtest.simulate_trades + equity_curve.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.events import entry_at_close
from features.exits import exit_phase
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


def _bets(bars, signals, cfg, side_col, size_col, sessions=None) -> pd.DataFrame:
    """All candidate bets across symbols with their local and union positions, in (union entry, symbol) order."""
    frames = []
    for sym, df in bars.items():
        sig = signals[sym]
        sig = sig[(sig.index >= df.index[0]) & (sig.index <= df.index[-1])]
        cand = _candidates(sig, side_col, size_col, cfg.SIZE_STEP)
        ex = _exits(df, cand, cfg, None if sessions is None else sessions.get(sym))
        ex["side"] = cand.loc[ex.index, "side"].astype(int)
        ex["size"] = cand.loc[ex.index, "size"]
        ex["sym"] = sym
        open_ = df["open"].to_numpy()
        ex["exit_phase"] = exit_phase(
            ex["barrier"].to_numpy(), ex["exit_px"].to_numpy(), open_[ex["exit_pos"].to_numpy()]
        )
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
    prints: dict[str, pd.DataFrame | None] | None = None,
    cash_yield: pd.Series | None = None,
    sessions: dict[str, pd.DatetimeIndex] | None = None,
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
        prints: symbol → auction prints (data.bars.load_prints; None for a symbol without), read under
            cfg.FILL_AUCTION "print" (module docstring).
        cash_yield: annual rate per NY session date credited on free cash (module docstring); None = no yield.
        sessions: symbol → the exchange calendar's session dates (the time model's session clock); None = the data's.

    Returns:
        (equity, trades, log): close-marked equity on the union timeline (attrs "turnover", "exposure" = share of
        bars with a position at any point, "avg_position" = mean Σ committed fraction of the held positions over the
        closes with a position — m·SIZE per position without the risk layer, as the single-mode backtest;
        "traded_notional" / "cost_paid" = Σ shares traded × reference price / Σ their one-way costs, in cash, over
        every fill: a rolled exit trades nothing, a roll's entry only the traded shares); one row per executed position (entry / exit union positions, fills, one-way `entry_cost_bp` /
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
    from data.timeframes import get_timeframe
    from risk.costs import FILL_CLASSES, auction_flags

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
    # U22: auction bars per symbol on the union timeline, and the official prints where FILL_AUCTION reads them
    minutes = get_timeframe(cfg.TIMEFRAME).minutes
    use_prints = cfg.FILL_AUCTION == "print"
    auc_open, auc_close, pr_open, pr_close = {}, {}, {}, {}
    for s in syms:
        io_, ic_ = auction_flags(bars[s].index, minutes)
        auc_open[s], auc_close[s] = np.zeros(n, bool), np.zeros(n, bool)
        auc_open[s][loc[s]], auc_close[s][loc[s]] = io_, ic_
        pr_open[s], pr_close[s] = np.full(n, np.nan), np.full(n, np.nan)
        pr = None if prints is None else prints.get(s)
        if use_prints and pr is not None and len(pr):
            pidx = pd.DatetimeIndex(pr.index)
            pday = (pidx.tz_convert("America/New_York").tz_localize(None) if pidx.tz is not None else pidx).normalize()
            bidx = bars[s].index
            bday = (bidx.tz_convert("America/New_York").tz_localize(None) if bidx.tz is not None else bidx).normalize()
            po = pd.Series(pr["open"].to_numpy(dtype=float), index=pday).reindex(bday).to_numpy()
            pc = pd.Series(pr["close"].to_numpy(dtype=float), index=pday).reindex(bday).to_numpy()
            pr_open[s][loc[s]] = np.where(io_, po, np.nan)
            pr_close[s][loc[s]] = np.where(ic_, pc, np.nan)
    fallbacks = 0

    def px_open(s: str, b: int) -> float:
        """Reference price of an open fill at bar b: the opening print at an opening-auction bar (FILL_AUCTION print),
        else the bar's open (counted as a fallback at an auction bar without a print)."""
        nonlocal fallbacks
        if use_prints and auc_open[s][b]:
            v = pr_open[s][b]
            if not np.isnan(v):
                return float(v)
            fallbacks += 1
        return opn[s][b]

    def px_close(s: str, b: int) -> float:
        nonlocal fallbacks
        if use_prints and auc_close[s][b]:
            v = pr_close[s][b]
            if not np.isnan(v):
                return float(v)
            fallbacks += 1
        return cls[s][b]

    def fill_class(s: str, b: int, kind: str) -> str:
        if kind == "open" and auc_open[s][b]:
            return "open_auction"
        if kind == "close" and auc_close[s][b]:
            return "close_auction"
        return kind

    book_notional = {k: np.zeros(n) for k in FILL_CLASSES}
    book_cost = {k: np.zeros(n) for k in FILL_CLASSES}
    rate = None
    if cash_yield is not None:
        cy = pd.Series(cash_yield, dtype=float)
        cidx = pd.DatetimeIndex(cy.index)
        cidx = (cidx.tz_convert("America/New_York").tz_localize(None) if cidx.tz is not None else cidx).normalize()
        rate = pd.Series(cy.to_numpy(), index=cidx)
    sess_day = session.tz_localize(None) if session.tz is not None else session
    interest_total, yield_missing = 0.0, 0

    bets = _bets(bars, signals, cfg, side_col, size_col, sessions)
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
    b_barrier = bets["barrier"].to_numpy() if len(bets) else np.array([], object)
    close_entry = entry_at_close(cfg)
    entry_phase = 2 if close_entry else 0

    def known_exit(bet: int, b: int) -> bool:
        """The bet's exit falls at bar b, at or before this run's entry fill, and is known at close[b−1]."""
        if b_exit_b[bet] != b or b_phase[bet] > entry_phase:
            return False
        return b_barrier[bet] in ("time", "hysteresis") or (b_barrier[bet] == "vertical" and cfg.EXIT_MODEL == "time")

    def exit_px(i: int, s: str, b: int) -> float:
        """The exit's reference price: an auction print for a MOC (`vertical` at a closing bar) or MOO (`time` at an
        opening bar) exit under FILL_AUCTION print, else the exit model's price."""
        if b_barrier[i] == "vertical" and auc_close[s][b]:
            return px_close(s, b)
        if b_barrier[i] in ("time", "hysteresis") and auc_open[s][b]:
            return px_open(s, b)
        return float(b_exit_px[i])

    cash = float(cfg.INIT_CASH)
    lots: dict[str, _Lot] = {}
    busy_until = dict.fromkeys(syms, -1)  # local position of the last bar the symbol's position occupies
    mark = dict.fromkeys(syms, np.nan)  # last close per symbol
    equity = np.full(n, np.nan)
    prev_eq, hwm, sess_start = cash, cash, cash
    flatten: set[str] = set()
    blocked_session = None
    turnover, pos_bars, pos_sum, held_bars = 0.0, 0, 0.0, 0
    notional, cost_cash = 0.0, 0.0  # Σ traded notional at the fills' reference prices and Σ one-way costs, in cash
    trades: list[dict] = []
    log = {k: np.zeros(n) for k in ("gross", "net", "long", "short", "count", "max_pos", "dd_mult", "gate")}
    open_info: dict[str, dict] = {}  # per open lot: entry details for the trade record
    cap = profile.position_cap
    tol = 1.0 + profile.drift_tol

    def unrealized(sym_px: dict[str, float]) -> float:
        return sum(lt.side * lt.qty * (sym_px[s] - lt.entry_fill) for s, lt in lots.items())

    def close_lot(
        s: str, b: int, px: float, reason: str, kind: str, frac_of_lot: float = 1.0, rolled: bool = False
    ) -> None:
        """Sell (part of) a lot at reference price px with adverse costs (fill kind `kind`; none when `rolled` into a
        new position at the same fill, which pays the traded shares' cost); realize P&L into cash."""
        nonlocal cash, turnover, notional, cost_cash
        lt = lots[s]
        q = lt.qty if frac_of_lot == 1.0 else lt.qty * frac_of_lot
        # equity with this lot marked at px (the others at their marks), before the fill's cost
        marks = {k: (px if k == s else mark[k]) for k in lots}
        E = cash + unrealized(marks)
        c = 0.0 if rolled else cost(s, b, kind)
        fill = px * (1.0 - lt.side * c)
        pnl = lt.side * q * (fill - lt.entry_fill)
        if profile.borrow_bps and lt.side < 0:
            pnl -= q * lt.entry_fill * profile.borrow_bps * 1e-4 * (b - lt.entry_b + 1) / cfg.bars_per_year
        if not rolled:
            turnover += q * fill / E
            notional += q * px
            cost_cash += q * px * c
            fcls = fill_class(s, b, kind)
            book_notional[fcls][b] += q * px
            book_cost[fcls][b] += q * px * c
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
                    "rolled": rolled,
                }
            )
        else:
            lt.qty -= q
            info["frac"] *= 1.0 - frac_of_lot
            info["trimmed"] = True

    def enter(f: float, s: str, i: int, b: int, px: float, kind: str, E_dec: float, roll: bool) -> None:
        """Open bet i at reference price px of bar b (fill kind `kind`) with f of the decision-time equity E_dec;
        `roll`: the symbol's position exiting at this fill is rolled into it (cost on the traded shares only)."""
        nonlocal turnover, notional, cost_cash
        side = int(b_side[i])
        c_in = cost(s, b, kind)
        fill = px * (1.0 + side * c_in)
        # a market-on-close order is sized from the last close known at the decision (the previous bar's close); an
        # open fill keeps sizing from its fill (the U7–U18 convention)
        ref = mark[s] if kind == "close" and cfg.MOC_SIZE_FROM == "decision" and not np.isnan(mark[s]) else px
        qty = f * E_dec / (ref * (1.0 + side * c_in))
        if roll:
            old = lots[s]
            if side == old.side and abs(f - open_info[s]["frac"]) <= 1e-12:
                qty = old.qty  # the same committed fraction: the position is held as it is (no order)
            traded = abs(side * qty - old.side * old.qty)
            close_lot(s, b, px, "barrier", kind, rolled=True)
            c_in = c_in * traded / qty  # the traded shares' cost, per share of the new position
            fill = px * (1.0 + side * c_in)
            turnover += traded * px / E_dec
            notional += traded * px
            book_notional[fill_class(s, b, kind)][b] += traded * px
        else:
            turnover += qty * fill / E_dec
            notional += qty * px
            book_notional[fill_class(s, b, kind)][b] += qty * px
        cost_cash += qty * px * c_in  # a roll's c_in is already per share of the new position (the traded shares' cost)
        book_cost[fill_class(s, b, kind)][b] += qty * px * c_in
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
            "entry_px": px,
            "entry_fill": fill,
            "entry_cost_bp": c_in * 1e4,
            "qty": qty,
            "pnl": 0.0,
            "exit_notional": 0.0,
            "trimmed": False,
        }

    for b in range(n):
        here = [s for s in syms if has[s][b]]
        if rate is not None and new_session[b] and b > 0:  # interest on the free cash held over the gap (ACT/360)
            r = rate.get(sess_day[b - 1], np.nan)
            if np.isnan(r):
                yield_missing += 1
            else:
                free = max(0.0, min(cash, prev_eq))
                interest = free * float(r) * (sess_day[b] - sess_day[b - 1]).days / 360.0
                cash += interest
                prev_eq += interest
                hwm = max(hwm, prev_eq)
                interest_total += interest
        # ── decisions at close[b−1]: every input is known at that close (prev_eq, hwm, marks, gate flags). Positions
        # that will gap through a barrier at open[b] are unknown then: they keep their slot and exposure below.
        dd = 1.0 - prev_eq / hwm
        dd_mult = profile.dd_multiplier(dd)
        log["dd_mult"][b] = dd_mult
        E_dec = prev_eq
        in_bar = bool(lots)
        held_dec = set(lots)
        freed = {s for s, lt in lots.items() if s in here and known_exit(lt.bet, b) and s not in flatten}
        gated = {s for s in here if s in flatten and s in lots}
        leaving = {s for s in freed if b_phase[lots[s].bet] == 0}  # gone at this open: no trims for or against it
        expo = {s: lt.side * lt.qty * mark[s] / E_dec for s, lt in lots.items() if s not in gated | leaving}
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
        # positions after the entries' fill, as decided (a freed position has exited by then)
        held = [expo[s] * scale[s] for s in expo if scale[s] > 0 and s not in freed]
        cand = []
        blocked = blocked_session is not None and session[b] == blocked_session
        for i in entries_at.get(b, ()):
            s = b_sym[i]
            if ((s in held_dec or b_entry_pos[i] <= busy_until[s]) and s not in freed) or blocked:
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
        # a freed position exiting at the entries' own fill is rolled into the new one (one order)
        rolls = {s for f, s, _ in cand if f > 0 and s in freed and b_phase[lots[s].bet] == entry_phase}
        # ── execution at open[b] ──────────────────────────────────────────────────────────────────
        # (1) gap exits at the open
        for s in here:
            lt = lots.get(s)
            if lt is not None and b_exit_b[lt.bet] == b and b_phase[lt.bet] == 0 and s not in rolls:
                close_lot(s, b, exit_px(lt.bet, s, b), "barrier", "open")
        # (2) gate flattening at this symbol's next open
        for s in here:
            if s in flatten:
                flatten.discard(s)
                if s in lots:
                    close_lot(s, b, px_open(s, b), "gate", "open")
                    busy_until[s] = int(np.searchsorted(loc[s], b))
        # (3) drift trims (a position that already gapped out at this open has nothing left to trim)
        for s in [s for s in scale if scale[s] < 1.0 and s in lots and s not in rolls]:
            if scale[s] <= 0:
                close_lot(s, b, px_open(s, b), "trim", "open")
                busy_until[s] = int(np.searchsorted(loc[s], b))
            else:
                close_lot(s, b, px_open(s, b), "trim", "open", frac_of_lot=1.0 - scale[s])
        # (4) entries at the open
        if cand and not close_entry:
            for f, s, i in cand:
                if f > 0:
                    enter(f, s, i, b, px_open(s, b), "open", E_dec, s in rolls)
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
                # a lot that a market-on-close entry rolls at this close is closed by that roll, not here; a lot
                # entered at this bar's open (a roll at phase 0) may still exit at its close (U22: `next_event`)
                if b_exit_b[lt.bet] == b and b_phase[lt.bet] == phase and not (phase == 2 and close_entry
                                                                             and s in rolls):  # fmt: skip
                    close_lot(s, b, exit_px(lt.bet, s, b), "barrier", "intra" if phase == 1 else "close")
        # (7) market-on-close entries at the close
        if cand and close_entry:
            for f, s, i in cand:
                if f > 0:
                    enter(f, s, i, b, px_close(s, b), "close", E_dec, s in rolls)
            in_bar |= bool(lots)
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
    out.attrs["traded_notional"] = notional  # the family test's edge-to-cost floor (SPEC §17.2)
    out.attrs["cost_paid"] = cost_cash
    out.attrs["auction_fallbacks"] = fallbacks
    out.attrs["cash_interest"] = interest_total
    out.attrs["cash_yield_missing_sessions"] = yield_missing
    # the per-session cost ledger (module docstring): starting equity = the previous session's last close (the
    # initial cash for the first), notional and cost paid per fill class
    sid = np.cumsum(new_session) - 1
    firsts = np.flatnonzero(new_session)
    eq_prev = np.r_[float(cfg.INIT_CASH), equity[:-1]]
    daily = {"equity_start": eq_prev[firsts]}
    for k in FILL_CLASSES:
        daily[f"notional_{k}"] = np.bincount(sid, weights=book_notional[k], minlength=len(firsts))
    for k in FILL_CLASSES:
        daily[f"cost_{k}"] = np.bincount(sid, weights=book_cost[k], minlength=len(firsts))
    out.attrs["daily_costs"] = pd.DataFrame(daily, index=pd.DatetimeIndex(sess_day[firsts], name="session"))
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["pnl_pct"] = tr["side"] * (tr["exit_fill"] / tr["entry_fill"] - 1.0)
        # trimmed positions and borrow: the per-unit return of the whole position = cash P&L / entry notional
        whole = tr["trimmed"] | ((tr["side"] < 0) & (profile.borrow_bps > 0))
        tr.loc[whole, "pnl_pct"] = tr["pnl"] / (tr["qty"] * tr["entry_fill"])
        tr["bars_held"] = tr["exit_b"] - tr["entry_b"] + 1
        tr["cost_bp"] = tr["entry_cost_bp"] + tr["exit_cost_bp"]  # round trip, one-way costs at the two fills
        tr["rolled"] = tr["rolled"].astype(bool)
        tr = tr.set_index("event").sort_values(["entry_b", "sym"], kind="stable")
    else:
        tr = pd.DataFrame(columns=["sym", "side", "size", "frac", "pnl", "pnl_pct", "bars_held", "cost_bp", "rolled"])
    return out, tr, pd.DataFrame(log, index=union)
