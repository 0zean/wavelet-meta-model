import numpy as np
import pandas as pd

from features.triple_barrier_labels import barrier_exits
from sizing import discretize
from utils.config import RunConfig


def _candidates(signals: pd.DataFrame, side_col: str, size_col: str | None, step: float) -> pd.DataFrame:
    """Events with a side, with their bet size m (1 without a size column), optionally discretized."""
    cand = signals.loc[signals[side_col] != 0, [side_col, "width"]].rename(columns={side_col: "side"})
    m = np.ones(len(cand)) if size_col is None else signals.loc[cand.index, size_col].to_numpy(dtype=float)
    if np.isnan(m).any() or (m < 0).any() or (m > 1).any():
        raise ValueError(f"bet sizes in {size_col!r} must lie in [0, 1]")
    cand["size"] = m if step is None else discretize(m, step)
    return cand[cand["size"] > 0]


def _exits(df: pd.DataFrame, cand: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    return barrier_exits(
        df,
        cand.index,
        cand["width"],
        side=cand["side"],
        vertical_bars=cfg.VERTICAL_BARS,
        hold_overnight=cfg.HOLD_OVERNIGHT,
    )


def _unit_pnl(trades: pd.DataFrame, cand: pd.DataFrame, slippage: float) -> pd.DataFrame:
    trades["side"] = cand.loc[trades.index, "side"].astype(int)
    trades["size"] = cand.loc[trades.index, "size"]
    trades["entry_fill"] = trades["entry_px"] * (1.0 + trades["side"] * slippage)
    trades["exit_fill"] = trades["exit_px"] * (1.0 - trades["side"] * slippage)
    trades["pnl_pct"] = trades["side"] * (trades["exit_fill"] / trades["entry_fill"] - 1.0)
    trades["bars_held"] = trades["exit_pos"] - trades["entry_pos"] + 1
    return trades


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: RunConfig,
    side_col: str = "trade_signal",
    size_col: str | None = None,
) -> pd.DataFrame:
    """
    Execute event signals with the same triple-barrier rules used for labeling (POSITION_MODE="single").

    Execution logic (see barrier_exits):
      • Signal at event bar t → enter at OPEN of bar t+1
      • Exit at the first barrier touched (adverse barrier on same-bar ties),
        else at the close of the vertical barrier (or the session close unless HOLD_OVERNIGHT)
      • One position at a time: events arriving while a position is open are skipped
      • Bet size m = the event's `size_col` discretized to SIZE_STEP (1 without a size column); m = 0 is no bet
        (it does not block later events)
      • Slippage is adverse on every fill: buys at price x (1 + pct), sells at price x (1 - pct)

    Args:
        df (pd.DataFrame): OHLC data.
        signals (pd.DataFrame): WFO output with `width` and a side column.
        cfg (RunConfig): Run configuration (barrier rules, SLIPPAGE_PCT, SIZE_STEP).
        side_col (str, optional): Column holding the side {-1, 0, +1}. Defaults to "trade_signal".
        size_col (str | None, optional): Column holding the raw bet size in [0, 1]. Defaults to None (m = 1).

    Returns:
        pd.DataFrame: One row per executed trade, indexed by signal time (`size` = m, `pnl_pct` per unit).
    """
    cand = _candidates(signals, side_col, size_col, cfg.SIZE_STEP)
    exits = _exits(df, cand, cfg)

    # Greedy non-overlapping selection: a new entry at open[t+1] needs the previous exit at or before bar t
    taken = np.zeros(len(exits), dtype=bool)
    last_exit = -1
    for i, (entry, exit_) in enumerate(zip(exits["entry_pos"].to_numpy(), exits["exit_pos"].to_numpy())):
        if entry > last_exit:
            taken[i] = True
            last_exit = exit_

    return _unit_pnl(exits[taken].copy(), cand, cfg.SLIPPAGE_PCT)


def equity_curve(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    init_cash: float,
    size: float,
) -> pd.Series:
    """
    Bar-by-bar mark-to-market equity: open positions are valued at each close,
    and at the exit bar at the exit fill. A trade commits size · m of equity (m = its `size` column, else 1).
    `attrs["turnover"]` = Σ traded notional / equity at the fill; `attrs["avg_position"]` = mean m over the
    closes at which a position is held (as in simulate_positions).

    Args:
        df (pd.DataFrame): OHLC data (the span to report).
        trades (pd.DataFrame): Output of simulate_trades, positions relative to df.
        init_cash (float): Starting equity (cfg.INIT_CASH).
        size (float): Fraction of equity committed per trade at m = 1 (cfg.SIZE).

    Returns:
        pd.Series: Equity indexed like df.
    """
    close = df["close"].to_numpy()
    equity = np.full(len(df), np.nan)
    cash = init_cash
    m_all = trades["size"].to_numpy() if "size" in trades else np.ones(len(trades))
    turnover, pos_bars, pos_sum = 0.0, 0, 0.0
    for (e, x, side, entry, exit_, exit_px), m in zip(
        trades[["entry_pos", "exit_pos", "side", "entry_fill", "exit_fill", "exit_px"]].itertuples(index=False), m_all
    ):
        equity[e - 1] = cash  # flat at the prior close, the entry happens at the next open
        qty = size * m * cash / entry
        equity[e:x] = cash + side * qty * (close[e:x] - entry)
        turnover += qty * entry / cash + qty * exit_ / (cash + side * qty * (exit_px - entry))
        cash += side * qty * (exit_ - entry)
        equity[x] = cash
        pos_bars, pos_sum = pos_bars + (x - e), pos_sum + m * (x - e)  # held at closes e … x−1
    equity[0] = init_cash if np.isnan(equity[0]) else equity[0]
    out = pd.Series(equity, index=df.index, name="equity").ffill()
    out.attrs["turnover"] = turnover
    out.attrs["avg_position"] = pos_sum / pos_bars if pos_bars else np.nan
    return out


def simulate_positions(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: RunConfig,
    side_col: str = "trade_signal",
    size_col: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Active-bet averaging (POSITION_MODE="average", López de Prado 2018 §10.4) with concurrent bets.

    Every event with a side and raw size m > 0 is a bet, live from its entry (open[t+1]) to its own barrier exit
    (barrier_exits, side-aware). The target exposure is f = discretize(mean of side·m over the live bets, SIZE_STEP)
    of equity (0 with no live bet). It is re-evaluated when bets start or end, and the position is traded only
    when f changes: target shares = f · SIZE · equity / fill, where equity is marked at the fill's reference price
    and the fill price carries adverse slippage (costs on every notional change, resizes included).
    Within a bar: first the open (entries, and exits at the open), then intrabar barrier exits in bet order, then
    vertical exits at the close (exits sharing a phase and price are one change). With non-overlapping bets this equals POSITION_MODE="single"
    without the skipping.

    Returns:
        tuple[pd.Series, pd.DataFrame]: Equity (close-marked; attrs "turnover", "exposure" = share of bars
            with a position at any point in the bar, as `bars_held` counts, "avg_position" = mean |f| over the
            closes at which a position is held) and the bets (per-unit `pnl_pct` as in simulate_trades, `size` = raw m).
    """
    cand = _candidates(signals, side_col, size_col, None)
    bets = _unit_pnl(_exits(df, cand, cfg).copy(), cand, cfg.SLIPPAGE_PCT)
    open_, close = df["open"].to_numpy(), df["close"].to_numpy()
    slip, step, size = cfg.SLIPPAGE_PCT, cfg.SIZE_STEP, cfg.SIZE

    # Per bar: (phase, price) → [Δ signed-size sum, Δ count]; phase 0 = the open (entries, gap exits),
    # 1 = intrabar barrier exits (bet order among them), 2 = vertical exits at the close — a feasible fill sequence
    changes: dict[int, dict[tuple[int, float], list[float]]] = {}
    sm = (bets["side"] * bets["size"]).to_numpy()
    for e, s in zip(bets["entry_pos"].to_numpy(), sm):
        g = changes.setdefault(e, {}).setdefault((0, open_[e]), [0.0, 0])
        g[0] += s
        g[1] += 1
    vertical = (bets["barrier"] == "vertical").to_numpy()
    for x, px, s, v in zip(bets["exit_pos"].to_numpy(), bets["exit_px"].to_numpy(), sm, vertical):
        phase = 2 if v else (0 if px == open_[x] else 1)
        g = changes.setdefault(x, {}).setdefault((phase, px), [0.0, 0])
        g[0] -= s
        g[1] -= 1

    equity = np.full(len(df), float(cfg.INIT_CASH))
    cash, q, f, total, count, prev = float(cfg.INIT_CASH), 0.0, 0.0, 0.0, 0, 0
    turnover, held, pos_bars, pos_sum = 0.0, 0, 0, 0.0
    for b in sorted(changes):
        equity[prev:b] = cash + q * close[prev:b]
        held += (b - prev) if q != 0 else 0
        if q != 0:
            pos_bars, pos_sum = pos_bars + (b - prev), pos_sum + abs(f) * (b - prev)
        in_bar = q != 0
        for (_, P), (d_sum, d_cnt) in sorted(changes[b].items(), key=lambda kv: kv[0][0]):  # stable: bet order
            total, count = total + d_sum, count + d_cnt
            if not count:
                total = 0.0  # no float residue from the running sum once every bet has ended
            # rounding to 1e-12 drops float residue of the running sum (e.g. 0.3 − 0.1 − 0.2) when SIZE_STEP = 0
            f_new = float(discretize(round(total / count, 12), step)) if count else 0.0
            if f_new == f:
                continue
            E = cash + q * P
            d = np.sign(f_new * size * E / P - q)
            fill = P * (1.0 + d * slip)
            q_new = f_new * size * E / fill
            cash -= (q_new - q) * fill
            turnover += abs(q_new - q) * fill / E
            q, f = q_new, f_new
            in_bar |= q != 0
        equity[b] = cash + q * close[b]
        held += in_bar
        if q != 0:
            pos_bars, pos_sum = pos_bars + 1, pos_sum + abs(f)
        prev = b + 1
    equity[prev:] = cash + q * close[prev:]
    held += (len(df) - prev) if q != 0 else 0
    out = pd.Series(equity, index=df.index, name="equity")
    out.attrs["turnover"] = turnover
    out.attrs["exposure"] = held / len(df)
    out.attrs["avg_position"] = pos_sum / pos_bars if pos_bars else np.nan
    return out, bets


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: RunConfig,
    size_col: str = "bet_size",
    spread_bars: pd.DataFrame | None = None,
) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """
    Backtest the meta-filtered, sized signals (`size_col`, cfg.POSITION_MODE) and, as the benchmark
    meta-labeling must beat, the unfiltered primary signal on the same events at m = 1. With an active
    cfg.RISK_PROFILE both go through the risk layer (risk.portfolio.simulate_portfolio, one symbol).

    Args:
        df (pd.DataFrame): OHLC data covering the OOS span.
        signals (pd.DataFrame): WFO output.
        cfg (RunConfig): Run configuration.
        size_col (str, optional): Bet-size column of the meta-filtered strategy. Defaults to "bet_size".
        spread_bars (pd.DataFrame | None, optional): 5Min bars of the symbol, with history before the OOS span,
            for the Corwin–Schultz half-spread (risk profiles with spread="cs"). Defaults to None.

    Returns:
        dict[str, tuple[pd.Series, pd.DataFrame]]: Strategy name → (equity, trades).
    """
    from risk.costs import half_spread
    from risk.portfolio import simulate_portfolio
    from risk.profiles import get_profile

    profile = get_profile(cfg.RISK_PROFILE)
    print(
        f"\n[BACKTEST]  Simulating barrier-exit trades (sizer {cfg.SIZER}, {cfg.POSITION_MODE} positions, "
        f"risk {profile.name}) ..."
    )
    hs = None
    if profile.spread == "cs":
        if spread_bars is None:
            raise ValueError(f"risk profile {profile.name!r} charges spreads: pass the symbol's 5Min spread_bars")
        hs = {"_": half_spread(df.index, spread_bars, profile.spread_window_days, profile.spread_floor)}
    results = {}
    for name, col, sz in (("Meta-filtered", "trade_signal", size_col), ("Primary only", "signed_dir", None)):
        if profile.active:
            eq, trades, _ = simulate_portfolio(
                {"_": df}, {"_": signals}, cfg, profile, side_col=col, size_col=sz, half_spreads=hs
            )
            results[name] = (eq, trades)
        elif cfg.POSITION_MODE == "single":
            trades = simulate_trades(df, signals, cfg, side_col=col, size_col=sz)
            results[name] = (equity_curve(df, trades, cfg.INIT_CASH, cfg.SIZE), trades)
        else:
            results[name] = simulate_positions(df, signals, cfg, side_col=col, size_col=sz)
    return results
