import numpy as np
import pandas as pd

from features.triple_barrier_labels import barrier_exits
from utils.config import config


def simulate_trades(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    side_col: str = "trade_signal",
    slippage: float = config.SLIPPAGE_PCT,
) -> pd.DataFrame:
    """
    Execute event signals with the same triple-barrier rules used for labeling.

    Execution logic (see barrier_exits):
      • Signal at event bar t → enter at OPEN of bar t+1
      • Exit at the first barrier touched (adverse barrier on same-bar ties),
        else at the close of the vertical barrier / session close
      • One position at a time: events arriving while a position is open are skipped
      • Slippage is adverse on every fill: buys at price x (1 + pct), sells at price x (1 - pct)

    Args:
        df (pd.DataFrame): OHLC data.
        signals (pd.DataFrame): WFO output with `width` and a side column.
        side_col (str, optional): Column holding the side {-1, 0, +1}. Defaults to "trade_signal".
        slippage (float, optional): One-way slippage. Defaults to config.SLIPPAGE_PCT.

    Returns:
        pd.DataFrame: One row per executed trade, indexed by signal time.
    """
    cand = signals[signals[side_col] != 0]
    exits = barrier_exits(df, cand.index, cand["width"], side=cand[side_col])

    # Greedy non-overlapping selection: a new entry at open[t+1] needs the previous exit at or before bar t
    taken = np.zeros(len(exits), dtype=bool)
    last_exit = -1
    for i, (entry, exit_) in enumerate(zip(exits["entry_pos"].to_numpy(), exits["exit_pos"].to_numpy())):
        if entry > last_exit:
            taken[i] = True
            last_exit = exit_

    trades = exits[taken].copy()
    trades["side"] = cand.loc[trades.index, side_col].astype(int)
    trades["entry_fill"] = trades["entry_px"] * (1.0 + trades["side"] * slippage)
    trades["exit_fill"] = trades["exit_px"] * (1.0 - trades["side"] * slippage)
    trades["pnl_pct"] = trades["side"] * (trades["exit_fill"] / trades["entry_fill"] - 1.0)
    trades["bars_held"] = trades["exit_pos"] - trades["entry_pos"] + 1
    return trades


def equity_curve(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    init_cash: float = config.INIT_CASH,
    size: float = config.SIZE,
) -> pd.Series:
    """
    Bar-by-bar mark-to-market equity: open positions are valued at each close,
    and at the exit bar at the exit fill.

    Args:
        df (pd.DataFrame): OHLC data (the span to report).
        trades (pd.DataFrame): Output of simulate_trades, positions relative to df.
        init_cash (float, optional): Starting equity. Defaults to config.INIT_CASH.
        size (float, optional): Fraction of equity committed per trade. Defaults to config.SIZE.

    Returns:
        pd.Series: Equity indexed like df.
    """
    close = df["close"].to_numpy()
    equity = np.full(len(df), np.nan)
    cash = init_cash
    for e, x, side, entry, exit_ in trades[["entry_pos", "exit_pos", "side", "entry_fill", "exit_fill"]].itertuples(
        index=False
    ):
        equity[e - 1] = cash  # flat at the prior close, the entry happens at the next open
        qty = size * cash / entry
        equity[e:x] = cash + side * qty * (close[e:x] - entry)
        cash += side * qty * (exit_ - entry)
        equity[x] = cash
    equity[0] = init_cash if np.isnan(equity[0]) else equity[0]
    return pd.Series(equity, index=df.index, name="equity").ffill()


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """
    Backtest the meta-filtered signals and, as the benchmark meta-labeling must
    beat, the unfiltered primary signal on the same events.

    Args:
        df (pd.DataFrame): OHLC data covering the OOS span.
        signals (pd.DataFrame): WFO output.

    Returns:
        dict[str, tuple[pd.Series, pd.DataFrame]]: Strategy name → (equity, trades).
    """
    print("\n[BACKTEST]  Simulating barrier-exit trades ...")
    results = {}
    for name, col in (("Meta-filtered", "trade_signal"), ("Primary only", "signed_dir")):
        trades = simulate_trades(df, signals, side_col=col)
        results[name] = (equity_curve(df, trades), trades)
    return results
