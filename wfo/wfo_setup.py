import numpy as np
import pandas as pd

from utils.config import config


def build_backtest_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convert the WFO trade_signal column into VectorBT-compatible entry/exit
    boolean arrays with next-bar open execution and slippage.

    Execution logic:
      • Signal at bar t → execute at OPEN of bar t+1
      • Long:  entry at open[t+1] x (1 + SLIPPAGE_PCT)
      • Short: entry at open[t+1] x (1 - SLIPPAGE_PCT)
      • Exit:  when trade_signal changes sign or goes to 0 (or VERTICAL_BARS)

    Args:
        df (pd.DataFrame): Original dataset
        signals (pd.DataFrame): WFO trade signal

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: entries, exits, direction. All pd.Series aligned to df.index
    """
    sig = signals["trade_signal"].reindex(df.index).fillna(0)

    # Entries: signal transitions from 0 to non-zero (or sign flip)
    prev = sig.shift(1).fillna(0)
    entries = (sig != 0) & ((prev == 0) | (np.sign(sig) != np.sign(prev)))

    # Exits: signal goes to 0 OR reverses sign
    exits = (sig == 0) & (prev != 0)
    # Also exit on sign flip (entry handles the new position)
    sign_flip = (sig != 0) & (prev != 0) & (np.sign(sig) != np.sign(prev))
    exits = exits | sign_flip

    direction = np.sign(sig)

    return entries, exits, direction


def apply_slippage(price: pd.Series, direction: pd.Series, pct: float = config.SLIPPAGE_PCT) -> pd.Series:
    """
    Apply directional slippage to a price series.
    Long entries are filled at price x (1 + pct),
    short entries at price x (1 - pct).

    Args:
        price (pd.Series): Price series
        direction (pd.Series): Direction of slippage
        pct (float, optional): Slippage percentage. Defaults to config.SLIPPAGE_PCT.

    Returns:
        pd.Series: Applied slippage price.
    """
    adj = price.copy()
    adj[direction > 0] *= 1.0 + pct
    adj[direction < 0] *= 1.0 - pct
    return adj
