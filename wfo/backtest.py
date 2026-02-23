import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

from utils.config import config
from wfo.wfo_metrics import compute_metrics
from wfo.wfo_setup import apply_slippage, build_backtest_signals


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """
    Execute the WFO signals in VectorBT.

    Strategy:
      • Long/Short signals from meta-model
      • Entry: next-bar OPEN (shift by 1)
      • Exit:  signal reversal or close (vertical barrier proxy)
      • Slippage: 1 bp one-way, zero commission

    Returns (portfolio, metrics_df).
    """
    print("\n[BACKTEST]  Building VectorBT portfolio ...")

    entries, exits, direction = build_backtest_signals(df, signals)

    # Next-bar open execution
    exec_price = df["open"].shift(-1).ffill()
    slipped = apply_slippage(exec_price, direction.shift(1).fillna(0))

    # Separate long and short signals
    long_entries = entries & (direction > 0)
    short_entries = entries & (direction < 0)
    long_exits = exits | (direction < 0)  # exit long on short signal too
    short_exits = exits | (direction > 0)

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        price=slipped,
        init_cash=config.INIT_CASH,
        size=config.SIZE,
        size_type=SizeType.Percent,  # size as % of equity
        upon_long_conflict="ignore",
        upon_short_conflict="ignore",
        freq="5T",
    )

    metrics = compute_metrics(pf)
    return pf, metrics
