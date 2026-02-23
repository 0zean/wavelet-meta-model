import numpy as np
import pandas as pd
import vectorbt as vbt


def compute_metrics(pf: vbt.Portfolio) -> pd.DataFrame:
    """
    Compute a comprehensive set of risk-adjusted performance metrics.

    Args:
        pf (vbt.Portfolio): VectorBT portfolio object.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics.
    """
    rets = pf.returns()
    ann = 252 * 78  # bars per year (252 days × 78 five-min bars)

    total_ret = pf.total_return()
    ann_ret = (1 + total_ret) ** (ann / max(len(rets), 1)) - 1
    vol = rets.std() * np.sqrt(ann)
    sharpe = (rets.mean() * ann - 0.0) / (rets.std() * np.sqrt(ann) + 1e-9)
    max_dd = pf.max_drawdown()
    calmar = ann_ret / abs(max_dd + 1e-9)
    win_rate = (rets > 0).mean()
    n_trades = pf.trades.count()
    trade_dur = pf.trades.duration.mean() if n_trades > 0 else np.nan

    metrics = pd.DataFrame(
        {
            "Total Return (%)": [total_ret * 100],
            "Annual Return (%)": [ann_ret * 100],
            "Annual Volatility (%)": [vol * 100],
            "Sharpe Ratio": [sharpe],
            "Max Drawdown (%)": [max_dd * 100],
            "Calmar Ratio": [calmar],
            "Win Rate (%)": [win_rate * 100],
            "Num Trades": [n_trades],
            "Avg Trade Duration": [trade_dur],
        }
    ).T
    metrics.columns = ["Value"]

    print("\n" + "═" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("═" * 50)
    print(metrics.to_string())
    print("═" * 50)
    return metrics
