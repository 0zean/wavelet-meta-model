import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features.triple_barrier_labels import barrier_exits
from utils.config import config

BARS_PER_YEAR = 252 * config.BARS_PER_DAY


def strategy_metrics(equity: pd.Series, trades: pd.DataFrame) -> pd.Series:
    """
    Risk-adjusted performance of one equity curve over its own span.

    Args:
        equity (pd.Series): Bar-level equity.
        trades (pd.DataFrame): Executed trades (empty for buy-and-hold).

    Returns:
        pd.Series: Metrics.
    """
    rets = equity.pct_change().fillna(0.0)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (BARS_PER_YEAR / max(len(rets), 1)) - 1
    vol = rets.std() * np.sqrt(BARS_PER_YEAR)
    max_dd = (equity / equity.cummax() - 1).min()
    n_trades = len(trades)
    return pd.Series(
        {
            "Total Return (%)": total_ret * 100,
            "Annual Return (%)": ann_ret * 100,
            "Annual Volatility (%)": vol * 100,
            "Sharpe Ratio": rets.mean() * BARS_PER_YEAR / (vol + 1e-12),
            "Max Drawdown (%)": max_dd * 100,
            "Calmar Ratio": ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
            "Num Trades": n_trades,
            "Win Rate (%)": (trades["pnl_pct"] > 0).mean() * 100 if n_trades else np.nan,
            "Avg Trade (bp)": trades["pnl_pct"].mean() * 1e4 if n_trades else np.nan,
            "Avg Bars Held": trades["bars_held"].mean() if n_trades else np.nan,
            "Exposure (%)": trades["bars_held"].sum() / len(equity) * 100 if n_trades else 0.0,
        }
    )


def compute_metrics(df: pd.DataFrame, results: dict[str, tuple[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    """
    Metrics table for each strategy plus buy-and-hold, all over the same OOS span.

    Args:
        df (pd.DataFrame): OHLC data covering the OOS span.
        results (dict): Output of run_backtest.

    Returns:
        pd.DataFrame: One column per strategy.
    """
    cols = {name: strategy_metrics(eq, trades) for name, (eq, trades) in results.items()}
    bh = config.INIT_CASH * df["close"] / df["close"].iloc[0]
    cols["Buy-and-Hold"] = strategy_metrics(bh, pd.DataFrame(columns=["pnl_pct", "bars_held"]))
    metrics = pd.DataFrame(cols)

    print("\n" + "═" * 70)
    print("  BACKTEST PERFORMANCE SUMMARY (OOS span)")
    print("═" * 70)
    print(metrics.round(3).to_string())
    print("═" * 70)
    return metrics


def signal_diagnostics(df: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    """
    Post-hoc OOS skill of the primary and meta models on every test event
    (evaluation only — outcomes are computed after all predictions are fixed).

    Args:
        df (pd.DataFrame): OHLC data.
        signals (pd.DataFrame): WFO output.

    Returns:
        pd.Series: Primary hit rate, meta AUC and meta precision.
    """
    out = barrier_exits(df, signals.index, signals["width"], side=signals["signed_dir"])
    sig = signals.loc[out.index]
    success = (sig["signed_dir"] * out["ret"] > config.META_MIN_RET).astype(int)
    approved = sig["trade_signal"] != 0
    diag = pd.Series(
        {
            "OOS events": len(sig),
            "Primary success rate": success.mean(),
            "Meta AUC": roc_auc_score(success, sig["meta_prob"]) if success.nunique() == 2 else np.nan,
            "Approved share": approved.mean(),
            "Approved success rate": success[approved].mean() if approved.any() else np.nan,
        }
    )
    print("\n[DIAG]  OOS signal quality")
    print(diag.round(4).to_string())
    return diag
