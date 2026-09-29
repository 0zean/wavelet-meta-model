import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features.triple_barrier_labels import barrier_exits
from models.zoo import PROB_CLIP
from utils.config import RunConfig
from validation.scoring import brier, neg_log_loss


def strategy_metrics(equity: pd.Series, trades: pd.DataFrame, bars_per_year: int) -> pd.Series:
    """
    Risk-adjusted performance of one equity curve over its own span.

    Args:
        equity (pd.Series): Bar-level equity.
        trades (pd.DataFrame): Executed trades (empty for buy-and-hold).
        bars_per_year (int): Annualisation factor (cfg.bars_per_year).

    Returns:
        pd.Series: Metrics.
    """
    rets = equity.pct_change().fillna(0.0)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (bars_per_year / max(len(rets), 1)) - 1
    vol = rets.std() * np.sqrt(bars_per_year)
    max_dd = (equity / equity.cummax() - 1).min()
    n_trades = len(trades)
    return pd.Series(
        {
            "Total Return (%)": total_ret * 100,
            "Annual Return (%)": ann_ret * 100,
            "Annual Volatility (%)": vol * 100,
            "Sharpe Ratio": rets.mean() * bars_per_year / (vol + 1e-12),
            "Max Drawdown (%)": max_dd * 100,
            "Calmar Ratio": ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
            "Num Trades": n_trades,
            "Win Rate (%)": (trades["pnl_pct"] > 0).mean() * 100 if n_trades else np.nan,
            "Avg Trade (bp)": trades["pnl_pct"].mean() * 1e4 if n_trades else np.nan,
            "Avg Bars Held": trades["bars_held"].mean() if n_trades else np.nan,
            "Exposure (%)": trades["bars_held"].sum() / len(equity) * 100 if n_trades else 0.0,
        }
    )


def compute_metrics(
    df: pd.DataFrame, results: dict[str, tuple[pd.Series, pd.DataFrame]], cfg: RunConfig
) -> pd.DataFrame:
    """
    Metrics table for each strategy plus buy-and-hold, all over the same OOS span.

    Args:
        df (pd.DataFrame): OHLC data covering the OOS span.
        results (dict): Output of run_backtest.
        cfg (RunConfig): Run configuration.

    Returns:
        pd.DataFrame: One column per strategy.
    """
    cols = {name: strategy_metrics(eq, trades, cfg.bars_per_year) for name, (eq, trades) in results.items()}
    bh = cfg.INIT_CASH * df["close"] / df["close"].iloc[0]
    cols["Buy-and-Hold"] = strategy_metrics(bh, pd.DataFrame(columns=["pnl_pct", "bars_held"]), cfg.bars_per_year)
    metrics = pd.DataFrame(cols)

    print("\n" + "═" * 70)
    print("  BACKTEST PERFORMANCE SUMMARY (OOS span)")
    print("═" * 70)
    print(metrics.round(3).to_string())
    print("═" * 70)
    return metrics


def meta_outcomes(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    Post-hoc meta-label of every OOS event (1 = the primary's side beat META_MIN_RET, same barrier rules as the
    backtest) next to its meta_prob. Events of folds whose meta-model was skipped (signals.attrs
    "meta_skipped_folds") are flagged `scored=False`: their meta_prob is a placeholder 0, not a prediction.
    """
    out = barrier_exits(
        df,
        signals.index,
        signals["width"],
        side=signals["signed_dir"],
        vertical_bars=cfg.VERTICAL_BARS,
        hold_overnight=cfg.HOLD_OVERNIGHT,
    )
    sig = signals.loc[out.index]
    skipped = signals.attrs.get("meta_skipped_folds", [])
    return pd.DataFrame(
        {
            "success": (sig["signed_dir"] * out["ret"] > cfg.META_MIN_RET).astype(int),
            "meta_prob": sig["meta_prob"],
            "approved": sig["trade_signal"] != 0,
            "ret": sig["signed_dir"] * out["ret"],
            "fold": sig["fold"],
            "scored": ~sig["fold"].isin(skipped),
        },
        index=out.index,
    )


def calibration_table(success: pd.Series, prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve: equal-width probability bins with count, mean predicted p and observed success rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    g = pd.DataFrame({"bin": b, "p": prob.to_numpy(), "y": success.to_numpy()}).groupby("bin")
    t = g.agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
    t.insert(0, "lo", edges[t.index])
    t.insert(1, "hi", edges[t.index + 1])
    return t.reset_index(drop=True)


def signal_diagnostics(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """
    Post-hoc OOS skill of the primary and meta models on every test event
    (evaluation only — outcomes are computed after all predictions are fixed).
    Meta log-loss / Brier / AUC use the events of folds with a fitted meta-model (probabilities clipped
    to [PROB_CLIP, 1 − PROB_CLIP] for the log-loss, as the zoo's calibrated output is).

    Args:
        df (pd.DataFrame): OHLC data.
        signals (pd.DataFrame): WFO output.
        cfg (RunConfig): Run configuration.

    Returns:
        pd.Series: Primary hit rate, meta log-loss, Brier, AUC and precision.
    """
    o = meta_outcomes(df, signals, cfg)
    s = o[o["scored"]]
    two = s["success"].nunique() == 2
    p = np.clip(s["meta_prob"], PROB_CLIP, 1 - PROB_CLIP)
    diag = pd.Series(
        {
            "OOS events": len(o),
            "Primary success rate": o["success"].mean(),
            "Meta-scored events": len(s),
            "Meta log-loss": -neg_log_loss(s["success"], p) if len(s) else np.nan,
            "Meta Brier": brier(s["success"], s["meta_prob"]) if len(s) else np.nan,
            "Meta AUC": roc_auc_score(s["success"], s["meta_prob"]) if two else np.nan,
            "Approved share": o["approved"].mean(),
            "Approved success rate": o.loc[o["approved"], "success"].mean() if o["approved"].any() else np.nan,
        }
    )
    print("\n[DIAG]  OOS signal quality")
    print(diag.round(4).to_string())
    return diag
