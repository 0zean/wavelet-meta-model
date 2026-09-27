from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from utils.config import RunConfig


def plot_results(
    df: pd.DataFrame,
    results: dict[str, tuple[pd.Series, pd.DataFrame]],
    signals: pd.DataFrame,
    cfg: RunConfig,
    save_to: str = "results/strategy_results.png",
) -> None:
    roll_bars = 5 * cfg.BARS_PER_DAY  # 1 week
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    colors = {"Meta-filtered": "steelblue", "Primary only": "gray"}
    meta_equity = results["Meta-filtered"][0]

    # 1. Equity curves vs Buy-and-hold
    ax1 = fig.add_subplot(gs[0, :])
    for name, (equity, _) in results.items():
        equity.plot(ax=ax1, label=name, color=colors.get(name), lw=1.5)
    bh = cfg.INIT_CASH * (df["close"] / df["close"].iloc[0])
    bh.plot(ax=ax1, label="Buy-and-Hold", color="orange", lw=1.0, linestyle="--")
    ax1.set_title("OOS Equity Curve vs Buy-and-Hold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Drawdown (meta-filtered)
    ax2 = fig.add_subplot(gs[1, 0])
    dd = (meta_equity / meta_equity.cummax() - 1) * 100
    dd.plot(ax=ax2, color="red", lw=1.0)
    ax2.fill_between(dd.index, dd, 0, alpha=0.2, color="red")
    ax2.set_title("Drawdown (Meta-filtered)", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    # 3. Trade signal distribution over OOS events
    ax3 = fig.add_subplot(gs[1, 1])
    sig_counts = signals["trade_signal"].value_counts().reindex([-1, 0, 1], fill_value=0)
    sig_counts.plot(kind="bar", ax=ax3, color=["red", "gray", "steelblue"])
    ax3.set_title("Trade Signal Distribution (OOS events)", fontsize=11)
    ax3.set_xlabel("Signal (-1=Short, 0=Rejected, 1=Long)")
    ax3.set_ylabel("Count")
    ax3.grid(alpha=0.3, axis="y")

    # 4. Rolling 1-week Sharpe
    ax4 = fig.add_subplot(gs[2, 0])
    roll_ret = meta_equity.pct_change().fillna(0)
    roll_sharpe = roll_ret.rolling(roll_bars).mean() / (roll_ret.rolling(roll_bars).std() + 1e-9)
    (roll_sharpe * np.sqrt(cfg.bars_per_year)).plot(ax=ax4, color="green", lw=1.0)
    ax4.axhline(0, color="black", lw=0.5, linestyle="--")
    ax4.set_title("Rolling 1-Week Sharpe (Meta-filtered)", fontsize=11)
    ax4.grid(alpha=0.3)

    # 5. Meta-model confidence histogram
    ax5 = fig.add_subplot(gs[2, 1])
    signals["meta_prob"].hist(ax=ax5, bins=40, color="purple", alpha=0.7)
    ax5.axvline(cfg.META_THRESH, color="red", lw=1.5, linestyle="--", label=f"Threshold={cfg.META_THRESH}")
    ax5.set_title("Meta-Model Confidence Distribution", fontsize=11)
    ax5.set_xlabel("P(trade is profitable)")
    ax5.legend()
    ax5.grid(alpha=0.3)

    plt.suptitle(f"{cfg.TIMEFRAME} Strategy: Full WFO Results", fontsize=15, fontweight="bold", y=1.01)
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT]  Saved to {save_to}")
    plt.close(fig)
