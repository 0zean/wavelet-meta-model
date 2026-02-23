import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from utils.config import config


def plot_results(
    df: pd.DataFrame,
    pf: vbt.Portfolio,
    signals: pd.DataFrame,
    save_to: str = "results/strategy_results.png",
) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Equity curve vs Buy-and-hold
    ax1 = fig.add_subplot(gs[0, :])
    equity = pf.value()
    bh = config.INIT_CASH * (df["close"] / df["close"].iloc[0])
    equity.reindex(df.index).plot(ax=ax1, label="Strategy", color="steelblue", lw=1.5)
    bh.plot(ax=ax1, label="Buy-and-Hold", color="orange", lw=1.0, linestyle="--")
    ax1.set_title("Equity Curve vs Buy-and-Hold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    pf.drawdown().reindex(df.index).plot(ax=ax2, color="red", lw=1.0)
    ax2.fill_between(df.index, pf.drawdown().reindex(df.index), 0, alpha=0.2, color="red")
    ax2.set_title("Drawdown", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    # 3. Trade signal distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sig_counts = signals["trade_signal"].value_counts().sort_index()
    sig_counts.plot(kind="bar", ax=ax3, color=["red", "gray", "steelblue"])
    ax3.set_title("Trade Signal Distribution", fontsize=11)
    ax3.set_xlabel("Signal (-1=Short, 0=No Trade, 1=Long)")
    ax3.set_ylabel("Count")
    ax3.grid(alpha=0.3, axis="y")

    # 4. Rolling Sharpe (63-bar ≈ 1 week of 5m bars)
    ax4 = fig.add_subplot(gs[2, 0])
    roll_ret = pf.returns().reindex(df.index).fillna(0)
    roll_sharpe = (roll_ret.rolling(78 * 5).mean() / (roll_ret.rolling(78 * 5).std() + 1e-9)) * np.sqrt(78 * 252)
    roll_sharpe.plot(ax=ax4, color="green", lw=1.0)
    ax4.axhline(0, color="black", lw=0.5, linestyle="--")
    ax4.set_title("Rolling 1-Week Sharpe", fontsize=11)
    ax4.grid(alpha=0.3)

    # ── 5. Meta-model confidence histogram ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    if "meta_prob" in signals.columns:
        signals["meta_prob"].hist(ax=ax5, bins=40, color="purple", alpha=0.7)
    ax5.axvline(config.META_THRESH, color="red", lw=1.5, linestyle="--", label=f"Threshold={config.META_THRESH}")
    ax5.set_title("Meta-Model Confidence Distribution", fontsize=11)
    ax5.set_xlabel("P(trade is profitable)")
    ax5.legend()
    ax5.grid(alpha=0.3)

    plt.suptitle("SPY 5-Minute Strategy: Full WFO Results", fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT]  Saved to {save_to}")
    plt.close(fig)
