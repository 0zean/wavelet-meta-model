"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SPY 5-Minute Strategy Framework                                             ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Pipeline:                                                                   ║
║    1. Fractional Differencing  (fracdiff, fit-on-train-only)                 ║
║    2. Causal MODWT features    (db1/Haar, J=4, S4 AR lags)                   ║
║    3. Technical features       (VWAP, RSI, Siegel slope)                     ║
║    4. Triple-Barrier labeling  (López de Prado 2018)                         ║
║    5. Dual primary models      (XGB classifier + XGB regressor)              ║
║    6. Meta-labeling            (XGB classifier on primary OOF preds)         ║
║    7. Expanding-window WFO     (walk-forward optimization engine)            ║
║    8. VectorBT backtest        (next-bar open execution + slippage)          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

from utils.data_loader import load_ohlcv, make_synthetic_spy
from utils.visuals import plot_results
from wfo.backtest import run_backtest
from wfo.wfo_engine import run_wfo


def main(data_path: str | None = None) -> None:
    """
    Main entry point to run the full pipeline end-to-end.

    Args:
        data_path (str | None, optional): Path to a CSV with OHLCV data.
        If None, synthetic SPY data is used. Defaults to None.
    """
    print("╔" + "═" * 58 + "╗")
    print("║  SPY 5-Minute Strategy  |  Fracdiff + MODWT + Meta-Label ║")
    print("╚" + "═" * 58 + "╝\n")

    # Load / generate data
    if data_path and Path(data_path).exists():
        df = load_ohlcv(data_path)
    else:
        print("[DATA]  No data path provided — using synthetic SPY (n=5000)")
        df = make_synthetic_spy(n=5000)

    # Walk-forward optimisation
    signals = run_wfo(df)

    # VectorBT backtest
    pf, metrics = run_backtest(df, signals)

    # Visualise
    plot_results(df, pf, signals)

    # Persist signals
    out_path = "results/wfo_signals.csv"
    signals.to_csv(out_path)
    print(f"[OUT]   WFO signals saved to {out_path}")

    return pf, metrics, signals


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
