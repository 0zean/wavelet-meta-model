"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SPY 5-Minute Strategy Framework                                             ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Pipeline:                                                                   ║
║    1. CUSUM event sampling     (σ-scaled threshold, causal)                  ║
║    2. Triple-Barrier labeling  (López de Prado 2018, entry at next open)     ║
║    3. Fractional Differencing  (fracdiff, d fit-on-train-only)               ║
║    4. Causal MODWT features    (db1/Haar, J=4, S4 AR lags)                   ║
║    5. Technical features       (VWAP, RSI, Siegel slope)                     ║
║    6. Dual primary models      (XGB classifier + XGB regressor)              ║
║    7. Meta-labeling            (XGB classifier on primary OOF preds)         ║
║    8. Expanding-window WFO     (purged, uniqueness-weighted)                 ║
║    9. Barrier-exit backtest    (next-bar open execution + slippage)          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

import pandas as pd

from utils.data_loader import load_ohlcv, make_synthetic_spy
from utils.visuals import plot_results
from wfo.backtest import run_backtest
from wfo.wfo_engine import run_wfo
from wfo.wfo_metrics import compute_metrics, signal_diagnostics


def main(data_path: str | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point to run the full pipeline end-to-end.

    Args:
        data_path (str | None, optional): Path to a CSV with OHLCV data.
        If None, synthetic SPY data is used. Defaults to None.

    Returns:
        tuple[dict, pd.DataFrame, pd.DataFrame]: Backtest results, metrics, WFO signals.
    """
    print("╔" + "═" * 58 + "╗")
    print("║  SPY 5-Minute Strategy  |  Fracdiff + MODWT + Meta-Label ║")
    print("╚" + "═" * 58 + "╝\n")

    # Load / generate data
    if data_path:
        if not Path(data_path).exists():
            raise FileNotFoundError(data_path)
        df = load_ohlcv(data_path)
    else:
        print("[DATA]  No data path provided — using synthetic SPY (n=5000)")
        df = make_synthetic_spy(n=5000)

    # Walk-forward optimisation
    signals = run_wfo(df)

    # Persist signals first so a later failure doesn't lose the WFO run
    out_path = Path("results/wfo_signals.csv")
    out_path.parent.mkdir(exist_ok=True)
    signals.to_csv(out_path)
    print(f"[OUT]   WFO signals saved to {out_path}")

    # Backtest, metrics and plots cover only the OOS span
    df_oos = df.loc[signals.index[0] :]
    results = run_backtest(df_oos, signals)
    metrics = compute_metrics(df_oos, results)
    signal_diagnostics(df, signals)
    plot_results(df_oos, results, signals)

    return results, metrics, signals


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
