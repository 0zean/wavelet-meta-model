"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Wavelet Meta-Labeling Strategy Framework (1Min … 1Day)                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Pipeline:                                                                   ║
║    1. CUSUM event sampling     (σ-scaled threshold, causal)                  ║
║    2. Triple-Barrier labeling  (López de Prado 2018, entry at next open)     ║
║    3. Fractional Differencing  (fracdiff, d fit-on-train-only)               ║
║    4. Feature zoo              (registry; causal MODWT S_J core required)    ║
║    5. Feature selection        (optional clustered MDA, train fold only)     ║
║    6. Primary signal zoo       (XGB clf + reg, or a fixed rule; --primary)   ║
║    7. Meta-labeling            (XGB classifier on primary OOF preds)         ║
║    8. Expanding-window WFO     (purged + embargoed, uniqueness-weighted)     ║
║    9. Barrier-exit backtest    (next-bar open execution + slippage)          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import pandas as pd

from data.bars import load_bars
from features import cache as feature_cache
from models.zoo import REGISTRY as ZOO
from primaries import REGISTRY as PRIMARIES
from primaries.diagnostics import primary_diagnostics
from utils.config import RunConfig
from utils.data_loader import load_ohlcv, make_synthetic_spy
from utils.visuals import plot_results
from wfo.backtest import run_backtest
from wfo.wfo_engine import run_wfo
from wfo.wfo_metrics import calibration_table, compute_metrics, meta_outcomes, signal_diagnostics


def main(
    data_path: str | None = None,
    symbol: str | None = None,
    timeframe: str = "5Min",
    start: str | None = None,
    end: str | None = None,
    cfg: RunConfig | None = None,
    out_dir: str = "results",
    market_symbol: str = "SPY",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point to run the full pipeline end-to-end.

    Data source, in priority order: a CSV at `data_path`, Alpaca bars for `symbol`
    (cached, RTH only), else synthetic SPY.

    Without `cfg`, CSV and synthetic data run with RunConfig.legacy_5min() (the
    pre-U2 behaviour) and Alpaca data with RunConfig.for_timeframe(timeframe).

    Args:
        data_path (str | None, optional): Path to a CSV with OHLCV data. Defaults to None.
        symbol (str | None, optional): Ticker to load from Alpaca. Defaults to None.
        timeframe (str, optional): Alpaca bar timeframe. Defaults to "5Min".
        start (str | None, optional): First NY trading day (Alpaca only).
        end (str | None, optional): Day after the last NY trading day (Alpaca only).
        cfg (RunConfig | None, optional): Run configuration. Defaults to None (see above).
        out_dir (str, optional): Directory for signals and plots. Defaults to "results".
        market_symbol (str, optional): Market bars for the 'cross_asset' feature group. Defaults to "SPY".

    Returns:
        tuple[dict, pd.DataFrame, pd.DataFrame]: Backtest results, metrics, WFO signals.
    """
    print("╔" + "═" * 58 + "╗")
    print("║  Wavelet Strategy  |  Fracdiff + MODWT + Meta-Label      ║")
    print("╚" + "═" * 58 + "╝\n")

    # Load / generate data
    if data_path:
        if not Path(data_path).exists():
            raise FileNotFoundError(data_path)
        if cfg is None and timeframe != "5Min":
            raise ValueError("CSV input runs with RunConfig.legacy_5min(); pass a cfg for other timeframes")
        df = load_ohlcv(data_path)
        cfg = cfg or RunConfig.legacy_5min()
    elif symbol:
        if not (start and end):
            raise ValueError("--start and --end are required with --symbol")
        df = load_bars(symbol, timeframe, start, end)
        if df.empty:
            raise ValueError(f"No {timeframe} bars for {symbol} in [{start}, {end})")
        print(f"[DATA]  Alpaca {symbol} {timeframe}: {len(df):,} bars  {df.index[0]} → {df.index[-1]}")
        cfg = cfg or RunConfig.for_timeframe(timeframe)
        if cfg.TIMEFRAME != timeframe:
            raise ValueError(f"cfg is for {cfg.TIMEFRAME} but the data is {timeframe}")
    else:
        print("[DATA]  No data path provided — using synthetic SPY (n=5000)")
        df = make_synthetic_spy(n=5000)
        cfg = cfg or RunConfig.legacy_5min()
    print(
        f"[CFG]   {cfg.TIMEFRAME}  vertical={cfg.VERTICAL_BARS} bars  hold_overnight={cfg.HOLD_OVERNIGHT}  "
        f"windows={cfg.INITIAL_TRAIN}/{cfg.VAL}/{cfg.TEST} {cfg.WINDOW_UNIT}  embargo={cfg.EMBARGO}"
    )

    print(f"[CFG]   features={cfg.FEATURE_GROUPS or 'legacy'}  selection={cfg.FEATURE_SELECTION}")
    print(f"[CFG]   primary={cfg.PRIMARY} {cfg.PRIMARY_PARAMS or ''}")
    print(f"[CFG]   models: meta={cfg.META_MODEL} primary={cfg.PRIMARY_MODEL}  meta_train={cfg.META_TRAIN}")

    # Feature context (cross-asset market bars) and cache (Alpaca data only: the key needs a symbol)
    context = {}
    if cfg.FEATURE_GROUPS and "cross_asset" in cfg.FEATURE_GROUPS:
        if not symbol or symbol == market_symbol:
            raise ValueError(f"'cross_asset' needs an Alpaca --symbol other than the market ({market_symbol})")
        context["market"] = load_bars(market_symbol, timeframe, start, end)

    # Walk-forward optimisation
    signals = run_wfo(
        df,
        cfg,
        context=context,
        symbol=symbol if not data_path else None,
        feature_cache_dir=feature_cache.DEFAULT_ROOT if symbol and not data_path else None,
    )

    # Persist signals first so a later failure doesn't lose the WFO run
    out_path = Path(out_dir) / "wfo_signals.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path)
    # meta_prob is a placeholder 0 in folds whose meta-model was skipped: the CSV alone cannot tell them apart
    (Path(out_dir) / "wfo_run.json").write_text(json.dumps({k: list(v) for k, v in signals.attrs.items()}))
    print(f"[OUT]   WFO signals saved to {out_path} (skipped folds in wfo_run.json)")

    # Backtest, metrics and plots cover only the OOS span
    df_oos = df.loc[signals.index[0] :]
    results = run_backtest(df_oos, signals, cfg)
    metrics = compute_metrics(df_oos, results, cfg)
    signal_diagnostics(df, signals, cfg)
    o = meta_outcomes(df, signals, cfg)
    o = o[o["scored"]]
    calib = calibration_table(o["success"], o["meta_prob"])
    calib.to_csv(Path(out_dir) / "meta_calibration.csv", index=False)
    print("\n[DIAG]  Meta calibration (OOS)")
    print(calib.round(4).to_string(index=False))
    diag = primary_diagnostics(df, signals, cfg, symbol=symbol or "")
    diag.to_csv(Path(out_dir) / "primary_diagnostics.csv", index=False)
    print("\n[DIAG]  Primary diagnostics (all folds)")
    print(diag.iloc[-1].drop(["symbol", "timeframe", "fold"]).to_string())
    plot_results(df_oos, results, signals, cfg, save_to=str(Path(out_dir) / "strategy_results.png"))

    return results, metrics, signals


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", nargs="?", default=None, help="OHLCV CSV (overrides --symbol)")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe", default="5Min")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="results", help="output directory for signals and plots")
    ap.add_argument("--features", help="comma-separated feature groups (default: RunConfig default zoo)")
    ap.add_argument("--select", choices=["none", "cmda"], help="per-fold feature selection")
    ap.add_argument("--primary", choices=sorted(PRIMARIES), help="primary signal (default ml_xgb)")
    ap.add_argument("--primary-params", help='fixed rule parameters as JSON, e.g. \'{"fast": 10, "slow": 40}\'')
    ap.add_argument("--meta-model", choices=sorted(ZOO), help="meta-model (models/zoo.py; default legacy)")
    ap.add_argument("--primary-model", choices=sorted(ZOO), help="ml_xgb direction classifier (default legacy)")
    ap.add_argument("--meta-train", choices=["val", "oof"], help="meta-model training rows (default val)")
    a = ap.parse_args()
    overrides = {}
    if a.features:
        overrides["FEATURE_GROUPS"] = tuple(g.strip() for g in a.features.split(","))
    if a.select:
        overrides["FEATURE_SELECTION"] = a.select
    if a.primary:
        overrides["PRIMARY"] = a.primary
    if a.primary_params:
        overrides["PRIMARY_PARAMS"] = json.loads(a.primary_params)
    for arg, name in (("meta_model", "META_MODEL"), ("primary_model", "PRIMARY_MODEL"), ("meta_train", "META_TRAIN")):
        if getattr(a, arg):
            overrides[name] = getattr(a, arg)
    cfg = None
    if overrides:
        if a.data_path and a.timeframe == "5Min":
            cfg = RunConfig.legacy_5min(**overrides)
        else:
            cfg = RunConfig.for_timeframe(a.timeframe, **overrides)
    main(a.data_path, a.symbol, a.timeframe, a.start, a.end, cfg=cfg, out_dir=a.out)
