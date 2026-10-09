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
║    9. Bet sizing               (calibrated p → size; sizing/, --sizer)       ║
║   10. Barrier-exit backtest    (next-bar open, slippage, fractional sizes)   ║
║   11. Risk layer               (vol target, caps, loss gates; --risk-profile)║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import pandas as pd

from data.bars import load_bars
from data.exo import load_series
from features import cache as feature_cache
from features.context import load_context
from features.registry import context_needs
from models.zoo import REGISTRY as ZOO
from primaries import REGISTRY as PRIMARIES
from primaries.diagnostics import primary_diagnostics
from risk.profiles import PROFILES
from sizing import REGISTRY as SIZERS
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
    print(f"[CFG]   sizing: {cfg.SIZER} step={cfg.SIZE_STEP} positions={cfg.POSITION_MODE}  risk={cfg.RISK_PROFILE}")
    print(f"[CFG]   costs: {cfg.COST_MODEL}")
    print(f"[CFG]   events: {cfg.EVENT_SAMPLER} {cfg.EVENT_PARAMS or ''}  exits: {cfg.EXIT_MODEL} {cfg.EXIT_PARAMS or ''}  "
          f"vol profile: {cfg.VOL_PROFILE}")  # fmt: skip
    if cfg.COST_MODEL == "quotes" and (not symbol or data_path):  # fail before the WFO, not after it
        raise ValueError("COST_MODEL='quotes' needs an Alpaca symbol (its quotes-table rows); pass --cost-model")

    # Feature context (market / sector bars, exo series; SPEC §15) and cache (Alpaca data only: the key needs a symbol)
    context = {}
    if cfg.FEATURE_GROUPS and context_needs(cfg.FEATURE_GROUPS, cfg.TIMEFRAME) != (set(), set()):
        if not symbol or data_path:
            raise ValueError(f"feature groups {cfg.FEATURE_GROUPS} need context: pass an Alpaca --symbol")
        if "cross_asset" in cfg.FEATURE_GROUPS and symbol == market_symbol:
            raise ValueError(f"'cross_asset' needs an Alpaca --symbol other than the market ({market_symbol})")
        context = load_context(symbol, timeframe, start, end, cfg.FEATURE_GROUPS, bars=load_bars, exo=load_series,
                               market=market_symbol)  # fmt: skip

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
    (Path(out_dir) / "wfo_run.json").write_text(
        json.dumps({k: v if isinstance(v, dict) else list(v) for k, v in signals.attrs.items()})
    )
    print(f"[OUT]   WFO signals saved to {out_path} (skipped folds in wfo_run.json)")

    # Backtest, metrics and plots cover only the OOS span
    df_oos = df.loc[signals.index[0] :]
    cost_data = None
    if cfg.COST_MODEL == "cs":  # the Corwin–Schultz half-spread is estimated from 5Min bars (risk/costs.py)
        if cfg.TIMEFRAME == "5Min":
            cost_data = df
        elif symbol and not data_path:
            cost_data = load_bars(symbol, "5Min", start, end)
        else:
            raise ValueError("COST_MODEL='cs' needs 5Min bars for the spread estimate")
    elif cfg.COST_MODEL == "quotes":  # SPEC §19: the symbol's rows of the quotes half-spread table
        from data.quotes import read_table

        table = read_table()
        cost_data = table[table["symbol"] == symbol.upper()].reset_index(drop=True)
    results = run_backtest(df_oos, signals, cfg, cost_data=cost_data)
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
    ap.add_argument("--sizer", choices=sorted(SIZERS), help="bet sizer (sizing/; default fixed)")
    ap.add_argument("--size-step", type=float, help="bet-size discretization step (default 0.1, 0 = off)")
    ap.add_argument("--position-mode", choices=["single", "average"], help="one position or active-bet averaging")
    ap.add_argument("--risk-profile", choices=sorted(PROFILES), help="risk layer (risk/profiles.py; default none)")
    ap.add_argument("--cost-model", choices=["slippage", "cs", "quotes"], help="costs (SPEC §19; default quotes)")
    ap.add_argument(
        "--vol-profile", choices=["none", "tod"], help="time-of-day volatility profile (SPEC §14; default none)"
    )
    ap.add_argument("--event-sampler", choices=["cusum", "dc", "schedule"], help="event sampler (SPEC §13)")
    ap.add_argument("--event-params", help='sampler parameters as JSON, e.g. {"entry_times": ["15:30"]}')
    ap.add_argument("--exit-model", choices=["triple_barrier", "time", "hysteresis"], help="exit model (SPEC §13)")
    ap.add_argument("--exit-params", help='exit parameters as JSON, e.g. {"exit_time": "close"}')
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
    if a.event_params:
        overrides["EVENT_PARAMS"] = json.loads(a.event_params)
    if a.exit_params:
        overrides["EXIT_PARAMS"] = json.loads(a.exit_params)
    for arg, name in (
        ("meta_model", "META_MODEL"),
        ("primary_model", "PRIMARY_MODEL"),
        ("meta_train", "META_TRAIN"),
        ("sizer", "SIZER"),
        ("size_step", "SIZE_STEP"),
        ("position_mode", "POSITION_MODE"),
        ("risk_profile", "RISK_PROFILE"),
        ("cost_model", "COST_MODEL"),
        ("vol_profile", "VOL_PROFILE"),
        ("event_sampler", "EVENT_SAMPLER"),
        ("exit_model", "EXIT_MODEL"),
    ):
        if getattr(a, arg) is not None:
            overrides[name] = getattr(a, arg)
    cfg = None
    if overrides:
        if a.data_path and a.timeframe == "5Min":
            cfg = RunConfig.legacy_5min(**overrides)
        else:
            cfg = RunConfig.for_timeframe(a.timeframe, **overrides)
    main(a.data_path, a.symbol, a.timeframe, a.start, a.end, cfg=cfg, out_dir=a.out)
