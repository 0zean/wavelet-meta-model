"""
Bet-sizer comparison (U7): one WFO per cell, every sizer × position mode backtested on the same OOS signals.

    uv run python -m sizing.compare --symbol SPY --timeframe 1Day --start 2016-01-01 --end 2025-10-01 \\
        --primary wavelet_trend --meta-model logit_l2 --meta-train oof --out results/sizing

The WFO fits every sizer per fold on the train-window OOF meta-probabilities (run_wfo(..., sizers=...)), so the
meta-probabilities, sides and events are identical across rows and only the sizes differ. Each (sizer, mode) row is
appended to `<out>/trials.jsonl` as a counted trial (stage "U7") with OOS return, vol, Sharpe, PSR(0) of the
per-bar returns, max drawdown, turnover, exposure and average bet size; `<out>/sizing_<cell>.csv` holds the table.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.bars import load_bars
from features import cache as feature_cache
from models.compare import _git_sha
from sizing import REGISTRY
from utils.config import RunConfig
from validation.stats import psr, return_moments
from wfo.backtest import run_backtest
from wfo.wfo_engine import run_wfo
from wfo.wfo_metrics import compute_metrics, signal_diagnostics

MODES = ("single", "average")


def sizer_rows(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig, sizers, modes=MODES) -> list[dict]:
    """Backtest metrics of each (sizer, position mode) on one WFO output (columns `bet_size[:name]`)."""
    df_oos = df.loc[signals.index[0] :]
    rows = []
    for name in sizers:
        col = "bet_size" if name == cfg.SIZER else f"bet_size:{name}"
        skipped = signals.attrs.get("sizer_skipped_folds" if name == cfg.SIZER else f"sizer_skipped_folds:{name}")
        for mode in modes:
            c = cfg.replace(SIZER=name, POSITION_MODE=mode)
            res = run_backtest(df_oos, signals, c, size_col=col)
            m = compute_metrics(df_oos, res, c)["Meta-filtered"]
            eq = res["Meta-filtered"][0]
            r = eq.pct_change().dropna().to_numpy()
            flat = np.std(r) == 0  # no trades: the moments are undefined
            mom = return_moments(r) if not flat else None
            rows.append(
                {
                    "sizer": name,
                    "position_mode": mode,
                    "sizer_skipped_folds": len(skipped),
                    "n_trades": int(m["Num Trades"]),
                    "avg_size": float(m["Avg Bet Size"]),
                    "ret_ann": float(m["Annual Return (%)"]) / 100,
                    "vol_ann": float(m["Annual Volatility (%)"]) / 100,
                    "sharpe": float(m["Sharpe Ratio"]),
                    "calmar": float(m["Calmar Ratio"]),
                    "max_dd": float(m["Max Drawdown (%)"]) / 100,
                    "turnover": float(m["Turnover (x/yr)"]),
                    "exposure": float(m["Exposure (%)"]) / 100,
                    "psr": np.nan if flat else float(psr(mom.sr, mom.n_obs, mom.skew, mom.kurt)),
                    "sr_skew": np.nan if flat else float(mom.skew),
                    "sr_kurt": np.nan if flat else float(mom.kurt),
                    "n_obs": len(r),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--primary", default="wavelet_trend")
    ap.add_argument("--meta-model", default="logit_l2")
    ap.add_argument("--meta-train", choices=["val", "oof"], default="oof")
    ap.add_argument("--sizers", default=",".join(REGISTRY), help="comma-separated sizers")
    ap.add_argument("--out", default="results/sizing")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizers = a.sizers.split(",")
    df = load_bars(a.symbol, a.timeframe, a.start, a.end)
    cfg = RunConfig.for_timeframe(
        a.timeframe, PRIMARY=a.primary, META_MODEL=a.meta_model, META_TRAIN=a.meta_train, SIZER=sizers[0]
    )
    cell = f"{a.symbol}_{a.timeframe}_{a.start}_{a.end}_{a.primary}_{a.meta_model}_{a.meta_train}"
    t = time.time()
    signals = run_wfo(df, cfg, symbol=a.symbol, feature_cache_dir=feature_cache.DEFAULT_ROOT, sizers=tuple(sizers))
    wfo_s = time.time() - t
    diag = signal_diagnostics(df, signals, cfg)
    df_oos = df.loc[signals.index[0] :]
    bench = compute_metrics(df_oos, run_backtest(df_oos, signals, cfg), cfg)  # primary-only and buy-and-hold
    shared = {
        "n_folds": int(signals["fold"].nunique()),
        "meta_skipped_folds": len(signals.attrs.get("meta_skipped_folds", [])),
        "n_oos_events": int(diag["OOS events"]),
        "meta_logloss": float(diag["Meta log-loss"]),
        "brier": float(diag["Meta Brier"]),
        "meta_auc": float(diag["Meta AUC"]),
        "sharpe_primary": float(bench.loc["Sharpe Ratio", "Primary only"]),
        "sharpe_buy_hold": float(bench.loc["Sharpe Ratio", "Buy-and-Hold"]),
    }
    started, sha = datetime.now(UTC).isoformat(), _git_sha()
    rows = sizer_rows(df, signals, cfg, sizers)
    with open(out / "trials.jsonl", "a") as f:
        for r in rows:
            spec = {
                "symbol": a.symbol,
                "timeframe": a.timeframe,
                "start": a.start,
                "end": a.end,
                "primary": a.primary,
                "meta_model": a.meta_model,
                "meta_train": a.meta_train,
                "sizer": r["sizer"],
                "position_mode": r["position_mode"],
                "size_step": cfg.SIZE_STEP,
                "seed": cfg.SEED,
            }
            row = {
                "stage": "U7",
                "status": "ok",
                "git_sha": sha,
                "started_at": started,
                "wfo_runtime_s": round(wfo_s, 1),
            }
            f.write(json.dumps({**row, **shared, **r, "spec": spec}) + "\n")
    table = pd.DataFrame([{**shared, **r} for r in rows])
    path = out / f"sizing_{cell}.csv"
    table.to_csv(path, index=False)
    cols = ["sizer", "position_mode", "n_trades", "avg_size", "ret_ann", "vol_ann", "sharpe", "psr", "max_dd",
            "turnover", "exposure", "sizer_skipped_folds"]  # fmt: skip
    print("\n" + table[cols].round(4).to_string(index=False))
    print(f"[SIZE]  primary-only Sharpe {shared['sharpe_primary']:.3f}  buy-and-hold {shared['sharpe_buy_hold']:.3f}")
    print(f"[SIZE]  Saved {path}")


if __name__ == "__main__":
    main()
