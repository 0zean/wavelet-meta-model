"""
Portfolio backtest runner (U8): one WFO per symbol, then every risk profile on one union timeline and equity curve.

    uv run python -m risk.run --symbols SPY,QQQ,IWM,DIA,TLT --timeframe 1Hour --start 2016-01-01 --end 2025-10-01 \\
        --primary wavelet_trend --meta-model logit_l2 --meta-train oof --out results/portfolio

Each symbol's WFO signals are saved under `<out>/signals/` (`--reuse` loads them instead of re-running). For every
risk profile the meta-filtered and the primary-only strategies are simulated (risk.portfolio.simulate_portfolio); the
equal-weight buy-and-hold of the symbols is the passive benchmark. Writes `<out>/portfolio_<cell>.csv` (metrics
table), `<out>/equity_<cell>.csv`, `<out>/trades_<cell>_<profile>.csv` and appends one counted trial per (profile,
strategy="meta") to `<out>/trials.jsonl` (stage "U8").
"""

import os

# One BLAS / OpenMP thread per WFO process: parallel workers each spinning a full thread pool oversubscribe the cores
# (measured: 1Hour folds ~0.6 s alone, ~300–700 s with six default-threaded workers). Set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.bars import load_bars
from features import cache as feature_cache
from models.compare import _git_sha
from risk.costs import fill_costs
from risk.portfolio import simulate_portfolio
from risk.profiles import PROFILES, get_profile
from utils.config import RunConfig
from wfo.wfo_engine import run_wfo
from wfo.wfo_metrics import signal_diagnostics, strategy_metrics


def _wfo_one(sym: str, a: argparse.Namespace, sig_dir: Path) -> str:
    """Run (or load) one symbol's WFO; stdout goes to `<sig_dir>/<sym>.log`."""
    path = sig_dir / f"{sym}_{a.timeframe}.pkl"
    if a.reuse and path.exists():
        return sym
    cfg = _cfg(a)
    with (
        open(sig_dir / f"{sym}_{a.timeframe}.log", "w", buffering=1, encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        df = load_bars(sym, a.timeframe, a.start, a.end)
        t = time.time()
        sig = run_wfo(df, cfg, symbol=sym, feature_cache_dir=feature_cache.DEFAULT_ROOT)
        sig.attrs["wfo_runtime_s"] = round(time.time() - t, 1)
        sig.attrs["diag"] = signal_diagnostics(df, sig, cfg).to_dict()
    sig.to_pickle(path)
    sig.to_csv(sig_dir / f"{sym}_{a.timeframe}.csv")
    return sym


def _cfg(a: argparse.Namespace, **kw) -> RunConfig:
    return RunConfig.for_timeframe(
        a.timeframe,
        PRIMARY=a.primary,
        META_MODEL=a.meta_model,
        META_TRAIN=a.meta_train,
        SIZER=a.sizer,
        COST_MODEL=a.cost_model,
        **kw,
    )


def risk_summary(log: pd.DataFrame, trades: pd.DataFrame, cfg: RunConfig) -> dict:
    """How often each part of the risk layer acted."""
    reasons = trades["exit_reason"].value_counts() if len(trades) else pd.Series(dtype=int)
    return {
        "gate_sessions": int(log.loc[log["gate"] > 0].index.normalize().nunique()),
        "throttled_share": float((log["dd_mult"] < 1).mean()),
        "max_gross": float(log["gross"].max()),
        "max_abs_net": float(log["net"].abs().max()),
        "max_position": float(log["max_pos"].max()),
        "max_count": int(log["count"].max()),
        "gate_exits": int(reasons.get("gate", 0)),
        "trim_exits": int(reasons.get("trim", 0)),
        "trimmed_positions": int(trades["trimmed"].sum()) if len(trades) else 0,
        "avg_frac": float(trades["frac"].mean()) if len(trades) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", required=True, help="comma-separated")
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--primary", default="wavelet_trend")
    ap.add_argument("--meta-model", default="logit_l2")
    ap.add_argument("--meta-train", choices=["val", "oof"], default="oof")
    ap.add_argument("--sizer", default="fixed")
    ap.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated risk profiles")
    ap.add_argument("--cost-model", choices=["slippage", "cs", "quotes"], default="quotes", help="SPEC §19")
    ap.add_argument("--jobs", type=int, default=5, help="parallel WFO processes")
    ap.add_argument("--reuse", action="store_true", help="load saved per-symbol signals when present")
    ap.add_argument("--out", default="results/portfolio")
    a = ap.parse_args()

    syms = sorted(s.strip() for s in a.symbols.split(","))
    out = Path(a.out)
    sig_dir = out / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    cell = f"{'-'.join(syms)}_{a.timeframe}_{a.start}_{a.end}_{a.primary}_{a.meta_model}_{a.meta_train}_{a.sizer}"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(a.jobs, len(syms))) as ex:
        for sym in ex.map(_wfo_one, syms, [a] * len(syms), [sig_dir] * len(syms)):
            print(f"[PORT]  {sym} signals ready ({time.time() - t0:.0f}s)")

    cfg0 = _cfg(a)
    bars, sigs, cost_data = {}, {}, {}
    for sym in syms:
        sig = pd.read_pickle(sig_dir / f"{sym}_{a.timeframe}.pkl")
        df = load_bars(sym, a.timeframe, a.start, a.end)
        bars[sym], sigs[sym] = df.loc[sig.index[0] :], sig
        if cfg0.COST_MODEL == "cs":
            cost_data[sym] = df if a.timeframe == "5Min" else load_bars(sym, "5Min", a.start, a.end)
        elif cfg0.COST_MODEL == "quotes":
            from data.quotes import read_table

            table = read_table()
            cost_data[sym] = table[table["symbol"] == sym].reset_index(drop=True)

    bh = pd.concat({s: b["close"] / b["close"].iloc[0] for s, b in bars.items()}, axis=1, sort=True)
    bh = bh.ffill().fillna(1.0).mean(axis=1) * cfg0.INIT_CASH
    cols, equities, rows = {}, {"Buy-and-Hold (EW)": bh}, []
    empty = pd.DataFrame(columns=["pnl_pct", "bars_held"])
    cols["Buy-and-Hold (EW)"] = strategy_metrics(bh, empty, cfg0.bars_per_year)
    for name in a.profiles.split(","):
        profile = get_profile(name)
        cfg = _cfg(a, RISK_PROFILE=name)
        costs = None
        if cfg.COST_MODEL != "slippage":
            costs = {s: fill_costs(bars[s].index, cfg, cost_data[s]) for s in syms}
        for strat, side, size in (("meta", "trade_signal", "bet_size"), ("primary", "signed_dir", None)):
            eq, trades, log = simulate_portfolio(bars, sigs, cfg, profile, side_col=side, size_col=size, costs=costs)
            label = f"{strat} | {name}"
            m = strategy_metrics(eq, trades, cfg.bars_per_year)
            cols[label], equities[label] = m, eq
            trades.to_csv(out / f"trades_{cell}_{strat}_{name}.csv")
            summ = risk_summary(log, trades, cfg)
            print(f"[PORT]  {label}: Sharpe {m['Sharpe Ratio']:.3f}  risk {summ}")
            if strat == "meta":
                rows.append((name, m, summ, eq))
    table = pd.DataFrame(cols)
    path = out / f"portfolio_{cell}.csv"
    table.to_csv(path)
    pd.DataFrame(equities).to_csv(out / f"equity_{cell}.csv")

    diag = {s: sigs[s].attrs.get("diag", {}) for s in syms}
    started, sha = datetime.now(UTC).isoformat(), _git_sha()
    with open(out / "trials.jsonl", "a", encoding="utf-8") as f:
        for name, m, summ, eq in rows:
            spec = {
                "symbols": syms,
                "timeframe": a.timeframe,
                "start": a.start,
                "end": a.end,
                "primary": a.primary,
                "meta_model": a.meta_model,
                "meta_train": a.meta_train,
                "sizer": a.sizer,
                "risk_profile": name,
                "seed": cfg0.SEED,
            }
            row = {
                "stage": "U8",
                "status": "ok",
                "git_sha": sha,
                "started_at": started,
                "n_trades": int(m["Num Trades"]),
                "ret_ann": float(m["Annual Return (%)"]) / 100,
                "vol_ann": float(m["Annual Volatility (%)"]) / 100,
                "sharpe": float(m["Sharpe Ratio"]),
                "sortino": float(m["Sortino Ratio"]),
                "calmar": float(m["Calmar Ratio"]),
                "max_dd": float(m["Max Drawdown (%)"]) / 100,
                "turnover": float(m["Turnover (x/yr)"]),
                "psr": float(m["PSR(0)"]),
                "n_obs": len(eq) - 1,
                "meta_auc": {s: float(d.get("Meta AUC", np.nan)) for s, d in diag.items()},
                "risk": summ,
            }
            f.write(json.dumps({**row, "spec": spec}) + "\n")
    print("\n" + table.round(3).to_string())
    print(f"[PORT]  Saved {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
