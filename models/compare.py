"""
Model-zoo comparison (U6): every model through the same WFO, plus a CPCV Sharpe distribution.

    uv run python -m models.compare --symbol SPY --timeframe 1Day --start 2016-01-01 --end 2025-10-01 \\
        --role meta --primary wavelet_trend --out results/model_zoo

Per model (one row, appended to `<out>/trials.jsonl` as a counted trial, stage "U6"):
- WFO (the run's config with META_MODEL, or PRIMARY_MODEL for `--role primary`, switched): OOS meta log-loss,
  Brier, AUC, meta precision (approved success rate), approved share, backtest Sharpe (meta-filtered and primary-only),
  and the OOS reliability curve (`<out>/calibration_<tag>.csv`).
- CPCV (CPCV_GROUPS, CPCV_TEST_GROUPS) over all development-window events: the model is fit (with its own inner
  HP search unless ZOO_FIXED_PARAMS fixes it, + calibration) on each purged train split and predicts the test groups; the φ stitched paths give a
  distribution of annualized Sharpe ratios of per-event net returns (approved events only for the meta role; every
  event on its predicted side for the primary role, where the classifier is fit on every non-zero-label train event:
  no active-day filter, unlike ml_xgb in the WFO). Per-fold feature groups (fracdiff d) are fit once on the whole
  window: d is label-free and shared by every model, so it cannot favour one. These Sharpes rank models; they are not
  a backtest (overlapping events are treated as independent bets).
"""

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.bars import load_bars
from features import cache as feature_cache
from features.events import sample_events
from features.exits import exit_frame
from features.feature_builder import FeatureSet
from features.triple_barrier_labels import average_uniqueness, triple_barrier_labels
from models.zoo import REGISTRY, inner_cv, make_model
from primaries import check_signal, make_primary
from utils.config import RunConfig
from validation.purged_cv import CombinatorialPurgedCV
from validation.stats import sharpe_ratio
from wfo.backtest import run_backtest
from wfo.wfo_engine import run_wfo
from wfo.wfo_metrics import calibration_table, compute_metrics, meta_outcomes, signal_diagnostics


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def wfo_row(df: pd.DataFrame, cfg: RunConfig, symbol: str) -> tuple[dict, pd.DataFrame]:
    """OOS WFO metrics for one configuration and its meta reliability curve."""
    signals = run_wfo(df, cfg, symbol=symbol, feature_cache_dir=feature_cache.DEFAULT_ROOT)
    diag = signal_diagnostics(df, signals, cfg)
    df_oos = df.loc[signals.index[0] :]
    metrics = compute_metrics(df_oos, run_backtest(df_oos, signals, cfg), cfg)
    o = meta_outcomes(df, signals, cfg)
    o = o[o["scored"]]
    row = {
        "n_folds": int(signals["fold"].nunique()),
        "meta_skipped_folds": len(signals.attrs.get("meta_skipped_folds", [])),
        "n_oos_events": int(diag["OOS events"]),
        "meta_logloss": float(diag["Meta log-loss"]),
        "brier": float(diag["Meta Brier"]),
        "meta_auc": float(diag["Meta AUC"]),
        "meta_precision": float(diag["Approved success rate"]),
        "approved_share": float(diag["Approved share"]),
        "primary_success": float(diag["Primary success rate"]),
        "n_trades": int(metrics.loc["Num Trades", "Meta-filtered"]),
        "sharpe": float(metrics.loc["Sharpe Ratio", "Meta-filtered"]),
        "sharpe_primary": float(metrics.loc["Sharpe Ratio", "Primary only"]),
        "sharpe_buy_hold": float(metrics.loc["Sharpe Ratio", "Buy-and-Hold"]),
    }
    return row, calibration_table(o["success"], o["meta_prob"])


def _events_frame(df: pd.DataFrame, cfg: RunConfig, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Development-window events with complete features: (X, labels, uniqueness weights)."""
    if cfg.VOL_PROFILE != "none":  # one whole-window CPCV has no fold to fit a profile on
        raise ValueError("models.compare samples events once for the whole window: VOL_PROFILE must be 'none'")
    events = sample_events(df, cfg, symbol=symbol)
    labels = triple_barrier_labels(df, events, cfg)
    fset = FeatureSet(cfg, symbol=symbol, cache_dir=feature_cache.DEFAULT_ROOT)
    X_all = fset.build(df).join(fset.transform(df, fset.fit(df)))
    X = X_all.reindex(labels.index).dropna()
    labels = labels.loc[X.index]
    labels["width"] = events.loc[X.index, "width"]
    return X, labels, average_uniqueness(labels, len(df))


def _side_returns(df: pd.DataFrame, labels: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Gross barrier-exit return of a long and of a short on every event (side-aware ties, as in the backtest)."""
    out = {}
    for name, s in (("long", 1), ("short", -1)):
        side = pd.Series(s, index=labels.index)
        ex = exit_frame(df, labels.index, labels["width"], cfg, side=side)
        out[name] = s * ex["ret"].reindex(labels.index)
    r = pd.DataFrame(out)
    if r.isna().any().any():
        raise RuntimeError("side-aware barrier exits are missing for some labelled events")
    return r


def cpcv_sharpes(df: pd.DataFrame, cfg: RunConfig, role: str, model: str, symbol: str) -> np.ndarray:
    """
    Annualized Sharpe ratio of each CPCV path (module docstring). role="meta": cfg.PRIMARY must be a fixed rule
    (its frame joins the features, as in the WFO); role="primary": the zoo model is the direction classifier.
    """
    X, labels, w = _events_frame(df, cfg, symbol)
    rets = _side_returns(df, labels, cfg)
    cost = 2 * cfg.SLIPPAGE_PCT
    if role == "meta":
        prim = make_primary(cfg)
        if cfg.PRIMARY == "ml_xgb":
            raise ValueError("CPCV for the meta role needs a fixed-rule primary (ml_xgb would need nested OOF)")
        frame = check_signal(prim.signal(df, X, cfg), X, prim.name)
        side = frame["signed_dir"].to_numpy()
        side_ret = np.where(side > 0, rets["long"], rets["short"])
        y = (side_ret > cfg.META_MIN_RET).astype(int)
        feats = pd.concat([X, frame], axis=1)
    else:
        keep = labels["label"] != 0
        X, labels, w, rets = X[keep], labels[keep], w[keep], rets[keep]
        y = (labels["label"] > 0).astype(int).to_numpy()
        feats = X

    t0 = labels["entry_pos"].to_numpy() - 1
    t1 = labels["exit_pos"].to_numpy()
    cv = CombinatorialPurgedCV.from_cfg(cfg, len(df))
    outs = []
    for s, (train, test) in enumerate(cv.split(t0, t1)):
        m = make_model(model, cfg, role=role)
        m.fit(feats.iloc[train], y[train], w.iloc[train].to_numpy(), inner_cv(labels.iloc[train], cfg))
        p = m.predict_proba(feats.iloc[test])
        if role == "meta":
            r = np.where(p >= cfg.META_THRESH, side_ret[test] - cost, 0.0)
        else:
            r = np.where(p >= cfg.CLF_THRESH, rets["long"].to_numpy()[test], rets["short"].to_numpy()[test]) - cost
        outs.append(r)
        print(f"[CPCV]  {model} split {s + 1}/{cv.n_splits}")
    paths = cv.assemble_paths(outs, len(labels))
    years = (df.index[-1] - df.index[0]).days / 365.25
    ann = np.sqrt(len(labels) / years)
    return np.array([sharpe_ratio(p) * ann if np.std(p) > 0 else 0.0 for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--role", choices=["meta", "primary"], default="meta")
    ap.add_argument("--models", default=",".join(REGISTRY), help="comma-separated zoo models")
    ap.add_argument("--primary", default="wavelet_trend")
    ap.add_argument("--meta-model", default="legacy", help="meta-model held fixed for --role primary")
    ap.add_argument("--meta-train", choices=["val", "oof"], default="val")
    ap.add_argument("--no-cpcv", action="store_true")
    ap.add_argument("--out", default="results/model_zoo")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_bars(a.symbol, a.timeframe, a.start, a.end)
    base = RunConfig.for_timeframe(a.timeframe, PRIMARY=a.primary, META_TRAIN=a.meta_train, META_MODEL=a.meta_model)
    rows = []
    for model in a.models.split(","):
        field = "META_MODEL" if a.role == "meta" else "PRIMARY_MODEL"
        cfg = base.replace(**{field: model})
        tag = f"{a.symbol}_{a.timeframe}_{a.start}_{a.end}_{a.role}_{model}_{a.primary}_{a.meta_train}"
        t = time.time()
        spec = {
            "symbol": a.symbol,
            "timeframe": a.timeframe,
            "start": a.start,
            "end": a.end,
            "role": a.role,
            "model": model,
            "primary": a.primary,
            "meta_model": cfg.META_MODEL,
            "primary_model": cfg.PRIMARY_MODEL,
            "meta_train": a.meta_train,
            "zoo_fixed_params": cfg.ZOO_FIXED_PARAMS,
            "calibration": cfg.CALIBRATION,
            "oof_meta": cfg.OOF_META,
            "seed": cfg.SEED,
        }
        row = {"stage": "U6", "status": "ok", "git_sha": _git_sha(), "started_at": datetime.now(UTC).isoformat()}
        try:
            metrics, calib = wfo_row(df, cfg, a.symbol)
            calib.to_csv(out / f"calibration_{tag}.csv", index=False)
            row |= metrics
        except Exception as e:  # noqa: BLE001 — a failing model is a recorded trial (status=error), never dropped
            row |= {"status": "error", "error": f"{type(e).__name__}: {e}"}
            print(f"[ZOO]  {model} failed: {row['error']}")
        if row["status"] == "ok" and not a.no_cpcv:
            try:  # a CPCV failure is recorded next to the (valid) WFO metrics
                sr = cpcv_sharpes(df, cfg, a.role, model, a.symbol)
                row |= {
                    "cpcv_sharpes": sr.round(4).tolist(),
                    "cpcv_sr_mean": float(sr.mean()),
                    "cpcv_sr_sd": float(sr.std(ddof=1)),
                    "cpcv_sr_min": float(sr.min()),
                    "cpcv_pct_positive": float((sr > 0).mean()),
                }
            except Exception as e:  # noqa: BLE001 — recorded, never dropped
                row["cpcv_error"] = f"{type(e).__name__}: {e}"
                print(f"[ZOO]  {model} CPCV failed: {row['cpcv_error']}")
        row["runtime_s"] = round(time.time() - t, 1)
        row = {**row, "spec": spec}
        with open(out / "trials.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        rows.append({**spec, **{k: v for k, v in row.items() if k not in ("spec", "cpcv_sharpes")}})

    table = pd.DataFrame(rows)
    path = out / f"comparison_{a.symbol}_{a.timeframe}_{a.start}_{a.end}_{a.role}_{a.primary}_{a.meta_train}.csv"
    table.to_csv(path, index=False)
    cols = [c for c in ("model", "meta_logloss", "brier", "meta_auc", "meta_precision", "sharpe", "cpcv_sr_mean",
                        "cpcv_sr_sd", "runtime_s", "status") if c in table]  # fmt: skip
    print("\n" + table[cols].round(4).to_string(index=False))
    print(f"[ZOO]  Saved {path}")


if __name__ == "__main__":
    main()
