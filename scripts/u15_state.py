"""
U15 diagnostic (PLAN2 U15 done-criteria, SPEC §15): the state and context groups on cached data over the
development window.

    uv run python scripts/u15_state.py

For SPY 5Min, SPY 1Day and AAPL 5Min (cross_asset with market SPY and sector XLK):
- every new group's columns pass the stationarity guard (check_group_output, also run by the feature build);
- per column: NaN count and the last NaN stamp (the warm-up), mean, std, min, max, |corr with close|;
- vol_state is per fold: fit on the first 5Min fold's train window (INITIAL_TRAIN sessions minus the embargo) and on
  the whole window, with the fitted GARCH parameters;
- point-in-time spot checks on SPY 5Min: the VIX seen by the bars of 2020-03-16 and 2020-03-17 (VIX closed 57.83 on
  03-13 and 82.69 on 03-16), and the 10-year yield change seen on the day after a large move.

Outputs: results/u15/state_features.json. Network: none (cached bars and exo series).
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.bars import load_bars
from data.exo import load_series
from features.context import load_context
from features.feature_builder import FeatureSet
from features.registry import REGISTRY, check_group_output
from utils.config import RunConfig
from wfo.wfo_engine import wfo_folds

OUT = ROOT / "results" / "u15"
START, END = "2016-01-04", "2025-10-01"
NEW = ["cross_asset", "vol_state", "calendar_events", "rates_credit"]


def column_stats(X: pd.DataFrame, close: pd.Series) -> dict:
    out = {}
    for c in X.columns:
        x = X[c]
        nan = x.isna()
        out[c] = {
            "n_nan": int(nan.sum()),
            "last_nan": str(x.index[nan][-1]) if nan.any() else None,
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
            "abs_corr_close": float(abs(x.corr(close))) if x.std() > 0 else None,
        }
    return out


def run(symbol: str, timeframe: str, groups: list[str]) -> dict:
    cfg = RunConfig.for_timeframe(timeframe)
    df = load_bars(symbol, timeframe, START, END)
    ctx = load_context(symbol, timeframe, START, END, ["wavelet_core", *groups], bars=load_bars, exo=load_series)
    fs = FeatureSet(cfg, ["wavelet_core", *groups], context=ctx)
    t0 = time.time()
    static = fs.build(df)  # runs check_group_output on every static group
    t_static = time.time() - t0
    rec = {"symbol": symbol, "timeframe": timeframe, "bars": len(df), "context": sorted(ctx),
           "exo": sorted(ctx.get("exo", {})), "static_s": round(t_static, 2)}  # fmt: skip
    if "sector" in ctx:
        rec["sector_bars"] = len(ctx["sector"])
    parts = [static.drop(columns=[c for c in static if c.startswith("wavelet_core__")])]
    if "vol_state" in groups:
        train_end, _, _, emb_tr, _ = next(iter(wfo_folds(df.index, cfg)))
        spec = REGISTRY["vol_state"]
        vctx = fs.group_context["vol_state"]
        fits = {}
        for label, train in (("first_fold", df.iloc[: train_end - emb_tr]), ("full", df)):
            t0 = time.time()
            state = spec.fn.fit(train, cfg)
            frame = spec.fn.transform(df, state, cfg, vctx)
            check_group_output("vol_state", frame, df)
            _, omega, alpha, beta, vbar = state
            fits[label] = {"train_bars": len(train), "omega": omega, "alpha": alpha, "beta": beta, "vbar": vbar,
                           "persistence": alpha + beta, "seconds": round(time.time() - t0, 2)}  # fmt: skip
            if label == "first_fold":
                parts.append(frame)
        rec["garch"] = fits
    X = pd.concat(parts, axis=1)
    rec["columns"] = column_stats(X, df["close"].astype(float))
    rec["stationarity_guard"] = "pass"
    return rec, X


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"window": [START, END], "runs": []}
    spy5, X5 = run("SPY", "5Min", ["vol_state", "calendar_events", "rates_credit"])
    out["runs"].append(spy5)
    spyd, _ = run("SPY", "1Day", ["vol_state", "calendar_events", "rates_credit"])
    out["runs"].append(spyd)
    aapl, _ = run("AAPL", "5Min", ["cross_asset"])
    out["runs"].append(aapl)

    vix = X5["vol_state__vix"]
    out["spot_checks"] = {
        "vix_2020-03-16_09:30": float(vix.loc["2020-03-16 09:30"]),
        "vix_2020-03-16_15:55": float(vix.loc["2020-03-16 15:55"]),
        "vix_2020-03-17_09:30": float(vix.loc["2020-03-17 09:30"]),
        "fomc_2020-03-03_09:55": float(X5.loc["2020-03-03 09:55", "calendar_events__fomc_day"]),
        "fomc_2020-03-03_10:00": float(X5.loc["2020-03-03 10:00", "calendar_events__fomc_day"]),
        "d_dgs10_2020-03-09_15:55": float(X5.loc["2020-03-09 15:55", "rates_credit__d_dgs10"]),
        "d_dgs10_2020-03-10_09:30": float(X5.loc["2020-03-10 09:30", "rates_credit__d_dgs10"]),
        "d_dgs10_2020-03-11_09:30": float(X5.loc["2020-03-11 09:30", "rates_credit__d_dgs10"]),
    }
    (OUT / "state_features.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in out["runs"]:
        print(f"{r['symbol']} {r['timeframe']}: {r['bars']:,} bars, context {r['context']}, static {r['static_s']} s")
        for c, s in r["columns"].items():
            print(f"  {c:<38} nan {s['n_nan']:>7,}  last nan {s['last_nan']!s:<26} mean {s['mean']:>10.4g}  "
                  f"std {s['std']:>9.4g}  |ρ close| {s['abs_corr_close'] or 0:.3f}")  # fmt: skip
        for k, g in r.get("garch", {}).items():
            print(f"  GARCH {k}: α {g['alpha']:.4f} β {g['beta']:.4f} persistence {g['persistence']:.4f} "
                  f"({g['train_bars']:,} train bars, {g['seconds']} s)")  # fmt: skip
    print(json.dumps(out["spot_checks"], indent=2))


if __name__ == "__main__":
    main()
