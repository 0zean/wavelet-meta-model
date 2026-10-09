"""
U12 cost table (PLAN2 U12 done-when): meta-model fits and seconds per walk-forward window, U12 defaults on vs off, and
windows per PWFO cell for the U11 grid vs the U12 grid.

    uv run python scripts/u12_cost.py --symbol SPY --out results/u12_cost.md [--jobs 9]

Per (timeframe, meta model, arm), one rolling combo (the timeframe's first U12 IS length, OOS 21) of a rule primary
(donchian_breakout, META_TRAIN=oof) is walked from its first window until N_MEASURED windows past the point where the
"on" arm calibrates from its rolling history; fits and seconds are the mean over those last N_MEASURED windows (the
same windows in every arm). Fits = estimator .fit calls (purged-CV folds, refits, OOF-meta refits). Arms:
  off        U11 behaviour: ZOO_FIXED_PARAMS={}, CALIBRATION=crossfit, OOF_META=refit, sizer fixed
  off+ecdf   the same with the ecdf sizer (its per-split OOF meta refits; REVIEW §6's worst case)
  on         U12 defaults (fixed parameters, rolling calibration), sizer fixed
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIMEFRAMES = ("30Min", "1Hour", "1Day")
MODELS = ("logit_l2", "rf_ldp_fast", "xgb")
OLD = {"ZOO_FIXED_PARAMS": {}, "CALIBRATION": "crossfit", "OOF_META": "refit"}
ARMS = {"off": {**OLD, "SIZER": "fixed"}, "off+ecdf": {**OLD, "SIZER": "ecdf"}, "on": {"SIZER": "fixed"}}
N_MEASURED = 3
U11_GRID = {"intraday": ((252, 378, 504, 756), (5, 10, 21, 63)), "1Day": ((1260, 1512), (5, 10, 21, 63))}


def _counting(fits: list):
    """Patch every meta estimator class's fit to count calls (class level: clones are counted too)."""
    import xgboost as xgb

    from models import zoo

    for cls in (zoo.ScaledLogit, zoo.SerialPredictRF, xgb.XGBClassifier):
        orig = cls.fit

        def fit(self, *a, _orig=orig, **k):
            fits.append(1)
            return _orig(self, *a, **k)

        cls.fit = fit


def measure(symbol: str, tf: str, model: str, start: str, end: str) -> list[dict]:
    from data.bars import get_calendar, load_bars
    from features import cache as feature_cache
    from utils.config import RunConfig
    from wfo import wfo_engine as eng
    from wfo.pwfo import make_grid, pwfo_windows

    fits: list = []
    _counting(fits)
    df = load_bars(symbol, tf, start, end)
    sessions = get_calendar(end).index
    base = {"PRIMARY": "donchian_breakout", "META_MODEL": model, "META_TRAIN": "oof"}
    cals: list = []
    real = eng.fit_meta_model
    eng.fit_meta_model = lambda *a, **k: cals.append(real(*a, **k)) or cals[-1]

    def walk(arm: str, only: list[int] | None):
        cfg = RunConfig.for_timeframe(tf, **base, **ARMS[arm])
        combo = make_grid(cfg)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            prep = eng.prepare(df, cfg, symbol=symbol, feature_cache_dir=feature_cache.DEFAULT_ROOT)
        wins = pwfo_windows(df.index, combo, embargo=cfg.EMBARGO, sessions=sessions)
        hist = eng.CalHistory() if cfg.CALIBRATION == "rolling" else None
        out = []
        for win in wins if only is None else [wins[w] for w in only]:
            n0, c0, t0 = len(fits), len(cals), time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                res = eng.fit_window(df, cfg, prep, win.w + 1, win.is_start, win.train_end, win.val_end, win.oos_end,
                                     win.train_embargo, win.val_embargo, cal_history=hist)  # fmt: skip
            m = cals[-1] if len(cals) > c0 else None
            out.append({"w": win.w, "status": res.status, "fits": len(fits) - n0, "s": time.perf_counter() - t0,
                        "cal": getattr(m, "calibration_", None)})  # fmt: skip
            if only is None and str(out[-1]["cal"]).startswith("rolling"):
                rolled = [o for o in out if str(o["cal"]).startswith("rolling")]
                if len(rolled) >= N_MEASURED:
                    break
        return combo, out

    combo, on = walk("on", None)
    measured = [o["w"] for o in on if str(o["cal"]).startswith("rolling")][-N_MEASURED:]
    rows = []
    for arm in ARMS:
        _, out = (combo, on) if arm == "on" else walk(arm, measured)  # no history needed: just those windows
        sel = [o for o in out if o["w"] in measured]
        rows.append({
            "timeframe": tf, "model": model, "arm": arm, "combo": combo.label, "windows": measured,
            "fits_per_window": float(np.mean([o["fits"] for o in sel])), "s_per_window": float(np.mean([o["s"] for o in sel])),
            "calibration": sorted({str(o["cal"]) for o in sel}),
        })  # fmt: skip
    eng.fit_meta_model = real
    return rows


def _md(df: pd.DataFrame) -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cols = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(fmt(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join([" | ".join(cols), " | ".join("---" for _ in cols), *rows])


def windows_per_cell(symbol: str, start: str, end: str) -> pd.DataFrame:
    from data.bars import get_calendar, load_bars
    from utils.config import RunConfig
    from wfo.pwfo import make_grid, pwfo_windows

    sessions = get_calendar(end).index
    rows = []
    for tf in TIMEFRAMES:
        df = load_bars(symbol, tf, start, end)
        new = RunConfig.for_timeframe(tf)
        is_g, oos_g = U11_GRID["1Day" if tf == "1Day" else "intraday"]
        old = new.replace(PWFO_IS_GRID=is_g, PWFO_OOS_GRID=oos_g, PWFO_DEFAULT=(is_g[0], 10))
        n = {}
        for arm, cfg in (("U11 grid", old), ("U12 grid", new)):
            n[arm] = sum(len(pwfo_windows(df.index, c, embargo=cfg.EMBARGO, sessions=sessions)) for c in make_grid(cfg))
        rows.append({"timeframe": tf, **n, "ratio": n["U11 grid"] / n["U12 grid"]})
    return pd.DataFrame(rows)


def main() -> None:
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-10-01")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results/u12_cost.md")
    a = ap.parse_args()

    wins = windows_per_cell(a.symbol, a.start, a.end)
    jobs = [(tf, m) for tf in TIMEFRAMES for m in MODELS]
    res = Parallel(n_jobs=a.jobs)(delayed(measure)(a.symbol, tf, m, a.start, a.end) for tf, m in jobs)
    t = pd.DataFrame([r for rows in res for r in rows])
    piv = t.pivot_table(index=["timeframe", "model"], columns="arm", values=["fits_per_window", "s_per_window"])
    piv[("fits_ratio", "off/on")] = piv[("fits_per_window", "off")] / piv[("fits_per_window", "on")]
    piv[("fits_ratio", "off+ecdf/on")] = piv[("fits_per_window", "off+ecdf")] / piv[("fits_per_window", "on")]
    piv[("s_ratio", "off/on")] = piv[("s_per_window", "off")] / piv[("s_per_window", "on")]
    piv[("s_ratio", "off+ecdf/on")] = piv[("s_per_window", "off+ecdf")] / piv[("s_per_window", "on")]
    md = [f"# U12 cost table — {a.symbol} {a.start} → {a.end}", "", "## Fits and seconds per window", "",
          _md(piv), "", "Measured windows / calibration:", "",
          _md(t[["timeframe", "model", "arm", "combo", "windows", "calibration"]]), "",
          "## Windows per PWFO cell", "", _md(wins), ""]  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
