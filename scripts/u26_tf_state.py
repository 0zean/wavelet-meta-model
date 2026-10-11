"""
U26 checks (PLAN3 §5 U26; SPEC §24). Status only: no strategy statistic is printed.

    uv run python scripts/u26_tf_state.py budget   # tf_state on SPY 2016-01 → 2025-09 (≤ 10 s), the stationarity
                                                   # guard on SPY / QQQ for tf_state and vol_state, warm-ups
    uv run python scripts/u26_tf_state.py null     # PLAN3 §1.1's session-shift null (scripts/plan3_diagnostics/
                                                   # periodicity2.py) re-run with features.tf_state.goertzel_power
                                                   # beside the original numpy power, on the same shifts

`null` re-examines the four configurations PLAN3 §1.1 recorded in K₀ (families/looks.jsonl,
periodicity2.py n = 4) on the same data and seed: no new look. The G1 splits are added by
`python -m families splits G1` after a look is recorded.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.bars import load_bars
from features.registry import REGISTRY, check_group_output
from features.tf_state import goertzel_power
from utils.config import RunConfig

START, END = "2016-01-04", "2025-10-01"
BUDGET_S = 10.0


def budget() -> None:
    import features.groups  # noqa: F401  (registers the groups)
    from data.exo import load_series
    from features.exo_align import Exo
    from features.state import VOL_STATE_EXO

    cfg = RunConfig.for_timeframe("5Min")
    for sym in ("SPY", "QQQ"):
        df = load_bars(sym, "5Min", START, END, allow_holdout=False)
        REGISTRY["tf_state"].fn(df.iloc[:5000], cfg, {})  # compile / warm
        t = time.perf_counter()
        f = REGISTRY["tf_state"].fn(df, cfg, {})
        el = time.perf_counter() - t
        check_group_output("tf_state", f, df)
        first = f.notna().idxmax()
        print(f"[U26]  {sym}: {len(df):,} bars; tf_state {el:.2f} s (budget {BUDGET_S:.0f} s: "
              f"{'ok' if el <= BUDGET_S else 'OVER'}); stationarity guard ok")  # fmt: skip
        print(f"[U26]  {sym}: first defined bar per column: "
              + ", ".join(f"{c.split('__')[1]} {str(v)[:10]}" for c, v in first.items()))  # fmt: skip
        exo = {}
        for name in VOL_STATE_EXO:
            src, n = name.split("/")
            exo[name] = load_series(src, n, "2015-06-01", END)
        vs = REGISTRY["vol_state"].fn
        state = vs.fit(df.iloc[: 20 * 78 * 21], cfg)
        g = vs.transform(df, state, cfg, {"exo": Exo(exo).restrict(VOL_STATE_EXO)})
        check_group_output("vol_state", g, df)
        new = g[["vol_state__skew", "vol_state__vvix", "vol_state__vvix_vix"]]
        print(f"[U26]  {sym}: vol_state guard ok; skew / vvix / vvix_vix NaN bars: {new.isna().sum().tolist()}")


def _full_sessions(sym: str) -> np.ndarray:
    """periodicity2.py's input: inside-session close → close log returns of the full 78-bar sessions, demeaned."""
    df = load_bars(sym, "5Min", START, END, allow_holdout=False)
    ny = df.index.tz_convert("America/New_York") if df.index.tz is not None else df.index
    date = pd.Series(ny.date, index=df.index)
    cnt = date.value_counts()
    df = df[date.isin(cnt[cnt == 78].index).to_numpy()]
    px = df["close"].to_numpy().reshape(-1, 78)
    r = np.log(px[:, 1:] / px[:, :-1])
    return r - r.mean()


def _pw(x: np.ndarray, f: float) -> float:
    k = np.arange(len(x))
    return np.abs(np.sum(x * np.exp(-2j * np.pi * f * k))) ** 2 / len(x)


def null() -> None:
    W = 40
    fr = {"1/d": 1 / 77, "2/d": 2 / 77, "3/d": 3 / 77, "1/w": 1 / 385}
    for sym in ("SPY", "QQQ"):
        r = _full_sessions(sym)
        D = r.shape[0]
        rng = np.random.default_rng(1)
        out = {f: [] for f in fr}
        dev = 0.0
        for s in range(0, D - W, W):
            seg = r[s : s + W]
            x = seg.reshape(-1)
            m = len(x)
            for name, f in fr.items():
                k = round(f * m)
                assert abs(k - f * m) < 1e-9, "every tested period is a Fourier bin of the window"
                p_np, p_tf = _pw(x, f), goertzel_power(x, k)
                null_np, null_tf = [], []
                for _ in range(60):
                    sh = np.stack([np.roll(row, rng.integers(77)) for row in seg]).reshape(-1)
                    null_np.append(_pw(sh, f))
                    null_tf.append(goertzel_power(sh, k))
                null_np, null_tf = np.array(null_np), np.array(null_tf)
                dev = max(dev, abs(p_tf / p_np - 1), float(np.max(np.abs(null_tf / null_np - 1))))
                out[name].append(((null_np >= p_np).mean(), (null_tf >= p_tf).mean(), p_tf / np.median(null_tf)))
        print(f"[U26]  {sym}: {D} full sessions, {len(out['1/d'])} windows of {W} sessions; max |goertzel / numpy − 1| "
              f"{dev:.1e}")  # fmt: skip
        for name in fr:
            a = np.array(out[name])
            same = np.array_equal(a[:, 0], a[:, 1])
            print(f"[U26]    {name}: share of windows with p < 0.05 = {(a[:, 1] < 0.05).mean():.2f} (numpy "
                  f"{(a[:, 0] < 0.05).mean():.2f}, every window's p equal: {same}); median power / null median "
                  f"{np.median(a[:, 2]):.2f}")  # fmt: skip


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    {"budget": budget, "null": null}[sys.argv[1]]()
