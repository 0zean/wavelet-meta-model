"""
U28 (PLAN3 §5 U28, SPEC §27): G6's power inputs and smoke run. Neither takes a look.

    uv run python scripts/u28_g6.py power
    uv run python scripts/u28_g6.py smoke [--jobs 8]

`power` computes the MDE line's ρ from the bars and the VIX state alone, as G3's was (no mean, Sharpe, alpha or p of
anything is computed or printed): the region's target on each session from the previous session's VIX close and
VX1 / VIX settlement (the 15 cells of G6's headline, the 0.5 scale before 2018-02-28), SVXY's and the SPY / QQQ 50/50
mix's close-to-close returns, and from them the region's uncentred second moments — σ_o² = mean(t² r_svxy²) and
E[t r_svxy r_mix] — i.e. the variance and covariance of the gross overlay a mean-zero stream would have. With G2's
registered core assumptions (σ_core 13.5 %, Sharpe 0.7) and k = 0.5: ρ(core + k·overlay, core), the MDE Δ Sharpe
(families.power.mde_delta_sharpe, Holm over 4 families) and the Δ Sharpe expected from PLAN2 F9's overlay Sharpe 0.8.

`smoke` runs families/G6.yaml as committed with its core families copied beside it, every window cut to
2016-01-04 → 2016-07-01 (the episodes moved inside it), into a scratch ledger / runner root / results directory that is
deleted afterwards. It prints statuses, member and engine-cell counts, the kill bars and the report's presence —
never a return, Sharpe, alpha, p or floor — so `families/looks.jsonl` is not touched.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOW = ("2016-01-04", "2025-10-01")
SMOKE_WINDOW = {"start": "2016-01-04", "end": "2016-07-01"}
SMOKE_EPISODES = [{"label": "smoke", "start": "2016-02-01", "end": "2016-03-01"}]
CORES = ("G2_core",)
SIGMA_CORE, SHARPE_CORE, SHARPE_OVERLAY, K = 0.135, 0.7, 0.8, 0.5  # G2's registered core; PLAN2 F9's 0.6–1.0


def _closes(source, sym: str) -> pd.Series:
    df = source.bars(sym, "1Day", *WINDOW, allow_holdout=False)
    c = df["close"].astype(float)
    c.index = df.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return c


def power() -> None:
    from experiments.runner import CachedBars
    from families.power import mde_delta_sharpe
    from primaries.mechanism import primary_config
    from primaries.vix_carry import VixCarry

    src = CachedBars()
    head = yaml.safe_load((ROOT / "families" / "G6.yaml").read_text(encoding="utf-8"))["headline"]["primary"]
    prim = VixCarry(**head["params"])
    primary_config("vix_carry", "5Min", head["params"])  # the headline's params validate
    days = _closes(src, "SVXY").index
    state = {}
    for name in ("VIX", "VX1_VIX", "VX2_VX1"):
        v = src.exo("cboe", name, "2015-06-01", WINDOW[1], allow_holdout=False)["value"]
        v.index = pd.DatetimeIndex(v.index).normalize()
        state[name] = v.reindex(days.union(v.index)).ffill().shift(1).reindex(days)  # the previous close, as of d
    X = pd.DataFrame({"vix_term__vix": state["VIX"], "vix_term__vx1_vix": state["VX1_VIX"],
                      "vix_term__vx2_vx1": state["VX2_VX1"]}, index=days).dropna()  # fmt: skip
    _, t = prim.rule(None, X, None)
    t = pd.Series(t, index=X.index)
    r_s = _closes(src, "SVXY").pct_change()
    r_m = 0.5 * _closes(src, "SPY").pct_change() + 0.5 * _closes(src, "QQQ").pct_change()
    j = pd.concat({"t": t, "s": r_s, "m": r_m}, axis=1).dropna()
    o = j["t"] * j["s"]
    sig_o = float(np.sqrt((o**2).mean() * 252))
    sig_m = float(np.sqrt((j["m"] ** 2).mean() * 252))
    rho_om = float((o * j["m"]).mean() * 252 / (sig_o * sig_m))
    sc, so = SIGMA_CORE, sig_o
    cov = sc * sc + K * rho_om * sc * so
    sp = float(np.sqrt(sc * sc + K * K * so * so + 2 * K * rho_om * sc * so))
    rho = cov / (sc * sp)
    n = len(j)
    mde = mde_delta_sharpe(n, rho, SHARPE_CORE, alpha=0.05, families=4, sided="one")
    expected = (SHARPE_CORE * sc + K * SHARPE_OVERLAY * so) / sp - SHARPE_CORE
    print(f"[POWER] sessions {n} ({j.index[0].date()} → {j.index[-1].date()}); region mean |target| "
          f"{j['t'].mean():.3f}, share of sessions with a position {(j['t'] > 0).mean():.3f}")  # fmt: skip
    print(f"[POWER] uncentred second moments: σ_overlay {sig_o:.4f}/yr (gross, at the region's target), σ_mix "
          f"{sig_m:.4f}, ρ(overlay, mix) {rho_om:.3f}")  # fmt: skip
    print(f"[POWER] core (G2's registered assumptions): σ {sc:.3f}, Sharpe {SHARPE_CORE}; k {K}: σ_portfolio "
          f"{sp:.4f}, ρ(core + k·overlay, core) {rho:.4f}")  # fmt: skip
    print(f"[POWER] MDE Δ Sharpe {mde['mde_delta_sharpe']:.3f} (se {mde['se_delta_sharpe']:.3f}, Holm over 4 "
          f"families, one-sided, 80 % power); expected Δ Sharpe {expected:.3f} at an overlay Sharpe of "
          f"{SHARPE_OVERLAY} (PLAN2 F9)")  # fmt: skip


def _copy(fid: str, tmp: Path, episodes=None) -> Path:
    doc = yaml.safe_load((ROOT / "families" / f"{fid}.yaml").read_text(encoding="utf-8"))
    doc.pop("registered", None)
    doc["window"] = SMOKE_WINDOW
    if episodes is not None and "episodes" in doc:
        doc["episodes"] = episodes
    doc.pop("sample_splits", None)  # date splits outside the smoke window
    path = tmp / f"{fid}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def smoke(jobs: int) -> None:
    from experiments.ledger import Ledger
    from families.looks import LOOKS_PATH
    from families.run import run_family

    looks_before = LOOKS_PATH.read_bytes()
    tmp = Path(tempfile.mkdtemp(prefix="u28_smoke_"))
    try:
        ledger = Ledger(tmp / "ledger.jsonl")
        spec = _copy("G6", tmp, SMOKE_EPISODES)
        for r in CORES:
            _copy(r, tmp)
        t = time.perf_counter()
        res = run_family(spec, ledger=ledger, root=tmp / "root", out_dir=tmp / "out", jobs=jobs,
                         feature_cache_dir=None, check_registration=False, quasi=False)  # fmt: skip
        wall = time.perf_counter() - t
        ev = res.get("evaluation") or {}
        print(f"[SMOKE] G6: window {SMOKE_WINDOW['start']} → {SMOKE_WINDOW['end']}: {wall:.1f} s, jobs {jobs}")
        print(
            f"[SMOKE]   error: {res['error']}" if "error" in res else "[SMOKE]   headline test computed (not printed)"
        )
        for label, v in ev.get("variants", {}).items():
            c = v.get("compare") or {}
            err = f" (statistic error: {c['error']})" if "error" in c else ""
            print(f"[SMOKE]   variant {label:<12} {'ok' if c else v.get('status')}{err}; members "
                  f"{len(res['members'].get(label, []))}")  # fmt: skip
        for rid, r in (res.get("refs") or {}).items():
            print(f"[SMOKE]   {rid}: " + ", ".join(f"{lb} {len(x['members'])} member(s)"
                                                for lb, x in r["variants"].items()))  # fmt: skip
        reg = res.get("region") or {}
        st = reg.get("status") or {}
        print(f"[SMOKE]   engine region: method {reg.get('method')}, {len(reg.get('names', []))} cells, "
              f"{sum(v == 'ok' for v in st.values())} ok, statistic {reg.get('statistic')}, costs {reg.get('costs')}")  # fmt: skip
        coh = ((ev.get("variants") or {}).get("headline") or {}).get("cells") or {}
        print(f"[SMOKE]   cells coherence evaluated over {coh.get('n_cells')} cells (share not printed)")
        print(f"[SMOKE]   kill recorded for {sum(len(v) for v in (res.get('kill') or {}).values())} variant member(s) "
              "(whether it fired is not printed)")  # fmt: skip
        eps = res.get("episodes") or {}
        print("[SMOKE]   episodes: " + ", ".join(f"{k} {len(e['rows'])} rows" for k, e in eps.items()))
        print(f"[SMOKE]   overlay-alone statistic {'computed' if res.get('overlay_alone') else 'MISSING'}; account "
              f"tradable {(res.get('account') or {}).get('tradable')}; portfolio accounts "
              f"{ {k: a.get('tradable') for k, a in (res.get('portfolio_account') or {}).items()} }")  # fmt: skip
        print(f"[SMOKE]   cost curve columns {sorted(res.get('cost_curve', {}))}; report "
              f"{'written' if (res.get('paths') or {}).get('md') else 'missing'}")  # fmt: skip
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert LOOKS_PATH.read_bytes() == looks_before, "the smoke changed families/looks.jsonl"
    print("[SMOKE] scratch outputs deleted; families/looks.jsonl unchanged")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["power", "smoke"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    power() if a.cmd == "power" else smoke(a.jobs)


if __name__ == "__main__":
    main()
