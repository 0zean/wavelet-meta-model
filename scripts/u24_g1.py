"""
U24 (PLAN3 §5 U24): the G1 smoke run. Status only — no look is taken.

    uv run python scripts/u24_g1.py smoke [--jobs 8]

`smoke` runs families/G1.yaml as committed, with its window cut to 2016-01-04 → 2016-07-01 (the first half-year of the
development window; no quasi-holdout cells), into a scratch ledger / runner root / results directory under a temporary
directory that is deleted afterwards. It prints each cell's and each variant's status, the region cells' parity
residual (a property of the code, not a performance statistic), shapes and timings — never a return, Sharpe, alpha, p
or floor — so it reads nothing of the development window's outcome and `families/looks.jsonl` is not touched. Its
purpose is to show that every variant, the sample split and the region's cell curve run before the spec is registered.
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

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SMOKE_WINDOW = {"start": "2016-01-04", "end": "2016-07-01"}


def smoke(jobs: int) -> None:
    from experiments.ledger import Ledger
    from families.looks import LOOKS_PATH
    from families.run import run_family

    looks_before = LOOKS_PATH.read_bytes()
    doc = yaml.safe_load((ROOT / "families" / "G1.yaml").read_text(encoding="utf-8"))
    doc["window"] = SMOKE_WINDOW
    tmp = Path(tempfile.mkdtemp(prefix="u24_smoke_"))
    try:
        spec = tmp / "G1.yaml"
        spec.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        t = time.perf_counter()
        res = run_family(spec, ledger=Ledger(tmp / "ledger.jsonl"), root=tmp / "root", out_dir=tmp / "out", jobs=jobs,
                         feature_cache_dir=None, check_registration=False, quasi=False)  # fmt: skip
        wall = time.perf_counter() - t
        ev = res.get("evaluation") or {}
        print(f"[SMOKE] window {SMOKE_WINDOW['start']} → {SMOKE_WINDOW['end']}: {wall:.1f} s wall-clock, jobs {jobs}")
        print(f"[SMOKE] error: {res['error']}" if "error" in res else "[SMOKE] headline test computed (not printed)")
        for label, v in ev.get("variants", {}).items():
            print(f"[SMOKE]   variant {label:<16} {'ok' if 'compare' in v else v.get('status')}")
        reg = res.get("region") or {}
        print(f"[SMOKE] region cells: {len(reg.get('cells', {}))} (instrument, cell) points × costs {reg.get('costs')}")
        for sym, p in (reg.get("parity") or {}).items():
            print(f"[SMOKE]   parity {sym}: max |Δ daily return| {p['max_abs_diff']!r} over {p['common_days']} days")
        d = res.get("described") or {}
        print(f"[SMOKE] sample splits: {sorted(d.get('sample_splits', {}))}; state splits: "
              f"{ {k: ('unavailable' if s.get('unavailable') else 'ok') for k, s in d.get('state_splits', {}).items()} }")  # fmt: skip
        print(f"[SMOKE] account check: {'ran' if res.get('account') else 'missing'}; trade stats: "
              f"{sorted((res.get('trade_stats') or {}).get('per_instrument', {}))}")  # fmt: skip
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert LOOKS_PATH.read_bytes() == looks_before, "the smoke changed families/looks.jsonl"
    print("[SMOKE] scratch outputs deleted; families/looks.jsonl unchanged")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    smoke(a.jobs)


if __name__ == "__main__":
    main()
