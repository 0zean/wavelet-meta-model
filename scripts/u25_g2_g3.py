"""
U25 (PLAN3 §5 U25): the G2 / G3 smoke run. Status only — no look is taken.

    uv run python scripts/u25_g2_g3.py smoke [--jobs 8]

`smoke` runs families/G2.yaml and families/G3.yaml as committed, each with its core / reference families
(G2_core, G2_core4, G1) copied beside it, every window cut to 2016-01-04 → 2017-03-01 (long enough for the 252-session
vol gate to close on some sessions; no quasi-holdout cells), into a scratch ledger / runner root / results directory
under a temporary directory that is deleted afterwards. It prints statuses, the core / reference member counts, the
region parity residual, the number of sessions on which each gated stream differs from its reference and the account
check's tradable flag — properties of the code and of the positions, never a return, Sharpe, alpha, p or floor — so it
reads nothing of the development window's outcome and `families/looks.jsonl` is not touched. Its purpose is to show
that every variant, core and reference runs before the specs are registered.
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

SMOKE_WINDOW = {"start": "2016-01-04", "end": "2017-03-01"}
SPECS = {"G2": ("G2_core", "G2_core4"), "G3": ("G1",)}


def _copy(fid: str, tmp: Path) -> Path:
    doc = yaml.safe_load((ROOT / "families" / f"{fid}.yaml").read_text(encoding="utf-8"))
    doc.pop("registered", None)
    doc["window"] = SMOKE_WINDOW
    path = tmp / f"{fid}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def smoke(jobs: int) -> None:
    from experiments.ledger import Ledger
    from families.looks import LOOKS_PATH
    from families.run import run_family

    looks_before = LOOKS_PATH.read_bytes()
    tmp = Path(tempfile.mkdtemp(prefix="u25_smoke_"))
    try:
        ledger = Ledger(tmp / "ledger.jsonl")
        for fid, refs in SPECS.items():
            spec = _copy(fid, tmp)
            for r in refs:
                _copy(r, tmp)
            t = time.perf_counter()
            res = run_family(spec, ledger=ledger, root=tmp / "root", out_dir=tmp / "out", jobs=jobs,
                             feature_cache_dir=None, check_registration=False, quasi=False)  # fmt: skip
            wall = time.perf_counter() - t
            ev = res.get("evaluation") or {}
            print(f"[SMOKE] {fid}: window {SMOKE_WINDOW['start']} → {SMOKE_WINDOW['end']}: {wall:.1f} s, jobs {jobs}")
            print(
                f"[SMOKE]   error: {res['error']}"
                if "error" in res
                else "[SMOKE]   headline test computed (not printed)"
            )
            for label, v in ev.get("variants", {}).items():
                c = v.get("compare") or {}
                extra = f", differs from its reference on {c['days_differ']} sessions" if "days_differ" in c else ""
                err = f" (statistic error: {c['error']})" if "error" in c else ""
                print(f"[SMOKE]   variant {label:<20} {'ok' if c else v.get('status')}{extra}{err}")
            for rid, r in (res.get("refs") or {}).items():
                print(f"[SMOKE]   {rid}: " + ", ".join(f"{lb} {len(x['members'])} member(s)"
                                                    for lb, x in r["variants"].items()))  # fmt: skip
            for sym, p in ((res.get("region") or {}).get("parity") or {}).items():
                print(
                    f"[SMOKE]   parity {sym}: max |Δ daily return| {p['max_abs_diff']!r} over {p['common_days']} days"
                )
            for label, a in (res.get("portfolio_account") or {}).items():
                print(f"[SMOKE]   account core + k × overlay, {label}: tradable {a.get('tradable')}")
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
    ap.add_argument("cmd", choices=["smoke"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    smoke(a.jobs)


if __name__ == "__main__":
    main()
