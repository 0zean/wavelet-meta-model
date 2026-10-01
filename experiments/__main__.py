"""
Experiment runner CLI (U10, SPEC §9).

    uv run python -m experiments run experiments/specs/u10_smoke.yaml --jobs 8
    uv run python -m experiments report [--stage A]
    uv run python -m experiments import-legacy            # fold results/**/trials.jsonl (U6–U9) into the ledger

`run --final` is the one-time holdout evaluation (stage E only; recorded in the ledger).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

from experiments.ledger import DEFAULT_PATH, Ledger
from experiments.runner import DEFAULT_ROOT, ROOT


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows writes a redirected stream in the ANSI code page otherwise
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        prog="experiments", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", default=str(DEFAULT_PATH))
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="artifacts root (cells/, signals/, report/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a spec")
    r.add_argument("spec")
    r.add_argument("--jobs", type=int, default=1, help="parallel cells (processes)")
    r.add_argument("--inner-jobs", type=int, default=1, help="parallel PWFO combos inside a cell")
    r.add_argument("--final", action="store_true", help="the one-time holdout evaluation (stage E)")
    r.add_argument("--retry-errors", action="store_true", help="re-run cells whose ledger row is an error")
    p = sub.add_parser("report", help="leaderboard, funnel, PBO per stage")
    p.add_argument("--stage", default=None)
    p.add_argument("--out", default=None, help="default <root>/report")
    p.add_argument("--top", type=int, default=50)
    i = sub.add_parser("import-legacy", help="fold pre-U10 trials.jsonl files into the ledger")
    i.add_argument("paths", nargs="*", help="default: every results/**/trials.jsonl")
    a = ap.parse_args()

    ledger = Ledger(a.ledger)
    if a.cmd == "run":
        from experiments.runner import run
        from experiments.spec import load_spec

        doc, cells = load_spec(a.spec)
        print(f"[EXP]  {a.spec}: {len(cells)} cell(s), stage {doc['stage']}")
        rows = run(
            cells, ledger=ledger, root=a.root, jobs=a.jobs, inner_jobs=a.inner_jobs, final=a.final,
            retry_errors=a.retry_errors, spec_name=doc.get("name") or Path(a.spec).stem,
        )  # fmt: skip
        status = {}
        for row in rows:
            status[row["status"]] = status.get(row["status"], 0) + 1
        print(f"[EXP]  wrote {len(rows)} ledger row(s) {status} → {ledger.path}")
    elif a.cmd == "report":
        from experiments.report import write_report

        paths = write_report(ledger, a.root, a.out or Path(a.root) / "report", a.stage, a.top)
        print(paths["md"].read_text(encoding="utf-8"))
        print(f"[EXP]  report → {paths['html']}")
    else:
        from experiments.legacy import import_legacy

        paths = a.paths or sorted((ROOT / "results").rglob("trials.jsonl"))
        n = import_legacy(paths, ledger, base=ROOT)
        print(f"[EXP]  imported {n} legacy row(s) from {len(paths)} file(s) → {ledger.path}")


if __name__ == "__main__":
    main()
