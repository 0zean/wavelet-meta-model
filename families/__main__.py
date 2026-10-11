"""
Family-test CLI (SPEC §17, PLAN2 U17).

    uv run python -m families register families/F1.yaml      # then commit the `registered` line
    uv run python -m families run families/F1.yaml --jobs 8 [--ledger L --root R --out O]
    uv run python -m families summary [--out O]               # program verdict over every family result
    uv run python -m families look <label> <n> [--note ...]   # append n looks to families/looks.jsonl (SPEC §22)
    uv run python -m families power --n-days 2400 --vol-ann 0.08 --families 4   # the MDE line (families/power.py)

`run` refuses an unregistered or modified spec and a run past the family's TRIAL_BUDGET or the program caps
(families/program.yaml). Its cells are stage F (rule cells); its ledger rows are one per variant (the counted trials).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    from experiments.ledger import DEFAULT_PATH
    from experiments.runner import DEFAULT_ROOT
    from families.run import DEFAULT_OUT

    ap = argparse.ArgumentParser(
        prog="families", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", default=str(DEFAULT_PATH))
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="runner artifacts root (cells/, signals/)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="family results (<out>/<id>/, program_summary.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("register", help="record the HEAD sha in a committed, clean family spec")
    g.add_argument("spec")
    r = sub.add_parser("run", help="run and test one registered family")
    r.add_argument("spec")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--no-quasi", action="store_true", help="skip the quasi-holdout slice cells")
    sub.add_parser("summary", help="program verdict over the family results in --out")
    lk = sub.add_parser("look", help="record configurations examined on the development window (families/looks.jsonl)")
    lk.add_argument("label")
    lk.add_argument("n", type=int)
    lk.add_argument("--note", default="")
    pw = sub.add_parser("power", help="the minimum detectable alpha of an overlay test (families/power.py)")
    pw.add_argument("rest", nargs=argparse.REMAINDER)
    a, extra = ap.parse_known_args()  # `power` passes its own flags through (families/power.py parses them)
    if extra and a.cmd != "power":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if a.cmd == "register":
        from families.spec import register

        reg = register(a.spec)
        print(f"[FAM]  {a.spec}: registered at {reg['sha']} ({reg['date']}); commit the file before running it")
    elif a.cmd == "run":
        from experiments.ledger import Ledger
        from families.run import run_family

        res = run_family(a.spec, ledger=Ledger(a.ledger), root=a.root, out_dir=a.out, jobs=a.jobs,
                         feature_cache_dir="default", quasi=not a.no_quasi)  # fmt: skip
        print(Path(res["paths"]["md"]).read_text(encoding="utf-8"))
    elif a.cmd == "look":
        from families.looks import looks_total, record_look

        row = record_look(a.label, a.n, a.note)
        print(f"[FAM]  look recorded: {row['label']} n={row['n']}; K = {looks_total()}")
    elif a.cmd == "power":
        from families.power import main as power_main

        power_main([*extra, *a.rest])
    else:
        from experiments.ledger import Ledger
        from families.report import program_summary
        from families.run import PROGRAM

        print(program_summary(a.out, ledger=Ledger(a.ledger), program=PROGRAM)["md"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
