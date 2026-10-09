"""
U12 parity (PLAN2 U12, SPEC §11.2 invariant): four U11 Stage C PWFO cells re-run with every new engine switch at its
old value must reproduce their U11 `daily_returns.csv` (and every combo's daily returns, `returns_pwfo.csv`) exactly.

    uv run python scripts/u12_parity.py build                 # tests/fixtures/u12_parity.json from the U11 cells
    uv run python scripts/u12_parity.py spec OUT.yaml [--legacy]
    uv run python scripts/u12_parity.py check ROOT LEDGER     # compare a parity run against the fixture

`spec` writes the fixture cells as a stage-C spec whose `overrides` pin the U12 switches to their old values
(OLD_SWITCHES); `--legacy` leaves them out, for code that predates the switches. Run it into a scratch root and
ledger (`python -m experiments --root R --ledger R/ledger.jsonl run OUT.yaml`), never the canonical ledger.
Hashes are sha256 of the file bytes with CRLF normalized to LF.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "u12_parity.json"
U11_CELLS = ROOT / "results" / "experiments" / "cells"
FILES = ("daily_returns.csv", "returns_pwfo.csv")
# U11 Stage C cell hashes; together they exercise the grid search (logit_l2 3 points, xgb 2, rf_ldp_fast 1), the
# cross-fitted calibration choice, the oof_meta_prob refit (ecdf), cmda, the ml_xgb primary and nested selection
CELLS = {
    "68878e5a62b3d84e": "NVDA 1Day ml_xgb logit_l2 cmda ecdf (OOF meta refit, 3-point grid)",
    "70babc42461bd4e1": "QQQ 1Hour donchian_breakout xgb ldp_sigmoid (2-point grid, intraday)",
    "2dfc205f85035126": "AMZN 1Hour wavelet_trend rf_ldp_fast ldp_sigmoid (1-point grid, forest)",
    "a52bf71cec3e9215": "SPY 1Day wavelet_trend logit_l2 fixed (full feature groups)",
}
OLD_SWITCHES = {"ZOO_FIXED_PARAMS": {}, "CALIBRATION": "crossfit", "OOF_META": "refit", "PWFO_COMBINE": "nested"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def label(spec: dict) -> str:
    s = spec
    return "_".join(["-".join(s["symbols"]), s["timeframe"], s["primary"]["name"], s["model"]["meta"],
                     s["meta_train"], s["sizer"], s["risk_profile"], "pwfo"])  # fmt: skip


def build() -> None:
    out = []
    for h, why in CELLS.items():
        d = U11_CELLS / h
        spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        out.append({
            "u11_cell_hash": h, "stage": spec["stage"], "why": why, "label": label(spec["spec"]),
            "spec": spec["spec"], "sha256": {f: file_hash(d / f) for f in FILES},
            "u11": {k: res.get(k) for k in ("sharpe", "n_obs", "n_trades", "pbo", "n_trials")},
        })  # fmt: skip
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"cells": out}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE} ({len(out)} cells)")


def spec(path: Path, legacy: bool) -> None:
    import yaml

    cells = []
    for c in json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]:
        s = copy.deepcopy(c["spec"])
        if not legacy:
            s["overrides"] = {**s["overrides"], **OLD_SWITCHES}
        cells.append(s)
    doc = {"name": "u12_parity" + ("_legacy" if legacy else ""), "stage": "C", "cells": cells}
    path.write_text(yaml.safe_dump(doc, sort_keys=True, width=1000), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells{', legacy' if legacy else ''})")


def check(root: Path, ledger: Path) -> int:
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_label = {r["label"]: r for r in rows if r.get("stage") == "C" and "label" in r}
    bad = 0
    for c in json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]:
        r = by_label.get(c["label"])
        if r is None or r.get("status") != "ok":
            print(f"MISSING  {c['label']}: {None if r is None else r.get('status')}")
            bad += 1
            continue
        d = root / "cells" / r["cell_hash"]
        for f, want in c["sha256"].items():
            got = file_hash(d / f) if (d / f).exists() else None
            ok = got == want
            bad += not ok
            print(f"{'OK      ' if ok else 'MISMATCH'}  {c['label']}  {f}  {(got or 'absent')[:16]} vs {want[:16]}")
    print("parity: PASS" if not bad else f"parity: FAIL ({bad})")
    return int(bad > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("spec")
    s.add_argument("out", type=Path)
    s.add_argument("--legacy", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("root", type=Path)
    c.add_argument("ledger", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "spec":
        spec(a.out, a.legacy)
    else:
        sys.exit(check(a.root, a.ledger))


if __name__ == "__main__":
    main()
