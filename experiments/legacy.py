"""
Fold the pre-U10 `trials.jsonl` files (stages U6–U9: models/compare, sizing/compare, risk/run, wfo/pwfo_run) into
the ledger, so the DSR's trial count includes every experiment run before the runner existed.

Each line becomes one `kind="legacy"` row: cell_hash = "legacy-" + hash of the line (re-importing is a no-op),
`n_trials` = 1, except the U9 nested-PWFO row (n_trials = 0: it is a selection among the combo rows, which are already
counted, including the `no_fit` ones). Comparable scalars are copied (Sharpe annualized, return, drawdown, AUC, ...);
the source row is kept verbatim under `original`. Legacy return moments are per bar, not per day, so the report
computes no DSR for legacy rows; their Sharpes do enter V[SR_n] (annualized / √252).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.ledger import Ledger, validate

COPY = ("status", "stage", "git_sha", "started_at", "runtime_s", "n_oos_events", "n_trades", "meta_logloss", "brier",
        "ret_ann", "vol_ann", "sharpe", "sortino", "calmar", "max_dd", "turnover", "wfe", "n_oos_windows", "pbo",
        "error")  # fmt: skip


def legacy_row(line: str, source: str) -> dict:
    r = json.loads(line)
    out = {k: r[k] for k in COPY if k in r}
    out["cell_hash"] = "legacy-" + hashlib.sha256(line.strip().encode()).hexdigest()[:16]
    out["kind"] = "legacy"
    out["source"] = source
    out["n_trials"] = 0 if r.get("combo") == "nested" else 1
    if "dsr" in r:
        out["pwfo_dsr"] = r["dsr"]
    auc = r.get("meta_auc")
    if isinstance(auc, dict):  # U8: per symbol
        vals = [v for v in auc.values() if v is not None and np.isfinite(v)]
        auc = float(np.mean(vals)) if vals else None
    out["meta_auc"] = auc
    spec = r.get("spec") or {}
    out["spec_json"] = json.dumps(spec, sort_keys=True)
    syms = spec.get("symbols") or [spec.get("symbol", "?")]
    bits = [
        "-".join(syms),
        spec.get("timeframe", "?"),
        spec.get("primary", ""),
        spec.get("model") or spec.get("meta_model", ""),
    ]
    bits += [str(spec[k]) for k in ("sizer", "position_mode", "risk_profile", "combo") if k in spec]
    out["label"] = "_".join(str(b) for b in bits if b)
    out["original"] = r
    return out


def import_legacy(paths, ledger: Ledger, base: Path | None = None) -> int:
    """Append every not-yet-imported line of `paths`; returns the number of rows added."""
    have = {r.get("cell_hash") for r in ledger.rows()}
    new = []
    for p in paths:
        p = Path(p)
        src = str(p.relative_to(base)) if base is not None and p.is_relative_to(base) else str(p)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = legacy_row(line, src)
            except json.JSONDecodeError:  # a killed pre-U10 run's partial line: reported, not imported
                print(f"[EXP]  {src}: skipped a malformed line: {line[:80]!r}")
                continue
            validate(row)  # unknown stage / status → raise before anything is appended
            if row["cell_hash"] not in have:
                have.add(row["cell_hash"])
                new.append(row)
    ledger.append(new)
    return len(new)
