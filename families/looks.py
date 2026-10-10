"""
The looks ledger (PLAN3 §4.6, SPEC §22): `families/looks.jsonl` counts every configuration examined on the
development window — diagnostics, pilots, anything whose result was read — so the family report can print K, the
number of looks, and a Bonferroni bound (p × K) next to the bootstrap p of a headline. Looks are appended BEFORE a
result is read (`python -m families look <label> <n> [--note ...]`); the seed is PLAN3 §1's K₀ = 60.

A look is not a trial: trials (family variants) are counted and budgeted by the ledger (families/run.py); looks are
the wider count of what was examined, including what never became a trial.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

LOOKS_PATH = Path(__file__).resolve().parent / "looks.jsonl"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_look(label: str, n: int, note: str = "", path: Path = LOOKS_PATH, *, when: str | None = None) -> dict:
    """Append one record: `label` (a script, a family, a pilot), `n` configurations examined, a note."""
    n = int(n)
    if n < 1 or not label:
        raise ValueError("a look needs a label and n >= 1")
    row = {"at": when or datetime.now(UTC).isoformat(), "label": str(label), "n": n, "note": str(note),
           "git_sha": _git_sha()}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def looks(path: Path = LOOKS_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def looks_total(path: Path = LOOKS_PATH) -> int:
    """K: the configurations examined so far."""
    return int(sum(int(r["n"]) for r in looks(path)))


def bonferroni(p: float, k: int) -> float:
    """The Bonferroni bound on a p-value over K looks (capped at 1)."""
    return float(min(1.0, p * max(int(k), 1)))
