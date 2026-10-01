"""
Append-only trial ledger (SPEC §9, U10): `results/ledger.jsonl`, one JSON object per line.

Rows are trials (`kind` = "wfo", "portfolio", "pwfo" or "legacy") or events (`event` = "holdout_access"). Only the
runner's parent process appends; each row is one write + fsync under an exclusive lock, after its cell finished,
so a killed run leaves complete rows plus at most one truncated last line. A truncated line is reported and
skipped by the reader and isolated by the next append (which starts a fresh line); it is never rewritten.

Trial count for the DSR (SPEC §9): N(stage) = Σ n_trials over the **distinct** cell hashes with a counted status
("ok" or "no_fit") in that stage or an earlier one (STAGES order). A PWFO cell is n_trials = its grid size; a cache
hit re-recorded under a later stage shares its hash and is counted once; a cell re-run under new code is a new hash
and counts again (over-counting is the safe direction).
"""

import contextlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from experiments.spec import stage_rank

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "results" / "ledger.jsonl"
COUNTED = ("ok", "no_fit")

if os.name == "nt":  # Windows: a byte-range lock on the file's first byte (msvcrt has no flock)
    import msvcrt

    def _lock(f, blocking: bool = True) -> None:
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if not blocking:
                    raise BlockingIOError from None
                time.sleep(0.05)

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(f, blocking: bool = True) -> None:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))

    def _unlock(f) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)


TRADING_DAYS = 252


def clean(x):
    """JSON-safe value: NaN / ±inf → None, numpy scalars → Python, recursively."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


class Ledger:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        """Every parseable row, in file order; malformed lines (a killed write) are reported on stderr and skipped."""
        if not self.path.exists():
            return []
        out, bad = [], []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad.append(i)
                    continue
                if not isinstance(row, dict):
                    bad.append(i)
                    continue
                out.append(row)
        if bad:
            print(f"[LEDGER]  {self.path}: skipped {len(bad)} malformed line(s) {bad[:10]}", file=sys.stderr)
        return out

    @contextlib.contextmanager
    def run_lock(self):
        """Exclusive for the duration of a run (a sidecar `.lock` file): two runs on one ledger would both miss the
        other's rows (duplicates) and could race the holdout check."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as f:
            try:
                _lock(f, blocking=False)
            except BlockingIOError:
                raise RuntimeError(f"another experiment run holds {self.path} (one run per ledger)") from None
            try:
                yield
            finally:
                _unlock(f)

    def append(self, rows: dict | Iterable[dict]) -> None:
        rows = [rows] if isinstance(rows, dict) else list(rows)
        if not rows:
            return
        for r in rows:
            validate(r)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(clean(r), sort_keys=True, allow_nan=False) + "\n" for r in rows)
        with open(self.path, "a+b") as f:
            _lock(f)
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":  # a killed write left a partial line: start a fresh one
                        data = "\n" + data
                f.write(data.encode())
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock(f)


def validate(row: dict) -> None:
    """A row the counting and resume logic can read: an event, or a trial with a hash, a known stage and a status."""
    if "event" in row:
        return
    for k in ("cell_hash", "stage", "status"):
        if not row.get(k):
            raise ValueError(f"ledger row without {k!r}: {str(row)[:200]}")
    stage_rank(row["stage"])
    if row["status"] not in (*COUNTED, "error"):
        raise ValueError(f"ledger row with unknown status {row['status']!r}")


def trials(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "event" not in r]


def counted(rows: list[dict], stage: str) -> dict[str, dict]:
    """Distinct counted trials (cell hash → first counted row) in `stage` or earlier."""
    rank = stage_rank(stage)
    out: dict[str, dict] = {}
    for r in trials(rows):
        if r.get("status") in COUNTED and stage_rank(r["stage"]) <= rank:
            out.setdefault(r["cell_hash"], r)
    return out


def n_trials(rows: list[dict], stage: str) -> int:
    return int(sum(int(r.get("n_trials", 1)) for r in counted(rows, stage).values()))


def trial_sharpes(row: dict) -> list[float]:
    """Per-period (daily) Sharpes of a row's trials: its combos' OOS Sharpes for a PWFO cell, else its own Sharpe."""
    vals = [c.get("sharpe") for c in row["combos"]] if row.get("combos") else [row.get("sharpe")]
    return [float(v) / np.sqrt(TRADING_DAYS) for v in vals if v is not None and np.isfinite(v)]


def var_trials(rows: list[dict], stage: str) -> float:
    """V[SR_n] for the DSR: variance (ddof 1) of the per-period Sharpes of the counted trials up to `stage`."""
    srs = [s for r in counted(rows, stage).values() if int(r.get("n_trials", 1)) > 0 for s in trial_sharpes(r)]
    return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0


def done(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(cell_hash, stage) → its latest trial row."""
    out = {}
    for r in trials(rows):
        out[(r["cell_hash"], r["stage"])] = r
    return out


def holdout_events(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("event") == "holdout_access"]
