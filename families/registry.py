"""
The claims registry (PLAN3 §5 U25, SPEC §26): `families/registry.yaml`, one entry per passing claim, the forward test's
input (U27 trades exactly these configurations from their freeze date).

    uv run python -m families registry G1 [--freeze-date 2026-10-12]

An entry is written from the family's result (results/families/<id>/result.json) only when the program verdict passes
it — Holm over the program's families (families/program.yaml) as `python -m families summary` computes it, plus its
floors, coherence and sign — and records what U27 must reproduce: the registered spec sha, the headline's member cell
hashes (and its core / reference members), the code hash and git sha of the run, the verdict's cost, the program
verdict's numbers, whether the family was registered as a diagnostic, and the freeze date. A re-entry replaces the
family's entry (a later run of the same registered spec); the file is rewritten whole, keys sorted.
"""

import datetime as dt
import json
from pathlib import Path

import yaml

from families.spec import HEADLINE, ROOT

REGISTRY = ROOT / "families" / "registry.yaml"


class RegistryError(RuntimeError):
    """A family that the program verdict does not pass, or without a result."""


def _program_row(fid: str, out_dir: Path, program) -> dict:
    from families.report import UNTESTED
    from families.run import program_caps
    from families.test import program_verdict

    listed = program_caps(program).get("families")
    verdicts = {}
    for p in sorted(Path(out_dir).glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if listed is not None and r["id"].split(".v")[0] not in listed:
            continue
        verdicts[r["id"]] = dict(r["evaluation"]["verdict"]) if "evaluation" in r else dict(UNTESTED)
    if fid not in verdicts:
        raise RegistryError(f"{fid} has no result in {out_dir} (or is not a family of the program)")
    table = program_verdict(verdicts)
    return table.set_index("family").loc[fid].to_dict()


def add_entry(fid: str, out_dir, path=REGISTRY, program=None, freeze_date: str | None = None) -> dict:
    """Write `fid`'s entry into the registry (module docstring); returns it."""
    from families.run import PROGRAM

    out_dir = Path(out_dir)
    result = json.loads((out_dir / fid / "result.json").read_text(encoding="utf-8"))
    row = _program_row(fid, out_dir, program or PROGRAM)
    if not row["passes"]:
        raise RegistryError(f"{fid} does not pass the program verdict (p_holm {row['p_holm']:.4f}, floors "
                            f"{row['floors_ok']}, coherent {row['coherent']}, positive {row['positive']})")  # fmt: skip
    v = result["evaluation"]["verdict"]
    entry = {
        "family": fid,
        "registered": result.get("registered"),
        "kind": v.get("kind"),
        "diagnostic": bool((result.get("power") or {}).get("diagnostic", False)),
        "at_cost": v.get("at_cost"),
        "members": list(result["members"][HEADLINE]),
        "variant_hash": result.get("variant_hashes", {}).get(HEADLINE),
        "refs": {
            rid: {lb: x["members"] for lb, x in r["variants"].items()} for rid, r in (result.get("refs") or {}).items()
        },
        "run_at": result.get("run_at"),
        "git_sha": result.get("git_sha"),
        "code_hash": result.get("code_hash"),
        "verdict": {
            "statistic": float(v["delta_ann"]),
            "p": float(row["p"]),
            "p_holm": float(row["p_holm"]),
            "floors_ok": bool(row["floors_ok"]),
            "coherent": bool(row["coherent"]),
        },
        "freeze_date": freeze_date or str(dt.date.today()),
    }
    path = Path(path)
    doc = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or {}
    claims = dict(doc.get("claims") or {})
    claims[fid] = entry
    header = ("# The claims registry (families/registry.py, PLAN3 U25): one entry per passing claim; U27's input.\n"
              "# Written by `python -m families registry <id>`; do not edit by hand.\n")  # fmt: skip
    path.write_text(header + yaml.safe_dump({"claims": claims}, sort_keys=True, allow_unicode=True), encoding="utf-8",
                    newline="\n")  # fmt: skip
    return entry
