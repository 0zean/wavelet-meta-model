"""
Experiment specs (SPEC §9, U10): a YAML file → a list of validated cells.

    stage: A                      # one of STAGES
    name: screen                  # free text, recorded in the ledger
    defaults:                     # every cell field not given falls back to DEFAULT_CELL
      timeframe: 1Hour
      start: 2016-01-01
      end: 2025-10-01
      model: {meta: logit_l2}
    grid:                         # cartesian product; dotted keys reach into primary / model / pwfo / overrides
      symbols: [SPY, QQQ, [SPY, QQQ]]   # a string = one symbol; a list = one portfolio cell
      primary.name: [wavelet_trend, sma_cross]
    cells:                        # optional explicit cells (each merged over defaults, then crossed with grid)
      - {sizer: linear}

Cell fields: symbols, timeframe, start, end ([start, end) NY dates), feature_groups ("default" or a list),
primary {name, params}, model {meta, primary}, meta_train, sizer, risk_profile, pwfo ({is_grid, oos_grid, expanding}
or null = one expanding WFO), seed, overrides (any other RunConfig field). A mechanism primary (SPEC §16) puts its
default sampler and exit (config_overrides) into `overrides`, under the cell's own, so the recorded cell is the run
configuration; its sizer defaults to `rule_size` (when the cell and the defaults set none), and a feature group its
rule reads (vol_state for a VIX source / gate) must be listed. Every cell's RunConfig is built at load time, so an invalid spec fails before anything runs.
"""

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from utils.config import DEFAULT_FEATURE_GROUPS, RunConfig

# Stage order for trial counting: N for a stage = trials of that stage and every earlier one (SPEC §9).
STAGES = ("U6", "U7", "U8", "U9", "U10", "A", "B", "C", "D", "E")
FINAL_STAGE = "E"

DEFAULT_CELL = {
    "symbols": None,
    "timeframe": None,
    "start": None,
    "end": None,
    "feature_groups": "default",
    "primary": {"name": "wavelet_trend", "params": {}},
    "model": {"meta": "logit_l2", "primary": "legacy"},
    "meta_train": "oof",
    "sizer": "fixed",
    "risk_profile": "none",
    "pwfo": None,
    "seed": 42,
    "overrides": {},
}
_REQUIRED = ("symbols", "timeframe", "start", "end")
# RunConfig fields a cell sets through its own keys; `overrides` may not set them a second time
_CELL_OWNED = {
    "TIMEFRAME", "FEATURE_GROUPS", "PRIMARY", "PRIMARY_PARAMS", "META_MODEL", "PRIMARY_MODEL", "META_TRAIN",
    "SIZER", "RISK_PROFILE", "SEED", "PWFO_IS_GRID", "PWFO_OOS_GRID", "PWFO_EXPANDING", "ALLOW_HOLDOUT",
}  # fmt: skip


def stage_rank(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}") from None


@dataclass(frozen=True)
class Cell:
    """One trial configuration: `spec` is the canonical (normalized, fully defaulted) cell dict."""

    spec: dict
    stage: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.spec["symbols"])

    @property
    def is_pwfo(self) -> bool:
        return self.spec["pwfo"] is not None

    def config(self, final: bool = False) -> RunConfig:
        s = self.spec
        kw = {
            "FEATURE_GROUPS": tuple(s["feature_groups"]),
            "PRIMARY": s["primary"]["name"],
            "PRIMARY_PARAMS": s["primary"]["params"],
            "META_MODEL": s["model"]["meta"],
            "PRIMARY_MODEL": s["model"]["primary"],
            "META_TRAIN": s["meta_train"],
            "SIZER": s["sizer"],
            "RISK_PROFILE": s["risk_profile"],
            "SEED": s["seed"],
            "ALLOW_HOLDOUT": final,
        }
        if s["pwfo"] is not None:
            p = s["pwfo"]
            kw |= {"PWFO_IS_GRID": p["is_grid"], "PWFO_OOS_GRID": p["oos_grid"], "PWFO_EXPANDING": p["expanding"]}
        return RunConfig.for_timeframe(s["timeframe"], **s["overrides"], **kw)

    def label(self) -> str:
        s = self.spec
        bits = ["-".join(s["symbols"]), s["timeframe"], s["primary"]["name"], s["model"]["meta"], s["meta_train"],
                s["sizer"], s["risk_profile"]]  # fmt: skip
        if s["pwfo"] is not None:
            bits.append("pwfo")
        return "_".join(bits)

    def spec_json(self) -> str:
        return json.dumps(self.spec, sort_keys=True)


def _date(x) -> str:
    return str(pd.Timestamp(str(x)).date())


def normalize(raw: dict) -> dict:
    """Fill defaults and canonicalize a cell dict (types, order, case) so equal configurations compare equal."""
    unknown = set(raw) - set(DEFAULT_CELL)
    if unknown:
        raise ValueError(f"unknown cell field(s) {sorted(unknown)}; expected {sorted(DEFAULT_CELL)}")
    c = copy.deepcopy(DEFAULT_CELL)
    for k, v in raw.items():
        if isinstance(c[k], dict) and isinstance(v, dict) and k not in ("overrides",):
            c[k] = {**c[k], **v}
        elif k == "primary" and isinstance(v, str):
            c[k] = {"name": v, "params": {}}
        else:
            c[k] = copy.deepcopy(v)
    missing = [k for k in _REQUIRED if c[k] is None]
    if missing:
        raise ValueError(f"cell is missing {missing}")
    syms = [c["symbols"]] if isinstance(c["symbols"], str) else list(c["symbols"])
    if not syms:
        raise ValueError("symbols is empty")
    c["symbols"] = sorted({s.strip().upper() for s in syms})
    c["start"], c["end"] = _date(c["start"]), _date(c["end"])
    if c["start"] >= c["end"]:
        raise ValueError(f"empty range [{c['start']}, {c['end']})")
    fg = c["feature_groups"]
    c["feature_groups"] = list(DEFAULT_FEATURE_GROUPS) if fg == "default" else [str(g) for g in fg]
    if set(c["primary"]) != {"name", "params"} or set(c["model"]) != {"meta", "primary"}:
        raise ValueError(
            f"primary needs {{name, params}} and model {{meta, primary}}; got {c['primary']}, {c['model']}"
        )
    c["primary"]["params"] = dict(c["primary"]["params"] or {})
    c["seed"] = int(c["seed"])
    if c["pwfo"] is not None:
        p = {"is_grid": None, "oos_grid": None, "expanding": False} | dict(c["pwfo"])
        if set(p) != {"is_grid", "oos_grid", "expanding"}:
            raise ValueError(f"pwfo takes is_grid, oos_grid, expanding; got {sorted(p)}")
        base = RunConfig.for_timeframe(c["timeframe"])  # the timeframe's default grid (SPEC §11.2)
        p["is_grid"] = [int(v) for v in (p["is_grid"] or base.PWFO_IS_GRID)]
        p["oos_grid"] = [int(v) for v in (p["oos_grid"] or base.PWFO_OOS_GRID)]
        p["expanding"] = bool(p["expanding"])
        c["pwfo"] = p
        if len(c["symbols"]) != 1:
            raise ValueError("a PWFO cell takes one symbol (multi-symbol PWFO is deferred to U11 Stage D)")
    owned = _CELL_OWNED & set(c["overrides"])
    if owned:
        raise ValueError(f"overrides may not set {sorted(owned)}: use the cell's own fields")
    from primaries import REGISTRY as PRIMARIES
    from primaries.mechanism import MechanismPrimary

    cls = PRIMARIES.get(c["primary"]["name"])
    if isinstance(cls, type) and issubclass(cls, MechanismPrimary):
        if c["timeframe"] not in cls.TIMEFRAMES:
            raise ValueError(f"primary {c['primary']['name']!r} runs on {list(cls.TIMEFRAMES)}, not {c['timeframe']!r}")
        p = cls(**c["primary"]["params"])
        c["overrides"] = {**p.config_overrides(c["timeframe"]), **c["overrides"]}
        if "sizer" not in raw:  # primary_config's default: the rule's own size hint
            c["sizer"] = "rule_size"
        missing = [g for g in p.needs_groups() if g not in c["feature_groups"]]
        if missing:
            raise ValueError(f"primary {c['primary']['name']!r} with {c['primary']['params']} reads the {missing} "
                             "feature group(s): add them to feature_groups")  # fmt: skip
    c["overrides"] = {k: list(v) if isinstance(v, tuple) else v for k, v in dict(c["overrides"]).items()}
    return json.loads(json.dumps(c, sort_keys=True))  # plain JSON types (tuples → lists, sorted keys)


def _set_dotted(d: dict, key: str, value) -> None:
    head, _, rest = key.partition(".")
    if not rest:
        d[head] = value
        return
    if head not in DEFAULT_CELL or not isinstance(DEFAULT_CELL[head], dict | type(None)):
        raise ValueError(f"grid key {key!r}: {head!r} is not a nested cell field")
    sub = d.setdefault(head, {})
    if sub is None:
        sub = d[head] = {}
    _set_dotted(sub, rest, value)


def expand(doc: dict) -> list[Cell]:
    """Cells of a spec document (module docstring), de-duplicated in order (identical configurations run once)."""
    extra = set(doc) - {"stage", "name", "defaults", "grid", "cells"}
    if extra:
        raise ValueError(f"unknown spec key(s) {sorted(extra)}")
    stage = str(doc.get("stage", ""))
    stage_rank(stage)
    defaults = doc.get("defaults") or {}
    grid = doc.get("grid") or {}
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"grid axis {k!r} must be a non-empty list")
    bases = [copy.deepcopy(defaults)]
    if doc.get("cells"):
        bases = []
        for ex in doc["cells"]:
            b = copy.deepcopy(defaults)
            for k, v in ex.items():
                _set_dotted(b, k, copy.deepcopy(v))
            bases.append(b)
    cells, seen = [], set()
    keys = list(grid)
    for base in bases:
        for combo in itertools.product(*(grid[k] for k in keys)):
            raw = copy.deepcopy(base)
            for k, v in zip(keys, combo, strict=True):
                _set_dotted(raw, k, copy.deepcopy(v))
            cell = Cell(normalize(raw), stage)
            cell.config()  # validate the RunConfig now (unknown model, bad override, ...)
            key = cell.spec_json()
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


def load_spec(path) -> tuple[dict, list[Cell]]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise TypeError(f"{path}: a spec is a YAML mapping")
    return doc, expand(doc)
