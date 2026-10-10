"""
Family specs (SPEC §17.1): `families/<id>.yaml` → a validated FamilySpec and its runner cells (stage F, rule cells).

    id: F3_overnight                     # = the file's stem; an amendment is a new file <id>.v2.yaml with a reason
    mechanism: "<two sentences>"
    registered: {sha: <git sha>, date: 2026-10-10}   # written by `python -m families register <file>`
    instruments: [SPY, QQQ]              # one cell per instrument, pooled equal-risk; or
    # basket: {symbols: [...]}           # one portfolio cell (F2, F10): its own stream
    timeframe: 5Min
    window: {start: 2016-01-04, end: 2025-10-01}      # optional; the development window (SPEC §11.1)
    headline:                            # the only tested variant
      primary: {name: overnight, params: {}}
      exit: {model: time, params: {...}}  # optional; params merge over the primary's default exit when the model is
      sampler: {params: {...}}           #   the default's; sampler params merge over the default EVENT_PARAMS
      cost_model: quotes                 # COST_MODEL
      risk_profile: none
      sizer: rule_size                   # optional (fixed | rule_size)
      overrides: {VOL_PROFILE: tod}      # optional, any other RunConfig field
      timeframe: 5Min                    # optional (U18): the cells' timeframe when it differs from the family's
      per_instrument:                    # optional (U18): a partial headline deep-merged into one instrument's cells
        TLT: {primary: {params: {window: month_end}}}
      model: {meta: rf_ldp_fast}         # optional (U18, F11): a meta-model makes the cells model cells (a walk-forward,
      pwfo: {is_grid: [504, 756], oos_grid: [21]}   # or PWFO with `pwfo`), any registered primary; feature_groups,
      feature_groups: [wavelet_core, trend]          # meta_train and seed as in experiment cells. Default: rule cells.
      legs:                              # optional (U18): one cell per instrument and leg (a partial headline each);
        fomc: {primary: {params: {release: fomc}}}         # an instrument's stream is the SUM of its legs' streams,
        cpi_nfp: {primary: {params: {release: cpi_nfp}}}   # so legs must never hold positions at the same time
    variants:                            # reported only; 1 + len(variants) <= TRIAL_BUDGET
      - {label: no_cost, cost_model: slippage}
      - {label: tod, overrides.VOL_PROFILE: tod}   # dotted keys reach into the headline
    state_splits: [vix_tercile, macro_day]
    response: [sharpe, max_dd, mean_per_trade_bp, hit_rate]
    sample_splits: [{label: sector, instruments: [XLE, XLF]}, {label: 2016_2019, start: 2016-01-04, end: 2020-01-01}]
    benchmark: constant_mix_er           # constant_mix_ew | constant_mix_er | buy_and_hold | cash (PLAN2's
                                         #   buy_and_hold_ew / buy_and_hold_er are aliases of the constant mixes)
    floors: {min_net_ret: 0.02, min_net_ret_vs_benchmark: null, min_edge_to_cost: 3.0, max_dd: null}
    test: {kind: overlay_alpha, sided: one, at_cost: 1.0,    # protocol v3 (U22, SPEC §22): kind sharpe_vs_benchmark
           alpha: 0.05, block_days: 21, n_boot: 5000,        #   (PLAN2) | overlay_alpha | marginal; sided one | two;
           coherence_share: 0.667, seed: 0}                  #   at_cost registered | <round-trip bp> | measured
    power: {n_days: 2400, vol_ann: 0.08, families: 4,       # required by overlay_alpha / marginal: the MDE line
            mde_alpha_bp_per_day: 3.17, expected_alpha_bp_per_day: 3.5, diagnostic: false}
    core: {family: F1, variant: headline, k: 1.0}            # marginal: the core family's stream (results dir)
    account: {kind: margin, equity: 30000, locate_bps: 0}    # the registered account (risk/account.py); default:
                                                             #   a margin account at the headline cells' INIT_CASH
    overlay: {...}                       # stage G (U19); stored, not read here
    TRIAL_BUDGET: 12

Registration (PLAN2 protocol 2). `register` records the HEAD commit in which the spec file is committed and clean;
`check_registered` refuses a spec whose `registered.sha` is missing, not an ancestor of HEAD, or whose content at that
commit (without the `registered` key) differs from the file, and a file that is untracked or has uncommitted changes.
The run therefore always uses the spec as committed before its first ledger row.
"""

import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from experiments.spec import RULE_STAGE, Cell, normalize

ROOT = Path(__file__).resolve().parent.parent
FAMILY_DIR = ROOT / "families"
DEV_WINDOW = ("2016-01-04", "2025-10-01")  # SPEC §11.1, [start, end)
QUASI_WINDOW = ("2025-10-01", "2026-10-01")  # the contaminated quasi-holdout slice: reported, never a gate
DEFAULT_BUDGET = 12
MAX_BUDGET = 12  # PLAN2 protocol 3: no family spec (or amendment) may grant itself more variants than this
BENCHMARKS = ("constant_mix_ew", "constant_mix_er", "buy_and_hold", "cash")
BENCHMARK_ALIASES = {"buy_and_hold_ew": "constant_mix_ew", "buy_and_hold_er": "constant_mix_er"}  # PLAN2 names
STATE_SPLITS = (
    "vix_tercile",
    "vix_median",
    "macro_day",
    "abs_move_tercile",
    "prior_day_sign",
    "day_of_week",
    "year",
    "gamma_sign",
)  # fmt: skip  (gamma_sign: no gamma proxy yet, reported as unavailable)
RESPONSES = ("sharpe", "sortino", "calmar", "ret_ann", "vol_ann", "max_dd", "skew", "lpm2", "mean_per_trade_bp",
             "hit_rate", "crisis_return", "exposure", "worst_day", "longest_flat_run")  # fmt: skip
TOP_KEYS = {"id", "mechanism", "registered", "instruments", "basket", "timeframe", "window", "headline", "variants",
            "state_splits", "response", "sample_splits", "benchmark", "floors", "test", "overlay", "TRIAL_BUDGET",
            "notes", "power", "core", "account"}  # fmt: skip
TEST_KINDS = ("sharpe_vs_benchmark", "overlay_alpha", "marginal")
POWER_KEYS = {"n_days", "vol_ann", "families", "mde_alpha_bp_per_day", "expected_alpha_bp_per_day", "diagnostic"}
MDE_TOLERANCE = 0.10  # a recorded MDE must agree with families.power.mde_alpha within this fraction
ACCOUNT_KEYS = {"kind", "equity", "locate_bps"}
HEADLINE_KEYS = {"primary", "exit", "sampler", "cost_model", "risk_profile", "sizer", "overrides", "timeframe",
                 "per_instrument", "legs", "model", "feature_groups", "pwfo", "meta_train", "seed"}  # fmt: skip
PATCH_KEYS = HEADLINE_KEYS - {"per_instrument", "timeframe", "legs"}  # what a per-instrument or leg patch may set
FLOOR_KEYS = {"min_net_ret": None, "min_net_ret_vs_benchmark": None, "min_edge_to_cost": None, "max_dd": None}
TEST_DEFAULTS = {"alpha": 0.05, "block_days": 21, "n_boot": 5000, "coherence_share": 0.667, "seed": 0,
                 "kind": "sharpe_vs_benchmark", "sided": "two", "at_cost": "registered"}  # fmt: skip
HEADLINE = "headline"


@dataclass(frozen=True)
class Variant:
    label: str
    config: dict  # the headline dict with the variant's changes applied
    changes: dict  # the variant's own (dotted) keys
    core: dict | None = None  # the `core` block with the variant's `core.*` changes applied (marginal kind)


@dataclass
class FamilySpec:
    id: str
    doc: dict
    path: Path | None
    instruments: tuple[str, ...]
    basket: bool
    timeframe: str
    window: tuple[str, str]
    variants: list[Variant]  # the headline first
    benchmark: str
    floors: dict
    test: dict
    state_splits: list[str]
    response: list[str]
    sample_splits: list[dict]
    budget: int
    registered: dict | None = None
    cells: dict[str, list[Cell]] = field(default_factory=dict)  # variant label → its cells (dev window)
    power: dict | None = None  # the MDE line (protocol v3)
    core: dict | None = None  # the core family of a `marginal` test
    account: dict = field(default_factory=dict)  # the registered account (risk.account)

    @property
    def base_id(self) -> str:
        """The family an amendment belongs to (`F1.v2` → `F1`): amendments share its budget."""
        return self.id.split(".v")[0]

    @property
    def headline(self) -> Variant:
        return self.variants[0]

    def trial_key(self, label: str) -> str:
        """`<base id>/<configuration hash>`: the variant's cells (symbols, window, primary, exit, costs, every override)
        without code or data, so an amendment that reuses a label with a new configuration is a new trial, and the same
        configuration re-run (new code, a re-registered amendment) is not."""
        cfg = sorted(c.spec_json() for c in self.cells[label])
        return f"{self.base_id}/{hashlib.sha256(json.dumps(cfg).encode()).hexdigest()[:12]}"


def _date(x) -> str:
    return str(pd.Timestamp(str(x)).date())


def _set_dotted(d: dict, key: str, value) -> None:
    head, _, rest = key.partition(".")
    if not rest:
        d[head] = copy.deepcopy(value)
        return
    sub = d.get(head)
    if sub is None:
        sub = d[head] = {}
    if not isinstance(sub, dict):
        raise TypeError(f"variant key {key!r}: {head!r} is not a mapping")
    _set_dotted(sub, rest, value)


def _primary_cls(name: str, rule: bool = True):
    """The primary's class; a rule cell (no meta-model) needs a mechanism primary (SPEC §16)."""
    from primaries import REGISTRY
    from primaries.mechanism import MechanismPrimary

    cls = REGISTRY.get(name)
    if cls is None:
        raise TypeError(f"unknown primary {name!r}")
    if rule and not (isinstance(cls, type) and issubclass(cls, MechanismPrimary)):
        raise TypeError(f"a family's rule primary must be a mechanism primary (SPEC §16), not {name!r}; a family "
                        "headline with a meta-model (`model`) may use any primary")  # fmt: skip
    return cls


def _model(config: dict) -> dict:
    m = config.get("model") or {}
    if not isinstance(m, dict) or set(m) - {"meta", "primary"}:
        raise ValueError(f"model takes meta and primary; got {m!r}")
    return {"meta": m.get("meta", "none"), "primary": m.get("primary", "legacy")}


def _merge(base: dict, patch: dict) -> dict:
    """`base` with `patch` deep-merged into it (mappings merge, anything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _patch(patch, what: str) -> dict:
    if not isinstance(patch, dict) or set(patch) - PATCH_KEYS:
        raise ValueError(f"{what} is a partial headline without per_instrument / legs / timeframe; got {patch!r}")
    return patch


def for_symbol(config: dict, symbol: str | None) -> list[dict]:
    """
    A variant config as it runs on `symbol`, one config per leg (one without `legs`): the base, then its
    `per_instrument[symbol]` patch, then the leg's patch (legs in label order); the mappings are dropped.
    """
    per = config.get("per_instrument") or {}
    if not isinstance(per, dict):
        raise TypeError("per_instrument maps an instrument to a partial headline")
    legs = config.get("legs")
    if legs is not None and (not isinstance(legs, dict) or not legs):
        raise ValueError(f"legs maps a leg label to a partial headline (at least one); got {legs!r}")
    base = {k: v for k, v in config.items() if k not in ("per_instrument", "legs")}
    patch = per.get(symbol) if symbol is not None else None
    if patch is not None:
        base = _merge(base, _patch(patch, f"per_instrument[{symbol}]"))
    if legs is None:
        return [base]
    return [_merge(base, _patch(legs[k], f"leg {k!r}")) for k in sorted(legs)]


def cell_raw(config: dict, symbols, timeframe: str, start: str, end: str) -> dict:
    """The runner cell (experiments.spec fields) of a variant config on `symbols` over [start, end)."""
    unknown = set(config) - HEADLINE_KEYS
    if unknown:
        raise ValueError(f"unknown headline / variant key(s) {sorted(unknown)}; expected {sorted(HEADLINE_KEYS)}")
    if config.get("per_instrument") or config.get("legs"):
        raise ValueError("cell_raw takes a config resolved for its symbol and leg (for_symbol)")
    timeframe = str(config.get("timeframe") or timeframe)
    prim = config.get("primary")
    if isinstance(prim, str):
        prim = {"name": prim, "params": {}}
    if not isinstance(prim, dict) or "name" not in prim:
        raise ValueError(f"headline.primary needs a name; got {prim!r}")
    prim = {"name": prim["name"], "params": dict(prim.get("params") or {})}
    from primaries.mechanism import MechanismPrimary

    mm = _model(config)
    rule = mm["meta"] == "none"
    cls = _primary_cls(prim["name"], rule)
    mech = isinstance(cls, type) and issubclass(cls, MechanismPrimary)
    if rule and any(config.get(k) is not None for k in ("pwfo", "feature_groups", "meta_train")):
        raise ValueError("pwfo, feature_groups and meta_train need a meta-model (`model.meta`); a rule cell has none")
    p = cls(**prim["params"]) if mech else None
    default = p.config_overrides(timeframe) if mech else {}
    over = dict(config.get("overrides") or {})
    for owned in ("COST_MODEL", "EXIT_MODEL", "EXIT_PARAMS", "EVENT_PARAMS"):
        if owned in over:
            raise ValueError(f"set {owned} through the headline's cost_model / exit / sampler keys, not overrides")
    if config.get("sampler") is not None:
        extra = set(config["sampler"]) - {"params"}
        if extra:
            raise ValueError(f"sampler takes params only; got {sorted(extra)}")
        over["EVENT_PARAMS"] = {**default.get("EVENT_PARAMS", {}), **(config["sampler"].get("params") or {})}
    if config.get("exit") is not None:
        ex = config["exit"]
        extra = set(ex) - {"model", "params"}
        if extra:
            raise ValueError(f"exit takes model and params; got {sorted(extra)}")
        model = ex.get("model", default.get("EXIT_MODEL"))
        base = default.get("EXIT_PARAMS", {}) if model == default.get("EXIT_MODEL") else {}
        over["EXIT_MODEL"] = model
        over["EXIT_PARAMS"] = {**base, **(ex.get("params") or {})}
    over["COST_MODEL"] = config.get("cost_model", "quotes")
    return {
        "symbols": list(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "feature_groups": (
            list(config["feature_groups"])
            if config.get("feature_groups") is not None
            else ["wavelet_core", *[g for g in (p.needs_groups() if mech else ()) if g != "wavelet_core"]]
        ),
        "primary": prim,
        "model": mm,
        "meta_train": config.get("meta_train", "oof"),
        "sizer": config.get("sizer", "rule_size" if mech else "fixed"),
        "risk_profile": config.get("risk_profile", "none"),
        "pwfo": None if config.get("pwfo") is None else {"expanding": False, **config["pwfo"]},
        "seed": config.get("seed", 42),
        "overrides": over,
    }


def make_cells(fam: FamilySpec, config: dict, start: str, end: str, symbols=None) -> list[Cell]:
    """Stage-F cells of one configuration: one per instrument (and leg), or the basket's one portfolio cell."""
    syms = list(symbols or fam.instruments)
    groups = [syms] if fam.basket else [[s] for s in syms]
    if fam.basket and (config.get("per_instrument") or config.get("legs")):
        raise ValueError("per_instrument and legs are for pooled instruments; a basket is one portfolio cell")
    cells = []
    for g in groups:
        for cfg in for_symbol(config, None if fam.basket else g[0]):
            cell = Cell(normalize(cell_raw(cfg, g, fam.timeframe, start, end)), RULE_STAGE)
            cell.config()  # validate the RunConfig now
            cells.append(cell)
    keys = [c.spec_json() for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("two legs of one instrument run the same cell: a leg must change the configuration")
    return cells


def parse(doc: dict, path: Path | None = None) -> FamilySpec:
    """Validate a family document and build every dev-window cell (an invalid variant fails here)."""
    if not isinstance(doc, dict):
        raise TypeError("a family spec is a YAML mapping")
    unknown = set(doc) - TOP_KEYS
    if unknown:
        raise ValueError(f"unknown family key(s) {sorted(unknown)}; expected {sorted(TOP_KEYS)}")
    fid = str(doc.get("id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_]+(\.v\d+)?", fid):
        raise ValueError(f"id {fid!r}: letters, digits, '_' and an optional amendment suffix '.v<k>'")
    if path is not None and Path(path).stem != fid:
        raise ValueError(f"id {fid!r} must equal the file's stem {Path(path).stem!r}")
    if not str(doc.get("mechanism") or "").strip():
        raise ValueError("a family states its mechanism")
    if ("instruments" in doc) == ("basket" in doc):
        raise ValueError("give exactly one of instruments (pooled) or basket (one portfolio cell)")
    basket = "basket" in doc
    syms = doc["basket"].get("symbols") if basket else doc["instruments"]
    if basket and set(doc["basket"]) - {"symbols"}:
        raise ValueError("basket takes symbols only (its weighting is the cell's sizing and risk profile)")
    if not syms or not isinstance(syms, list):
        raise ValueError("instruments / basket.symbols must be a non-empty list")
    up = [str(s).strip().upper() for s in syms]
    if len(set(up)) != len(up):
        raise ValueError(f"duplicate instruments {up}: a pooled stream would count one twice")
    instruments = tuple(sorted(up))
    tf = str(doc.get("timeframe") or "")
    win = doc.get("window") or {}
    if set(win) - {"start", "end"}:
        raise ValueError("window takes start and end")
    window = (_date(win.get("start", DEV_WINDOW[0])), _date(win.get("end", DEV_WINDOW[1])))
    if window[1] > DEV_WINDOW[1]:
        raise ValueError(f"window ends {window[1]}, after the development window's end {DEV_WINDOW[1]}")
    headline = doc.get("headline")
    if not isinstance(headline, dict):
        raise TypeError("a family has a headline mapping")
    core = doc.get("core")
    variants = [Variant(HEADLINE, copy.deepcopy(headline), {}, copy.deepcopy(core))]
    for v in doc.get("variants") or []:
        if not isinstance(v, dict) or not isinstance(v.get("label"), str) or not v["label"]:
            raise ValueError(f"every variant has a string label (quote one YAML reads as a number); got {v!r}")
        label = str(v["label"])
        changes = {k: x for k, x in v.items() if k != "label"}
        if not changes:
            raise ValueError(f"variant {label!r} changes nothing")
        if any(k.split(".")[0] in ("instruments", "basket", "timeframe_symbols") for k in changes):
            raise ValueError(f"variant {label!r}: other instruments are a sample split, not a variant")
        cfg = copy.deepcopy(headline)
        vcore = copy.deepcopy(core)
        for k, x in changes.items():
            if k == "core" or k.startswith("core."):
                if core is None:
                    raise ValueError(f"variant {label!r} changes `core`, which the family does not set")
                if k == "core":
                    raise ValueError(f"variant {label!r}: change core.k (or core.variant), not the whole core block")
                _set_dotted(vcore, k.partition(".")[2], x)
            else:
                _set_dotted(cfg, k, x)
        variants.append(Variant(label, cfg, changes, vcore))
    labels = [v.label for v in variants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate variant labels {labels}")
    budget = int(doc.get("TRIAL_BUDGET", DEFAULT_BUDGET))
    if not 1 <= budget <= MAX_BUDGET:
        raise ValueError(f"TRIAL_BUDGET must be in [1, {MAX_BUDGET}], got {budget}")
    if len(variants) > budget:
        raise ValueError(f"{len(variants)} trials (headline + {len(variants) - 1} variants) exceed TRIAL_BUDGET "
                         f"{budget}")  # fmt: skip
    bench = doc.get("benchmark", "constant_mix_ew")
    bench = BENCHMARK_ALIASES.get(bench, bench)
    if bench not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS} (or a PLAN2 alias), got {bench!r}")
    floors = dict(FLOOR_KEYS)
    extra = set(doc.get("floors") or {}) - set(FLOOR_KEYS)
    if extra:
        raise ValueError(f"unknown floor(s) {sorted(extra)}; expected {sorted(FLOOR_KEYS)}")
    given = dict(doc.get("floors") or {})
    cls = _primary_cls((headline.get("primary") or {}).get("name") if isinstance(headline.get("primary"), dict)
                       else headline.get("primary"), _model(headline)["meta"] == "none")  # fmt: skip
    prim_params = headline["primary"].get("params") or {} if isinstance(headline["primary"], dict) else {}
    long_only = bool(getattr(cls, "LONG_ONLY", False) or prim_params.get("long_only", False))
    if "min_net_ret" not in given and "min_net_ret_vs_benchmark" not in given:  # PLAN2 U17 defaults
        given["min_net_ret_vs_benchmark" if long_only else "min_net_ret"] = 0.8 if long_only else 0.02
    given.setdefault("min_edge_to_cost", 3.0)
    floors |= given
    if floors["min_net_ret"] is None and floors["min_net_ret_vs_benchmark"] is None:
        raise ValueError("a family needs a net-return floor (min_net_ret or min_net_ret_vs_benchmark)")
    if floors["min_edge_to_cost"] is None or not floors["min_edge_to_cost"] > 0:
        raise ValueError("a family needs min_edge_to_cost > 0 (the magnitude floor; default 3)")
    if floors["max_dd"] is not None and not 0 < floors["max_dd"] < 1:
        raise ValueError("floors.max_dd is a drawdown magnitude in (0, 1)")
    test = dict(TEST_DEFAULTS)
    extra = set(doc.get("test") or {}) - set(TEST_DEFAULTS)
    if extra:
        raise ValueError(f"unknown test setting(s) {sorted(extra)}; expected {sorted(TEST_DEFAULTS)}")
    test |= doc.get("test") or {}
    if not 0 < test["alpha"] < 1 or not 0 < test["coherence_share"] <= 1 or test["n_boot"] < 99:
        raise ValueError(f"test settings out of range: {test}")
    if test["kind"] not in TEST_KINDS:
        raise ValueError(f"test.kind must be one of {TEST_KINDS}, got {test['kind']!r}")
    if test["sided"] not in ("one", "two"):
        raise ValueError(f"test.sided must be 'one' or 'two', got {test['sided']!r}")
    ac = test["at_cost"]
    if not (ac in ("registered", "measured") or (isinstance(ac, (int, float)) and not isinstance(ac, bool)
                                                 and ac > 0)):  # fmt: skip
        raise ValueError(f"test.at_cost must be 'registered', 'measured' or a round-trip cost in bp > 0, got {ac!r}")
    power = _power(doc.get("power"), test)
    if test["kind"] in ("overlay_alpha", "marginal") and power is None:
        raise ValueError(f"test.kind {test['kind']!r} needs the `power` block (the MDE line, PLAN3 §4.5)")
    if test["kind"] == "marginal":
        if not isinstance(core, dict) or not core.get("family"):
            raise ValueError("test.kind 'marginal' needs core: {family, variant (headline), k (1.0)}")
        extra = set(core) - {"family", "variant", "k"}
        if extra:
            raise ValueError(f"core takes family, variant and k; got {sorted(extra)}")
        for v in variants:
            kk = (v.core or {}).get("k", 1.0)
            if isinstance(kk, bool) or not isinstance(kk, (int, float)) or not kk > 0:
                raise ValueError(f"variant {v.label!r}: core.k must be a number > 0, got {kk!r}")
    elif core is not None:
        raise ValueError("`core` is read by test.kind 'marginal' only")
    states = list(doc.get("state_splits") or [])
    bad = [s for s in states if s not in STATE_SPLITS]
    if bad:
        raise ValueError(f"unknown state split(s) {bad}; expected {STATE_SPLITS}")
    response = list(doc.get("response") or ["sharpe", "max_dd"])
    bad = [r for r in response if r not in RESPONSES]
    if bad:
        raise ValueError(f"unknown response measure(s) {bad}; expected {RESPONSES}")
    samples = []
    for s in doc.get("sample_splits") or []:
        if not isinstance(s, dict) or not isinstance(s.get("label"), str) or set(s) - {"label", "instruments", "start",
                                                                                      "end"}:  # fmt: skip
            raise ValueError(
                f"a sample split is {{label, instruments}} or {{label, start, end}} with a string label "
                f"(quote one YAML reads as a number); got {s!r}"
            )
        if ("instruments" in s) == ("start" in s or "end" in s):
            raise ValueError(f"sample split {s['label']!r}: give instruments or a date range, not both / neither")
        s = dict(s)
        if "instruments" in s:
            s["instruments"] = sorted({str(x).upper() for x in s["instruments"]})
        else:
            s["start"], s["end"] = _date(s.get("start", window[0])), _date(s.get("end", window[1]))
        samples.append(s)
    known = set(instruments) | {x for s in samples for x in s.get("instruments", ())}
    for v in variants:
        stray = set(v.config.get("per_instrument") or {}) - known
        if stray:
            raise ValueError(f"variant {v.label!r}: per_instrument names {sorted(stray)}, not an instrument of the "
                             "family or of a sample split")  # fmt: skip
    reg = doc.get("registered")
    fam = FamilySpec(fid, doc, Path(path) if path else None, instruments, basket, tf, window, variants, bench,
                     floors, test, states, response, samples, budget, reg, power=power, core=core)  # fmt: skip
    fam.cells = {v.label: make_cells(fam, v.config, *window) for v in variants}
    fam.account = _account(doc.get("account"), fam.cells[HEADLINE][0].config().INIT_CASH)
    seen: dict[str, str] = {}
    for v in variants:  # a variant identical to the headline or to another variant would pad the coherence share
        key = json.dumps([sorted(c.spec_json() for c in fam.cells[v.label]), v.core], sort_keys=True)
        if key in seen:
            raise ValueError(f"variant {v.label!r} runs the same configuration as {seen[key]!r}")
        seen[key] = v.label
    return fam


def _power(block, test: dict) -> dict | None:
    """The MDE line (PLAN3 §4.5): validated against families.power.mde_alpha; a family expecting less than its MDE
    must declare itself a diagnostic."""
    if block is None:
        return None
    from families.power import mde_alpha

    if (
        not isinstance(block, dict)
        or set(block) - POWER_KEYS
        or not {"n_days", "vol_ann", "families", "mde_alpha_bp_per_day", "expected_alpha_bp_per_day"} <= set(block)
    ):
        raise ValueError(f"power takes n_days, vol_ann, families, mde_alpha_bp_per_day, expected_alpha_bp_per_day "
                         f"and diagnostic; got {block!r}")  # fmt: skip
    n, vol, fams = int(block["n_days"]), float(block["vol_ann"]), int(block["families"])
    ref = mde_alpha(n, vol, alpha=test["alpha"], families=fams, sided=test["sided"])
    rec, exp = float(block["mde_alpha_bp_per_day"]), float(block["expected_alpha_bp_per_day"])
    if abs(rec - ref["mde_bp_per_day"]) > MDE_TOLERANCE * ref["mde_bp_per_day"]:
        raise ValueError(f"power.mde_alpha_bp_per_day {rec} disagrees with families.power.mde_alpha "
                         f"{ref['mde_bp_per_day']:.3f} (n {n}, vol {vol}, {fams} families, {test['sided']}-sided)")  # fmt: skip
    diag = bool(block.get("diagnostic", False))
    if exp < ref["mde_bp_per_day"] and not diag:
        raise ValueError(f"expected alpha {exp} bp/day is below the MDE {ref['mde_bp_per_day']:.2f}: register the "
                         "family as a diagnostic (power.diagnostic: true), not as a test")  # fmt: skip
    return {**ref, "mde_alpha_bp_per_day": rec, "expected_alpha_bp_per_day": exp, "diagnostic": diag}


def _account(block, init_cash: float) -> dict:
    """The registered account: a margin account at the headline cells' INIT_CASH unless the spec says otherwise."""
    block = dict(block or {})
    extra = set(block) - ACCOUNT_KEYS
    if extra:
        raise ValueError(f"account takes kind, equity and locate_bps; got {sorted(extra)}")
    kind = block.get("kind", "margin")
    if kind not in ("margin", "cash"):
        raise ValueError(f"account.kind must be 'margin' or 'cash', got {kind!r}")
    eq = float(block.get("equity", init_cash))
    locate = float(block.get("locate_bps", 0.0))
    if not eq > 0 or locate < 0:
        raise ValueError("account.equity must be > 0 and locate_bps >= 0")
    return {"kind": kind, "equity": eq, "locate_bps": locate, "name": f"{kind}_{eq / 1000:g}k"}


def load_family(path) -> FamilySpec:
    path = Path(path)
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")), path)


# ── Registration ─────────────────────────────────────────────────────────────


class RegistrationError(RuntimeError):
    """A family spec that is not pre-registered as committed (PLAN2 protocol 2)."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # UTF-8, as the spec files are read: the locale's codec (cp1252 on Windows) would make a non-ASCII spec (an en
    # dash in a mechanism) differ from its own registered version
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          check=check)  # fmt: skip


def _strip(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "registered"}


def _committed_clean(path: Path, repo: Path) -> str:
    rel = path.resolve().relative_to(repo.resolve()).as_posix()
    if _git(repo, "ls-files", "--error-unmatch", rel, check=False).returncode:
        raise RegistrationError(f"{rel} is not committed: commit the spec before registering / running it")
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        raise RegistrationError(f"{rel} has uncommitted changes: the run must use the spec as committed")
    return rel


def check_registered(path, repo: Path = ROOT) -> dict:
    """The spec's `registered` block after the checks in the module docstring; raises RegistrationError."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    reg = doc.get("registered")
    if not isinstance(reg, dict) or not reg.get("sha") or not reg.get("date"):
        raise RegistrationError(f"{path.name} is not registered: run `python -m families register {path}` and commit")
    rel = _committed_clean(path, repo)
    sha = str(reg["sha"])
    if _git(repo, "merge-base", "--is-ancestor", sha, "HEAD", check=False).returncode:
        raise RegistrationError(f"{path.name}: registered sha {sha} is not an ancestor of HEAD")
    shown = _git(repo, "show", f"{sha}:{rel}", check=False)
    if shown.returncode:
        raise RegistrationError(f"{path.name}: the file does not exist at the registered commit {sha}")
    if _strip(yaml.safe_load(shown.stdout) or {}) != _strip(doc):
        raise RegistrationError(f"{path.name} differs from its registered version at {sha}: an amendment is a new "
                                "file (<id>.v2.yaml) with a stated reason")  # fmt: skip
    return reg


def register(path, repo: Path = ROOT, today: str | None = None) -> dict:
    """Write `registered: {sha: HEAD, date}` into a committed, clean spec (the next commit records it)."""
    path = Path(path)
    load_family(path)  # a spec that does not validate is not registered
    _committed_clean(path, repo)
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if doc.get("registered"):
        raise RegistrationError(f"{path.name} is already registered at {doc['registered']}")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    reg = {"sha": sha, "date": today or str(dt.date.today())}
    line = f"registered: {{sha: '{sha}', date: '{reg['date']}'}}\n"
    lines = text.splitlines(keepends=True)
    at = next((i + 1 for i, ln in enumerate(lines) if re.match(r"id\s*:", ln)), 0)
    path.write_text("".join(lines[:at] + [line] + lines[at:]), encoding="utf-8", newline="\n")
    if _strip(yaml.safe_load(path.read_text(encoding="utf-8"))) != _strip(doc):
        path.write_text(text, encoding="utf-8", newline="\n")
        raise RegistrationError("could not insert the registered line without changing the spec")
    return reg
