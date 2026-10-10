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
    variants:                            # reported only; 1 + len(variants) <= TRIAL_BUDGET
      - {label: no_cost, cost_model: slippage}
      - {label: tod, overrides.VOL_PROFILE: tod}   # dotted keys reach into the headline
    state_splits: [vix_tercile, macro_day]
    response: [sharpe, max_dd, mean_per_trade_bp, hit_rate]
    sample_splits: [{label: sector, instruments: [XLE, XLF]}, {label: 2016_2019, start: 2016-01-04, end: 2020-01-01}]
    benchmark: buy_and_hold_ew           # buy_and_hold_ew | buy_and_hold_er | cash
    floors: {min_net_ret: 0.02, min_net_ret_vs_benchmark: null, min_edge_to_cost: 3.0, max_dd: null}
    test: {alpha: 0.05, block_days: 21, n_boot: 2000, coherence_share: 0.667, seed: 0}
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
BENCHMARKS = ("buy_and_hold_ew", "buy_and_hold_er", "cash")
STATE_SPLITS = (
    "vix_tercile",
    "vix_median",
    "macro_day",
    "abs_move_tercile",
    "prior_day_sign",
    "day_of_week",
    "gamma_sign",
)  # fmt: skip  (gamma_sign: no gamma proxy yet, reported as unavailable)
RESPONSES = ("sharpe", "sortino", "calmar", "ret_ann", "vol_ann", "max_dd", "skew", "lpm2", "mean_per_trade_bp",
             "hit_rate", "crisis_return", "exposure")  # fmt: skip
TOP_KEYS = {"id", "mechanism", "registered", "instruments", "basket", "timeframe", "window", "headline", "variants",
            "state_splits", "response", "sample_splits", "benchmark", "floors", "test", "overlay", "TRIAL_BUDGET",
            "notes"}  # fmt: skip
HEADLINE_KEYS = {"primary", "exit", "sampler", "cost_model", "risk_profile", "sizer", "overrides"}
FLOOR_KEYS = {"min_net_ret": None, "min_net_ret_vs_benchmark": None, "min_edge_to_cost": None, "max_dd": None}
TEST_DEFAULTS = {"alpha": 0.05, "block_days": 21, "n_boot": 2000, "coherence_share": 0.667, "seed": 0}
HEADLINE = "headline"


@dataclass(frozen=True)
class Variant:
    label: str
    config: dict  # the headline dict with the variant's changes applied
    changes: dict  # the variant's own (dotted) keys


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


def _primary_cls(name: str):
    from primaries import REGISTRY
    from primaries.mechanism import MechanismPrimary

    cls = REGISTRY.get(name)
    if not (isinstance(cls, type) and issubclass(cls, MechanismPrimary)):
        raise TypeError(f"a family's primary must be a mechanism primary (SPEC §16), not {name!r}")
    return cls


def cell_raw(config: dict, symbols, timeframe: str, start: str, end: str) -> dict:
    """The runner cell (experiments.spec fields) of a variant config on `symbols` over [start, end)."""
    unknown = set(config) - HEADLINE_KEYS
    if unknown:
        raise ValueError(f"unknown headline / variant key(s) {sorted(unknown)}; expected {sorted(HEADLINE_KEYS)}")
    prim = config.get("primary")
    if isinstance(prim, str):
        prim = {"name": prim, "params": {}}
    if not isinstance(prim, dict) or "name" not in prim:
        raise ValueError(f"headline.primary needs a name; got {prim!r}")
    prim = {"name": prim["name"], "params": dict(prim.get("params") or {})}
    p = _primary_cls(prim["name"])(**prim["params"])
    default = p.config_overrides(timeframe)
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
        "feature_groups": ["wavelet_core", *[g for g in p.needs_groups() if g != "wavelet_core"]],
        "primary": prim,
        "model": {"meta": "none", "primary": "legacy"},
        "sizer": config.get("sizer", "rule_size"),
        "risk_profile": config.get("risk_profile", "none"),
        "overrides": over,
    }


def make_cells(fam: FamilySpec, config: dict, start: str, end: str, symbols=None) -> list[Cell]:
    """Stage-F cells of one configuration: one per instrument, or the basket's one portfolio cell."""
    syms = list(symbols or fam.instruments)
    groups = [syms] if fam.basket else [[s] for s in syms]
    cells = []
    for g in groups:
        cell = Cell(normalize(cell_raw(config, g, fam.timeframe, start, end)), RULE_STAGE)
        cell.config()  # validate the RunConfig now
        cells.append(cell)
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
    variants = [Variant(HEADLINE, copy.deepcopy(headline), {})]
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
        for k, x in changes.items():
            _set_dotted(cfg, k, x)
        variants.append(Variant(label, cfg, changes))
    labels = [v.label for v in variants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate variant labels {labels}")
    budget = int(doc.get("TRIAL_BUDGET", DEFAULT_BUDGET))
    if not 1 <= budget <= MAX_BUDGET:
        raise ValueError(f"TRIAL_BUDGET must be in [1, {MAX_BUDGET}], got {budget}")
    if len(variants) > budget:
        raise ValueError(f"{len(variants)} trials (headline + {len(variants) - 1} variants) exceed TRIAL_BUDGET "
                         f"{budget}")  # fmt: skip
    bench = doc.get("benchmark", "buy_and_hold_ew")
    if bench not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {bench!r}")
    floors = dict(FLOOR_KEYS)
    extra = set(doc.get("floors") or {}) - set(FLOOR_KEYS)
    if extra:
        raise ValueError(f"unknown floor(s) {sorted(extra)}; expected {sorted(FLOOR_KEYS)}")
    given = dict(doc.get("floors") or {})
    cls = _primary_cls((headline.get("primary") or {}).get("name") if isinstance(headline.get("primary"), dict)
                       else headline.get("primary"))  # fmt: skip
    prim_params = headline["primary"].get("params") or {} if isinstance(headline["primary"], dict) else {}
    long_only = bool(cls.LONG_ONLY or prim_params.get("long_only", False))
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
    reg = doc.get("registered")
    fam = FamilySpec(fid, doc, Path(path) if path else None, instruments, basket, tf, window, variants, bench,
                     floors, test, states, response, samples, budget, reg)  # fmt: skip
    fam.cells = {v.label: make_cells(fam, v.config, *window) for v in variants}
    seen: dict[str, str] = {}
    for v in variants:  # a variant identical to the headline or to another variant would pad the coherence share
        key = json.dumps(sorted(c.spec_json() for c in fam.cells[v.label]))
        if key in seen:
            raise ValueError(f"variant {v.label!r} runs the same configuration as {seen[key]!r}")
        seen[key] = v.label
    return fam


def load_family(path) -> FamilySpec:
    path = Path(path)
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")), path)


# ── Registration ─────────────────────────────────────────────────────────────


class RegistrationError(RuntimeError):
    """A family spec that is not pre-registered as committed (PLAN2 protocol 2)."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


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
