"""
Run one family (SPEC §17, PLAN2 U17): registered spec → stage-F rule cells through experiments.runner → pooled variant
streams → one ledger row per variant (the counted trial) → the family test → results/families/<id>/.

Cells (all stage F rule cells, n_trials 0 each, run in one runner batch):
- every variant × instrument (or the basket's one portfolio cell) over the family window;
- the headline on each instrument sample split (reported);
- the headline over [window start, 2026-10-01) for the quasi-holdout slice (reported; its days before 2025-10-01
  are not used: the test reads the dev-window cells only).

Trial accounting (SPEC §11.1): a variant is one trial whatever its instrument count. Each variant gets a ledger row
(kind "family_variant", n_trials 1, its pooled stream's statistics, `trial_key` = <family>/<configuration hash>, members = its cell
hashes); its artifacts (daily_returns.csv, result.json) live under <root>/cells/<variant hash>/. Budgets count
distinct trial keys with a counted status in stages F / G: per family (amendments `<id>.v<k>` share their family's)
≤ TRIAL_BUDGET, and program-wide ≤ families/program.yaml's caps; a run that would exceed one is refused before any
cell runs. A ledger already holding rows of this id under another registered sha is refused (an amendment is a new
file).
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments import ledger as L
from families import test as T
from families.spec import HEADLINE, QUASI_WINDOW, ROOT, FamilySpec, check_registered, load_family, make_cells

PROGRAM = ROOT / "families" / "program.yaml"
DEFAULT_OUT = ROOT / "results" / "families"
FAMILY_STAGES_COUNTED = ("F", "G")
KIND = "family_variant"


class BudgetError(RuntimeError):
    """A family run that would exceed its TRIAL_BUDGET or the program caps (PLAN2 protocol 3)."""


# ── Budgets ──────────────────────────────────────────────────────────────────


def program_caps(path=PROGRAM) -> dict:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {"max_families": int(doc["max_families"]), "max_trials": int(doc["max_trials"])}


def counted_keys(rows: list[dict]) -> dict[str, set[str]]:
    """Base family id → the distinct trial keys of its counted variant rows in stages F / G."""
    out: dict[str, set[str]] = {}
    for r in L.trials(rows):
        if r.get("kind") == KIND and r.get("stage") in FAMILY_STAGES_COUNTED and r.get("status") in L.COUNTED:
            out.setdefault(r["base_family"], set()).add(r["trial_key"])
    return out


def check_budget(fam: FamilySpec, rows: list[dict], caps: dict) -> dict:
    """Refuse a run past the family's budget or the program caps; returns the accounting after the run."""
    for r in L.trials(rows):
        if r.get("kind") == KIND and r.get("family") == fam.id and fam.registered and \
                r.get("registered_sha") != str(fam.registered.get("sha")):  # fmt: skip
            raise BudgetError(f"{fam.id} already has ledger rows under registered sha {r.get('registered_sha')}: "
                              "an amendment is a new file (<id>.v2.yaml)")  # fmt: skip
    keys = counted_keys(rows)
    # an amendment cannot grant its family more trials than an earlier version had
    budget = min([fam.budget] + [int(r["budget"]) for r in L.trials(rows) if r.get("kind") == KIND
                                 and r.get("base_family") == fam.base_id and r.get("budget") is not None])  # fmt: skip
    mine = keys.get(fam.base_id, set()) | {fam.trial_key(v.label) for v in fam.variants}
    if len(mine) > budget:
        raise BudgetError(f"{fam.base_id}: {len(mine)} trials (with earlier versions' {len(keys.get(fam.base_id, ()))})"
                          f" exceed TRIAL_BUDGET {budget}")  # fmt: skip
    keys[fam.base_id] = mine
    n_fam, n_trials = len(keys), sum(len(v) for v in keys.values())
    if n_fam > caps["max_families"]:
        raise BudgetError(f"{n_fam} families exceed the program cap {caps['max_families']}")
    if n_trials > caps["max_trials"]:
        raise BudgetError(f"{n_trials} program trials exceed the cap {caps['max_trials']}")
    return {"family_trials": len(mine), "family_budget": budget, "program_families": n_fam,
            "program_trials": n_trials, **caps}  # fmt: skip


# ── Data ─────────────────────────────────────────────────────────────────────


def bh_returns(source, symbols, timeframe: str, start: str, end: str) -> dict[str, pd.Series]:
    """Buy-and-hold daily returns (last close of each NY session) of each symbol from the cells' own bars."""
    out = {}
    for s in symbols:
        df = source.bars(s, timeframe, start, end, allow_holdout=False)
        day = df.index.tz_convert("America/New_York").normalize() if df.index.tz is not None else df.index.normalize()
        out[s] = df["close"].groupby(day).last().pct_change().dropna()
    return out


def load_state(source, start: str, end: str) -> pd.DataFrame | None:
    """Day state for the reported splits: the previous VIX close (`vix_prev`) and the macro-day flag (FOMC, CPI,
    NFP). None for an input the source cannot give (tests' fake sources)."""
    days = pd.bdate_range(start, end, inclusive="left")
    out = pd.DataFrame(index=days)
    try:
        vix = source.exo("cboe", "VIX", start, end, allow_holdout=False)["value"]
        vix.index = pd.DatetimeIndex(vix.index).normalize()
        out["vix_prev"] = vix.reindex(days.union(vix.index)).ffill().shift(1).reindex(days)
    except Exception as e:  # noqa: BLE001 — a reported split only: say so in the result
        print(f"[FAM]  no VIX for the state splits ({type(e).__name__}: {e})")
    try:
        from data.events import read_events

        ev = read_events()
        macro = pd.DatetimeIndex(pd.to_datetime(ev.loc[ev["kind"].isin(["FOMC", "CPI", "NFP"]), "date"]))
        out["macro"] = days.isin(macro)
    except Exception as e:  # noqa: BLE001
        print(f"[FAM]  no event calendar for the state splits ({type(e).__name__}: {e})")
    return out if out.shape[1] else None


def _read(root: Path, h: str, name: str) -> pd.DataFrame | None:
    p = root / "cells" / h / name
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


def _stream(root: Path, h: str) -> pd.Series:
    s = _read(root, h, "daily_returns.csv")["ret"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    return s


# ── Run ──────────────────────────────────────────────────────────────────────


def families_code_hash() -> str:
    """The family layer's own source (pooling, statistics): a change re-records the variant rows."""
    h = hashlib.sha256()
    for path in sorted((ROOT / "families").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _vhash(fam: FamilySpec, label: str, members: list[str]) -> str:
    return hashlib.sha256(json.dumps([fam.id, label, sorted(members), families_code_hash()]).encode()).hexdigest()[:16]


def _pooled(fam: FamilySpec, cells, rows_by_hash, hashes, root, weights) -> dict:
    """Members → status, pooled stream, summed cost totals, trades, per-instrument streams."""
    members = [hashes[(c.stage, c.spec_json())] for c in cells]
    stat = [rows_by_hash.get(h, {}).get("status", "error") for h in members]
    out = {"members": members, "status": "ok" if all(s == "ok" for s in stat) else
           ("error" if "error" in stat else "no_fit")}  # fmt: skip
    if out["status"] != "ok":
        return out
    streams = {c.symbols[0] if not fam.basket else "_basket": _stream(root, h) for c, h in zip(cells, members)}
    out["per_instrument"] = {} if fam.basket else streams
    out["stream"] = streams["_basket"].rename("ret") if fam.basket else T.pool(streams, weights)
    # cash totals per unit of each member's starting capital, mixed with the pooled stream's weights (a basket: its own)
    w = [1.0] if fam.basket else [float(weights[c.symbols[0]]) for c in cells]
    out["costs"] = {k: float(sum(wi * rows_by_hash[h][k] / rows_by_hash[h]["init_cash"] for wi, h in zip(w, members)))
                    for k in ("pnl", "cost_paid", "traded_notional")}  # fmt: skip
    out["skipped_segments"] = {k: v for h in members for k, v in (rows_by_hash[h].get("skipped_segments") or {}).items()
                               if v}  # fmt: skip
    tr = [_read(root, h, "trades.csv") for h in members]
    out["trades"] = pd.concat([t for t in tr if t is not None and len(t)]) if any(
        t is not None and len(t) for t in tr) else None  # fmt: skip
    return out


def run_family(
    path,
    *,
    ledger: L.Ledger,
    root,
    out_dir=DEFAULT_OUT,
    jobs: int = 1,
    source=None,
    feature_cache_dir=None,
    repo: Path = ROOT,
    check_registration: bool = True,
    program=PROGRAM,
    quasi: bool = True,
) -> dict:
    """Run and test one family (module docstring); returns the result dict also written to <out_dir>/<id>/."""
    from experiments.runner import CachedBars, code_hash, git_sha, run
    from features import cache as feature_cache

    fam = load_family(path)
    if check_registration:
        fam.registered = check_registered(path, repo)
    source = source or CachedBars()
    root = Path(root)
    feature_cache_dir = feature_cache.DEFAULT_ROOT if feature_cache_dir == "default" else feature_cache_dir
    accounting = check_budget(fam, ledger.rows(), program_caps(program))
    start, end = fam.window

    batch = [c for v in fam.variants for c in fam.cells[v.label]]
    split_cells = {s["label"]: make_cells(fam, fam.headline.config, start, end, s["instruments"])
                   for s in fam.sample_splits if "instruments" in s}  # fmt: skip
    quasi_cells = make_cells(fam, fam.headline.config, start, QUASI_WINDOW[1]) if quasi else []
    batch += [c for cs in split_cells.values() for c in cs] + quasi_cells
    hashes: dict[tuple, str] = {}
    run(batch, ledger=ledger, root=root, source=source, jobs=jobs, spec_name=fam.id,
        feature_cache_dir=feature_cache_dir, family=True, on_hash=lambda c, h: hashes.__setitem__((c.stage, c.spec_json()), h))  # fmt: skip
    rows_by_hash = {h: r for (h, st), r in L.done(ledger.rows()).items() if st == "F"}

    bh = bh_returns(source, fam.instruments, fam.timeframe, start, end)
    weights = T.risk_weights(bh)
    bench = T.benchmark(fam.benchmark, bh, weights)

    pooled = {v.label: _pooled(fam, fam.cells[v.label], rows_by_hash, hashes, root, weights) for v in fam.variants}
    sha, chash = git_sha(), code_hash()
    stamp = datetime.now(UTC).isoformat()
    done = L.done(ledger.rows())
    for v in fam.variants:
        p = pooled[v.label]
        vh = _vhash(fam, v.label, p["members"])
        p["hash"] = vh
        if (vh, "F") in done and done[(vh, "F")]["status"] == p["status"]:
            continue
        row = {"stage": "F", "status": p["status"], "kind": KIND, "cell_hash": vh, "label": f"{fam.id}:{v.label}",
               "family": fam.id, "base_family": fam.base_id, "variant": v.label, "trial_key": fam.trial_key(v.label),
               "registered_sha": str((fam.registered or {}).get("sha")), "members": p["members"], "n_trials": 1,
               "spec_json": json.dumps(v.config, sort_keys=True, default=str), "git_sha": sha, "code_hash": chash,
               "started_at": stamp, "instruments": list(fam.instruments), "weights": weights.to_dict(),
               "budget": fam.budget}  # fmt: skip
        if p["status"] == "ok":
            row |= T.stream_stats(p["stream"]) | p["costs"]
            d = root / "cells" / vh
            d.mkdir(parents=True, exist_ok=True)
            p["stream"].rename("ret").to_csv(d / "daily_returns.csv")
            (d / "result.json").write_text(json.dumps(L.clean(row), indent=2, sort_keys=True), encoding="utf-8")
        ledger.append(row)
        print(f"[FAM]  F {p['status']:<7} {vh}  {row['label']}")

    failed = {k: p["status"] for k, p in pooled.items() if p["status"] != "ok"}
    result: dict = {"id": fam.id, "registered": fam.registered, "spec": fam.doc, "window": list(fam.window),
                    "instruments": list(fam.instruments), "basket": fam.basket, "benchmark": fam.benchmark,
                    "weights": weights.to_dict(), "accounting": accounting, "git_sha": sha, "code_hash": chash,
                    "run_at": stamp, "variant_hashes": {k: p["hash"] for k, p in pooled.items()},
                    "spec_path": str(Path(path).resolve()), "repo": str(Path(repo).resolve()),
                    "members": {k: p["members"] for k, p in pooled.items()}}  # fmt: skip
    if HEADLINE in failed:
        result["error"] = f"headline cells did not complete ({failed[HEADLINE]}); no test"
        return _save(result, None, out_dir)
    ok = [v for v in fam.variants if pooled[v.label]["status"] == "ok"]
    streams = {v.label: pooled[v.label]["stream"] for v in ok}
    costs = {v.label: pooled[v.label]["costs"] for v in ok}
    sub = FamilySpec(**{**fam.__dict__, "variants": ok})
    ev = T.evaluate(sub, streams, bench, costs)
    if failed:  # a variant that did not run is reported, and counts against coherence as not sharing the sign
        for k, st in failed.items():
            ev["variants"][k] = {"status": st}
        ev["verdict"]["coherence"] = T.coherence(
            {**{k: {"delta_ann": v["compare"]["delta_ann"] if k == HEADLINE else v["delta_common"],
                    "floors_ok": v["floors"]["ok"]}
                for k, v in ev["variants"].items() if "compare" in v},
             **{k: {"delta_ann": np.nan, "floors_ok": False} for k in failed}}, fam.test["coherence_share"])  # fmt: skip
    ev["verdict"]["dsr"], ev["verdict"]["dsr_n"], ev["verdict"]["dsr_v"] = program_dsr(ledger.rows(), streams[HEADLINE])

    sample_streams = {}
    for label, cs in split_cells.items():
        syms = [x for c in cs for x in c.symbols]  # a basket split is one portfolio cell over all its symbols
        sbh = bh_returns(source, syms, fam.timeframe, start, end)
        sw = T.risk_weights(sbh)
        hs = [hashes[(c.stage, c.spec_json())] for c in cs]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            r = (_stream(root, hs[0]).rename("ret") if fam.basket
                 else T.pool({c.symbols[0]: _stream(root, h) for c, h in zip(cs, hs)}, sw))  # fmt: skip
            sample_streams[label] = (r, T.benchmark(fam.benchmark, sbh, sw))
    qslice = None
    if quasi_cells:
        hs = [hashes[(c.stage, c.spec_json())] for c in quasi_cells]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            qs = {("_basket" if fam.basket else c.symbols[0]): _stream(root, h) for c, h in zip(quasi_cells, hs)}
            qr = qs["_basket"] if fam.basket else T.pool(qs, weights)
            qbh = bh_returns(source, fam.instruments, fam.timeframe, *QUASI_WINDOW)
            qslice = (qr.loc[QUASI_WINDOW[0] :], T.benchmark(fam.benchmark, qbh, weights))
            result["quasi_hashes"] = hs
    state = load_state(source, start, end)
    desc = T.describe(fam, streams, bench, {k: pooled[k].get("trades") for k in streams},
                      pooled[HEADLINE].get("per_instrument", {}), state, sample_streams, qslice)  # fmt: skip
    result |= {
        "evaluation": ev,
        "described": desc,
        "skipped_segments": {k: p["skipped_segments"] for k, p in pooled.items() if p.get("skipped_segments")},
    }
    frame = pd.concat({**{k: T.day_index(s) for k, s in streams.items()}, "benchmark": T.day_index(bench)}, axis=1,
                      sort=True)  # fmt: skip
    return _save(result, frame, out_dir)


def program_dsr(rows: list[dict], headline: pd.Series) -> tuple[float, int, float]:
    """DSR of a headline stream (SPEC §17.3): N = the program's counted trials (distinct trial keys, stages F / G),
    V = the variance of their per-period Sharpes (the latest row of each key)."""
    from validation.stats import dsr, return_moments

    latest: dict[str, dict] = {}
    for r in L.trials(rows):
        if r.get("kind") == KIND and r.get("stage") in FAMILY_STAGES_COUNTED and r.get("status") in L.COUNTED:
            latest[r["trial_key"]] = r
    srs = [r["sharpe"] / np.sqrt(T.TRADING_DAYS) for r in latest.values() if r.get("sharpe") is not None]
    n = max(len(latest), 1)
    v = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    try:
        m = return_moments(T.day_index(headline).to_numpy())
        return float(dsr(m.sr, n, v, m.n_obs, m.skew, m.kurt)), n, v
    except ValueError:
        return np.nan, n, v


def _save(result: dict, frame: pd.DataFrame | None, out_dir) -> dict:
    from families.report import write_family_report

    out = Path(out_dir) / result["id"]
    out.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        frame.to_csv(out / "streams.csv")
    (out / "result.json").write_text(json.dumps(L.clean(result), indent=2, sort_keys=True, default=str),
                                     encoding="utf-8")  # fmt: skip
    result["paths"] = write_family_report(result, out)
    print(f"[FAM]  {result['id']} → {out}")
    return result
