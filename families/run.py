"""
Run one family (SPEC §17, PLAN2 U17; protocol v3 SPEC §22, U22): registered spec → stage-F rule cells through
experiments.runner → pooled variant streams → one ledger row per variant (the counted trial) → the family test →
results/families/<id>/.

Cells (all stage F rule cells, n_trials 0 each, run in one runner batch):
- every variant × instrument (or the basket's one portfolio cell) over the family window;
- the headline on each instrument sample split (reported);
- the headline over [window start, 2026-10-01) for the quasi-holdout slice (reported; its days before 2025-10-01
  are not used: the test reads the dev-window cells only).

Trial accounting (SPEC §11.1): a variant is one trial whatever its instrument count. Each variant gets a ledger row
(kind "family_variant", n_trials 1, its pooled stream's statistics, `trial_key` = <family>/<configuration hash>, members = its cell
hashes); its artifacts (daily_returns.csv, result.json) live under <root>/cells/<variant hash>/. Budgets count
distinct trial keys with a counted status in stages F / G: per family (amendments `<id>.v<k>` share their family's)
≤ TRIAL_BUDGET, and program-wide ≤ families/program.yaml's caps over the program's listed families (U22: a family
outside the list is refused; the PLAN2 families' rows do not count). A ledger already holding rows of this id under
another registered sha is refused (an amendment is a new file).

U22 additions: equal-risk weights from the trailing 252 sessions before the window (the window's first 252 sessions
when the data start with it, recorded as `weights_basis`); excess returns against the T-bill accrual (FRED DTB3);
the cost curve (every variant re-priced at 0.3 / 1.0 / 2.3 bp round trip and at the measured profile from the
members' per-session cost ledgers; the verdict reads the spec's `test.at_cost`); the account check (risk.account)
on the headline's positions; N_eff; the looks ledger's K and Bonferroni bound; the `marginal` kind's core stream
from the core family's results.

U24 additions (SPEC §25): `weighting: equal` pools the instruments 1/n; a region headline (a primary with cells, e.g.
region_trend) gets its specification curve over every (instrument, cell): each cell's position series re-simulated by
wfo.position_backtest on the member's bars, costs, prints, cash yield, clock and loss gate at the booked costs and at
each round-trip cost, with the region (the cells' mean) re-simulated beside them as a parity check against the member
cell's stream; with `test.coherence: cells` the verdict's coherence is the share of cells with the headline's alpha
sign at the verdict's cost. Trade statistics (round trips per day from the cost ledgers, long / short P&L from the
trades), per-instrument alpha at the verdict's cost, the MDE at the realized volatility, and the state inputs of the
vol_quintile and opex_day splits.
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
from families.spec import (
    DEV_WINDOW,
    HEADLINE,
    QUASI_WINDOW,
    ROOT,
    FamilySpec,
    check_registered,
    load_family,
    make_cells,
)

PROGRAM = ROOT / "families" / "program.yaml"
DEFAULT_OUT = ROOT / "results" / "families"
FAMILY_STAGES_COUNTED = ("F", "G")
KIND = "family_variant"
WEIGHT_SESSIONS = 252
MEASURED_COST_CSV = ROOT / "data" / "costs" / "measured_cost.csv"  # class, cost_bp (U27's reconciliation)


class BudgetError(RuntimeError):
    """A family run that would exceed its TRIAL_BUDGET or the program caps (PLAN2 protocol 3)."""


class CostLedgerMissing(RuntimeError):
    """A member cell without a per-session cost ledger (daily_costs.csv): a PWFO member, or a pre-U22 cell."""


# ── Budgets ──────────────────────────────────────────────────────────────────


def program_caps(path=PROGRAM) -> dict:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fams = doc.get("families")
    if fams is not None and (not isinstance(fams, list) or not all(isinstance(f, str) for f in fams)):
        raise ValueError(f"{path}: `families` must be a list of family ids")
    return {"max_families": int(doc["max_families"]), "max_trials": int(doc["max_trials"]),
            "families": None if fams is None else [str(f) for f in fams], "program": doc.get("program")}  # fmt: skip


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
    listed = caps.get("families")
    if listed is not None and fam.base_id not in listed:
        raise BudgetError(f"{fam.base_id} is not a family of program {caps.get('program')!r} ({listed}): list it in "
                          "families/program.yaml first")  # fmt: skip
    keys = counted_keys(rows)
    if listed is not None:
        keys = {k: v for k, v in keys.items() if k in listed}
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
            "program_trials": n_trials, "max_families": caps["max_families"], "max_trials": caps["max_trials"],
            "program": caps.get("program")}  # fmt: skip


# ── Data ─────────────────────────────────────────────────────────────────────


def bh_returns(source, symbols, timeframe: str, start: str, end: str) -> dict[str, pd.Series]:
    """Buy-and-hold daily returns (last close of each NY session) of each symbol from the cells' own bars."""
    out = {}
    for s in symbols:
        df = source.bars(s, timeframe, start, end, allow_holdout=False)
        day = df.index.tz_convert("America/New_York").normalize() if df.index.tz is not None else df.index.normalize()
        out[s] = df["close"].groupby(day).last().pct_change().dropna()
    return out


def family_weights(source, symbols, timeframe: str, window: tuple[str, str],
                   weighting: str = "equal_risk") -> tuple[pd.Series, dict]:  # fmt: skip
    """
    `weighting` "equal" (U24): 1/n per instrument, no data read.

    Equal-risk weights (T.risk_weights) over the weighting window (SPEC §22.1): the WEIGHT_SESSIONS sessions before
    the family window when the data reach back that far (a window starting more than 400 days after the development
    window's start), else the family window's first WEIGHT_SESSIONS sessions (Alpaca's history starts with the
    development window). Either way the weights depend on the instruments' own volatility only.
    """
    if weighting == "equal":
        syms = sorted(symbols)
        return pd.Series(1.0 / len(syms), index=syms), {"window": "equal", "sessions": 0, "n": len(syms)}
    start, end = window
    s0 = pd.Timestamp(start)
    if s0 - pd.Timedelta(days=400) >= pd.Timestamp(DEV_WINDOW[0]):
        bh = bh_returns(source, symbols, timeframe, str((s0 - pd.Timedelta(days=400)).date()), start)
        bh = {k: v.iloc[-WEIGHT_SESSIONS:] for k, v in bh.items()}
        basis = {"window": "trailing", "sessions": WEIGHT_SESSIONS, "end": start}
    else:
        bh = bh_returns(source, symbols, timeframe, start, end)
        bh = {k: v.iloc[:WEIGHT_SESSIONS] for k, v in bh.items()}
        basis = {"window": "first", "sessions": WEIGHT_SESSIONS, "start": start}
    n = {k: len(v) for k, v in bh.items()}
    basis["n_sessions"] = n
    return T.risk_weights(bh), basis


VOL_RANK_SESSIONS = 252  # the causal distribution of the vol_quintile split (PLAN3 §3 G1)


def _ny_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert("America/New_York").normalize().tz_localize(None)
    return index.normalize()


def vol_rank(df: pd.DataFrame, sessions: int = VOL_RANK_SESSIONS) -> pd.Series:
    """
    Per session date (tz-naive NY): the mid-rank percentile, in (0, 1), of the PREVIOUS session's σ of 5Min log returns
    (inside the session) among the σ of the `sessions` sessions ending with it — known at the open; NaN until that many
    sessions exist (PLAN3 §3 G1's trailing-vol quintile).
    """
    day = _ny_day(df.index)
    r = np.log(df["close"].astype(float)).groupby(day).diff()
    sig = r.groupby(day).std(ddof=1)
    x = sig.to_numpy()
    out = np.full(len(x), np.nan)
    for d in range(sessions, len(x)):  # session d reads σ of sessions d − sessions … d − 1 (the last one is x[d − 1])
        w = x[d - sessions : d]
        w = w[np.isfinite(w)]
        v = x[d - 1]
        if np.isfinite(v) and len(w) >= sessions // 2:
            out[d] = ((w < v).sum() + 0.5 * (w == v).sum()) / len(w)
    return pd.Series(out, index=sig.index, name="vol_rank")


def load_state(source, start: str, end: str, instruments=(), timeframe: str | None = None,
               need=()) -> pd.DataFrame | None:  # fmt: skip
    """Day state for the reported splits: the previous VIX close (`vix_prev`), the macro-day flag (FOMC, CPI,
    NFP), and (U24) the OPEX flag and, for an intraday family whose splits name vol_quintile, the instruments' mean
    vol_rank. None for an input the source cannot give (tests' fake sources)."""
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
        out["opex"] = days.isin(pd.DatetimeIndex(pd.to_datetime(ev.loc[ev["kind"] == "OPEX", "date"])))
    except Exception as e:  # noqa: BLE001
        print(f"[FAM]  no event calendar for the state splits ({type(e).__name__}: {e})")
    if "vol_quintile" in need and instruments and timeframe not in (None, "1Day"):
        try:
            ranks = [vol_rank(source.bars(s, timeframe, start, end, allow_holdout=False)) for s in instruments]
            out["vol_rank"] = pd.concat(ranks, axis=1, sort=True).mean(axis=1).reindex(days)
        except Exception as e:  # noqa: BLE001 — a reported split only
            print(f"[FAM]  no vol_rank for the state splits ({type(e).__name__}: {e})")
    return out if out.shape[1] else None


def load_rf(source, start: str, end: str, sessions: pd.DatetimeIndex) -> tuple[pd.Series | None, str]:
    """The T-bill accrual per session (T.rf_accrual of FRED DTB3 as of each session; SPEC §22.7), or None with the
    reason when the source has no such series (a test source)."""
    from experiments.runner import rate_asof

    try:
        lookback = str((pd.Timestamp(start) - pd.Timedelta(days=14)).date())
        dtb3 = source.exo("fred", "DTB3", lookback, end, allow_holdout=False)
        rate = rate_asof(dtb3, sessions) / 100.0
        return T.rf_accrual(rate, sessions), "FRED DTB3 as of each session's open, ACT/360"
    except Exception as e:  # noqa: BLE001 — recorded in the result, never silent
        return None, f"unavailable ({type(e).__name__}: {e}): raw returns"


def measured_profile(path: Path = MEASURED_COST_CSV) -> dict[str, float] | None:
    """The measured one-way cost per fill class (bp) from the forward test's reconciliation, or None before it."""
    if not Path(path).exists():
        return None
    t = pd.read_csv(path, dtype={"class": str})
    out = {str(c): float(v) for c, v in zip(t["class"], t["cost_bp"])}
    missing = [c for c in T.FILL_CLASSES if c not in out]
    if missing:
        raise ValueError(f"{path}: no measured cost for fill class(es) {missing}")
    return out


def _read(root: Path, h: str, name: str) -> pd.DataFrame | None:
    p = root / "cells" / h / name
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, float_precision="round_trip")  # the written floats, exactly


def _stream(root: Path, h: str) -> pd.Series:
    s = _read(root, h, "daily_returns.csv")["ret"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    return s


def _daily_costs(root: Path, h: str) -> pd.DataFrame | None:
    dc = _read(root, h, "daily_costs.csv")
    if dc is None:
        return None
    dc.index = pd.DatetimeIndex(pd.to_datetime(dc.index)).normalize()
    return dc


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


def _check_disjoint(sym: str, trades: list[pd.DataFrame | None]) -> None:
    """Raise when two legs' positions on one symbol overlap in time (entry before another leg's open position exits)."""
    parts = []
    for i, t in enumerate(trades):
        if t is None or not len(t):
            continue
        if not {"entry_time", "exit_time"} <= set(t):
            raise ValueError(f"{sym}: leg trades without entry_time / exit_time (re-run the cells)")
        parts.append(pd.DataFrame({"entry": pd.to_datetime(t["entry_time"], utc=True),
                                   "exit": pd.to_datetime(t["exit_time"], utc=True), "leg": i}))  # fmt: skip
    if len(parts) < 2:
        return
    iv = pd.concat(parts, ignore_index=True).sort_values(["entry", "exit"], kind="stable")
    for i, g in iv.groupby("leg"):
        others = iv[iv["leg"] != i]
        # times are bar stamps and a close fill exits at its bar's stamp, so a position holds its entry and exit
        # bars inclusive: two legs clash when they share any bar (conservative: back-to-back legs on one bar clash)
        e, x = others["entry"].to_numpy(), others["exit"].to_numpy()
        for a, b in zip(g["entry"].to_numpy(), g["exit"].to_numpy()):
            hit = (e <= b) & (x >= a)
            if hit.any():
                raise ValueError(f"{sym}: legs hold positions at the same time ({pd.Timestamp(a)} … "
                                 f"{pd.Timestamp(b)}); legs must be disjoint in time")  # fmt: skip


def _by_symbol(cells, members: list[str], root: Path, cost=None) -> dict[str, pd.Series]:
    """Symbol → its daily stream: the sum of its legs' streams on the union of their days (one cell without legs),
    each leg re-priced at `cost` (T.reprice) when given. Legs are separate full-size positions, so their positions on
    one symbol must never overlap in time (raises)."""
    by: dict[str, list[str]] = {}
    for c, h in zip(cells, members):
        by.setdefault(c.symbols[0], []).append(h)
    streams = {}
    for sym, hs in by.items():
        if len(hs) > 1:
            _check_disjoint(sym, [_read(root, h, "trades.csv") for h in hs])
        parts = []
        for h in hs:
            s = T.day_index(_stream(root, h))
            if cost is not None:
                dc = _daily_costs(root, h)
                if dc is None:
                    raise CostLedgerMissing(f"cell {h} has no daily_costs.csv (a PWFO member, or a pre-U22 cell)")
                s = T.reprice(s, dc, cost)[0]
            parts.append(s)
        frame = pd.concat(parts, axis=1, sort=True).fillna(0.0)
        streams[sym] = frame.sum(axis=1).rename("ret")
    return streams


def _pooled(fam: FamilySpec, cells, rows_by_hash, hashes, root, weights, cost=None) -> dict:
    """Members → status, pooled stream, summed cost totals, trades, per-instrument streams; at the booked costs, or
    re-priced at `cost` (a round-trip bp, or a per-class profile) from the members' cost ledgers."""
    members = [hashes[(c.stage, c.spec_json())] for c in cells]
    stat = [rows_by_hash.get(h, {}).get("status", "error") for h in members]
    out = {"members": members, "status": "ok" if all(s == "ok" for s in stat) else
           ("error" if "error" in stat else "no_fit")}  # fmt: skip
    if out["status"] != "ok":
        return out
    if fam.basket:
        s = T.day_index(_stream(root, members[0]))
        if cost is not None:
            dc = _daily_costs(root, members[0])
            if dc is None:
                raise CostLedgerMissing(f"cell {members[0]} has no daily_costs.csv (a PWFO member, or a pre-U22 cell)")
            s = T.reprice(s, dc, cost)[0]
        streams = {"_basket": s}
    else:
        streams = _by_symbol(cells, members, root, cost)
    out["per_instrument"] = {} if fam.basket else streams
    out["stream"] = streams["_basket"].rename("ret") if fam.basket else T.pool(streams, weights)
    # cash totals per unit of each member's starting capital, mixed with the pooled stream's weights (a basket: its own)
    w = [1.0] if fam.basket else [float(weights[c.symbols[0]]) for c in cells]
    totals = {}
    for wi, h in zip(w, members):
        r = rows_by_hash[h]
        pnl, paid, notional = r["pnl"], r["cost_paid"], r["traded_notional"]
        if cost is not None:
            dc = _daily_costs(root, h)
            if dc is None:
                raise CostLedgerMissing(f"cell {h} has no daily_costs.csv (a PWFO member, or a pre-U22 cell)")
            _, tot = T.reprice(T.day_index(_stream(root, h)), dc, cost)
            pnl, paid = pnl + tot["pnl_delta"], tot["cost_paid"]
        for k, v in (("pnl", pnl), ("cost_paid", paid), ("traded_notional", notional)):
            totals[k] = totals.get(k, 0.0) + wi * v / r["init_cash"]
    out["costs"] = {k: float(v) for k, v in totals.items()}
    out["skipped_segments"] = {k: v for h in members for k, v in (rows_by_hash[h].get("skipped_segments") or {}).items()
                               if v}  # fmt: skip
    out["auction_fallbacks"] = int(sum(int(rows_by_hash[h].get("auction_fallbacks") or 0) for h in members))
    out["counters"] = {k: int(sum(int(rows_by_hash[h].get(k) or 0) for h in members))
                       for k in ("asof_before_first", "stress_fills", "cash_yield_missing_sessions")}  # fmt: skip
    out["data_notes"] = {
        f"{h[:8]}:{k}": v for h in members for k, v in (rows_by_hash[h].get("data_notes") or {}).items()
    }
    tr = [_read(root, h, "trades.csv") for h in members]
    out["trades"] = pd.concat([t for t in tr if t is not None and len(t)]) if any(
        t is not None and len(t) for t in tr) else None  # fmt: skip
    out["member_trades"] = [(c.symbols[0] if not fam.basket else "_basket", t, float(rows_by_hash[h]["init_cash"]))
                            for c, h, t in zip(cells, members, tr)]  # fmt: skip
    return out


def _account_check(fam: FamilySpec, pooled: dict, weights: pd.Series) -> dict:
    """The registered account's check (risk.account) on the headline's member positions."""
    from risk.account import AccountProfile, check_account

    acc = fam.account
    profile = AccountProfile(acc["name"], acc["kind"], acc["equity"], locate_bps=acc["locate_bps"])
    members = [(sym, 1.0 if fam.basket else float(weights[sym]), tr, init)
               for sym, tr, init in pooled.get("member_trades", [])]  # fmt: skip
    try:
        return check_account(members, pooled["stream"], profile)
    except ValueError as e:  # trades without the bar stamps (pre-U22 cells): say so
        return {"profile": profile.name, "kind": profile.kind, "equity": profile.equity, "tradable": None,
                "reasons": [f"not checked: {e}"]}  # fmt: skip


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
    from families.looks import bonferroni, looks_total
    from features import cache as feature_cache

    fam = load_family(path)
    if check_registration:
        fam.registered = check_registered(path, repo)
    source = source or CachedBars()
    root = Path(root)
    feature_cache_dir = feature_cache.DEFAULT_ROOT if feature_cache_dir == "default" else feature_cache_dir
    accounting = check_budget(fam, ledger.rows(), program_caps(program))
    start, end = fam.window
    measured = measured_profile()
    at_cost = fam.test["at_cost"]
    if at_cost == "measured" and measured is None:
        raise ValueError(f"{fam.id}: test.at_cost 'measured' needs {MEASURED_COST_CSV} (the forward test's "
                         "reconciliation); until then register at_cost 1.0")  # fmt: skip

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
    weights, weights_basis = family_weights(source, fam.instruments, fam.timeframe, fam.window, fam.weighting)
    bench = T.benchmark(fam.benchmark, bh, weights)
    rf, rf_note = load_rf(source, start, end, pd.DatetimeIndex(T.day_index(bench).index))

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
               "budget": fam.budget, "weights_basis": weights_basis}  # fmt: skip
        if p["status"] == "ok":
            row |= T.stream_stats(p["stream"]) | p["costs"]
            d = root / "cells" / vh
            d.mkdir(parents=True, exist_ok=True)
            p["stream"].rename("ret").to_csv(d / "daily_returns.csv")
            (d / "result.json").write_text(json.dumps(L.clean(row), indent=2, sort_keys=True), encoding="utf-8")
        ledger.append(row)
        print(f"[FAM]  F {p['status']:<7} {vh}  {row['label']}")

    failed = {k: p["status"] for k, p in pooled.items() if p["status"] != "ok"}
    K = looks_total()
    result: dict = {"id": fam.id, "registered": fam.registered, "spec": fam.doc, "window": list(fam.window),
                    "instruments": list(fam.instruments), "basket": fam.basket, "benchmark": fam.benchmark,
                    "weights": weights.to_dict(), "weights_basis": weights_basis, "accounting": accounting,
                    "git_sha": sha, "code_hash": chash, "run_at": stamp,
                    "variant_hashes": {k: p["hash"] for k, p in pooled.items()},
                    "spec_path": str(Path(path).resolve()), "repo": str(Path(repo).resolve()),
                    "members": {k: p["members"] for k, p in pooled.items()}, "test": dict(fam.test),
                    "power": fam.power, "account_profile": fam.account, "rf": rf_note, "looks": {"K": K},
                    "measured_profile": measured}  # fmt: skip
    if HEADLINE in failed:
        result["error"] = f"headline cells did not complete ({failed[HEADLINE]}); no test"
        return _save(result, None, out_dir)
    ok = [v for v in fam.variants if pooled[v.label]["status"] == "ok"]
    sub = FamilySpec(**{**fam.__dict__, "variants": ok})

    # the core stream of a marginal test (the core family's result) and the k per variant
    core_stream, k_by_variant = None, None
    if fam.test["kind"] == "marginal":
        core_stream = _core_stream(fam.core, out_dir)
        k_by_variant = {v.label: float((v.core or fam.core).get("k", 1.0)) for v in fam.variants}

    # the cost curve: every variant at the booked costs and at each round-trip cost (and the measured profile)
    curve: dict = {"registered": {v.label: (pooled[v.label]["stream"], pooled[v.label]["costs"]) for v in ok}}
    for c in T.COST_CURVE_BP:
        curve[str(c)] = {v.label: _at_cost(fam, v, rows_by_hash, hashes, root, weights, c) for v in ok}
    if measured is not None:
        curve["measured"] = {v.label: _at_cost(fam, v, rows_by_hash, hashes, root, weights, measured) for v in ok}
    key = "registered" if at_cost == "registered" else ("measured" if at_cost == "measured" else str(float(at_cost)))
    if key not in curve:  # a spec cost off the standard curve
        curve[key] = {v.label: _at_cost(fam, v, rows_by_hash, hashes, root, weights, float(at_cost)) for v in ok}
    if any(x is None for x in curve[key].values()):
        raise ValueError(f"{fam.id}: test.at_cost {at_cost!r} needs every member's cost ledger (daily_costs.csv); "
                         "a PWFO member has none: register at_cost 'registered'")  # fmt: skip
    streams = {k: s for k, (s, _) in curve[key].items()}
    costs = {k: c for k, (_, c) in curve[key].items()}
    region = None
    if _is_region(fam):  # U24: the region's cells, re-simulated (specification curve, cells coherence)
        region = region_cells(fam, fam.cells[HEADLINE], pooled[HEADLINE]["members"], rows_by_hash, root, source, rf,
                              weights, at_cost)  # fmt: skip
    cell_alphas = None
    if region is not None and key in region["costs"]:
        cell_alphas = {k: v[key]["alpha_bp"] for k, v in region["cells"].items()}
    ev = T.evaluate(sub, streams, bench, costs, rf, core_stream, k_by_variant, cell_alphas)
    if failed:  # a variant that did not run is reported, and counts against coherence as not sharing the sign
        for k, st in failed.items():
            ev["variants"][k] = {"status": st}
        ckey = "variant_coherence" if ev["verdict"].get("coherence_kind") == "cells" else "coherence"
        ev["verdict"][ckey] = T.coherence(
            {**{k: {"delta_ann": v["compare"]["delta_ann"] if k == HEADLINE else v["delta_common"],
                    "floors_ok": v["floors"]["ok"]}
                for k, v in ev["variants"].items() if "compare" in v},
             **{k: {"delta_ann": np.nan, "floors_ok": False} for k in failed}}, fam.test["coherence_share"])  # fmt: skip
    ev["verdict"]["dsr"], ev["verdict"]["dsr_n"], ev["verdict"]["dsr_v"] = program_dsr(ledger.rows(), streams[HEADLINE])
    ev["verdict"]["at_cost"] = key
    ev["verdict"]["bonferroni_p"] = bonferroni(ev["verdict"]["p"], K)
    result["looks"]["bonferroni_p"] = ev["verdict"]["bonferroni_p"]
    cost_curve = {}
    for ck, per in curve.items():
        cost_curve[ck] = {}
        for label, sc in per.items():
            if sc is None:
                cost_curve[ck][label] = {"unavailable": "no cost ledger (a PWFO member, or a pre-U22 cell)"}
                continue
            s, c = sc
            entry = {"stats": T.stream_stats(s), "floors": T.floors(s, bench, c, fam.floors, rf), "costs": c}
            if label == HEADLINE:
                entry["compare"] = T.compare(s, bench, fam.benchmark, fam.test, rf, core_stream,
                                             (k_by_variant or {}).get(HEADLINE, 1.0))  # fmt: skip
            cost_curve[ck][label] = entry
    result["cost_curve"] = cost_curve

    sample_streams = {}
    for label, cs in split_cells.items():
        syms = sorted({x for c in cs for x in c.symbols})  # a basket split is one cell; legs share a symbol
        sbh = bh_returns(source, syms, fam.timeframe, start, end)
        sw, _ = family_weights(source, syms, fam.timeframe, fam.window, fam.weighting)
        hs = [hashes[(c.stage, c.spec_json())] for c in cs]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            r = _stream(root, hs[0]).rename("ret") if fam.basket else T.pool(_by_symbol(cs, hs, root), sw)
            sample_streams[label] = (r, T.benchmark(fam.benchmark, sbh, sw))
    qslice = None
    if quasi_cells:
        hs = [hashes[(c.stage, c.spec_json())] for c in quasi_cells]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            qs = {"_basket": _stream(root, hs[0])} if fam.basket else _by_symbol(quasi_cells, hs, root)
            qr = qs["_basket"] if fam.basket else T.pool(qs, weights)
            qbh = bh_returns(source, fam.instruments, fam.timeframe, *QUASI_WINDOW)
            qslice = (qr.loc[QUASI_WINDOW[0] :], T.benchmark(fam.benchmark, qbh, weights))
            result["quasi_hashes"] = hs
    state = load_state(source, start, end, fam.instruments, fam.timeframe, fam.state_splits)
    desc = T.describe(fam, streams, bench, {k: pooled[k].get("trades") for k in streams},
                      pooled[HEADLINE].get("per_instrument", {}), state, sample_streams, qslice)  # fmt: skip
    result |= {
        "evaluation": ev,
        "described": desc,
        "skipped_segments": {k: p["skipped_segments"] for k, p in pooled.items() if p.get("skipped_segments")},
        "n_eff": T.n_eff(pooled[HEADLINE].get("per_instrument", {}), weights) if not fam.basket else None,
        "account": _account_check(fam, pooled[HEADLINE], weights),
        "auction_fallbacks": {k: p.get("auction_fallbacks", 0) for k, p in pooled.items() if p["status"] == "ok"},
        "counters": {k: p.get("counters", {}) for k, p in pooled.items() if p["status"] == "ok"},
        "data_notes": {k: p.get("data_notes", {}) for k, p in pooled.items() if p.get("data_notes")},
        "region": region,
        "trade_stats": trade_stats(fam, pooled[HEADLINE], root, weights),
        "per_instrument_alpha": _per_instrument_alpha(fam, pooled[HEADLINE]["members"], root, rf, at_cost),
        "power_realized": _power_realized(fam, streams[HEADLINE], rf),
    }
    frame = pd.concat({**{k: T.day_index(s) for k, s in streams.items()}, "benchmark": T.day_index(bench),
                       **({"rf": T.day_index(rf)} if rf is not None else {})}, axis=1, sort=True)  # fmt: skip
    return _save(result, frame, out_dir)


def _cost_arg(at_cost):
    """The spec's at_cost as _by_symbol / T.reprice take it: None for the booked costs, else the round-trip bp."""
    return None if at_cost == "registered" else (measured_profile() if at_cost == "measured" else float(at_cost))


def _per_instrument_alpha(fam, members: list[str], root: Path, rf, at_cost) -> dict:
    """Each instrument's headline stream at the verdict's cost: mean daily excess return (bp) with its Newey–West 95 %
    interval, annualized Sharpe of the excess, days (reported)."""
    from families.stats import mean_ci_nw, sharpe_ann

    if fam.basket:
        return {}
    out = {}
    for sym, r in _by_symbol(fam.cells[HEADLINE], members, root, _cost_arg(at_cost)).items():
        x = T.excess(r, rf)
        m, lo, hi = mean_ci_nw(x.to_numpy())
        out[sym] = {"alpha_bp": m * 1e4, "ci_bp": [lo * 1e4, hi * 1e4], "sharpe": sharpe_ann(x), "n_days": len(x)}
    return out


def _power_realized(fam, headline: pd.Series, rf) -> dict | None:
    """The MDE line recomputed at the headline's realized days and excess volatility (reported, never a gate: the
    registered line is the spec's)."""
    if not fam.power:
        return None
    from families.power import mde_alpha

    x = T.excess(headline, rf)
    vol = float(x.std(ddof=1) * np.sqrt(T.TRADING_DAYS)) if len(x) > 2 else np.nan
    if not vol > 0:
        return None
    return mde_alpha(len(x), vol, alpha=fam.test["alpha"], families=int(fam.power["families"]), sided=fam.test["sided"])


def trade_stats(fam, pooled: dict, root: Path, weights: pd.Series) -> dict:
    """
    The headline's trading statistics (reported): per instrument and pooled with the stream's weights, round trips per
    day (Σ traded notional / the session's starting equity / 2, from the cost ledger, averaged over sessions), the
    share of sessions with a fill, and the long / short trading P&L (Σ trade pnl by side, as bp per day of the member's
    starting cash; trades without a side column are skipped).
    """
    if fam.basket or pooled.get("status") != "ok":
        return {}
    per = {}
    for h, (sym, tr, init) in zip(pooled["members"], pooled["member_trades"]):
        dc = _daily_costs(root, h)
        row = {}
        if dc is not None:
            notional = sum(dc[f"notional_{k}"] for k in T.FILL_CLASSES)
            row["round_trips_per_day"] = float((notional / dc["equity_start"]).mean() / 2.0)
            row["sessions_traded"] = float((notional > 0).mean())
            n_days = len(dc)
        else:
            n_days = len(T.day_index(_stream(root, h)))
        if tr is not None and len(tr) and "side" in tr and "pnl" in tr:
            side = tr["side"].astype(float)
            for name, m in (("long", side > 0), ("short", side < 0)):
                row[f"{name}_pnl_bp_per_day"] = float(tr.loc[m, "pnl"].sum() / init / max(n_days, 1) * 1e4)
                row[f"{name}_trades"] = int(m.sum())
        per[sym] = row
    keys = sorted({k for r in per.values() for k in r if not k.endswith("_trades")})
    pooled_row = {k: float(sum(float(weights[s]) * r[k] for s, r in per.items() if k in r)) for k in keys}
    return {"per_instrument": per, "pooled": pooled_row}


def _is_region(fam) -> bool:
    """A headline whose cells a region primary supplies (pooled rule cells of a primary with `cell_matrix`)."""
    if fam.basket or fam.headline.config.get("legs") or fam.headline.config.get("per_instrument"):
        return False
    from primaries import REGISTRY

    cfg = fam.cells[HEADLINE][0].config()
    return cfg.META_MODEL == "none" and hasattr(REGISTRY.get(cfg.PRIMARY), "cell_matrix")


def region_cells(fam, cells, members: list[str], rows_by_hash: dict, root: Path, source, rf, weights,
                 at_cost) -> dict:  # fmt: skip
    """
    The specification curve of a region headline (U24, SPEC §25): per instrument cell, every region cell's position
    series (primary.cell_matrix on the member's bars) re-simulated by wfo.position_backtest from the first bar of the
    member's first live session, on the member's own inputs (runner.load_cell_data: bars, as-of quotes, prints, cash
    yield, the calendar clock, stress flags; the risk profile's loss gate), at the booked costs and at every round-trip
    cost of the cost curve (and at_cost when off the curve). The region (the cells' mean, × the vol scale) is
    re-simulated beside them: its stream at the booked costs must equal the member's (`parity`: max |Δ daily return|).
    Per (instrument, cell) and cost: the mean daily excess return in bp with its Newey–West 95 % interval and the
    annualized excess Sharpe; the same for the pooled cell (the instruments' cell streams with the family weights) and
    for the region.
    """
    from experiments.runner import load_cell_data
    from families.stats import mean_ci_nw, sharpe_ann
    from primaries import make_primary
    from risk.costs import fill_costs
    from risk.profiles import get_profile
    from wfo.position_backtest import decision_grid, position_backtest

    curve = [*T.COST_CURVE_BP]
    if at_cost not in ("registered", "measured") and float(at_cost) not in curve:
        curve.append(float(at_cost))
    out: dict = {"cells": {}, "pooled_cells": {}, "region": {}, "parity": {}, "names": [],
                 "costs": ["registered", *map(str, curve)],
                 "note": "each cell re-simulated by wfo.position_backtest on the member's inputs; the measured profile "
                         "is per fill class and is not re-simulated here"}  # fmt: skip
    by_sym: dict[str, dict[str, pd.DataFrame]] = {}
    for c, h in zip(cells, members):
        sym = c.symbols[0]
        cfg = c.config()
        data = load_cell_data(c, cfg, source, final=False)
        df = data["bars"][sym]
        p = make_primary(cfg)
        dec, m = p.cell_matrix(df, cfg)
        names = p.cell_names()
        out["names"] = names
        day = _ny_day(df.index)
        live_day = day[df.index.get_indexer([pd.Timestamp(rows_by_hash[h]["live_start"])])[0]]
        b0 = int(np.flatnonzero(day == live_day)[0])  # the first bar of the member's first live session
        bars = df.iloc[b0:]
        keep = dec >= b0
        frame = pd.DataFrame(m[keep], index=df.index[dec[keep]], columns=names)
        frame["_region"] = frame[names].to_numpy().mean(axis=1) * p.vol_scale(df)[dec[keep]]
        clock = data.get("clock")
        kw = {"prints": (data.get("prints") or {}).get(sym), "cash_yield": data.get("cash_yield"),
              "profile": get_profile(cfg.RISK_PROFILE), "sessions": clock, "grid": decision_grid(bars, cfg, clock)}  # fmt: skip
        booked = None
        if cfg.COST_MODEL != "slippage":
            booked = fill_costs(bars.index, cfg, (data.get("cost") or {}).get(sym), stress=data.get("stress"))
        rets = {"registered": position_backtest(bars, frame, cfg, costs=booked, **kw).ret}
        for bp in curve:
            rets[str(bp)] = position_backtest(bars, frame, cfg, costs=bp / 2.0 * 1e-4, **kw).ret
        member = T.day_index(_stream(root, h))
        reg = T.day_index(rets["registered"]["_region"])
        j = pd.concat([member.rename("m"), reg.rename("r")], axis=1, sort=True)
        both = j.dropna()
        out["parity"][sym] = {"max_abs_diff": float((both["m"] - both["r"]).abs().max()) if len(both) else None,
                              "member_days": int(j["m"].notna().sum()), "resim_days": int(j["r"].notna().sum()),
                              "common_days": len(both)}  # fmt: skip
        by_sym[sym] = rets

    def stat(r: pd.Series) -> dict:
        x = T.excess(r, rf)
        mm, lo, hi = mean_ci_nw(x.to_numpy())
        return {"alpha_bp": mm * 1e4, "ci_bp": [lo * 1e4, hi * 1e4], "sharpe": sharpe_ann(x)}

    for ck in out["costs"]:
        for col in [*out["names"], "_region"]:
            per = {sym: T.day_index(rets[ck][col]) for sym, rets in by_sym.items()}
            pooled_s = T.pool(per, weights)
            if col == "_region":
                for sym, r in per.items():
                    out["region"].setdefault(sym, {})[ck] = stat(r)
                out["region"].setdefault("pooled", {})[ck] = stat(pooled_s)
                continue
            for sym, r in per.items():
                out["cells"].setdefault(f"{sym}:{col}", {})[ck] = stat(r)
            out["pooled_cells"].setdefault(col, {})[ck] = stat(pooled_s)
    return out


def _at_cost(fam, v, rows_by_hash, hashes, root, weights, cost) -> tuple[pd.Series, dict] | None:
    """The variant's pooled stream and totals re-priced at `cost`; None when a member has no cost ledger."""
    try:
        p = _pooled(fam, fam.cells[v.label], rows_by_hash, hashes, root, weights, cost)
    except CostLedgerMissing:
        return None
    return p["stream"], p["costs"]


def _core_stream(core: dict, out_dir) -> pd.Series:
    """The core family's variant stream from its results (streams.csv) for a `marginal` test."""
    p = Path(out_dir) / str(core["family"]) / "streams.csv"
    if not p.exists():
        raise FileNotFoundError(f"core family {core['family']!r} has no results at {p}: run it first")
    frame = pd.read_csv(p, index_col=0, parse_dates=True)
    label = str(core.get("variant", HEADLINE))
    if label not in frame:
        raise ValueError(f"core family {core['family']!r} has no variant {label!r} in {p}")
    return frame[label].dropna().rename("core")


def program_dsr(rows: list[dict], headline: pd.Series) -> tuple[float, int, float]:
    """DSR of a headline stream (a ledger diagnostic since U22, SPEC §22.6): N = the program's counted trials
    (distinct trial keys, stages F / G), V = the variance of their per-period Sharpes (the latest row of each key)."""
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
