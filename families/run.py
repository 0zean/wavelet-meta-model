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
on the headline's positions; N_eff; the looks ledger's K and Bonferroni bound.

U24 additions (SPEC §25): `weighting: equal` pools the instruments 1/n; a region headline (a primary with cells, e.g.
region_trend) gets its specification curve over every (instrument, cell): each cell's position series re-simulated by
wfo.position_backtest on the member's bars, costs, prints, cash yield, clock and loss gate at the booked costs and at
each round-trip cost, with the region (the cells' mean) re-simulated beside them as a parity check against the member
cell's stream; with `test.coherence: cells` the verdict's coherence is the share of cells with the headline's alpha
sign at the verdict's cost. Trade statistics (round trips per day from the cost ledgers, long / short P&L from the
trades), per-instrument alpha at the verdict's cost, the MDE at the realized volatility, and the state inputs of the
vol_quintile and opex_day splits.

U25 additions (SPEC §26): the `marginal` kind's core and the `paired` kind's reference are other registered family
specs (families/<id>.yaml, registration checked) whose cells run in this family's batch — stage F rule cells, never a
family variant row, so they count no trial and enter no Holm. Each variant's core (`core.family` / `core.variant`, the
variant `benchmark` = the core family's benchmark mix) is pooled with the core family's own weights at its booked
costs; the reference is pooled at the same cost as the stream at every point of the cost curve (a gated stream and
its ungated reference are priced alike). A marginal family's report adds, per variant, the account check of core +
k × overlay together (the overlay members' positions × k plus the core's; PLAN3 G2's buying-power check).

U28 additions (SPEC §27): an ENGINE region (a primary with `cell_params`, families.spec.region_kind "engine": G6's
positions are held overnight, which wfo.position_backtest does not model) runs every region cell as its own stage-F
rule cell in the batch (the headline with the primary's params replaced by the cell's; no ledger variant row, no
trial); the specification curve reads each cell's stream re-priced from its own cost ledger at every cost, and with
`test.coherence: cells` the verdict's coherence is the share of cells whose headline statistic (the alpha, or with a
`marginal` test the Δ Sharpe of core + k × cell vs the core) has the headline's sign. A marginal family also reports
the overlay's own alpha against zero (the `overlay_alpha` statistics, never tested); every family reports the members'
drawdown-kill bars (risk.profiles kill_dd) and the spec's `episodes` day by day (`episode_paths`).
"""

import copy
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
    TERCILE_SPLITS,
    FamilySpec,
    check_registered,
    load_family,
    make_cells,
    region_kind,
)
from features.tf_state import COLUMNS as TF_COLUMNS
from features.tf_state import tf_state_days

PROGRAM = ROOT / "families" / "program.yaml"
DEFAULT_OUT = ROOT / "results" / "families"
FAMILY_STAGES_COUNTED = ("F", "G")
KIND = "family_variant"
WEIGHT_SESSIONS = 252
MEASURED_COST_CSV = ROOT / "data" / "costs" / "measured_cost.csv"  # class, cost_bp (U27's reconciliation)
PARITY_TOL = 1e-9  # the region's re-simulation vs its members (exactly 0 in practice; SPEC §25)


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
    from features.kernels import prior_rank, session_layout, session_sigma

    day = _ny_day(df.index)
    sess = session_layout(df.index, 5)[0]
    # U25: the one definition G3's gate reads too (features.kernels)
    out = prior_rank(session_sigma(df["close"].to_numpy(dtype=float), sess), sessions)
    return pd.Series(out, index=pd.DatetimeIndex(day.unique()), name="vol_rank")


def trailing_rank(x: pd.Series, sessions: int = VOL_RANK_SESSIONS) -> pd.Series:
    """The mid-rank percentile of each value among the `sessions` values ending with it (NaN skipped; NaN with fewer
    than half of them): a mean of several instruments' ranks is not uniform, its own trailing rank is (U24 review)."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for d in range(len(v)):
        w = v[max(0, d - sessions + 1) : d + 1]
        w = w[np.isfinite(w)]
        if np.isfinite(v[d]) and len(w) >= sessions // 2:
            out[d] = ((w < v[d]).sum() + 0.5 * (w == v[d]).sum()) / len(w)
    return pd.Series(out, index=x.index, name=x.name)


def load_state(source, start: str, end: str, instruments=(), timeframe: str | None = None,
               need=()) -> pd.DataFrame | None:  # fmt: skip
    """Day state for the reported splits: the previous VIX close (`vix_prev`), the macro-day flag (FOMC, CPI,
    NFP), and (U24) the OPEX flag and, for an intraday family whose splits name vol_quintile, the instruments' mean
    vol_rank; (U26) for the tercile splits `<column>_rank` (the trailing rank of the instruments' mean tf_state column,
    or of the CBOE close, as of the open) and for gamma_sign `ac1`. None for an input the source cannot give (tests'
    fake sources)."""
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
            mean_rank = pd.concat(ranks, axis=1, sort=True).mean(axis=1)
            out["vol_rank"] = (mean_rank if len(ranks) == 1 else trailing_rank(mean_rank)).reindex(days)
        except Exception as e:  # noqa: BLE001 — a reported split only
            print(f"[FAM]  no vol_rank for the state splits ({type(e).__name__}: {e})")
    # U26 (SPEC §24): tf_state columns (the instruments' mean, known at the open) and the previous CBOE closes; a
    # tercile split reads the column's trailing rank, gamma_sign the mean lag-1 autocorrelation itself
    cols = {TERCILE_SPLITS[k] for k in need if k in TERCILE_SPLITS}
    ranked = set(cols)
    if "gamma_sign" in need:
        cols.add("ac1")
    tf_cols = sorted(cols & set(TF_COLUMNS))
    if tf_cols and instruments and timeframe not in (None, "1Day"):
        try:
            per = [tf_state_days(source.bars(s, timeframe, start, end, allow_holdout=False)) for s in instruments]
            for c in tf_cols:
                mean = pd.concat([p[c] for p in per], axis=1, sort=True).mean(axis=1)
                if c == "ac1":
                    out["ac1"] = mean.reindex(days)
                if c in ranked:
                    out[f"{c}_rank"] = trailing_rank(mean).reindex(days)
        except Exception as e:  # noqa: BLE001 — a reported split only
            print(f"[FAM]  no tf_state for the state splits ({type(e).__name__}: {e})")
    for c in sorted(cols & {"skew", "vvix", "vvix_vix"}):
        try:
            out[f"{c}_rank"] = _as_of_open(trailing_rank(_cboe_close(source, c, start, end)), days)
        except Exception as e:  # noqa: BLE001 — a reported split only
            print(f"[FAM]  no {c} for the state splits ({type(e).__name__}: {e})")
    return out if out.shape[1] else None


def _cboe_close(source, name: str, start: str, end: str) -> pd.Series:
    """SKEW, VVIX or VVIX / VIX (on common dates) by observation date, from 400 days before `start` (so a trailing
    rank exists on the window's first day)."""

    def close(n: str) -> pd.Series:
        lookback = str((pd.Timestamp(start) - pd.Timedelta(days=400)).date())
        v = source.exo("cboe", n, lookback, end, allow_holdout=False)["value"]
        v.index = pd.DatetimeIndex(v.index).normalize()
        return v

    return (close("VVIX") / close("VIX") if name == "vvix_vix" else close(name.upper())).dropna()


def _as_of_open(v: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """The value of the last observation dated before each day (a 16:20 close is known at the next open; `vix_prev`)."""
    return v.reindex(days.union(v.index)).ffill().shift(1).reindex(days)


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
    refs = ref_families(fam, path, repo, check_registration)  # U25: core / reference families, run in this batch
    batch += [c for rid, labels in _ref_labels(fam).items() for lb in labels if lb != "benchmark"
              for c in refs[rid].cells[lb]]  # fmt: skip
    ecells = engine_cells(fam) if _is_engine_region(fam) else {}  # U28: an engine region's cells, run in the batch
    batch += [c for cs in ecells.values() for c in cs]
    batch = list({(c.stage, c.spec_json()): c for c in batch}.values())  # one run per distinct cell
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

    # U25: the core (marginal; at its booked costs) or the reference (paired; at the stream's cost) of each variant
    ref_pool = RefStreams(refs, rows_by_hash, hashes, root, source)
    bad = ref_pool.failed(_ref_labels(fam))
    if bad:
        result["error"] = f"core / reference cells did not complete ({bad}); no test"
        return _save(result, None, out_dir)
    k_by_variant = None
    if fam.test["kind"] == "marginal":
        k_by_variant = {v.label: float(v.core.get("k", 1.0)) for v in fam.variants}

    def refs_at(ck: str) -> dict | None:
        """Variant label → its core / reference stream for the cost-curve column `ck`."""
        if fam.test["kind"] == "marginal":
            return {v.label: ref_pool.stream(v.core["family"], v.core.get("variant", HEADLINE), None) for v in ok}
        if fam.test["kind"] == "paired":
            rb = fam.reference
            return {v.label: ref_pool.stream(rb["family"], rb.get("variant", HEADLINE), _curve_cost(ck, measured))
                    for v in ok}  # fmt: skip
        return None

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
    if ecells:  # U28: an engine region's cells, each its own engine stream
        region = engine_region(fam, ecells, rows_by_hash, hashes, root, rf=rf, weights=weights, curve=curve,
                               measured=measured, refs_at=refs_at, k=(k_by_variant or {}).get(HEADLINE, 1.0))  # fmt: skip
    cell_alphas = None
    if region is not None and not region["parity_ok"]:
        print(f"[FAM]  {fam.id}: the region's re-simulation does not reproduce its members ({region['parity']}): the "
              "cells are not used for coherence")  # fmt: skip
    elif region is not None and key in region["costs"]:
        stat = region.get("statistic", "alpha_bp")  # U28: an engine region under a marginal test: the Δ Sharpe
        cell_alphas = {k: (v[key][stat] if v.get(key) is not None else np.nan) for k, v in region["cells"].items()}
    refs_key = refs_at(key)
    ev = T.evaluate(sub, streams, bench, costs, rf, refs_key, k_by_variant, cell_alphas)
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
        rk = refs_at(ck) or {}
        for label, sc in per.items():
            if sc is None:
                cost_curve[ck][label] = {"unavailable": "no cost ledger (a PWFO member, or a pre-U22 cell)"}
                continue
            s, c = sc
            kk = (k_by_variant or {}).get(label, 1.0)
            entry = {"stats": T.stream_stats(s), "floors": T.floors(s, bench, c, fam.floors, rf, rk.get(label), kk),
                     "costs": c}  # fmt: skip
            if label == HEADLINE or rk:  # U25: a core / reference makes every variant's statistic worth a column
                entry["compare"] = T.compare(s, bench, fam.benchmark, fam.test, rf, rk.get(label), kk)
            cost_curve[ck][label] = entry
    result["cost_curve"] = cost_curve
    if refs:
        result["refs"] = ref_pool.describe(_ref_labels(fam))
    if fam.test["kind"] == "marginal":
        result["portfolio_account"] = {
            v.label: _portfolio_account(fam, v, pooled[v.label], weights, ref_pool, rf) for v in ok
        }

    sample_streams = {}
    for label, cs in split_cells.items():
        syms = sorted({x for c in cs for x in c.symbols})  # a basket split is one cell; legs share a symbol
        sbh = bh_returns(source, syms, fam.timeframe, start, end)
        sw, _ = family_weights(source, syms, fam.timeframe, fam.window, fam.weighting)
        hs = [hashes[(c.stage, c.spec_json())] for c in cs]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            if fam.basket:
                r = T.day_index(_stream(root, hs[0])).rename("ret")
                if _cost_arg(at_cost) is not None:
                    r = T.reprice(r, _daily_costs(root, hs[0]), _cost_arg(at_cost))[0]
            else:
                r = T.pool(_by_symbol(cs, hs, root, _cost_arg(at_cost)), sw)
            sample_streams[label] = (r, T.benchmark(fam.benchmark, sbh, sw))
    qslice = None
    if quasi_cells:
        hs = [hashes[(c.stage, c.spec_json())] for c in quasi_cells]
        if all(rows_by_hash.get(h, {}).get("status") == "ok" for h in hs):
            if fam.basket:
                qs = {"_basket": T.day_index(_stream(root, hs[0]))}
                if _cost_arg(at_cost) is not None:
                    qs["_basket"] = T.reprice(qs["_basket"], _daily_costs(root, hs[0]), _cost_arg(at_cost))[0]
            else:
                qs = _by_symbol(quasi_cells, hs, root, _cost_arg(at_cost))
            qr = qs["_basket"] if fam.basket else T.pool(qs, weights)
            qbh = bh_returns(source, fam.instruments, fam.timeframe, *QUASI_WINDOW)
            qslice = (qr.loc[QUASI_WINDOW[0] :], T.benchmark(fam.benchmark, qbh, weights))
            result["quasi_hashes"] = hs
    state = load_state(source, start, end, fam.instruments, fam.timeframe, fam.state_splits)
    per_inst = {} if fam.basket else _by_symbol(fam.cells[HEADLINE], pooled[HEADLINE]["members"], root,
                                                 _cost_arg(at_cost))  # fmt: skip
    desc = T.describe(fam, streams, bench, {k: pooled[k].get("trades") for k in streams}, per_inst, state,
                      sample_streams, qslice, rf)  # fmt: skip
    desc["at_cost"] = key
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
        "power_realized": _power_realized(
            fam, streams[HEADLINE], rf, (refs_key or {}).get(HEADLINE), (k_by_variant or {}).get(HEADLINE, 1.0)
        ),
        "kill": {
            k: {h[:16]: rows_by_hash[h].get("killed_at") for h in p["members"]}
            for k, p in pooled.items()
            if p["status"] == "ok"
        },
    }
    if fam.test["kind"] == "marginal":  # U28: the overlay's own alpha against zero (reported, never tested)
        result["overlay_alone"] = T.compare(streams[HEADLINE], bench, fam.benchmark, {**fam.test, "kind": "overlay_alpha"},
                                            rf)  # fmt: skip
    if fam.episodes:
        result["episodes"] = episode_paths(fam, streams[HEADLINE], (refs_key or {}).get(HEADLINE),
                                           (k_by_variant or {}).get(HEADLINE, 1.0), rf, pooled[HEADLINE], weights,
                                           source)  # fmt: skip
    extra = {}
    if fam.test["kind"] == "marginal":  # each distinct core, raw at its booked costs
        for v in ok:
            lb = f"core:{v.core['family']}/{v.core.get('variant', HEADLINE)}"
            extra[lb] = T.day_index(refs_key[v.label])
    elif fam.test["kind"] == "paired":
        extra[f"reference:{fam.reference['family']}/{fam.reference.get('variant', HEADLINE)}"] = T.day_index(
            refs_key[HEADLINE]
        )
    frame = pd.concat({**{k: T.day_index(s) for k, s in streams.items()}, "benchmark": T.day_index(bench), **extra,
                       **({"rf": T.day_index(rf)} if rf is not None else {})}, axis=1, sort=True)  # fmt: skip
    return _save(result, frame, out_dir)


SPLIT_PARITY_TOL = 1e-12


def add_splits(family: str, splits, look: str, out_dir=DEFAULT_OUT, source=None, check=("year", "vol_quintile"),
               looks_path=None) -> dict:  # fmt: skip
    """
    U26 (SPEC §24): add reported state splits to a family's stored result and rewrite the report's state-splits
    section, without re-running it (no cell, no ledger row, no trial). Refused unless:
    - `look` labels a families/looks.jsonl row recorded after the run and after any earlier addition (the splits are
      read on the development window, so the look comes first);
    - none of `splits` is already stored with groups (no overwrite);
    - at least one stored split of `check` exists and is reproduced to SPLIT_PARITY_TOL from the headline's excess
      stream rebuilt from the saved streams.csv (the stream at the verdict's cost and the T-bill accrual describe()
      used);
    - every split's input loads (none would be written "unavailable").
    Nothing is written on a refusal. The verdict, the floors and every other stored number are left as they are;
    `result["splits_added"]` records what was added, the look, the check, when and at which commit.
    """
    from experiments.runner import CachedBars, code_hash, git_sha
    from families.looks import LOOKS_PATH, looks
    from families.report import load_result, replace_splits, stream_frame
    from families.spec import STATE_SPLITS

    out = Path(out_dir) / family
    result = load_result(out / "result.json")
    bad = [s for s in splits if s not in STATE_SPLITS]
    if bad or not splits:
        raise ValueError(f"unknown or no state split(s) {bad or splits}; expected {STATE_SPLITS}")
    after = max([str(result.get("run_at", ""))] + [str(a["at"]) for a in result.get("splits_added", [])])
    if not any(r["label"] == look and str(r["at"]) > after for r in looks(looks_path or LOOKS_PATH)):
        raise ValueError(f"{family}: no look labelled {look!r} recorded after {after or 'the run'}: record it first "
                         "(python -m families look <label> <n>)")  # fmt: skip
    stored = result["described"]["state_splits"]
    again = [k for k in splits if "groups" in stored.get(k, {})]
    if again:
        raise ValueError(f"{family}: split(s) {again} are already stored; an addition never overwrites")
    frame = stream_frame(out)
    h = T.excess(frame[HEADLINE].dropna(), frame.get("rf", None))
    bench = frame["benchmark"].dropna()
    start, end = result["window"]
    timeframe = result["spec"].get("timeframe")
    check = [k for k in check if k in stored and not stored[k].get("unavailable")]
    if not check:
        raise ValueError(f"{family}: no stored split to reproduce the stream with; nothing written")
    state = load_state(source or CachedBars(), start, end, result["instruments"], timeframe, [*splits, *check])
    for k in check:
        g = T._groups(k, h.index, state, bench)
        again, before = T.split_table(h, g) if g is not None else [], stored[k]["groups"]
        same = [a["group"] for a in again] == [b["group"] for b in before] and all(
            a["n_days"] == b["n_days"] and abs(a["mean_bp"] - b["mean_bp"]) <= SPLIT_PARITY_TOL * 1e4
            for a, b in zip(again, before, strict=True))  # fmt: skip
        if not same:
            raise ValueError(f"{family}: the stored split {k!r} is not reproduced from streams.csv; nothing written")
    groups = {k: T._groups(k, h.index, state, bench) for k in splits}
    missing = [k for k, g in groups.items() if g is None]
    if missing:
        raise ValueError(
            f"{family}: the input of split(s) {missing} did not load (see the [FAM] lines); nothing written"
        )
    added = {k: {"groups": T.split_table(h, g)} for k, g in groups.items()}
    result["described"]["state_splits"] = {**stored, **added}
    result.setdefault("splits_added", []).append({
        "at": datetime.now(UTC).isoformat(), "splits": list(splits), "look": look, "checked": check,
        "git_sha": git_sha(), "code_hash": code_hash()})  # fmt: skip
    (out / "result.json").write_text(json.dumps(L.clean(result), indent=2, sort_keys=True, default=str),
                                     encoding="utf-8")  # fmt: skip
    result["paths"] = replace_splits(result, out)
    print(f"[FAM]  {family}: {len(added)} split(s) added; checked {check}; look {look!r} → {out}")
    return result


def _curve_cost(ck: str, measured):
    """A cost-curve column key as _pooled takes it: None (booked), the measured profile, or a round-trip bp."""
    return None if ck == "registered" else (measured if ck == "measured" else float(ck))


def _ref_labels(fam: FamilySpec) -> dict[str, list[str]]:
    """Core / reference family id → the variant labels this family reads from it (`benchmark` included)."""
    out: dict[str, list[str]] = {}
    blocks = [v.core for v in fam.variants] if fam.test["kind"] == "marginal" else \
        ([fam.reference] if fam.test["kind"] == "paired" else [])  # fmt: skip
    for b in blocks:
        lb = str(b.get("variant", HEADLINE))
        if lb not in out.setdefault(str(b["family"]), []):
            out[str(b["family"])].append(lb)
    return out


def _ref_shas(fam: FamilySpec) -> dict[str, set]:
    """Core / reference family id → the registered shas the consuming spec pins for it (`sha` in its blocks)."""
    blocks = [v.core for v in fam.variants] if fam.test["kind"] == "marginal" else \
        ([fam.reference] if fam.test["kind"] == "paired" else [])  # fmt: skip
    out: dict[str, set] = {}
    for b in blocks:
        out.setdefault(str(b["family"]), set()).add(b.get("sha"))
    return out


def ref_families(fam: FamilySpec, path, repo: Path, check_registration: bool) -> dict[str, FamilySpec]:
    """The core / reference families (families/<id>.yaml beside this spec), each validated, registration-checked when
    this family's is, on this family's window and timeframe-independent; a variant label they do not have raises.
    With the registration check, the consuming spec must pin each one's registered sha (`sha` in its core / reference
    block) and the file must be registered at exactly that sha (U25 review: otherwise a core or reference could be
    edited and re-registered after the family's result was seen, at no trial cost)."""
    out = {}
    shas = _ref_shas(fam)
    for rid, labels in _ref_labels(fam).items():
        rpath = Path(path).resolve().parent / f"{rid}.yaml"
        if not rpath.exists():
            raise FileNotFoundError(f"{fam.id}: core / reference family {rid!r} has no spec at {rpath}")
        rfam = load_family(rpath)
        if check_registration:
            rfam.registered = check_registered(rpath, repo)
            pinned = shas.get(rid, {None})
            if pinned != {str(rfam.registered["sha"])}:
                raise ValueError(f"{fam.id}: core / reference {rid} is registered at {rfam.registered['sha']}, but the "
                                 f"spec pins {sorted(map(str, pinned))}: a changed core / reference is a new family "
                                 "version (an amendment)")  # fmt: skip
        if rfam.window != fam.window:
            raise ValueError(f"{fam.id}: core / reference {rid} runs over {rfam.window}, not the family's {fam.window}")
        missing = [lb for lb in labels if lb != "benchmark" and lb not in rfam.cells]
        if missing:
            raise ValueError(f"{fam.id}: core / reference family {rid!r} has no variant(s) {missing}")
        out[rid] = rfam
    return out


class RefStreams:
    """U25: the core / reference families' pooled streams from this batch's cells (each family's own weights), cached
    per (family, variant, cost)."""

    def __init__(self, refs: dict[str, FamilySpec], rows_by_hash: dict, hashes: dict, root: Path, source):
        self.refs, self.rows_by_hash, self.hashes, self.root, self.source = refs, rows_by_hash, hashes, root, source
        self._w: dict[str, pd.Series] = {}
        self._cache: dict = {}

    def weights(self, rid: str) -> pd.Series:
        if rid not in self._w:
            r = self.refs[rid]
            self._w[rid] = family_weights(self.source, r.instruments, r.timeframe, r.window, r.weighting)[0]
        return self._w[rid]

    def pooled(self, rid: str, label: str, cost=None) -> dict:
        key = (rid, label, json.dumps(cost, sort_keys=True))
        if key not in self._cache:
            r = self.refs[rid]
            if label == "benchmark":  # the core family's benchmark mix: held throughout, no trades, no costs
                bh = bh_returns(self.source, r.instruments, r.timeframe, *r.window)
                self._cache[key] = {"status": "ok", "members": [], "benchmark": True,
                                    "stream": T.benchmark(r.benchmark, bh, self.weights(rid)).rename("ret")}  # fmt: skip
            else:
                self._cache[key] = _pooled(r, r.cells[label], self.rows_by_hash, self.hashes, self.root,
                                           self.weights(rid), cost)  # fmt: skip
        return self._cache[key]

    def stream(self, rid: str, label: str, cost=None) -> pd.Series:
        return self.pooled(rid, label, cost)["stream"]

    def failed(self, labels: dict[str, list[str]]) -> dict:
        return {f"{rid}/{lb}": st for rid, lbs in labels.items() for lb in lbs
                if (st := self.pooled(rid, lb)["status"]) != "ok"}  # fmt: skip

    def describe(self, labels: dict[str, list[str]]) -> dict:
        out = {}
        for rid, lbs in labels.items():
            r = self.refs[rid]
            out[rid] = {"registered": r.registered, "instruments": list(r.instruments), "window": list(r.window),
                        "weights": self.weights(rid).to_dict(), "benchmark": r.benchmark,
                        "variants": {lb: {"members": self.pooled(rid, lb)["members"],
                                          "stats": T.stream_stats(self.stream(rid, lb)),
                                          "costs": self.pooled(rid, lb).get("costs")} for lb in lbs}}  # fmt: skip
        return out


def _portfolio_account(fam, v, pooled: dict, weights: pd.Series, ref_pool: RefStreams, rf) -> dict:
    """
    The registered account's check of core + k × overlay held together (PLAN3 G2): the overlay members' positions × k
    × their weights plus the core members' × the core family's weights (a `benchmark` core: each instrument held at its
    weight from the first session to the last), on the portfolio's equity path. `plan3_2x` is PLAN3 U25's criterion,
    overlay notional + core notional ≤ 2 × equity at every decision (the Reg-T overnight limit; a margin account at or
    above the PDT floor may use 4× intraday, which the account check itself applies).
    """
    from risk.account import AccountProfile, check_account

    acc = fam.account
    profile = AccountProfile(acc["name"], acc["kind"], acc["equity"], locate_bps=acc["locate_bps"])
    k = float(v.core.get("k", 1.0))
    rid, lb = v.core["family"], v.core.get("variant", HEADLINE)
    core = ref_pool.pooled(rid, lb)
    cw = ref_pool.weights(rid)
    members = [(sym, k * float(weights[sym]), tr, init) for sym, tr, init in pooled.get("member_trades", [])]
    if core.get("benchmark"):
        days = T.day_index(core["stream"]).index
        t0 = pd.Timestamp(days[0]).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        t1 = pd.Timestamp(days[-1]).tz_localize("America/New_York") + pd.Timedelta(hours=16)
        for sym in cw.index:
            tr = pd.DataFrame({"side": [1], "frac": [1.0], "entry_time": [t0.tz_convert("UTC")],
                               "exit_time": [t1.tz_convert("UTC")], "rolled": [False]})  # fmt: skip
            members.append((f"core:{sym}", float(cw[sym]), tr, acc["equity"]))
    else:
        members += [(f"core:{sym}", float(cw[sym]), tr, init) for sym, tr, init in core.get("member_trades", [])]
    j = pd.concat([T.day_index(core["stream"]).rename("c"), T.excess(pooled["stream"], rf).rename("a")], axis=1,
                  sort=True).fillna(0.0)  # fmt: skip
    port = j["c"] + k * j["a"]
    try:
        out = check_account(members, port, profile)
    except ValueError as e:  # trades without the bar stamps (pre-U22 cells): say so
        return {"profile": profile.name, "tradable": None, "reasons": [f"not checked: {e}"], "k": k}
    gi = out["buying_power"]["max_gross_intraday"]
    out |= {"k": k, "core": f"{rid}/{lb}", "plan3_2x": {"max_gross": gi, "ok": bool(gi <= 2.0 + 1e-9)}}
    return out


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


def _power_realized(fam, headline: pd.Series, rf, ref: pd.Series | None = None, k: float = 1.0) -> dict | None:
    """The MDE line recomputed at the headline's realized days and excess volatility (reported, never a gate: the
    registered line is the spec's); for the paired kinds (U25) at the realized correlation of the two compared streams
    (core + k·overlay with the core; the stream with its reference) and the reference's realized Sharpe."""
    if not fam.power:
        return None
    from families.power import mde_alpha, mde_delta_sharpe
    from families.spec import SHARPE_DIFF_KINDS
    from families.stats import sharpe_ann

    if fam.test["kind"] in SHARPE_DIFF_KINDS:
        if ref is None:
            return None
        j = pd.concat([T.excess(headline, rf).rename("a"), T.excess(ref, rf).rename("c")], axis=1, sort=True).dropna()
        a = j["c"] + k * j["a"] if fam.test["kind"] == "marginal" else j["a"]
        rho = float(a.corr(j["c"]))
        if len(j) < 3 or not -1 < rho < 1:
            return None
        return mde_delta_sharpe(len(j), rho, sharpe_ann(j["c"]), alpha=fam.test["alpha"],
                                families=int(fam.power["families"]), sided=fam.test["sided"])  # fmt: skip
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


def _region_cls(fam):
    """The headline's primary class when its cells can be read (pooled rule cells, no legs / per_instrument)."""
    if fam.basket or fam.headline.config.get("legs") or fam.headline.config.get("per_instrument"):
        return None
    from primaries import REGISTRY

    cfg = fam.cells[HEADLINE][0].config()
    return REGISTRY.get(cfg.PRIMARY) if cfg.META_MODEL == "none" else None


def _is_region(fam) -> bool:
    """A headline whose cells a region primary supplies, re-simulated by the replica (U24: `cell_matrix`)."""
    return region_kind(_region_cls(fam)) == "replica"


def _is_engine_region(fam) -> bool:
    """U28: a headline whose region cells run as engine cells (`cell_params`)."""
    return region_kind(_region_cls(fam)) == "engine"


def engine_cells(fam) -> dict[str, list]:
    """U28: region cell name → its stage-F cells over the family window: the headline with the primary's params
    replaced by the single cell's (primary.cell_params), one cell per instrument."""
    from primaries import make_primary

    prim = make_primary(fam.cells[HEADLINE][0].config())
    out = {}
    for name, params in prim.cell_params():
        cfg = copy.deepcopy(fam.headline.config)
        cfg["primary"] = {**cfg["primary"], "params": copy.deepcopy(params)}
        out[name] = make_cells(fam, cfg, *fam.window)
    return out


def engine_region(fam, ecells: dict[str, list], rows_by_hash: dict, hashes: dict, root: Path, rf, weights,
                  curve: dict, measured, refs_at, k: float) -> dict:  # fmt: skip
    """
    U28 (SPEC §27): the specification curve of an engine region. Each cell's pooled stream (its members re-priced from
    their own cost ledgers at every cost-curve column, as the variants are) → the mean daily excess return in bp with
    its Newey–West 95 % interval, the annualized excess Sharpe and, under a `marginal` test, the Δ Sharpe of core + k ×
    cell vs the core on their common days (the core as the headline reads it: `refs_at(column)`); the region is the
    headline's stream. The cells are engine runs, not a replica, so there is no parity to check (`parity_ok` True,
    `method` "engine"). `statistic` names the coherence statistic (`delta_sharpe` under a marginal test, else
    `alpha_bp`). A cell that did not complete is listed in `status` and has no entry (coherence counts it against).
    """
    from families.stats import mean_ci_nw, sharpe_ann

    marginal = fam.test["kind"] == "marginal"
    out: dict = {"method": "engine", "statistic": "delta_sharpe" if marginal else "alpha_bp", "cells": {},
                 "pooled_cells": {}, "region": {}, "names": list(ecells), "costs": list(curve), "parity": {},
                 "parity_ok": True, "status": {},
                 "note": "each cell run by the engine as its own rule cell (risk profile, costs, fills and cash yield "
                         "of the headline) and re-priced from its cost ledger at each column; the region is the "
                         "headline stream"}  # fmt: skip

    def stat(r: pd.Series, core: pd.Series | None) -> dict:
        x = T.excess(r, rf)
        mm, lo, hi = mean_ci_nw(x.to_numpy())
        s = {"alpha_bp": mm * 1e4, "ci_bp": [lo * 1e4, hi * 1e4], "sharpe": sharpe_ann(x), "n_days": len(x)}
        if core is not None:
            j = pd.concat([x.rename("a"), T.excess(core, rf).rename("c")], axis=1, sort=True).dropna()
            s["delta_sharpe"] = sharpe_ann(j["c"] + float(k) * j["a"]) - sharpe_ann(j["c"]) if len(j) > 2 else np.nan
        return s

    for ck, per in curve.items():
        core = (refs_at(ck) or {}).get(HEADLINE) if marginal else None
        cost = _curve_cost(ck, measured)
        for name, cells in ecells.items():
            p = _pooled(fam, cells, rows_by_hash, hashes, root, weights, cost)
            out["status"][name] = p["status"]
            if p["status"] != "ok":
                for c in cells:
                    out["cells"].setdefault(f"{c.symbols[0]}:{name}", {})
                continue
            for sym, r in p["per_instrument"].items():
                out["cells"].setdefault(f"{sym}:{name}", {})[ck] = stat(r, core)
            out["pooled_cells"].setdefault(name, {})[ck] = stat(p["stream"], core)
        if per.get(HEADLINE) is not None:
            out["region"].setdefault("pooled", {})[ck] = stat(per[HEADLINE][0], core)
    return out


def held_fraction(member_trades: list, weights: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """U28: the pooled position held at each session's close (Σ weight × side × the fraction committed at entry over
    the positions open across 16:00 NY), from the members' trades (entry_time / exit_time bar stamps)."""
    out = pd.Series(0.0, index=days)
    closes = (pd.DatetimeIndex(days).tz_localize("America/New_York") + pd.Timedelta(hours=16)).tz_convert("UTC")
    for sym, tr, _ in member_trades:
        if tr is None or not len(tr) or not {"entry_time", "exit_time", "side", "frac"} <= set(tr):
            continue
        w = float(weights.get(sym, 1.0))
        e = pd.to_datetime(tr["entry_time"], utc=True).to_numpy()
        x = pd.to_datetime(tr["exit_time"], utc=True).to_numpy()
        s = (tr["side"].astype(float) * tr["frac"].astype(float)).to_numpy()
        c = closes.to_numpy()
        held = (e[None, :] <= c[:, None]) & (x[None, :] > c[:, None])
        out += w * (held * s[None, :]).sum(axis=1)
    return out


def episode_paths(fam, headline: pd.Series, core: pd.Series | None, k: float, rf, pooled_h: dict, weights,
                  source) -> dict:  # fmt: skip
    """
    U28 (SPEC §27; PLAN3 G6's February 2018 and March 2020 paths): per spec episode [start, end), day by day, the
    headline's net return at the verdict's cost and its cumulative path, the position held at the close, the VIX close
    and VX1 / VIX settlement of the day (the next session's decision reads them) and, with a core (marginal), the core
    and core + k × overlay (the overlay's excess, as the test adds it). Reported only.
    """
    h = T.day_index(headline)
    state = {}
    for name in ("VIX", "VX1_VIX"):
        try:
            v = source.exo("cboe", name, fam.window[0], fam.window[1], allow_holdout=False)["value"]
            v.index = pd.DatetimeIndex(v.index).normalize()
            state[name] = v
        except Exception as e:  # noqa: BLE001 — a reported table only: say so
            print(f"[FAM]  no {name} for the episode paths ({type(e).__name__}: {e})")
    pos = held_fraction(pooled_h.get("member_trades") or [], weights, pd.DatetimeIndex(h.index))
    rfd = T.day_index(rf) if rf is not None else None
    c = T.day_index(core) if core is not None else None
    out = {}
    for ep in fam.episodes:
        d = h.loc[(h.index >= pd.Timestamp(ep["start"])) & (h.index < pd.Timestamp(ep["end"]))]
        rows, cum, pcum, peak, worst_dd = [], 1.0, 1.0, 1.0, 0.0
        for day, r in d.items():
            cum *= 1.0 + float(r)
            peak = max(peak, cum)
            worst_dd = min(worst_dd, cum / peak - 1.0)
            row = {"date": str(day.date()), "position": float(pos.get(day, np.nan)), "ret": float(r), "cum": cum - 1.0,
                   "vix": float(state["VIX"].get(day, np.nan)) if "VIX" in state else None,
                   "vx1_vix": float(state["VX1_VIX"].get(day, np.nan)) if "VX1_VIX" in state else None}  # fmt: skip
            if c is not None:
                cr = float(c.get(day, np.nan))
                ex = float(r) - (float(rfd.get(day, 0.0)) if rfd is not None else 0.0)
                pr = cr + float(k) * ex
                pcum *= 1.0 + (pr if np.isfinite(pr) else 0.0)
                row |= {"core": cr, "portfolio": pr, "portfolio_cum": pcum - 1.0}
            rows.append(row)
        summary = {"days": len(d), "total": cum - 1.0, "worst_day": float(d.min()) if len(d) else np.nan,
                   "max_dd": worst_dd, "days_held": int((pos.reindex(d.index).abs() > 0).sum())}  # fmt: skip
        if c is not None:
            cc = c.reindex(d.index)
            summary |= {"core_total": float((1.0 + cc.fillna(0.0)).prod() - 1.0), "portfolio_total": pcum - 1.0}
        out[ep["label"]] = {"start": ep["start"], "end": ep["end"], "rows": rows, "summary": summary}
    return out


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
    measured = measured_profile()
    if at_cost == "measured" and measured is None:
        raise ValueError(f"test.at_cost 'measured' needs {MEASURED_COST_CSV}")
    out: dict = {"cells": {}, "pooled_cells": {}, "region": {}, "parity": {}, "names": [],
                 "costs": ["registered", *map(str, curve), *(["measured"] if measured is not None else [])],
                 "note": "each cell re-simulated by wfo.position_backtest on the member's inputs; `measured` prices "
                         "decision fills at the profile's `open` class and the closing fill at `close_auction` "
                         "(`close` under FILL_AUCTION last_bar)"}  # fmt: skip
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
        at = df.index.get_indexer([pd.Timestamp(rows_by_hash[h]["live_start"])])[0]
        if at < 0:
            raise ValueError(f"{sym}: the member's live_start {rows_by_hash[h]['live_start']} is not a bar of its data")
        live_day = day[at]
        b0 = int(np.flatnonzero(day == live_day)[0])  # the first bar of the member's first live session
        bars = df.iloc[b0:]
        keep = dec >= b0
        frame = pd.DataFrame(m[keep], index=df.index[dec[keep]], columns=names)
        frame["_region"] = frame[names].to_numpy().mean(axis=1) * p.target_scale(df)[dec[keep]]
        clock = data.get("clock")
        kw = {"prints": (data.get("prints") or {}).get(sym), "cash_yield": data.get("cash_yield"),
              "profile": get_profile(cfg.RISK_PROFILE), "sessions": clock, "grid": decision_grid(bars, cfg, clock)}  # fmt: skip
        booked = None
        if cfg.COST_MODEL != "slippage":
            booked = fill_costs(bars.index, cfg, (data.get("cost") or {}).get(sym), stress=data.get("stress"))
        rets = {"registered": position_backtest(bars, frame, cfg, costs=booked, **kw).ret}
        for bp in curve:
            rets[str(bp)] = position_backtest(bars, frame, cfg, costs=bp / 2.0 * 1e-4, **kw).ret
        if measured is not None:  # decisions fill at bar opens after 09:30 (class `open`); the last exit at the close
            close_cls = "close_auction" if cfg.FILL_AUCTION == "print" else "close"
            mc = pd.DataFrame({"open": measured["open"] * 1e-4, "close": measured[close_cls] * 1e-4}, index=bars.index)
            rets["measured"] = position_backtest(bars, frame, cfg, costs=mc, **kw).ret
        member = T.day_index(_stream(root, h))
        reg = T.day_index(rets["registered"]["_region"])
        j = pd.concat([member.rename("m"), reg.rename("r")], axis=1, sort=True)
        both = j.dropna()
        out["parity"][sym] = {"max_abs_diff": float((both["m"] - both["r"]).abs().max()) if len(both) else None,
                              "member_days": int(j["m"].notna().sum()), "resim_days": int(j["r"].notna().sum()),
                              "common_days": len(both)}  # fmt: skip
        by_sym[sym] = rets
    out["parity_ok"] = all(v["max_abs_diff"] is not None and v["max_abs_diff"] <= PARITY_TOL and v["common_days"] > 0
                           for v in out["parity"].values())  # fmt: skip

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
