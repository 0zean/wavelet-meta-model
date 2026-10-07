"""
Ledger report (SPEC §9, U10): stage funnel, leaderboard with DSR, PBO per stage → Markdown + HTML + CSV.

DSR of a row = PSR(SR₀) of its daily returns with N = n_trials through the row's stage (experiments.ledger) and
V[SR_n] = variance of the per-period Sharpes of those trials; `dsr_all` uses every trial in the ledger (the deflation
a pick made from the whole table faces). Rows without daily return moments (legacy imports, no
trades) get no DSR. PBO per stage = CSCV (validation/pbo.py, PBO_BLOCKS) over the daily returns of the stage's runner
cells (`<root>/cells/<hash>/daily_returns.csv`) on the days all of them cover, identical streams counted once.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import ledger as L
from experiments.spec import STAGES, normalize, stage_rank
from utils.config import DEFAULT_FEATURE_GROUPS, RunConfig
from validation.pbo import pbo
from validation.stats import dsr

TRADING_DAYS = 252


def row_dsr(row: dict, rows: list[dict], through: str | None = None) -> tuple[float, int, float]:
    """(DSR, N, V) of one trial row against the ledger's trials through `through` (default: the row's stage)."""
    through = through or row["stage"]
    n, v = L.n_trials(rows, through), L.var_trials(rows, through)
    need = ("sharpe", "n_obs", "sr_skew", "sr_kurt")
    if row.get("kind") == "legacy" or any(row.get(k) is None for k in need) or row["n_obs"] < 2:
        return np.nan, n, v
    sr = row["sharpe"] / np.sqrt(TRADING_DAYS)
    return float(dsr(sr, max(n, 1), v, int(row["n_obs"]), row["sr_skew"], row["sr_kurt"])), n, v


def leaderboard(rows: list[dict], stage: str | None = None) -> pd.DataFrame:
    latest = L.done(rows)  # (hash, stage) → latest row
    last = max((r["stage"] for r in L.trials(rows)), key=stage_rank, default="U6")
    recs = []
    for (_, st), r in latest.items():
        if r.get("status") != "ok" or (stage is not None and st != stage):
            continue
        d, n, _ = row_dsr(r, rows)
        d_all, n_all, _ = row_dsr(r, rows, last)
        recs.append(
            {
                "stage": st,
                "label": r.get("label", ""),
                "kind": r.get("kind"),
                "sharpe": r.get("sharpe"),
                "psr": r.get("psr"),
                "dsr": d,
                "n_trials": n,
                "dsr_all": d_all,  # against every trial in the ledger (a pick made from this table faces them all)
                "n_trials_all": n_all,
                "ret_ann": r.get("ret_ann"),
                "max_dd": r.get("max_dd"),
                "meta_auc": r.get("meta_auc"),
                "n_trades": r.get("n_trades"),
                "n_obs": r.get("n_obs"),
                "wfe": r.get("wfe"),
                "pbo": r.get("pbo"),
                "cache_hit": bool(r.get("cache_hit", False)),
                "cell_hash": r["cell_hash"],
            }
        )
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["_rank"] = df["stage"].map(stage_rank)
    return (
        df.sort_values(["_rank", "dsr", "sharpe"], ascending=[True, False, False], na_position="last")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def funnel(rows: list[dict]) -> pd.DataFrame:
    latest = list(L.done(rows).values())
    recs = []
    for st in STAGES:
        rs = [r for r in latest if r["stage"] == st]
        if not rs:
            continue
        status = pd.Series([r.get("status") for r in rs]).value_counts()
        recs.append(
            {
                "stage": st,
                "cells": len(rs),
                "ok": int(status.get("ok", 0)),
                "no_fit": int(status.get("no_fit", 0)),
                "error": int(status.get("error", 0)),
                "cache_hits": sum(bool(r.get("cache_hit")) for r in rs),
                "trials_in_stage": int(sum(int(r.get("n_trials", 1)) for r in rs if r.get("status") in L.COUNTED)),
                "n_trials_cumulative": L.n_trials(rows, st),
            }
        )
    return pd.DataFrame(recs)


def stage_pbo(rows: list[dict], root, n_blocks: int | None = None) -> pd.DataFrame:
    n_blocks = n_blocks or RunConfig().PBO_BLOCKS
    latest = L.done(rows)
    recs = []
    for st in STAGES:
        cols = {}
        for (h, s), r in latest.items():
            path = Path(root) / "cells" / h / "daily_returns.csv"
            if s == st and r.get("status") == "ok" and r.get("kind") != "legacy" and path.exists():
                cols[h] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if not cols:
            continue
        common = pd.DataFrame(cols).dropna(how="any")
        # identical return streams (one configuration re-run under new code) would enter CSCV as tied duplicates
        common = common.T.drop_duplicates().T
        rec = {"stage": st, "n_cells": len(cols), "n_distinct": common.shape[1], "n_common_days": len(common),
               "pbo": np.nan}  # fmt: skip
        if common.shape[1] >= 2 and len(common) >= 2 * n_blocks:
            res = pbo(common, n_blocks)
            rec |= {"pbo": res.pbo, "prob_oos_loss": res.prob_oos_loss, "degradation_slope": res.degradation_slope}
        recs.append(rec)
    return pd.DataFrame(recs)


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def _markdown(df: pd.DataFrame) -> str:
    cells = [[str(c) for c in df.columns]] + [
        ["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)
    ]
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in row) + " |" for row in cells]
    return "\n".join([lines[0], "|" + "---|" * len(df.columns), *lines[1:]])


def write_report(ledger: L.Ledger, root, out, stage: str | None = None, top: int = 50) -> dict[str, Path]:
    rows = ledger.rows()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lb, fn, pb = leaderboard(rows, stage), funnel(rows), stage_pbo(rows, root)
    errors = [r for r in L.done(rows).values() if r.get("status") == "error"]
    holdout = L.holdout_events(rows)
    lb.to_csv(out / "leaderboard.csv", index=False)
    sections = [
        ("Stage funnel", fn),
        ("PBO per stage", pb),
        (f"Leaderboard (top {top} per stage by DSR)", lb.groupby("stage", sort=False).head(top) if len(lb) else lb),
    ]
    notes = [
        (
            f"Ledger: {ledger.path} — {len(L.trials(rows))} trial rows, {len(errors)} cell(s) in error "
            "(errors are not counted in N)."
        ),
        "N = distinct counted (ok / no_fit) cell hashes through the stage, a PWFO cell counting its grid size.",
        "Holdout accessed: " + (", ".join(f"{e['batch_id']} at {e['started_at']}" for e in holdout) or "never") + ".",
    ]
    md = ["# Experiment ledger report", "", *[f"- {n}" for n in notes], ""]
    for title, df in sections:
        md += [f"## {title}", "", _markdown(_fmt(df)) if len(df) else "_none_", ""]
    if errors:
        md += ["## Errors", ""] + [f"- `{r['cell_hash']}` {r.get('label', '')}: {r.get('error')}" for r in errors]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    css = (
        "body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#0b0b0b;"
        "background:#fcfcfb}table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:8px 0 24px}"
        "th,td{padding:4px 10px;border-bottom:1px solid #e4e3dc;text-align:right}th{color:#52514e;font-weight:600}"
        "td:nth-child(2){text-align:left}.wrap{overflow-x:auto}"
        "@media (prefers-color-scheme:dark){body{background:#151514;color:#ecebe6}th,td{border-color:#33322f}}"
    )
    body = ["<h1>Experiment ledger report</h1><ul>", *[f"<li>{html.escape(n)}</li>" for n in notes], "</ul>"]
    for title, df in sections:
        table = _fmt(df).to_html(index=False, border=0, escape=True) if len(df) else "<p><em>none</em></p>"
        body += [f"<h2>{html.escape(title)}</h2><div class='wrap'>{table}</div>"]
    if errors:
        body += ["<h2>Errors</h2><ul>"]
        body += [f"<li><code>{r['cell_hash']}</code> {html.escape(str(r.get('label', '')))}: "
                 f"{html.escape(str(r.get('error')))}</li>" for r in errors]  # fmt: skip
        body += ["</ul>"]
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>Experiment report</title><style>{css}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )
    (out / "report.html").write_text(page, encoding="utf-8")
    return {"md": out / "report.md", "html": out / "report.html", "csv": out / "leaderboard.csv"}


# ── U11 5Min pilot rule (PLAN U11, pre-registered 2026-10-01) ────────────────

PILOT_AUC_GATE = 0.52  # Stage A's survivor AUC gate


def weighted_auc(inv: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC = P(p⁺ > p⁻) + ½ P(tie) with event weights `w`; `inv` = each event's rank among the distinct probabilities
    (np.unique(prob, return_inverse=True)[1]), y = 0/1 outcome."""
    pos = np.bincount(inv, weights=w * y)
    neg = np.bincount(inv, weights=w * (1 - y))
    return float((pos * (np.cumsum(neg) - 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def session_bootstrap_auc(success, prob, sessions, n_boot: int = 2000, q: float = 0.95, seed: int = 0):
    """(AUC, one-sided upper bound): the q-quantile of the AUC over `n_boot` resamples of whole sessions with
    replacement (a session's events move together: overlapping barriers and shared intraday regime make them
    dependent). Resamples without both outcomes are skipped."""
    y = np.asarray(success, dtype=float)
    inv = np.unique(np.asarray(prob, dtype=float), return_inverse=True)[1].ravel()
    codes, uniq = pd.factorize(pd.Index(sessions))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
            boots.append(weighted_auc(inv, y, w))
    return weighted_auc(inv, y, np.ones_like(y)), float(np.quantile(boots, q))


def pilot_rule(spec_path, root, timeframe: str = "5Min", source=None, n_boot: int = 2000) -> tuple[pd.DataFrame, str]:
    """
    The pre-registered 5Min rule over the `timeframe` cells of `spec_path`, from their cached per-symbol WFO signals
    (`<root>/signals/<key>.pkl`, so it can run before the cells' ledger rows exist): per cell the OOS meta AUC on
    scored events (as the runner's Meta AUC) and its session-bootstrap one-sided 95 % upper bound. Verdict "drop"
    iff every cell's bound is < PILOT_AUC_GATE (a cell without scored events fails the gate); "undecided" while
    any cell's signals are missing.
    """
    from experiments.runner import CachedBars, load_cell_data, signals_key
    from experiments.spec import load_spec
    from wfo.wfo_metrics import meta_outcomes

    recs = []
    for cell in [c for c in load_spec(spec_path)[1] if c.spec["timeframe"] == timeframe]:
        cfg = cell.config(False)
        ((sym, df),) = load_cell_data(cell, cfg, source or CachedBars(), False)["bars"].items()
        path = Path(root) / "signals" / f"{signals_key(sym, cfg, df)}.pkl"
        rec = {"label": cell.label(), "n_events": 0, "n_sessions": 0, "auc": np.nan, "auc_ub95": np.nan}
        if not path.exists():
            recs.append(rec | {"status": "missing"})
            continue
        o = meta_outcomes(df, pd.read_pickle(path), cfg)
        s = o[o["scored"]]
        rec |= {"status": "ok", "n_events": len(s)}
        if s["success"].nunique() == 2:
            sessions = s.index.tz_convert("America/New_York").date
            auc, ub = session_bootstrap_auc(s["success"], s["meta_prob"], sessions, n_boot)
            rec |= {"n_sessions": len(set(sessions)), "auc": auc, "auc_ub95": ub}
        recs.append(rec)
    table = pd.DataFrame(recs)
    table["below_gate"] = table["auc_ub95"].lt(PILOT_AUC_GATE) | (table["status"].eq("ok") & table["auc_ub95"].isna())
    if table["status"].eq("missing").any():
        verdict = "undecided"
    else:
        verdict = "drop" if table["below_gate"].all() else "run"
    return table, verdict


# ── U11 Stage A survivors (PLAN U11, pre-registered 2026-10-03) ──────────────

SURVIVOR_AUC = 0.515  # Stage A's 0.52 gate less the xgb proxy's offset to rf_ldp (pilot: 0.0049 ± 0.0016)
SURVIVOR_PSR = 0.5  # i.e. a positive Sharpe after costs
SURVIVOR_K = 20
SURVIVOR_PER_PAIR = 2  # per (symbol, timeframe): its primaries share events and features


def stage_a_survivors(rows: list[dict], stage: str = "A", meta: str = "xgb", k: int = SURVIVOR_K,
                      per_pair: int = SURVIVOR_PER_PAIR) -> pd.DataFrame:  # fmt: skip
    """
    Every `ok` row of `stage` with the screen's meta-model, one per (symbol, timeframe, primary): `passed` = meta AUC
    > SURVIVOR_AUC and PSR > SURVIVOR_PSR (and a DSR); `survivor` = the passed rows ranked by DSR (then Sharpe, then
    label), at most `per_pair` per (symbol, timeframe), the first `k`. Rows of other meta-models (rf_ldp fits
    recorded before the amendment) count as trials but select nothing: the better of two models per cell would be
    an extra, uncounted selection.
    """
    recs = []
    for (h, st), r in L.done(rows).items():
        if st != stage or r.get("status") != "ok" or r.get("kind") == "legacy":
            continue
        spec = json.loads(r["spec_json"])
        if spec["model"]["meta"] != meta or len(spec["symbols"]) != 1:
            continue
        d, n, _ = row_dsr(r, rows)
        recs.append({"symbol": spec["symbols"][0], "timeframe": spec["timeframe"], "primary": spec["primary"]["name"],
                     "meta_auc": r.get("meta_auc"), "psr": r.get("psr"), "sharpe": r.get("sharpe"), "dsr": d,
                     "n_trials": n, "n_trades": r.get("n_trades"), "label": r.get("label", ""), "cell_hash": h})  # fmt: skip
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    num = df[["meta_auc", "psr", "dsr"]].apply(pd.to_numeric, errors="coerce")
    df["passed"] = num["meta_auc"].gt(SURVIVOR_AUC) & num["psr"].gt(SURVIVOR_PSR) & num["dsr"].notna()
    df = df.sort_values(["dsr", "sharpe", "label"], ascending=[False, False, True], na_position="last")
    picked = df[df["passed"]].groupby(["symbol", "timeframe"], sort=False).head(per_pair).head(k)
    df["survivor"] = df["cell_hash"].isin(picked["cell_hash"])
    return df.reset_index(drop=True)


# ── U11 Stage B1 → B2 selection (PLAN U11, pre-registered 2026-10-03, before any B1 ledger row) ────────────────

B1_MODELS = ("xgb", "logit_l2", "rf_ldp_fast", "catboost")  # xgb = the survivor's Stage A row (not refit)
B1_PSR = 0.5
B1_TOP_DAYS = 5  # the best 5 days must not hold >= 100 % of the summed daily P&L


def alpha_stats(ret: pd.Series, bh: pd.Series, top_days: int = B1_TOP_DAYS) -> dict:
    """
    `ret` (strategy daily returns) against `bh` (the symbol's buy-and-hold daily returns): OLS beta on the days both
    cover, the annualized Sharpe of ret − beta·bh (alpha Sharpe), and the share of Σ ret earned on the `top_days`
    best days (inf when Σ ret ≤ 0).
    """
    j = pd.concat([ret, bh], axis=1, keys=["r", "b"], sort=True).dropna()
    var = j["b"].var()
    beta = float(j["r"].cov(j["b"]) / var) if len(j) > 2 and var > 0 else 0.0
    a = j["r"] - beta * j["b"]
    sd = a.std()
    alpha_sr = float(a.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    tot = float(ret.sum())
    share = float(ret.nlargest(top_days).sum() / tot) if tot > 0 else np.inf
    return {"beta": beta, "alpha_sr": alpha_sr, "top_days_share": share}


def _cached_bh(spec: dict, allow_holdout: bool = False) -> pd.Series:
    """Buy-and-hold daily returns of the cell's symbol: last close of each NY session, from the cell's own bars."""
    from experiments.runner import CachedBars

    df = CachedBars().bars(spec["symbols"][0], spec["timeframe"], spec["start"], spec["end"],
                           allow_holdout=allow_holdout)  # fmt: skip
    day = df.index.tz_convert("America/New_York").normalize()
    return df["close"].groupby(day).last().pct_change()


def _daily_returns(root, h: str) -> pd.Series:
    s = pd.read_csv(Path(root) / "cells" / h / "daily_returns.csv", index_col=0)["ret"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").normalize()
    return s


def stage_b1_selection(rows: list[dict], root, bh_returns=_cached_bh) -> pd.DataFrame:
    """
    One row per Stage A survivor (`stage_a_survivors(...)["passed"]`, PLAN amendment 2026-10-03). Candidates: its
    Stage A xgb row plus the `ok` stage-B rows whose spec is the survivor's with only `model.meta` changed to another
    of B1_MODELS. `model` = the candidate with the highest OOS meta AUC (ties → PSR, then label); `passed` iff that
    row has PSR > B1_PSR, alpha Sharpe > 0 and top-days share < 1 (alpha_stats on its daily returns vs buy-and-hold).
    `n_models` < len(B1_MODELS) means a model's row is missing (not run, error, no_fit): the rule is final only at 4.
    """
    surv = stage_a_survivors(rows)
    if surv.empty:
        return surv
    surv = surv[surv["passed"]]
    a_rows = {h: r for (h, st), r in L.done(rows).items() if st == "A"}

    def key(spec: dict) -> str:
        return json.dumps({**spec, "model": {**spec["model"], "meta": "xgb"}}, sort_keys=True)

    by_key = {key(json.loads(a_rows[h]["spec_json"])): h for h in surv["cell_hash"]}
    cands = {h: [a_rows[h]] for h in surv["cell_hash"]}
    for (_h, st), r in L.done(rows).items():
        if st != "B" or r.get("status") != "ok":
            continue
        spec = json.loads(r["spec_json"])
        if spec["model"]["meta"] in B1_MODELS[1:] and key(spec) in by_key:
            cands[by_key[key(spec)]].append(r)
    recs = []
    for s in surv.itertuples():
        c = cands[s.cell_hash]
        aucs = {json.loads(r["spec_json"])["model"]["meta"]: r.get("meta_auc") for r in c}
        best, picked = _pick_and_gate(c, root, bh_returns)
        rec = {"symbol": s.symbol, "timeframe": s.timeframe, "primary": s.primary,
               "model": json.loads(best["spec_json"])["model"]["meta"], "n_models": len(c), **picked,
               "a_cell_hash": s.cell_hash, **{f"auc_{m}": aucs.get(m, np.nan) for m in B1_MODELS}}  # fmt: skip
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["passed", "psr"], ascending=[False, False]).reset_index(drop=True)


def _pick_and_gate(cands: list[dict], root, bh_returns) -> tuple[dict, dict]:
    """The candidate row with the highest OOS meta AUC (ties → PSR, then label) and its gates (B1 / B2 rules)."""
    best = min(cands, key=lambda r: (-(r.get("meta_auc") or -np.inf), -(r.get("psr") or -np.inf),
                                     r.get("label", "")))  # fmt: skip
    return best, _gated(best, root, bh_returns)


def _gated(row: dict, root, bh_returns) -> dict:
    """A row's selection stats and the B1 gates: PSR > B1_PSR, alpha Sharpe > 0, top-days share < 1."""
    st = alpha_stats(_daily_returns(root, row["cell_hash"]), bh_returns(json.loads(row["spec_json"])))
    rec = {"meta_auc": row.get("meta_auc"), "psr": row.get("psr"), "sharpe": row.get("sharpe"), **st,
           "label": row.get("label", ""), "cell_hash": row["cell_hash"]}  # fmt: skip
    rec["passed"] = bool((rec["psr"] or 0) > B1_PSR and rec["alpha_sr"] > 0 and rec["top_days_share"] < 1)
    return rec


# ── U11 Stage B2 → B3 selection (PLAN U11, pre-registered 2026-10-04, before any B2 ledger row) ────────────────

B2_FULL = (*DEFAULT_FEATURE_GROUPS, "wavelet_ext", "structural", "calendar")  # cross_asset: no market bars in runner
B2_ARMS = {"default": (DEFAULT_FEATURE_GROUPS, "none"), "cmda": (DEFAULT_FEATURE_GROUPS, "cmda"),
           "full": (B2_FULL, "none"), "full_cmda": (B2_FULL, "cmda")}  # fmt: skip  # "default" = the B1 row


def feature_arm(spec: dict) -> str | None:
    """The B2 arm a cell's (feature_groups, FEATURE_SELECTION) is, or None."""
    fs = (tuple(spec["feature_groups"]), spec["overrides"].get("FEATURE_SELECTION", "none"))
    return next((a for a, (g, sel) in B2_ARMS.items() if fs == (tuple(g), sel)), None)


def stage_b2_selection(rows: list[dict], root, bh_returns=_cached_bh) -> pd.DataFrame:
    """
    One row per B1 passer (`stage_b1_selection(...)["passed"]`). Candidates: its B1 row (arm "default") plus the `ok`
    stage-B rows whose spec is that row's with only the feature arm changed (B2_ARMS). `arm` = the candidate with the
    highest OOS meta AUC; `passed` by the B1 gates. `n_arms` < len(B2_ARMS) means an arm's row is missing.
    """
    b1 = stage_b1_selection(rows, root, bh_returns)
    if b1.empty:
        return b1
    b1 = b1[b1["passed"]]
    by_hash = {h: r for (h, _st), r in L.done(rows).items()}

    def key(spec: dict) -> str:
        o = {k: v for k, v in spec["overrides"].items() if k != "FEATURE_SELECTION"}
        return json.dumps({**spec, "feature_groups": None, "overrides": o}, sort_keys=True)

    base = {key(json.loads(by_hash[h]["spec_json"])): h for h in b1["cell_hash"]}
    cands = {h: {h: by_hash[h]} for h in b1["cell_hash"]}
    for (h, st), r in L.done(rows).items():
        if st != "B" or r.get("status") != "ok":
            continue
        spec = json.loads(r["spec_json"])
        if feature_arm(spec) not in (None, "default") and key(spec) in base:
            cands[base[key(spec)]][h] = r
    recs = []
    for s in b1.itertuples():
        c = list(cands[s.cell_hash].values())
        aucs = {feature_arm(json.loads(r["spec_json"])) or "default": r.get("meta_auc") for r in c}
        best, picked = _pick_and_gate(c, root, bh_returns)
        rec = {"symbol": s.symbol, "timeframe": s.timeframe, "primary": s.primary, "model": s.model,
               "arm": feature_arm(json.loads(best["spec_json"])) or "default", "n_arms": len(c), **picked,
               "b1_cell_hash": s.cell_hash, **{f"auc_{a}": aucs.get(a, np.nan) for a in B2_ARMS}}  # fmt: skip
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["passed", "psr"], ascending=[False, False]).reset_index(drop=True)


# ── U11 Stage B3 → C selection (PLAN U11, pre-registered 2026-10-05, before any B3 ledger row) ─────────────────

B3_SIZERS = ("linear", "ldp_sigmoid", "ecdf")  # vs the B2 row's `fixed` (not refit); single position mode


def stage_b3_selection(rows: list[dict], root, bh_returns=_cached_bh) -> pd.DataFrame:
    """
    One row per B2 passer (`stage_b2_selection(...)["passed"]`). Candidates: its B2 row (sizer `fixed`) plus the `ok`
    stage-B rows whose spec is that row's with only `sizer` changed to one of B3_SIZERS. `sizer` = `fixed` unless a
    sizer row passes the gates with a higher PSR than `fixed` — then the highest-PSR such row (ties → label);
    `passed` = the chosen row's gates. `n_sizers` < 1 + len(B3_SIZERS) means a sizer's row is missing.
    """
    b2 = stage_b2_selection(rows, root, bh_returns)
    if b2.empty:
        return b2
    b2 = b2[b2["passed"]]
    by_hash = {h: r for (h, _st), r in L.done(rows).items()}

    def key(spec: dict) -> str:
        return json.dumps({**spec, "sizer": None}, sort_keys=True)

    base = {key(json.loads(by_hash[h]["spec_json"])): h for h in b2["cell_hash"]}
    cands = {h: [] for h in b2["cell_hash"]}
    for (h, st), r in L.done(rows).items():
        if st != "B" or r.get("status") != "ok":
            continue
        spec = json.loads(r["spec_json"])
        if spec.get("sizer") in B3_SIZERS and key(spec) in base:
            cands[base[key(spec)]].append(r)
    recs = []
    for s in b2.itertuples():
        fixed = _gated(by_hash[s.cell_hash], root, bh_returns)
        sized = {json.loads(r["spec_json"])["sizer"]: _gated(r, root, bh_returns) for r in cands[s.cell_hash]}
        better = [(z, g) for z, g in sized.items() if g["passed"] and (g["psr"] or 0) > (fixed["psr"] or 0)]
        sizer, picked = min(better, key=lambda zg: (-zg[1]["psr"], zg[1]["label"])) if better else ("fixed", fixed)
        rec = {"symbol": s.symbol, "timeframe": s.timeframe, "primary": s.primary, "model": s.model, "arm": s.arm,
               "sizer": sizer, "n_sizers": 1 + len(sized), **picked, "b2_cell_hash": s.cell_hash,
               **{f"psr_{z}": (fixed if z == "fixed" else sized.get(z, {})).get("psr", np.nan)
                  for z in ("fixed", *B3_SIZERS)}}  # fmt: skip
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["passed", "psr"], ascending=[False, False]).reset_index(drop=True)


# Stage C finalists (PLAN U11, registered 2026-10-06 after B3 finished, before any B3 result was looked at)
C_PER_SYMBOL = 2  # a symbol's cells share its price path: Stage D needs diverse finalists


def stage_c_finalists(b3: pd.DataFrame, per_symbol: int = C_PER_SYMBOL) -> pd.DataFrame:
    """`b3` (stage_b3_selection) with `finalist`: its passed rows ranked by PSR (ties → label), the first
    `per_symbol` of each symbol."""
    if b3.empty:
        return b3.assign(finalist=pd.Series(dtype=bool))
    ranked = b3[b3["passed"]].sort_values(["psr", "label"], ascending=[False, True])
    picked = ranked.groupby("symbol", sort=False).head(per_symbol)
    return b3.assign(finalist=b3["cell_hash"].isin(picked["cell_hash"]))


# ── U11 Stage C grid and C → D selection (PLAN U11, pre-registered 2026-10-06, before any Stage C cell ran) ──────

# (IS grid, OOS grid, PWFO_DEFAULT) per timeframe. U9's default IS {63, 126, 252, 504} cannot fit a 1Day window
# (~0.25 events / session: train >= 200 events needs ~800+ sessions) and its 63 / 126 rows are unfittable intraday.
C_GRIDS = {"1Day": ((1260, 1512), (5, 10, 21, 63), (1512, 10))}
C_GRID_INTRADAY = ((252, 378, 504, 756), (5, 10, 21, 63), (252, 10))
C_PBO_MAX = 0.5  # PBO across the cell's combos must be < this (NaN fails)


def c_grid(timeframe: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, int]]:
    return C_GRIDS.get(timeframe, C_GRID_INTRADAY)


def stage_c_cell(spec: dict) -> dict:
    """A B3 row's spec as its Stage C PWFO cell: the same spec with `pwfo` = its timeframe's grid (rolling) and
    PWFO_DEFAULT set explicitly."""
    is_grid, oos_grid, default = c_grid(spec["timeframe"])
    return {**spec, "pwfo": {"is_grid": list(is_grid), "oos_grid": list(oos_grid), "expanding": False},
            "overrides": {**spec["overrides"], "PWFO_DEFAULT": list(default)}}  # fmt: skip


def stage_c_selection(rows: list[dict], finalists: pd.DataFrame, root, bh_returns=_cached_bh) -> pd.DataFrame:
    """
    One row per Stage C finalist (`finalists`: stage_b3_selection rows with `finalist`; only those are used). Its
    Stage C row = the stage-C ledger row whose spec is `stage_c_cell(finalist spec)` (ok or no_fit). `passed` iff that
    row is `ok` and its nested-PWFO live stream (daily_returns.csv) has PSR > B1_PSR, alpha Sharpe > 0, top-days share
    < 1 (the B gates) and the cell's PBO across combos < C_PBO_MAX. `has_row` False = not run yet (or error): final
    only when every finalist has its row.
    """
    fin = finalists[finalists["finalist"]]
    by_hash = {h: r for (h, _st), r in L.done(rows).items()}
    want = {}
    for h in fin["cell_hash"]:
        want[json.dumps(normalize(stage_c_cell(json.loads(by_hash[h]["spec_json"]))), sort_keys=True)] = h
    got = {}
    for (_h, st), r in L.done(rows).items():
        if st == "C" and r.get("status") in ("ok", "no_fit"):
            key = json.dumps(json.loads(r["spec_json"]), sort_keys=True)
            if key in want:
                got[want[key]] = r
    recs = []
    for s in fin.itertuples():
        base = {"symbol": s.symbol, "timeframe": s.timeframe, "primary": s.primary, "model": s.model, "arm": s.arm,
                "sizer": s.sizer, "b3_psr": s.psr, "b3_cell_hash": s.cell_hash}  # fmt: skip
        r = got.get(s.cell_hash)
        if r is None or r["status"] != "ok":
            recs.append({**base, "has_row": r is not None, "status": None if r is None else r["status"],
                         "passed": False})  # fmt: skip
            continue
        g = _gated(r, root, bh_returns)
        pbo_ = r.get("pbo")
        pbo_ = np.nan if pbo_ is None else float(pbo_)
        picks = r.get("picks") or {}
        recs.append({**base, "has_row": True, "status": "ok", **{k: v for k, v in g.items() if k != "meta_auc"},
                     "pbo": pbo_, "pwfo_dsr": r.get("pwfo_dsr"), "n_trials": r.get("n_trials"),
                     "top_pick": max(picks, key=picks.get) if picks else None,
                     "passed": bool(g["passed"] and pbo_ < C_PBO_MAX)})  # fmt: skip
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values(["passed", "b3_psr"], ascending=[False, False]).reset_index(drop=True)


# ── U11 Stage E: holdout cells and per-finalist verdict (PLAN U11, pre-registered 2026-10-07, before E ran) ──────

E_END = "2026-09-27"  # the cached bars' coverage end: no Alpaca top-up (which could re-adjust the history)
E_DSR = 0.95  # "edge": holdout DSR (N = every counted trial in the ledger, E included) > E_DSR and alpha Sharpe > 0
E_ALPHA = 0.05  # else "positive, not significant after deflation": Holm-adjusted 1 − PSR(0) < E_ALPHA over the E rows
E_C_TAIL = 5  # the C stream's last days close its positions at the C data end; compared up to here


def stage_e_cell(spec: dict) -> dict:
    """A Stage C cell's spec as its Stage E cell: the same procedure run through E_END, with a partial last OOS
    window so the holdout is scored to the end of the data."""
    return {**spec, "end": E_END, "overrides": {**spec["overrides"], "PWFO_PARTIAL_LAST": True}}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def _stream(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0)["ret"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").normalize()
    return s


def stage_e_selection(rows: list[dict], c_rows: list[dict], root, bh_returns=None) -> pd.DataFrame:
    """
    One row per Stage C cell in `c_rows` (the 14 finalists' C rows). Its E row = the stage-E ledger row whose spec is
    `stage_e_cell` of its C spec. Holdout statistics are the E row's (its daily_returns.csv = holdout days only).
    `verdict`: "edge" iff DSR (N, V over every counted trial of the ledger) > E_DSR and alpha Sharpe > 0; else
    "positive_not_significant" iff the Holm-adjusted 1 − PSR(0) over all E rows < E_ALPHA; else "no_edge".
    `c_max_diff` = max |E full stream − C stream| on the C days before its last E_C_TAIL (integrity: E continues C).
    """
    bh_returns = bh_returns or (lambda spec: _cached_bh(spec, allow_holdout=True))
    want = {json.dumps(normalize(stage_e_cell(json.loads(r["spec_json"]))), sort_keys=True): r for r in c_rows}
    got = {}
    for (_h, st), r in L.done(rows).items():
        if st == "E" and r.get("status") in ("ok", "no_fit"):
            key = json.dumps(json.loads(r["spec_json"]), sort_keys=True)
            if key in want:
                got[want[key]["cell_hash"]] = r
    recs = []
    for c in c_rows:
        spec = json.loads(c["spec_json"])
        base = {"symbol": spec["symbols"][0], "timeframe": spec["timeframe"], "primary": spec["primary"]["name"],
                "model": spec["model"]["meta"], "sizer": spec["sizer"], "c_sharpe": c.get("sharpe"),
                "c_cell_hash": c["cell_hash"]}  # fmt: skip
        e = got.get(c["cell_hash"])
        if e is None or e["status"] != "ok":
            recs.append({**base, "has_row": e is not None, "status": None if e is None else e["status"]})
            continue
        g = _gated(e, root, bh_returns)
        d, n, _v = row_dsr(e, rows, through=STAGES[-1])
        full = _stream(Path(root) / "cells" / e["cell_hash"] / "daily_returns_full.csv")
        cs = _stream(Path(root) / "cells" / c["cell_hash"] / "daily_returns.csv").iloc[:-E_C_TAIL]
        diff = (full.reindex(cs.index) - cs).abs()
        recs.append({**base, "has_row": True, "status": "ok", "n_obs": e.get("n_obs"),
                     **{k: g[k] for k in ("sharpe", "psr", "beta", "alpha_sr", "top_days_share", "label", "cell_hash")},
                     "dsr": d, "n_trials_total": n, "c_max_diff": float(diff.max()) if diff.notna().all() else np.inf,
                     "n_pre_holdout_days": e.get("n_pre_holdout_days")})  # fmt: skip
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    ok = out["status"].eq("ok")
    out["p_holm"] = np.nan
    if ok.any():
        out.loc[ok, "p_holm"] = holm(1 - out.loc[ok, "psr"].astype(float).to_numpy())
    edge = ok & (out.get("dsr", np.nan) > E_DSR) & (out.get("alpha_sr", np.nan) > 0)
    out["verdict"] = np.where(edge, "edge", np.where(ok & (out["p_holm"] < E_ALPHA), "positive_not_significant",
                                                     np.where(ok, "no_edge", None)))  # fmt: skip
    return out.sort_values(["c_sharpe"], ascending=False).reset_index(drop=True)


if __name__ == "__main__":  # python -m experiments.report pilot <spec> [--root <root>]
    import argparse

    from experiments.runner import DEFAULT_ROOT

    ap = argparse.ArgumentParser(prog="experiments.report")
    ap.add_argument("cmd", choices=["pilot", "survivors", "b1", "b2", "b3", "c-spec", "c", "e-spec", "e"])
    ap.add_argument("spec", nargs="?", help="pilot: the spec holding the pilot cells")
    ap.add_argument("--ledger", default=str(L.DEFAULT_PATH))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--timeframe", default="5Min")
    a = ap.parse_args()
    out = Path(a.root) / "report"
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "survivors":
        table = stage_a_survivors(L.Ledger(a.ledger).rows())
        table.to_csv(out / "stage_a_survivors.csv", index=False)
        n_pass = int(table["passed"].sum()) if len(table) else 0
        rule = (
            f"meta AUC > {SURVIVOR_AUC}, PSR > {SURVIVOR_PSR}; top {SURVIVOR_K} by DSR, at most "
            f"{SURVIVOR_PER_PAIR} per (symbol, timeframe); xgb rows only"
        )
        md = ["# Stage A survivors", "", f"Rule: {rule}. {len(table)} cells, {n_pass} passed the gates.", "",
              _markdown(_fmt(table[table["survivor"]])) if n_pass else "_none_"]  # fmt: skip
        (out / "stage_a_survivors.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if a.cmd == "b1":
        table = stage_b1_selection(L.Ledger(a.ledger).rows(), a.root)
        table.to_csv(out / "stage_b1_selection.csv", index=False)
        n_pass, short = int(table["passed"].sum()), int(table["n_models"].lt(len(B1_MODELS)).sum())
        rule = (f"per Stage A survivor the model ({', '.join(B1_MODELS)}) with the highest OOS meta AUC; passes if "
                f"PSR > {B1_PSR}, alpha Sharpe > 0 and best-{B1_TOP_DAYS}-days share of P&L < 100 %")  # fmt: skip
        status = f"{len(table)} survivors, {n_pass} passed" + (
            f"; NOT FINAL: {short} lack a model row" if short else ""
        )
        md = ["# Stage B1 selection", "", f"Rule: {rule}. {status}.", "", _markdown(_fmt(table))]
        (out / "stage_b1_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if a.cmd == "b3":
        table = stage_c_finalists(stage_b3_selection(L.Ledger(a.ledger).rows(), a.root))
        table.to_csv(out / "stage_b3_selection.csv", index=False)
        n_pass, short = int(table["passed"].sum()), int(table["n_sizers"].lt(1 + len(B3_SIZERS)).sum())
        rule = (f"per B2 passer `fixed` unless a sizer ({', '.join(B3_SIZERS)}) passes the gates (PSR > {B1_PSR}, "
                f"alpha Sharpe > 0, best-{B1_TOP_DAYS}-days share < 100 %) with a higher PSR — then the "
                f"highest-PSR such sizer; Stage C finalists = the top {C_PER_SYMBOL} passers per symbol by PSR")  # fmt: skip
        status = f"{len(table)} B2 passers, {n_pass} passed, {int(table['finalist'].sum())} finalists" + (
            f"; NOT FINAL: {short} lack a sizer row" if short else ""
        )
        md = ["# Stage B3 selection", "", f"Rule: {rule}. {status}.", "", _markdown(_fmt(table))]
        (out / "stage_b3_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if a.cmd in ("e-spec", "e"):
        rows = L.Ledger(a.ledger).rows()
        c_rows = [r for (_h, st), r in L.done(rows).items()
                  if st == "C" and r.get("spec_name") == "u11_c" and r.get("status") == "ok"]  # fmt: skip
        if a.cmd == "e-spec":  # experiments/specs/u11_e.yaml (or the given path); slowest cells first
            cells = [stage_e_cell(json.loads(r["spec_json"])) for r in sorted(c_rows, key=lambda r: -r["runtime_s"])]
            path = Path(a.spec or "experiments/specs/u11_e.yaml")
            head = [
                "# U11 Stage E (PLAN U11, pre-registered 2026-10-07): the holdout, once. All 14 Stage C finalists (none",
                f"# passed C -> D; the user chose to evaluate all of them), each its Stage C cell run through {E_END}",
                "# (cached bars' end) with PWFO_PARTIAL_LAST; ledger statistics = holdout days only. Run with --final.",
                "# Generated by `python -m experiments.report e-spec`.",
                "name: u11_e", "stage: E", "cells:",
            ]  # fmt: skip
            path.write_text("\n".join(head + [f"  - {json.dumps(c, sort_keys=True)}" for c in cells]) + "\n",
                            encoding="utf-8", newline="\n")  # fmt: skip
            print(f"{len(cells)} cells → {path}")
            raise SystemExit(0)
        table = stage_e_selection(rows, c_rows, a.root)
        table.to_csv(out / "stage_e_verdicts.csv", index=False)
        short = int((~table["has_row"]).sum())
        rule = (f"edge iff holdout DSR (N = every counted trial) > {E_DSR} and alpha Sharpe > 0; else "
                f"positive_not_significant iff Holm-adjusted 1 − PSR(0) < {E_ALPHA} over the {len(table)} finalists; "
                "else no_edge")  # fmt: skip
        counts = table["verdict"].value_counts().to_dict() if "verdict" in table else {}
        status = f"{len(table)} finalists, verdicts {counts}" + (f"; NOT FINAL: {short} lack a row" if short else "")
        md = ["# Stage E (holdout) verdicts", "", f"Rule: {rule}. {status}.", "", _markdown(_fmt(table))]
        (out / "stage_e_verdicts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if a.cmd in ("c-spec", "c"):
        rows = L.Ledger(a.ledger).rows()
        fin = stage_c_finalists(stage_b3_selection(rows, a.root))
        if a.cmd == "c-spec":  # experiments/specs/u11_c.yaml (or the given path); slowest cells first
            by_hash = {h: r for (h, _st), r in L.done(rows).items()}
            cells = [stage_c_cell(json.loads(by_hash[h]["spec_json"])) for h in fin.loc[fin["finalist"], "cell_hash"]]
            cells.sort(key=lambda c: (c["model"]["meta"] != "catboost", c["sizer"] != "ecdf", c["timeframe"] == "1Day"))
            path = Path(a.spec or "experiments/specs/u11_c.yaml")
            head = [
                "# U11 Stage C (PLAN U11, pre-registered 2026-10-06): the 14 Stage C finalists (stage_b3_selection.csv,",
                "# finalist = True), each the B3 row's spec with a rolling PWFO grid: intraday IS {252, 378, 504, 756},",
                "# 1Day IS {1260, 1512}, x OOS {5, 10, 21, 63}; every combo a trial. Slowest (catboost, ecdf) first.",
                "# Generated by `python -m experiments.report c-spec`.",
                "name: u11_c", "stage: C", "cells:",
            ]  # fmt: skip
            path.write_text("\n".join(head + [f"  - {json.dumps(c, sort_keys=True)}" for c in cells]) + "\n",
                            encoding="utf-8", newline="\n")  # fmt: skip
            print(f"{len(cells)} cells → {path}")
            raise SystemExit(0)
        table = stage_c_selection(rows, fin, a.root)
        table.to_csv(out / "stage_c_selection.csv", index=False)
        n_pass, short = int(table["passed"].sum()), int((~table["has_row"]).sum())
        rule = (f"per finalist, its nested-PWFO live stream passes the B gates (PSR > {B1_PSR}, alpha Sharpe > 0, "
                f"best-{B1_TOP_DAYS}-days share < 100 %) and its PBO across combos < {C_PBO_MAX}")  # fmt: skip
        status = f"{len(table)} finalists, {n_pass} passed" + (f"; NOT FINAL: {short} lack a row" if short else "")
        md = ["# Stage C selection", "", f"Rule: {rule}. {status}.", "", _markdown(_fmt(table))]
        (out / "stage_c_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if a.cmd == "b2":
        table = stage_b2_selection(L.Ledger(a.ledger).rows(), a.root)
        table.to_csv(out / "stage_b2_selection.csv", index=False)
        n_pass, short = int(table["passed"].sum()), int(table["n_arms"].lt(len(B2_ARMS)).sum())
        rule = (f"per B1 passer the feature arm ({', '.join(B2_ARMS)}) with the highest OOS meta AUC; passes if "
                f"PSR > {B1_PSR}, alpha Sharpe > 0 and best-{B1_TOP_DAYS}-days share of P&L < 100 %")  # fmt: skip
        status = f"{len(table)} B1 passers, {n_pass} passed" + (
            f"; NOT FINAL: {short} lack an arm row" if short else ""
        )
        md = ["# Stage B2 selection", "", f"Rule: {rule}. {status}.", "", _markdown(_fmt(table))]
        (out / "stage_b2_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        raise SystemExit(0)
    if not a.spec:
        ap.error("pilot needs a spec")
    table, verdict = pilot_rule(a.spec, a.root, a.timeframe)
    stem = f"pilot_{Path(a.spec).stem}_{a.timeframe}"
    table.to_csv(out / f"{stem}.csv", index=False)
    gate = f"every one-sided 95 % session-bootstrap upper bound on OOS meta AUC < {PILOT_AUC_GATE}"
    md = [f"# {a.timeframe} pilot rule ({a.spec})", "", f"Verdict: **{verdict}** — gate: {gate}.", "",
          _markdown(_fmt(table))]  # fmt: skip
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n→ {out / stem}.csv / .md")
