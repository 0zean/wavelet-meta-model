"""
Ledger report (SPEC §9, U10): stage funnel, leaderboard with DSR, PBO per stage → Markdown + HTML + CSV.

DSR of a row = PSR(SR₀) of its daily returns with N = n_trials through the row's stage (experiments.ledger) and
V[SR_n] = variance of the per-period Sharpes of those trials; `dsr_all` uses every trial in the ledger (the deflation
a pick made from the whole table faces). Rows without daily return moments (legacy imports, no
trades) get no DSR. PBO per stage = CSCV (validation/pbo.py, PBO_BLOCKS) over the daily returns of the stage's runner
cells (`<root>/cells/<hash>/daily_returns.csv`) on the days all of them cover, identical streams counted once.
"""

import html
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import ledger as L
from experiments.spec import STAGES, stage_rank
from utils.config import RunConfig
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
