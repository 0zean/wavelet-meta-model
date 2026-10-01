"""
PWFO runner (U9): every (IS, OOS) combo of one symbol's walk-forward in parallel, nested selection, PBO and DSR.

    uv run python -m wfo.pwfo_run --symbol SPY --timeframe 1Hour --start 2016-01-01 --end 2025-10-01 \\
        --primary wavelet_trend --meta-model logit_l2 --meta-train oof --out results/pwfo

Windows count exchange-calendar sessions (data/cache/calendar.json). Writes to `<out>/`: `summary_<cell>.csv` (one
row per combo), `windows_<cell>.csv`, `returns_<cell>.csv` (daily OOS returns per combo), `choice_<cell>.csv`
(nested-selection decisions), `pwfo_<cell>.csv` (stitched PWFO daily returns), `stats_<cell>.json`,
`heatmaps_<cell>.png` and per-combo logs under `logs/`; appends one counted trial per run combo plus one for the
nested PWFO to `<out>/trials.jsonl` (stage "U9").
"""

import os

# One BLAS / OpenMP thread per process (see risk/run.py: parallel workers otherwise oversubscribe the cores)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.bars import get_calendar, load_bars
from features import cache as feature_cache
from models.compare import _git_sha
from utils.config import RunConfig
from wfo.pwfo import PWFOResult, run_pwfo

NEG, MID, POS = ("#a8302f", "#e34948"), "#f0efec", ("#2a78d6", "#184f95")  # diverging red ↔ gray ↔ blue
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def plot_heatmaps(summary: pd.DataFrame, cfg: RunConfig, title: str, path: Path) -> None:
    """IS × OOS heat-maps of OOS Sharpe and WFE (diverging at 0); empty = no window could be fit, † = < 50 windows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [NEG[0], NEG[1], MID, POS[0], POS[1]])
    cmap.set_bad(SURFACE)
    is_g, oos_g = sorted(summary["is_len"].unique()), sorted(summary["oos_len"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (col, name) in zip(axes, (("oos_sharpe", "OOS Sharpe (ann.)"), ("wfe", "WFE (OOS / IS ann. return)"))):
        grid = summary.pivot_table(index="is_len", columns="oos_len", values=col, aggfunc="first", dropna=False)
        grid = grid.reindex(index=is_g, columns=oos_g).astype(float)
        weak = summary.pivot_table(index="is_len", columns="oos_len", values="weak", aggfunc="first")
        ok = summary.pivot_table(index="is_len", columns="oos_len", values="n_ok_windows", aggfunc="first")
        v = grid.to_numpy()
        lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
        lim = min(lim, 2.0) if col == "wfe" else lim  # colour saturates at |WFE| = 2; the text keeps the value
        ax.set_facecolor(SURFACE)
        ax.imshow(np.ma.masked_invalid(v), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i, is_len in enumerate(is_g):
            for j, oos_len in enumerate(oos_g):
                x = v[i, j]
                if ok.loc[is_len, oos_len] == 0:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", ec="#c3c2b7", lw=0))
                    ax.text(j, i, "no fit", ha="center", va="center", color=MUTED, fontsize=9)
                    continue
                s = "n/a" if np.isnan(x) else f"{x:.2f}"
                s += "†" if weak.loc[is_len, oos_len] else ""
                dark = not np.isnan(x) and abs(x) > 0.55 * lim
                ax.text(j, i, s, ha="center", va="center", color="white" if dark else INK, fontsize=10)
        ax.set_xticks(range(len(oos_g)), [str(o) for o in oos_g], color=MUTED)
        ax.set_yticks(range(len(is_g)), [str(i) for i in is_g], color=MUTED)
        ax.set_xlabel("OOS window = retrain cadence (sessions)", color=MUTED)
        ax.set_ylabel("IS window (sessions)", color=MUTED)
        ax.set_title(name, color=INK, loc="left", fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.text(
        0.01,
        0.01,
        f"† fewer than {cfg.PWFO_MIN_WINDOWS} fitted OOS windows   "
        f"n/a: WFE undefined (mean IS return not > 0 at t ≥ {cfg.PWFO_WFE_MIN_T:g})   "
        "no fit: too few events in every window",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def write_outputs(res: PWFOResult, cfg: RunConfig, out: Path, cell: str, title: str) -> None:
    res.summary.to_csv(out / f"summary_{cell}.csv")
    res.windows.to_csv(out / f"windows_{cell}.csv", index=False)
    res.returns.to_csv(out / f"returns_{cell}.csv")
    res.choice.to_csv(out / f"choice_{cell}.csv", index=False)
    res.pwfo.to_csv(out / f"pwfo_{cell}.csv")
    (out / f"stats_{cell}.json").write_text(json.dumps(res.stats, indent=2, default=float))
    plot_heatmaps(res.summary, cfg, title, out / f"heatmaps_{cell}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--primary", default="wavelet_trend")
    ap.add_argument("--meta-model", default="logit_l2")
    ap.add_argument("--meta-train", choices=["val", "oof"], default="oof")
    ap.add_argument("--sizer", default="fixed")
    ap.add_argument("--is-grid", default=None, help="comma-separated IS sessions (default cfg.PWFO_IS_GRID)")
    ap.add_argument("--oos-grid", default=None, help="comma-separated OOS sessions (default cfg.PWFO_OOS_GRID)")
    ap.add_argument("--expanding", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/pwfo")
    a = ap.parse_args()

    kw = {"PRIMARY": a.primary, "META_MODEL": a.meta_model, "META_TRAIN": a.meta_train, "SIZER": a.sizer}
    if a.is_grid:
        kw["PWFO_IS_GRID"] = tuple(int(x) for x in a.is_grid.split(","))
    if a.oos_grid:
        kw["PWFO_OOS_GRID"] = tuple(int(x) for x in a.oos_grid.split(","))
    cfg = RunConfig.for_timeframe(a.timeframe, PWFO_EXPANDING=a.expanding, **kw)
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    mode = "expanding" if a.expanding else "rolling"
    cell = f"{a.symbol}_{a.timeframe}_{a.start}_{a.end}_{a.primary}_{a.meta_model}_{a.meta_train}_{a.sizer}_{mode}"

    df = load_bars(a.symbol, a.timeframe, a.start, a.end)
    sessions = get_calendar(a.end).index
    t0 = time.time()
    res = run_pwfo(
        df,
        cfg,
        sessions=sessions,
        jobs=a.jobs,
        log_dir=out / "logs",
        symbol=a.symbol,
        feature_cache_dir=feature_cache.DEFAULT_ROOT,
    )
    runtime = time.time() - t0
    title = (
        f"PWFO {a.symbol} {a.timeframe} {a.start} → {a.end}  ({a.primary} / {a.meta_model} / {a.meta_train}, {mode})"
    )
    write_outputs(res, cfg, out, cell, title)

    spec = {
        "symbols": [a.symbol],
        "timeframe": a.timeframe,
        "start": a.start,
        "end": a.end,
        "primary": a.primary,
        "meta_model": a.meta_model,
        "meta_train": a.meta_train,
        "sizer": a.sizer,
        "risk_profile": cfg.RISK_PROFILE,
        "seed": cfg.SEED,
        "pwfo": {"is_grid": cfg.PWFO_IS_GRID, "oos_grid": cfg.PWFO_OOS_GRID, "expanding": a.expanding},
    }
    started, sha = datetime.now(UTC).isoformat(), _git_sha()
    run_id = f"{cell}@{started}"  # every row of one invocation; a rerun appends a new run_id (no silent dedup)

    def num(x):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)

    with open(out / "trials.jsonl", "a") as f:
        # Every grid combo is a trial (the DSR counts them all): a combo with no fittable window is a "no_fit" row
        for label, s in res.summary.iterrows():
            if s["n_ok_windows"] == 0:
                row = {"stage": "U9", "status": "no_fit", "git_sha": sha, "started_at": started, "run_id": run_id,
                       "combo": label, "n_oos_windows": int(s["n_oos_windows"]), "n_trials_dsr": res.stats["n_combos"]}  # fmt: skip
                f.write(json.dumps({**row, "spec": {**spec, "combo": label}}) + "\n")
                continue
            row = {
                "stage": "U9",
                "status": "ok",
                "run_id": run_id,
                "n_trials_dsr": res.stats["n_combos"],
                "git_sha": sha,
                "started_at": started,
                "combo": label,
                "n_trades": int(s["n_trades"]),
                "ret_ann": num(s["oos_ret_ann"]),
                "sharpe": num(s["oos_sharpe"]),
                "sortino": num(s["oos_sortino"]),
                "max_dd": num(s["max_dd"]),
                "turnover": num(s["turnover"]),
                "psr": num(s["psr0"]),
                "wfe": num(s["wfe"]),
                "n_oos_windows": int(s["n_oos_windows"]),
                "n_obs": int(s["n_oos_days"]),
            }
            f.write(json.dumps({**row, "spec": {**spec, "combo": label}}) + "\n")
        st = res.stats
        row = {
            "stage": "U9",
            "status": "ok",
            "git_sha": sha,
            "started_at": started,
            "combo": "nested",
            "run_id": run_id,
            "n_trials_dsr": st["n_combos"],
            "ret_ann": num(st.get("pwfo_ret_ann")),
            "sharpe": num(st.get("pwfo_sharpe")),
            "max_dd": num(st.get("pwfo_max_dd")),
            "psr": num(st.get("pwfo_psr0")),
            "dsr": num(st.get("pwfo_dsr")),
            "pbo": num(st.get("pbo")),
            "n_obs": st["n_live_days"],
            "runtime_s": round(runtime, 1),
        }
        f.write(json.dumps({**row, "spec": {**spec, "combo": "nested"}}) + "\n")

    cols = ["n_oos_windows", "n_ok_windows", "oos_sharpe", "oos_ret_ann", "max_dd", "wfe", "wfe_sharpe",
            "pct_profitable_oos", "is_oos_spearman", "n_trades"]  # fmt: skip
    print(res.summary.reindex(columns=cols).round(3).to_string())
    print(json.dumps(res.stats, indent=2, default=float))
    print(f"[PWFO]  Saved to {out} ({runtime:.0f}s)")


if __name__ == "__main__":
    main()
