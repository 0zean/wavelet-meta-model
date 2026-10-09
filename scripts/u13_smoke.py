"""
U13 live smoke (PLAN2 U13 done-criteria): fetch the exogenous series, check them, plot them, and compare the quotes
half-spreads with Corwin–Schultz.

    uv run python scripts/u13_smoke.py exo      # CBOE indexes vs FRED VIXCLS, VX1/VIX contango, FRED coverage
    uv run python scripts/u13_smoke.py quotes   # time-of-day profile for SPY QQQ IWM AAPL NVDA; SPY vs Corwin–Schultz

Outputs go to results/u13/ (PNG + JSON). Network: cdn-api.cboe.com, cboe.com, fred.stlouisfed.org (exo); none for
quotes (the checked-in table and the cached 5Min bars).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import exo

OUT = ROOT / "results" / "u13"
DEV = ("2016-01-01", "2025-10-01")


def smoke_exo() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, end = "2016-01-01", "2026-10-01"
    kw = {"allow_holdout": True}  # a caching smoke, like data.fetch: the checks below use the development window
    s = {n: exo.load_series("cboe", n, start, end, **kw) for n in (*exo.CBOE_INDEXES, *exo.VX_SERIES)}
    s |= {n: exo.load_series("fred", n, start, end, **kw) for n in exo.FRED_SERIES}
    vix, vixcls = s["VIX"]["value"], s["VIXCLS"]["value"]
    common = vix.index.intersection(vixcls.index)
    diff = (vix[common] - vixcls[common]).abs()
    dev = s["VX1_VIX"]["value"].loc[DEV[0] : DEV[1]]
    out = {
        "coverage": {n: [str(d.index[0].date()), str(d.index[-1].date()), len(d)] for n, d in s.items()},
        "vix_vs_vixcls": {
            "common_days": len(common),
            "max_abs_diff": float(diff.max()),
            "n_diff_ge_0.01": int((diff >= 0.01).sum()),
            "worst": {str(k.date()): float(v) for k, v in diff.nlargest(5).items()},
        },
        "vx1_vix_dev": {"median": float(dev.median()), "share_gt_1": float((dev > 1).mean()), "n": len(dev)},
        "vx2_vx1_dev_median": float(s["VX2_VX1"]["value"].loc[DEV[0] : DEV[1]].median()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for n in exo.CBOE_INDEXES:
        ax[0].plot(s[n]["value"], lw=0.7, label=n)
    ax[0].plot(vixcls, lw=0.7, ls="--", color="k", label="FRED VIXCLS")
    ax[0].set_ylabel("index")
    ax[0].legend(ncol=5, fontsize=8)
    ax[1].plot(diff, lw=0.7)
    ax[1].set_ylabel("|VIX − VIXCLS|")
    ax[2].plot(s["VX1_VIX"]["value"], lw=0.7, label="VX1 / VIX")
    ax[2].plot(s["VX2_VX1"]["value"], lw=0.7, label="VX2 / VX1")
    ax[2].axhline(1, color="k", lw=0.5)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exo_smoke.png", dpi=110)
    (OUT / "exo_smoke.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def smoke_quotes(symbols=("SPY", "QQQ", "IWM", "AAPL", "NVDA")) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.bars import load_bars
    from data.quotes import read_table
    from risk.costs import session_spread

    t = read_table()
    regular = t[t["bin"].str.match(r"^\d\d:\d\d$")]
    out: dict = {"profile_bp": {}, "spy_vs_cs": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, len(symbols), figsize=(4 * len(symbols), 3.5), sharey=False)
    for a, sym in zip(ax, symbols):
        r = regular[regular["symbol"] == sym]
        prof = r.groupby("bin")["half_spread_bp"].median()
        out["profile_bp"][sym] = {k: round(float(v), 4) for k, v in prof.items()}
        for y, g in r.groupby("year"):
            a.plot(g["bin"], g["half_spread_bp"], lw=0.7, label=str(y))
        a.set_title(f"{sym} median half-spread (bp)")
        a.set_xticks(range(0, 26, 6))
        a.tick_params(axis="x", labelsize=7)
    ax[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "quotes_profile.png", dpi=110)
    # SPY at 10:00–15:30 (bins 10:00 … 15:15) vs the U8 Corwin–Schultz half-spread from 5Min bars, per year
    spy = regular[(regular["symbol"] == "SPY") & regular["bin"].between("10:00", "15:15")]
    q = spy.groupby("year")["half_spread_bp"].median()
    bars = load_bars("SPY", "5Min", *DEV)
    cs = (session_spread(bars) / 2 * 1e4).groupby(lambda d: d.year).mean()
    out["spy_vs_cs"] = {int(y): {"quotes_bp": round(float(q.get(y, np.nan)), 4), "cs_bp": round(float(cs[y]), 4)}
                        for y in cs.index}  # fmt: skip
    out["spy_quotes_median_bp_dev"] = float(spy[spy["year"] <= 2025]["half_spread_bp"].median())
    out["spy_cs_mean_bp_dev"] = float(cs.mean())
    (OUT / "quotes_smoke.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["exo", "quotes"])
    a = p.parse_args()
    res = smoke_exo() if a.what == "exo" else smoke_quotes()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
