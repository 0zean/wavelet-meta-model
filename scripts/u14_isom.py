"""
U14 diagnostic (PLAN2 U14 done-criteria, SPEC §14): SPY 5Min over the development window.

    uv run python scripts/u14_isom.py

- the time-of-day profile s(b) fit on the whole window, and per year (stability);
- the median |r| per 30-minute slot, and the open slot vs the lunch slot;
- ISOM (events per 30-minute slot) of CUSUM (CUSUM_MULT 1.5) and DC (dc_mult 2) with and without VOL_PROFILE="tod",
  and each sampler's share of events in the first hour. The profile here is fit on the whole window (a diagnostic,
  not a fold); in a run it is fit per fold on train bars.

Outputs: results/u14/isom_spy_5min.json and .png. Network: none (cached bars).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.bars import load_bars
from features.events import sample_events
from features.vol_profile import VolProfile, bar_slots, isom_counts, ny_dates
from utils.config import RunConfig

OUT = ROOT / "results" / "u14"
START, END = "2016-01-04", "2025-10-01"
SLOT30 = [f"{(570 + 30 * k) // 60:02d}:{(570 + 30 * k) % 60:02d}" for k in range(13)]


def by30(counts78: np.ndarray) -> list[int]:
    return [int(counts78[6 * k : 6 * k + 6].sum()) for k in range(13)]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_bars("SPY", "5Min", START, END)
    n_sess = len(np.unique(ny_dates(df.index)))
    prof = VolProfile.fit(df, 5)
    r = np.abs(np.log(df["close"]).diff())
    slot = bar_slots(df.index, 5)
    med30 = pd.Series(r.to_numpy()).groupby(slot // 6).median()
    years = sorted(set(df.index.year))
    per_year = {str(y): VolProfile.fit(df[df.index.year == y], 5).s for y in years}

    out = {
        "window": [START, END],
        "n_bars": len(df),
        "n_sessions": n_sess,
        "profile_s": [round(float(v), 4) for v in prof.s],
        "profile_s_open_over_lunch": round(float(prof.s[0] / prof.s[30:36].mean()), 3),
        "median_abs_ret_30min_bp": {SLOT30[k]: round(float(v) * 1e4, 3) for k, v in med30.items()},
        "open_slot_over_lunch_slot": {
            "09:30 vs 12:30 (incl. the overnight gap bar)": round(float(med30[0] / med30[6]), 3),
            "09:30 vs 12:30 (09:35-09:55 only)": round(
                float(pd.Series(r.to_numpy())[(slot >= 1) & (slot < 6)].median() / med30[6]), 3
            ),
        },
        "per_year_s0_over_lunch": {y: round(float(s[0] / s[30:36].mean()), 3) for y, s in per_year.items()},
        "samplers": {},
    }
    for sampler, params in (("cusum", {}), ("dc", {"dc_mult": 2.0})):
        cfg = RunConfig.for_timeframe("5Min", EVENT_SAMPLER=sampler, EVENT_PARAMS=params)
        for name, p in (("none", None), ("tod", prof)):
            ev = sample_events(df, cfg, profile=p).index
            c = isom_counts(ev, 5)
            out["samplers"][f"{sampler}/{name}"] = {
                "n_events": len(ev),
                "events_per_session": round(len(ev) / n_sess, 2),
                "first_hour_share": round(float(c[:12].sum() / c.sum()), 4),
                "isom_30min": dict(zip(SLOT30, by30(c), strict=True)),
                "iaom_30min": {k: round(v / n_sess, 4) for k, v in zip(SLOT30, by30(c), strict=True)},
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "isom_spy_5min.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(prof.s, lw=2, color="k", label="2016–2025")
    for y, s in per_year.items():
        ax[0].plot(s, lw=0.6, alpha=0.5, label=y)
    ax[0].set_title("SPY 5Min time-of-day profile s(b)")
    ax[0].set_xlabel("slot (5-minute bar of the session)")
    ax[0].legend(ncol=6, fontsize=7)
    x = np.arange(13)
    for i, key in enumerate(out["samplers"]):
        v = list(out["samplers"][key]["iaom_30min"].values())
        ax[1].bar(x + (i - 1.5) * 0.2, v, width=0.2, label=key)
    ax[1].set_xticks(x, SLOT30, rotation=45)
    ax[1].set_title("IAOM: events per session by 30-minute slot")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "isom_spy_5min.png", dpi=120)

    print(json.dumps({k: v for k, v in out.items() if k != "profile_s"}, indent=2))


if __name__ == "__main__":
    main()
