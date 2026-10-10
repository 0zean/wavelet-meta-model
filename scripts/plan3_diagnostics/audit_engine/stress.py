import json

from common import *

from data.exo import load_series

vix = load_series("cboe", "VIX", "2016-01-01", "2025-10-01")["value"]
vix.index = pd.DatetimeIndex(vix.index).normalize()
SLIP = 1.0
for fam in ["F3", "F5", "F7", "F8"]:
    r = json.load(open(f"results/families/{fam}/result.json"))
    w = r["weights"]
    mem = r["members"]["headline"]
    ev = r["evaluation"]["variants"]["headline"]
    tot = {"cost": 0, "extra21": 0, "extra4": 0, "n": 0, "nstress": 0, "cost_stress": 0}
    for h in mem:
        cell = (
            json.load(open(f"results/experiments/cells/{h}/spec.json"))
            if os.path.exists(f"results/experiments/cells/{h}/spec.json")
            else None
        )
        tr = pd.read_csv(f"results/experiments/cells/{h}/trades.csv")
        sym = tr["sym"].iloc[0] if "sym" in tr and tr["sym"].iloc[0] != "_" else (cell["symbols"][0] if cell else None)
        if isinstance(cell, dict) and "symbols" in cell:
            sym = cell["symbols"][0] if isinstance(cell["symbols"], list) else cell["symbols"]
        wt = w.get(sym, 1.0)
        et = (
            pd.to_datetime(tr["entry_time"] if "entry_time" in tr else tr["event"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        xt = (
            pd.to_datetime(tr["exit_time"] if "exit_time" in tr else tr["event"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        notional_in = tr["qty"] * tr["entry_px"]
        notional_out = tr["qty"] * tr["exit_px"]

        # auction fills pay no slippage: half-spread = cost_bp - SLIP when cost_bp > SLIP + floor? approximate: hs = cost_bp - SLIP if cost_bp>=SLIP+0.25-1e-9 else cost_bp
        def hs(c):
            return np.where(c >= SLIP + 0.25 - 1e-9, c - SLIP, c)

        ci = notional_in * tr["entry_cost_bp"] / 1e4
        co = notional_out * tr["exit_cost_bp"] / 1e4
        hi = notional_in * hs(tr["entry_cost_bp"]) / 1e4
        ho = notional_out * hs(tr["exit_cost_bp"]) / 1e4
        si = vix.reindex(et).to_numpy() >= 30
        so = vix.reindex(xt).to_numpy() >= 30
        init = 10000.0
        tot["cost"] += wt * (ci.sum() + co.sum()) / init
        tot["cost_stress"] += wt * (ci[si].sum() + co[so].sum()) / init
        tot["extra21"] += wt * 1.1 * (hi[si].sum() + ho[so].sum()) / init
        tot["extra4"] += wt * 3.0 * (hi[si].sum() + ho[so].sum()) / init
        tot["n"] += len(tr)
        tot["nstress"] += int(si.sum())
    yrs = ev["stats"]["n_obs"] / 252
    c = ev["costs"]
    print(
        f"{fam}: trades {tot['n']} (on VIX>=30 entry days {tot['nstress']}); pooled cost_paid/init {c['cost_paid']:.4f} (mine {tot['cost']:.4f}); "
        f"stress share of cost {tot['cost_stress'] / tot['cost']:.1%}; extra cost if stress hs x2.1: {tot['extra21'] / yrs * 1e4:.1f} bp/yr, x4: {tot['extra4'] / yrs * 1e4:.1f} bp/yr; "
        f"net ret_ann {ev['stats']['ret_ann']:.4f}; edge floor {ev['floors']['min_edge_to_cost']['value']:.2f}"
    )
