from common import *

for f in ["F3", "F5"]:
    h = [r for r in ROWS if r.get("label") == f"{f}:headline"][-1]
    w = h["weights"]
    tot = dict(pnl=0, cost_paid=0, traded_notional=0)
    for m in h["members"]:
        r = BYH[m]
        sym = json.loads(r["spec_json"])["symbols"][0]
        n = r["traded_notional"]
        c = r["cost_paid"]
        p = r["pnl"]
        print(
            f,
            sym,
            f"turnover/yr per $ {n / r['init_cash'] / 9.7:.0f}x  cost {c / n * 1e4:.3f}bp  gross edge {(p + c) / n * 1e4:.3f}bp  ratio {(p + c) / c:.2f}  net edge {p / n * 1e4:.3f}bp",
        )
        for k in tot:
            tot[k] += w[sym] * r[k] / r["init_cash"]
    e = (tot["pnl"] + tot["cost_paid"]) / tot["traded_notional"]
    c = tot["cost_paid"] / tot["traded_notional"]
    print(
        f,
        "pooled: gross",
        round(e * 1e4, 4),
        "bp cost",
        round(c * 1e4, 4),
        "bp ratio",
        round(e / c, 3),
        "| ledger row:",
        round((h["pnl"] + h["cost_paid"]) / h["cost_paid"], 3),
    )
    # per-trade cost from trades.csv
    for m in h["members"]:
        t = pd.read_csv(REPO / f"results/experiments/cells/{m}/trades.csv")
        sym = json.loads(BYH[m]["spec_json"])["symbols"][0]
        cols = [c for c in t.columns if "cost" in c]
        print("   ", sym, "trades", len(t), {c: round(t[c].median(), 3) for c in cols})
