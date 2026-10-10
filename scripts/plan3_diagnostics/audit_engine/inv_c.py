import contextlib
import io

from common import *

from features.events import sample_events
from utils.config import RunConfig

q = io.StringIO()
df = bars("SPY", "5Min")
tbl = read_table()
tbl = tbl[tbl.symbol == "SPY"]


def gross(tr):
    return tr["side"] * (tr["exit_px"] / tr["entry_px"] - 1)


# (1) triple barrier, random sides
cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", RISK_PROFILE="none")
with contextlib.redirect_stdout(q):
    ev = sample_events(df, cfg)
rng = np.random.default_rng(1)
sig = pd.DataFrame({"width": ev["width"]}).dropna()
sig["trade_signal"] = rng.choice([-1, 1], len(sig))
sig["bet_size"] = 1.0
fc = fill_costs(df.index, cfg, tbl)
res = {}
for k, s in (("orig", sig), ("flip", sig.assign(trade_signal=-sig.trade_signal))):
    with contextlib.redirect_stdout(q):
        eq, tr, _ = simulate_portfolio({"SPY": df}, {"SPY": s}, cfg, get_profile("none"), costs={"SPY": fc})
    res[k] = tr
a, b = res["orig"], res["flip"]
print("TB: trades", len(a), len(b), "same events", a.index.equals(b.index))
ga, gb = gross(a), gross(b)
print("TB: sum gross orig %.6f flip %.6f sum %.6f" % (ga.sum(), gb.sum(), ga.sum() + gb.sum()))
diff = ga + gb
nz = diff.abs() > 1e-12
print(
    "TB: trades where flip is not exact negation:",
    int(nz.sum()),
    " their reasons orig:",
    a.exit_reason[nz].value_counts().to_dict(),
    "flip:",
    b.exit_reason[nz].value_counts().to_dict(),
)
print("TB: mean asymmetry on those (bp): %.3f" % (diff[nz].mean() * 1e4))
# (2) intraday_momentum with time exits (F5 headline-like) on SPY 5Min, plain sigma
cfg2 = primary_config("intraday_momentum", "5Min", {}, META_MODEL="none", COST_MODEL="quotes", RISK_PROFILE="none")
with contextlib.redirect_stdout(q):
    s2 = rule_signals(df, cfg2, symbol="SPY")
fc2 = fill_costs(df.index, cfg2, tbl)
out = {}
for k, s in (("orig", s2), ("flip", s2.assign(trade_signal=-s2.trade_signal))):
    with contextlib.redirect_stdout(q):
        eq, tr, _ = simulate_portfolio({"SPY": df}, {"SPY": s}, cfg2, get_profile("none"), costs={"SPY": fc2})
    out[k] = (eq, tr)
ga, gb = gross(out["orig"][1]), gross(out["flip"][1])
print("TIME: trades", len(ga), len(gb), " max|g_orig + g_flip| =", (ga + gb).abs().max())
print("TIME: gross sum orig %.6f flip %.6f" % (ga.sum(), gb.sum()))
na, nb = out["orig"][1].pnl.sum(), out["flip"][1].pnl.sum()
ca, cb = out["orig"][0].attrs["cost_paid"], out["flip"][0].attrs["cost_paid"]
print("TIME: net cash orig %.2f flip %.2f ; cost orig %.2f flip %.2f" % (na, nb, ca, cb))
print("TIME: gross cash orig %.2f flip %.2f" % (na + ca, nb + cb))
print("TIME mean gross bp per trade %.3f, mean rt cost bp %.3f" % (ga.mean() * 1e4, out["orig"][1].cost_bp.mean()))
