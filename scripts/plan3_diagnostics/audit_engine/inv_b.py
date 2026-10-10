import contextlib
import io

from common import *

from features.events import sample_events
from utils.config import RunConfig

df = bars("SPY", "5Min")
cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", RISK_PROFILE="none")
with contextlib.redirect_stdout(io.StringIO()):
    ev = sample_events(df, cfg)
rng = np.random.default_rng(12345)
sig = pd.DataFrame({"width": ev["width"]})
sig["trade_signal"] = rng.choice([-1, 1], len(sig))
sig["bet_size"] = 1.0
sig = sig.dropna()
tbl = read_table()
tbl = tbl[tbl.symbol == "SPY"]
fc = fill_costs(df.index, cfg, tbl)
with contextlib.redirect_stdout(io.StringIO()):
    eq, tr, _ = simulate_portfolio({"SPY": df}, {"SPY": sig}, cfg, get_profile("none"), costs={"SPY": fc})
g = tr["side"] * (tr["exit_px"] / tr["entry_px"] - 1)
net = tr["pnl_pct"]
print("cusum events", len(ev), "trades", len(tr))
print(
    "gross mean bp %.4f  se %.4f  t %.2f"
    % (g.mean() * 1e4, g.std() / np.sqrt(len(g)) * 1e4, g.mean() / (g.std() / np.sqrt(len(g))))
)
print("net mean bp %.4f" % (net.mean() * 1e4))
print("mean round-trip cost bp (entry+exit) %.4f ; gross-net %.4f" % (tr.cost_bp.mean(), (g - net).mean() * 1e4))
hs = (tr.cost_bp - 2 * cfg.SLIPPAGE_PCT * 1e4) / 2
print("mean one-way half-spread bp %.4f; slippage 1 bp per side" % hs.mean())
print(
    "Sum check: equity change",
    eq.iloc[-1] - cfg.INIT_CASH,
    " sum pnl",
    tr.pnl.sum(),
    " cost_paid",
    eq.attrs["cost_paid"],
    " notional",
    eq.attrs["traded_notional"],
)
print("cost_paid/notional bp", eq.attrs["cost_paid"] / eq.attrs["traded_notional"] * 1e4)
print(tr.exit_reason.value_counts())
# by exit reason gross
print(g.groupby(tr.exit_reason).agg(["mean", "count"]) * [1e4, 1])
# separate long vs short
print("gross by side bp", (g.groupby(tr.side).mean() * 1e4).to_dict())
