import contextlib
import io

from common import *

q = io.StringIO()
df = bars("SPY", "5Min")
base = dict(META_MODEL="none", COST_MODEL="slippage", SLIPPAGE_PCT=0.0, RISK_PROFILE="none")
on = primary_config("overnight", "5Min", {}, **base)
idc = primary_config(
    "overnight", "5Min", {}, **base, EVENT_PARAMS={"entry_times": ["09:30"]}, EXIT_PARAMS={"exit_time": "close"}
)
with contextlib.redirect_stdout(q):
    s_on, e_on, t_on = run_rule(df, on)
    s_id, e_id, t_id = run_rule(df, idc)
r_on, r_id = daily_returns(e_on), daily_returns(e_id)
day = df.index.normalize()
c = df.close.groupby(day).last()
o = df.open.groupby(day).first()
cc = c.pct_change()
j = pd.concat([r_on.rename("on"), r_id.rename("id"), cc.rename("cc")], axis=1, sort=True).dropna()
j = j.iloc[2:-1]
prod = (1 + j.on) * (1 + j.id) - 1
print("days", len(j), " max |(1+on)(1+id)-1 - cc| =", (prod - j.cc).abs().max())
print(" max |on+id - cc| =", (j.on + j.id - j.cc).abs().max(), " mean =", (j.on + j.id - j.cc).mean())
# per-trade gross check
g_on = t_on.exit_px / t_on.entry_px - 1
print("overnight trades", len(t_on), "intraday trades", len(t_id))
print("ann. mean overnight %.4f intraday %.4f cc %.4f" % (j.on.mean() * 252, j.id.mean() * 252, j.cc.mean() * 252))
# which days are missing from either stream
print(
    "cc days",
    cc.dropna().shape[0],
    "on days with trade P&L != 0",
    int((r_on != 0).sum()),
    "id days !=0",
    int((r_id != 0).sum()),
)
d = prod - j.cc
bad = d[d.abs() > 1e-9]
print("days with deviation:", len(bad))
for k, v in bad.items():
    sub = df[df.index.normalize() == k]
    print(
        k.date(),
        "%.5f" % v,
        "bars",
        len(sub),
        "first",
        sub.index[0].strftime("%H:%M"),
        "last",
        sub.index[-1].strftime("%H:%M"),
        "on %.5f id %.5f cc %.5f" % (j.on[k], j.id[k], j.cc[k]),
    )
