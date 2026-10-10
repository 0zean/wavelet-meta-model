import contextlib
import io

from common import *

df = bars("SPY", "1Day")
cfg = primary_config(
    "vol_target",
    "1Day",
    {"sigma_target": 1e6},
    META_MODEL="none",
    COST_MODEL="slippage",
    SLIPPAGE_PCT=0.0,
    RISK_PROFILE="none",
)
with contextlib.redirect_stdout(io.StringIO()):
    sig, eq, tr = run_rule(df, cfg)
print("events", len(sig), "sided", int((sig.signed_dir != 0).sum()), "bet sizes", sig.bet_size.unique()[:5])
print("trades", len(tr), "rolled", int(tr.rolled.sum()), "non-rolled", int((~tr.rolled).sum()))
dr = daily_returns(eq)
bh = df["close"].pct_change()
j = pd.concat([dr.rename("s"), bh.rename("b")], axis=1).dropna()
d = j.s - j.b
print("days", len(j), "first day", j.index[0].date())
print(
    "first-day strat vs bh:",
    j.iloc[0].to_dict(),
    " open->close:",
    df.close.iloc[df.index.get_loc(j.index[0])] / df.open.iloc[df.index.get_loc(j.index[0])] - 1,
)
print("max |diff| excluding first day:", d.iloc[1:].abs().max(), "argmax", d.iloc[1:].abs().idxmax())
print(
    "final equity/initial",
    eq.iloc[-1] / cfg.INIT_CASH,
    " bh from entry open:",
    df.close.iloc[-1] / df.open.loc[j.index[0]],
)
# worst days
print(d.abs().sort_values().tail(5))
# does the position ever exit before the end? exits not rolled
print(tr[~tr.rolled][["entry_b", "exit_b", "exit_reason"]].tail())
print("last day in eq", eq.index[-1], "last bar df", df.index[-1])
