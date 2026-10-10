import contextlib
import io

from common import *

q = io.StringIO()
df = bars("SPY", "1Day")
cfg = primary_config(
    "vol_target", "1Day", {"sigma_target": 1e6}, META_MODEL="none", COST_MODEL="quotes", RISK_PROFILE="none"
)
with contextlib.redirect_stdout(q):
    sig = rule_signals(df, cfg, symbol="SPY")
rng = np.random.default_rng(7)
sig = sig.copy()
sig["trade_signal"] = rng.choice([-1, 0, 1, 1], len(sig)).astype(float)
sig["bet_size"] = np.where(sig.trade_signal != 0, rng.choice([0.3, 0.5, 0.5, 1.0], len(sig)), 0.0)
sig.attrs["live_start"] = sig.index[0]
for prof in ("none", "basket"):
    c = cfg.replace(RISK_PROFILE=prof)
    with contextlib.redirect_stdout(q):
        s2, eq, tr = run_rule(df, c, signals=sig)
    b = df.loc[sig.index[0] :]
    fc = fill_costs(b.index, c, read_table().query("symbol=='SPY'"))
    # independent path: signed shares held after each open
    pos = np.zeros(len(b))
    for _, t in tr.iterrows():
        pos[int(t.entry_b) : int(t.exit_b)] += (
            t.side * t.qty
        )  # held from entry open to exit open (exit_b exclusive: time exits fill at the open)
    prev = np.r_[0.0, pos[:-1]]
    d = np.abs(pos - prev)
    exp_cost = (d * b.open.to_numpy() * fc["open"].to_numpy()).sum()
    exp_notional = (d * b.open.to_numpy()).sum()
    print(
        prof, "trades", len(tr), "rolled", int(tr.rolled.sum()), "flips", int(((tr.side.shift() * tr.side) < 0).sum())
    )
    print(
        "   cost_paid %.6f  independent %.6f  | notional %.2f independent %.2f"
        % (eq.attrs["cost_paid"], exp_cost, eq.attrs["traded_notional"], exp_notional)
    )
    # P&L identity: equity change = sum pnl; and = sum over bars of pos*(Δprice) - cost?
    print("   eq change %.6f sum pnl %.6f" % (eq.iloc[-1] - c.INIT_CASH, tr.pnl.sum()))
    if prof == "basket":
        print("   borrow on shorts included in pnl; short trades", int((tr.side < 0).sum()))
