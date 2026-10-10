# Net Sharpe of the band (half-hourly) and SAR(N=6) rules under three round-trip cost assumptions, and a 4-leg equal-weight portfolio.
import importlib.util
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sp", sys.argv[1])
sp = importlib.util.module_from_spec(spec)
src = open(sys.argv[1]).read().split("res = {}")[0]
exec(src, sp.__dict__)


def rules_with_turnover(C, O):
    D = C.shape[0]
    lc = np.log(C)
    fwd = np.full((D, 78), np.nan)
    fwd[:, :76] = np.log(O[:, 2:]) - np.log(O[:, 1:77])
    fwd[:, 76] = np.log(C[:, 77]) - np.log(O[:, 77])
    mv = lc - np.log(O[:, :1])
    amv = np.abs(mv)
    band = np.full_like(amv, np.nan)
    for d in range(14, D):
        band[d] = amv[d - 14 : d].mean(0)
    r = np.diff(lc, axis=1)
    sd = np.full(D, np.nan)
    for d in range(5, D):
        sd[d] = r[d - 5 : d].std()
    pos = np.where(mv > band, 1.0, np.where(mv < -band, -1.0, 0.0))
    hh = np.zeros_like(pos)
    cur = np.zeros(D)
    for j in range(78):
        if (j + 1) % 6 == 0:
            cur = pos[:, j]
        hh[:, j] = cur
    N = 6
    k = np.arange(N) - (N - 1) / 2
    w = k / np.sum(k * k)
    s = np.full_like(lc, np.nan)
    for j in range(N - 1, 78):
        s[:, j] = lc[:, j - N + 1 : j + 1] @ w
    v = s / sd[:, None]
    p2 = np.where(v > 0.75, 1.0, np.where(v < -0.75, -1.0, 0.0))
    sar = np.zeros_like(p2)
    cur = np.zeros(D)
    for j in range(78):
        cur = np.where(p2[:, j] != 0, p2[:, j], cur)
        sar[:, j] = cur
    out = {}
    for name, p in [("band_hh", hh), ("sar6", sar)]:
        p = np.nan_to_num(p.copy())
        p[:, :6] = 0
        out[name] = np.nansum(p * fwd, axis=1)
        out[name + "_rt"] = np.abs(np.diff(np.c_[np.zeros(D), p, np.zeros(D)], axis=1)).sum(1) / 2
    out["ok"] = ~np.isnan(band[:, 0]) & ~np.isnan(sd)
    return pd.DataFrame(out)


legs = {}
for sym in ["SPY", "QQQ"]:
    C, O, dates = sp.load(sym)
    t = rules_with_turnover(C, O)
    t.index = dates
    t = t[t["ok"]]
    for name in ["band_hh", "sar6"]:
        line = f"{sym} {name:8s}"
        for rt_cost in [0.3e-4, 1.0e-4, 2.3e-4]:
            net = t[name] - t[name + "_rt"] * rt_cost
            line += f" | cost {rt_cost * 1e4:.1f}bp: net {net.mean() * 1e4:+.2f} bp/d Sharpe {net.mean() / net.std() * np.sqrt(252):+.2f}"
            legs[(sym, name, rt_cost)] = net
        print(line)
for rt_cost in [0.3e-4, 1.0e-4, 2.3e-4]:
    port = (
        pd.concat([legs[(s, n, rt_cost)] for s in ["SPY", "QQQ"] for n in ["band_hh", "sar6"]], axis=1).dropna().mean(1)
    )
    print(
        f"4-leg equal-weight portfolio, cost {rt_cost * 1e4:.1f} bp/rt: net {port.mean() * 1e4:+.2f} bp/d, Sharpe {port.mean() / port.std() * np.sqrt(252):+.2f}, "
        f"ann ret {port.mean() * 252 * 100:+.1f}% at {port.std() * np.sqrt(252) * 100:.1f}% vol, worst day {port.min() * 1e4:.0f} bp, "
        f"max DD {((port.cumsum() - port.cumsum().cummax()).min()) * 100:.1f}% (arith)"
    )
