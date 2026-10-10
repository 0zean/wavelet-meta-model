# Position-based diagnostic: daily gross P&L (bp of notional) of simple always-evaluated intraday rules, by year, with turnover.
# Rules decide at bar j's close and hold from bar j+1 open to bar j+2 open (re-evaluated every bar); flat at the 15:55 close.
import sys

import numpy as np
import pandas as pd

sym = sys.argv[1]
z = np.load(f"data/cache/sip/all/5Min/{sym}.npz")
keys = list(z.keys())
tk = "timestamp" if "timestamp" in keys else keys[0]
ts = pd.to_datetime(z[tk], utc=True).tz_convert("America/New_York")
df = pd.DataFrame({k: z[k] for k in keys if k != tk}, index=ts)
df = df[(df.index >= "2016-01-04") & (df.index < "2025-10-01")]
df["date"] = df.index.date
cnt = df.groupby("date").size()
df = df[df["date"].isin(cnt[cnt == 78].index)]
C = df["close"].to_numpy().reshape(-1, 78)
O = df["open"].to_numpy().reshape(-1, 78)
V = df["volume"].to_numpy().reshape(-1, 78)
dates = pd.to_datetime(sorted(set(df["date"])))
yrs = dates.year
D = C.shape[0]
lc = np.log(C)
# return earned by a position decided at bar j close: open_{j+1} -> open_{j+2}; last decision bar j=76 earns open_77 -> close_77
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
vwap = np.cumsum(C * V, axis=1) / np.cumsum(V, axis=1)


def ls_slope(x, N):
    k = np.arange(N) - (N - 1) / 2
    w = k / np.sum(k * k)
    s = np.full_like(x, np.nan)
    for j in range(N - 1, x.shape[1]):
        s[:, j] = x[:, j - N + 1 : j + 1] @ w
    return s


def report(pos, label, first=6):
    pos = pos.copy()
    pos[:, :first] = 0
    pos = np.nan_to_num(pos)
    pnl = np.nansum(pos * fwd, axis=1)  # daily gross bp (log units)
    trades = np.abs(np.diff(np.c_[np.zeros(D), pos, np.zeros(D)], axis=1)).sum(1) / 2  # round trips per day
    ok = ~np.isnan(band[:, 0]) & ~np.isnan(sd)
    p = pnl[ok]
    y = yrs[ok]
    tr = trades[ok]
    t = p.mean() / p.std() * np.sqrt(len(p))
    sh = p.mean() / p.std() * np.sqrt(252)
    exp = (np.abs(pos[ok]) > 0).mean()
    byy = " ".join(f"{yy}:{p[y == yy].mean() * 1e4:+.1f}" for yy in sorted(set(y)))
    print(
        f"{label:58s} gross {p.mean() * 1e4:5.2f} bp/d  t {t:5.2f}  Sharpe {sh:5.2f}  rt/day {tr.mean():4.2f}  exposure {exp:.2f}  | {byy}"
    )


print(
    f"{sym}: {D} sessions; gross = before costs; a round trip on SPY costs ~0.3 bp (quoted half-spreads) to ~2.3 bp (repo default 1 bp slippage/side)"
)
for VM in [1.0, 1.5]:
    pos = np.where(mv > VM * band, 1.0, np.where(mv < -VM * band, -1.0, 0.0))
    report(pos, f"band VM={VM}, flat inside, every bar")
    # half-hourly decisions only (Zarattini): hold the last half-hour decision
    hh = np.zeros_like(pos)
    idx = [j for j in range(78) if (j + 1) % 6 == 0]  # bars ending 10:00, 10:30, ...
    cur = np.zeros(D)
    for j in range(78):
        if j in idx:
            cur = pos[:, j]
        hh[:, j] = cur
    report(hh, f"band VM={VM}, flat inside, half-hourly decisions")
    # with VWAP trailing stop: long only while close > vwap, short only while close < vwap
    stop = np.where((pos > 0) & (C < vwap), 0.0, np.where((pos < 0) & (C > vwap), 0.0, pos))
    report(stop, f"band VM={VM}, + flat if wrong side of VWAP, every bar")
for N in [6, 12, 24]:
    v = ls_slope(lc, N) / sd[:, None]
    for k in [0.75, 1.0, 1.5]:
        pos = np.where(v > k, 1.0, np.where(v < -k, -1.0, 0.0))
        report(pos, f"LS velocity N={N} |v|>{k}, flat inside")
        # stop-and-reverse (RMV style): hold the last nonzero signal until the opposite one
        sar = np.zeros_like(pos)
        cur = np.zeros(D)
        for j in range(78):
            cur = np.where(pos[:, j] != 0, pos[:, j], cur)
            sar[:, j] = cur
        report(sar, f"LS velocity N={N} |v|>{k}, stop-and-reverse")
report(np.ones((D, 78)), "control: always long 10:00-close (intraday drift)")
