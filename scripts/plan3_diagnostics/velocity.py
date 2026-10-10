# Mechanism check: does a noise-exceeding intraday move (velocity or distance-from-open vs time-of-day noise) continue?
# Forward signed return over the next H bars (and to the 15:55 close), conditional on |signal| > thr. 2016-2025, by year.
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
dates = pd.to_datetime(sorted(set(df["date"])))
yrs = dates.year
D = C.shape[0]
lc = np.log(C)
# time-of-day noise band (Zarattini): mean |close_{t-i,slot} / open_{t-i} - 1| over prior 14 sessions, per slot
mv = np.abs(lc - np.log(O[:, :1]))  # |move from open| per slot, per day
band = np.full_like(mv, np.nan)
for d in range(14, D):
    band[d] = mv[d - 14 : d].mean(0)
# velocity: least-squares slope over N bars, normalized by the slot-free rolling sd of 5Min returns over the prior 5 sessions
r = np.diff(lc, axis=1)  # 77 returns per day
sd = np.full(D, np.nan)
for d in range(5, D):
    sd[d] = r[d - 5 : d].std()


def ls_slope(x, N):
    k = np.arange(N) - (N - 1) / 2
    w = k / np.sum(k * k)
    s = np.full_like(x, np.nan)
    for j in range(N - 1, x.shape[1]):
        s[:, j] = x[:, j - N + 1 : j + 1] @ w
    return s


rows = []
H = 12  # forward horizon in bars (1 hour)
entries = range(
    6, 78 - H - 1
)  # decide at bar j (10:00 or later), enter at next bar open, measure to j+1+H close, and to the close


def fwd(j):
    e = np.log(O[:, j + 1])
    return np.log(C[:, j + 1 + H]) - e, np.log(C[:, -1]) - e


def summarize(mask, side, label):
    res = {}
    for name, h in [("1h", 0), ("close", 1)]:
        vals = []
        ds = []
        for j in entries:
            f = fwd(j)[h]
            m = mask[:, j] & ~np.isnan(f)
            vals.append((side[:, j] * f)[m])
            ds.append(np.flatnonzero(m))
        v = np.concatenate(vals)
        dd = np.concatenate(ds)
        # one observation per day: average the day's signed forward returns (removes overlap) then t across days
        day_mean = pd.Series(v).groupby(dd).mean()
        day_yr = yrs[day_mean.index.to_numpy()]
        t = day_mean.mean() / day_mean.std() * np.sqrt(len(day_mean))
        byyear = {y: round(day_mean[day_yr == y].mean() * 1e4, 1) for y in sorted(set(day_yr))}
        res[name] = (round(v.mean() * 1e4, 2), round(t, 2), len(v), len(day_mean), byyear)
    print(f"{label}:")
    for name, (m, t, n, nd, by) in res.items():
        print(f"   fwd {name}: mean {m} bp/trade, t(by day) {t}, n_obs {n}, n_days {nd}; by year bp: {by}")


# (a) Zarattini band: |move from open| > VM * band
for VM in [1.0, 1.5]:
    sig = lc - np.log(O[:, :1])
    mask = np.abs(sig) > VM * band
    side = np.sign(sig)
    summarize(mask, side, f"{sym} band breakout VM={VM} (enter next bar open, 10:00-14:50 decisions)")
# (b) LS velocity over N bars > k * sd
for N in [6, 12, 24]:
    s = ls_slope(lc, N)
    vn = s / sd[:, None]
    for k in [1.0, 1.5]:
        mask = np.abs(vn) > k
        side = np.sign(vn)
        summarize(mask, side, f"{sym} LS velocity N={N} |v|>{k} sd")
# (c) unconditional (control): every bar, side = sign of the last 1-hour return
s12 = lc - np.roll(lc, 12, axis=1)
s12[:, :12] = np.nan
summarize(~np.isnan(s12), np.sign(s12), f"{sym} control: sign of trailing 1h return, every bar")
