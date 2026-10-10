# Intraday structure diagnostics on cached SIP 5Min bars (read directly from the npz; dev window only).
import sys

import numpy as np
import pandas as pd

sym = sys.argv[1]
z = np.load(f"data/cache/sip/all/5Min/{sym}.npz")
keys = list(z.keys())
ts = pd.to_datetime(z["timestamp"] if "timestamp" in keys else z[keys[0]], utc=True).tz_convert("America/New_York")
df = pd.DataFrame({k: z[k] for k in keys if k != "timestamp"}, index=ts)
df = df[(df.index >= "2016-01-04") & (df.index < "2025-10-01")]
df["date"] = df.index.date
# keep only full 78-bar sessions
cnt = df.groupby("date").size()
full = cnt[cnt == 78].index
df = df[df["date"].isin(full)]
px = df["close"].to_numpy().reshape(-1, 78)
op = df["open"].to_numpy().reshape(-1, 78)
dates = pd.to_datetime(sorted(set(df["date"])))
r = np.log(px[:, 1:] / px[:, :-1])  # 77 intrabar returns per session (bar close to close)
r0 = np.log(px[:, 0] / op[:, 0])  # first bar open->close
gap = np.log(op[1:, 0] / px[:-1, -1])  # overnight: prev 15:55 close -> 09:30 open
intra = np.log(px[:, -1] / op[:, 0])  # open -> 15:55 close
print(f"{sym}: {len(dates)} full sessions {dates[0].date()}..{dates[-1].date()}")
yr = dates.year
print("year  overnight bp/d  intraday bp/d  (mean, t)")
for y in sorted(set(yr)):
    m = yr[1:] == y
    g = gap[m]
    i = intra[1:][m]
    print(
        f"{y}  {g.mean() * 1e4:6.2f} (t {g.mean() / g.std() * np.sqrt(len(g)):4.1f})   {i.mean() * 1e4:6.2f} (t {i.mean() / i.std() * np.sqrt(len(i)):4.1f})"
    )
# same-slot (lag 78 = next day same time) autocorrelation vs other lags, on the flattened RTH return series
flat = r.reshape(-1)


def ac(x, lag):
    a, b = x[:-lag], x[lag:]
    return np.corrcoef(a, b)[0, 1]


n = len(flat)
print("lag-k autocorrelation of 5Min returns (x1e3); 2se =", round(2 / np.sqrt(n) * 1e3, 2))
print({k: round(ac(flat, k) * 1e3, 2) for k in [1, 2, 3, 6, 12, 77, 154, 231, 308, 385]})
# per-slot mean return (seasonality in the mean), t-stat across sessions
mu = r.mean(0)
se = r.std(0, ddof=1) / np.sqrt(r.shape[0])
t = mu / se
slots = [f"{9 * 60 + 35 + 5 * i:04d}" for i in range(77)]
big = np.argsort(-np.abs(t))[:8]
print("largest |t| slot means (bar ending at HH:MM, mean bp, t):")
for i in sorted(big):
    hhmm = pd.Timestamp("09:30") + pd.Timedelta(minutes=5 * (i + 1))
    print(f"  {hhmm:%H:%M}  {mu[i] * 1e4:6.2f}  t {t[i]:5.2f}")


# Goertzel power at the 1-day period (78 bars) and harmonics on 40-session windows vs circular-shift null
def goertzel_pow(x, f):
    k = np.arange(len(x))
    return np.abs(np.sum(x * np.exp(-2j * np.pi * f * k))) ** 2 / len(x)


W = 40 * 77
xs = r.reshape(-1)
rng = np.random.default_rng(0)
freqs = {"1d": 1 / 77, "2/d": 2 / 77, "3/d": 3 / 77, "1w": 1 / (5 * 77)}
ratios = {k: [] for k in freqs}
for s in range(0, len(xs) - W, W):
    seg = xs[s : s + W]
    seg = seg - seg.mean()
    for k, f in freqs.items():
        p = goertzel_pow(seg, f)
        null = [goertzel_pow(rng.permutation(seg), f) for _ in range(50)]
        ratios[k].append(p / np.mean(null))
print("Goertzel power ratio to permutation null (median over windows, share > 95th pct of null):")
for k in freqs:
    v = np.array(ratios[k])
    print(f"  {k}: median {np.median(v):.2f}, share>~3.0 (95th pct of exp): {(v > 3).mean():.2f}, n={len(v)}")
