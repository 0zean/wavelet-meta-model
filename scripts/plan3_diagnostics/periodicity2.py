# Daily-cycle power test with a null that keeps within-session autocorrelation: circularly shift each session by a random offset.
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
px = df["close"].to_numpy().reshape(-1, 78)
r = np.log(px[:, 1:] / px[:, :-1])
r = r - r.mean()
D = r.shape[0]
rng = np.random.default_rng(1)


def pw(x, f):
    k = np.arange(len(x))
    return np.abs(np.sum(x * np.exp(-2j * np.pi * f * k))) ** 2 / len(x)


W = 40  # sessions per window
out = {f: [] for f in ["1/d", "2/d", "3/d", "1/w"]}
fr = {"1/d": 1 / 77, "2/d": 2 / 77, "3/d": 3 / 77, "1/w": 1 / 385}
for s in range(0, D - W, W):
    seg = r[s : s + W]
    x = seg.reshape(-1)
    for name, f in fr.items():
        p = pw(x, f)
        null = []
        for _ in range(60):
            sh = np.stack([np.roll(row, rng.integers(77)) for row in seg]).reshape(-1)
            null.append(pw(sh, f))
        null = np.array(null)
        out[name].append(((null >= p).mean(), p / np.median(null)))
print(f"{sym}: {D} sessions, {len(out['1/d'])} windows of {W} sessions")
for name in out:
    a = np.array(out[name])
    print(
        f"  {name}: share of windows with p<0.05 = {(a[:, 0] < 0.05).mean():.2f} (expect 0.05); median power/null median = {np.median(a[:, 1]):.2f}"
    )
# mean-return seasonality strength: variance of slot means vs sampling variance
mu = r.mean(0)
v_slot = mu.var()
v_samp = (r.var(0, ddof=1) / D).mean()
print(f"  slot-mean variance / sampling variance = {v_slot / v_samp:.2f} (1.0 = no seasonality in the mean)")
