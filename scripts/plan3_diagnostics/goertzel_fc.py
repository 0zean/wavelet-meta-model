# Meyers' adaptive n-cycle Goertzel next-bar forecast (EPF 1560 bars, periods 4..390, top-n amplitudes): does fp(k)-ep(k) predict the next bar(s)?
import sys
import time

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
C = df["close"].to_numpy()
n = len(C)
W = 1560
periods = np.arange(4, 391)
f = 1 / periods
k = np.arange(W)
E = np.exp(-2j * np.pi * np.outer(f, k))  # (386, W)
i = np.arange(W)


def epf(Y):  # endpoint flatten rows of Y (B, W)
    a = Y[:, :1]
    b = (Y[:, -1:] - a) / (W - 1)
    return Y - (a + b * i)


starts = np.arange(W, n - 13, 13)  # one window every 13 bars (6 per session), forecasting bar start
t0 = time.time()
fc = {1: [], 3: [], 5: [], 10: []}
nxt1 = []
nxt6 = []
dayidx = []
B = 512
for s0 in range(0, len(starts), B):
    ss = starts[s0 : s0 + B]
    Y = np.stack([C[s - W : s] for s in ss])
    Yf = epf(Y)
    G = Yf @ E.T  # (B, 386) complex Goertzel at each frequency
    amp = 2 * np.abs(G) / W
    ph = np.angle(G)
    for ncy in fc:
        top = np.argsort(-amp, axis=1)[:, :ncy]
        a = np.take_along_axis(amp, top, 1)
        p = np.take_along_axis(ph, top, 1)
        ff = f[top]
        fp = (a * np.cos(2 * np.pi * ff * (W) + p)).sum(1)
        ep = (a * np.cos(2 * np.pi * ff * (W - 1) + p)).sum(1)
        fc[ncy].append(fp - ep)
    nxt1.append(np.log(C[ss] / C[ss - 1]))
    nxt6.append(np.log(C[np.minimum(ss + 5, n - 1)] / C[ss - 1]))
    dayidx.append(ss // 78)
nxt1 = np.concatenate(nxt1)
nxt6 = np.concatenate(nxt6)
dayidx = np.concatenate(dayidx)
print(f"{sym}: {len(starts)} windows in {time.time() - t0:.0f}s")
for ncy in fc:
    v = np.concatenate(fc[ncy])
    s = np.sign(v)
    for name, nx in [("next bar", nxt1), ("next 6 bars", nxt6)]:
        c = np.corrcoef(v, nx)[0, 1]
        m = (s * nx).mean() * 1e4
        dm = pd.Series(s * nx).groupby(dayidx).sum()
        t = dm.mean() / dm.std() * np.sqrt(len(dm))
        print(f"  top-{ncy} cycles -> {name:12s}: corr {c:+.4f}, mean signed {m:+.2f} bp, t(by day) {t:+.2f}")
