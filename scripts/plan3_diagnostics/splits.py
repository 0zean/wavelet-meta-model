# State splits and cross-rule correlation for the two intraday-continuation rules (gross, bp/day).
import numpy as np
import pandas as pd


def load(sym):
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
    return C, O, dates


def rules(C, O):
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
        p = p.copy()
        p[:, :6] = 0
        out[name] = np.nansum(np.nan_to_num(p) * fwd, axis=1)
    out["rv5"] = sd
    out["cc"] = np.r_[np.nan, np.diff(np.log(C[:, -1]))]
    out["ok"] = ~np.isnan(band[:, 0]) & ~np.isnan(sd)
    return pd.DataFrame(out)


res = {}
for sym in ["SPY", "QQQ"]:
    C, O, dates = load(sym)
    t = rules(C, O)
    t.index = dates
    res[sym] = t[t["ok"]]
    print(
        f"{sym}: corr(band_hh, sar6) daily = {t['band_hh'].corr(t['sar6']):.2f}; corr with close-to-close = {t['band_hh'].corr(t['cc']):.2f} / {t['sar6'].corr(t['cc']):.2f}"
    )
    q = pd.qcut(t["rv5"].shift(1), 5, labels=False)  # prior-day trailing vol quintile (known before the session)
    for name in ["band_hh", "sar6"]:
        g = t.groupby(q)[name].agg(["mean", "std", "count"])
        g["bp"] = g["mean"] * 1e4
        g["t"] = g["mean"] / g["std"] * np.sqrt(g["count"])
        print(
            f"  {name} by prior trailing-vol quintile (bp/day, t): "
            + "  ".join(f"q{int(i)}: {row.bp:+.1f} ({row.t:+.1f})" for i, row in g.iterrows())
        )
    # post-publication slice for the band rule (Zarattini first version 2024-05-10)
    post = t[t.index >= "2024-05-13"]
    for name in ["band_hh", "sar6"]:
        x = post[name]
        print(
            f"  {name} 2024-05-13 -> 2025-09-30: {x.mean() * 1e4:+.2f} bp/day, t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)}"
        )
both = res["SPY"][["band_hh", "sar6"]].join(res["QQQ"][["band_hh", "sar6"]], lsuffix="_spy", rsuffix="_qqq").dropna()
print("cross-symbol corr:", both.corr().round(2).to_dict())
