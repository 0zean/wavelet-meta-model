from common import *

for f in FAMS:
    d = streams(f)
    r = result(f)
    w = pd.Series(r["weights"])
    hd = d["headline"]
    first, last = hd.first_valid_index(), hd.last_valid_index()
    inner = d.loc[first:last]
    gaps = int(inner["headline"].isna().sum())
    bgaps = int(inner["benchmark"].isna().sum())
    print(
        f"== {f}: headline {first.date()}..{last.date()} n={hd.notna().sum()} interior NaN headline={gaps} bench={bgaps}; bench n={d.benchmark.notna().sum()}"
    )
    if r["basket"]:
        continue
    pi, row = per_instrument(f)
    starts = {k: str(v.index.min().date()) for k, v in pi.items()}
    nz = {k: str(v[v != 0].index.min().date()) for k, v in pi.items()}
    print("  per-instrument stream starts", starts, "first nonzero", nz)
    pooled = T.pool(pi, w)
    diff = (pooled - T.day_index(hd).reindex(pooled.index)).abs().max()
    print("  rebuilt pooled vs streams.csv max abs diff", diff)
    F = pd.concat(pi, axis=1).fillna(0.0).loc[first:last]
    sd = F.std()
    C = F.corr()
    Sig = F.cov()
    wv = w[F.columns]
    sp = np.sqrt(wv @ Sig @ wv)
    rc = wv * (Sig @ wv) / sp**2
    neff = (wv * sd).sum() ** 2 / sp**2
    print("  strategy vol ann", (sd * np.sqrt(252)).round(4).to_dict())
    print("  risk contributions", rc.round(3).to_dict())
    print("  corr\n", C.round(2).to_string())
    print(f"  N_eff (Sharpe gain^2 if equal per-instrument SR) = {neff:.2f} of {len(F.columns)}")
    # equal-risk on strategy vols would give
    w2 = (1 / sd) / (1 / sd).sum()
    sp2 = np.sqrt(w2 @ Sig @ w2)
    print(f"  N_eff with weights 1/strategy-vol = {((w2 * sd).sum() / sp2) ** 2:.2f}")
    print("  per-inst SR ann", {k: round(T.sharpe_ann(v.loc[first:last]), 3) for k, v in pi.items()})
