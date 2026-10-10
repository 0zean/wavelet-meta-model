import sys

from common import *
from scipy.stats import norm

from families.stats import circular_block_indices, sharpe_test

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
fams = sys.argv[2].split(",") if len(sys.argv) > 2 else FAMS
out = []
for f in fams:
    j = pair(f)
    a = j["a"].to_numpy()
    b = j["b"].to_numpy()
    n = len(a)
    base = sharpe_test(a, b)
    se = base["se_ann"]
    mde05 = (norm.ppf(0.975) + norm.ppf(0.8)) * se
    mdeH = (norm.ppf(1 - 0.05 / 16) + norm.ppf(0.8)) * se
    srb = b.mean() / b.std()
    HI = {
        "F1": 0.30,
        "F2": 0.80 - 0.862,
        "F3": 0.39,
        "F4": 1.0 - 0.670,
        "F5": 1.0 - 0.827,
        "F7": 0.80 - 0.866,
        "F8": 0.90 - 0.705,
        "F11": 0.0,
    }
    for label, dl in [("null", 0.0), ("mde05", mde05), ("mdeH", mdeH), ("plan2_hi", HI[f])]:
        mu = (srb + dl / np.sqrt(252)) * a.std()
        a0 = a - a.mean() + mu
        rng = np.random.default_rng(12345)
        ps, ds = [], []
        for k in range(NS):
            idx = circular_block_indices(n, 21, 1, rng)[0]
            extra = rng.integers(0, n, 1)[0]
            idx = np.concatenate([idx, (extra + np.arange(n - len(idx))) % n])  # pad to length n
            r = sharpe_test(a0[idx], b[idx], seed=k)
            ps.append(r["p"])
            ds.append(r["delta_ann"])
        ps, ds = np.array(ps), np.array(ds)
        row = dict(
            fam=f,
            case=label,
            delta_true=round(dl, 3),
            se_ann=round(se, 3),
            rej05_2s=np.mean(ps < 0.05),
            pow05_pos=np.mean((ps < 0.05) & (ds > 0)),
            powH_pos=np.mean((ps < 0.05 / 8) & (ds > 0)),
            mean_dhat=round(ds.mean(), 3),
        )
        print(row, flush=True)
        out.append(row)
pd.DataFrame(out).to_csv(f"power_{'_'.join(fams)}.csv", index=False)
