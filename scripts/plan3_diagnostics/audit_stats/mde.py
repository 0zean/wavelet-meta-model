from common import *
from scipy.stats import norm

from families.stats import newey_west_alpha, sharpe_test

z = lambda a: norm.ppf(1 - a)
RF = 0.0215  # assumed mean 3M T-bill 2016-01..2025-09 (approx., not from repo data)
rows = []
for f in FAMS:
    j = pair(f)
    a, b = j.a.values, j.b.values
    yrs = len(a) / 252
    r = sharpe_test(a, b)
    se = r["se_ann"]
    q = (r["ci_ann"][1] - r["delta_ann"]) / se
    nw = newey_west_alpha(a, b, 5)
    resid = a - nw["beta"] * b
    se_alpha_ann = nw["alpha_ann"] / nw["alpha_t"]
    ar_se = se_alpha_ann / (resid.std() * np.sqrt(252))
    rows.append(
        dict(
            fam=f,
            se=round(se, 3),
            q_boot=round(q, 2),
            mde_2s05=round((z(0.025) + z(0.2)) * se, 2),
            mde_holm2s=round((z(0.05 / 16) + z(0.2)) * se, 2),
            mde_holm1s=round((z(0.05 / 8) + z(0.2)) * se, 2),
            need_SR_holm2s=round(T.sharpe_ann(b) + (z(0.05 / 16) + z(0.2)) * se, 2),
            mde_AR_05=round((z(0.025) + z(0.2)) * ar_se, 2),
            mde_AR_holm1s=round((z(0.05 / 8) + z(0.2)) * ar_se, 2),
            rf_bias=round(RF / (b.std() * np.sqrt(252)), 2),
        )
    )
print(pd.DataFrame(rows).set_index("fam").to_string())
