from common import *

from families.stats import newey_west_alpha, sharpe_ann, sharpe_test

rows = []
for f in FAMS:
    j = pair(f)
    a, b = j["a"], j["b"]
    T_ = len(j)
    nw5 = newey_west_alpha(a, b, lags=5)
    nw21 = newey_west_alpha(a, b, lags=21)
    rho = a.corr(b)
    comb = a + b
    lw_c = sharpe_test(comb.values, b.values)
    lw = sharpe_test(a.values, b.values)
    exp = a != 0
    ex = a[exp]
    resid = a - nw5["beta"] * b
    rows.append(
        dict(
            fam=f,
            n=T_,
            yrs=round(T_ / 252, 2),
            sr=round(sharpe_ann(a), 3),
            sr_b=round(sharpe_ann(b), 3),
            rho=round(rho, 3),
            vol_a=round(a.std() * np.sqrt(252), 4),
            vol_b=round(b.std() * np.sqrt(252), 4),
            beta=round(nw5["beta"], 3),
            alpha_ann=round(nw5["alpha_ann"], 4),
            t_nw5=round(nw5["alpha_t"], 2),
            t_nw21=round(nw21["alpha_t"], 2),
            appraisal=round(nw5["alpha_ann"] / (resid.std() * np.sqrt(252)), 3),
            sr_comb=round(sharpe_ann(comb), 3),
            d_comb=round(sharpe_ann(comb) - sharpe_ann(b), 3),
            p_comb=round(lw_c["p"], 3),
            expo=round(exp.mean(), 3),
            bp_per_exp_day=round(ex.mean() * 1e4, 2),
            sr_exp_days=round(sharpe_ann(ex), 3),
            bench_sr_exp_days=round(sharpe_ann(b[exp]), 3),
            delta=round(lw["delta_ann"], 3),
            se_ann=round(lw["se_ann"], 3),
            p=round(lw["p"], 4),
        )
    )
df = pd.DataFrame(rows).set_index("fam")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(df.T.to_string())
df.to_csv("diag.csv")
