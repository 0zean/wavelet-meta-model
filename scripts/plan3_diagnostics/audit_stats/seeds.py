from common import *

from families.stats import holm, sharpe_test

ps = {}
for f in FAMS:
    j = pair(f)
    ps[f] = [sharpe_test(j.a.values, j.b.values, seed=s)["p"] for s in range(20)]
P = pd.DataFrame(ps)
print(P.describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
H = np.array([holm(P.loc[s].values) for s in range(20)])
print(
    "F7 Holm over seeds: min %.4f max %.4f share<0.05 %.2f"
    % (H[:, FAMS.index("F7")].min(), H[:, FAMS.index("F7")].max(), (H[:, FAMS.index("F7")] < 0.05).mean())
)
