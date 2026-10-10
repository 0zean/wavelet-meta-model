from common import *

from families import stats as S
from validation.stats import dsr, expected_max_sharpe, psr, return_moments

# 1. gradient check
rng = np.random.default_rng(0)
u = np.array([0.0004, 0.0003, 0.0004**2 + 1e-4, 0.0003**2 + 2e-4])
d, g = S._f_grad(u, True)
num = np.array([(S._f_grad(u + e, True)[0] - S._f_grad(u - e, True)[0]) / (2 * e.sum()) for e in np.eye(4) * 1e-9])
print("grad analytic", g, "\nnumeric      ", num)
# 2. se vs IID delta method (Memmel 2003 / JK) on F1 and F4
for f in ["F1", "F4", "F8"]:
    j = pair(f)
    a, b = j.a.values, j.b.values
    Y = S._y(a, b)
    _, g = S._f_grad(S._moments(Y), True)
    se_iid = np.sqrt(g @ np.cov(Y.T, bias=True) @ g / len(a)) * np.sqrt(252)
    se_hac = np.sqrt(g @ S.hac_bartlett(Y, 21) @ g / len(a)) * np.sqrt(252)
    r = S.sharpe_test(a, b)
    q = (r["ci_ann"][1] - r["delta_ann"]) / r["se_ann"]
    # count non-finite bootstrap stats: replicate inner loop
    rr = np.random.default_rng(0)
    idx = S.circular_block_indices(len(a), 21, 2000, rr)
    Ys = Y[idx]
    u2 = Ys.mean(1)
    va = u2[:, 2] - u2[:, 0] ** 2
    vb = u2[:, 3] - u2[:, 1] ** 2
    print(
        f,
        "se iid",
        round(se_iid, 3),
        "se HAC21",
        round(se_hac, 3),
        "boot crit q",
        round(q, 3),
        "zero-var resamples",
        int((va <= 0).sum() + (vb <= 0).sum()),
    )
    # seed sensitivity of p
    print("   p over seeds", [round(S.sharpe_test(a, b, seed=s)["p"], 4) for s in range(5)])
    # block sensitivity
    print("   p by block", {bl: round(S.sharpe_test(a, b, block=bl)["p"], 4) for bl in (5, 10, 21, 42, 63)})
# 3. DSR
v = 0.0010746747475198408
for n in (8, 44):
    sr0 = expected_max_sharpe(n, v)
    print("N", n, "SR0 per-day", round(sr0, 4), "annualized", round(sr0 * np.sqrt(252), 3))
print(
    "pure-noise V at T=2400 per-day:",
    1 / 2400,
    "-> sd ann",
    round(np.sqrt(1 / 2400 * 252), 3),
    "; program V sd ann",
    round(np.sqrt(v * 252), 3),
)
for n in (8, 44):
    sr0n = expected_max_sharpe(n, 1 / 2400)
    print("N", n, "SR0 if V were pure noise (ann)", round(sr0n * np.sqrt(252), 3))
# DSR of the benchmark itself at N=44
for f in ["F1", "F3"]:
    j = pair(f)
    m = return_moments(j.b.values)
    print(
        f,
        "benchmark SR ann",
        round(m.sr * np.sqrt(252), 3),
        "DSR(N=44,V)",
        round(dsr(m.sr, 44, v, m.n_obs, m.skew, m.kurt), 3),
        "PSR0",
        round(psr(m.sr, m.n_obs, m.skew, m.kurt), 4),
    )
