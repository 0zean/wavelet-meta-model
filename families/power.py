"""
Power of the overlay-alpha test (PLAN3 §4.5, SPEC §22; ports the statistics audit's power.py / mde.py).

    uv run python -m families power --n-days 2400 --vol-ann 0.08 --families 4 [--expected-bp 3.0] [--sims 200]

`mde_alpha` is the analytic minimum detectable alpha of a one-sided test of a daily mean against zero at 80 % power:
with σ_d the daily volatility of the overlay and n its days, se = σ_d / √n and MDE = (z_{1−α/k} + z_{power}) · se,
where k is the number of families Holm must hold simultaneously (its most stringent step is Bonferroni's α/k).
`simulate_power` checks it on `n_sims` streams with the planted alpha (iid normal, or GARCH(1,1) with Student-t
innovations: the fat-tailed, clustered daily returns of a long-volatility overlay) through the real
`families.stats.mean_test`. A family spec records the MDE it was given (`power.mde_alpha_bp_per_day`); the spec
loader refuses one that disagrees with `mde_alpha` by more than 10 %, and a family whose expected magnitude is below
its MDE must declare itself a diagnostic (`power.diagnostic: true`).
"""

import argparse

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


def mde_alpha(
    n_days: int, vol_ann: float, *, alpha: float = 0.05, power: float = 0.8, families: int = 1, sided: str = "one"
) -> dict:
    """Minimum detectable daily alpha (bp per day; annualized × 252) and its ingredients."""
    if n_days < 2 or not vol_ann > 0 or not 0 < alpha < 1 or not 0 < power < 1 or families < 1:
        raise ValueError("need n_days >= 2, vol_ann > 0, 0 < alpha, power < 1, families >= 1")
    se_daily = vol_ann / np.sqrt(TRADING_DAYS) / np.sqrt(n_days)
    a = alpha / families
    z_a = norm.ppf(1 - a) if sided == "one" else norm.ppf(1 - a / 2)
    z_p = norm.ppf(power)
    mde = (z_a + z_p) * se_daily
    return {
        "n_days": int(n_days), "vol_ann": float(vol_ann), "alpha": alpha, "power": power, "families": families,
        "sided": sided, "se_bp_per_day": float(se_daily * 1e4), "z_alpha": float(z_a), "z_power": float(z_p),
        "mde_bp_per_day": float(mde * 1e4), "mde_ann": float(mde * TRADING_DAYS),
        "mde_sharpe_ann": float(mde / (vol_ann / np.sqrt(TRADING_DAYS)) * np.sqrt(TRADING_DAYS)),
    }  # fmt: skip


def garch_t(n: int, vol_daily: float, rng, nu: float = 5.0, a: float = 0.05, b: float = 0.90) -> np.ndarray:
    """GARCH(1,1) daily returns with variance targeting (unconditional σ = vol_daily) and unit-variance Student-t(ν)
    innovations; the first 500 draws are burn-in."""
    omega = vol_daily**2 * (1 - a - b)
    z = rng.standard_t(nu, n + 500) * np.sqrt((nu - 2) / nu)
    v = np.empty(n + 500)
    v[0] = vol_daily**2
    r = np.empty(n + 500)
    r[0] = np.sqrt(v[0]) * z[0]
    for t in range(1, n + 500):
        v[t] = omega + a * r[t - 1] ** 2 + b * v[t - 1]
        r[t] = np.sqrt(v[t]) * z[t]
    return r[500:]


def simulate_power(
    n_days: int,
    vol_ann: float,
    alpha_bp_per_day: float,
    *,
    n_sims: int = 200,
    alpha: float = 0.05,
    families: int = 1,
    block: int = 21,
    n_boot: int = 500,
    seed: int = 0,
    stream: str = "iid",
) -> dict:
    """Rejection rate of the one-sided overlay-alpha test at level alpha / families on streams with the planted
    daily alpha (0 → the test's size). `stream`: "iid" normal or "garch_t"."""
    from families.stats import mean_test

    vol_d = vol_ann / np.sqrt(TRADING_DAYS)
    mu = alpha_bp_per_day * 1e-4
    rej = 0
    level = alpha / families
    for i in range(n_sims):
        rng = np.random.default_rng(seed * 100_003 + i)
        r = rng.normal(0.0, vol_d, n_days) if stream == "iid" else garch_t(n_days, vol_d, rng)
        r = r + mu  # the planted alpha on top of the simulated noise
        res = mean_test(r, block=block, n_boot=n_boot, alpha=level, seed=i, sided="one")
        rej += res["p"] < level
    return {"n_sims": n_sims, "stream": stream, "alpha_bp_per_day": alpha_bp_per_day, "level": level,
            "rejection_rate": rej / n_sims}  # fmt: skip


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-days", type=int, required=True)
    ap.add_argument("--vol-ann", type=float, required=True, help="realized annualized volatility of the overlay stream")
    ap.add_argument("--families", type=int, default=1, help="families tested at once (Holm)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--expected-bp", type=float, default=None, help="the family's expected alpha, bp per day")
    ap.add_argument("--sims", type=int, default=0, help="simulation check of the MDE (0 = analytic only)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--stream", choices=["iid", "garch_t"], default="iid")
    a = ap.parse_args(argv)
    out = mde_alpha(a.n_days, a.vol_ann, alpha=a.alpha, power=a.power, families=a.families)
    print(f"[POWER] n {out['n_days']} days, vol {out['vol_ann']:.3f}: se {out['se_bp_per_day']:.2f} bp/day; MDE "
          f"{out['mde_bp_per_day']:.2f} bp/day = {out['mde_ann']:.2%}/yr (Sharpe {out['mde_sharpe_ann']:.2f}) at "
          f"{a.power:.0%} power, one-sided alpha {a.alpha}/{a.families}")  # fmt: skip
    if a.expected_bp is not None:
        out["expected_bp_per_day"] = a.expected_bp
        out["diagnostic"] = a.expected_bp < out["mde_bp_per_day"]
        print(f"[POWER] expected {a.expected_bp:.2f} bp/day → " + ("a DIAGNOSTIC (below the MDE)" if out["diagnostic"]
                                                                   else "a test (at or above the MDE)"))  # fmt: skip
    if a.sims:
        for planted, key in ((0.0, "size"), (out["mde_bp_per_day"], "power_at_mde")):
            r = simulate_power(a.n_days, a.vol_ann, planted, n_sims=a.sims, alpha=a.alpha, families=a.families,
                               n_boot=a.n_boot, stream=a.stream)  # fmt: skip
            out[key] = r["rejection_rate"]
            print(f"[POWER] {key}: {r['rejection_rate']:.3f} over {a.sims} {a.stream} simulations")
    return out


if __name__ == "__main__":
    main()
