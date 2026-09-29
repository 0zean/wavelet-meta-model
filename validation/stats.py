"""
Sharpe-ratio significance statistics (SPEC §6): Probabilistic Sharpe Ratio (Bailey & López de Prado 2012),
Deflated Sharpe Ratio (Bailey & López de Prado 2014) and the minimum track record length.

Conventions: Sharpe ratios are **per period** (not annualized); `kurt` is the **raw** (non-excess) kurtosis,
3 for a Normal. Sample moments are the plain (biased, population) estimators, as in the papers.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class ReturnMoments:
    sr: float  # per-period Sharpe ratio (mean / std, ddof = 1)
    skew: float
    kurt: float  # raw kurtosis
    n_obs: int


def sharpe_ratio(returns) -> float:
    """Per-period Sharpe ratio mean / std (ddof = 1) of excess returns. Raises on NaN or zero variance."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("need a 1-D series of at least 2 returns")
    if np.isnan(r).any():
        raise ValueError("returns contain NaN")
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("returns have zero variance; the Sharpe ratio is undefined")
    return float(r.mean() / sd)


def return_moments(returns) -> ReturnMoments:
    """Sharpe ratio, skewness and raw kurtosis of a return series (the PSR/DSR inputs)."""
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r)
    return ReturnMoments(sr, float(skew(r)), float(kurtosis(r, fisher=False)), r.size)


def _sr_std(sr: float, n_obs: int, skew_: float, kurt: float) -> float:
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2; got {n_obs}")
    v = 1 - skew_ * sr + (kurt - 1) / 4 * sr**2
    if not v > 0:
        raise ValueError(f"non-positive Sharpe variance term {v:.4g} (sr={sr}, skew={skew_}, kurt={kurt})")
    return float(np.sqrt(v / (n_obs - 1)))


def psr(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0, sr_star: float = 0.0) -> float:
    """
    PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ): the probability that the true per-period
    Sharpe ratio exceeds `sr_star`, given the estimate `sr` over `n_obs` returns with skewness γ₃ and raw kurtosis γ₄.
    """
    return float(norm.cdf((sr - sr_star) / _sr_std(sr, n_obs, skew, kurt)))


def expected_max_sharpe(n_trials: int, var_trials: float, mean_trials: float = 0.0) -> float:
    """
    SR₀ = E[max SR_n] ≈ E[SR_n] + √V[SR_n]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))) for N independent trials
    (γ = Euler–Mascheroni). With one trial there is no selection, so SR₀ = E[SR_n].
    """
    if n_trials < 1 or int(n_trials) != n_trials:
        raise ValueError(f"n_trials must be a positive integer; got {n_trials}")
    if var_trials < 0:
        raise ValueError(f"var_trials must be >= 0; got {var_trials}")
    if n_trials == 1:
        return float(mean_trials)
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(mean_trials + np.sqrt(var_trials) * z)


def dsr(sr: float, n_trials: int, var_trials: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Deflated Sharpe ratio: PSR(SR₀), SR₀ = `expected_max_sharpe(n_trials, var_trials)` under the null that every
    trial's true Sharpe is zero. `sr` and `var_trials` must be per-period (variance of the trials' per-period SRs).
    """
    return psr(sr, n_obs, skew, kurt, sr_star=expected_max_sharpe(n_trials, var_trials))


def min_track_record_length(
    sr: float, sr_star: float = 0.0, skew: float = 0.0, kurt: float = 3.0, alpha: float = 0.05
) -> float:
    """
    MinTRL = 1 + (1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²)·(Φ⁻¹(1−α) / (SR̂ − SR*))²: the number of returns needed for
    PSR(SR*) ≥ 1 − α. Infinite when SR̂ ≤ SR*.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if sr <= sr_star:
        return float("inf")
    v = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if not v > 0:
        raise ValueError(f"non-positive Sharpe variance term {v:.4g} (sr={sr}, skew={skew}, kurt={kurt})")
    return float(1 + v * (norm.ppf(1 - alpha) / (sr - sr_star)) ** 2)
