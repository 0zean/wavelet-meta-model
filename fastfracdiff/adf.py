"""
Augmented Dickey–Fuller test with a constant (statsmodels' `adfuller(x, regression="c")`), fast lag selection.

Regression: Δx_t = c + γ x_{t-1} + Σ_{j=1..p} φ_j Δx_{t-j} + e_t; the statistic is γ's t-value. With
autolag="AIC", p ∈ [0, maxlag] minimizes AIC over the models fit on the common sample of the widest one (the
statsmodels convention); the chosen model is then refit on all the rows its p allows. The candidate models are
nested prefixes of the columns [1, x_{t-1}, Δx_{t-1}, …, Δx_{t-maxlag}], so one factorization serves them all:
with the non-constant columns centred (which leaves every slope and its standard error unchanged, the constant being
in every model) and scaled, Z = [x̃_{t-1}, Δx̃_{t-1}, …], ZᵀZ = LLᵀ and z = L⁻¹Zᵀỹ, the residual sum of squares of
the model with the first j of them is ỹᵀỹ − Σ_{i<j} z_i². One BLAS product and a (maxlag+1)² Cholesky replace
maxlag + 1 SVD-based regressions (~170x at n = 190k); centring keeps ZᵀZ well conditioned (t-values agree with
statsmodels to ~1e-12).
"""

from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import norm

# MacKinnon (1994) p-value surface, constant-only regression, one variable (statsmodels' tsa/adfvalues.py)
_TAU_MAX, _TAU_MIN, _TAU_STAR = 2.74, -18.83, -1.61
_SMALLP = np.array([2.1659, 1.4412, 3.8269]) * np.array([1, 1, 1e-2])
_LARGEP = np.array([1.7339, 9.3202, -1.2745, -1.0368]) * np.array([1, 1e-1, 1e-1, 1e-2])


class ADFResult(NamedTuple):
    stat: float  # t-value of γ
    pvalue: float  # MacKinnon approximate p-value
    usedlag: int  # p
    nobs: int  # rows of the final regression


def mackinnonp(stat: float) -> float:
    """MacKinnon's approximate p-value of an ADF t-statistic (constant, no trend, N = 1)."""
    if stat > _TAU_MAX:
        return 1.0
    if stat < _TAU_MIN:
        return 0.0
    coef = _SMALLP if stat <= _TAU_STAR else _LARGEP
    return float(norm.cdf(np.polynomial.polynomial.polyval(stat, coef)))


def _design(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """[x_{t-1}, Δx_{t-1}, …, Δx_{t-lags}] and Δx_t for t = lags+1 … n-1, each column and Δx_t centred and the
    columns scaled to unit norm (the constant's role without its column; a t-value is scale-free)."""
    dx = np.diff(x)
    n = dx.size - lags
    Z = np.empty((n, lags + 1))
    Z[:, 0] = x[lags:-1]
    if lags:
        Z[:, 1:] = sliding_window_view(dx, lags)[:n, ::-1]
    Z -= Z.mean(axis=0)
    Z /= np.sqrt((Z * Z).sum(axis=0))
    y = dx[lags:]
    return Z, y - y.mean()


def adfuller(x, maxlag: int | None = None, autolag: str | None = "AIC") -> ADFResult:
    """
    ADF test with a constant. maxlag defaults to ceil(12 (n/100)^¼) capped at n//2 - 2 (Schwert, as statsmodels);
    autolag "AIC" picks the lag by AIC (ties → the shorter), None uses maxlag.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("adfuller takes a 1-D series")
    cap = x.size // 2 - 2
    if maxlag is None:
        maxlag = min(cap, int(np.ceil(12.0 * (x.size / 100.0) ** 0.25)))
        if maxlag < 0:
            raise ValueError("series too short for the ADF regression")
    elif maxlag > cap:
        raise ValueError(f"maxlag must be at most n//2 - 2 = {cap}")
    if autolag is None:
        lag = maxlag
    elif autolag.upper() == "AIC":
        Z, y = _design(x, maxlag)
        z = solve_triangular(np.linalg.cholesky(Z.T @ Z), Z.T @ y, lower=True)
        k = np.arange(2, maxlag + 3)  # columns in each candidate: 1, x_{t-1} and 0 … maxlag lags
        rss = y @ y - np.concatenate(([0.0], np.cumsum(z**2)))[k - 1]
        aic = y.size * np.log(rss / y.size) + 2 * k  # statsmodels' AIC less a constant common to all candidates
        lag = int(np.argmin(aic))
    else:
        raise ValueError(f"unsupported autolag: {autolag!r}")
    Z, y = _design(x, lag)
    c = cho_factor(Z.T @ Z)
    beta = cho_solve(c, Z.T @ y)
    resid = y - Z @ beta
    s2 = resid @ resid / (y.size - Z.shape[1] - 1)
    var = s2 * cho_solve(c, np.eye(Z.shape[1])[:, 0])[0]  # Var(γ) = s² [(ZᵀZ)⁻¹]₀₀
    stat = float(beta[0] / np.sqrt(var))
    return ADFResult(stat, mackinnonp(stat), lag, y.size)
