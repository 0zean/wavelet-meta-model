"""
Statistics of the family test (SPEC §17.2, §22): the Ledoit & Wolf (2008) Sharpe-ratio difference test by a
studentized circular block bootstrap, a one-sample version (Sharpe > 0), the same studentized block bootstrap of a
daily mean against zero (`mean_test`: the overlay-alpha test of protocol v3, U22), alpha against a benchmark by OLS
with Newey–West standard errors, Holm step-down adjustment and James–Stein shrinkage of per-instrument Sharpe ratios.
Every test takes `sided`: "two" (|statistic|) or "one" (H1: the statistic is positive).

Conventions: daily returns; Sharpe ratios are per period (mean / population std) unless named `_ann`
(× √252). Every random draw takes an explicit seed.

Ledoit & Wolf (2008), "Robust performance hypothesis testing with the Sharpe ratio", J. Empirical Finance 15:
Δ = SR_a − SR_b = f(u) with u = (μ_a, μ_b, γ_a, γ_b) the first and second moments of y_t = (a_t, b_t, a_t², b_t²),
f(u) = μ_a/√(γ_a − μ_a²) − μ_b/√(γ_b − μ_b²), s(Δ̂) = √(∇f(û)ᵀ Ψ̂ ∇f(û) / T) (delta method). Ψ̂ (the long-run
covariance of y) is the Bartlett-kernel HAC with bandwidth b (the block length) on the data, and on each circular
block-bootstrap sample of l = ⌊T/b⌋ blocks the block-sum estimator Ψ* = (1/l) Σ_j ζ_j ζ_jᵀ, ζ_j = b^(−1/2) Σ_{t ∈ j}
(y*_t − ȳ*) (their §3.2). The two-sided p-value is (1 + #{|Δ* − Δ̂|/s(Δ*) ≥ |Δ̂|/s(Δ̂)}) / (M + 1); the 1 − α
interval is Δ̂ ± q_{1−α}(|Δ* − Δ̂|/s(Δ*))·s(Δ̂). Blocks resample whole stretches of sessions, so the autocorrelation a
multi-day holding period puts into a daily stream is kept within blocks (block 21 sessions ≫ any family's hold).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
_CHUNK = 250  # bootstrap samples per vectorized chunk (memory: chunk · T · 4 floats)


def circular_block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, l·block) indices of circular block-bootstrap samples, l = ⌊n / block⌋ blocks of `block` sessions."""
    if not 1 <= block <= n // 2:
        raise ValueError(f"block must be in [1, n/2]; got block={block}, n={n}")
    n_blocks = n // block
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    return ((starts[..., None] + np.arange(block)) % n).reshape(n_boot, n_blocks * block)


def _moments(Y: np.ndarray) -> np.ndarray:
    return Y.mean(axis=-2)


def _f_grad(u: np.ndarray, two: bool) -> tuple[np.ndarray, np.ndarray]:
    """Δ and ∇Δ at moments u (…, k): k = 4 (a, b, a², b²) for a difference, k = 2 (a, a²) for one Sharpe."""
    if two:
        ma, mb, ga, gb = (u[..., i] for i in range(4))
        va, vb = ga - ma**2, gb - mb**2
        d = ma / np.sqrt(va) - mb / np.sqrt(vb)
        g = np.stack([ga / va**1.5, -gb / vb**1.5, -ma / (2 * va**1.5), mb / (2 * vb**1.5)], axis=-1)
    else:
        ma, ga = u[..., 0], u[..., 1]
        va = ga - ma**2
        d = ma / np.sqrt(va)
        g = np.stack([ga / va**1.5, -ma / (2 * va**1.5)], axis=-1)
    return d, g


def hac_bartlett(Y: np.ndarray, bandwidth: int) -> np.ndarray:
    """Bartlett-kernel long-run covariance of the rows of Y (T, k): Γ0 + Σ_{j<b} (1 − j/b)(Γj + Γjᵀ)."""
    Z = Y - Y.mean(axis=0)
    T = len(Z)
    S = Z.T @ Z / T
    for j in range(1, bandwidth):
        G = Z[j:].T @ Z[:-j] / T
        S += (1 - j / bandwidth) * (G + G.T)
    return S


def _y(a: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    return np.column_stack([a, a**2]) if b is None else np.column_stack([a, b, a**2, b**2])


def sharpe_test(
    a,
    b=None,
    *,
    block: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    sided: str = "two",
) -> dict:
    """
    Ledoit–Wolf studentized circular block-bootstrap test of H0: SR_a = SR_b (paired daily returns on the same days),
    or of H0: SR_a = 0 when b is None (module docstring). Returns per-period and annualized Δ̂, its standard error,
    the p-value (two-sided on |t|, or one-sided against H1: Δ > 0), the 1 − alpha interval (annualized; one-sided: the
    lower bound and +inf), n_obs, block and n_boot.
    """
    if sided not in ("one", "two"):
        raise ValueError(f"sided must be 'one' or 'two', got {sided!r}")
    a = np.asarray(a, dtype=float)
    two = b is not None
    b = None if b is None else np.asarray(b, dtype=float)
    if two and a.shape != b.shape:
        raise ValueError(f"paired streams must have the same length; got {a.shape} and {b.shape}")
    if np.isnan(a).any() or (two and np.isnan(b).any()):
        raise ValueError("returns contain NaN: align the streams first")
    T = len(a)
    if T < 2 * block:
        raise ValueError(f"need at least 2 blocks of {block} returns; got {T}")
    if a.std() == 0 or (two and b.std() == 0):
        raise ValueError("a stream has zero variance: its Sharpe ratio is undefined")
    Y = _y(a, b)
    d_hat, g = _f_grad(_moments(Y), two)
    se = float(np.sqrt(g @ hac_bartlett(Y, block) @ g / T))
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = circular_block_indices(T, block, m, rng)
        Ys = Y[idx]  # (m, L, k)
        L = Ys.shape[1]
        u = _moments(Ys)
        d, gs = _f_grad(u, two)
        zeta = (Ys - u[:, None, :]).reshape(m, L // block, block, Y.shape[1]).sum(axis=2) / np.sqrt(block)
        psi = np.einsum("mjk,mjl->mkl", zeta, zeta) / zeta.shape[1]
        s = np.sqrt(np.einsum("mk,mkl,ml->m", gs, psi, gs) / L)
        stats.append((d - d_hat) / s)
    stats = np.concatenate(stats)
    stats = stats[np.isfinite(stats)]
    ann = np.sqrt(TRADING_DAYS)
    out = _boot_pvalue(float(d_hat), se, stats, alpha, sided)
    return {
        "delta": float(d_hat),
        "delta_ann": float(d_hat * ann),
        "se": se,
        "se_ann": se * ann,
        "t": float(d_hat / se),
        "p": out["p"],
        "ci_ann": (float(out["ci"][0] * ann), float(out["ci"][1] * ann)),
        "n_obs": T,
        "block": block,
        "n_boot": n_boot,
        "sided": sided,
    }


def _boot_pvalue(stat: float, se: float, boot: np.ndarray, alpha: float, sided: str) -> dict:
    """p-value and 1 − alpha interval of `stat` (standard error `se`) from the centred studentized bootstrap
    statistics `boot` = (θ* − θ̂)/s*: two-sided on |·|, one-sided against H1: θ > 0 (the interval is then the lower
    confidence bound and +inf)."""
    t_obs = stat / se
    if sided == "two":
        ab = np.abs(boot)
        p = float((1 + np.sum(ab >= abs(t_obs))) / (len(ab) + 1))
        q = float(np.quantile(ab, 1 - alpha))
        return {"p": p, "ci": (stat - q * se, stat + q * se)}
    p = float((1 + np.sum(boot >= t_obs)) / (len(boot) + 1))
    q = float(np.quantile(boot, 1 - alpha))  # P(θ̂ − θ ≤ q·se) = 1 − α → θ ≥ θ̂ − q·se
    return {"p": p, "ci": (stat - q * se, float("inf"))}


def mean_test(
    r, *, block: int = 21, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0, sided: str = "one"
) -> dict:
    """
    Studentized circular block-bootstrap test of H0: E[r] = 0 for a daily stream (the overlay-alpha test, SPEC §22):
    t̂ = mean / se with the Bartlett HAC standard error (bandwidth = the block), the bootstrap statistics
    (mean* − mean̂) / se* with the block-sum variance estimator on each resample of ⌊T/b⌋ blocks (as sharpe_test),
    p one-sided against H1: E[r] > 0 (or two-sided), the 1 − alpha interval. Returns the mean per day and per
    year (× 252) in fractions, the standard error, t, p, the interval (annualized), and the Newey–West (lag 5) t as a
    cross-check.
    """
    if sided not in ("one", "two"):
        raise ValueError(f"sided must be 'one' or 'two', got {sided!r}")
    x = np.asarray(r, dtype=float)
    if np.isnan(x).any():
        raise ValueError("returns contain NaN: align the stream first")
    T = len(x)
    if T < 2 * block:
        raise ValueError(f"need at least 2 blocks of {block} returns; got {T}")
    if x.std() == 0:
        raise ValueError("the stream has zero variance: nothing to test")
    m_hat = float(x.mean())
    Y = x[:, None]
    se = float(np.sqrt(hac_bartlett(Y, block)[0, 0] / T))
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = circular_block_indices(T, block, m, rng)
        xs = x[idx]  # (m, L)
        L = xs.shape[1]
        means = xs.mean(axis=1)
        zeta = (xs - means[:, None]).reshape(m, L // block, block).sum(axis=2) / np.sqrt(block)
        s = np.sqrt((zeta**2).mean(axis=1) / L)
        stats.append((means - m_hat) / s)
    boot = np.concatenate(stats)
    boot = boot[np.isfinite(boot)]
    out = _boot_pvalue(m_hat, se, boot, alpha, sided)
    nw = mean_ci_nw(x)
    nw_se = (nw[2] - nw[0]) / norm.ppf(0.975) if np.isfinite(nw[2]) else np.nan
    return {
        "mean": m_hat,
        "mean_ann": m_hat * TRADING_DAYS,
        "mean_bp_per_day": m_hat * 1e4,
        "se": se,
        "t": m_hat / se,
        "p": out["p"],
        "ci_ann": (float(out["ci"][0] * TRADING_DAYS), float(out["ci"][1] * TRADING_DAYS)),
        "t_nw": float(m_hat / nw_se) if nw_se and np.isfinite(nw_se) and nw_se > 0 else np.nan,
        "n_obs": T,
        "block": block,
        "n_boot": n_boot,
        "sided": sided,
    }


def newey_west_alpha(y, x, lags: int = 5) -> dict:
    """
    OLS y = α + β·x + ε with Newey–West (Bartlett, `lags`) standard errors: α (per day and annualized × 252), β,
    t-statistics and the two-sided normal p-value of α.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lags + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lags + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = beta / se
    return {
        "alpha": float(beta[0]),
        "alpha_ann": float(beta[0] * TRADING_DAYS),
        "alpha_t": float(t[0]),
        "alpha_p": float(2 * norm.sf(abs(t[0]))),
        "beta": float(beta[1]),
        "beta_t": float(t[1]),
        "lags": lags,
    }


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    from experiments.report import holm as _holm

    return _holm(np.asarray(p, dtype=float))


def james_stein(sr, n_obs) -> np.ndarray:
    """
    Positive-part James–Stein shrinkage of k per-period Sharpe ratios toward their mean (Efron & Morris 1975):
    SR_i^JS = m + max(0, 1 − (k − 3)·σ̄² / Σ(SR_i − m)²)·(SR_i − m), σ̄² = the mean sampling variance
    (1 + SR_i²/2) / T_i. With k ≤ 3 the factor is 1 (no shrinkage: the estimator needs k ≥ 4).
    """
    sr = np.asarray(sr, dtype=float)
    n = np.broadcast_to(np.asarray(n_obs, dtype=float), sr.shape)
    k = len(sr)
    if k <= 3:
        return sr.copy()
    m = sr.mean()
    ss = float(((sr - m) ** 2).sum())
    var = float(np.mean((1 + sr**2 / 2) / n))
    c = max(0.0, 1 - (k - 3) * var / ss) if ss > 0 else 0.0
    return m + c * (sr - m)


def sharpe_ann(r) -> float:
    """Annualized Sharpe (ddof 1) of daily returns; NaN for fewer than 2 or a flat series."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def mean_ci_nw(r, lags: int = 5, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of r with a Newey–West (Bartlett, `lags`) `level` interval: (mean, lo, hi)."""
    r = np.asarray(r, dtype=float)
    if r.size < 3:
        return float(r.mean()) if r.size else np.nan, np.nan, np.nan
    z = r - r.mean()
    s = (z @ z) / r.size
    for j in range(1, min(lags, r.size - 1) + 1):
        s += 2 * (1 - j / (lags + 1)) * (z[j:] @ z[:-j]) / r.size
    se = np.sqrt(max(s, 0.0) / r.size)
    q = norm.ppf(0.5 + level / 2)
    return float(r.mean()), float(r.mean() - q * se), float(r.mean() + q * se)


def drawdown(r: pd.Series) -> float:
    curve = np.cumprod(1 + np.asarray(r, dtype=float))
    return float((curve / np.maximum.accumulate(curve) - 1).min()) if curve.size else np.nan
