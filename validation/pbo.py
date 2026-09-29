"""
Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
(Bailey, Borwein, López de Prado & Zhu 2017; SPEC §6).

Given a T×N performance matrix M (rows = time periods in order, columns = strategy configurations/trials):
1. cut the rows into S contiguous blocks (S even, `np.array_split` sizes);
2. for each of the C(S, S/2) choices of S/2 blocks as in-sample (IS), the rest is out-of-sample (OOS);
3. n* = the column with the best IS performance (ties → lowest column index);
4. ω = (OOS rank of n* among the N columns, 1 = worst, ties averaged) / (N + 1), λ = log(ω / (1 − ω));
5. PBO = share of combinations with λ < 0 (the IS winner ranks below the OOS median), a combination with λ = 0
   (exactly the median rank, possible when N is odd or with ties) counting ½ — so pure noise gives PBO ≈ 0.5 for
   every N instead of (N+1)/(2N).
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # P(λ < 0) + ½·P(λ = 0)
    logits: np.ndarray  # λ per combination
    is_perf: np.ndarray  # IS performance of n* per combination
    oos_perf: np.ndarray  # OOS performance of n* per combination
    best: np.ndarray  # n* per combination
    prob_oos_loss: float  # P(OOS performance of n* < 0)
    degradation_slope: float  # OLS slope of oos_perf on is_perf (negative = IS gains do not carry over)
    n_combinations: int


def _block_sharpe(counts: np.ndarray, sums: np.ndarray, sqs: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Per-column Sharpe (ddof = 1) from pooled block moments of the shifted returns x − shift; zero variance → 0 if the
    mean is 0 else ±inf. Shifting each column by its first value keeps the variance exact for flat columns (all
    shifted values are 0) and limits cancellation in Σx² − (Σx)²/n.
    """
    mean_c = sums / counts
    mean = mean_c + shift
    var = np.maximum((sqs - sums * mean_c) / (counts - 1), 0.0)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = mean / sd
    flat = np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
    return np.where(sd > 0, sr, flat)


def pbo(perf_matrix, n_blocks: int = 16, metric: Callable[[np.ndarray], np.ndarray] | None = None) -> PBOResult:
    """
    CSCV Probability of Backtest Overfitting.

    Args:
        perf_matrix (array-like): T×N per-period returns (or other performance) of N trials over T periods.
        n_blocks (int): S, even, ≤ T. Default 16 (SPEC §6).
        metric (callable, optional): maps a (t×N) row subset to N scores (higher = better). Default = the
            per-period Sharpe ratio, computed from pooled block moments.

    Returns:
        PBOResult
    """
    M = perf_matrix.to_numpy(dtype=float) if isinstance(perf_matrix, pd.DataFrame) else np.asarray(perf_matrix, float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T periods × N trials)")
    T, N = M.shape
    if N < 2:
        raise ValueError(f"need at least 2 trials (columns); got {N}")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError(f"n_blocks must be an even integer >= 2; got {n_blocks}")
    if T < 2 * n_blocks:
        raise ValueError(f"need at least 2 rows per block: T={T} < 2·n_blocks={2 * n_blocks}")
    if not np.isfinite(M).all():
        raise ValueError("perf_matrix contains NaN or inf")

    blocks = np.array_split(np.arange(T), n_blocks)
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    is_mask = np.zeros((len(combos), n_blocks), dtype=bool)
    for c, cols in enumerate(combos):
        is_mask[c, list(cols)] = True

    if metric is None:
        cnt = np.array([b.size for b in blocks], dtype=float)[:, None]
        shift = M[0]
        Mc = M - shift
        s = np.stack([Mc[b].sum(0) for b in blocks])
        q = np.stack([(Mc[b] ** 2).sum(0) for b in blocks])

        def perf(mask):
            m = mask.astype(float)
            return _block_sharpe(m @ cnt, m @ s, m @ q, shift)

        is_scores, oos_scores = perf(is_mask), perf(~is_mask)
    else:
        is_scores = np.empty((len(combos), N))
        oos_scores = np.empty((len(combos), N))
        for c, mask in enumerate(is_mask):
            is_rows = np.concatenate([blocks[j] for j in np.flatnonzero(mask)])
            oos_rows = np.concatenate([blocks[j] for j in np.flatnonzero(~mask)])
            is_scores[c], oos_scores[c] = metric(M[is_rows]), metric(M[oos_rows])
        if np.isnan(is_scores).any() or np.isnan(oos_scores).any():
            raise ValueError("metric returned NaN")

    rows = np.arange(len(combos))
    best = is_scores.argmax(1)
    ranks = rankdata(oos_scores, axis=1, method="average")[rows, best]
    omega = ranks / (N + 1)
    logits = np.log(omega / (1 - omega))
    is_best, oos_best = is_scores[rows, best], oos_scores[rows, best]
    finite = np.isfinite(is_best) & np.isfinite(oos_best)
    x, y = is_best[finite], oos_best[finite]
    slope = np.polyfit(x, y, 1)[0] if x.size >= 2 and np.ptp(x) > 0 else np.nan
    return PBOResult(
        pbo=float(np.mean(logits < 0) + 0.5 * np.mean(logits == 0)),
        logits=logits,
        is_perf=is_best,
        oos_perf=oos_best,
        best=best,
        prob_oos_loss=float(np.mean(oos_best < 0)),
        degradation_slope=float(slope),
        n_combinations=len(combos),
    )
