"""
Purged k-fold cross-validation (López de Prado 2018, §7.4; SPEC §5).

Minimal version introduced in U3 for clustered-MDA feature selection; U5 extends
this module (CPCV, property tests, sklearn adapter).
"""

from collections.abc import Iterator

import numpy as np


class PurgedKFold:
    """
    Contiguous, unshuffled k-fold over samples sorted by event time. Each sample i spans
    bars [t0_i, t1_i] (event bar, exit bar). For a test fold spanning [a, b]
    (a = min test t0, b = max test t1), a train sample is kept only if its span ends
    before a (t1 < a) or it starts after the embargo (t0 > b + embargo) — so no train
    span overlaps the test span, and the `embargo` bars after it are skipped too.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits, self.embargo = n_splits, embargo

    def split(self, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0, t1 = np.asarray(t0), np.asarray(t1)
        if len(t0) != len(t1):
            raise ValueError("t0 and t1 must have the same length")
        if np.any(np.diff(t0) < 0):
            raise ValueError("samples must be sorted by event time t0")
        if np.any(t1 < t0):
            raise ValueError("every span needs t1 >= t0")
        if len(t0) < self.n_splits:
            raise ValueError(f"{len(t0)} samples < n_splits={self.n_splits}")
        for test in np.array_split(np.arange(len(t0)), self.n_splits):
            a, b = t0[test].min(), t1[test].max()
            train = np.flatnonzero((t1 < a) | (t0 > b + self.embargo))
            yield train, test
