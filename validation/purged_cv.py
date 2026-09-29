"""
Purged cross-validation (López de Prado 2018, §7.4 and §12; SPEC §5).

Every sample i spans bars [t0_i, t1_i] (event bar, exit bar) — integer bar positions, samples sorted by t0.
A train sample is dropped when its span overlaps a test span (purge) or when it starts within `embargo` bars
after a contiguous block of test samples ends (embargo). Both splitters yield positional index arrays;
`bind(t0, t1)` adapts them to scikit-learn's `cv=` protocol.
"""

import math
from collections.abc import Iterator
from itertools import combinations

import numpy as np


def embargo_bars(embargo_pct: float, n_bars: int) -> int:
    """Embargo h = ceil(embargo_pct · n_bars) bars (SPEC §5); the product is rounded to 1e-9 first so float error
    (0.07 · 100 = 7.000000000000001) cannot add a bar."""
    if not 0 <= embargo_pct < 1:
        raise ValueError(f"embargo_pct must be in [0, 1); got {embargo_pct}")
    if n_bars < 0:
        raise ValueError(f"n_bars must be >= 0; got {n_bars}")
    return math.ceil(round(embargo_pct * n_bars, 9))


def _check_spans(t0, t1, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = np.asarray(t0), np.asarray(t1)
    if t0.ndim != 1 or t0.shape != t1.shape:
        raise ValueError("t0 and t1 must be 1-D and the same length")
    if not (np.issubdtype(t0.dtype, np.integer) and np.issubdtype(t1.dtype, np.integer)):
        raise TypeError(f"t0/t1 must be integer bar positions; got {t0.dtype}/{t1.dtype}")
    if np.any(np.diff(t0) < 0):
        raise ValueError("samples must be sorted by event time t0")
    if np.any(t1 < t0):
        raise ValueError("every span needs t1 >= t0")
    if len(t0) < min_samples:
        raise ValueError(f"{len(t0)} samples < {min_samples} required")
    return t0, t1


def purged_train(t0: np.ndarray, t1: np.ndarray, test: np.ndarray, embargo: int) -> np.ndarray:
    """
    Train indices left after purging and embargoing against `test`.

    `test` may be non-contiguous (CPCV). It is split into runs of consecutive indices; since samples are sorted
    by t0, a sample outside a run overlaps some test span of the run iff it overlaps the run's hull
    [min t0, max t1], so purging against each hull is exact. The embargo follows each run.

    Args:
        t0, t1 (np.ndarray): Sorted integer spans.
        test (np.ndarray): Sorted test indices.
        embargo (int): Bars embargoed after each test run.

    Returns:
        np.ndarray: Sorted train indices.
    """
    keep = np.ones(len(t0), dtype=bool)
    keep[test] = False
    runs = np.split(test, np.flatnonzero(np.diff(test) > 1) + 1)
    for run in runs:
        if run.size:
            a, b = t0[run].min(), t1[run].max()
            keep &= (t1 < a) | (t0 > b + embargo)
    return np.flatnonzero(keep)


class _Splitter:
    embargo: int

    def _test_sets(self, n: int) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    def n_splits(self) -> int:
        raise NotImplementedError

    def _min_samples(self) -> int:
        raise NotImplementedError

    def split(self, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train, test) positional indices for samples with spans [t0, t1]."""
        t0, t1 = _check_spans(t0, t1, self._min_samples())
        for test in self._test_sets(len(t0)):
            yield purged_train(t0, t1, test, self.embargo), test

    def bind(self, t0, t1) -> "BoundSplitter":
        """A scikit-learn compatible splitter (`split(X, y, groups)`, `get_n_splits`) for these spans."""
        return BoundSplitter(self, t0, t1)


class PurgedKFold(_Splitter):
    """
    Contiguous, unshuffled k-fold. For a test fold spanning [a, b] (a = min test t0, b = max test t1), a train
    sample is kept only if t1 < a or t0 > b + embargo.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self._n_splits, self.embargo = n_splits, embargo

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def _min_samples(self) -> int:
        return self._n_splits

    def _test_sets(self, n: int) -> list[np.ndarray]:
        return np.array_split(np.arange(n), self._n_splits)


class CombinatorialPurgedCV(_Splitter):
    """
    CPCV(N, k): the samples are cut into N contiguous groups; every one of the C(N, k) choices of k groups is a
    test set, the rest (purged and embargoed around each contiguous test run) is train. Splits are enumerated in
    `itertools.combinations` order. Each group is tested in φ = C(N−1, k−1) splits, which yields φ complete
    backtest paths (`path_splits`, `assemble_paths`).
    """

    def __init__(self, n_groups: int = 10, k: int = 2, embargo: int = 0):
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not 1 <= k < n_groups:
            raise ValueError(f"k must be in [1, n_groups); got k={k}, n_groups={n_groups}")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_groups, self.k, self.embargo = n_groups, k, embargo
        self.combos = list(combinations(range(n_groups), k))

    @classmethod
    def from_cfg(cls, cfg, n_bars: int) -> "CombinatorialPurgedCV":
        """CPCV(CPCV_GROUPS, CPCV_TEST_GROUPS) with embargo = ceil(CV_EMBARGO_PCT · n_bars)."""
        return cls(cfg.CPCV_GROUPS, cfg.CPCV_TEST_GROUPS, embargo_bars(cfg.CV_EMBARGO_PCT, n_bars))

    @property
    def n_splits(self) -> int:
        return len(self.combos)

    @property
    def n_paths(self) -> int:
        """φ = C(N, k)·k / N = C(N−1, k−1)."""
        return math.comb(self.n_groups - 1, self.k - 1)

    def _min_samples(self) -> int:
        return self.n_groups

    def groups(self, n: int) -> list[np.ndarray]:
        """The N contiguous sample groups (`np.array_split` sizes)."""
        return np.array_split(np.arange(n), self.n_groups)

    def _test_sets(self, n: int) -> list[np.ndarray]:
        g = self.groups(n)
        return [np.concatenate([g[j] for j in c]) for c in self.combos]

    def path_splits(self) -> np.ndarray:
        """
        (φ, N) array: entry [p, g] is the split whose predictions fill group g on path p. The splits testing
        group g, in enumeration order, are assigned to paths 0..φ−1, so every (split, tested group) pair is
        used exactly once.
        """
        out = np.empty((self.n_paths, self.n_groups), dtype=int)
        for g in range(self.n_groups):
            out[:, g] = [s for s, c in enumerate(self.combos) if g in c]
        return out

    def assemble_paths(self, test_values: list[np.ndarray], n: int) -> np.ndarray:
        """
        Stitch per-split OOS values into φ full-length backtest paths.

        Args:
            test_values (list[np.ndarray]): One array per split (in `split` order), aligned to that split's test
                indices (the first axis has len(test) entries).
            n (int): Number of samples.

        Returns:
            np.ndarray: shape (φ, n, …) — path p's OOS value for every sample.
        """
        if len(test_values) != self.n_splits:
            raise ValueError(f"expected {self.n_splits} split outputs, got {len(test_values)}")
        groups, tests = self.groups(n), self._test_sets(n)
        vals = [np.asarray(v) for v in test_values]
        for s, (v, t) in enumerate(zip(vals, tests)):
            if v.shape[:1] != t.shape:
                raise ValueError(f"split {s}: {v.shape[0]} values for {t.size} test samples")
        out = np.full((self.n_paths, n, *vals[0].shape[1:]), np.nan)
        for p, row in enumerate(self.path_splits()):
            for g, s in enumerate(row):
                pos = np.searchsorted(tests[s], groups[g])
                out[p, groups[g]] = vals[s][pos]
        return out


class BoundSplitter:
    """scikit-learn `cv=` adapter: a splitter bound to fixed spans; X must have len(t0) rows in span order."""

    def __init__(self, splitter: _Splitter, t0, t1):
        self.splitter = splitter
        self.t0, self.t1 = _check_spans(t0, t1, splitter._min_samples())

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.splitter.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if X is not None and len(X) != len(self.t0):
            raise ValueError(f"X has {len(X)} rows but the splitter is bound to {len(self.t0)} spans")
        yield from self.splitter.split(self.t0, self.t1)
