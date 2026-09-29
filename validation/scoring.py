"""
Probabilistic scoring and purged cross-validated evaluation (SPEC §5).

Models are **selected** on sample-weighted neg-log-loss (default) or Brier; AUC, F1 and accuracy are
**reported only** — they ignore calibration, which bet sizing (U7) depends on.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from validation.purged_cv import _Splitter

SELECTION_METRICS = ("neg_log_loss", "brier")


def _check(y, p, w) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("y and p must be 1-D and the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary {0, 1}")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p must be probabilities in [0, 1]")
    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.shape != y.shape or (w < 0).any() or not w.sum() > 0:
            raise ValueError("sample weights must be non-negative, aligned to y and not all zero")
    return y, p, w


def neg_log_loss(y, p, sample_weight=None) -> float:
    """−(weighted mean log-loss) of P(y=1) = p; higher is better."""
    y, p, w = _check(y, p, sample_weight)
    return -float(log_loss(y, p, sample_weight=w, labels=[0, 1]))


def brier(y, p, sample_weight=None) -> float:
    """Weighted mean squared error of p against y; lower is better."""
    y, p, w = _check(y, p, sample_weight)
    return float(brier_score_loss(y, p, sample_weight=w, pos_label=1))


def selection_score(metric: str, y, p, sample_weight=None) -> float:
    """Model-selection score, **higher is better**: neg-log-loss, or −Brier."""
    if metric == "neg_log_loss":
        return neg_log_loss(y, p, sample_weight)
    if metric == "brier":
        return -brier(y, p, sample_weight)
    raise ValueError(f"unknown selection metric {metric!r}; choose from {SELECTION_METRICS}")


def score_report(y, p, sample_weight=None, threshold: float = 0.5) -> dict[str, float]:
    """Selection metrics plus report-only AUC / F1 / accuracy (at `threshold`). AUC is NaN with one class."""
    y, p, w = _check(y, p, sample_weight)
    hard = (p >= threshold).astype(int)
    return {
        "neg_log_loss": neg_log_loss(y, p, w),
        "brier": brier(y, p, w),
        "auc": float(roc_auc_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else float("nan"),
        "f1": float(f1_score(y, hard, sample_weight=w, zero_division=0)),
        "accuracy": float(accuracy_score(y, hard, sample_weight=w)),
    }


@dataclass(frozen=True)
class FoldPrediction:
    train: np.ndarray
    test: np.ndarray
    proba: np.ndarray  # P(y=1) on test


def purged_cv_predict(estimator, X, y, cv: _Splitter, t0, t1, sample_weight=None) -> Iterator[FoldPrediction]:
    """
    Fit a fresh clone of `estimator` on each purged train split (with train sample weights) and predict
    P(y=1) on its test split. Rows of X / y / weights must be in span order. A split whose train set has one
    class raises (no silent skip).
    """
    Xa = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y = np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if len(Xa) != len(y) or len(y) != len(np.asarray(t0)) or (w is not None and len(w) != len(y)):
        raise ValueError("X, y, sample_weight and t0/t1 must have the same length")
    for s, (train, test) in enumerate(cv.split(t0, t1)):
        if np.unique(y[train]).size < 2:
            raise ValueError(f"split {s}: the purged train set has a single class ({len(train)} samples)")
        model = clone(estimator)
        kw = {} if w is None else {"sample_weight": w[train]}
        model.fit(Xa[train], y[train], **kw)
        classes = list(model.classes_)
        yield FoldPrediction(train, test, model.predict_proba(Xa[test])[:, classes.index(1)])


def purged_cv_score(
    estimator, X, y, cv: _Splitter, t0, t1, sample_weight=None, metric: str = "neg_log_loss"
) -> np.ndarray:
    """Per-split `selection_score` (test weights = the test rows' sample weights), higher is better."""
    if metric not in SELECTION_METRICS:
        raise ValueError(f"unknown selection metric {metric!r}; choose from {SELECTION_METRICS}")
    y_ = np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return np.array(
        [
            selection_score(metric, y_[f.test], f.proba, None if w is None else w[f.test])
            for f in purged_cv_predict(estimator, X, y_, cv, t0, t1, w)
        ]
    )
