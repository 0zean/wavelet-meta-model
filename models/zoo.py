"""
Model zoo (SPEC §4, U6): one protocol for the primary direction classifier and the meta-model.

    model = make_model("rf_ldp", cfg, role="meta")
    model.fit(X, y, sample_weight, inner_cv(labels, cfg))  # HP search + calibration, all inside (X, y)
    p = model.predict_proba(X_new)                          # calibrated P(y=1), 1-D

Every zoo model except `legacy` does the same thing inside `fit`, using only the rows it is given:
1. For each point of its small hyper-parameter grid, purged-CV out-of-fold P(y=1) (`purged_cv_predict`,
   fresh clone per split, train-row weights) scored with `cfg.SELECTION_METRIC` (test-row weights); the best mean
   score wins (ties → the first grid point).
2. Calibration: isotonic vs sigmoid (Platt on logit p), each cross-fitted on the winner's OOF predictions over
   the same purged splits (fit on the train rows' OOF p, scored on the test rows' OOF p); lower weighted Brier wins
   (ties → sigmoid).
3. The winner is refit on all rows; the chosen calibrator is fit on all OOF predictions (a model never scores
   its own training rows) and maps the refit model's raw P(y=1).

Feature columns with no value in any fitting row (a rule primary's `clf_prob`) are dropped; any other NaN raises.

`legacy` is the pre-U6 XGBoost (META_PARAMS for the meta role, CLF_PARAMS for the primary role): no search,
no calibration — the baseline and the legacy regression path.
"""

from collections.abc import Callable
from typing import ClassVar, Literal, Protocol

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from utils.config import RunConfig
from validation.purged_cv import BoundSplitter, PurgedKFold, embargo_bars
from validation.scoring import brier, purged_cv_predict, selection_score

Role = Literal["meta", "primary"]
CALIBRATIONS = ("sigmoid", "isotonic")  # tie order: sigmoid first
PROB_CLIP = 1e-3  # calibrated probabilities are clipped to [PROB_CLIP, 1 − PROB_CLIP] (isotonic yields exact 0/1)


class ZooFitError(ValueError):
    """The model cannot be fit on these rows (one class overall or in an inner purged train split)."""


class ZooModel(Protocol):
    name: str

    def fit(self, X, y, sample_weight, cv: BoundSplitter) -> "ZooModel": ...

    def predict_proba(self, X) -> np.ndarray: ...  # calibrated P(y=1), 1-D


REGISTRY: dict[str, Callable[..., ZooModel]] = {}


def zoo_model(name: str) -> Callable:
    def register(cls):
        if name in REGISTRY:
            raise ValueError(f"zoo model {name!r} registered twice")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return register


def make_model(name: str, cfg: RunConfig, role: Role = "meta") -> ZooModel:
    """A fresh, unfitted zoo model seeded with cfg.SEED."""
    if role not in ("meta", "primary"):
        raise ValueError(f"role must be 'meta' or 'primary', got {role!r}")
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown zoo model {name!r}; expected one of {sorted(REGISTRY)}") from None
    return cls(cfg, role)


def inner_cv(labels: pd.DataFrame, cfg: RunConfig) -> BoundSplitter:
    """
    PurgedKFold(ZOO_CV_SPLITS) bound to the label spans [event bar, exit bar] (rows in event order).
    Embargo = ceil(CV_EMBARGO_PCT · bars spanned by these labels).
    """
    t0 = labels["entry_pos"].to_numpy() - 1  # an event at bar t enters at t + 1
    t1 = labels["exit_pos"].to_numpy()
    n_bars = int(t1.max() - t0.min() + 1) if len(t0) else 0
    return PurgedKFold(cfg.ZOO_CV_SPLITS, embargo_bars(cfg.CV_EMBARGO_PCT, n_bars)).bind(t0, t1)


def proba(model, X) -> np.ndarray:
    """P(y=1) of a fitted scikit-learn style classifier."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


# ── Calibrators ──────────────────────────────────────────────────────────────


class _Sigmoid:
    """Platt scaling on logit(p) (weighted, effectively unpenalized)."""

    def fit(self, p, y, w) -> "_Sigmoid":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y, sample_weight=w)
        return self

    def predict(self, p) -> np.ndarray:
        return self.lr.predict_proba(_logit(p))[:, 1]


class _Isotonic:
    def fit(self, p, y, w) -> "_Isotonic":
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y, sample_weight=w)
        return self

    def predict(self, p) -> np.ndarray:
        return self.iso.predict(p)


_CALIBRATORS = {"sigmoid": _Sigmoid, "isotonic": _Isotonic}


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _clip(p) -> np.ndarray:
    return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


# ── Base: purged-CV HP search + calibration ─────────────────────────────────


def _as_xyw(X, y, sample_weight) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xa = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    ya = np.asarray(y)
    if not np.isin(ya, (0, 1)).all():
        raise ValueError("y must be binary {0, 1}")
    ya = ya.astype(int)
    w = np.ones(len(ya)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if not (len(Xa) == len(ya) == len(w)):
        raise ValueError("X, y and sample_weight must have the same length")
    if np.unique(ya).size < 2:
        raise ZooFitError(f"single class in {len(ya)} fitting rows")
    return Xa, ya, w


class _Tuned:
    """Zoo model with a hyper-parameter grid searched by purged CV and a CV-chosen calibrator (module docstring)."""

    name: str
    grid: ClassVar[dict[str, list]] = {}

    def __init__(self, cfg: RunConfig, role: Role = "meta"):
        self.cfg, self.role = cfg, role
        self.est = self.cal = None

    def estimator(self, params: dict, w: np.ndarray):
        """A fresh scikit-learn classifier for one grid point (`w` = the fitting rows' weights)."""
        raise NotImplementedError

    def fit(self, X, y, sample_weight, cv: BoundSplitter) -> "_Tuned":
        Xa, ya, w = _as_xyw(X, y, sample_weight)
        # Columns with no value in any fitting row carry nothing (e.g. a rule primary's clf_prob); any other NaN raises
        self.cols_ = ~np.isnan(Xa).all(axis=0)
        Xa = self._cols(Xa)
        if not isinstance(cv, BoundSplitter) or len(cv.t0) != len(ya):
            raise ValueError("cv must be a BoundSplitter bound to the fitting rows' spans")
        splits = list(cv.splitter.split(cv.t0, cv.t1))
        for i, (train, _) in enumerate(splits):
            if np.unique(ya[train]).size < 2:
                raise ZooFitError(
                    f"{self.name}: inner split {i} has a single-class purged train set ({train.size} rows)"
                )
        metric = self.cfg.SELECTION_METRIC

        rows, oofs = [], []
        for params in ParameterGrid(self.grid):
            oof = np.full(len(ya), np.nan)
            scores = []
            for f in purged_cv_predict(self.estimator(params, w), Xa, ya, cv.splitter, cv.t0, cv.t1, w):
                oof[f.test] = f.proba
                scores.append(selection_score(metric, ya[f.test], f.proba, w[f.test]))
            rows.append({"params": params, "score": float(np.mean(scores)), "score_sd": float(np.std(scores))})
            oofs.append(oof)
        best = int(np.argmax([r["score"] for r in rows]))  # first maximum → ties go to the earlier grid point
        oof = oofs[best]
        if np.isnan(oof).any():
            raise RuntimeError(f"{self.name}: purged CV left {int(np.isnan(oof).sum())} rows without an OOF prediction")

        cal_brier = {}
        for method in CALIBRATIONS:
            b = []
            for train, test in splits:
                c = _CALIBRATORS[method]().fit(oof[train], ya[train], w[train])
                b.append(brier(ya[test], _clip(c.predict(oof[test])), w[test]))
            cal_brier[method] = float(np.mean(b))
        method = min(CALIBRATIONS, key=lambda m: cal_brier[m])  # min returns the first on ties

        self.best_index_, self.best_params_ = best, rows[best]["params"]
        self.cv_results_ = pd.DataFrame(rows)
        self.calibration_, self.calibration_brier_ = method, cal_brier
        self.oof_raw_ = oof
        self.est = self.estimator(self.best_params_, w).fit(Xa, ya, sample_weight=w)
        self.cal = _CALIBRATORS[method]().fit(oof, ya, w)
        return self

    def _cols(self, Xa: np.ndarray) -> np.ndarray:
        if Xa.ndim != 2 or Xa.shape[1] != self.cols_.size:
            raise ValueError(f"{self.name}: expected {self.cols_.size} feature columns, got shape {Xa.shape}")
        Xa = Xa[:, self.cols_]
        if np.isnan(Xa).any():
            raise ValueError(f"{self.name}: NaN in {int(np.isnan(Xa).any(axis=1).sum())} rows of used features")
        return Xa

    def predict_raw(self, X) -> np.ndarray:
        if self.est is None:
            raise RuntimeError(f"{self.name}.predict_proba called before fit")
        return proba(self.est, self._cols(np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)))

    def predict_proba(self, X) -> np.ndarray:
        raw = self.predict_raw(X)
        return _clip(self.cal.predict(raw))


# ── Models ───────────────────────────────────────────────────────────────────


class ScaledLogit(ClassifierMixin, BaseEstimator):
    """StandardScaler (weighted, fit on the fitting rows only) + logistic regression; l1_ratio 1 = L1, 0 = L2."""

    def __init__(self, C: float = 1.0, l1_ratio: float = 0.0, random_state: int | None = None):
        self.C, self.l1_ratio, self.random_state = C, l1_ratio, random_state

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(X, sample_weight=sample_weight)
        solver = "saga" if self.l1_ratio > 0 else "lbfgs"
        self.lr_ = LogisticRegression(
            C=self.C, l1_ratio=self.l1_ratio, solver=solver, max_iter=5000, tol=1e-4, random_state=self.random_state
        ).fit(self.scaler_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.scaler_.transform(X))


@zoo_model("logit_l1")
class LogitL1(_Tuned):
    grid: ClassVar[dict[str, list]] = {"C": [0.01, 0.1, 1.0]}

    def estimator(self, params, w):
        return ScaledLogit(l1_ratio=1.0, random_state=self.cfg.SEED, **params)


@zoo_model("logit_l2")
class LogitL2(_Tuned):
    grid: ClassVar[dict[str, list]] = {"C": [0.01, 0.1, 1.0]}

    def estimator(self, params, w):
        return ScaledLogit(l1_ratio=0.0, random_state=self.cfg.SEED, **params)


class _SerialPredict:
    """Forest whose predict_proba runs single-threaded: threaded prediction sums the trees' probabilities in
    completion order, so repeated runs differ in the last bits (fitting stays parallel and seeded)."""

    def predict_proba(self, X):
        n_jobs, self.n_jobs = self.n_jobs, 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = n_jobs


class SerialPredictRF(_SerialPredict, RandomForestClassifier):
    pass


class SerialPredictET(_SerialPredict, ExtraTreesClassifier):
    pass


@zoo_model("rf_ldp")
class RfLdp(_Tuned):
    """López de Prado's forest: balanced_subsample, bootstrap of avg-uniqueness size (mean of the weights, which
    are average uniqueness in the pipeline), min_weight_fraction_leaf 0.05; max_features 1 vs sqrt."""

    grid: ClassVar[dict[str, list]] = {"max_features": [1, "sqrt"]}
    n_estimators: ClassVar[int] = 500

    def estimator(self, params, w):
        return SerialPredictRF(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            max_samples=float(np.clip(w.mean(), 1e-3, 1.0)),
            min_weight_fraction_leaf=0.05,
            n_jobs=-1,
            random_state=self.cfg.SEED,
            **params,
        )


@zoo_model("rf_ldp_fast")
class RfLdpFast(RfLdp):
    """rf_ldp at ~1/5 of the cost (U11; late SPY 5Min fold 8.9 s vs 43.5 s): max_features fixed at 1 — what
    rf_ldp's purged CV chose in 81 % of its 21,948 Stage A fold fits — so 4 CV fits + 1 refit instead of 9, and 200
    trees instead of 500 (OOS meta_prob vs 500 trees on 636 SPY 5Min events: r = 0.993, trade decisions 99.7 % equal)."""

    grid: ClassVar[dict[str, list]] = {"max_features": [1]}
    n_estimators: ClassVar[int] = 200


@zoo_model("extra_trees")
class ExtraTrees(_Tuned):
    grid: ClassVar[dict[str, list]] = {"min_weight_fraction_leaf": [0.01, 0.05]}

    def estimator(self, params, w):
        return SerialPredictET(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.cfg.SEED,
            **params,
        )


@zoo_model("xgb")
class Xgb(_Tuned):
    """The pipeline's XGBoost (META_PARAMS or CLF_PARAMS by role) with max_depth searched."""

    grid: ClassVar[dict[str, list]] = {"max_depth": [2, 4]}

    def estimator(self, params, w):
        base = self.cfg.META_PARAMS if self.role == "meta" else self.cfg.CLF_PARAMS
        return xgb.XGBClassifier(**{**base, **params})


@zoo_model("lightgbm")
class LightGbm(_Tuned):
    grid: ClassVar[dict[str, list]] = {"num_leaves": [7, 31]}

    def estimator(self, params, w):
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=20,
            deterministic=True,
            force_row_wise=True,
            n_jobs=4,
            verbose=-1,
            random_state=self.cfg.SEED,
            **params,
        )


@zoo_model("catboost")
class CatBoost(_Tuned):
    """CatBoost: symmetric (oblivious) trees — every node of a level splits on the same feature, a strong structural
    regularizer — with its default L2 leaf regularization; depth searched. Plain boosting: ordered boosting (each
    row's gradient from a model that never saw it) cost 12x xgb on a late SPY 5Min fold (185 s vs 16 s), plain 2x
    (33 s). Single-threaded and seeded (deterministic on CPU); writes no training files."""

    grid: ClassVar[dict[str, list]] = {"depth": [4, 6]}

    def estimator(self, params, w):
        return cb.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            boosting_type="Plain",
            random_seed=self.cfg.SEED,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **params,
        )


@zoo_model("legacy")
class Legacy:
    """Pre-U6 XGBoost: fixed META_PARAMS (meta) / CLF_PARAMS (primary), no search, no calibration; `cv` unused."""

    def __init__(self, cfg: RunConfig, role: Role = "meta"):
        self.params = cfg.META_PARAMS if role == "meta" else cfg.CLF_PARAMS
        self.est = None

    def fit(self, X, y, sample_weight, cv=None) -> "Legacy":
        self.est = xgb.XGBClassifier(**self.params)
        self.est.fit(X, y, sample_weight=sample_weight, verbose=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.est is None:
            raise RuntimeError("legacy.predict_proba called before fit")
        return proba(self.est, X)


def describe(model: ZooModel) -> str:
    """One-line summary of a fitted model's selection (for logs)."""
    if isinstance(model, _Tuned):
        cb = model.calibration_brier_
        return (
            f"{model.name} {model.best_params_}  calibration={model.calibration_} "
            f"(cv Brier sigmoid={cb['sigmoid']:.4f} isotonic={cb['isotonic']:.4f})"
        )
    return model.name
