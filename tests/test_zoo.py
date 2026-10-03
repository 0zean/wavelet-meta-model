"""
U6: model zoo — protocol and registry, purged-CV HP search and calibration kept inside the fitting rows,
scaling / weight routing for the linear models, the WFO config switches (META_MODEL, PRIMARY_MODEL, META_TRAIN),
out-of-fold primary signals, OOS scoring and determinism.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from features.triple_barrier_labels import average_uniqueness, sample_events, triple_barrier_labels
from models import zoo
from models.meta_model import fit_meta_model
from models.zoo import PROB_CLIP, REGISTRY, ScaledLogit, ZooFitError, inner_cv, make_model
from primaries import make_primary
from tests.test_primaries import WFO_CFG, WFO_COLUMNS, random_walk_after
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from validation.purged_cv import PurgedKFold
from validation.scoring import score_report
from wfo import wfo_engine
from wfo.wfo_engine import purged, run_wfo, wfo_folds
from wfo.wfo_metrics import calibration_table, meta_outcomes, signal_diagnostics

CFG = RunConfig.for_timeframe("1Day")
TUNED = sorted(n for n in REGISTRY if n != "legacy")


def planted(n: int = 900, seed: int = 0, strength: float = 1.2):
    """Linear logit signal in the first two of 6 columns; spans of 5 bars every 3 bars (overlapping)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < 1 / (1 + np.exp(-strength * (X[:, 0] - 0.7 * X[:, 1])))).astype(int)
    w = rng.uniform(0.3, 1.0, n)
    t0 = np.arange(n) * 3
    return X, y, w, t0, t0 + 5


def cv_for(t0, t1, k: int = 4, embargo: int = 3):
    return PurgedKFold(k, embargo).bind(t0, t1)


# ── Registry, config ─────────────────────────────────────────────────────────


def test_registry_and_config_switches():
    assert set(REGISTRY) == {
        "legacy",
        "logit_l1",
        "logit_l2",
        "rf_ldp",
        "rf_ldp_fast",
        "extra_trees",
        "xgb",
        "lightgbm",
        "catboost",
    }
    with pytest.raises(ValueError, match="unknown zoo model"):
        make_model("nope", CFG)
    with pytest.raises(ValueError, match="role"):
        make_model("xgb", CFG, role="both")
    with pytest.raises(ValueError, match="META_MODEL"):
        CFG.replace(META_MODEL="nope")
    with pytest.raises(ValueError, match="PRIMARY_MODEL applies only"):
        CFG.replace(PRIMARY="sma_cross", PRIMARY_MODEL="rf_ldp")
    with pytest.raises(ValueError, match="META_TRAIN"):
        CFG.replace(META_TRAIN="all")
    with pytest.raises(ValueError, match="ZOO_CV_SPLITS"):
        CFG.replace(ZOO_CV_SPLITS=1)
    assert CFG.META_MODEL == CFG.PRIMARY_MODEL == "legacy" and CFG.META_TRAIN == "val"


def test_inner_cv_uses_label_spans_and_embargo():
    labels = pd.DataFrame({"entry_pos": [11, 21, 31, 41, 51, 61, 71, 81], "exit_pos": [15, 30, 35, 50, 60, 64, 90, 99]})
    cv = inner_cv(labels, CFG.replace(ZOO_CV_SPLITS=2, CV_EMBARGO_PCT=0.05))
    np.testing.assert_array_equal(cv.t0, labels["entry_pos"] - 1)
    np.testing.assert_array_equal(cv.t1, labels["exit_pos"])
    assert cv.get_n_splits() == 2 and cv.splitter.embargo == 5  # ceil(0.05 · (99 − 10 + 1)) = 5


# ── Protocol behaviour of every model ────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_model_learns_is_bounded_and_deterministic(name):
    X, y, w, t0, t1 = planted()
    tr, te = slice(0, 700), slice(700, None)
    a = make_model(name, CFG).fit(X[tr], y[tr], w[tr], cv_for(t0[tr], t1[tr]))
    b = make_model(name, CFG).fit(X[tr], y[tr], w[tr], cv_for(t0[tr], t1[tr]))
    p = a.predict_proba(X[te])
    assert p.shape == (200,)
    np.testing.assert_array_equal(p, b.predict_proba(X[te]))  # same seed → identical
    assert ((p > 0) & (p < 1)).all()
    if name != "legacy":
        assert (p >= PROB_CLIP).all() and (p <= 1 - PROB_CLIP).all()
    assert score_report(y[te], p)["auc"] > 0.75


def test_legacy_is_the_pre_u6_xgboost():
    X, y, w, *_ = planted(400)
    got = make_model("legacy", CFG).fit(X, y, w).predict_proba(X)
    want = xgb.XGBClassifier(**CFG.META_PARAMS).fit(X, y, sample_weight=w).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(got, want)
    prim = make_model("legacy", CFG, role="primary")
    assert prim.params == CFG.CLF_PARAMS


def test_unfitted_and_single_class_raise():
    X, _, w, t0, t1 = planted(200)
    with pytest.raises(RuntimeError, match="before fit"):
        make_model("logit_l2", CFG).predict_proba(X)
    with pytest.raises(ZooFitError, match="single class"):
        make_model("logit_l2", CFG).fit(X, np.zeros(200, int), w, cv_for(t0, t1))
    y_blocks = np.r_[np.zeros(150, int), np.ones(50, int)]  # the last fold holds every positive
    with pytest.raises(ZooFitError, match="inner split 3"):
        make_model("logit_l2", CFG).fit(X, y_blocks, w, cv_for(t0, t1))


def test_all_nan_column_dropped_other_nan_raises():
    X, y, w, t0, t1 = planted(300)
    Xn = np.c_[X, np.full(300, np.nan)]  # a rule primary's clf_prob
    m = make_model("logit_l2", CFG).fit(Xn, y, w, cv_for(t0, t1))
    ref = make_model("logit_l2", CFG).fit(X, y, w, cv_for(t0, t1))
    np.testing.assert_allclose(m.predict_proba(Xn), ref.predict_proba(X))
    Xn[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN in 1 rows"):
        m.predict_proba(Xn)
    with pytest.raises(ValueError, match="feature columns"):
        m.predict_proba(X)


# ── HP search and calibration stay inside the fitting rows ───────────────────


def test_hp_search_sees_only_the_fitting_rows_and_picks_the_best(monkeypatch):
    """Every estimator fit during HP search / refit gets a subset of the rows passed to fit, and the grid point
    with the best purged-CV score wins (ties → first)."""
    X, y, w, t0, t1 = planted(600)
    seen = []
    orig = ScaledLogit.fit

    def spy(self, Xf, yf, sample_weight=None):
        seen.append(Xf.copy())
        return orig(self, Xf, yf, sample_weight)

    monkeypatch.setattr(ScaledLogit, "fit", spy)
    monkeypatch.setattr(zoo.LogitL2, "grid", {"C": [1e-6, 1.0]})  # 1e-6 ≈ a constant predictor
    m = make_model("logit_l2", CFG).fit(X[:400], y[:400], w[:400], cv_for(t0[:400], t1[:400]))
    fit_rows = {r.tobytes() for r in X[:400]}
    assert len(seen) == 2 * 4 + 1 and all({r.tobytes() for r in s} <= fit_rows for s in seen)
    assert m.best_params_ == {"C": 1.0}
    assert list(m.cv_results_["score"]) == sorted(m.cv_results_["score"])  # 1.0 scored higher

    monkeypatch.setattr(zoo.LogitL2, "grid", {"C": [1.0, 1.0]})  # identical fits → tie → first
    tie = make_model("logit_l2", CFG).fit(X[:400], y[:400], w[:400], cv_for(t0[:400], t1[:400]))
    assert tie.cv_results_["score"].nunique() == 1 and tie.best_index_ == 0


def test_calibration_is_fit_on_out_of_fold_predictions(monkeypatch):
    """On pure-noise labels a memorizing forest's in-sample probabilities are near 0/1; a calibrator fit on its
    in-sample output would pass that through. Calibrated on OOF predictions, new-data output stays at the base rate."""
    monkeypatch.setattr(zoo.ExtraTrees, "grid", {"min_weight_fraction_leaf": [0.0]})  # fully grown trees
    rng = np.random.default_rng(3)
    n = 800
    X, y = rng.normal(size=(n, 6)), (rng.random(n) < 0.4).astype(int)
    t0 = np.arange(n) * 2
    m = make_model("extra_trees", CFG).fit(X, y, np.ones(n), cv_for(t0, t0 + 1))
    raw_in = m.predict_raw(X)
    assert np.abs(raw_in - y).mean() < 0.3  # the forest memorizes its own rows
    p = m.predict_proba(rng.normal(size=(2000, 6)))
    assert abs(p.mean() - 0.4) < 0.05 and p.std() < 0.08
    np.testing.assert_allclose(m.cal.predict(m.oof_raw_).mean(), 0.4, atol=0.03)


def test_calibration_method_is_the_cv_brier_argmin():
    """A step-shaped score → isotonic; each model records both CV Briers and picks the lower."""
    rng = np.random.default_rng(4)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = (rng.random(n) < np.where(x > 0.3, 0.9, 0.2)).astype(int)
    t0 = np.arange(n) * 2
    m = make_model("logit_l2", CFG).fit(x.reshape(-1, 1), y, np.ones(n), cv_for(t0, t0 + 1))
    assert m.calibration_ == "isotonic"
    assert m.calibration_brier_["isotonic"] < m.calibration_brier_["sigmoid"]
    X, y2, w, t0, t1 = planted()
    for name in TUNED:
        mm = make_model(name, CFG).fit(X, y2, w, cv_for(t0, t1))
        assert mm.calibration_ == min(("sigmoid", "isotonic"), key=lambda k: mm.calibration_brier_[k])


# ── Linear models: scaling and weights ───────────────────────────────────────


def test_scaled_logit_scales_inside_fit_with_weights():
    X, y, w, *_ = planted(500)
    m = ScaledLogit(C=0.1).fit(X, y, w)
    np.testing.assert_allclose(m.scaler_.mean_, np.average(X, axis=0, weights=w))
    shifted = ScaledLogit(C=0.1).fit(X * 1000 + 5, y, w)  # scale-free: the scaler is part of the model
    np.testing.assert_allclose(shifted.predict_proba(X * 1000 + 5), m.predict_proba(X), atol=1e-6)


@pytest.mark.parametrize("l1", [0.0, 1.0])
def test_scaled_logit_routes_sample_weight(l1):
    """Zero weight on half the rows == fitting on the other half."""
    X, y, *_ = planted(600)
    w = np.r_[np.ones(300), np.zeros(300)]
    a = ScaledLogit(C=0.5, l1_ratio=l1, random_state=0).fit(X, y, w)
    b = ScaledLogit(C=0.5, l1_ratio=l1, random_state=0).fit(X[:300], y[:300])
    np.testing.assert_allclose(a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1], atol=1e-3)


def test_rf_ldp_bootstrap_size_is_average_uniqueness():
    w = planted(300)[2]
    est = make_model("rf_ldp", CFG).estimator({"max_features": 1}, w)
    assert est.max_samples == pytest.approx(w.mean()) and est.class_weight == "balanced_subsample"
    assert est.min_weight_fraction_leaf == 0.05 and est.n_estimators == 500


# ── Meta-model and WFO wiring ────────────────────────────────────────────────


def test_fit_meta_model_skips_on_inner_single_class_and_needs_spans():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)), index=idx)
    prim = pd.DataFrame({"signed_dir": 1}, index=idx)
    lbl = pd.Series(np.r_[np.zeros(150, int), np.ones(50, int)], index=idx)
    w = pd.Series(1.0, index=idx)
    spans = pd.DataFrame({"entry_pos": np.arange(200) * 3 + 1, "exit_pos": np.arange(200) * 3 + 4}, index=idx)
    cfg = CFG.replace(META_MODEL="logit_l2")
    assert fit_meta_model(X, prim, lbl, w, cfg, spans) is None  # counted skip (the WFO records the fold)
    with pytest.raises(ValueError, match="needs the label spans"):
        fit_meta_model(X, prim, lbl.sample(frac=1, random_state=0).set_axis(idx), w, cfg)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_daily(1400)


@pytest.mark.parametrize(
    "over",
    [
        {"META_MODEL": "logit_l1"},
        {"META_MODEL": "lightgbm", "META_TRAIN": "oof"},
        {"PRIMARY": "ml_xgb", "PRIMARY_MODEL": "logit_l2", "META_MODEL": "rf_ldp"},
        {"PRIMARY": "ml_xgb", "META_MODEL": "xgb", "META_TRAIN": "oof"},
    ],
)
def test_wfo_runs_with_zoo_models_and_is_deterministic(daily, over):
    small = {k: {**getattr(CFG, k), "n_estimators": 50} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
    cfg = RunConfig.for_timeframe("1Day", **{"PRIMARY": "wavelet_trend", **WFO_CFG, **small, **over})
    a = run_wfo(daily, cfg)
    assert list(a.columns) == WFO_COLUMNS and a["fold"].nunique() == 3
    if not a.attrs["meta_skipped_folds"]:
        assert ((a["meta_prob"] >= PROB_CLIP) & (a["meta_prob"] <= 1 - PROB_CLIP)).all()
    pd.testing.assert_frame_equal(a, run_wfo(daily, cfg))  # deterministic given the seed


def test_wfo_with_zoo_models_is_causal(daily):
    """Perturbing every bar after an OOS event leaves the WFO rows up to it unchanged (tuned meta, OOF training)."""
    cfg = RunConfig.for_timeframe(
        "1Day", PRIMARY="donchian_breakout", META_MODEL="logit_l2", META_TRAIN="oof", **WFO_CFG
    )
    folds = wfo_folds(daily.index, cfg)
    ev = daily.index[folds[-1].val_end + 30]
    c = daily.index.get_loc(ev)
    a = run_wfo(daily, cfg)
    b = run_wfo(random_walk_after(daily, c, seed=5), cfg)
    pd.testing.assert_frame_equal(a.loc[:ev], b.loc[:ev])
    assert len(a.loc[:ev]) > len(a) / 2


def test_oof_primary_never_signals_an_event_it_was_fit_on(daily, monkeypatch):
    """META_TRAIN="oof": each fitting event's primary frame comes from a primary fit on a purged train split that
    excludes it (and every span overlapping it); the refit primary for test sees all fitting events."""
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="ml_xgb", META_TRAIN="oof", **WFO_CFG)
    calls = []
    cls = type(make_primary(cfg))
    fit, sig = cls.fit, cls.signal

    def spy_fit(self, df, X, labels, weights, cfg, *, val=None):
        calls.append(("fit", id(self), labels[["entry_pos", "exit_pos"]].copy(), len(df)))
        return fit(self, df, X, labels, weights, cfg, val=val)

    def spy_sig(self, df, X, cfg):
        calls.append(("signal", id(self), X.index, len(df)))
        return sig(self, df, X, cfg)

    monkeypatch.setattr(cls, "fit", spy_fit)
    monkeypatch.setattr(cls, "signal", spy_sig)
    frames = {}
    orig = wfo_engine.oof_primary

    def spy_oof(df, X, labels, weights, cfg, select=None, fit_start=0):
        frames["labels"] = labels
        return orig(df, X, labels, weights, cfg, select=select, fit_start=fit_start)

    monkeypatch.setattr(wfo_engine, "oof_primary", spy_oof)
    daily3 = daily.iloc[: wfo_folds(daily.index, cfg)[0].test_end]
    run_wfo(daily3, cfg)

    f = wfo_folds(daily3.index, cfg)[0]
    fit_rows = frames["labels"]
    assert fit_rows.equals(purged(fit_rows, 0, f.val_end, f.val_embargo))  # purged at the val end
    fits, oof_calls = {}, []
    for kind, i, obj, n in calls:  # in call order: an object id may be reused after the previous one is freed
        if kind == "fit":
            fits[i] = obj
        elif obj.isin(fit_rows.index).all():
            oof_calls.append((fits[i], obj, n))
    assert len(oof_calls) == cfg.ZOO_CV_SPLITS
    covered = pd.Index([])
    for tr, idx, n in oof_calls:
        te = fit_rows.loc[idx]
        assert n == f.val_end  # bars up to the fitting split's end only
        assert not tr.index.isin(idx).any()
        # purge: no train span overlaps the test block's hull
        a, b = te["entry_pos"].min() - 1, te["exit_pos"].max()
        assert ((tr["exit_pos"] < a) | (tr["entry_pos"] - 1 > b)).all()
        covered = covered.append(idx)
    assert covered.sort_values().equals(fit_rows.index)
    final = [lab for kind, i, lab, n in calls if kind == "fit"][-1]
    assert final.index.equals(fit_rows.index)


def test_oof_rule_primary_frame_is_the_plain_signal(daily):
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="sma_cross", META_TRAIN="oof", **WFO_CFG)
    lab = triple_barrier_labels(daily, sample_events(daily, cfg), cfg).iloc[100:400]
    X = pd.DataFrame(index=lab.index)
    got = wfo_engine.oof_primary(daily, X, lab, average_uniqueness(lab, len(daily)), cfg)
    pd.testing.assert_frame_equal(got, make_primary(cfg).signal(daily, X, cfg))


# ── OOS scoring ──────────────────────────────────────────────────────────────


def test_calibration_table_hand_example():
    p = pd.Series([0.05, 0.15, 0.12, 0.95, 0.91, 1.0])
    y = pd.Series([0, 0, 1, 1, 1, 0])
    t = calibration_table(y, p, n_bins=10)
    assert list(t["n"]) == [1, 2, 3]
    np.testing.assert_allclose(t["lo"], [0.0, 0.1, 0.9])
    np.testing.assert_allclose(t["mean_p"], [0.05, 0.135, (0.95 + 0.91 + 1.0) / 3])
    np.testing.assert_allclose(t["observed"], [0.0, 0.5, 2 / 3])


def test_signal_diagnostics_exclude_skipped_folds(daily):
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="wavelet_trend", **WFO_CFG)
    sig = run_wfo(daily, cfg)
    full = signal_diagnostics(daily, sig, cfg)
    sig.attrs["meta_skipped_folds"] = [int(sig["fold"].iloc[0])]
    o = meta_outcomes(daily, sig, cfg)
    part = signal_diagnostics(daily, sig, cfg)
    assert part["Meta-scored events"] == o["scored"].sum() < full["Meta-scored events"]
    s = o[o["scored"]]
    assert part["Meta Brier"] == pytest.approx(((s["meta_prob"] - s["success"]) ** 2).mean())


# ── Review fixes: weights, calibration choice, clipping, cmda in OOF, primary skip ──


def test_selection_calibration_and_refit_are_weighted_and_cross_fitted():
    """Recompute every stored number independently: per-grid-point weighted purged-CV scores, the calibrators'
    cross-fitted weighted Briers (fit on train OOF p, scored on test OOF p), the weighted refit and calibrator."""
    from validation.scoring import brier, purged_cv_predict, selection_score

    X, y, _, t0, t1 = planted(500, seed=7)
    w = np.random.default_rng(1).uniform(0.05, 1.0, 500) ** 3  # strongly unequal weights
    cv = cv_for(t0, t1)
    m = make_model("logit_l2", CFG).fit(X, y, w, cv)
    splits = list(cv.split(X))
    for (_, row), params in zip(m.cv_results_.iterrows(), m.cv_results_["params"], strict=True):
        est = m.estimator(params, w)
        want = np.mean(
            [
                selection_score(CFG.SELECTION_METRIC, y[f.test], f.proba, w[f.test])
                for f in purged_cv_predict(est, X, y, cv.splitter, cv.t0, cv.t1, w)
            ]
        )
        assert row["score"] == pytest.approx(want, abs=1e-12)
    oof = m.oof_raw_
    for method, got in m.calibration_brier_.items():
        want = np.mean(
            [
                brier(y[te], zoo._clip(zoo._CALIBRATORS[method]().fit(oof[tr], y[tr], w[tr]).predict(oof[te])), w[te])
                for tr, te in splits
            ]
        )
        assert got == pytest.approx(want, abs=1e-12)
    ref = m.estimator(m.best_params_, w).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(m.predict_raw(X), zoo.proba(ref, X), atol=1e-9)
    cal = zoo._CALIBRATORS[m.calibration_]().fit(oof, y, w)
    np.testing.assert_allclose(m.predict_proba(X), zoo._clip(cal.predict(m.predict_raw(X))), atol=1e-12)
    unweighted = zoo._CALIBRATORS[m.calibration_]().fit(oof, y, None)
    assert not np.allclose(cal.predict(oof), unweighted.predict(oof), atol=1e-4)  # the weights matter here


def test_calibrated_output_is_clipped(monkeypatch):
    """Separable data: isotonic calibration maps to exact 0/1, which the output clips to PROB_CLIP."""
    monkeypatch.setattr(zoo, "CALIBRATIONS", ("isotonic",))
    n = 400
    x = np.linspace(-1, 1, n)
    y = (x > 0).astype(int)
    t0 = np.arange(n) * 2
    m = make_model("logit_l2", CFG).fit(x.reshape(-1, 1), y, np.ones(n), cv_for(t0, t0 + 1))
    p = m.predict_proba(np.array([[-1.0], [1.0]]))
    assert m.cal.predict(m.predict_raw(np.array([[-1.0]])))[0] < PROB_CLIP  # unclipped would be ~0
    np.testing.assert_allclose(p, [PROB_CLIP, 1 - PROB_CLIP])


def test_oof_feature_selection_never_sees_the_scored_split(daily, monkeypatch):
    """META_TRAIN="oof" + cmda: every OOF split's selection runs on that split's train events only; the final
    selection (for the refit primary, meta and test) runs on all fitting events."""
    cfg = RunConfig.for_timeframe(
        "1Day", PRIMARY="ml_xgb", META_TRAIN="oof", FEATURE_SELECTION="cmda", CMDA_TREES=20, **WFO_CFG
    )
    cfg = cfg.replace(
        **{k: {**getattr(cfg, k), "n_estimators": 30} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
    )
    log = []
    sel = wfo_engine.select_features

    def spy_sel(X, lab, w, cfg, n):
        log.append(("select", lab.index))
        return sel(X, lab, w, cfg, n)

    cls = type(make_primary(cfg))
    sig = cls.signal

    def spy_sig(self, df, X, cfg):
        log.append(("signal", X.index, list(X.columns)))
        return sig(self, df, X, cfg)

    monkeypatch.setattr(wfo_engine, "select_features", spy_sel)
    monkeypatch.setattr(cls, "signal", spy_sig)
    run_wfo(daily.iloc[: wfo_folds(daily.index, cfg)[0].test_end], cfg)
    selects = [e for e in log if e[0] == "select"]
    assert len(selects) == cfg.ZOO_CV_SPLITS + 1  # one per OOF split + the final one
    for i in range(cfg.ZOO_CV_SPLITS):  # select(train) → signal(test), in order
        (_, train_idx), (_, test_idx, _) = log[2 * i], log[2 * i + 1]
        assert not train_idx.isin(test_idx).any()
    assert selects[-1][1].isin(pd.Index(np.concatenate([e[1] for e in log[1 : 2 * cfg.ZOO_CV_SPLITS : 2]]))).all()


def test_primary_fit_failure_skips_the_fold_and_is_recorded(daily, monkeypatch):
    import primaries.ml_xgb as ml

    cfg = RunConfig.for_timeframe("1Day", PRIMARY="ml_xgb", **WFO_CFG)
    fit = ml.fit_primary_classifier
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ZooFitError("single class")
        return fit(*args)

    monkeypatch.setattr(ml, "fit_primary_classifier", flaky)
    sig = run_wfo(daily, cfg)
    assert sig.attrs["primary_skipped_folds"] == [2] and sorted(sig["fold"].unique()) == [1, 3]


def test_cli_writes_skipped_folds_sidecar(daily, tmp_path, monkeypatch):
    import json

    import wavelet_meta_model as app

    monkeypatch.setattr(app, "make_synthetic_spy", lambda n: daily)
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="sma_cross", **WFO_CFG)
    monkeypatch.setattr(app, "plot_results", lambda *a, **k: None)
    app.main(cfg=cfg, out_dir=str(tmp_path))
    run = json.loads((tmp_path / "wfo_run.json").read_text())
    assert run == {"meta_skipped_folds": [], "primary_skipped_folds": [], "sizer_skipped_folds": []}
