"""
U12 (PLAN2, SPEC §11.2): harness speed — fixed zoo hyper-parameters, rolling / no calibration, OOF reuse for the
sizers, PWFO combo averaging, the flat runner pool, the compiled CUSUM recursion — and the parity fixture that pins
the old behaviour.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.runner as R
import wfo.wfo_engine as eng
from experiments import ledger as L
from experiments.spec import expand, load_spec
from features.groups import cusum_state
from features.triple_barrier_labels import bar_volatility, cusum_events
from models import zoo
from models.meta_model import fit_meta_model, oof_meta_prob
from models.zoo import PROB_CLIP, REGISTRY, make_model
from sizing import REGISTRY as SIZERS
from tests.test_experiments import SMALL as RUN_SMALL
from tests.test_experiments import FakeSource, doc
from tests.test_runconfig import perturb_after, synthetic_daily
from tests.test_zoo import cv_for, planted
from utils.config import RunConfig
from utils.data_loader import load_ohlcv
from wfo.pwfo import Combo, check_grid, make_grid, run_combo, run_pwfo
from wfo.wfo_engine import CalHistory, prepare

ROOT = Path(__file__).resolve().parent.parent
OLD = {"ZOO_FIXED_PARAMS": {}, "CALIBRATION": "crossfit", "OOF_META": "refit", "PWFO_COMBINE": "nested"}
FIXED = {"logit_l2": {"C": 0.1}}
SMALL_XGB = {k: {**getattr(RunConfig(), k), "n_estimators": 50} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
# 1Day walk-forward small enough for synthetic data; a tuned meta-model on OOF rows
WCFG = RunConfig.for_timeframe(
    "1Day", PRIMARY="wavelet_trend", META_MODEL="logit_l2", META_TRAIN="oof", INITIAL_TRAIN=700, VAL=350, TEST=100,
    MIN_TRAIN_EVENTS=50, MIN_VAL_EVENTS=30, **SMALL_XGB, **OLD,
)  # fmt: skip


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_daily(1400)


# ── RunConfig ────────────────────────────────────────────────────────────────


def test_new_switches_are_validated():
    with pytest.raises(ValueError, match="not a zoo model with a grid"):
        RunConfig(ZOO_FIXED_PARAMS={"nope": {}})
    with pytest.raises(ValueError, match="not a zoo model with a grid"):
        RunConfig(ZOO_FIXED_PARAMS={"legacy": {}})
    with pytest.raises(ValueError, match="exactly"):
        RunConfig(ZOO_FIXED_PARAMS={"logit_l2": {"C": 0.1, "l1_ratio": 0.0}})
    for name, bad in (("CALIBRATION", "isotonic"), ("OOF_META", "x"), ("PWFO_COMBINE", "best")):
        with pytest.raises(ValueError, match=name):
            RunConfig(**{name: bad})
    fixed = {"logit_l2": {"C": 0.1}}
    cfg = RunConfig(ZOO_FIXED_PARAMS=fixed)
    fixed["logit_l2"]["C"] = 9.0  # the config holds its own copy
    assert REGISTRY["logit_l2"].param_grid(cfg) == [{"C": 0.1}]
    assert REGISTRY["logit_l2"].param_grid(RunConfig(ZOO_FIXED_PARAMS={})) == [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]


def test_train_fit_sizers_with_oof_reuse_get_the_oof_predictions_they_need(daily, monkeypatch):
    """A train-fit sizer (cfg.SIZER or an extra one) under OOF_META="reuse" makes every window's meta-model keep its
    purged-CV OOF predictions (need_oof), also when rolling calibration and fixed parameters need none."""
    cfg = WCFG.replace(ZOO_FIXED_PARAMS=FIXED, CALIBRATION="rolling", OOF_META="reuse")
    seen, real = [], eng.fit_meta_model
    monkeypatch.setattr(
        eng, "fit_meta_model", lambda *a, **k: seen.append((k["need_oof"], real(*a, **k))) or seen[-1][1]
    )
    sig = eng.run_wfo(daily, cfg, sizers=("ecdf",))
    assert "bet_size:ecdf" in sig and all(need for need, _ in seen)
    assert all(m.oof_raw_ is not None for _, m in seen if m is not None)
    assert any(m.calibration_.startswith("rolling") for _, m in seen if m is not None)
    seen.clear()
    eng.run_wfo(daily, cfg.replace(SIZER="ecdf"))
    assert seen and all(need for need, _ in seen)
    seen.clear()
    eng.run_wfo(daily, cfg)  # fixed sizer: no OOF needed, one fit per window
    assert seen and not any(need for need, _ in seen)
    assert all(m.oof_raw_ is None for _, m in seen if m is not None and m.calibration_.startswith("rolling"))


# ── Fixed hyper-parameters, calibration modes ────────────────────────────────


@pytest.fixture
def logit_fits(monkeypatch):
    n = []
    orig = zoo.ScaledLogit.fit

    def fit(self, *a, **k):
        n.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(zoo.ScaledLogit, "fit", fit)
    return n


@pytest.mark.parametrize(
    "fixed, calibration, need_oof, n_fits",
    [({}, "crossfit", False, 3 * 4 + 1), (FIXED, "crossfit", False, 4 + 1), (FIXED, "none", False, 1),
     ({}, "none", False, 3 * 4 + 1), (FIXED, "none", True, 4 + 1)],
)  # fmt: skip
def test_fits_per_window(logit_fits, fixed, calibration, need_oof, n_fits):
    X, y, w, t0, t1 = planted()
    cfg = RunConfig(ZOO_FIXED_PARAMS=fixed)
    m = make_model("logit_l2", cfg, calibration=calibration, need_oof=need_oof).fit(X, y, w, cv_for(t0, t1))
    assert len(logit_fits) == n_fits
    assert (m.oof_raw_ is None) == (n_fits == 1)
    if calibration == "none":
        assert m.calibration_ == "none" and m.calibration_brier_ == {}
        np.testing.assert_array_equal(m.predict_proba(X), np.clip(m.predict_raw(X), PROB_CLIP, 1 - PROB_CLIP))
    else:
        assert m.calibration_ in ("sigmoid", "isotonic")


def test_fixed_params_at_the_grid_winner_reproduce_the_search():
    X, y, w, t0, t1 = planted()
    cfg = RunConfig(ZOO_FIXED_PARAMS={})
    a = make_model("logit_l2", cfg).fit(X, y, w, cv_for(t0, t1))
    b = make_model("logit_l2", cfg.replace(ZOO_FIXED_PARAMS={"logit_l2": a.best_params_})).fit(X, y, w, cv_for(t0, t1))
    np.testing.assert_array_equal(a.oof_raw_, b.oof_raw_)
    assert a.calibration_ == b.calibration_
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def _meta_inputs(n=600):
    X, y, w, t0, t1 = planted(n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X_fit = pd.DataFrame(X[:, :4], index=idx, columns=[f"f{i}" for i in range(4)])
    prim = pd.DataFrame(X[:, 4:], index=idx, columns=["p0", "p1"])
    spans = pd.DataFrame({"entry_pos": t0 + 1, "exit_pos": t1}, index=idx)
    return X_fit, prim, pd.Series(y, index=idx), pd.Series(w, index=idx), spans


def test_oof_reuse_is_the_fitted_models_calibrated_oof_without_a_fit(logit_fits):
    X_fit, prim, y, w, spans = _meta_inputs()
    cfg = RunConfig.for_timeframe("1Day", META_MODEL="logit_l2", **{**OLD, "OOF_META": "reuse"})
    meta = fit_meta_model(X_fit, prim, y, w, cfg, spans)
    n = len(logit_fits)
    p = oof_meta_prob(X_fit, prim, y, w, cfg, spans, meta=meta)
    assert len(logit_fits) == n  # no fit
    np.testing.assert_array_equal(p.to_numpy(), np.clip(meta.cal.predict(meta.oof_raw_), PROB_CLIP, 1 - PROB_CLIP))
    assert p.index.equals(y.index)
    refit = oof_meta_prob(X_fit, prim, y, w, cfg.replace(OOF_META="refit"), spans, meta=meta)
    assert len(logit_fits) > n and refit.index.equals(y.index)  # the U7–U11 path refits per split


def test_oof_reuse_raises_without_oof_predictions():
    X_fit, prim, y, w, spans = _meta_inputs()
    cfg = RunConfig.for_timeframe("1Day", META_MODEL="logit_l2", ZOO_FIXED_PARAMS=FIXED, CALIBRATION="none",
                                  OOF_META="reuse")  # fmt: skip
    meta = fit_meta_model(X_fit, prim, y, w, cfg, spans)
    assert meta.oof_raw_ is None
    with pytest.raises(RuntimeError, match="no OOF predictions"):
        oof_meta_prob(X_fit, prim, y, w, cfg, spans, meta=meta)


def test_rolling_calibration_attaches_a_platt_map_on_the_pairs_or_cross_fits():
    X_fit, prim, y, w, spans = _meta_inputs()
    cfg = RunConfig.for_timeframe("1Day", META_MODEL="logit_l2", ZOO_FIXED_PARAMS=FIXED, CALIBRATION="rolling",
                                  MIN_VAL_EVENTS=50)  # fmt: skip
    rng = np.random.default_rng(3)
    pairs = pd.DataFrame({"raw": rng.uniform(0.2, 0.8, 80), "y": rng.integers(0, 2, 80), "w": rng.uniform(0.5, 1, 80)})
    m = fit_meta_model(X_fit, prim, y, w, cfg, spans, cal_pairs=pairs)
    assert m.calibration_ == "rolling (80 pairs)" and m.oof_raw_ is None  # one fit, calibrated from the past
    ref = zoo._Sigmoid().fit(pairs["raw"].to_numpy(), pairs["y"].to_numpy(), pairs["w"].to_numpy())
    Xm = pd.concat([X_fit, prim], axis=1)
    np.testing.assert_array_equal(
        m.predict_proba(Xm), np.clip(ref.predict(m.predict_raw(Xm)), PROB_CLIP, 1 - PROB_CLIP)
    )
    for few in (pairs.iloc[:49], pairs.assign(y=1), None):  # too few, one class, no history: cross-fit
        assert fit_meta_model(X_fit, prim, y, w, cfg, spans, cal_pairs=few).calibration_ in ("sigmoid", "isotonic")


def test_cal_history_hands_out_only_resolved_pairs_inside_the_fit_span():
    bars = np.arange(10, 110, 10)  # events at bars 10 … 100, entered next bar, exit 5 bars later
    idx = pd.date_range("2020-01-01", periods=len(bars), freq="D")
    labels = pd.DataFrame({"entry_pos": bars + 1, "exit_pos": bars + 5}, index=idx)
    h = CalHistory()
    h.add(pd.Series(np.linspace(0.3, 0.7, 6), index=idx[:6]), pd.Series([0, 1, 0, 1, 1, 0], index=idx[:6]))
    h.add(pd.Series(np.linspace(0.4, 0.6, 4), index=idx[6:]), pd.Series([1, 0, 1, 0], index=idx[6:]))
    p = h.pairs(labels, fit_start=30, end=80, embargo=2, n_bars=200)
    # bar >= 30, bar < 80 and exit < 78: bars 30 … 70 (bar 70 exits at 75)
    assert list(p.index) == list(idx[2:7]) and set(p.columns) == {"raw", "y", "w"}
    assert p["y"].tolist() == [0, 1, 1, 0, 1] and (p["w"] > 0).all()
    assert h.pairs(labels, 30, 75, 0, 200).index.equals(idx[2:6])  # bar 70 exits at 75: not before 75
    h.add(pd.Series([0.5], index=idx[:1]), pd.Series([1], index=idx[:1]))
    with pytest.raises(RuntimeError, match="two windows"):
        h.pairs(labels, 0, 200, 0, 200)


def test_rolling_calibration_uses_only_earlier_resolved_oos_pairs(daily, monkeypatch):
    """Every pair a rolling window's calibrator sees: an earlier window's OOS event, inside the window's IS span, its
    barrier exit before the fit's val end minus the embargo."""
    cfg = WCFG.replace(ZOO_FIXED_PARAMS=FIXED, CALIBRATION="rolling")
    calls, orig = [], eng.CalHistory.pairs

    def spy(self, labels, fit_start, end, embargo, n_bars):
        p = orig(self, labels, fit_start, end, embargo, n_bars)
        known = pd.concat(self.raw).index if self.raw else pd.DatetimeIndex([])
        calls.append((fit_start, end, embargo, p, known))
        return p

    monkeypatch.setattr(eng.CalHistory, "pairs", spy)
    seen = []
    real = eng.fit_meta_model
    monkeypatch.setattr(eng, "fit_meta_model", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    prep = prepare(daily, cfg)
    run = run_combo(daily, cfg, prep, Combo(500, 50, 170))
    assert len(calls) == (run.windows["n_fit_events"] > 0).sum() >= 10  # one per window that fit a meta-model
    used = 0
    for fit_start, end, emb, p, known in calls:
        lab = prep.labels.loc[p.index]
        assert p.index.isin(known).all()
        assert ((lab["entry_pos"] - 1) >= fit_start).all() and (lab["exit_pos"] < end - emb).all()
        used += len(p) >= cfg.MIN_VAL_EVENTS
    cal = [m.calibration_ for m in seen if m is not None]
    assert cal[0] in ("sigmoid", "isotonic") and used > 0 and sum(c.startswith("rolling") for c in cal) == used


# ── PWFO combo averaging ─────────────────────────────────────────────────────

AVG_CFG = WCFG.replace(
    ZOO_FIXED_PARAMS=FIXED, CALIBRATION="rolling", PWFO_COMBINE="average", PWFO_IS_GRID=(500, 700),
    PWFO_OOS_GRID=(25, 50), PBO_BLOCKS=4,
)  # fmt: skip


@pytest.fixture(scope="module")
def avg_result(daily):
    return run_pwfo(daily, AVG_CFG)


def test_average_is_the_mean_of_the_combo_columns_on_their_common_span(avg_result):
    r = avg_result
    common = r.returns.dropna(axis=0, how="any")
    assert r.returns.shape[1] == 4 and len(common) < len(r.returns)  # the combos' OOS spans differ
    pd.testing.assert_series_equal(r.pwfo["ret"], common.mean(axis=1), check_names=False)
    assert r.pwfo.index[0] == max(r.returns[c].first_valid_index() for c in r.returns)
    assert r.pwfo.index[-1] == min(r.returns[c].last_valid_index() for c in r.returns)
    assert not r.pwfo["burn_in"].any() and (r.pwfo["combo"] == "average").all()
    assert r.choice.empty and r.stats["n_decisions"] == 0 and r.stats["picks"] == {}
    assert r.stats["n_live_days"] == len(common) and 0 <= r.stats["pbo"] <= 1 and r.stats["pbo_n_combos"] == 4
    assert r.stats["combine"] == "average" and r.stats["n_windows"] == r.summary["n_oos_windows"].sum()
    ok = r.windows["status"] == "ok"  # every fitted window says how its meta-model was calibrated
    assert r.windows.loc[ok, "calibration"].isin(["rolling", "sigmoid", "isotonic"]).all()
    assert r.stats["calibration_windows"] == r.windows.loc[ok, "calibration"].value_counts().to_dict()
    assert r.stats["calibration_windows"].get("rolling", 0) > 0
    assert 0 <= r.stats["pwfo_dsr"] <= r.stats["pwfo_psr0"] <= 1 and r.stats["dsr_n_trials"] == 4


def test_average_pwfo_with_rolling_calibration_is_causal(daily, avg_result):
    c = 1250
    other = run_pwfo(perturb_after(daily, c + 1), AVG_CFG)
    day = daily.index[c].normalize()
    a, b = avg_result.pwfo.loc[:day], other.pwfo.loc[:day]
    assert len(a) > 200
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(avg_result.returns.loc[:day], other.returns.loc[:day])


def test_nested_needs_its_default_combo_and_average_does_not():
    cfg = WCFG.replace(PWFO_IS_GRID=(500,), PWFO_OOS_GRID=(25,), PWFO_DEFAULT=(252, 10))
    with pytest.raises(ValueError, match="PWFO_DEFAULT"):
        check_grid(cfg, make_grid(cfg))
    check_grid(cfg.replace(PWFO_COMBINE="average"), make_grid(cfg))


# ── Flat runner pool ─────────────────────────────────────────────────────────


@pytest.fixture
def env(tmp_path):
    return {"ledger": L.Ledger(tmp_path / "ledger.jsonl"), "root": tmp_path / "exp", "feature_cache_dir": None,
            "holdout_marker": tmp_path / "holdout_marker.jsonl"}  # fmt: skip


@pytest.mark.parametrize("combine", ["nested", "average"])
def test_flat_pool_records_every_pwfo_cell_and_equals_serial(env, tmp_path, combine):
    """2 PWFO cells × 2 combos with jobs=4: four combo tasks in one pool, two `ok` rows, the serial rows and files."""
    pwfo = {"is_grid": [400, 600], "oos_grid": [100]}
    over = {**RUN_SMALL, "PWFO_DEFAULT": [400, 100], "SELECT_LOOKBACK": 60, "SELECT_EVERY": 20, "PBO_BLOCKS": 4,
            "PWFO_COMBINE": combine}  # fmt: skip
    spec = expand(doc(grid={"symbols": ["AAA", "BBB"]}, pwfo=pwfo, overrides=over))
    serial = R.run(spec, source=FakeSource(), **env)
    root2 = tmp_path / "e2"
    par = R.run(spec, source=FakeSource(), ledger=L.Ledger(tmp_path / "l2.jsonl"), root=root2, feature_cache_dir=None,
                holdout_marker=env["holdout_marker"], jobs=4)  # fmt: skip
    assert [r["status"] for r in par] == ["ok", "ok"] and all(r["kind"] == "pwfo" for r in par)
    keys = (
        "sharpe",
        "n_obs",
        "n_trades",
        "pbo",
        "pwfo_dsr",
        "n_trials",
        "n_oos_windows",
        "combos",
        "combine",
        "n_run_combos",
        "calibration_windows",
    )
    pick = lambda rs: {r["cell_hash"]: L.clean({k: r.get(k) for k in keys}) for r in rs}  # NaN → None
    assert pick(serial) == pick(par)
    for r in par:
        a, b = env["root"] / "cells" / r["cell_hash"], root2 / "cells" / r["cell_hash"]
        for f in ("daily_returns.csv", "returns_pwfo.csv", "pwfo_pwfo.csv", "windows_pwfo.csv"):
            assert (a / f).read_bytes() == (b / f).read_bytes(), f
        assert sorted(p.name for p in (b / "logs").iterdir()) == ["IS400_OOS100.log", "IS600_OOS100.log"]
        assert r["runtime_s"] > 0 and json.loads((b / "result.json").read_text())["status"] == "ok"


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_failing_combo_makes_its_cell_an_error_row(env, jobs):
    pwfo = {"is_grid": [400, 5000], "oos_grid": [100]}  # IS 5000 sessions: the data holds no window
    over = {**RUN_SMALL, "PWFO_COMBINE": "average"}
    rows = R.run(expand(doc(grid={"symbols": ["AAA", "BBB"]}, pwfo=pwfo, overrides=over)), source=FakeSource(),
                 jobs=jobs, **env)  # fmt: skip
    assert [r["status"] for r in rows] == ["error", "error"]
    for r in rows:
        assert "no full window" in Path(r["error_path"]).read_text(encoding="utf-8")
        assert "no full window" in r["error"] and "\n" not in r["error"]


# ── Compiled CUSUM ───────────────────────────────────────────────────────────


def _events_loop(close: pd.Series, threshold: pd.Series) -> pd.DatetimeIndex:
    """cusum_events before U12 (the reference)."""
    ret = np.log(close).diff().to_numpy()
    h = threshold.reindex(close.index).to_numpy()
    events = []
    s_pos = s_neg = 0.0
    for t in range(1, len(ret)):
        if np.isnan(h[t]) or np.isnan(ret[t]):
            continue
        s_pos = max(0.0, s_pos + ret[t])
        s_neg = min(0.0, s_neg + ret[t])
        if s_pos > h[t]:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -h[t]:
            s_neg = 0.0
            events.append(t)
    return close.index[events]


def _state_loop(close: pd.Series, threshold: pd.Series):
    """cusum_state before U12 (the reference)."""
    ret = np.log(close.astype(float)).diff().to_numpy()
    h = threshold.to_numpy()
    pos, neg = np.full(len(ret), np.nan), np.full(len(ret), np.nan)
    s_pos = s_neg = 0.0
    for t in range(1, len(ret)):
        if np.isnan(h[t]) or np.isnan(ret[t]) or h[t] == 0:
            continue
        s_pos = max(0.0, s_pos + ret[t])
        s_neg = min(0.0, s_neg + ret[t])
        pos[t], neg[t] = s_pos / h[t], s_neg / h[t]
        if s_pos > h[t]:
            s_pos = 0.0
        elif s_neg < -h[t]:
            s_neg = 0.0
    return pos, neg


def _series(seed: int, n: int = 20_000):
    """Fat-tailed returns with NaN gaps, a noisy threshold (its jumps let both sides cross on one bar), zeros."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(3, n) * 1e-3
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2020-01-01", periods=n, freq="5min"))
    close[rng.random(n) < 0.01] = np.nan
    h = pd.Series(2.5e-3 * np.exp(rng.normal(0, 0.6, n)), index=close.index)
    h[:100] = np.nan
    h[rng.random(n) < 0.005] = 0.0
    return close, h


@pytest.mark.parametrize("seed", range(10))
def test_compiled_cusum_equals_the_loop(seed):
    close, h = _series(seed)
    ev = cusum_events(close, h)
    assert len(ev) > 500 and ev.equals(_events_loop(close, h))
    for a, b in zip(cusum_state(close, h), _state_loop(close, h), strict=True):
        np.testing.assert_array_equal(a, b)


def test_compiled_cusum_equals_the_loop_on_spy_5min():
    close = load_ohlcv(str(ROOT / "data" / "data.csv"))["close"]
    h = 1.5 * bar_volatility(close, 100)
    ev = cusum_events(close, h)
    assert len(ev) > 1000 and ev.equals(_events_loop(close, h))
    for a, b in zip(cusum_state(close, h), _state_loop(close, h), strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_up_crossing_takes_the_bar_and_the_down_side_carries_on():
    # bar 3: S+ = 0.4 > h = 0.3 → event, S− = −0.5 < −0.3 is not reset (the loop's elif); bar 5: S− < −0.4 → event
    ret = [np.nan, 0.9, -0.5, 0.0, 0.0, 0.0]
    close = pd.Series(np.exp(np.nancumsum(ret)), index=pd.date_range("2020-01-01", periods=6, freq="D"))
    h = pd.Series([np.nan, 1.0, 1.0, 0.3, 1.0, 0.4], index=close.index)
    ev = cusum_events(close, h)
    assert list(ev) == list(close.index[[3, 5]]) and ev.equals(_events_loop(close, h))


# ── Parity fixture ───────────────────────────────────────────────────────────


def _parity_module():
    spec = importlib.util.spec_from_file_location("u12_parity", ROOT / "scripts" / "u12_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parity_spec_runs_the_fixture_cells_with_every_old_switch(tmp_path):
    P = _parity_module()
    fixture = json.loads(P.FIXTURE.read_text(encoding="utf-8"))["cells"]
    assert len(fixture) == 4 and len({c["label"] for c in fixture}) == 4
    for c in fixture:
        assert set(c["sha256"]) == {"daily_returns.csv", "returns_pwfo.csv"}
        assert all(len(h) == 64 for h in c["sha256"].values()) and c["stage"] == "C"
    P.spec(tmp_path / "parity.yaml", legacy=False)
    _, cells = load_spec(tmp_path / "parity.yaml")
    assert [c.label() for c in cells] == [c["label"] for c in fixture]
    cfgs = [c.config() for c in cells]
    for cell, cfg, fx in zip(cells, cfgs, fixture, strict=True):
        assert {k: getattr(cfg, k) for k in OLD} == OLD
        rest = {k: v for k, v in cell.spec["overrides"].items() if k not in OLD}
        assert {**cell.spec, "overrides": rest} == fx["spec"]
        assert cell.is_pwfo and cfg.PWFO_COMBINE == "nested"
    # the cells exercise the old paths the switches replace: grid searches of 3, 2 and 1 points, the per-split OOF
    # meta refit of a train-fit sizer, and cmda
    assert {len(REGISTRY[cfg.META_MODEL].param_grid(cfg)) for cfg in cfgs} == {1, 2, 3}
    assert any(SIZERS[cfg.SIZER].needs_train for cfg in cfgs)
    assert any(cfg.FEATURE_SELECTION == "cmda" for cfg in cfgs) and any(cfg.PRIMARY == "ml_xgb" for cfg in cfgs)
