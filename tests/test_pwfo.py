"""
U9: Power Walk-Forward — window generator, legacy WFO as one combo, rolling-window isolation, per-combo statistics,
nested selection without look-ahead, and the full PWFO (PBO, DSR, causality).
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

import wfo.wfo_engine as eng
from data.bars import HoldoutError
from tests.test_runconfig import perturb_after, synthetic_daily
from utils.config import RunConfig
from wfo.pwfo import (
    Combo,
    combo_stats,
    make_grid,
    nested_select,
    pwfo_windows,
    run_combo,
    run_pwfo,
    stitch,
    unit_bounds,
)
from wfo.wfo_engine import prepare, purged, run_wfo, wfo_folds

SMALL = {k: {**getattr(RunConfig(), k), "n_estimators": 50} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
CFG = RunConfig.for_timeframe(
    "1Day",
    PRIMARY="wavelet_trend",
    INITIAL_TRAIN=700,
    VAL=350,
    TEST=100,
    MIN_TRAIN_EVENTS=50,
    MIN_VAL_EVENTS=30,
    COST_MODEL="slippage",
    **SMALL,
)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_daily(1400)


def intraday_index(days: list[str], half: set[str] = frozenset(), per_day: int = 7) -> pd.DatetimeIndex:
    """Hourly-style bars (per_day full sessions, 3 on half-days) on the given NY dates."""
    stamps = []
    for d in days:
        n = 3 if d in half else per_day
        stamps += list(pd.date_range(f"{d} 09:30", periods=n, freq="h", tz="America/New_York"))
    return pd.DatetimeIndex(stamps)


# ── Window generator ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize("combo", [Combo(20, 5, 7), Combo(30, 10, 10), Combo(12, 1, 4)])
def test_windows_do_not_overlap_tile_and_respect_the_embargo(combo, expanding):
    cal = pd.bdate_range("2024-01-02", periods=90)
    half = {str(cal[17].date()), str(cal[40].date())}
    dropped = {str(cal[25].date())}  # an exchange session the data layer dropped
    days = [str(d.date()) for d in cal if str(d.date()) not in dropped]
    idx = intraday_index(days, half)
    bounds = unit_bounds(idx, "days", cal)
    assert len(bounds) == len(cal) + 1 and bounds[26] == bounds[25]  # the dropped session holds no bars
    wins = pwfo_windows(idx, combo, embargo=2, expanding=expanding, sessions=cal)
    n_full = (len(cal) - combo.is_len) // combo.oos_len
    assert len(wins) == n_full
    for k, w in enumerate(wins):
        b = combo.is_len + k * combo.oos_len
        assert w.is_start == (0 if expanding else bounds[k * combo.oos_len])
        # IS = [is_start, val_end), OOS = [val_end, oos_end); an OOS of only the dropped session holds no bars
        assert w.is_start <= w.train_end < w.val_end <= w.oos_end
        assert w.val_end < w.oos_end or (combo.oos_len == 1 and b == 25)
        assert w.train_end == bounds[b - combo.val_len] and w.val_end == bounds[b]
        assert w.oos_end == bounds[b + combo.oos_len]
        # every boundary is a session's first bar
        for p in (w.is_start, w.train_end, w.val_end):
            assert p == 0 or idx[p].normalize() != idx[p - 1].normalize()
        # embargo = the bars of the 2 calendar sessions before each split end (shorter across a half-day / hole)
        assert w.train_embargo == w.train_end - bounds[b - combo.val_len - 2]
        assert w.val_embargo == w.val_end - bounds[b - 2]
    for a, b in pairwise(wins):
        assert b.val_end == a.oos_end  # OOS windows tile the span contiguously
    assert wins[0].val_end == bounds[combo.is_len] and wins[-1].oos_end == bounds[combo.is_len + n_full * combo.oos_len]


def test_half_day_counts_as_one_session_and_purge_respects_the_embargo():
    cal = pd.bdate_range("2024-01-02", periods=12)
    half = {str(cal[5].date())}
    idx = intraday_index([str(d.date()) for d in cal], half)
    (w,) = pwfo_windows(idx, Combo(6, 6, 2), embargo=1, sessions=cal)[:1]
    assert w.val_end == 5 * 7 + 3  # sessions 0–4 full, session 5 is the half-day
    assert w.val_embargo == 3  # the last IS session is the half-day: 3 bars
    labels = pd.DataFrame({"entry_pos": np.arange(1, len(idx)), "exit_pos": np.arange(1, len(idx)) + 2})
    kept = purged(labels, w.train_end, w.val_end, w.val_embargo)
    assert (kept["exit_pos"] < w.val_end - w.val_embargo).all() and len(kept)


def test_window_inputs_are_validated():
    idx = intraday_index([str(d.date()) for d in pd.bdate_range("2024-01-02", periods=30)])
    with pytest.raises(ValueError, match="val < IS"):
        pwfo_windows(idx, Combo(10, 5, 10), embargo=0)
    with pytest.raises(ValueError, match="embargo"):
        pwfo_windows(idx, Combo(10, 5, 2), embargo=2)
    with pytest.raises(ValueError, match="not exchange sessions"):
        unit_bounds(idx, "days", pd.bdate_range("2024-01-03", periods=30))
    assert pwfo_windows(idx, Combo(28, 5, 7), embargo=1) == []  # no full OOS window


@pytest.mark.parametrize("unit", ["days", "bars"])
def test_legacy_folds_are_one_expanding_combo(daily, unit):
    cfg = CFG if unit == "days" else CFG.replace(WINDOW_UNIT="bars", EMBARGO=3)
    folds = wfo_folds(daily.index, cfg)
    wins = pwfo_windows(
        daily.index,
        Combo(cfg.INITIAL_TRAIN + cfg.VAL, cfg.TEST, cfg.VAL),
        embargo=cfg.EMBARGO,
        expanding=True,
        unit=unit,
    )
    assert [(w.train_end, w.val_end, w.oos_end, w.train_embargo, w.val_embargo) for w in wins] == [
        tuple(f) for f in folds
    ]
    assert all(w.is_start == 0 for w in wins)


def test_grid_defaults():
    cfg = RunConfig.for_timeframe("1Hour")  # SPEC §11.2 (U12)
    grid = make_grid(cfg)
    assert [(c.is_len, c.oos_len) for c in grid] == [(504, 21), (756, 21)] and cfg.PWFO_DEFAULT == (504, 21)
    assert {c.is_len: c.val_len for c in grid} == {504: 168, 756: 252}  # VAL/(TRAIN+VAL) = 1/3
    daily = RunConfig.for_timeframe("1Day")
    assert [(c.is_len, c.oos_len) for c in make_grid(daily)] == [(1260, 21), (1512, 21)]
    assert daily.PWFO_DEFAULT == (1260, 21) and daily.PWFO_COMBINE == "average"
    u11 = cfg.replace(PWFO_IS_GRID=(63, 126, 252, 504), PWFO_OOS_GRID=(5, 10, 21, 63))  # the U9–U11 grid
    assert {c.is_len: c.val_len for c in make_grid(u11)} == {63: 21, 126: 42, 252: 84, 504: 168}
    with pytest.raises(ValueError):
        cfg.replace(PWFO_VAL_FRAC=1.0)


# ── One combo ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("unit", ["days", "bars"])
def test_legacy_wfo_is_reproduced_as_a_single_combo(daily, unit):
    cfg = CFG if unit == "days" else CFG.replace(WINDOW_UNIT="bars", EMBARGO=3)
    ref = run_wfo(daily, cfg)
    run = run_combo(
        daily,
        cfg.replace(PWFO_EXPANDING=True),
        prepare(daily, cfg),
        Combo(cfg.INITIAL_TRAIN + cfg.VAL, cfg.TEST, cfg.VAL),
        unit=unit,
    )
    pd.testing.assert_frame_equal(run.signals, ref.drop(columns=[]), check_exact=True)
    assert (run.windows["status"] == "ok").all()


@pytest.mark.parametrize("meta_train", ["val", "oof"])
def test_rolling_window_fits_only_on_its_own_window(daily, meta_train, monkeypatch):
    cfg = CFG.replace(META_TRAIN=meta_train)
    seen = {}
    real_fit, real_fset_fit = eng.fit_meta_model, eng.FeatureSet.fit

    def spy_meta(X, *a, **k):
        seen.setdefault("meta", []).append(X.index)
        return real_fit(X, *a, **k)

    def spy_fset(self, train_df):
        seen.setdefault("fset", []).append(train_df.index)
        return real_fset_fit(self, train_df)

    monkeypatch.setattr(eng, "fit_meta_model", spy_meta)
    monkeypatch.setattr(eng.FeatureSet, "fit", spy_fset)
    combo = Combo(500, 100, 170)
    run = run_combo(daily, cfg, prepare(daily, cfg), combo)
    wins = pwfo_windows(daily.index, combo, embargo=cfg.EMBARGO)
    assert len(seen["meta"]) == len(wins) == (run.windows["status"] == "ok").sum()
    for w, meta_ix, fset_ix in zip(wins, seen["meta"], seen["fset"], strict=True):
        lo, hi = daily.index[w.is_start], daily.index[w.val_end]
        assert meta_ix.min() >= lo and meta_ix.max() < hi
        assert fset_ix[0] == lo and fset_ix[-1] < daily.index[w.train_end - w.train_embargo]
        ins = run.in_sample[w.w]
        assert ins.index.min() >= lo and ins.index.max() < hi
        oos = run.signals[run.signals["fold"] == w.w + 1]
        assert oos.index.min() >= hi and oos.index.max() <= daily.index[w.oos_end - 1]


def test_combo_stats_are_consistent(daily):
    combo = Combo(500, 100, 170)
    run = run_combo(daily, CFG, prepare(daily, CFG), combo)
    st = combo_stats(daily, run, CFG)
    s, win = st.summary, st.windows
    assert s["n_oos_windows"] == len(win) == 9 and s["weak"]  # < PWFO_MIN_WINDOWS
    assert s["n_oos_days"] == len(st.oos_daily) == 900  # the windows tile the stitched stream
    np.testing.assert_allclose((1 + win["oos_ret"]).prod(), (1 + st.oos_daily).prod(), rtol=1e-12)
    assert win["is_ret_ann"].notna().all() and win["is_sharpe"].notna().all()
    mean_is = win["is_ret_ann"].mean()
    t = mean_is / win["is_ret_ann"].std(ddof=1) * np.sqrt(len(win))
    assert s["is_ret_ann_mean"] == pytest.approx(mean_is) and s["is_ret_t"] == pytest.approx(t)
    assert s["n_is_nonpos"] == (win["is_ret_ann"] <= 0).sum()
    if mean_is > 0 and t >= CFG.PWFO_WFE_MIN_T:
        assert s["wfe"] == pytest.approx(s["oos_ret_ann"] / mean_is)
    else:
        assert np.isnan(s["wfe"])
    assert s["pct_profitable_oos"] == pytest.approx((win["oos_ret"] > 0).mean())
    traded = win["n_oos_trades"] > 0
    assert s["pct_windows_traded"] == pytest.approx(traded.mean())
    assert s["pct_profitable_traded"] == pytest.approx((win.loc[traded, "oos_ret"] > 0).mean())


def test_wfe_needs_a_significantly_positive_in_sample():
    from wfo.pwfo import wfe

    assert wfe(0.1, [0.2, 0.2, 0.2], 2.0)[0] == pytest.approx(0.5)  # zero spread: t = inf
    assert wfe(-0.1, [0.19, 0.2, 0.21], 2.0)[0] == pytest.approx(-0.5)
    assert np.isnan(wfe(0.1, [0.0, 0.0], 2.0)[0]) and np.isnan(wfe(0.1, [-0.2, -0.1], 2.0)[0])
    assert np.isnan(wfe(0.1, [np.nan], 2.0)[0])
    # review regression: a positive IS mean indistinguishable from 0 (SPY IS504_OOS5: 0.0029, t ≈ 0.85) gave −6.5
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.067, 389)
    noise += 0.0029 - noise.mean()
    w, t = wfe(-0.019, noise, 2.0)
    assert np.isnan(w) and 0 < t < 2
    assert wfe(-0.019, noise, 0.0)[0] == pytest.approx(-0.019 / 0.0029)


def test_unfitted_windows_count_flat_in_sample_and_weak_counts_fitted_windows(daily, monkeypatch):
    """Review regressions: the WFE's IS mean covers the same windows as its OOS (skipped = 0), `weak` counts fitted."""
    combo = Combo(500, 100, 170)
    real = eng.fit_window

    def skip_some(df, cfg, prep, fold, *a, **k):
        if fold % 3 == 0:
            return eng.WindowFit("insufficient_events", None, None, False, (), 0)
        return real(df, cfg, prep, fold, *a, **k)

    monkeypatch.setattr("wfo.pwfo.fit_window", skip_some)
    cfg = CFG.replace(PWFO_MIN_WINDOWS=8)
    run = run_combo(daily, cfg, prepare(daily, cfg), combo)
    st = combo_stats(daily, run, cfg)
    win, s = st.windows, st.summary
    ok = win["status"] == "ok"
    assert (~ok).sum() == 3 and s["n_ok_windows"] == 6 and s["n_oos_windows"] == 9 and s["weak"]
    assert s["is_ret_ann_mean"] == pytest.approx(win["is_ret_ann"].where(ok, 0.0).mean())
    assert (win.loc[~ok, "oos_ret"] == 0).all()


# ── Nested selection ─────────────────────────────────────────────────────────


def rand_streams(seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-02", periods=n, tz="America/New_York")
    starts = {"IS63_OOS5": 0, "IS126_OOS5": 30, "IS252_OOS10": 10, "IS504_OOS21": 60}
    ends = {"IS63_OOS5": n, "IS126_OOS5": n - 3, "IS252_OOS10": n, "IS504_OOS21": n - 17}
    rets = pd.DataFrame(index=days, columns=list(starts), dtype=float)
    rows = []
    for k, c in enumerate(starts):
        a, b = starts[c], ends[c]
        rets.iloc[a:b, k] = rng.normal(0.0003 * k, 0.01, b - a)
        for p in range(a, b, 7):
            rows.append({"combo": c, "oos_start": days[p], "is_sharpe": rng.normal(1, 0.5)})
    is_len = {c: int(c[2:].split("_")[0]) for c in starts}
    return rets, pd.DataFrame(rows), is_len


@pytest.mark.parametrize("seed", range(4))
def test_nested_selection_has_no_look_ahead(seed):
    rets, is_sr, is_len = rand_streams(seed)
    base = nested_select(rets, is_sr, is_len, "IS252_OOS10", every=10, lookback=50)
    assert (~base["burn_in"]).sum() > 10
    assert base.loc[base["burn_in"], "chosen"].eq("IS252_OOS10").all()
    for a, b in pairwise(base.itertuples()):
        assert rets.index.get_loc(b.start) == rets.index.get_loc(a.end) + 1  # periods tile
    rng = np.random.default_rng(100 + seed)
    for k in rng.choice(len(base), 8, replace=False):
        d = base["start"].iloc[k]
        pert = rets.copy()
        pert.loc[d:] = pert.loc[d:] * rng.normal(0, 30, pert.loc[d:].shape) + rng.normal(0, 0.1)  # this window on
        pert_is = is_sr.copy()
        late = pert_is["oos_start"] >= d
        pert_is.loc[late, "is_sharpe"] = rng.normal(0, 5, late.sum())
        got = nested_select(pert, pert_is, is_len, "IS252_OOS10", every=10, lookback=50)
        pd.testing.assert_frame_equal(got.iloc[: k + 1], base.iloc[: k + 1])  # incl. the window starting at d


def test_nested_selection_picks_the_trailing_best_and_breaks_ties():
    days = pd.bdate_range("2020-01-02", periods=60, tz="America/New_York")
    rets = pd.DataFrame(0.0, index=days, columns=["IS252_OOS10", "IS63_OOS5", "IS126_OOS5"])
    is_sr = pd.DataFrame({"combo": ["IS63_OOS5"], "oos_start": [days[0]], "is_sharpe": [1.0]})
    is_len = {"IS252_OOS10": 252, "IS63_OOS5": 63, "IS126_OOS5": 126}
    ch = nested_select(rets, is_sr, is_len, "IS252_OOS10", every=10, lookback=20)
    assert ch["burn_in"].tolist() == [True, True] + [False] * 4
    assert (ch.loc[~ch["burn_in"], "chosen"] == "IS63_OOS5").all()  # all flat: Sharpe and WFE tie → shortest IS
    rng = np.random.default_rng(0)
    rets.iloc[:, :] = rng.normal(0, 0.01, rets.shape)
    rets.loc[days[30:40], "IS126_OOS5"] += 0.05  # best over [days 30, 40) → chosen at day 40
    ch = nested_select(rets, is_sr, is_len, "IS252_OOS10", every=10, lookback=20)
    assert ch.set_index("start").loc[days[40], "chosen"] == "IS126_OOS5"
    pw = stitch(rets, ch)
    assert pw.index.equals(days) and (pw["ret"] == [rets.loc[d, c] for d, c in zip(pw.index, pw["combo"])]).all()
    with pytest.raises(ValueError, match="default"):
        nested_select(rets.drop(columns="IS252_OOS10"), is_sr, is_len, "IS252_OOS10", 10, 20)


# ── Full PWFO ────────────────────────────────────────────────────────────────

PW_CFG = CFG.replace(
    PWFO_IS_GRID=(40, 500, 700),
    PWFO_OOS_GRID=(25, 50),
    PWFO_DEFAULT=(700, 25),
    SELECT_EVERY=10,
    SELECT_LOOKBACK=60,
    PBO_BLOCKS=4,
)


@pytest.fixture(scope="module")
def pwfo_result(daily):
    return run_pwfo(daily, PW_CFG)


def test_pwfo_reports_pbo_dsr_and_infeasible_combos(pwfo_result):
    r = pwfo_result
    assert r.stats["n_combos"] == 6 and r.stats["dsr_n_trials"] == 6  # every tried combo is a trial
    # IS = 40 sessions holds too few events for MIN_TRAIN_EVENTS: reported, never selected
    infeasible = r.summary[r.summary["is_len"] == 40]
    assert (infeasible["n_ok_windows"] == 0).all() and infeasible["n_oos_windows"].gt(0).all()
    assert infeasible["weak"].all()
    assert r.stats["n_run_combos"] == 4 and set(r.returns.columns) == set(r.summary.index[r.summary["is_len"] > 40])
    assert set(r.choice["chosen"]) <= set(r.returns.columns)
    assert 0 <= r.stats["pbo"] <= 1 and r.stats["pbo_n_combos"] == 4
    assert 0 <= r.stats["pwfo_dsr"] <= r.stats["pwfo_psr0"] <= 1  # deflating can only lower the PSR
    assert r.stats["n_live_days"] + r.stats["n_burn_in_days"] == len(r.pwfo)
    assert r.pwfo["burn_in"].iloc[: r.stats["n_burn_in_days"]].all()
    assert (r.windows.groupby("combo").size() == r.summary["n_oos_windows"]).all()


def test_pwfo_is_causal(daily, pwfo_result):
    c = 1250
    other = run_pwfo(perturb_after(daily, c + 1), PW_CFG)
    day = daily.index[c].normalize()
    a, b = pwfo_result.pwfo.loc[:day], other.pwfo.loc[:day]
    assert len(a) > 300
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pwfo_result.returns.loc[:day], other.returns.loc[:day])


def test_pwfo_respects_the_holdout():
    idx = pd.bdate_range("2024-01-02", "2026-10-03", tz="America/New_York")
    with pytest.raises(HoldoutError):
        run_pwfo(synthetic_daily(len(idx)).set_axis(idx), PW_CFG)


@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize("combo", [Combo(20, 7, 7), Combo(30, 10, 10), Combo(12, 1, 4)])
def test_partial_last_window_runs_to_the_end_of_the_data(combo, expanding):
    cal = pd.bdate_range("2024-01-02", periods=90)
    idx = intraday_index([str(d.date()) for d in cal])
    full = pwfo_windows(idx, combo, embargo=2, expanding=expanding, sessions=cal)
    part = pwfo_windows(idx, combo, embargo=2, expanding=expanding, sessions=cal, partial_last=True)
    rest = (len(cal) - combo.is_len) % combo.oos_len
    assert part[: len(full)] == full and len(part) == len(full) + (rest > 0)  # the full windows are unchanged
    assert part[-1].oos_end == len(idx)  # the OOS windows now tile to the last bar
    if rest:
        last = part[-1]
        assert last.val_end == full[-1].oos_end and (last.oos_end - last.val_end) == rest * 7
        assert last.w == full[-1].w + 1 and last.train_end == full[-1].train_end + combo.oos_len * 7
