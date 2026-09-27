"""
U2: injected RunConfig, overnight policy, trading-day WFO windows, embargo, holdout guard,
and an end-to-end check that the WFO causality test catches a one-bar look-ahead.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from data.bars import HOLDOUT_START, HoldoutError
from features.triple_barrier_labels import barrier_exits, sample_events, triple_barrier_labels
from utils.config import TIMEFRAME_DEFAULTS, RunConfig
from utils.data_loader import load_ohlcv
from wfo.backtest import simulate_trades
from wfo.wfo_engine import purged, run_wfo, wfo_folds

LEGACY = RunConfig.legacy_5min()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_ohlcv("data/data.csv")


def perturb_after(df: pd.DataFrame, start: int, seed: int = 0) -> pd.DataFrame:
    """Multiply prices from bar `start` onward by a random walk and scramble volume."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = len(df) - start
    shock = np.exp(np.cumsum(rng.normal(0, 0.003, k)))
    for col in ("open", "high", "low", "close"):
        out.iloc[start:, out.columns.get_loc(col)] *= shock
    out.iloc[start:, out.columns.get_loc("volume")] = rng.integers(1e5, 1e7, k).astype(float)
    return out


def daily_bars(rows: list[tuple[float, float, float, float]], start: str = "2024-01-02") -> pd.DataFrame:
    """1Day bars stamped at NY midnight of consecutive business days (Fri → Mon skips the weekend)."""
    idx = pd.bdate_range(start, periods=len(rows), tz="America/New_York")
    out = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    out["volume"] = 1.0
    return out


def synthetic_daily(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.004, n))  # overnight gaps
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2012-01-03", periods=n, tz="America/New_York")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1e6}, index=idx)


# ── RunConfig ────────────────────────────────────────────────────────────────


def test_for_timeframe_matches_spec_table():
    for tf, d in TIMEFRAME_DEFAULTS.items():
        cfg = RunConfig.for_timeframe(tf)
        assert (cfg.TIMEFRAME, cfg.BARS_PER_DAY, cfg.VERTICAL_BARS, cfg.VOL_SPAN, cfg.HOLD_OVERNIGHT) == (
            tf,
            d.bars_per_day,
            d.vertical_bars,
            d.vol_span,
            d.hold_overnight,
        )
        assert (cfg.WINDOW_UNIT, cfg.INITIAL_TRAIN, cfg.VAL, cfg.TEST, cfg.EMBARGO) == (
            "days",
            d.initial_train_days,
            d.val_days,
            d.test_days,
            1,
        )
    assert RunConfig.for_timeframe("1Day").HOLD_OVERNIGHT
    assert not any(RunConfig.for_timeframe(tf).HOLD_OVERNIGHT for tf in ("1Min", "5Min", "15Min", "30Min", "1Hour"))


def test_legacy_5min_is_pre_u2_config():
    windows = (LEGACY.WINDOW_UNIT, LEGACY.INITIAL_TRAIN, LEGACY.VAL, LEGACY.TEST, LEGACY.EMBARGO)
    assert windows == ("bars", 2000, 1000, 500, 0)
    assert (LEGACY.BARS_PER_DAY, LEGACY.VERTICAL_BARS, LEGACY.VOL_SPAN, LEGACY.HOLD_OVERNIGHT) == (78, 12, 100, False)
    assert LEGACY.CLF_PARAMS["random_state"] == 42


def test_config_validation_and_overrides():
    with pytest.raises(ValueError, match="HOLD_OVERNIGHT"):
        RunConfig.for_timeframe("1Day", HOLD_OVERNIGHT=False)
    with pytest.raises(ValueError, match="Unknown timeframe"):
        RunConfig.for_timeframe("2Hour")
    with pytest.raises(ValueError, match="EMBARGO"):
        RunConfig.for_timeframe("5Min", EMBARGO=21)
    with pytest.raises(ValueError, match="WINDOW_UNIT"):
        RunConfig.for_timeframe("5Min", WINDOW_UNIT="weeks")
    cfg = RunConfig.for_timeframe("1Hour", VERTICAL_BARS=3, SEED=7)
    assert cfg.VERTICAL_BARS == 3 and cfg.VOL_SPAN == 70
    assert {cfg.CLF_PARAMS["random_state"], cfg.REG_PARAMS["random_state"], cfg.META_PARAMS["random_state"]} == {7}
    with pytest.raises(AttributeError):
        cfg.VERTICAL_BARS = 5  # frozen


# ── Overnight policy ─────────────────────────────────────────────────────────


def test_daily_bars_hold_overnight_and_exit_in_a_later_session():
    bars = daily_bars([(100, 100, 100, 100)] + [(100, 100.5, 99.5, 100)] * 5)  # Tue 01-02 … Tue 01-09
    ev = bars.index[[0]]
    w = pd.Series(0.05, index=bars.index)
    out = barrier_exits(bars, ev, w, vertical_bars=3, hold_overnight=True).iloc[0]
    assert out["barrier"] == "vertical"
    assert out["entry_pos"] == 1 and out["exit_pos"] == 3  # entered next session's open, held 3 sessions
    assert out["t1"] == pd.Timestamp("2024-01-05", tz="America/New_York")
    # Without the overnight policy a daily event can never enter (the entry is always next session)
    assert barrier_exits(bars, ev, w, vertical_bars=3, hold_overnight=False).empty


def test_daily_gap_through_barrier_fills_at_next_open_across_weekend():
    # Fri 01-05 event, entry Mon 01-08 open 100, Tue opens at 94 below the 98 barrier
    rows = [(100, 100, 100, 100)] * 4 + [(100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100), (94, 95, 93, 94)]
    bars = daily_bars(rows)
    ev = bars.index[[3]]
    out = barrier_exits(bars, ev, pd.Series(0.02, index=bars.index), vertical_bars=5, hold_overnight=True)
    assert out.empty  # window runs past the data: incomplete, not truncated
    bars = daily_bars(rows + [(94, 94, 94, 94)] * 3)
    out = barrier_exits(bars, ev, pd.Series(0.02, index=bars.index), vertical_bars=5, hold_overnight=True).iloc[0]
    assert bars.index[out["entry_pos"]] == pd.Timestamp("2024-01-08", tz="America/New_York")  # Monday
    assert out["barrier"] == "lower"
    assert out["exit_px"] == 94  # gap fill at the open, not the barrier
    assert out["t1"] == pd.Timestamp("2024-01-10", tz="America/New_York")


def test_daily_sampling_keeps_session_end_events_intraday_drops_them(df):
    d = synthetic_daily(400)
    cfg = RunConfig.for_timeframe("1Day")
    ev = sample_events(d, cfg)
    assert len(ev) > 20  # every daily bar is a session's last bar; none may be skipped
    labels = triple_barrier_labels(d, ev, cfg)
    assert (labels["exit_pos"] - labels["entry_pos"] >= 1).any()  # multi-session holds exist

    ev5 = sample_events(df, LEGACY)
    day = df.index.normalize()
    last_bar = pd.Series(day, index=df.index).shift(-1) != day
    assert not last_bar.loc[ev5.index].any()


def test_intraday_never_holds_overnight(df):
    labels = triple_barrier_labels(df, sample_events(df, LEGACY), LEGACY)
    assert (labels.index.normalize() == labels["t1"].dt.normalize()).all()
    sig = labels[["width"]].assign(trade_signal=np.where(np.arange(len(labels)) % 2, 1, -1))
    trades = simulate_trades(df, sig, LEGACY)
    assert (df.index[trades["entry_pos"]].normalize() == df.index[trades["exit_pos"]].normalize()).all()


# ── Trading-day windows and embargo ──────────────────────────────────────────


def intraday_index(sessions: list[tuple[str, int]]) -> pd.DatetimeIndex:
    """5Min RTH bars; each session is (date, number of bars) — 42 bars = 13:00 early close."""
    parts = [pd.date_range(f"{d} 09:30", periods=n, freq="5min", tz="America/New_York") for d, n in sessions]
    return parts[0].append(parts[1:])


def test_wfo_folds_in_trading_days_with_half_day():
    days = pd.bdate_range("2024-11-18", periods=12)
    lens = [78] * 12
    lens[8] = 42  # 2024-11-29, day after Thanksgiving (the holiday itself is simply absent)
    idx = intraday_index([(d.strftime("%Y-%m-%d"), n) for d, n in zip(days, lens)])
    cfg = RunConfig.for_timeframe("5Min", INITIAL_TRAIN=4, VAL=3, TEST=2, EMBARGO=1)
    folds = wfo_folds(idx, cfg)
    starts = np.r_[0, np.cumsum(lens)]
    assert [(f.train_end, f.val_end, f.test_end) for f in folds] == [
        (starts[4], starts[7], starts[9]),
        (starts[6], starts[9], starts[11]),
    ]
    for f in folds:  # every boundary is a session's first bar
        for b in (f.train_end, f.val_end):
            assert idx[b].strftime("%H:%M") == "09:30"
    # embargo = the whole preceding session: 78 bars normally, 42 after the half-day
    assert folds[0].train_embargo == 78 and folds[0].val_embargo == 78
    assert folds[1].val_embargo == 42  # val_end = session 9, the session before it is the half-day
    # 2-day test windows hold whole sessions (the first includes the half-day) and tile the span
    assert [f.test_end - f.val_end for f in folds] == [78 + 42, 78 + 78]
    assert all(b.val_end == a.test_end for a, b in pairwise(folds))


def test_wfo_folds_bars_mode_is_legacy_range():
    idx = pd.date_range("2024-01-02", periods=4500, freq="5min")
    folds = wfo_folds(idx, LEGACY)
    assert [f.train_end for f in folds] == list(range(2000, 4500 - 1000 - 500 + 1, 500))
    assert all(f.val_end == f.train_end + 1000 and f.test_end == f.val_end + 500 for f in folds)
    assert all(f.train_embargo == f.val_embargo == 0 for f in folds)


def test_embargo_removes_exactly_the_exits_before_the_boundary():
    # events at bars 0..99, each held 5 bars (entry t+1, exit t+5)
    t = np.arange(100)
    labels = pd.DataFrame({"entry_pos": t + 1, "exit_pos": t + 5}, index=pd.RangeIndex(100))
    base = purged(labels, 0, 80)
    emb = purged(labels, 0, 80, embargo=10)
    assert base.index.max() == 74  # exit 79 < 80
    assert emb.index.max() == 64  # exit 69 < 70
    removed = base.index.difference(emb.index)
    assert list(removed) == list(range(65, 75))
    assert (labels.loc[removed, "exit_pos"] >= 70).all() and (labels.loc[removed, "exit_pos"] < 80).all()
    # embargo sits before the NEXT split; the start of the split is untouched
    assert purged(labels, 20, 80, embargo=10).index.min() == purged(labels, 20, 80).index.min() == 20


# ── Holdout guard ────────────────────────────────────────────────────────────


def test_holdout_guard():
    cfg = RunConfig.for_timeframe("1Day")
    ok = pd.bdate_range("2025-09-01", "2025-09-30", tz="America/New_York")
    cfg.holdout_guard(ok)
    reach = pd.bdate_range("2025-09-01", HOLDOUT_START, tz="America/New_York")
    with pytest.raises(HoldoutError):
        cfg.holdout_guard(reach)
    with pytest.raises(HoldoutError):
        cfg.holdout_guard(reach.tz_localize(None))
    cfg.replace(ALLOW_HOLDOUT=True).holdout_guard(reach)
    with pytest.raises(HoldoutError):
        run_wfo(synthetic_daily(len(reach)).set_axis(reach), cfg)


# ── End-to-end ───────────────────────────────────────────────────────────────


def test_daily_wfo_end_to_end_holds_across_sessions():
    d = synthetic_daily(1400)
    cfg = RunConfig.for_timeframe("1Day", INITIAL_TRAIN=700, VAL=350, TEST=100, MIN_TRAIN_EVENTS=50, MIN_VAL_EVENTS=30)
    sig = run_wfo(d, cfg)
    assert sig["fold"].nunique() == 3
    assert sig.index.min() >= d.index[1050]
    trades = simulate_trades(d, sig, cfg, side_col="signed_dir")
    assert (trades["bars_held"] > 1).any()  # entered one session, exited in a later one


def _wfo_cut_after_oos_event(base: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    """An OOS event in the last test fold and the bar right after it (where perturbation starts)."""
    folds = wfo_folds(base.index, LEGACY)
    ev = sample_events(base, LEGACY)
    pos = base.index.get_indexer(ev.index)
    in_test = ev.index[(pos >= folds[-1].val_end + 100) & (pos < folds[-1].test_end - 100)]
    event = in_test[0]
    return event, base.index.get_loc(event) + 1


def _wfo_prefix_diff(base: pd.DataFrame) -> None:
    """Assert the WFO rows up to an OOS event are unchanged when every later bar is perturbed."""
    event, start = _wfo_cut_after_oos_event(base)
    a = run_wfo(base, LEGACY)
    b = run_wfo(perturb_after(base, start), LEGACY)
    assert event in a.index
    cols = ["clf_prob", "magnitude", "meta_prob", "trade_signal", "width"]
    pd.testing.assert_frame_equal(a.loc[:event, cols], b.loc[:event, cols])


def test_wfo_causality_test_catches_one_bar_lookahead(df, monkeypatch):
    """The WFO-level causality check passes on the real features and fails once a
    feature peeks one bar ahead — so the check is sensitive enough to trust."""
    base = df.iloc[:4500]
    _wfo_prefix_diff(base)  # clean

    def leaky(d, cfg, fset):
        return fset.build(d).assign(leak=np.log(d["close"]).diff().shift(-1))  # next bar's return

    monkeypatch.setattr("wfo.wfo_engine.build_features", leaky)
    with pytest.raises(AssertionError, match="are different"):
        _wfo_prefix_diff(base)


def _fit_targets(d: pd.DataFrame, cfg: RunConfig, monkeypatch) -> dict[str, list[pd.DataFrame]]:
    """Run the WFO recording what each model is fitted on: train targets/weights, and every meta-model input."""
    import wfo.wfo_engine as eng

    rec: dict[str, list[pd.DataFrame]] = {}
    fit_clf, fit_meta = eng.fit_primary_classifier, eng.fit_meta_model

    def clf(X_tr, y_tr, w_tr, *args):
        rec["train"] = [y_tr.to_frame(), w_tr.to_frame()]
        return fit_clf(X_tr, y_tr, w_tr, *args)

    def meta(X_vl, prim, lbl, w, cfg):
        rec["val"] = [X_vl, prim, lbl.to_frame(), w.to_frame()]
        return fit_meta(X_vl, prim, lbl, w, cfg)

    monkeypatch.setattr(eng, "fit_primary_classifier", clf)
    monkeypatch.setattr(eng, "fit_meta_model", meta)
    eng.run_wfo(d, cfg)
    return rec


def _assert_fit_unaffected(df, cfg, monkeypatch) -> None:
    """Perturbing every bar from the start of a fitting split's embargo onward must leave
    what that split's model is fitted on unchanged (days mode: embargo = the last session)."""
    (f,) = wfo_folds(df.index, cfg)
    for split, start in (("train", f.train_end - f.train_embargo), ("val", f.val_end - f.val_embargo)):
        a = _fit_targets(df, cfg, monkeypatch)[split]
        b = _fit_targets(perturb_after(df, start), cfg, monkeypatch)[split]
        for x, y in zip(a, b, strict=True):
            pd.testing.assert_frame_equal(x, y)


def test_days_mode_purge_and_embargo_keep_fitting_targets_causal(df, monkeypatch):
    cfg = RunConfig.for_timeframe("5Min", INITIAL_TRAIN=20, VAL=10, TEST=5, EMBARGO=1)
    base = df.iloc[: wfo_folds(df.index, cfg)[0].test_end]  # exactly one fold
    _assert_fit_unaffected(base, cfg, monkeypatch)

    # Sensitivity: the same check fails once the embargo is not applied
    import wfo.wfo_engine as eng

    purge = eng.purged
    monkeypatch.setattr(eng, "purged", lambda labels, start, end, embargo=0: purge(labels, start, end))
    with pytest.raises(AssertionError):
        _assert_fit_unaffected(base, cfg, monkeypatch)
