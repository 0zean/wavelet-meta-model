"""
U4: primary-signal zoo — causality of every primary's side (SPEC §8), synthetic sanity checks,
the registry/config contract, the WFO config switch and output schema, and the diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

from features.events import sample_events
from features.feature_builder import build_features
from features.triple_barrier_labels import average_uniqueness, barrier_exits, triple_barrier_labels
from primaries import PRIMARY_COLUMNS, REGISTRY, base, check_signal, make_primary, side
from primaries.diagnostics import primary_diagnostics
from primaries.rules import RulePrimary
from utils.config import RunConfig
from utils.data_loader import load_ohlcv
from wfo.wfo_engine import purged, run_wfo, wfo_folds

CFG5 = RunConfig.for_timeframe("5Min")
CFGD = RunConfig.for_timeframe("1Day")
RULES = sorted(n for n in REGISTRY if n != "ml_xgb")
GROUPS = ["wavelet_core", "trend", "volatility"]  # static groups only: no per-fold fit needed
WFO_COLUMNS = [*PRIMARY_COLUMNS, "meta_prob", "trade_signal", "width", "fold", "primary", "bet_size"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_ohlcv("data/data.csv").iloc[:3000]


def random_walk_after(df: pd.DataFrame, c: int, seed: int) -> pd.DataFrame:
    """Rows > c replaced by an independent random walk (prices) and random volume."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = len(df) - c - 1
    close = df["close"].iloc[c] * np.exp(np.cumsum(rng.normal(0, 0.002, k)))
    open_ = close * np.exp(rng.normal(0, 0.001, k))
    out.iloc[c + 1 :, out.columns.get_loc("close")] = close
    out.iloc[c + 1 :, out.columns.get_loc("open")] = open_
    out.iloc[c + 1 :, out.columns.get_loc("high")] = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1e-3, k)))
    out.iloc[c + 1 :, out.columns.get_loc("low")] = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1e-3, k)))
    out.iloc[c + 1 :, out.columns.get_loc("volume")] = rng.integers(1e4, 1e7, k).astype(float)
    return out


def daily_from_close(close: np.ndarray, spread: float = 0.002, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + spread * rng.random(n))
    lo = np.minimum(open_, close) * (1 - spread * rng.random(n))
    idx = pd.bdate_range("2012-01-03", periods=n, tz="America/New_York")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1e6}, index=idx)


def synthetic_daily(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2012-01-03", periods=n, tz="America/New_York")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1e6}, index=idx)


def rule_signal(name: str, d: pd.DataFrame, cfg: RunConfig, start: int, **params) -> pd.DataFrame:
    """The rule's primary frame at every bar from `start` on."""
    p = make_primary(cfg.replace(PRIMARY=name, PRIMARY_PARAMS=params))
    return p.signal(d, d.iloc[start:], cfg)


# ── Causality (SPEC §8) of every primary's output ────────────────────────────


@pytest.mark.parametrize("name", RULES)
def test_rule_primary_is_causal(df, name):
    start = 300  # past every rule window and the σ warm-up
    ref = rule_signal(name, df, CFG5, start)
    assert set(ref["signed_dir"]) == {-1, 1}
    for i, c in enumerate(np.sort(np.random.default_rng(7).integers(1000, len(df) - 50, 3))):
        for label, d2 in (("random walk", random_walk_after(df, c, seed=i)), ("truncated", df.iloc[: c + 1])):
            other = rule_signal(name, d2, CFG5, start)
            pd.testing.assert_frame_equal(
                other.loc[: df.index[c]], ref.loc[: df.index[c]], check_exact=False, rtol=0, atol=1e-12, obj=label
            )


def test_ml_primary_is_causal(df):
    """ml_xgb fit on a fixed train prefix: its frame at rows <= c ignores every bar after c."""
    fit_end, cfg = 1500, CFG5.replace(CLF_PARAMS={**CFG5.CLF_PARAMS, "n_estimators": 50})
    cfg = cfg.replace(REG_PARAMS={**cfg.REG_PARAMS, "n_estimators": 50})

    def frame(d: pd.DataFrame) -> pd.DataFrame:
        X = build_features(d, cfg, GROUPS)
        lab = purged(triple_barrier_labels(d, sample_events(d, cfg), cfg), 0, fit_end)
        X_tr = X.reindex(lab.index).dropna()
        lab = lab.loc[X_tr.index]
        p = make_primary(cfg).fit(d.iloc[:fit_end], X_tr, lab, average_uniqueness(lab, len(d)), cfg, val=(X_tr, lab))
        rows = X.iloc[fit_end:].dropna()
        return check_signal(p.signal(d, rows, cfg), rows, "ml_xgb")

    ref = frame(df)
    for c in (2000, 2600):
        other = frame(random_walk_after(df, c, seed=c))
        cut = df.index[c]
        pd.testing.assert_frame_equal(other.loc[:cut], ref.loc[:cut], check_exact=False, rtol=0, atol=1e-12)


def test_causality_check_catches_a_leaky_rule(df, monkeypatch):
    """The rule check is sensitive: a rule reading the next bar's close fails it."""

    class Leaky(RulePrimary):
        def score(self, d, cfg):
            return np.log(d["close"]).diff().shift(-1).fillna(0.0)

    Leaky.name = "leaky"
    monkeypatch.setitem(REGISTRY, "leaky", Leaky)
    c = 2000
    ref = rule_signal("leaky", df, CFG5, 300)
    other = rule_signal("leaky", random_walk_after(df, c, 0), CFG5, 300)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(other.loc[: df.index[c]], ref.loc[: df.index[c]])


@pytest.mark.parametrize("name", RULES)
def test_rules_ignore_labels(df, name):
    """Fixed rules never tune: fitting on any labels leaves the side unchanged."""
    p = make_primary(CFG5.replace(PRIMARY=name))
    before = p.signal(df, df.iloc[300:], CFG5)
    lab = pd.DataFrame({"label": 1, "ret": 1.0}, index=df.index[300:])
    after = p.fit(df, df.iloc[300:], lab, pd.Series(1.0, index=lab.index), CFG5).signal(df, df.iloc[300:], CFG5)
    pd.testing.assert_frame_equal(before, after)


# ── Synthetic sanity ─────────────────────────────────────────────────────────


def trend(n: int, drift: float, noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return daily_from_close(100 * np.exp(np.cumsum(drift + noise * rng.normal(size=n))), seed=seed)


@pytest.mark.parametrize("name", ["sma_cross", "wavelet_trend", "donchian_breakout"])
@pytest.mark.parametrize("noise", [0.0, 0.002])
def test_trend_rules_follow_a_monotone_trend(name, noise):
    for drift, want in ((0.01, 1), (-0.01, -1)):
        d = trend(400, drift, noise)
        sides = rule_signal(name, d, CFGD, 100)["signed_dir"]
        assert (sides == want).all(), f"{name} drift={drift}: {sides.value_counts().to_dict()}"


def test_wavelet_trend_level_mode_follows_trend():
    for drift, want in ((0.01, 1), (-0.01, -1)):
        assert (
            rule_signal("wavelet_trend", trend(400, drift, 0.002), CFGD, 100, mode="level")["signed_dir"] == want
        ).all()


def test_donchian_breakout_strength_exceeds_one_on_breakout():
    close = np.r_[100 + 0.5 * np.sin(np.arange(60)), 105.0]  # range-bound, then a jump above the channel
    s = rule_signal("donchian_breakout", daily_from_close(close), CFGD, 55)
    assert s["signed_dir"].iloc[-1] == 1 and s["magnitude"].iloc[-1] > 1


@pytest.mark.parametrize("jump, want", [(1.05, -1), (0.95, 1)])
def test_bollinger_mr_fades_a_spike(jump, want):
    rng = np.random.default_rng(3)
    close = 100 * np.exp(0.001 * rng.normal(size=200))
    close[150] = close[149] * jump
    s = rule_signal("bollinger_mr", daily_from_close(close), CFGD, 100)
    assert s["signed_dir"].iloc[50] == want and s["magnitude"].iloc[50] > 2
    # ... and fades a steady trend too (the close sits above its rolling mean)
    assert (rule_signal("bollinger_mr", trend(300, 0.01, 0.002), CFGD, 100)["signed_dir"] == -1).all()


def test_ml_primary_learns_a_planted_side():
    """ml_xgb recovers side = sign(x0) when the label is sign(x0)."""
    rng = np.random.default_rng(0)
    n = 1200
    idx = pd.bdate_range("2012-01-03", periods=n, tz="America/New_York")
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["x0", "x1", "x2"])
    y = np.where(X["x0"] > 0, 1, -1)
    lab = pd.DataFrame({"label": y, "ret": y * 0.01 * (1 + rng.random(n))}, index=idx)
    d = daily_from_close(100 * np.exp(np.cumsum(0.01 * rng.normal(size=n))))
    cfg = CFGD.replace(LOW_MOVE_PCTILE=0.0)
    p = make_primary(cfg).fit(
        d.iloc[:800],
        X.iloc[:800],
        lab.iloc[:800],
        pd.Series(1.0, index=idx[:800]),
        cfg,
        val=(X.iloc[800:], lab.iloc[800:]),
    )
    got = side(p, d, X.iloc[800:], cfg)
    assert (got == np.sign(X["x0"].iloc[800:])).mean() > 0.95


# ── Registry / config contract ───────────────────────────────────────────────


def test_registry_and_params():
    assert set(REGISTRY) == {"ml_xgb", "sma_cross", "bollinger_mr", "wavelet_trend", "donchian_breakout"}
    assert RunConfig().PRIMARY == "ml_xgb" and RunConfig.legacy_5min().PRIMARY == "ml_xgb"
    with pytest.raises(ValueError, match="unknown primary"):
        make_primary(CFG5.replace(PRIMARY="astrology"))
    with pytest.raises(ValueError, match="unknown params"):
        make_primary(CFG5.replace(PRIMARY="sma_cross", PRIMARY_PARAMS={"fsat": 5}))
    with pytest.raises(ValueError, match="fast must be < slow"):
        make_primary(CFG5.replace(PRIMARY="sma_cross", PRIMARY_PARAMS={"fast": 50, "slow": 20}))
    with pytest.raises(ValueError, match="takes no PRIMARY_PARAMS"):
        make_primary(CFG5.replace(PRIMARY_PARAMS={"n": 1}))
    with pytest.raises(ValueError, match="mode"):
        make_primary(CFG5.replace(PRIMARY="wavelet_trend", PRIMARY_PARAMS={"mode": "vibes"}))
    assert make_primary(CFG5.replace(PRIMARY="sma_cross", PRIMARY_PARAMS={"fast": 5})).params == {"fast": 5, "slow": 50}
    # PRIMARY_PARAMS is copied per config (no sharing through replace)
    a = CFG5.replace(PRIMARY_PARAMS={"x": 1})
    b = a.replace(SEED=1)
    b.PRIMARY_PARAMS["x"] = 2
    assert a.PRIMARY_PARAMS == {"x": 1}


def test_unfitted_ml_primary_raises(df):
    p = make_primary(CFG5)
    with pytest.raises(RuntimeError, match="before fit"):
        p.signal(df, df.iloc[:5], CFG5)


def test_check_signal_and_rule_frame_reject_bad_output():
    idx = pd.bdate_range("2020-01-01", periods=4)
    X = pd.DataFrame(index=idx)
    good = base.rule_frame(pd.Series([1.0, -2.0, 0.0, 3.0], index=idx), idx, "r")
    assert list(good["signed_dir"]) == [1, -1, 1, 1]  # an exact tie goes long
    check_signal(good, X, "r")
    with pytest.raises(ValueError, match="outside"):
        check_signal(good.assign(signed_dir=[1, 0, 1, -1]), X, "r")
    with pytest.raises(ValueError, match="columns"):
        check_signal(good.drop(columns="clf_prob"), X, "r")
    with pytest.raises(ValueError, match="indexed"):
        check_signal(good.iloc[::-1], X, "r")
    with pytest.raises(ValueError, match="undefined"):
        base.rule_frame(pd.Series([1.0, np.nan, 1.0, 1.0], index=idx), idx, "r")


# ── WFO switch and schema ────────────────────────────────────────────────────

WFO_CFG = {"INITIAL_TRAIN": 700, "VAL": 350, "TEST": 100, "MIN_TRAIN_EVENTS": 50, "MIN_VAL_EVENTS": 30}


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_daily(1400)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_wfo_runs_with_every_primary(daily, name):
    cfg = RunConfig.for_timeframe("1Day", PRIMARY=name, **WFO_CFG)
    sig = run_wfo(daily, cfg)
    assert list(sig.columns) == WFO_COLUMNS
    assert (sig["primary"] == name).all() and sig["fold"].nunique() == 3
    assert set(sig["signed_dir"]) <= {-1, 1}
    if name != "ml_xgb":  # the WFO passes the rule's side through unchanged
        want = make_primary(cfg).signal(daily, daily.loc[sig.index], cfg)["signed_dir"]
        pd.testing.assert_series_equal(sig["signed_dir"], want, check_names=False, check_dtype=False)
        assert sig["clf_prob"].isna().all()


def test_wfo_with_rule_primary_is_causal(daily):
    """Perturbing every bar after an OOS event leaves the WFO rows up to it unchanged."""
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="donchian_breakout", **WFO_CFG)
    folds = wfo_folds(daily.index, cfg)
    ev = sample_events(daily, cfg)
    pos = daily.index.get_indexer(ev.index)
    event = ev.index[(pos >= folds[-1].val_end + 20) & (pos < folds[-1].test_end - 20)][0]
    c = daily.index.get_loc(event)
    a = run_wfo(daily, cfg)
    b = run_wfo(random_walk_after(daily, c, seed=5), cfg)
    assert event in a.index
    pd.testing.assert_frame_equal(a.loc[:event], b.loc[:event])


def test_wfo_rejects_unknown_primary_before_work(daily):
    with pytest.raises(ValueError, match="unknown primary"):
        run_wfo(daily, RunConfig.for_timeframe("1Day", PRIMARY="nope", **WFO_CFG))


# ── Diagnostics ──────────────────────────────────────────────────────────────


def test_diagnostics_definitions(daily):
    cfg = CFGD
    ev = sample_events(daily, cfg)
    r = {
        s: s
        * barrier_exits(
            daily, ev.index, ev["width"], side=pd.Series(s, index=ev.index), vertical_bars=10, hold_overnight=True
        )["ret"]
        for s in (1, -1)
    }
    idx = r[1].index
    oracle = pd.Series(np.where(r[1] >= r[-1], 1, -1), index=idx)
    sig = pd.DataFrame(
        {
            "signed_dir": oracle,
            "width": ev.loc[idx, "width"],
            "fold": np.repeat([1, 2], [len(idx) // 2, len(idx) - len(idx) // 2]),
            "primary": "oracle",
        },
        index=idx,
    )
    t = primary_diagnostics(daily, sig, cfg, symbol="SYN")
    assert list(t["fold"]) == [1, 2, "all"]
    allrow = t.iloc[-1]
    assert allrow["n_events"] == len(idx) and allrow["recall"] == 1.0  # perfect foresight catches every opportunity
    assert allrow["precision"] == pytest.approx(allrow["opportunity"])

    always_long = sig.assign(signed_dir=1)
    a = primary_diagnostics(daily, always_long, cfg).iloc[-1]
    assert a["long_share"] == 1.0 and a["turnover"] == 0.0
    assert a["precision"] == pytest.approx((r[1] > cfg.META_MIN_RET).mean())
    assert a["net_bp"] == pytest.approx((r[1].mean() - 2 * cfg.SLIPPAGE_PCT) * 1e4)
    alt = sig.assign(signed_dir=np.resize([1, -1], len(idx)))
    assert primary_diagnostics(daily, alt, cfg).iloc[-1]["turnover"] == 1.0


def test_diagnostics_hand_computed_example():
    """Three events on flat bars: a same-bar tie (adverse for both sides), a +1 bp vertical exit
    (below the 2 bp cost threshold) and a +1% upside move taken short."""
    n = 40
    bars = pd.DataFrame(
        100.0,
        index=pd.bdate_range("2020-01-02", periods=n, tz="America/New_York"),
        columns=["open", "high", "low", "close"],
    )
    bars.iloc[6, bars.columns.get_indexer(["high", "low"])] = [101.5, 98.5]  # event A (t=5): tie on the entry bar
    bars.iloc[25, bars.columns.get_indexer(["high", "close"])] = [100.01, 100.01]  # event B (t=15): vertical exit +1 bp
    bars.iloc[28, bars.columns.get_loc("high")] = 101.2  # event C (t=26): upper barrier +1%
    bars["volume"] = 1.0
    idx = bars.index[[5, 15, 26]]
    sig = pd.DataFrame({"signed_dir": [1, 1, -1], "width": 0.01, "fold": 1, "primary": "hand"}, index=idx)
    row = primary_diagnostics(bars, sig, CFGD).iloc[-1]
    assert row["n_events"] == 3
    assert row["long_share"] == pytest.approx(2 / 3)
    assert row["precision"] == 0.0  # A −1%, B +1 bp (< 2 bp cost), C −1%
    assert row["opportunity"] == pytest.approx(1 / 3)  # only C (long +1%)
    assert row["recall"] == 0.0
    assert row["turnover"] == 0.5
    assert row["net_bp"] == pytest.approx(((-0.01 + 0.0001 - 0.01) / 3 - 2 * CFGD.SLIPPAGE_PCT) * 1e4)


def test_rule_param_types_fail_fast():
    for params in ({"window": 20.0}, {"window": True}, {"window": 1}):
        with pytest.raises(ValueError, match="integer >= 2"):
            make_primary(CFG5.replace(PRIMARY="bollinger_mr", PRIMARY_PARAMS=params))
    with pytest.raises(ValueError, match="integer >= 2"):
        make_primary(CFG5.replace(PRIMARY="sma_cross", PRIMARY_PARAMS={"fast": 1.5}))


def test_rule_scores_match_hand_formulas(df):
    """Each rule's signed score (signal column) at one bar, from raw prices (db1, J=4: S_4 = 16-bar mean)."""
    t = 1000
    c, hi, lo = (df[k].to_numpy() for k in ("close", "high", "low"))
    lp = np.log(c)
    sigma = np.log(df["close"]).diff().ewm(span=CFG5.VOL_SPAN, min_periods=CFG5.VOL_SPAN).std().iloc[t]
    s4 = lambda k: lp[k - 15 : k + 1].mean()
    H, L = hi[t - 20 : t].max(), lo[t - 20 : t].min()
    want = {
        ("sma_cross", ()): np.log(c[t - 19 : t + 1].mean() / c[t - 49 : t + 1].mean()) / sigma,
        ("bollinger_mr", ()): -(c[t] - c[t - 19 : t + 1].mean()) / c[t - 19 : t + 1].std(ddof=1),
        ("wavelet_trend", ()): (s4(t) - s4(t - 1)) / sigma,
        ("wavelet_trend", (("mode", "level"),)): (lp[t] - s4(t)) / sigma,
        ("donchian_breakout", ()): (c[t] - (H + L) / 2) / ((H - L) / 2),
    }
    for (name, params), w in want.items():
        got = rule_signal(name, df, CFG5, t, **dict(params))["signal"].iloc[0]
        assert got == pytest.approx(w, rel=1e-9), (name, params)
