"""
U3: feature registry, the feature zoo, the feature cache, purged k-fold and clustered MDA.

The causality test (SPEC §8) runs over every registered group: for random cut points c,
the rows <= c must be identical when every bar after c is replaced by an independent
random walk, and when the series is truncated at c.
"""

import numpy as np
import pandas as pd
import pytest
import pywddff.pywddff as pyw

import features.groups as G
from features import cache as feature_cache
from features.causal_modwt import wavelet_ar_features
from features.feature_builder import FeatureSet, build_features
from features.registry import REGISTRY, check_group_output, resolve_groups
from features.selection import clustered_mda
from utils.config import DEFAULT_FEATURE_GROUPS, RunConfig
from utils.data_loader import load_ohlcv
from validation.purged_cv import PurgedKFold
from wfo.wfo_engine import purged, run_wfo, wfo_folds

CFG5 = RunConfig.for_timeframe("5Min")
CFGD = RunConfig.for_timeframe("1Day")
ALL_GROUPS = sorted(REGISTRY)


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


def synthetic_daily(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1e6, 5e6, n).astype(float)
    idx = pd.bdate_range("2012-01-03", periods=n, tz="America/New_York")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def market_for(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return df.assign(close=df["close"] * np.exp(np.cumsum(rng.normal(0, 1e-3, len(df)))))


def synthetic_exo(index: pd.DatetimeIndex, seed: int = 0) -> dict[str, pd.DataFrame]:
    """
    Every registered group's exo series around the bars' dates (U15): cboe / fred rows on business days with random
    positive values and the real available_at rules (data.exo.available_at); calendar kinds as synthetic schedules
    public from 1 January of their year, FOMC / CPI / NFP with release instants, plus one unscheduled FOMC statement
    at 10:00 on a mid-sample day, public only from that instant.
    """
    from data.alpaca_source import NY_TZ
    from data.exo import available_at
    from features.exo_align import bar_days

    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(np.unique(bar_days(index)))
    span = pd.bdate_range(days[0] - pd.Timedelta(days=45), days[-1] + pd.Timedelta(days=120), name="date")
    names = sorted(set().union(*(REGISTRY[g].exo for g in REGISTRY)))
    out = {}
    for name in names:
        src, series = name.split("/")
        if src != "calendar":
            v = 15 * np.exp(np.cumsum(rng.normal(0, 0.03, len(span))))
            out[name] = pd.DataFrame({"value": v, "available_at": available_at(src, series, span)}, index=span)
            continue
        series = series.removesuffix("_SCHEDULE")  # the schedule of a kind = its rows (no cancellations here)
        step, offset, at = {"FOMC": (30, 3, "14:00"), "CPI": (21, 7, "08:30"), "NFP": (21, 12, "08:30"),
                            "OPEX": (21, 15, None), "TOM": (5, 1, None), "PRE_HOLIDAY": (37, 9, None)}[series]  # fmt: skip
        dates = span[offset::step]
        value = np.resize([-1.0, 1.0, 2.0, 3.0], len(dates)) if series == "TOM" else np.ones(len(dates))
        jan1 = pd.DatetimeIndex([pd.Timestamp(year=d.year, month=1, day=1) for d in dates])
        avail = jan1.tz_localize(NY_TZ).tz_convert("UTC")
        event_at = pd.DatetimeIndex([pd.NaT] * len(dates), tz="UTC")
        if at is not None:
            event_at = (
                (dates + pd.Timedelta(hours=int(at[:2]), minutes=int(at[3:]))).tz_localize(NY_TZ).tz_convert("UTC")
            )
        f = pd.DataFrame({"value": value, "event_at": event_at, "available_at": avail}, index=dates)
        if series == "FOMC":  # an unscheduled statement: public at its instant (data.events)
            d = days[len(days) // 2]
            if d not in f.index:
                inst = (d + pd.Timedelta(hours=10)).tz_localize(NY_TZ).tz_convert("UTC")
                f = pd.concat([f, pd.DataFrame({"value": 1.0, "event_at": [inst], "available_at": [inst]}, index=[d])])
        out[name] = f.sort_index().rename_axis("date")
    return out


def group_frame(
    name: str,
    d: pd.DataFrame,
    cfg: RunConfig,
    market: pd.DataFrame | None,
    fit_end: int = 1500,
    exo: dict | None = None,
    sector: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One group's frame through the FeatureSet's context rules (market, optional sector, an Exo view)."""
    spec = REGISTRY[name]
    context = {"market": market, "exo": exo if exo is not None else synthetic_exo(d.index)}
    if sector is not None:
        context["sector"] = sector
    ctx = FeatureSet(cfg, ["wavelet_core", name] if name != "wavelet_core" else [name], context=context).group_context[
        name
    ]
    if spec.per_fold:
        state = spec.fn.fit(d.iloc[:fit_end], cfg)  # fixed train prefix, before every cut
        return spec.fn.transform(d, state, cfg, ctx) if spec.needs else spec.fn.transform(d, state, cfg)
    return spec.fn(d, cfg, ctx)


# ── Causality (SPEC §8) over every registered group ──────────────────────────


@pytest.mark.parametrize("name", ALL_GROUPS)
def test_every_group_is_causal(df, name):
    cuts = np.sort(np.random.default_rng(123).integers(1600, len(df) - 50, 3))
    market, exo = market_for(df), synthetic_exo(df.index)  # exo is perturbed separately (tests/test_u15.py)
    base = group_frame(name, df, CFG5, market, exo=exo)
    assert base.notna().any().all(), f"{name}: a column is all-NaN on the fixture"
    for i, c in enumerate(cuts):
        variants = {
            "random walk": (random_walk_after(df, c, seed=i), random_walk_after(market, c, seed=100 + i)),
            "truncated": (df.iloc[: c + 1], market.iloc[: c + 1]),
        }
        for label, (d2, m2) in variants.items():
            other = group_frame(name, d2, CFG5, m2, exo=exo)
            np.testing.assert_allclose(
                other.iloc[: c + 1].to_numpy(),
                base.iloc[: c + 1].to_numpy(),
                rtol=0,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{name}: rows <= {c} changed ({label})",
            )


def test_causality_check_catches_a_one_bar_leak(df):
    """The check above is sensitive: a feature using the next bar's return fails it."""
    c = 2000
    leak = lambda d: np.log(d["close"]).diff().shift(-1).to_numpy()
    a, b = leak(df)[: c + 1], leak(random_walk_after(df, c, 0))[: c + 1]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name", [g for g in ALL_GROUPS if not REGISTRY[g].intraday_only])
def test_daily_groups_are_causal(name):
    d = synthetic_daily(900)
    m, exo = market_for(d), synthetic_exo(d.index)
    base = group_frame(name, d, CFGD, m, fit_end=600, exo=exo)
    c = 850
    other = group_frame(name, random_walk_after(d, c, 3), CFGD, random_walk_after(m, c, 4), fit_end=600, exo=exo)
    np.testing.assert_allclose(other.iloc[: c + 1], base.iloc[: c + 1], rtol=0, atol=1e-12, equal_nan=True)


# ── Registry contract ────────────────────────────────────────────────────────


def test_registry_contents_and_required_group():
    assert set(DEFAULT_FEATURE_GROUPS) <= set(REGISTRY)
    assert REGISTRY["wavelet_core"].required
    assert [g for g in REGISTRY if REGISTRY[g].required] == ["wavelet_core"]
    assert REGISTRY["intraday"].intraday_only and REGISTRY["fracdiff"].per_fold


def test_feature_set_without_wavelet_core_raises(df):
    with pytest.raises(ValueError, match="wavelet_core"):
        build_features(df, CFG5, ["trend", "volatility"])
    with pytest.raises(ValueError, match="wavelet_core"):
        FeatureSet(RunConfig.for_timeframe("5Min", FEATURE_GROUPS=("trend",)))
    with pytest.raises(ValueError, match="unknown"):
        build_features(df, CFG5, ["wavelet_core", "astrology"])
    with pytest.raises(ValueError, match="duplicate"):
        build_features(df, CFG5, ["wavelet_core", "trend", "trend"])
    with pytest.raises(ValueError, match="needs context"):
        build_features(df, CFG5, ["wavelet_core", "cross_asset"])


def test_build_features_5min_and_1day(df, capsys):
    f5 = build_features(df, CFG5, [g for g in DEFAULT_FEATURE_GROUPS if g != "fracdiff"])
    assert {c.split("__")[0] for c in f5.columns} == set(DEFAULT_FEATURE_GROUPS) - {"fracdiff"}
    assert f5.index.equals(df.index) and f5.iloc[300:].notna().all().all()

    d = synthetic_daily(600)
    capsys.readouterr()
    fd = build_features(d, CFGD, ["wavelet_core", "trend", "intraday", "volatility"])
    assert "Skipping intraday-only group 'intraday' on 1Day" in capsys.readouterr().out
    assert {c.split("__")[0] for c in fd.columns} == {"wavelet_core", "trend", "volatility"}
    assert fd.iloc[300:].notna().all().all()
    assert [s.name for s in resolve_groups(DEFAULT_FEATURE_GROUPS, "1Day")] == [
        g for g in DEFAULT_FEATURE_GROUPS if g != "intraday"
    ]


def test_stationarity_guard_rejects_price_levels(df):
    legacy_lags = wavelet_ar_features(df["close"], "db1", 4, 1).add_prefix("trend__")
    with pytest.raises(ValueError, match="raw price level"):
        check_group_output("trend", legacy_lags, df)
    with pytest.raises(ValueError, match="prefix"):
        check_group_output("trend", pd.DataFrame({"x": 0.0}, index=df.index), df)
    with pytest.raises(ValueError, match="inf"):
        check_group_output("trend", pd.DataFrame({"trend__x": np.inf}, index=df.index), df)


@pytest.mark.parametrize("name", ALL_GROUPS)
def test_no_group_is_a_price_level(df, name):
    """Every group passes the stationarity guard on the fixture (the build runs it on the static ones)."""
    feats = group_frame(name, df, CFG5, market_for(df), sector=market_for(df, seed=11))
    check_group_output(name, feats, df, REGISTRY[name].level_check)
    feats = feats.loc[:, feats.std() > 0]
    corr = feats.corrwith(df["close"]).abs()
    assert (corr.fillna(0) < 0.99).all(), corr.sort_values().tail(3)


# ── Implementations ──────────────────────────────────────────────────────────


def test_wavelet_core_is_legacy_lags_relative_to_close(df):
    core = REGISTRY["wavelet_core"].fn(df, CFG5, {})
    lags = wavelet_ar_features(df["close"], CFG5.WAVELET_FILTER, CFG5.WAVELET_J, CFG5.AR_LAGS)
    for k in range(1, CFG5.AR_LAGS + 1):
        np.testing.assert_allclose(
            core[f"wavelet_core__s4_lag_{k}"], np.log(lags[f"w_lag_{k}"] / df["close"]), equal_nan=True
        )


@pytest.mark.parametrize("filt", G.WAVELET_EXT_FILTERS)
def test_causal_modwt_matches_pywddff(df, filt):
    x = np.log(df["close"].iloc[:600])
    details, smooth = G.causal_modwt(x, filt, 3)
    ref = pyw.modwt(x=x.to_numpy(), filter=filt, J=3, remove_bc=True)
    n = ref.shape[0]
    for j in range(3):
        np.testing.assert_allclose(details[j].to_numpy()[-n:], ref[:, j], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(smooth.to_numpy()[-n:], ref[:, -1], rtol=1e-10)
    assert smooth.iloc[: len(x) - n].isna().all()  # exactly the boundary coefficients are NaN


def test_roll_and_range_estimators_on_known_processes():
    rng = np.random.default_rng(0)
    n, spread, sigma = 20_000, 0.002, 0.001
    mid = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    trade = mid * (1 + rng.choice([-1, 1], n) * spread / 2)  # bid-ask bounce
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    d = pd.DataFrame({"open": trade, "high": trade, "low": trade, "close": trade, "volume": 1e5}, index=idx)
    roll = G.microstructure(d, CFG5, {})["microstructure__roll_spread"]
    assert roll.median() == pytest.approx(spread, rel=0.25)

    # Parkinson σ on a fine-grained GBM path sampled into bars
    fine = 100 * np.exp(np.cumsum(rng.normal(0, sigma / np.sqrt(50), n * 50)))
    b = fine.reshape(n, 50)
    bars = pd.DataFrame(
        {"open": b[:, 0], "high": b.max(1), "low": b.min(1), "close": b[:, -1], "volume": 1e5}, index=idx
    )
    vol = G.volatility(bars, CFG5, {})
    assert vol["volatility__parkinson_20"].median() == pytest.approx(sigma, rel=0.15)
    assert vol["volatility__ewm_sigma"].median() == pytest.approx(sigma, rel=0.1)


def test_intraday_bar_fraction_marks_the_1hour_stub():
    idx = pd.DatetimeIndex([f"2024-01-02 {t}" for t in ("09:30", "10:30", "14:30", "15:30")], tz="America/New_York")
    d = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
    out = G.intraday(d, RunConfig.for_timeframe("1Hour"), {})
    assert out["intraday__bar_frac"].tolist() == [1.0, 1.0, 1.0, 0.5]


def test_cross_asset_alignment_uses_last_market_bar_at_or_before_t():
    idx = pd.date_range("2024-01-02 09:30", periods=300, freq="5min")
    d = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0}, index=idx)
    d["close"] = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, 300)))
    mkt = d.iloc[::2].copy()  # market missing every other bar
    mkt["close"] = np.arange(1, len(mkt) + 1, dtype=float)
    out = G.cross_asset(d, CFG5, {"market": mkt})
    ret = out["cross_asset__mkt_ret_1"]
    # on bars the market lacks, its close is carried forward (return 0), never taken from the next bar
    assert (ret.iloc[1::2] == 0).all()
    assert ret.iloc[2] == pytest.approx(np.log(2 / 1))


# ── Feature cache ────────────────────────────────────────────────────────────


def test_feature_cache_round_trip_and_invalidation(df, tmp_path, monkeypatch):
    groups = ["wavelet_core", "trend"]
    a = build_features(df, CFG5, groups, symbol="SPY", cache_dir=tmp_path)
    files = sorted(tmp_path.rglob("*.npz"))
    assert [f.parent.name for f in files] == ["trend", "wavelet_core"]

    # a cache hit never calls the group function
    import features.registry as reg

    boom = reg.FeatureGroup("trend", lambda *a: (_ for _ in ()).throw(RuntimeError("recomputed")))
    monkeypatch.setitem(reg.REGISTRY, "trend", boom)
    b = build_features(df, CFG5, groups, symbol="SPY", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(a, b)

    # different data or a feature-relevant cfg field → miss (recompute raises); model fields → hit
    with pytest.raises(RuntimeError, match="recomputed"):
        build_features(df.iloc[:-1], CFG5, groups, symbol="SPY", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="recomputed"):
        build_features(df, CFG5.replace(VERTICAL_BARS=6), groups, symbol="SPY", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="recomputed"):
        build_features(df, CFG5, groups, symbol="QQQ", cache_dir=tmp_path)
    build_features(df, CFG5.replace(SEED=7, TEST=5), groups, symbol="SPY", cache_dir=tmp_path)


def test_cache_key_covers_code_and_context(df):
    k = feature_cache.cache_key("cross_asset", "AAPL", CFG5, df, {"market": market_for(df)})
    assert k != feature_cache.cache_key("cross_asset", "AAPL", CFG5, df, {"market": market_for(df, seed=8)})
    assert len(feature_cache.code_hash()) == 64
    feature_cache.code_hash.cache_clear()
    import importlib.metadata as md

    real = md.version
    try:
        md.version = lambda lib: "0.0-other" if lib == "pandas" else real(lib)
        other = feature_cache.code_hash()
    finally:
        md.version = real
        feature_cache.code_hash.cache_clear()
    assert other != feature_cache.code_hash()  # a library upgrade invalidates the cache


def _cache_hammer(root, write: bool) -> pd.DataFrame | None:
    idx = pd.date_range("2020-01-01", periods=20_000, freq="min", tz="UTC")
    feats = pd.DataFrame({"a": np.arange(len(idx), dtype=float)}, index=idx)
    out = None
    for _ in range(150):
        if write:
            feature_cache.save(root, "X", "5Min", "g", "k", feats)
        else:
            out = feature_cache.load(root, "X", "5Min", "g", "k", idx)
    return out


def test_cache_tolerates_concurrent_writers_and_readers_of_a_key(tmp_path):
    """Cells sharing a symbol and timeframe save / load the same groups at once; on Windows os.replace onto a file
    another process has open raises PermissionError (Stage A cells died of it)."""
    from concurrent.futures import ProcessPoolExecutor

    _cache_hammer(tmp_path, write=True)
    with ProcessPoolExecutor(4) as ex:
        futs = [ex.submit(_cache_hammer, tmp_path, w) for w in (True, True, False, False)]
        res = [f.result() for f in futs]  # re-raises a worker's error
    assert all(r is None or r["a"].iloc[-1] == 19_999 for r in res)
    assert not list(tmp_path.rglob("*.npz"))[1:]  # one file, no temp leftovers


# ── Purged k-fold ────────────────────────────────────────────────────────────


def test_purged_kfold_never_overlaps_test_span_or_embargo():
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(20, 200))
        t0 = np.sort(rng.integers(0, 2000, n))
        t1 = t0 + rng.integers(0, 40, n)
        emb = int(rng.integers(0, 30))
        seen = []
        for train, test in PurgedKFold(int(rng.integers(2, 6)), emb).split(t0, t1):
            a, b = t0[test].min(), t1[test].max()
            assert not np.intersect1d(train, test).size
            assert ((t1[train] < a) | (t0[train] > b + emb)).all()
            # purge is exact: every excluded non-test sample overlaps [a, b + emb]
            out = np.setdiff1d(np.arange(n), np.r_[train, test])
            assert ((t1[out] >= a) & (t0[out] <= b + emb)).all()
            seen.append(test)
        assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(n))


# ── Clustered MDA ────────────────────────────────────────────────────────────


def test_clustered_mda_keeps_the_informative_cluster():
    rng = np.random.default_rng(0)
    n = 1500
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "wavelet_core__a": rng.normal(size=n),
            "sig__x1": signal + rng.normal(0, 0.1, n),
            "sig__x2": signal + rng.normal(0, 0.1, n),
            **{f"noise__n{i}": rng.normal(size=n) for i in range(6)},
        }
    )
    y = np.where(signal + rng.normal(0, 0.5, n) > 0, 1, -1)
    t = np.arange(n) * 3
    labels = pd.DataFrame({"label": y, "entry_pos": t + 1, "exit_pos": t + 4}, index=X.index)
    sel = clustered_mda(X, labels, pd.Series(1.0, index=X.index), CFG5, n_bars=3 * n)
    assert {"sig__x1", "sig__x2", "wavelet_core__a"} <= set(sel.kept)
    assert len([c for c in sel.kept if c.startswith("noise")]) <= 2
    assert any(set(cols) == {"sig__x1", "sig__x2"} for cols in sel.clusters.values())


def test_cmda_in_wfo_uses_only_purged_train_rows(monkeypatch):
    import wfo.wfo_engine as eng

    d = load_ohlcv("data/data.csv")
    cfg = RunConfig.for_timeframe(
        "5Min", INITIAL_TRAIN=30, VAL=12, TEST=5, FEATURE_SELECTION="cmda", CMDA_TREES=30, MIN_VAL_EVENTS=60
    )
    (f,) = wfo_folds(d.index, cfg)[:1]
    d = d.iloc[: f.test_end]
    calls = []
    orig = eng.select_features

    def spy(X_tr, lab_tr, w_tr, cfg, n_bars):
        sel = orig(X_tr, lab_tr, w_tr, cfg, n_bars)
        calls.append((X_tr.index, lab_tr.copy(), n_bars, sel.kept, X_tr.copy(), sel.importance))
        return sel

    monkeypatch.setattr(eng, "select_features", spy)
    sig = run_wfo(d, cfg)
    assert len(calls) == 1 and len(sig)
    idx, lab, n_bars, kept, X0, imp0 = calls[0]
    assert n_bars == f.train_end
    allowed = purged(lab, 0, f.train_end, f.train_embargo).index
    assert idx.isin(allowed).all() and len(idx) == len(lab)
    assert (d.index.get_indexer(idx) < f.train_end).all()
    assert (lab["exit_pos"] < f.train_end - f.train_embargo).all()
    assert any(c.startswith("wavelet_core__") for c in kept)

    # CMDA's inputs and output are identical when every bar from the train embargo onward is
    # perturbed — random walks (several seeds) and a ramp confined to the embargo bars, which
    # moved fracdiff's d when it was fit on the embargo too
    start = f.train_end - f.train_embargo
    ramp = d.copy()
    k = f.train_embargo
    for col in ("open", "high", "low", "close"):
        ramp.iloc[start : start + k, ramp.columns.get_loc(col)] *= np.linspace(1.0, 1.08, k)
    for d2 in [random_walk_after(d, start - 1, seed=s) for s in (0, 4, 5, 9)] + [ramp]:
        calls.clear()
        run_wfo(d2, cfg)
        _, _, _, kept2, X2, imp2 = calls[0]
        pd.testing.assert_frame_equal(X2, X0)
        pd.testing.assert_frame_equal(imp2, imp0)
        assert kept2 == kept


def test_cmda_requires_registry_features(df):
    with pytest.raises(ValueError, match="cmda"):
        RunConfig.legacy_5min(FEATURE_SELECTION="cmda")
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)), columns=["a", "b", "c"])
    labels = pd.DataFrame({"label": 1, "entry_pos": np.arange(200) + 1, "exit_pos": np.arange(200) + 2})
    with pytest.raises(ValueError, match="wavelet_core"):
        clustered_mda(X, labels, pd.Series(1.0, index=X.index), CFG5, n_bars=400)
