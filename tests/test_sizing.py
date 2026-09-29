"""
U7: bet sizing — sizer properties (monotone, bounded, zero below META_THRESH), train-window-only fitting (OOF meta
probabilities of the fitting events), discretization, fractional and averaged-position backtests (hand-computed
examples, costs on every resize, causality) and the WFO `bet_size` column.
"""

import numpy as np
import pandas as pd
import pytest

from features.triple_barrier_labels import barrier_exits
from models import meta_model
from models.meta_model import oof_meta_prob
from sizing import REGISTRY, SizerFitError, discretize, make_sizer
from sizing import sizers as sizers_mod
from tests.test_primaries import WFO_CFG, random_walk_after
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from wfo import wfo_engine
from wfo.backtest import equity_curve, run_backtest, simulate_positions, simulate_trades
from wfo.wfo_engine import purged, run_wfo, wfo_folds

CFG = RunConfig.for_timeframe("1Day")
RNG = np.random.default_rng(0)
P_TRAIN = RNG.uniform(0.3, 0.85, 400)
RET_TRAIN = RNG.normal(0.001, 0.02, 400)


def fitted(name: str, cfg: RunConfig = CFG):
    s = make_sizer(name, cfg)
    return s.fit(P_TRAIN, RET_TRAIN) if s.needs_train else s


# ── Sizers ───────────────────────────────────────────────────────────────────


def test_registry_and_config():
    assert set(REGISTRY) == {"fixed", "linear", "ldp_sigmoid", "ecdf", "kelly_capped"}
    with pytest.raises(ValueError, match="unknown sizer"):
        make_sizer("nope", CFG)
    for bad in ({"SIZER": "nope"}, {"SIZE_STEP": 1.5}, {"KELLY_FRACTION": 0}, {"POSITION_MODE": "multi"}):
        with pytest.raises(ValueError):
            CFG.replace(**bad)
    with pytest.raises(ValueError, match="META_THRESH"):
        CFG.replace(META_THRESH=1.0)


@pytest.mark.parametrize("tau", [0.5, 0.55, 0.62])
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_sizer_monotone_bounded_zero_below_threshold(name, tau):
    cfg = CFG.replace(META_THRESH=tau)
    s = fitted(name, cfg)
    p = np.r_[np.linspace(0, 1, 2001), RNG.uniform(0, 1, 2000), tau, np.nextafter(tau, 0), 0.0, 1.0]
    p.sort()
    m = s.size(p)
    assert ((m >= 0) & (m <= 1)).all()
    assert (np.diff(m) >= 0).all()  # non-decreasing in p
    assert (m[p < tau] == 0).all()
    if name in ("linear", "ecdf"):  # these also start from 0 at the threshold itself
        assert s.size([tau])[0] == 0 or (name == "ecdf" and (s.ref_ == tau).any())
    if name == "ldp_sigmoid":
        assert (m[p <= 0.5] == 0).all()
    with pytest.raises(ValueError, match="NaN"):
        s.size([0.6, np.nan])


def test_sizer_formulas():
    p = np.array([0.4, 0.5, 0.6, 0.75, 0.9])
    np.testing.assert_array_equal(fitted("fixed").size(p), [0, 1, 1, 1, 1])
    np.testing.assert_allclose(fitted("linear").size(p), [0, 0, 0.2, 0.5, 0.8])
    from scipy.stats import norm

    z = (p - 0.5) / np.sqrt(p * (1 - p))
    np.testing.assert_allclose(fitted("ldp_sigmoid").size(p), np.maximum(0, 2 * norm.cdf(z) - 1) * (p >= 0.5))
    # ecdf: rank among the train OOF probabilities that reach τ
    e = make_sizer("ecdf", CFG).fit([0.2, 0.55, 0.6, 0.7, 0.8], np.zeros(5))
    np.testing.assert_allclose(e.size([0.5, 0.55, 0.65, 0.8, 0.95]), [0, 0.25, 0.5, 1, 1])
    # kelly: net = ret − 2·slippage on approved events only; b = mean win / mean loss
    slip2 = 2 * CFG.SLIPPAGE_PCT
    k = make_sizer("kelly_capped", CFG).fit(
        [0.6, 0.7, 0.8, 0.9, 0.3], np.array([0.03, 0.01, -0.01, -0.02, 5.0]) + slip2
    )
    assert k.b_ == pytest.approx(0.02 / 0.015)
    np.testing.assert_allclose(k.size([0.75]), [0.25 * (0.75 - 0.25 / k.b_)])


def test_train_fit_sizers_refuse_undefined_inputs():
    with pytest.raises(SizerFitError, match="ecdf"):
        make_sizer("ecdf", CFG).fit([0.1, 0.4], [0.0, 0.0])
    with pytest.raises(SizerFitError, match="wins and losses"):
        make_sizer("kelly_capped", CFG).fit([0.6, 0.7], [0.01, 0.02])  # no loss
    with pytest.raises(SizerFitError, match="wins and losses"):
        make_sizer("kelly_capped", CFG).fit([0.6, 0.2], [-0.01, 0.05])  # the only win is not approved


def test_ecdf_depends_only_on_train_probabilities():
    """The fitted ECDF is a function of the train inputs alone: scoring different test sets never changes it."""
    e = make_sizer("ecdf", CFG).fit(P_TRAIN, RET_TRAIN)
    grid = np.linspace(0.5, 1, 50)
    before = e.size(grid)
    e.size(RNG.uniform(0.5, 1, 10_000))
    np.testing.assert_array_equal(e.size(grid), before)
    np.testing.assert_array_equal(e.ref_, np.sort(P_TRAIN[P_TRAIN >= 0.5]))


def test_discretize():
    m = np.array([0.0, 0.04, 0.05, 0.06, 0.33, 0.97, 1.0, -0.33, -0.97])
    d = discretize(m, 0.1)
    np.testing.assert_allclose(d, [0, 0, 0, 0.1, 0.3, 1.0, 1.0, -0.3, -1.0])
    assert discretize([1.0], 0.1)[0] == 1.0  # exactly (fixed sizer stays all-in)
    np.testing.assert_array_equal(discretize(m, 0), m)
    assert (np.abs(discretize(RNG.uniform(-1, 1, 1000), 0.25)) <= 1).all()


# ── Backtest: fractional sizes ───────────────────────────────────────────────


def bars(oc: list[tuple[float, float]], start: str = "2024-01-02 09:30") -> pd.DataFrame:
    o, c = np.array(oc, dtype=float).T
    idx = pd.date_range(start, periods=len(oc), freq="5min")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, idx)


def signals_at(df: pd.DataFrame, pos: list[int], side: list[int], size: list[float], width: float = 0.5):
    idx = df.index[pos]
    return pd.DataFrame({"trade_signal": side, "signed_dir": side, "bet_size": size, "width": width}, index=idx).astype(
        {"trade_signal": int, "signed_dir": int}
    )


THREE = bars(
    [(100, 100), (100, 102), (102, 104), (104, 104), (105, 103), (103, 100), (100, 100), (100, 99), (99, 110)]
    + [(110, 110)] * 3
)
BT = CFG.replace(TIMEFRAME="5Min", HOLD_OVERNIGHT=False, VERTICAL_BARS=2, SIZE_STEP=0.05, SLIPPAGE_PCT=1e-4)


def test_fractional_sizing_three_trade_hand_example():
    """
    Events at bars 0, 3, 6 → enter at the next open, exit at the close of the 2nd held bar (barriers ±50% untouched).
    Slippage 1 bp adverse. Starting equity 10,000.
      T1 long 0.5:   49.99500 sh at 100.01 (= 0.5·10,000/100.01); exit 104·0.9999 = 103.9896 → equity 10,198.96010
      T2 short 0.25: 24.28567 sh at 104.9895 (= 0.25·10,198.96/104.9895); cover at 100.01 → 10,319.89058
      T3 long 1.0:  103.18859 sh at 100.01; exit 110·0.9999 = 109.989 → 11,349.60949
    Mid-trade marks: bar 1 = 10,000 + 49.995·(102 − 100.01) = 10,099.49005; bar 7 = 10,319.89 + 103.1886·(99 − 100.01).
    """
    sig = signals_at(THREE, [0, 3, 6], [1, -1, 1], [0.52, 0.24, 0.98])  # → 0.5, 0.25, 1.0 at step 0.05
    trades = simulate_trades(THREE, sig, BT, size_col="bet_size")
    np.testing.assert_allclose(trades["size"], [0.5, 0.25, 1.0])
    np.testing.assert_array_equal(trades["entry_pos"], [1, 4, 7])
    eq = equity_curve(THREE, trades, 10_000, 1.0)
    want = {
        0: 10_000,
        1: 10_099.4900509949,
        2: 10_198.9601039896,
        4: 10_247.276438306097,
        5: 10_319.890582365571,
        7: 10_215.670109530962,
        8: 11_349.6094916889,
        11: 11_349.6094916889,
    }
    for i, v in want.items():
        assert eq.iloc[i] == pytest.approx(v, rel=1e-12), i
    # The averaging engine gives the same curve when bets do not overlap
    eq2, _ = simulate_positions(THREE, sig, BT, size_col="bet_size")
    np.testing.assert_allclose(eq2.to_numpy(), eq.to_numpy(), rtol=1e-12)


def test_zero_size_bet_is_skipped_and_does_not_block():
    sig = signals_at(THREE, [0, 1, 3], [1, 1, -1], [0.01, 0.6, 0.3])  # 0.01 → 0 at step 0.05
    trades = simulate_trades(THREE, sig, BT, size_col="bet_size")
    assert list(trades.index) == list(THREE.index[[1, 3]])  # event 0 rounds to 0: no bet, bar 1's event enters
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        simulate_trades(THREE, sig.assign(bet_size=1.5), BT, size_col="bet_size")


def u6_equity(df, trades, init_cash, size):
    """The pre-U7 equity loop, verbatim (reference for the fixed-sizer regression)."""
    close = df["close"].to_numpy()
    equity = np.full(len(df), np.nan)
    cash = init_cash
    for e, x, side, entry, exit_ in trades[["entry_pos", "exit_pos", "side", "entry_fill", "exit_fill"]].itertuples(
        index=False
    ):
        equity[e - 1] = cash
        qty = size * cash / entry
        equity[e:x] = cash + side * qty * (close[e:x] - entry)
        cash += side * qty * (exit_ - entry)
        equity[x] = cash
    equity[0] = init_cash if np.isnan(equity[0]) else equity[0]
    return pd.Series(equity, index=df.index, name="equity").ffill()


def test_fixed_sizer_reproduces_pre_u7_backtest_bitwise():
    df = synthetic_daily(600)
    cfg = CFG.replace(SIZE=0.7)
    ev = df.index[RNG.choice(np.arange(5, 580), 150, replace=False)].sort_values()
    side = RNG.choice([-1, 0, 1], len(ev))
    p = RNG.uniform(0.3, 0.9, len(ev))
    sig = pd.DataFrame({"trade_signal": side, "signed_dir": np.where(side == 0, 1, side), "width": 0.03}, index=ev)
    sig["bet_size"] = make_sizer("fixed", cfg).size(p) * (side != 0)
    sig["trade_signal"] = np.where(p >= cfg.META_THRESH, sig["trade_signal"], 0)
    res = run_backtest(df, sig, cfg)
    for name, col in (("Meta-filtered", "trade_signal"), ("Primary only", "signed_dir")):
        cand = sig[sig[col] != 0]
        ex = barrier_exits(df, cand.index, cand["width"], side=cand[col], vertical_bars=10, hold_overnight=True)
        eq, trades = res[name]
        assert trades["entry_pos"].is_monotonic_increasing and len(trades) <= len(ex)
        assert eq.to_numpy().tobytes() == u6_equity(df, trades, cfg.INIT_CASH, cfg.SIZE).to_numpy().tobytes()


# ── Backtest: active-bet averaging ───────────────────────────────────────────


RAMP = bars([(100 + i, 101 + i) for i in range(10)])
AVG = BT.replace(VERTICAL_BARS=4, SIZE_STEP=0.1, POSITION_MODE="average")


def test_active_bet_averaging_hand_example_charges_every_resize():
    """
    Bet A (event bar 0, long 0.4) lives over bars 1–4 (exit at close[4] = 105), bet B (event bar 2, long 0.8)
    over bars 3–6 (exit at close[6] = 107). Target: 0.4 from open[1]; mean(0.4, 0.8) = 0.6 from open[3];
    0.8 after A's exit at close[4]; 0 after B's exit at close[6]. Each change is filled with 1 bp adverse slippage.
    """
    s = 1e-4
    sig = signals_at(RAMP, [0, 2], [1, 1], [0.4, 0.8])
    eq, bets = simulate_positions(RAMP, sig, AVG, size_col="bet_size")
    assert len(bets) == 2 and list(bets["exit_pos"]) == [4, 6]

    cash, q, traded = 10_000.0, 0.0, 0.0

    def trade(f, P):
        nonlocal cash, q, traded
        E = cash + q * P
        fill = P * (1 + np.sign(f * E / P - q) * s)
        q_new = f * E / fill
        cash -= (q_new - q) * fill
        traded += abs(q_new - q) * fill / E
        q = q_new

    marks = {0: 10_000.0}
    trade(0.4, 101)  # open[1]
    marks |= {1: cash + q * 102, 2: cash + q * 103}
    q_before = q
    trade(0.6, 103)  # open[3]: resize, pays 1 bp on the added notional only
    assert cash == pytest.approx(10_000 - q_before * 101 * (1 + s) - (q - q_before) * 103 * (1 + s))
    marks[3] = cash + q * 104
    trade(0.8, 105)  # close[4], A's vertical exit
    marks |= {4: cash + q * 105, 5: cash + q * 106}
    trade(0.0, 107)  # close[6]
    marks |= {6: cash, 9: cash}
    for i, v in marks.items():
        assert eq.iloc[i] == pytest.approx(v, rel=1e-12), i
    assert eq.attrs["turnover"] == pytest.approx(traded, rel=1e-12)
    assert eq.attrs["exposure"] == pytest.approx(6 / 10)  # bars 1–6 held
    # Without slippage the same bets end strictly higher (every change above paid 1 bp)
    eq0, _ = simulate_positions(RAMP, sig, AVG.replace(SLIPPAGE_PCT=0.0), size_col="bet_size")
    assert eq0.iloc[-1] > eq.iloc[-1]


def test_averaging_opposite_sides_net_out_and_small_changes_do_not_trade():
    sig = signals_at(RAMP, [0, 1], [1, -1], [0.5, 0.5])
    eq, _ = simulate_positions(RAMP, sig, AVG, size_col="bet_size")
    # mean(+0.5, −0.5) = 0 while both live (open[2] → close[4]): flat, so equity is constant over bars 2–3;
    # after A's exit at close[4] the position is B's short 0.5 alone
    assert eq.iloc[1] != eq.iloc[2] == eq.iloc[3]
    assert eq.iloc[5] < eq.iloc[4]  # short on a rising ramp
    # 0.42 and 0.44 both discretize to 0.4: B's arrival and A's exit do not trade (turnover = entry + exit only)
    sig = signals_at(RAMP, [0, 2], [1, 1], [0.42, 0.44])
    eq, _ = simulate_positions(RAMP, sig, AVG, size_col="bet_size")
    single, _ = simulate_positions(
        RAMP, signals_at(RAMP, [0], [1], [0.4]), AVG.replace(VERTICAL_BARS=6), "trade_signal", "bet_size"
    )
    np.testing.assert_allclose(eq.to_numpy(), single.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("mode", ["single", "average"])
def test_sized_backtest_is_causal(mode):
    """Perturbing bars and signals after bar c leaves equity up to c unchanged (size acts from entry at t+1)."""
    df = synthetic_daily(500)
    cfg = CFG.replace(POSITION_MODE=mode)
    ev = df.index[np.sort(RNG.choice(np.arange(5, 480), 200, replace=False))]
    sig = pd.DataFrame(
        {"trade_signal": RNG.choice([-1, 1], 200), "bet_size": RNG.uniform(0, 1, 200), "width": 0.03}, index=ev
    )
    sig["signed_dir"] = sig["trade_signal"]
    c = 300
    df2 = random_walk_after(df, c, seed=3)
    late = sig.index > df.index[c]
    sig2 = sig.copy()
    sig2.loc[late, "bet_size"] = RNG.uniform(0, 1, late.sum())
    sig2.loc[late, "trade_signal"] = -sig2.loc[late, "trade_signal"]
    a, b = run_backtest(df, sig, cfg), run_backtest(df2, sig2, cfg)
    for k in a:
        np.testing.assert_array_equal(a[k][0].iloc[: c + 1].to_numpy(), b[k][0].iloc[: c + 1].to_numpy())
    # and a size change at event t first moves equity at bar t+1 (entry at open[t+1]), never at t
    t = df.index.get_loc(ev[50])
    sig3 = sig.copy()
    sig3.iloc[50, sig3.columns.get_loc("bet_size")] = 1.0 if sig.iloc[50]["bet_size"] < 0.5 else 0.2
    e1, e3 = run_backtest(df, sig, cfg)["Meta-filtered"][0], run_backtest(df, sig3, cfg)["Meta-filtered"][0]
    diff = np.flatnonzero(e1.to_numpy() != e3.to_numpy())
    assert diff.size == 0 or diff[0] >= t + 1


# ── WFO integration ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_daily(1400)


def wfo_cfg(**over) -> RunConfig:
    small = {k: {**getattr(CFG, k), "n_estimators": 50} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
    return RunConfig.for_timeframe("1Day", **{"PRIMARY": "wavelet_trend", **WFO_CFG, **small, **over})


def test_wfo_bet_size_columns(daily):
    cfg = wfo_cfg(SIZER="ecdf")
    sig = run_wfo(daily, cfg, sizers=("fixed", "kelly_capped", "ecdf"))
    assert {"bet_size", "bet_size:fixed", "bet_size:kelly_capped"} <= set(sig.columns)
    assert "bet_size:ecdf" not in sig  # cfg.SIZER is `bet_size`
    for col in ("bet_size", "bet_size:fixed", "bet_size:kelly_capped"):
        assert sig[col].between(0, 1).all()
        assert (sig.loc[sig["trade_signal"] == 0, col] == 0).all()
    np.testing.assert_array_equal(sig["bet_size:fixed"], (sig["trade_signal"] != 0).astype(float))
    assert isinstance(sig.attrs["sizer_skipped_folds"], list) and "sizer_skipped_folds:kelly_capped" in sig.attrs
    pd.testing.assert_frame_equal(sig, run_wfo(daily, cfg, sizers=("fixed", "kelly_capped", "ecdf")))


def test_wfo_fixed_sizer_leaves_signals_unchanged(daily):
    """Adding sizers changes nothing but the size columns."""
    cfg = wfo_cfg()
    a = run_wfo(daily, cfg)
    b = run_wfo(daily, cfg, sizers=("ecdf",))
    pd.testing.assert_frame_equal(a, b.drop(columns=["bet_size:ecdf"]), check_like=False)
    np.testing.assert_array_equal(a["bet_size"], (a["trade_signal"] != 0).astype(float))


@pytest.mark.parametrize("meta_train", ["val", "oof"])
def test_sizer_inputs_are_train_window_oof(daily, monkeypatch, meta_train):
    """ECDF / Kelly are fit on OOF meta-probabilities of the fold's fitting events only (purged at the fitting split's
    end, never a test event), each predicted by a meta-model whose training rows exclude it and every overlapping span."""
    cfg = wfo_cfg(SIZER="ecdf", META_TRAIN=meta_train, META_MODEL="logit_l2")
    seen, fits = [], []
    fit = sizers_mod.Ecdf.fit

    def spy_fit(self, p_oof, ret):
        seen.append((p_oof.copy(), ret.copy()))
        return fit(self, p_oof, ret)

    orig_oof = meta_model.oof_meta_prob

    def spy_oof(X_fit, prim_fit, lbl, w, cfg, spans):
        fits.append(spans.loc[lbl.index])
        return orig_oof(X_fit, prim_fit, lbl, w, cfg, spans)

    monkeypatch.setattr(sizers_mod.Ecdf, "fit", spy_fit)
    monkeypatch.setattr(wfo_engine, "oof_meta_prob", spy_oof)
    sig = run_wfo(daily, cfg)
    folds = wfo_folds(daily.index, cfg)
    assert len(seen) == len(folds) - len(sig.attrs["meta_skipped_folds"])
    assert not sig.attrs["meta_skipped_folds"]
    for k, ((p, ret), spans, f) in enumerate(zip(seen, fits, folds), start=1):
        assert p.index.equals(spans.index)
        assert spans.equals(purged(spans, 0, f.val_end, f.val_embargo))  # every exit before the fitting split's end
        assert not p.index.isin(sig.index[sig["fold"] == k]).any()  # none of the fold's test events
        assert p.index.max() < daily.index[f.val_end]
        assert ret.index.equals(p.index) and np.isfinite(ret).all()


def test_oof_meta_prob_is_out_of_fold(daily, monkeypatch):
    """Each OOF probability comes from a model that never trained on that event or any span overlapping its block."""
    cfg = wfo_cfg()
    n = 300
    rng = np.random.default_rng(1)
    idx = daily.index[:n]
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=list("abc"))
    prim = pd.DataFrame({"signed_dir": rng.choice([-1, 1], n)}, index=idx)
    y = pd.Series((X["a"] + rng.normal(0, 1, n) > 0).astype(int), index=idx)
    w = pd.Series(1.0, index=idx)
    spans = pd.DataFrame({"entry_pos": np.arange(n) + 1, "exit_pos": np.arange(n) + 6}, index=idx)
    trained = []
    from models.zoo import Legacy

    orig = Legacy.fit

    def spy(self, Xm, yy, ww, cv=None):
        trained.append(Xm.index)
        return orig(self, Xm, yy, ww, cv)

    monkeypatch.setattr(Legacy, "fit", spy)
    p = oof_meta_prob(X, prim, y, w, cfg, spans)
    assert p.between(0, 1).all() and len(trained) == cfg.ZOO_CV_SPLITS
    from models.zoo import inner_cv

    for tr_idx, (_, test) in zip(trained, inner_cv(spans, cfg).split(X)):
        te = spans.iloc[test]
        tr = spans.loc[tr_idx]
        assert not tr_idx.isin(te.index).any()
        a, b = te["entry_pos"].min() - 1, te["exit_pos"].max()
        assert ((tr["exit_pos"] < a) | (tr["entry_pos"] - 1 > b)).all()


def test_sizer_fit_failure_is_a_counted_no_trade_fold(daily, monkeypatch):
    def boom(self, p, r):
        raise SizerFitError("forced")

    monkeypatch.setattr(sizers_mod.KellyCapped, "fit", boom)
    sig = run_wfo(daily, wfo_cfg(SIZER="kelly_capped"))
    scored = sorted(set(sig["fold"]) - set(sig.attrs["meta_skipped_folds"]))
    assert sig.attrs["sizer_skipped_folds"] == scored
    assert (sig["bet_size"] == 0).all()


def test_wfo_with_sizer_is_causal(daily):
    cfg = wfo_cfg(PRIMARY="donchian_breakout", SIZER="kelly_capped", META_TRAIN="oof")
    folds = wfo_folds(daily.index, cfg)
    ev = daily.index[folds[-1].val_end + 30]
    c = daily.index.get_loc(ev)
    a = run_wfo(daily, cfg, sizers=("ecdf",))
    b = run_wfo(random_walk_after(daily, c, seed=5), cfg, sizers=("ecdf",))
    pd.testing.assert_frame_equal(a.loc[:ev], b.loc[:ev])
    assert (a.loc[:ev, "bet_size"] > 0).any()


# ── Review regressions (U7 adversarial review) ───────────────────────────────


def ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    o, h, lo, c = np.array(rows, dtype=float).T
    idx = pd.date_range("2024-01-02 09:30", periods=len(rows), freq="5min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, idx)


def test_same_bar_exits_fill_intrabar_barriers_before_the_close():
    """A (wide) exits at the vertical close[3] = 101; B hits its upper barrier 102 inside bar 3. The barrier fill
    happens first (target mean → A alone = 1.0 at 102), then the close (flat at 101): 10,018.01, not 10,077.91."""
    df = ohlc(
        [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100), (100, 103, 99.8, 101)]
        + [(101, 101, 101, 101)] * 2
    )
    cfg = AVG.replace(VERTICAL_BARS=3)
    sig = signals_at(df, [0, 1], [1, 1], [1.0, 0.2]).assign(width=[0.5, 0.02])
    eq, bets = simulate_positions(df, sig, cfg, size_col="bet_size")
    assert list(bets["barrier"]) == ["vertical", "upper"] and list(bets["exit_pos"]) == [3, 3]
    s, cash, q = 1e-4, 10_000.0, 0.0
    for f, P in ((1.0, 100), (0.6, 100), (1.0, 102), (0.0, 101)):
        E = cash + q * P
        fill = P * (1 + np.sign(f * E / P - q) * s)
        q_new = f * E / fill
        cash, q = cash - (q_new - q) * fill, q_new
    assert eq.iloc[-1] == pytest.approx(cash, rel=1e-12)
    assert eq.iloc[-1] == pytest.approx(10_018.01, abs=0.01)


def test_step_zero_leaves_no_float_residue_position():
    """0.3 − 0.1 − 0.2 = −2.8e-17 in floats: with SIZE_STEP = 0 the netted-out target must still be flat."""
    ramp = bars([(100 + i, 101 + i) for i in range(12)])
    sig = signals_at(ramp, [0, 1, 2], [1, -1, -1], [0.3, 0.1, 0.2])
    eq, _ = simulate_positions(ramp, sig, AVG.replace(VERTICAL_BARS=6, SIZE_STEP=0.0), size_col="bet_size")
    # live: A bars 1–6, B 2–7, C 3–8; mean 0 → flat from open[3] until A's exit at close[6]
    assert eq.iloc[3] == eq.iloc[4] == eq.iloc[5]
    # held: bars 1–3 (flattened at open[3]) and 6–8 (short −0.15 then −0.2 after A's exit) = 6 of 12 bars
    assert eq.attrs["exposure"] == pytest.approx(6 / 12)


@pytest.mark.parametrize("step", [0.3, 0.4, 0.7, 1.5, -0.1])
def test_size_step_must_keep_full_bets_whole(step):
    with pytest.raises(ValueError, match="SIZE_STEP"):
        CFG.replace(SIZE_STEP=step)
    for ok in (0, 0.05, 0.1, 0.125, 0.25, 0.5, 1.0):
        assert discretize([1.0], CFG.replace(SIZE_STEP=ok).SIZE_STEP)[0] == 1.0


def test_threshold_is_applied_in_the_probability_dtype():
    """Legacy meta_prob is float32: an exact float32(τ) tie that meta_predict approves must get a non-zero size."""
    cfg = CFG.replace(META_THRESH=0.52)
    p = np.array([0.52], dtype=np.float32)
    assert (p >= cfg.META_THRESH)[0]  # meta_predict's comparison (pandas / numpy, float32)
    assert make_sizer("fixed", cfg).size(p)[0] == 1.0


def test_avg_bet_size_means_the_same_in_both_modes():
    sig = signals_at(THREE, [0, 3, 6], [1, -1, 1], [0.52, 0.24, 0.98])
    eq1 = equity_curve(THREE, simulate_trades(THREE, sig, BT, size_col="bet_size"), 10_000, 1.0)
    eq2, _ = simulate_positions(THREE, sig, BT, size_col="bet_size")
    want = (0.5 * 1 + 0.25 * 1 + 1.0 * 1) / 3  # each trade holds one close (entry bar) before its exit bar
    assert eq1.attrs["avg_position"] == pytest.approx(want) and eq2.attrs["avg_position"] == pytest.approx(want)
