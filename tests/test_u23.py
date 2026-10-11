"""
U23 (SPEC §23, PLAN3 §5 U23): the intraday kernels against their oracles (scipy's repeated median, the least-squares
slope, a pandas band / VWAP / σ reference), causality of every kernel and of the region primary with planted one-bar
peeks, session-bounded windows, the region primary's schedule and stop-and-reverse state, and the position backtest's
parity with the portfolio simulator through the rule pass (synthetic with the loss gate firing; cached SPY 2024).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import siegelslopes

import features.kernels as K
from features.events import sample_events
from features.exits import exit_frame
from primaries import make_primary
from primaries.mechanism import primary_config
from risk.portfolio import simulate_portfolio
from risk.profiles import RiskProfile, get_profile
from tests.test_risk import const_costs
from tests.test_u14 import NY, intraday
from tests.test_u22 import _assert_fill_timing, _NoNetwork
from wfo.position_backtest import decision_grid, position_backtest
from wfo.rule_pass import rule_signals

NS = [6, 9, 12, 18, 24]


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=NY)


def _layout(df):
    return K.session_layout(df.index, 5)


def _walk(n_sessions: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    df = intraday(n_sessions, seed=seed)
    _, start, _, _ = _layout(df)
    return np.log(df["close"].to_numpy()), start


def perturb_after(df: pd.DataFrame, c: int, seed: int) -> pd.DataFrame:
    """Rows > c replaced by an independent random walk (prices) and random volume."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = len(df) - c - 1
    close = df["close"].iloc[c] * np.exp(np.cumsum(rng.normal(0, 0.004, k)))
    open_ = close * np.exp(rng.normal(0, 0.001, k))
    for col, v in (("close", close), ("open", open_), ("high", np.maximum(open_, close) * 1.001),
                   ("low", np.minimum(open_, close) * 0.999), ("volume", rng.integers(1e4, 1e7, k).astype(float))):  # fmt: skip
        out.iloc[c + 1 :, out.columns.get_loc(col)] = v
    return out


# ── Oracles ──────────────────────────────────────────────────────────────────


def test_rmedv_equals_scipy_siegelslopes_on_4400_pairs_and_the_meyers_worked_examples():
    x, start = _walk(60, seed=1)
    out = K.rmedv_all(x, start, NS)
    rng = np.random.default_rng(2)
    checked, worst = 0, 0.0
    for a, n in enumerate(NS):
        ok = np.flatnonzero(np.arange(len(x)) - start >= n - 1)
        for t in rng.choice(ok, 880, replace=False):
            ref = siegelslopes(x[t - n + 1 : t + 1], np.arange(n, dtype=float)).slope
            worst = max(worst, abs(out[a, t] - ref))
            checked += 1
    assert checked == 4400 and worst <= 1e-9
    assert worst == 0.0  # the same pairwise slopes and medians: bit-identical in practice
    # Meyers (2005 p.2; 2025 p.2): repeated median slope exactly 1.0 despite the planted outliers
    for y in ([1, 2, 3, 4, 5, 15, 12, 8, 9, 10], [1, 2, 10, 4, 5, 6, 7, 8, 9, 18, 11, 12, 13, 18, 15, 20]):
        y = np.asarray(y, dtype=float)
        assert K.rmedv_all(y, np.zeros(len(y), np.int64), [len(y)])[0, -1] == 1.0
    ramp = np.arange(40) * 3.0 + 17.0
    r = K.rmedv_all(ramp, np.zeros(40, np.int64), NS)
    for a, n in enumerate(NS):
        assert np.all(np.isnan(r[a, : n - 1])) and np.allclose(r[a, n - 1 :], 3.0, rtol=0, atol=1e-12)
    assert np.allclose(K.rmedv_all(-ramp, np.zeros(40, np.int64), [6])[0, 5:], -3.0)
    # even counts average the two middle values (scipy / numpy), not the lower one
    y = np.array([0.0, 1.0, 5.0, 6.0])
    assert K.rmedv_all(y, np.zeros(4, np.int64), [4])[0, -1] == siegelslopes(y).slope == 2.25


def test_sg_velocity_degree_one_is_the_least_squares_slope_and_higher_degrees_the_next_bar_derivative():
    x, start = _walk(40, seed=3)
    out = K.sg_velocity_all(x, start, NS, 1)
    worst = 0.0
    for a, n in enumerate(NS):
        for t in range(n - 1 + 78 * 5, len(x), 7):
            if t - start[t] >= n - 1:
                worst = max(worst, abs(out[a, t] - np.polyfit(np.arange(n), x[t - n + 1 : t + 1], 1)[0]))
    assert worst <= 1e-12
    u = np.arange(30, dtype=float)
    z = np.zeros(30, np.int64)
    # exact on polynomials: Meyers' Velocity(T+1) is the fit's derivative one bar past the window (offset 1); offset 0
    # is the derivative at the last point
    for deg, f, df_ in ((2, lambda t: 0.5 * t**2 - t, lambda t: t - 1), (3, lambda t: t**3 / 100 + t,
                                                                          lambda t: 3 * t**2 / 100 + 1)):  # fmt: skip
        for off in (1, 0):
            v = K.sg_velocity_all(f(u), z, [9, 12], deg, off)
            assert np.allclose(v[:, 11:], df_(u + off)[11:], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(K.sg_weights(12, 1, 1), K.sg_weights(12, 1, 0), rtol=1e-12)  # degree 1: any offset
    assert abs(K.sg_weights(12, 2).sum()) < 1e-14
    with pytest.raises(ValueError, match="degree"):
        K.sg_weights(4, 4)


def _with_holes() -> pd.DataFrame:
    """40 sessions with a missing bar, a 13:00 early close and missing afternoon bars (the band skips them)."""
    df = intraday(40, seed=5)
    drop = [df.index[78 * 7 + 30], *df.index[78 * 12 + 42 : 78 * 13], df.index[78 * 30 + 77]]
    return df.drop(drop)


def _band_reference(df: pd.DataFrame, lookback=14, min_count=7) -> tuple[pd.Series, pd.Series]:
    day = df.index.tz_convert(NY).normalize()
    slot = ((df.index.tz_convert(NY).hour * 60 + df.index.tz_convert(NY).minute) - 570) // 5
    open_d = df.groupby(day)["open"].transform("first")
    move = (df["close"] / open_d - 1).abs()
    grid = pd.DataFrame({"d": day, "b": slot, "m": move.to_numpy()}).pivot(index="d", columns="b", values="m")
    band_g = grid.rolling(lookback, min_periods=min_count).mean().shift(1)
    band_g.iloc[:lookback] = np.nan
    band = pd.Series(band_g.stack(future_stack=True).reindex(list(zip(day, slot))).to_numpy(), index=df.index)
    prev_c = df.groupby(day)["close"].last().shift(1).reindex(day).to_numpy()
    hi, lo = np.maximum(open_d.to_numpy(), prev_c), np.minimum(open_d.to_numpy(), prev_c)
    c = df["close"].to_numpy()
    dist = []
    for ci, h, l_, b in zip(c, hi, lo, band.to_numpy()):
        if np.isnan(b) or np.isnan(h):
            dist.append(np.nan)
        elif ci > h:
            dist.append((ci / h - 1) / b)
        elif ci < l_:
            dist.append((ci / l_ - 1) / b)
        else:
            dist.append(0.0)
    return pd.Series(dist, index=df.index), band


def test_band_state_vwap_and_prior_sd_match_pandas_references():
    df = _with_holes()
    sess, _, _, slot = _layout(df)
    dist, band = K.band_state(df["open"], df["close"], sess, slot, 14)
    ref_d, ref_b = _band_reference(df)
    last20 = sess >= sess.max() - 19
    assert np.isfinite(dist[last20]).mean() > 0.95
    np.testing.assert_allclose(band[last20], ref_b.to_numpy()[last20], rtol=1e-12, atol=0)
    np.testing.assert_allclose(dist[last20], ref_d.to_numpy()[last20], rtol=1e-10, atol=1e-13)
    assert np.all(np.isnan(band[sess < 14]))
    day = df.index.tz_convert(NY).normalize()
    pv = (df["close"] * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()
    np.testing.assert_allclose(K.session_vwap(df["close"], df["volume"], sess), pv.to_numpy(), rtol=1e-13)
    x = np.log(df["close"].to_numpy())
    v = K.sg_velocity_all(x, _layout(df)[1], [6, 12], 1)
    sd = K.prior_sd(v, sess, 5)
    for d in (5, 13, 31, 39):
        for a in range(2):
            want = np.std(v[a, (sess >= d - 5) & (sess < d)][np.isfinite(v[a, (sess >= d - 5) & (sess < d)])], ddof=1)
            assert np.allclose(sd[a, sess == d], want, rtol=1e-12)
    assert np.all(np.isnan(sd[:, sess < 5]))
    np.testing.assert_array_equal(K.prior_sd(v, sess, np.int64(5)), sd)  # numpy integers are integers
    np.testing.assert_array_equal(K.band_state(df["open"], df["close"], sess, slot, np.int64(14))[0], dist)


# ── Session boundaries and causality ─────────────────────────────────────────


def test_a_window_never_crosses_a_session_a_planted_overnight_gap_leaves_the_nth_velocity_unaffected():
    df = intraday(10, seed=6)
    gapped = df.copy()
    gapped.iloc[78 * 6 :, :4] *= 1.05  # +5 % overnight gap into session 6, carried by every later bar
    for d in (df, gapped):
        assert np.isfinite(np.log(d["close"])).all()
    sess, start, pos, _ = _layout(df)
    s6 = sess == 6
    for fn in (lambda x: K.rmedv_all(x, start, NS), lambda x: K.sg_velocity_all(x, start, NS, 1),
               lambda x: K.sg_velocity_all(x, start, NS, 2)):  # fmt: skip
        a, b = fn(np.log(df["close"].to_numpy())), fn(np.log(gapped["close"].to_numpy()))
        for k, n in enumerate(NS):
            assert np.all(np.isnan(b[k, s6 & (pos < n - 1)]))  # the first N − 1 bars of the session
            assert np.isfinite(b[k, s6 & (pos >= n - 1)]).all()
            # from the N-th bar on the window is inside the session: the gap does not reach it
            np.testing.assert_allclose(b[k, s6 & (pos >= n - 1)], a[k, s6 & (pos >= n - 1)], rtol=1e-8, atol=1e-13)
    # the test bites: windows allowed to straddle the open (start = 0) read the gap into the first N − 1 bars
    z = np.zeros(len(df), np.int64)
    a0, b0 = (K.rmedv_all(np.log(d["close"].to_numpy()), z, [6]) for d in (df, gapped))
    early = s6 & (pos < 5)
    assert np.isfinite(b0[0, early]).all() and np.abs(b0[0, early] - a0[0, early]).max() > 1e-3


def _kernels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    sess, start, _, slot = _layout(df)
    x = np.log(df["close"].to_numpy())
    dist, band = K.band_state(df["open"], df["close"], sess, slot, 14)
    return {
        "rmedv": K.rmedv_all(x, start, NS),
        "sgv1": K.sg_velocity_all(x, start, NS, 1),
        "sgv2": K.sg_velocity_all(x, start, NS, 2),
        "band": np.vstack([dist, band]),
        "vwap": K.session_vwap(df["close"], df["volume"], sess)[None, :],
        "sd": K.prior_sd(K.rmedv_all(x, start, [6]), sess, 5),
        "rmedv_norm": K.rmedv_normalized(K.rmedv_all(x, start, NS), NS, sess, 5),
        "poly_norm": K.poly_normalized(K.sg_velocity_all(x, start, NS, 2), sess, 5),
    }


def _assert_kernels_causal(df: pd.DataFrame, kernels, cuts) -> None:
    a = kernels(df)
    for k, c in enumerate(cuts):
        b = kernels(perturb_after(df, c, seed=30 + k))
        for name in a:
            np.testing.assert_array_equal(a[name][:, : c + 1], b[name][:, : c + 1], err_msg=f"{name} cut {c}")


CUTS = (78 * 16 + 9, 78 * 17 + 40, 78 * 19 + 0, 78 * 19 + 77)  # mid-session, a session's first and last bar


def test_every_kernel_is_causal_and_a_planted_one_bar_peek_in_any_kernel_is_caught():
    df = intraday(22, seed=7)
    _assert_kernels_causal(df, _kernels, CUTS)
    for name in _kernels(df):

        def peek(d, name=name):
            out = _kernels(d)
            # bar t reads bar t + 1's value; a trailing SD is constant within a session (a one-bar shift reads a value
            # known at the next open, which is no leak), so its plant reads the next session's: the current one included
            out[name] = np.roll(out[name], -78 if name == "sd" else -1, axis=1)
            return out

        with pytest.raises(AssertionError, match=name):
            _assert_kernels_causal(df, peek, CUTS)


def _cfg(params=None, **kw):
    """The region on short synthetic series: a 5-session normalization window (the default 21 would leave most of a
    30-session sample in warm-up)."""
    params = {"norm_sessions": 5, **(params or {})}
    return primary_config("region_trend", "5Min", params, META_MODEL="none", VOL_SPAN=20, **kw)


def test_region_trend_is_causal_at_each_decision_and_a_peeking_cell_is_caught(monkeypatch):
    df = intraday(26, seed=8)
    cfg = _cfg()
    p = make_primary(cfg)
    dec, m = p.cell_matrix(df, cfg)
    picks = dec[np.unique(np.linspace(len(dec) // 2, len(dec) - 2, 8).astype(int))]

    def check(p):
        dec_a, m_a = p.cell_matrix(df, cfg)
        for k, t in enumerate(picks):
            dec_b, m_b = p.cell_matrix(perturb_after(df, int(t), seed=50 + k), cfg)
            i = int(np.flatnonzero(dec_a == t)[0])
            assert dec_b[i] == t
            np.testing.assert_array_equal(m_a[: i + 1], m_b[: i + 1], err_msg=f"decision {df.index[t]}")

    check(p)
    assert (m != 0).mean() > 0.2  # the cells trade on this walk
    orig = K.rmedv_all
    monkeypatch.setattr(K, "rmedv_all", lambda x, st, ns: np.roll(orig(x, st, ns), -1, axis=1))  # a one-bar peek
    with pytest.raises(AssertionError, match="decision"):
        check(make_primary(cfg))


def test_region_rule_is_the_mean_cell_and_the_schedule_decides_at_the_bar_ending_at_each_mark():
    df = intraday(20, seed=9)
    cfg = _cfg()
    p = make_primary(cfg)
    assert len(p.cell_names()) == 43 and p.marks()[0] == "10:00" and p.marks()[-1] == "15:30" and len(p.marks()) == 12
    sig = rule_signals(df, cfg, symbol="X")
    hhmm = sorted(set(sig.index.strftime("%H:%M")))
    assert hhmm[0] == "09:55" and hhmm[-1] == "15:25" and len(hhmm) == 12
    cells = p.cells(df, sig, cfg)
    target = cells.mean(axis=1).to_numpy()
    np.testing.assert_allclose(sig["trade_signal"] * sig["bet_size"], target, rtol=0, atol=1e-15)
    # the fill is at the next bar's open (10:00 for the 09:55 decision), the exit at the next decision's entry bar
    # or the closing auction; the U22 fill-timing invariant holds
    ev = sample_events(df, cfg).dropna(subset=["width"])
    side = pd.Series(np.where(np.arange(len(ev)) % 2 == 0, 1, -1), index=ev.index)
    frame = exit_frame(df, ev.index, ev["width"], cfg, side=side)
    _assert_fill_timing(df, cfg, frame, ev.index)
    assert set(frame["barrier"]) == {"time", "vertical"}
    vert = frame["exit_pos"].to_numpy()[frame["barrier"].to_numpy() == "vertical"]
    assert set(df.index[vert].strftime("%H:%M")) == {"15:55"}
    # exit 15:30: the 15:30 mark is a flat decision; cadence 15 adds the quarter hours
    p2 = make_primary(_cfg({"exit": "15:30"}))
    assert p2.marks()[-1] == "15:30"
    dec2, m2 = p2.cell_matrix(df, _cfg({"exit": "15:30"}))
    last = df.index[dec2].strftime("%H:%M") == "15:25"
    assert last.any() and (m2[last] == 0).all()
    # a session without its exit bar (a 13:00 early close; a missing 15:30 bar) is flat from its last decision's fill
    holes = df.drop([*df.index[78 * 15 + 42 : 78 * 16], df.index[78 * 17 + 72]])
    dec3, m3 = p2.cell_matrix(holes, _cfg({"exit": "15:30"}))
    d3 = holes.index[dec3].tz_convert(NY).normalize()
    for day in (d3.unique()[15], d3.unique()[17]):
        lastk = np.flatnonzero(d3 == day)[-1]
        assert holes.index[dec3[lastk]].strftime("%H:%M") in ("12:50", "14:55") and (m3[lastk] == 0).all()
    assert len(make_primary(_cfg({"cadence": 15})).marks()) == 24  # 10:00, 10:15 … 15:45
    for bad in ({"first": "09:30"}, {"exit": "09:45"}, {"estimators": ["band", "band"]}, {"cadence": 7}):
        with pytest.raises(ValueError):
            _cfg(bad)


def test_stop_and_reverse_state_never_crosses_a_session_and_flat_inside_and_vwap_stop_behave():
    v = np.array([[2.0, 0.1, np.nan, -0.2, 0.3, -2.0, 0.0]])
    close = np.array([10.0, 10, 10, 10, 10, 10, 10])
    vwap = np.full(7, 9.0)
    dsess = np.array([0, 0, 0, 0, 1, 1, 1])
    flat = np.zeros(7, bool)
    from primaries.region import _velocity_cells

    out = np.empty((7, 1))
    _velocity_cells(v, close, vwap, dsess, flat, np.array([1.0]), True, False, out)
    assert out[:, 0].tolist() == [1, 1, 1, 1, 0, -1, -1]  # held through no-signal / NaN; the new session starts flat
    _velocity_cells(v, close, vwap, dsess, flat, np.array([1.0]), False, False, out)
    assert out[:, 0].tolist() == [1, 0, 0, 0, 0, -1, 0]
    _velocity_cells(v, close, np.full(7, 11.0), dsess, flat, np.array([1.0]), True, True, out)
    assert out[:, 0].tolist() == [0, 0, 0, 0, 0, -1, -1]  # longs below the VWAP are stopped; the state stays flat
    _velocity_cells(v, close, vwap, dsess, np.array([0, 0, 1, 0, 0, 0, 0], bool), np.array([1.0]), True, False, out)
    assert out[:, 0].tolist() == [1, 1, 0, 0, 0, -1, -1]  # a forced-flat decision resets the state


def test_each_velocity_keeps_meyers_normalization_unit_sd_at_every_n_from_the_previous_sessions_only():
    df = intraday(60, seed=10)
    sess, start, _, _ = _layout(df)
    x = np.log(df["close"].to_numpy())
    raw_r, raw_p = K.rmedv_all(x, start, NS), K.sg_velocity_all(x, start, NS, 2)
    # by hand: RMedV · √N · mean_N 1 / sd(RMedV · √N); the polynomial velocity / its own SD (no √N)
    sc = raw_r * np.sqrt(NS)[:, None]
    want_r = sc * np.mean(1 / K.prior_sd(sc, sess, 21), axis=0)
    np.testing.assert_allclose(K.rmedv_normalized(raw_r, NS, sess, 21), want_r, rtol=1e-14)
    np.testing.assert_allclose(K.poly_normalized(raw_p, sess, 21), raw_p / K.prior_sd(raw_p, sess, 21), rtol=1e-14)
    # on a random walk the published scalings bring every N (and degree) to about one SD: per N exactly in the
    # calibration window (the polynomial), within the √N law's spread for the one-scalar RMedV multiplier
    later = sess >= 21
    for v, tol in ((K.rmedv_normalized(raw_r, NS, sess, 21), 0.2), (K.poly_normalized(raw_p, sess, 21), 0.15)):
        sds = [np.nanstd(row[later]) for row in v]
        assert max(abs(s_ - 1) for s_ in sds) < tol, sds
    # the scale of session d reads sessions d − k … d − 1 only: scaling session 30's moves leaves its own SD alone
    big = df.copy()
    s30 = sess == 30
    big.loc[s30, "close"] = np.exp(x[s30][0] + 3 * (x[s30] - x[s30][0]))
    b = K.prior_sd(K.rmedv_all(np.log(big["close"].to_numpy()), start, NS), sess, 21)
    a = K.prior_sd(raw_r, sess, 21)
    assert np.array_equal(a[:, sess <= 30], b[:, sess <= 30], equal_nan=True)
    assert not np.allclose(a[:, sess == 31], b[:, sess == 31])


# ── Position backtest: parity with the portfolio simulator ───────────────────


def _daily(eq: pd.Series) -> np.ndarray:
    d = eq.groupby(eq.index.tz_convert(NY).normalize()).last().to_numpy()
    return d / np.r_[eq.attrs.get("init", 10_000.0), d[:-1]] - 1


def _parity(df, cfg, profile, costs, prints=None, cash_yield=None, symbol="X"):
    sig = rule_signals(df, cfg, symbol=symbol)
    bars = df.loc[sig.index[0] :]
    c = costs(bars) if callable(costs) else costs
    eq, _, _ = simulate_portfolio({symbol: bars}, {symbol: sig}, cfg, profile, costs={symbol: c},
                                  prints={symbol: prints}, cash_yield=cash_yield)  # fmt: skip
    res = position_backtest(bars, sig["trade_signal"] * sig["bet_size"], cfg, costs=c, prints=prints,
                            cash_yield=cash_yield, profile=None if profile.name == "none" else profile)  # fmt: skip
    eq.attrs["init"] = cfg.INIT_CASH
    sim = _daily(eq)
    pb = res.ret.iloc[:, 0].to_numpy()
    assert len(sim) == len(pb)
    return sim, pb, eq, res


def test_position_backtest_reproduces_the_simulator_with_the_loss_gate_firing_and_the_cash_yield():
    df = intraday(30, seed=13, u_shape=True)
    df["close"] *= np.exp(np.random.default_rng(12).normal(0, 4e-3, len(df)))  # volatile: −2 % sessions occur
    days = pd.DatetimeIndex(sorted(set(df.index.tz_convert(NY).normalize().tz_localize(None))))
    cy = pd.Series(0.05, index=days)
    gate = RiskProfile("gate", daily_loss=0.02)
    for params in ({}, {"estimators": ["rmedv"], "n_list": [6], "theta_list": [0.75]}):
        cfg = _cfg(params, COST_MODEL="quotes", INIT_CASH=30_000)
        sims = {}
        for prof in (get_profile("none"), gate):
            sim, pb, eq, res = _parity(df, cfg, prof, lambda b: const_costs(b, 2e-4), cash_yield=cy)
            assert np.max(np.abs(sim - pb)) < 1e-10, (params, prof.name)
            assert res.cost.to_numpy().sum() == pytest.approx(eq.attrs["cost_paid"], rel=1e-12)
            assert res.notional.to_numpy().sum() == pytest.approx(eq.attrs["traded_notional"], rel=1e-12)
            sims[prof.name] = sim
        if params:  # the single cell loses > 2 % on some sessions: the gate is exercised, not just configured
            assert sims["none"].min() < -0.02 and np.abs(sims["none"] - sims["gate"]).max() > 1e-3
    with pytest.raises(ValueError, match="daily_loss only"):
        position_backtest(df, pd.Series(dtype=float), cfg, profile=get_profile("standard"))
    with pytest.raises(ValueError, match="scheduled decision"):
        position_backtest(df, pd.Series(1.0, index=df.index[:1]), cfg)
    with pytest.raises(ValueError, match="NaN"):
        position_backtest(df, pd.Series(np.nan, index=decision_grid(df, cfg).index[:3]), cfg)


def test_position_backtest_runs_many_series_and_scalar_costs_and_flat_series_earn_only_the_yield():
    df = intraday(8, seed=13)
    cfg = _cfg({}, INIT_CASH=10_000)
    grid = decision_grid(df, cfg)
    tg = pd.DataFrame({"long": 1.0, "flat": 0.0, "half": 0.5}, index=grid.index)
    days = grid.days
    res = position_backtest(df, tg, cfg, costs=0.0, cash_yield=pd.Series(0.036, index=days), grid=grid)
    gap = np.diff(days.to_numpy().astype("datetime64[D]").astype(int))
    want = np.cumprod(np.r_[1.0, 1 + 0.036 * gap / 360]) * 10_000
    np.testing.assert_allclose(res.equity["flat"], want, rtol=1e-14)
    # at zero cost, a constant long held 10:00 → close earns the session's 10:00 open → close (the closing print
    # absent: the bar close), compounded with the yield
    o = df["open"].to_numpy()[grid.dec[df.index[grid.dec].strftime("%H:%M") == "09:55"] + 1]
    c = df.groupby(df.index.normalize())["close"].last().to_numpy()
    assert (res.fills["long"] == 2).all() and (res.fills["half"] == 2).all()
    r_long = res.equity["long"].to_numpy() / np.r_[10_000, res.equity["long"].to_numpy()[:-1]]
    np.testing.assert_allclose(r_long, (1 + np.r_[0, 0.036 * gap / 360]) * (c / o), rtol=1e-12)
    # costs: a scalar one-way cost on every fill; the half-size series pays half the long's cost per unit equity
    rc = position_backtest(df, tg, cfg, costs=1e-4, grid=grid)
    assert (rc.cost["flat"] == 0).all() and rc.cost["half"].sum() == pytest.approx(rc.cost["long"].sum() / 2, rel=0.02)


@pytest.fixture(scope="module")
def spy_2024():
    from data.bars import DEFAULT_CACHE_DIR, load_bars, load_prints

    base = Path(DEFAULT_CACHE_DIR) / "sip" / "all"
    if not (base / "5Min" / "SPY.npz").exists() or not (base / "1DayPrint" / "SPY.npz").exists():
        pytest.skip("no cached SPY bars / prints")
    from data.exo import load_series
    from data.quotes import read_asof_table
    from experiments.runner import rate_asof

    df = load_bars("SPY", "5Min", "2024-01-01", "2025-01-01", source=_NoNetwork())
    prints = load_prints("SPY", "2024-01-01", "2025-01-01", source=_NoNetwork())
    days = pd.DatetimeIndex(sorted(set(df.index.tz_convert(NY).normalize().tz_localize(None))))
    try:
        dtb3 = load_series("fred", "DTB3", "2023-12-15", "2025-01-01", http=_NoNetwork())
        quotes = read_asof_table()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cached DTB3 / quotes table unavailable: {e}")
    return df, prints, rate_asof(dtb3, days) / 100.0, quotes[quotes["symbol"] == "SPY"].reset_index(drop=True)


@pytest.mark.parametrize("params", [{}, {"estimators": ["rmedv"], "n_list": [12], "theta_list": [1.0]},
                                    {"estimators": ["sgv"], "n_list": [24], "theta_list": [0.75]},
                                    {"estimators": ["band"], "vm_list": [1.25]}])  # fmt: skip
def test_position_backtest_equals_the_simulator_on_spy_2024_for_three_cells_and_the_region(spy_2024, params):
    """PLAN3 U23 done-when: max |Δ daily P&L| < 1e-10 at SIZE_STEP 0 (quotes as-of costs, closing prints, T-bill)."""
    from risk.costs import fill_costs

    df, prints, cy, quotes = spy_2024
    cfg = primary_config("region_trend", "5Min", params, META_MODEL="none", COST_MODEL="quotes", INIT_CASH=30_000)
    assert cfg.SIZE_STEP == 0 and cfg.FILL_AUCTION == "print" and cfg.COST_TABLE == "asof"
    for prof in (get_profile("none"), RiskProfile("gate", daily_loss=0.02)):
        sim, pb, eq, res = _parity(df, cfg, prof, lambda b: fill_costs(b.index, cfg, quotes), prints=prints,
                                   cash_yield=cy, symbol="SPY")  # fmt: skip
        assert np.max(np.abs(sim - pb)) < 1e-10
        assert res.cost.to_numpy().sum() == pytest.approx(eq.attrs["cost_paid"], rel=1e-12)
        assert res.missing_prints == 0 and eq.attrs["auction_fallbacks"] == 0
