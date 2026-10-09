"""
U8: risk layer and portfolio backtest — profile "none" reproduces the U7 backtest bitwise, caps hold (property test),
the daily loss gate flattens and blocks re-entry for the session, the vol target scales inversely with σ, the drawdown
throttle, costs on every notional change (entries, exits, trims, flattening), cross-symbol alignment and accounting,
causality, the Corwin–Schultz half-spread and the added metrics.
"""

import numpy as np
import pandas as pd
import pytest

from risk.costs import corwin_schultz, half_spread, session_spread
from risk.portfolio import simulate_portfolio
from risk.profiles import PROFILES, RiskProfile, get_profile
from tests.test_primaries import random_walk_after
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from wfo.backtest import equity_curve, run_backtest, simulate_trades
from wfo.wfo_metrics import strategy_metrics

CFG = RunConfig.for_timeframe("1Day")
NONE = PROFILES["none"]


def rand_signals(df: pd.DataFrame, n: int, rng, width=(0.02, 0.06), lo: int = 5, hi: int | None = None):
    hi = hi or len(df) - 20
    ev = df.index[np.sort(rng.choice(np.arange(lo, hi), n, replace=False))]
    side = rng.choice([-1, 1], n)
    sig = pd.DataFrame(
        {"trade_signal": side, "signed_dir": side, "bet_size": rng.uniform(0, 1, n), "width": rng.uniform(*width, n)},
        index=ev,
    )
    sig.loc[rng.uniform(size=n) < 0.2, "trade_signal"] = 0
    sig.loc[sig["trade_signal"] == 0, "bet_size"] = 0.0
    return sig


def const_spread(df: pd.DataFrame, h: float) -> pd.Series:
    return pd.Series(h, index=df.index)


def const_costs(df: pd.DataFrame, c) -> pd.DataFrame:
    """One-way cost c (a scalar or per-bar array) on every kind of fill (risk.costs.fill_costs layout)."""
    return pd.DataFrame({k: np.broadcast_to(np.asarray(c, float), len(df)) for k in ("open", "intra", "close")},
                        index=df.index)  # fmt: skip


# ── Profiles and config ──────────────────────────────────────────────────────


def test_profiles_and_config():
    assert not NONE.active and PROFILES["standard"].active
    s = PROFILES["standard"]
    assert (s.vol_target, s.max_position, s.max_symbol, s.max_gross, s.max_net, s.max_concurrent) == (
        0.005, 0.20, 0.25, 1.0, 1.0, 10
    )  # fmt: skip
    assert [s.dd_multiplier(d) for d in (0, 0.0999, 0.10, 0.15, 0.20, 0.5)] == [1, 1, 0.5, 0.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="unknown risk profile"):
        get_profile("nope")
    for bad in ({"vol_target": 0}, {"max_gross": 0}, {"max_concurrent": 0}, {"daily_loss": 1.0},
                {"dd_tiers": ((0.2, 0.5), (0.1, 0.0))}, {"spread_window_days": 0}):  # fmt: skip
        with pytest.raises(ValueError):
            RiskProfile("x", **bad)
    with pytest.raises(ValueError, match="unknown risk profile"):
        CFG.replace(RISK_PROFILE="nope")
    with pytest.raises(ValueError, match="POSITION_MODE"):
        CFG.replace(RISK_PROFILE="standard", POSITION_MODE="average")
    df = synthetic_daily(80)
    with pytest.raises(ValueError, match="cost data"):
        run_backtest(df, rand_signals(df, 10, np.random.default_rng(0)), CFG.replace(COST_MODEL="cs"))
    with pytest.raises(ValueError, match="COST_MODEL"):
        CFG.replace(COST_MODEL="spread")
    with pytest.raises(ValueError, match="POSITION_MODE"):
        CFG.replace(COST_MODEL="quotes", POSITION_MODE="average")


# ── Regression: profile "none" == U7 single-mode backtest ─────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_no_risk_layer_reproduces_u7_backtest_bitwise(seed):
    rng = np.random.default_rng(seed)
    df = synthetic_daily(700, seed=seed)
    cfg = CFG.replace(SIZE=[1.0, 0.7, 1.0, 0.5][seed], SIZE_STEP=[0.1, 0.05, 0, 1][seed])
    sig = rand_signals(df, 250, rng)
    for side, size in (("trade_signal", "bet_size"), ("signed_dir", None)):
        tr = simulate_trades(df, sig, cfg, side_col=side, size_col=size)
        eq = equity_curve(df, tr, cfg.INIT_CASH, cfg.SIZE)
        eq2, tr2, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, NONE, side_col=side, size_col=size)
        assert eq.to_numpy().tobytes() == eq2.to_numpy().tobytes()
        np.testing.assert_array_equal(tr["entry_pos"], tr2["entry_b"])
        np.testing.assert_array_equal(tr["exit_pos"], tr2["exit_b"])
        np.testing.assert_array_equal(tr["pnl_pct"], tr2["pnl_pct"])
        np.testing.assert_allclose(eq2.attrs["turnover"], eq.attrs["turnover"], rtol=1e-12)
        np.testing.assert_allclose(eq2.attrs["avg_position"], eq.attrs["avg_position"] * cfg.SIZE, rtol=1e-12)


# ── Caps (property test) ─────────────────────────────────────────────────────

CAPS = RiskProfile(
    "caps", max_position=0.3, max_symbol=0.35, max_gross=0.8, max_net=0.5, max_concurrent=3, drift_tol=0.1
)


def universe(k: int, n: int, seed: int, drop_every: int = 0):
    bars = {f"S{i}": synthetic_daily(n, seed=seed * 10 + i) for i in range(k)}
    if drop_every:  # a symbol with missing sessions (cross-symbol alignment)
        b = bars["S0"]
        bars["S0"] = b[np.arange(len(b)) % drop_every != 1]
    rng = np.random.default_rng(seed)
    sigs = {s: rand_signals(b, 120, rng) for s, b in bars.items()}
    return bars, sigs


@pytest.mark.parametrize("seed", range(6))
def test_caps_are_never_exceeded(seed):
    bars, sigs = universe(5, 500, seed, drop_every=4 if seed % 2 else 0)
    cfg = CFG.replace(SIZE=1.0, SIZE_STEP=0.1)
    eq, tr, log = simulate_portfolio(bars, sigs, cfg, CAPS)
    tol, eps = 1 + CAPS.drift_tol, 1e-12
    # positions: committed at entry within the position cap; after every open, on decision-time marks
    assert (tr["frac"] <= CAPS.position_cap + eps).all()
    assert (log["max_pos"] <= CAPS.position_cap * tol + eps).all()
    assert (log["gross"] <= CAPS.max_gross * tol + eps).all()
    assert (log["net"].abs() <= CAPS.max_net * tol + eps).all()
    assert (log["count"] <= CAPS.max_concurrent).all()
    # each side's gross ≤ max_net (so |net| holds whichever positions exit)
    assert (log[["long", "short"]] <= CAPS.max_net * tol + eps).all().all()
    # at every bar with an entry the aggregate caps hold exactly
    entry_bars = np.unique(tr["entry_b"].to_numpy())
    assert (log["gross"].to_numpy()[entry_bars] <= CAPS.max_gross + eps).all()
    assert (log["net"].abs().to_numpy()[entry_bars] <= CAPS.max_net + eps).all()
    # ... and each side that received an entry holds its cap exactly (the other side is only held to the drift band)
    for b, g in tr.groupby("entry_b"):
        for side in g["side"].unique():
            assert log[["long", "short"]].iloc[b]["long" if side > 0 else "short"] <= CAPS.max_net + eps
    # independent count from the trade intervals (a lower bound: exits at the bar itself are excluded)
    n = len(eq)
    held = np.zeros(n, int)
    for e, x in zip(tr["entry_b"], tr["exit_b"]):
        held[e:x] += 1
    assert held.max() <= CAPS.max_concurrent
    assert (log["gross"] > 0.5).any() and tr["frac"].max() > 0.29  # the caps actually bound
    # one position per symbol
    for s, g in tr.groupby("sym"):
        assert (g["entry_b"].to_numpy()[1:] > g["exit_b"].to_numpy()[:-1]).all()


def test_accounting_and_cross_symbol_alignment():
    """Equity on the union timeline = realized P&L + open positions marked at each symbol's last close."""
    bars, sigs = universe(3, 400, 7, drop_every=3)
    cfg = CFG.replace(SIZE=0.5)
    eq, tr, _ = simulate_portfolio(bars, sigs, cfg, NONE)
    union = eq.index
    assert union.equals(bars["S1"].index) and len(bars["S0"]) < len(union)
    last_close = {s: b["close"].reindex(union).ffill().to_numpy() for s, b in bars.items()}
    ref = np.full(len(union), float(cfg.INIT_CASH))
    for _, t in tr.iterrows():
        ref[t["exit_b"] :] += t["pnl"]
        span = slice(t["entry_b"], t["exit_b"])
        ref[span] += t["side"] * t["qty"] * (last_close[t["sym"]][span] - t["entry_fill"])
        # entry at the symbol's own next bar after the event, never at a union bar where it has no data
        b = bars[t["sym"]]
        assert union[t["entry_b"]] == b.index[b.index.get_loc(t.name) + 1]
        assert union[t["exit_b"]] in b.index
    np.testing.assert_allclose(eq.to_numpy(), ref, rtol=1e-12)
    assert set(tr["sym"]) == {"S0", "S1", "S2"}


# ── Vol target, drawdown throttle ────────────────────────────────────────────


def test_vol_target_scales_inversely_with_sigma():
    rng = np.random.default_rng(3)
    df = synthetic_daily(600)
    sig = rand_signals(df, 150, rng, width=(0.005, 0.2))
    vt = RiskProfile("vt", vol_target=0.04)
    cfg = CFG.replace(SIZE_STEP=0, BARRIER_MULT=1.5)
    _, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, vt)
    sigma_hold = sig.loc[tr.index, "width"] / cfg.BARRIER_MULT
    expected = cfg.SIZE * tr["size"] * np.minimum(1.0, vt.vol_target / sigma_hold)
    np.testing.assert_allclose(tr["frac"], expected, rtol=1e-12)
    hi = sigma_hold > vt.vol_target
    assert hi.sum() > 5 and (~hi).sum() > 5
    # doubling σ halves the notional (above the target)
    sig2 = sig.copy()
    sig2["width"] *= 2
    _, tr2, _ = simulate_portfolio({"X": df}, {"X": sig2}, cfg, vt)
    both = tr.index.intersection(tr2.index)
    both = both[(sig.loc[both, "width"] / cfg.BARRIER_MULT > vt.vol_target).to_numpy()]
    np.testing.assert_allclose(tr2.loc[both, "frac"] / tr.loc[both, "frac"], 0.5, rtol=1e-12)


def daily_bars(oc: list[tuple[float, float]]) -> pd.DataFrame:
    o, c = np.array(oc, dtype=float).T
    idx = pd.bdate_range("2024-01-02", periods=len(oc), tz="America/New_York")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, idx)


def sig_at(df, pos, side, size, width=0.5):
    side = np.asarray(side)
    return pd.DataFrame(
        {"trade_signal": side, "signed_dir": side, "bet_size": size, "width": width}, index=df.index[pos]
    )


def test_drawdown_throttle_tiers():
    # −15% trade → DD 15% → next bet at half size; its −29% move → DD > 20% → no further entries (none blocked)
    df = daily_bars(
        [(100, 100), (100, 85), (85, 85), (85, 85), (85, 60), (60, 60), (60, 60), (60, 61)] + [(61, 61)] * 4
    )
    cfg = CFG.replace(VERTICAL_BARS=2, SLIPPAGE_PCT=0.0)
    sig = sig_at(df, [0, 3, 6, 7], [1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])
    prof = RiskProfile("dd", dd_tiers=((0.10, 0.5), (0.20, 0.0)))
    eq, tr, log = simulate_portfolio({"X": df}, {"X": sig}, cfg, prof)
    assert list(tr["frac"]) == [1.0, 0.5]
    assert eq.iloc[2] == pytest.approx(8500) and eq.iloc[5] == pytest.approx(8500 - 0.5 * 8500 / 85 * 25)
    assert list(log["dd_mult"].iloc[[1, 3, 4, 6, 7]]) == [1.0, 0.5, 0.5, 0.0, 0.0]
    # the throttle applies after the position cap (SPEC order): 0.8 cap → −12% → min(1, 0.8) · 0.5, not min(0.5, 0.8)
    capped = RiskProfile("dd", max_position=0.8, dd_tiers=((0.10, 0.5), (0.20, 0.0)))
    _, tr2, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, capped)
    assert list(tr2["frac"].iloc[:2]) == [0.8, 0.4]


# ── Daily loss gate ──────────────────────────────────────────────────────────


def intraday(sessions: list[list[tuple[float, float]]]) -> pd.DataFrame:
    idx, rows = [], []
    for d, sess in enumerate(sessions):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        idx += list(pd.date_range(day + pd.Timedelta("09:30:00"), periods=len(sess), freq="1h", tz="America/New_York"))
        rows += sess
    o, c = np.array(rows, dtype=float).T
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, pd.DatetimeIndex(idx)
    )


def test_daily_loss_gate_flattens_and_blocks_the_session_intraday():
    df = intraday([[(100, 100), (100, 100), (100, 96), (96, 97), (97, 97), (97, 97), (97, 97)],
                   [(97, 97), (97, 98), (98, 99), (99, 99), (99, 99), (99, 99), (99, 99)]])  # fmt: skip
    cfg = RunConfig.for_timeframe("1Hour", VERTICAL_BARS=4, SLIPPAGE_PCT=1e-4)
    # long at open[1]; −4% at close[2] → gate; the events at 2 and 3 (entries at 3 and 4) are blocked; next session ok
    sig = sig_at(df, [0, 2, 3, 7], [1, 1, -1, 1], [1.0, 1.0, 1.0, 1.0])
    prof = RiskProfile("gate", daily_loss=0.02)
    eq, tr, log = simulate_portfolio({"X": df}, {"X": sig}, cfg, prof)
    assert log["gate"].iloc[2] == 1
    first, second = tr.iloc[0], tr.iloc[1]
    assert first["exit_reason"] == "gate" and first["exit_b"] == 3 and first["exit_px"] == 96
    assert first["exit_fill"] == 96 * (1 - 1e-4)
    assert len(tr) == 2 and second["entry_b"] == 8  # the next session's event trades
    assert (eq.iloc[3:8] == eq.iloc[3]).all()  # flat for the rest of the session
    # without the gate the first bet runs to its vertical barrier (blocking the events at 2 and 3 itself)
    _, tr0, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, NONE)
    assert tr0.iloc[0]["exit_b"] == 4 and list(tr0["entry_b"]) == [1, 8]
    # the event at 3 (entry at 4) comes after the gate's flatten at 3: only the gate blocks it
    assert tr.iloc[0]["exit_b"] < 4


def test_daily_loss_gate_blocks_the_next_session_on_daily_bars():
    df = daily_bars([(100, 100), (100, 97), (97, 97), (97, 98), (98, 99)] + [(99, 99)] * 6)
    cfg = CFG.replace(VERTICAL_BARS=3, SLIPPAGE_PCT=0.0)
    sig = sig_at(df, [0, 1, 2], [1, 1, 1], [1.0, 1.0, 1.0])
    _, tr, log = simulate_portfolio({"X": df}, {"X": sig}, cfg, RiskProfile("g", daily_loss=0.02))
    assert log["gate"].iloc[1] == 1
    assert tr.iloc[0]["exit_reason"] == "gate" and tr.iloc[0]["exit_b"] == 2
    assert len(tr) == 2 and tr.iloc[1]["entry_b"] == 3  # the event at 1 (entry at 2) is blocked, 2 → 3 trades


# ── Costs ────────────────────────────────────────────────────────────────────


def test_costs_on_every_notional_change_including_trims():
    """Entry, drift trim and exit each pay SLIPPAGE_PCT + half-spread on the traded notional (hand computed)."""
    df = intraday([[(100, 100), (100, 130), (131, 131), (131, 132), (132, 132), (132, 132), (132, 132)]])
    cfg = RunConfig.for_timeframe("1Hour", VERTICAL_BARS=4, SLIPPAGE_PCT=1e-4, SIZE_STEP=0.1)
    sig = sig_at(df, [0], [1], [0.5], width=0.9)
    h = 2e-4
    prof = RiskProfile("c", max_position=0.5, drift_tol=0.1)
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, prof, costs={"X": const_costs(df, 1e-4 + h)})
    c = 1e-4 + h
    E0 = 10_000.0
    fill = 100 * (1 + c)
    q = 0.5 * E0 / fill
    E1 = E0 + q * (130 - fill)
    expo = q * 130 / E1  # 0.565 > 0.5 · 1.1 → trim at open[2] back to 0.5 on decision marks
    assert expo > 0.55
    keep = 0.5 / expo
    trim_fill = 131 * (1 - c)
    cash = E0 + q * (1 - keep) * (trim_fill - fill)
    q2 = q * keep
    exit_fill = 132 * (1 - c)  # vertical barrier: close of the 4th held bar
    cash += q2 * (exit_fill - fill)
    t = tr.iloc[0]
    assert t["entry_fill"] == pytest.approx(fill, rel=1e-15) and t["trimmed"]
    assert t["exit_fill"] == pytest.approx(exit_fill, rel=1e-15)
    assert eq.iloc[4] == pytest.approx(cash, rel=1e-12)
    assert eq.iloc[2] == pytest.approx(cash - q2 * (exit_fill - fill) + q2 * (131 - fill), rel=1e-12)
    turnover = (
        q * fill / E0
        + q * (1 - keep) * trim_fill / (E1 + q * (131 - 130))
        + q2 * exit_fill / (cash - q2 * (exit_fill - fill) + q2 * (132 - fill))
    )
    assert eq.attrs["turnover"] == pytest.approx(turnover, rel=1e-12)


def test_spread_needs_estimates_at_every_fill():
    df = synthetic_daily(100)
    sig = rand_signals(df, 20, np.random.default_rng(1))
    prof = RiskProfile("c")
    hs = const_spread(df, 1e-4)
    hs.iloc[:50] = np.nan
    with pytest.raises(ValueError, match="no cost estimate"):
        simulate_portfolio({"X": df}, {"X": sig}, CFG, prof, costs={"X": const_costs(df, hs.to_numpy())})
    with pytest.raises(ValueError, match="cost frame for every symbol"):
        simulate_portfolio({"X": df}, {"X": sig}, CFG, prof, costs={})


# ── Causality ────────────────────────────────────────────────────────────────


def test_portfolio_backtest_is_causal():
    """Perturbing every symbol's bars and signals after bar c leaves equity (and risk decisions) up to c unchanged."""
    bars, sigs = universe(4, 500, 11)
    prof = RiskProfile("all", vol_target=0.02, max_position=0.4, max_gross=1.0, max_net=0.6, max_concurrent=3,
                       dd_tiers=((0.05, 0.5), (0.3, 0.0)), daily_loss=0.01)  # fmt: skip
    cfg = CFG.replace(SIZE_STEP=0.1)
    rng = np.random.default_rng(5)
    for c in (150, 300, 420):
        bars2, sigs2 = {}, {}
        for s, b in bars.items():
            bars2[s] = random_walk_after(b, c, seed=c + int(s[1:]))
            sg = sigs[s].copy()
            late = sg.index > b.index[c]
            sg.loc[late, "bet_size"] = rng.uniform(0, 1, late.sum())
            sg.loc[late, "trade_signal"] = -sg.loc[late, "trade_signal"]
            sigs2[s] = sg
        hs = {s: const_costs(b, const_spread(b, 1e-4).where(np.arange(len(b)) <= c, 9e-4)) for s, b in bars.items()}
        hs1 = {s: const_costs(b, 1e-4) for s, b in bars.items()}
        e1, _, l1 = simulate_portfolio(bars, sigs, cfg, prof, costs=hs1)
        e2, _, l2 = simulate_portfolio(bars2, sigs2, cfg, prof, costs=hs)
        np.testing.assert_array_equal(e1.iloc[: c + 1].to_numpy(), e2.iloc[: c + 1].to_numpy())
        pd.testing.assert_frame_equal(l1.iloc[: c + 1], l2.iloc[: c + 1])
        assert (l1["gate"] > 0).any() and (l1["dd_mult"] < 1).any()


# ── Corwin–Schultz half-spread ───────────────────────────────────────────────


def test_corwin_schultz_formula_and_gap_adjustment():
    df = pd.DataFrame({"high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.0, 101.0]},
                      index=pd.date_range("2024-01-02 09:30", periods=2, freq="5min"))  # fmt: skip
    beta = np.log(101 / 99) ** 2 + np.log(102 / 100) ** 2
    gamma = np.log(102 / 99) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    expected = max(0.0, 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
    s = corwin_schultz(df)
    assert np.isnan(s.iloc[0]) and s.iloc[1] == pytest.approx(expected, rel=1e-12)
    # a bar entirely above the prior close is shifted down by the gap: same estimate as the gap-free pair
    gap = df.copy()
    gap.loc[gap.index[1], ["high", "low"]] = [104.0, 102.0]
    ref = df.copy()
    ref.loc[ref.index[1], ["high", "low"]] = [102.0, 100.0]
    assert corwin_schultz(gap).iloc[1] == pytest.approx(corwin_schultz(ref).iloc[1], rel=1e-12)
    wide = df.copy()
    wide.loc[wide.index[1], ["high", "low"]] = [100.5, 90.0]  # 2-bar range ≫ 1-bar ranges → negative → 0
    assert corwin_schultz(wide).iloc[1] == 0.0


def five_min(n_days: int, seed: int, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    idx = pd.DatetimeIndex(
        [t for d in days for t in pd.date_range(d + pd.Timedelta("09:30:00"), periods=78, freq="5min")]
    ).tz_localize("America/New_York")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))
    o = np.r_[100, c[:-1]]
    spr = rng.uniform(2e-4, 1e-3, len(idx))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spr), "low": np.minimum(o, c) * (1 - spr),
                         "close": c, "volume": 1.0}, index=idx)  # fmt: skip


def test_half_spread_uses_prior_sessions_only():
    m5 = five_min(40, 0)
    daily_idx = pd.DatetimeIndex(sorted(set(m5.index.normalize())))
    hs = half_spread(daily_idx, m5, window_days=5, floor=0.5e-4)
    assert hs.iloc[:5].isna().all() and hs.iloc[5:].notna().all()
    sess = session_spread(m5)
    np.testing.assert_allclose(hs.iloc[5], max(0.5e-4, sess.iloc[0:5].mean() / 2), rtol=1e-12)
    # perturbing session d and later leaves the estimates of sessions ≤ d unchanged (hourly bars too)
    d = 20
    cut = int(np.searchsorted(m5.index, daily_idx[d]))
    m5b = random_walk_after(m5, cut - 1, seed=1)
    hourly_idx = m5.index[::12]
    for idx in (daily_idx, hourly_idx):
        a, b = half_spread(idx, m5, 5, 0.5e-4), half_spread(idx, m5b, 5, 0.5e-4)
        upto = idx.normalize() <= daily_idx[d]
        np.testing.assert_array_equal(a[upto].to_numpy(), b[upto].to_numpy())
        assert not np.array_equal(a[~upto].to_numpy(), b[~upto].to_numpy())
    # floor
    assert (half_spread(daily_idx, m5, 5, floor=1.0).dropna() == 1.0).all()


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_added_metrics():
    eq = pd.Series([100.0, 110, 99, 99, 105, 120, 108, 108, 109])
    trades = pd.DataFrame({"pnl": [10.0, -11, 21, -12], "pnl_pct": [0.1, -0.1, 0.2, -0.1], "bars_held": 2})
    m = strategy_metrics(eq, trades, bars_per_year=252)
    r = eq.pct_change().fillna(0.0)
    assert m["Sortino Ratio"] == pytest.approx(
        r.mean() * 252 / (np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252))
    )
    assert m["Max DD Duration (days)"] == 3  # closes 2, 3, 4 below the peak of 110
    assert m["Profit Factor"] == pytest.approx(31 / 23)
    assert np.isnan(m["Tail Ratio"])  # fewer than 20 non-zero returns
    assert 0 < m["PSR(0)"] < 1
    flat = strategy_metrics(pd.Series([100.0] * 5), trades.iloc[:0], 252)
    assert np.isnan(flat["PSR(0)"]) and np.isnan(flat["Profit Factor"])


def test_run_backtest_routes_through_the_risk_layer():
    rng = np.random.default_rng(2)
    df = synthetic_daily(300)
    sig = rand_signals(df, 60, rng)
    cfg = CFG.replace(RISK_PROFILE="standard", COST_MODEL="cs")
    m5 = five_min(300, 3, start="2012-01-03")  # same business-day calendar as synthetic_daily
    d = df.iloc[100:]
    res = run_backtest(d, sig, cfg, cost_data=m5)
    tr = res["Meta-filtered"][1]
    assert len(tr) > 10 and (tr["frac"] <= 0.2 + 1e-12).all()
    hs = half_spread(d.index, m5, 21, 0.5e-4).to_numpy()[tr["entry_b"].to_numpy()]
    np.testing.assert_allclose(tr["entry_fill"] / tr["entry_px"] - 1, tr["side"] * (cfg.SLIPPAGE_PCT + hs), rtol=1e-9)


def test_entry_decisions_at_an_open_do_not_depend_on_that_open():
    """Review MINOR 1: slots and gross/net headroom for fills at open[b] are decided at close[b−1] — positions that
    gap through a barrier at open[b] still occupy their slot and exposure for that decision."""
    bars, sigs = universe(6, 400, 21)
    for s in sigs:
        sigs[s]["width"] = 0.012  # narrow barriers: many gap exits at the open
        sigs[s]["trade_signal"] = sigs[s]["signed_dir"]
        sigs[s]["bet_size"] = 1.0
    prof = RiskProfile("tight", max_position=0.3, max_gross=0.7, max_net=0.5, max_concurrent=2)
    cfg = CFG.replace(SIZE_STEP=0.1)
    _, tr, _ = simulate_portfolio(bars, sigs, cfg, prof)
    for c in range(50, 380, 7):
        bars2 = {s: random_walk_after(b, c, seed=c + i) for i, (s, b) in enumerate(bars.items())}
        _, tr2, _ = simulate_portfolio(bars2, sigs, cfg, prof)
        a = tr.loc[tr["entry_b"] <= c + 1, ["sym", "entry_b", "frac"]].reset_index(drop=True)
        b = tr2.loc[tr2["entry_b"] <= c + 1, ["sym", "entry_b", "frac"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_trimmed_trade_return_is_the_whole_position():
    """Review MINOR 3: a trimmed trade's pnl_pct is its cash P&L over its entry notional (trim fills included)."""
    df = intraday([[(100, 100), (100, 130), (131, 131), (131, 132), (132, 132), (132, 132), (132, 132)]])
    cfg = RunConfig.for_timeframe("1Hour", VERTICAL_BARS=4, SLIPPAGE_PCT=1e-4)
    sig = sig_at(df, [0], [1], [0.5], width=0.9)
    prof = RiskProfile("c", max_position=0.5, drift_tol=0.1)
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, prof)
    t = tr.iloc[0]
    assert t["trimmed"] and t["pnl"] == pytest.approx(eq.iloc[-1] - cfg.INIT_CASH, rel=1e-12)
    assert t["pnl_pct"] == pytest.approx(t["pnl"] / (t["qty"] * t["entry_fill"]), rel=1e-12)
    assert t["pnl_pct"] != pytest.approx(t["exit_fill"] / t["entry_fill"] - 1)
