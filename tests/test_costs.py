"""U13 cost model (SPEC §19): auction flags, the quotes-table lookup, per-fill costs and their regression paths."""

import numpy as np
import pandas as pd
import pytest

from risk.costs import auction_flags, fill_costs, half_spread, quotes_half_spread
from risk.portfolio import simulate_portfolio
from risk.profiles import PROFILES
from tests.test_risk import five_min, rand_signals
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from wfo.backtest import equity_curve, run_backtest, simulate_trades

NY = "America/New_York"
SLIP = 1e-4
REGULAR_BINS = [f"{(570 + 15 * k) // 60:02d}:{(570 + 15 * k) % 60:02d}" for k in range(26)]  # 09:30 … 15:45
AUCTION = {"open_auction": 5.0, "close_auction": 7.0, "day": 3.0}


def bin_bp(b: str) -> float:
    """A distinct half-spread (bp) per regular bin: 1.0 at 09:30, +0.1 per bin."""
    return 1.0 + 0.1 * REGULAR_BINS.index(b)


def const_quotes_table(symbols, years=range(1990, 2036), regular=bin_bp, auction=None) -> pd.DataFrame:
    """A quotes table (data.quotes layout) with the same values for every symbol and year."""
    auction = {**AUCTION, **(auction or {})}
    rows = [(s, y, b, regular(b), 10) for s in symbols for y in years for b in REGULAR_BINS]
    rows += [(s, y, b, v, 10) for s in symbols for y in years for b, v in auction.items()]
    return pd.DataFrame(rows, columns=["symbol", "year", "bin", "half_spread_bp", "n"])


def session_5min(day: str, close: str = "16:00", price=100.0) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", f"{day} {close}", freq="5min", inclusive="left", tz=NY)
    p = np.full(len(idx), price) if np.isscalar(price) else np.asarray(price, float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)


def signals_at(df: pd.DataFrame, positions, sides, width=0.5) -> pd.DataFrame:
    idx = df.index[list(positions)]
    s = np.asarray(sides)
    return pd.DataFrame({"trade_signal": s, "signed_dir": s, "bet_size": 1.0, "width": width}, index=idx)


# ── Auction flags ────────────────────────────────────────────────────────────


def test_auction_flags_on_full_and_early_close_sessions():
    full, early = session_5min("2024-07-02"), session_5min("2024-07-03", close="13:00")
    idx = full.index.append(early.index)
    is_open, is_close = auction_flags(idx, 5)
    assert list(idx[is_open].strftime("%m-%d %H:%M")) == ["07-02 09:30", "07-03 09:30"]
    assert list(idx[is_close].strftime("%m-%d %H:%M")) == ["07-02 15:55", "07-03 12:55"]  # the 07-02 12:55 bar is not
    hourly = pd.DatetimeIndex([f"2024-07-02 {h}:30" for h in range(9, 16)] + [f"2024-07-03 {h}:30" for h in range(9, 13)],
                              tz=NY)  # fmt: skip
    _, close_h = auction_flags(hourly, 60)
    assert list(hourly[close_h].strftime("%m-%d %H:%M")) == ["07-02 15:30", "07-03 12:30"]  # stub bars
    daily = pd.DatetimeIndex(["2024-07-02", "2024-07-03"], tz=NY)
    assert all(a.all() for a in auction_flags(daily, None))


# ── Quotes table lookup ──────────────────────────────────────────────────────


def test_quotes_lookup_uses_the_nearest_earlier_year_floors_and_refuses_earlier_years():
    t = pd.DataFrame({"year": [2017, 2017, 2019], "bin": ["10:00", "day", "10:00"], "half_spread_bp": [2.0, 0.1, 4.0]})
    got = quotes_half_spread(np.array([2017, 2018, 2019, 2024]), np.array(["10:00"] * 4, dtype=object), t)
    np.testing.assert_allclose(got, [2e-4, 2e-4, 4e-4, 4e-4])
    assert quotes_half_spread(np.array([2019]), np.array(["day"], dtype=object), t)[0] == pytest.approx(0.25e-4)
    with pytest.raises(ValueError, match="no year <= 2016"):
        quotes_half_spread(np.array([2016]), np.array(["10:00"], dtype=object), t)
    with pytest.raises(ValueError, match="no '11:00' bin"):
        quotes_half_spread(np.array([2019]), np.array(["11:00"], dtype=object), t)


# ── Fills (PLAN2 U13 done-criterion: cost = half-spread + SLIPPAGE_PCT, by hand) ─────────────────────────────────


def test_quotes_fills_intraday_by_hand():
    """Off-auction fills pay SLIPPAGE_PCT + the bin's half-spread; a close at the session's last bar pays the closing
    auction proxy only."""
    df = pd.concat([session_5min("2024-07-02"), session_5min("2024-07-05")])  # bets need data after their session
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_PCT=SLIP, VERTICAL_BARS=12)
    # event at 10:00 → entry at open[10:05] (bin 10:00), vertical exit at the close of the 12th bar (11:00 → bin 11:00)
    # event at 15:20 → entry at open[15:25] (bin 15:15), vertical barrier truncated at the session close: 15:55 bar
    sig = signals_at(df, [6, 70], [1, -1])
    tr = run_backtest(df, sig, cfg, cost_data=const_quotes_table(["X"]))["Meta-filtered"][1]
    a, b = tr.iloc[0], tr.iloc[1]
    assert df.index[a["entry_b"]].strftime("%H:%M") == "10:05" and df.index[a["exit_b"]].strftime("%H:%M") == "11:00"
    assert a["entry_fill"] == pytest.approx(100 * (1 + SLIP + bin_bp("10:00") * 1e-4), rel=1e-14)
    end = df.index[a["exit_b"]] + pd.Timedelta(minutes=5)  # a close fill's bin is its bar's end: 11:05 → 11:00
    assert a["exit_fill"] == pytest.approx(100 * (1 - SLIP - bin_bp(end.strftime("%H:00")) * 1e-4), rel=1e-14)
    assert df.index[b["exit_b"]].strftime("%H:%M") == "15:55" and b["side"] == -1
    assert b["entry_fill"] == pytest.approx(100 * (1 - SLIP - bin_bp("15:15") * 1e-4), rel=1e-14)
    assert b["exit_fill"] == pytest.approx(100 * (1 + AUCTION["close_auction"] * 1e-4), rel=1e-14)  # no slippage
    assert b["cost_bp"] == pytest.approx(SLIP * 1e4 + bin_bp("15:15") + AUCTION["close_auction"])


def test_quotes_fills_on_daily_bars_by_hand():
    """Daily bars: entries at the opening auction, vertical exits at the closing auction (proxies only), intrabar
    barrier exits SLIPPAGE_PCT + the `day` half-spread."""
    idx = pd.bdate_range("2024-03-04", periods=30, tz=NY)
    px = np.full(30, 100.0)
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)
    df.iloc[15, df.columns.get_loc("high")] = 120.0  # the upper barrier of a long entered at 13 is touched intrabar
    cfg = RunConfig.for_timeframe("1Day", COST_MODEL="quotes", SLIPPAGE_PCT=SLIP, VERTICAL_BARS=5)
    sig = signals_at(df, [2, 12], [1, 1], width=0.1)
    tr = run_backtest(df, sig, cfg, cost_data=const_quotes_table(["X"]))["Meta-filtered"][1]
    v, u = tr.iloc[0], tr.iloc[1]
    assert v["exit_reason"] == "vertical" and u["exit_reason"] == "upper"
    for t in (v, u):
        assert t["entry_fill"] == pytest.approx(100 * (1 + AUCTION["open_auction"] * 1e-4), rel=1e-14)
    assert v["exit_fill"] == pytest.approx(100 * (1 - AUCTION["close_auction"] * 1e-4), rel=1e-14)
    assert u["exit_fill"] == pytest.approx(u["exit_px"] * (1 - SLIP - AUCTION["day"] * 1e-4), rel=1e-14)


def test_quotes_floor_and_missing_rows():
    df = session_5min("2024-07-02")
    cfg = RunConfig.for_timeframe("5Min", SLIPPAGE_PCT=SLIP)
    c = fill_costs(df.index, cfg, const_quotes_table(["X"], regular=lambda b: 0.01))
    assert c.loc[df.index[10], "open"] == pytest.approx(SLIP + 0.25e-4)  # floored at 0.25 bp
    with pytest.raises(ValueError, match="cost data"):
        fill_costs(df.index, cfg, None)
    with pytest.raises(ValueError, match="no year <= 2024"):
        fill_costs(df.index, cfg, const_quotes_table(["X"], years=[2025]))


# ── Regression paths ─────────────────────────────────────────────────────────


def test_slippage_cost_frames_reproduce_the_slippage_backtest_exactly():
    """The portfolio simulator with a constant SLIPPAGE_PCT cost frame (profile none) equals simulate_trades — so the
    quotes path differs from the pre-U13 backtest only through the cost numbers."""
    df = synthetic_daily(400)
    sig = rand_signals(df, 80, np.random.default_rng(3))
    # U22: the auction prints and the cash yield live in the portfolio simulator, so the legacy path needs the old
    # values of those switches (scripts/u12_parity.py OLD_SWITCHES)
    cfg = RunConfig.for_timeframe("1Day", COST_MODEL="slippage", SIZE_STEP=0.1, FILL_AUCTION="last_bar",
                                  CASH_YIELD="none")  # fmt: skip
    costs = {"X": pd.DataFrame(cfg.SLIPPAGE_PCT, index=df.index, columns=["open", "intra", "close"])}
    eq, _, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, PROFILES["none"], size_col="bet_size", costs=costs)
    ref_tr = simulate_trades(df, sig, cfg, size_col="bet_size")
    ref = equity_curve(df, ref_tr, cfg.INIT_CASH, cfg.SIZE)
    np.testing.assert_array_equal(eq.to_numpy(), ref.to_numpy())
    res = run_backtest(df, sig, cfg)["Meta-filtered"]
    np.testing.assert_array_equal(res[0].to_numpy(), ref.to_numpy())
    assert "frac" not in res[1]  # COST_MODEL="slippage" + profile none: the pre-U13 single-position backtest


def test_cs_model_is_the_u8_spread_charge_on_every_fill():
    df = synthetic_daily(300).iloc[100:]
    m5 = five_min(300, 3, start="2012-01-03")
    cfg = RunConfig.for_timeframe("1Day", COST_MODEL="cs", RISK_PROFILE="standard")
    c = fill_costs(df.index, cfg, m5)
    expect = cfg.SLIPPAGE_PCT + half_spread(df.index, m5, 21, 0.5e-4).to_numpy()
    for k in ("open", "intra", "close"):
        np.testing.assert_array_equal(c[k].to_numpy(), expect)


def test_legacy_config_keeps_slippage_costs():
    assert RunConfig.legacy_5min().COST_MODEL == "slippage" and RunConfig().COST_MODEL == "quotes"


def test_cost_frames_with_no_bets_give_a_flat_curve():
    """Review regression: a window without a single bet (an IS slice of the PWFO) must not trip the cost checks."""
    df = synthetic_daily(60)
    sig = signals_at(df, [5], [0])  # no side: no bet
    costs = {"X": pd.DataFrame(1e-4, index=df.index, columns=["open", "intra", "close"])}
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, RunConfig.for_timeframe("1Day"), PROFILES["none"],
                                   costs=costs)  # fmt: skip
    assert tr.empty and (eq == eq.iloc[0]).all()
