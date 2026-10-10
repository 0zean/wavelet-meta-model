"""
U22 (PLAN3 §5; SPEC §20–§22): auction-print fills, the per-kind slippage / as-of cost table / stress multiplier and
the cost curve, the account profile, protocol v3 (overlay alpha, marginal, excess returns, power, looks), continuous
targets with the `next_event` exit, the engine-audit items (generic fill timing with the planted mutants, the
session clock, slip-through barriers, the cash yield, the fetch default and the forward truncation).
"""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import features.exits as EX
from data.bars import HOLDOUT_START, log_forward_access, truncate_forward
from data.store import save_bars
from families import test as T
from families.power import mde_alpha, simulate_power
from families.spec import parse
from families.stats import mean_test, sharpe_test
from features.events import sample_events, schedule_events, session_clock
from features.exits import exit_frame, exit_params
from features.triple_barrier_labels import barrier_exits
from primaries.mechanism import MechanismPrimary, primary_config
from risk.account import AccountProfile, check_account, positions
from risk.costs import FILL_CLASSES, asof_half_spread, fill_costs
from risk.portfolio import simulate_portfolio
from risk.profiles import get_profile
from tests.test_costs import AUCTION, SLIP, bin_bp, const_quotes_table, session_5min, signals_at
from tests.test_risk import const_costs
from tests.test_u14 import intraday
from tests.test_u16 import moc, sched
from tests.test_u17 import SPEC
from utils.config import RunConfig
from wfo.rule_pass import rule_signals

NY = "America/New_York"
ts = lambda s: pd.Timestamp(s, tz=NY)


def _prints(df: pd.DataFrame, d_open: float = 0.0, d_close: float = 0.0) -> pd.DataFrame:
    """Auction prints per session of `df`: the bar open / close shifted by d_open / d_close."""
    day = df.index.normalize()
    return pd.DataFrame({"open": df["open"].groupby(day).first() + d_open,
                         "close": df["close"].groupby(day).last() + d_close, "volume": 1.0})  # fmt: skip


# ── Auction fills (SPEC §20) ─────────────────────────────────────────────────


def test_moc_and_moo_fills_use_the_prints_under_print_and_the_bars_under_last_bar():
    df = pd.concat([session_5min("2024-07-01"), session_5min("2024-07-02")])
    df.loc[ts("2024-07-01 15:55"), "close"] = 101.0
    df.loc[ts("2024-07-02 09:30"), "open"] = 103.0
    pr = _prints(df, d_open=0.05, d_close=-0.04)  # print close 100.96, next print open 103.05
    sig = signals_at(df, [df.index.get_loc(ts("2024-07-01 15:50"))], [1])
    costs = {"X": const_costs(df, 0.0)}
    out = {}
    for mode in ("print", "last_bar"):
        cfg = moc(COST_MODEL="quotes", FILL_AUCTION=mode)
        eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), costs=costs, prints={"X": pr})
        out[mode] = (tr.iloc[0]["entry_px"], tr.iloc[0]["exit_px"], eq.attrs["auction_fallbacks"], eq.iloc[-1])
    assert out["print"][:2] == (pytest.approx(100.96), pytest.approx(103.05))
    assert out["last_bar"][:2] == (101.0, 103.0)
    assert out["print"][2] == 0 and out["last_bar"][2] == 0
    q = 10_000 / 100.0  # sized from the 15:50 close (U22)
    assert out["print"][3] == pytest.approx(10_000 + q * (103.05 - 100.96), rel=1e-12)
    # a session without a print falls back to the bar price and is counted
    cfg = moc(COST_MODEL="quotes", FILL_AUCTION="print")
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), costs=costs,
                                   prints={"X": pr.iloc[:1]})  # fmt: skip
    assert tr.iloc[0]["entry_px"] == pytest.approx(100.96) and tr.iloc[0]["exit_px"] == 103.0
    assert eq.attrs["auction_fallbacks"] == 1
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), costs=costs, prints=None)
    assert eq.attrs["auction_fallbacks"] == 2 and (tr.iloc[0]["entry_px"], tr.iloc[0]["exit_px"]) == (101.0, 103.0)


def test_intrabar_and_gap_fills_never_read_a_print():
    """A 10:00 → vertical-at-11:00 trade (no auction bar) and a gap exit at a 09:30 open keep the bar prices."""
    df = pd.concat([session_5min("2024-07-02"), session_5min("2024-07-03")])
    pr = _prints(df, d_open=1.0, d_close=1.0)
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", VERTICAL_BARS=12, FILL_AUCTION="print")
    sig = signals_at(df, [6], [1])
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), costs={"X": const_costs(df, 0.0)},
                                   prints={"X": pr})  # fmt: skip
    t = tr.iloc[0]
    assert t["exit_reason"] == "vertical" and df.index[t["exit_b"]] == ts("2024-07-02 11:00")
    assert t["entry_px"] == 100.0 and t["exit_px"] == 100.0 and eq.attrs["auction_fallbacks"] == 0


class _NoNetwork:
    def __getattr__(self, name):
        raise AssertionError("the test must run on cached bars only")


@pytest.fixture(scope="module")
def spy_cached():
    from data.bars import DEFAULT_CACHE_DIR, load_bars, load_prints

    base = Path(DEFAULT_CACHE_DIR) / "sip" / "all"
    if not (base / "5Min" / "SPY.npz").exists() or not (base / "1DayPrint" / "SPY.npz").exists():
        pytest.skip("no cached SPY bars / prints")
    return (
        load_bars("SPY", "5Min", "2024-02-26", "2024-03-09", source=_NoNetwork()),
        load_prints("SPY", "2024-02-26", "2024-03-09", source=_NoNetwork()),
    )


def test_real_spy_2024_03_04_moc_fill_is_the_official_close_under_print(spy_cached):
    """PLAN3 U22 done-when (the audit's five-session check): the 15:55-decided MOC fill on SPY 2024-03-04 is priced
    at the 1Day close under `print` (raw 512.30; adjusted 496.04 today) and at the 15:55 bar's close under
    `last_bar` (raw 512.25; adjusted 496.00), and the next session's MOO exit at that session's opening print."""
    bars, prints = spy_cached
    bar_close = bars.loc[ts("2024-03-04 15:55"), "close"]
    print_close = prints.loc[ts("2024-03-04 00:00"), "close"]
    print_open = prints.loc[ts("2024-03-05 00:00"), "open"]
    assert print_close != bar_close and 0.3 < (print_close / bar_close - 1) * 1e4 < 1.5
    cfg = primary_config("overnight", "5Min", META_MODEL="none", COST_MODEL="quotes", INIT_CASH=30_000)
    sig = rule_signals(bars, cfg, symbol="SPY")
    sig = sig.loc[[ts("2024-03-04 15:50")]]
    costs = {"SPY": fill_costs(bars.index, cfg, const_quotes_table(["SPY"]))}
    for mode, want_in, want_out in (("print", print_close, print_open),
                                    ("last_bar", bar_close, bars.loc[ts("2024-03-05 09:30"), "open"])):  # fmt: skip
        _, tr, _ = simulate_portfolio({"SPY": bars}, {"SPY": sig}, cfg.replace(FILL_AUCTION=mode), get_profile("none"),
                                      side_col="trade_signal", size_col="bet_size", costs=costs,
                                      prints={"SPY": prints})  # fmt: skip
        assert tr.iloc[0]["entry_px"] == want_in and tr.iloc[0]["exit_px"] == want_out, mode
    for d in ("2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"):  # the audit's five sessions
        gap = abs(prints.loc[ts(d), "close"] / bars.loc[ts(f"{d} 15:55"), "close"] - 1) * 1e4
        assert gap < 3.0, (d, gap)


# ── Costs: per-kind slippage, the as-of table, stress sessions, the cost curve ─


def test_per_kind_slippage_and_the_default_reproduces_slippage_pct():
    df = session_5min("2024-07-02")
    base = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_PCT=SLIP)
    c0 = fill_costs(df.index, base, const_quotes_table(["X"]))
    assert c0.loc[df.index[10], "open"] == pytest.approx(SLIP + bin_bp("10:15") * 1e-4)
    c1 = fill_costs(
        df.index, base.replace(SLIPPAGE_BP={"open": 0.0, "intra": 2.0, "close": 0.5}), const_quotes_table(["X"])
    )
    assert c1.loc[df.index[10], "open"] == pytest.approx(bin_bp("10:15") * 1e-4)
    assert c1.loc[df.index[10], "intra"] == pytest.approx(2e-4 + bin_bp("10:15") * 1e-4)
    assert c1.loc[df.index[10], "close"] == pytest.approx(0.5e-4 + bin_bp("10:15") * 1e-4)  # 10:20 bar ends 10:25
    assert c1.loc[df.index[0], "open"] == pytest.approx(AUCTION["open_auction"] * 1e-4)  # auctions: no slippage
    with pytest.raises(FileNotFoundError, match="measured"):
        fill_costs(df.index, base.replace(SLIPPAGE_BP="measured"), const_quotes_table(["X"]))
    with pytest.raises(ValueError, match="SLIPPAGE_BP"):
        base.replace(SLIPPAGE_BP={"open": 1.0})


def _asof_table(symbols=("X",)) -> pd.DataFrame:
    rows = []
    for s in symbols:
        for i, asof in enumerate(("2016-04-01", "2016-07-01", "2017-01-01")):
            for b, v in (
                ("09:30", 1.0 + i),
                ("open_auction", 3.0 + i),
                ("close_auction", 2.0 + i),
                ("day", 1.5 + i),
                *((f"{(570 + 15 * k) // 60:02d}:{(570 + 15 * k) % 60:02d}", 1.0 + i) for k in range(1, 26)),
            ):
                rows.append((s, asof, b, v, 10))
    return pd.DataFrame(rows, columns=["symbol", "asof", "bin", "half_spread_bp", "n"])


def test_asof_lookup_uses_the_latest_table_at_or_before_the_fill_and_counts_the_first_quarter():
    days = np.array(["2016-02-10", "2016-04-01", "2016-06-30", "2016-07-01", "2018-05-05"], dtype="M8[D]")
    v, before = asof_half_spread(days, np.array(["09:30"] * 5, dtype=object), _asof_table())
    np.testing.assert_allclose(v * 1e4, [1.0, 1.0, 1.0, 2.0, 3.0])
    assert before == 1  # 2016-02-10 is priced with the first (2016-04-01) table
    df = pd.concat([session_5min("2016-02-10"), session_5min("2018-05-07")])
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_BP={"open": 0, "intra": 0, "close": 0})
    c = fill_costs(df.index, cfg, _asof_table())
    assert c.attrs["asof_before_first"] == 1 and c.loc[df.index[0], "open"] == pytest.approx(3e-4)  # one session
    assert c.loc[df.index[-1], "close"] == pytest.approx(4e-4)  # 2018: the 2017-01-01 table's close auction
    with pytest.raises(ValueError, match="no 'day' bin"):
        asof_half_spread(days[:1], np.array(["day"], dtype=object), _asof_table().query("bin != 'day'"))


def test_build_asof_table_pools_the_weeks_completed_before_each_quarter():
    from data.quotes import build_asof_table

    rows = []
    for d, hs in (
        ("2016-02-08", 1.0),
        ("2016-05-09", 3.0),
        ("2016-08-08", 5.0),
        ("2016-11-07", 7.0),
        ("2017-02-06", 9.0),
    ):
        m = pd.Timestamp(f"{d} 10:00", tz=NY)
        rows.append({"symbol": "X", "label": "10:00", "mark": m.tz_convert("UTC"), "bid": 100.0,
                     "ask": 100.0 * (1 + 2 * hs * 1e-4), "bid_size": 1, "ask_size": 1, "quote_ts": m.tz_convert("UTC"),
                     "window_s": 5.0})  # fmt: skip
    t = build_asof_table(pd.DataFrame(rows), weeks=4)
    got = t[t["bin"] == "10:00"].set_index("asof")["half_spread_bp"].round(3).to_dict()
    # as of 2016Q2 only February's week; 2017Q2 pools May 2016 … Feb 2017 (four weeks: 3, 5, 7, 9 → median 6)
    want = {"2016-04-01": 1.0, "2016-07-01": 2.0, "2017-01-01": 4.0, "2017-04-01": 6.0}
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=0.01), k  # (ask − bid) / (ask + bid) is hs to first order
    assert list(t["asof"].unique()) == sorted(t["asof"].unique())
    assert set(t.columns) == {"symbol", "asof", "bin", "half_spread_bp", "n"}


def test_stress_multiplier_scales_the_half_spread_on_flagged_sessions_only():
    df = pd.concat([session_5min("2024-07-02"), session_5min("2024-07-03", close="13:00")])
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_PCT=SLIP, STRESS_MULT=2.1)
    with pytest.raises(ValueError, match="stress-session flags"):
        fill_costs(df.index, cfg, const_quotes_table(["X"]))
    stress = pd.Series({pd.Timestamp("2024-07-03"): True})
    c = fill_costs(df.index, cfg, const_quotes_table(["X"]), stress=stress)
    assert c.loc[df.index[10], "open"] == pytest.approx(SLIP + bin_bp("10:15") * 1e-4)  # 07-02: calm
    i = df.index.get_loc(ts("2024-07-03 10:20"))
    assert c.loc[df.index[i], "open"] == pytest.approx(SLIP + 2.1 * bin_bp("10:15") * 1e-4)  # the spread only
    assert c.loc[df.index[0], "open"] == pytest.approx(AUCTION["open_auction"] * 1e-4)
    assert c.attrs["stress_fills"] == int((df.index.normalize() == ts("2024-07-03")).sum())


def _two_day_round_trips(seed: int = 0):
    df = intraday(6, seed=seed)
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", VERTICAL_BARS=12, SLIPPAGE_PCT=SLIP)
    rng = np.random.default_rng(seed)
    pos = sorted(rng.choice(np.arange(5, 60), 8, replace=False))
    sig = signals_at(df, pos, rng.choice([-1, 1], 8), width=0.002)  # some touches intrabar, some vertical exits
    costs = {"X": fill_costs(df.index, cfg, const_quotes_table(["X"]))}
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), costs=costs)
    return df, cfg, eq, tr


def test_the_cost_ledger_books_every_fill_by_class_and_reprice_reproduces_the_stream():
    from wfo.pwfo import daily_returns

    _df, cfg, eq, _tr = _two_day_round_trips()
    dc = eq.attrs["daily_costs"]
    assert list(dc.columns) == ["equity_start", *[f"notional_{k}" for k in FILL_CLASSES],
                                *[f"cost_{k}" for k in FILL_CLASSES]]  # fmt: skip
    assert sum(dc[f"notional_{k}"].sum() for k in FILL_CLASSES) == pytest.approx(eq.attrs["traded_notional"])
    assert sum(dc[f"cost_{k}"].sum() for k in FILL_CLASSES) == pytest.approx(eq.attrs["cost_paid"])
    assert (
        dc["equity_start"].iloc[0] == cfg.INIT_CASH
        and dc["equity_start"].iloc[1] == eq.groupby(eq.index.normalize()).last().iloc[0]
    )
    assert dc["notional_intra"].sum() > 0 and dc["notional_close_auction"].sum() >= 0
    r = daily_returns(eq)
    # re-pricing at the booked costs per class changes nothing; at zero it adds the booked costs back (gross)
    same, tot = T.reprice(r, dc, {k: 0.0 for k in FILL_CLASSES})
    assert tot["cost_paid"] == 0.0 and tot["pnl_delta"] == pytest.approx(eq.attrs["cost_paid"])
    gross_adj = (sum(dc[f"cost_{k}"] for k in FILL_CLASSES) / dc["equity_start"]).to_numpy()
    np.testing.assert_allclose(same.to_numpy() - T.day_index(r).to_numpy(), gross_adj, atol=1e-15)
    _at1, tot1 = T.reprice(r, dc, 1.0)  # 1 bp round trip = 0.5 bp per side on every fill
    assert tot1["cost_paid"] == pytest.approx(0.5e-4 * eq.attrs["traded_notional"])


# ── Cash yield ───────────────────────────────────────────────────────────────


def test_cash_yield_credits_free_cash_over_the_gap_act_360():
    df = pd.concat([session_5min("2024-07-01"), session_5min("2024-07-03")])  # Monday, Wednesday: two days
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="slippage", CASH_YIELD="tbill")
    sig = signals_at(df, [], [])
    rate = pd.Series({pd.Timestamp("2024-07-01"): 0.054, pd.Timestamp("2024-07-03"): 0.05})
    eq, _, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), cash_yield=rate)
    want = 10_000 * 0.054 * 2 / 360
    assert eq.attrs["cash_interest"] == pytest.approx(want) and eq.iloc[-1] == pytest.approx(10_000 + want)
    assert eq.iloc[0] == 10_000  # credited at the second session's first bar
    eq0, _, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), cash_yield=None)
    assert eq0.iloc[-1] == 10_000
    # a short's proceeds beyond the equity earn nothing: free cash = min(cash, equity)
    short = signals_at(df, [5], [-1], width=10.0)
    cfg2 = RunConfig.for_timeframe("5Min", COST_MODEL="slippage", SLIPPAGE_PCT=0.0, VERTICAL_BARS=100,
                                   HOLD_OVERNIGHT=True)  # fmt: skip
    eq1, tr, _ = simulate_portfolio({"X": df}, {"X": short}, cfg2, get_profile("none"), cash_yield=rate)
    assert len(tr) == 1 and eq1.attrs["cash_interest"] == pytest.approx(10_000 * 0.054 * 2 / 360, rel=1e-9)


# ── Account profile ──────────────────────────────────────────────────────────


def _trades(rows) -> pd.DataFrame:
    """rows = (side, frac, entry 'YYYY-MM-DD HH:MM', exit 'YYYY-MM-DD HH:MM', rolled)."""
    return pd.DataFrame([{"side": s, "frac": f, "entry_time": ts(e), "exit_time": ts(x), "rolled": r,
                          "entry_b": i, "exit_b": i + 1} for i, (s, f, e, x, r) in enumerate(rows)])  # fmt: skip


def test_pdt_rule_flags_the_fourth_day_trade_under_25k_and_not_above():
    rows = [(1, 0.25, f"2024-07-0{d} 10:00", f"2024-07-0{d} 15:55", False) for d in (1, 1, 1, 1, 2)]
    tr = _trades(rows)
    pos = positions(tr)
    assert pos["day_trade"].all() and len(pos) == 5
    small = check_account([("X", 1.0, tr, 10_000.0)], None, AccountProfile("m10k", equity=10_000))
    assert not small["tradable"] and small["day_trades"]["first_pdt_flag"] == "2024-07-01"
    assert small["day_trades"]["max_in_window"] == 5 and "pattern-day-trader" in small["reasons"][0]
    assert small["cost_otherwise"]["blocked_day_trades"] >= 1
    big = check_account([("X", 1.0, tr, 10_000.0)], None, AccountProfile("m30k", equity=30_000))
    assert big["tradable"] and big["day_trades"]["first_pdt_flag"] is None and not big["day_trades"]["pdt_applies"]
    # a chain of rolled trades (the flag on the trade whose exit rolled) is one position; held overnight it is no
    # day trade; a side flip inside a session is two positions, both day trades (U22 review)
    chain = _trades(
        [
            (1, 1.0, "2024-07-01 15:55", "2024-07-02 15:55", True),
            (1, 1.0, "2024-07-02 15:55", "2024-07-03 15:55", False),
        ]
    )
    assert len(positions(chain)) == 1 and not positions(chain)["day_trade"].iloc[0]
    flip = _trades(
        [
            (1, 1.0, "2024-07-01 10:00", "2024-07-01 11:00", True),
            (-1, 1.0, "2024-07-01 11:00", "2024-07-01 15:55", False),
        ]
    )
    fp = positions(flip)
    assert len(fp) == 2 and fp["day_trade"].all() and list(fp["side"]) == [1, -1]
    # exits at an instant precede entries: a roll from A to B at one 09:30 open is 0.5× gross, not 1×
    members = [("A", 0.5, _trades([(1, 1.0, "2024-07-01 09:30", "2024-07-02 09:30", False)]), 1e4),
               ("B", 0.5, _trades([(1, 1.0, "2024-07-02 09:30", "2024-07-03 09:30", False)]), 1e4)]  # fmt: skip
    assert check_account(members, None, AccountProfile("m", equity=30_000))["buying_power"][
        "max_gross_intraday"
    ] == pytest.approx(0.5)


def test_cash_account_refuses_an_unsettled_re_entry_and_shorts_and_buying_power_is_checked():
    tr = _trades(
        [
            (1, 1.0, "2024-07-01 10:00", "2024-07-01 11:00", False),
            (1, 1.0, "2024-07-01 12:00", "2024-07-01 15:55", False),
        ]
    )
    cash = check_account([("X", 1.0, tr, 30_000.0)], None, AccountProfile("c", kind="cash"))
    assert not cash["tradable"] and cash["settlement"]["violation_at"] is not None
    assert "good-faith" in " ".join(cash["reasons"])
    # the same with T+1 settled: a re-entry the next session is fine
    tr2 = _trades(
        [
            (1, 1.0, "2024-07-01 10:00", "2024-07-01 11:00", False),
            (1, 1.0, "2024-07-02 12:00", "2024-07-02 15:55", False),
        ]
    )
    assert (
        check_account([("X", 1.0, tr2, 30_000.0)], None, AccountProfile("c", kind="cash"))["settlement"]["violation_at"]
        is None
    )
    sh = check_account([("X", 1.0, _trades([(-1, 0.5, "2024-07-01 10:00", "2024-07-01 11:00", False)]), 30_000.0)], None,
                       AccountProfile("c", kind="cash"))  # fmt: skip
    assert sh["settlement"]["short_in_cash"] and not sh["tradable"]
    # buying power: two members at 1× each on half the capital = 1× gross; a 3× overnight position breaks Reg-T
    members = [("A", 0.5, _trades([(1, 1.0, "2024-07-01 10:00", "2024-07-02 10:00", False)]), 1e4),
               ("B", 0.5, _trades([(-1, 1.0, "2024-07-01 10:00", "2024-07-02 10:00", False)]), 1e4)]  # fmt: skip
    ok = check_account(members, None, AccountProfile("m", equity=30_000))
    assert ok["tradable"] and ok["buying_power"]["max_gross_overnight"] == pytest.approx(1.0)
    big = check_account([("A", 1.0, _trades([(1, 3.0, "2024-07-01 10:00", "2024-07-02 10:00", False)]), 1e4)], None,
                        AccountProfile("m", equity=30_000))  # fmt: skip
    assert not big["tradable"] and big["buying_power"]["first_breach"] == "2024-07-01"
    # locate fees are an estimate, never deducted
    lo = check_account([("A", 1.0, _trades([(-1, 1.0, "2024-07-01 10:00", "2024-07-02 10:00", False)]), 1e4)], None,
                       AccountProfile("m", equity=30_000, locate_bps=50.0))  # fmt: skip
    assert lo["locate"]["cost_frac_per_year"] > 0 and lo["tradable"]


# ── Protocol v3 statistics ──────────────────────────────────────────────────


def test_overlay_alpha_test_has_the_right_size_and_power_at_its_mde():
    size = simulate_power(1000, 0.08, 0.0, n_sims=200, n_boot=299, stream="iid", seed=11)["rejection_rate"]
    assert 0.02 <= size <= 0.09
    garch = simulate_power(1000, 0.08, 0.0, n_sims=100, n_boot=299, stream="garch_t", seed=12)["rejection_rate"]
    assert 0.0 <= garch <= 0.12
    mde = mde_alpha(1000, 0.08)["mde_bp_per_day"]
    power = simulate_power(1000, 0.08, mde, n_sims=100, n_boot=299, stream="iid", seed=13)["rejection_rate"]
    assert power >= 0.65
    m = mde_alpha(2400, 0.08, families=4)
    assert 3.0 < m["mde_bp_per_day"] < 3.4 and 0.9 < m["mde_sharpe_ann"] < 1.1  # PLAN3 §4.5: Sharpe ≈ 1 at Holm


def test_one_sided_tests_and_the_mean_test_by_hand():
    rng = np.random.default_rng(3)
    r = rng.normal(2e-4, 0.005, 1200)
    one, two = mean_test(r, n_boot=999, sided="one"), mean_test(r, n_boot=999, sided="two")
    assert one["mean_ann"] == pytest.approx(r.mean() * 252) and one["p"] < two["p"] < 0.1
    assert one["ci_ann"][1] == float("inf") and one["ci_ann"][0] < one["mean_ann"]
    assert abs(one["t"] - one["t_nw"]) < 0.5
    with pytest.raises(ValueError, match="sided"):
        mean_test(r, sided="left")
    a, b = rng.normal(5e-4, 0.01, 1500), rng.normal(1e-4, 0.01, 1500)
    s1, s2 = sharpe_test(a, b, n_boot=999, sided="one"), sharpe_test(a, b, n_boot=999, sided="two")
    assert s1["delta_ann"] == s2["delta_ann"] and s1["p"] < s2["p"] and s1["ci_ann"][1] == float("inf")


def _days(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_marginal_reproduces_the_paired_sharpe_test_and_excess_returns_tilt_the_sharpe():
    d = _days(1000)
    rng = np.random.default_rng(5)
    core = pd.Series(rng.normal(4e-4, 0.01, 1000), index=d)
    a = pd.Series(core.to_numpy() * 0.5 + rng.normal(3e-4, 0.006, 1000), index=d)
    overlay = a - core  # core + 1 × overlay = a
    test = {"kind": "marginal", "sided": "one", "block_days": 21, "n_boot": 499, "alpha": 0.05, "seed": 0}
    c = T.compare(overlay, core, "cash", test, rf=None, core=core, k=1.0)
    ref = sharpe_test(a.to_numpy(), core.to_numpy(), block=21, n_boot=499, seed=0, sided="one")
    assert c["delta_ann"] == pytest.approx(ref["delta_ann"]) and c["p"] == ref["p"]
    assert c["portfolio_sharpe"] == pytest.approx(T.sharpe_ann(a)) and "dd_difference" in c
    # excess returns: a constant rf lowers every Sharpe by rf / σ (annualized)
    rf = pd.Series(0.03 / 252, index=d)
    bench = pd.Series(rng.normal(5e-4, 0.012, 1000), index=d)
    raw = T.compare(a, bench, "constant_mix_er", {**test, "kind": "sharpe_vs_benchmark", "sided": "two"})
    ex = T.compare(a, bench, "constant_mix_er", {**test, "kind": "sharpe_vs_benchmark", "sided": "two"}, rf=rf)
    assert raw["bench_sharpe"] - ex["bench_sharpe"] == pytest.approx(
        0.03 / (bench.std(ddof=1) * np.sqrt(252)), rel=1e-6
    )
    oa = T.compare(a, bench, "constant_mix_er", {**test, "kind": "overlay_alpha"}, rf=rf)
    assert oa["statistic"] == "alpha_ann" and oa["delta_ann"] == pytest.approx((a - rf).mean() * 252)
    assert oa["alpha_bp_per_day"] == pytest.approx((a - rf).mean() * 1e4) and oa["p"] < 0.05

    sess = pd.DatetimeIndex(["2024-07-01", "2024-07-03", "2024-07-08"])
    rate = pd.Series([0.036, 0.036, 0.04], index=sess)
    acc = T.rf_accrual(rate, sess)
    np.testing.assert_allclose(acc.to_numpy(), [0.0, 0.036 * 2 / 360, 0.036 * 5 / 360])
    d = _days(4)
    bh = {"A": pd.Series([0.1, 0.0, 0.0, 0.0], index=d), "B": pd.Series([0.0, 0.0, 0.0, 0.0], index=d)}
    w = pd.Series({"A": 0.5, "B": 0.5})
    drift = T.benchmark("buy_and_hold", bh, w)
    mix = T.benchmark("constant_mix_er", bh, w)
    assert drift.iloc[0] == pytest.approx(0.05) == mix.iloc[0]
    bh2 = {"A": pd.Series([0.1, 0.1, 0.0, 0.0], index=d), "B": pd.Series([0.0, 0.0, 0.0, 0.0], index=d)}
    assert T.benchmark("buy_and_hold", bh2, w).iloc[1] == pytest.approx(0.055 / 1.05)  # A's weight drifted to 0.55/1.05
    assert T.benchmark("constant_mix_er", bh2, w).iloc[1] == pytest.approx(0.05)
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 0.01, 500), index=_days(500))
    same = {"A": x, "B": x}
    ind = {"A": x, "B": pd.Series(rng.normal(0, 0.01, 500), index=_days(500))}
    assert T.n_eff(same, pd.Series({"A": 0.5, "B": 0.5}))["n_eff"] == pytest.approx(1.0)
    assert 1.7 < T.n_eff(ind, pd.Series({"A": 0.5, "B": 0.5}))["n_eff"] < 2.3
    coh = T.cells_coherence({"c1": 1.0, "c2": 0.5, "c3": -0.2}, 0.7, 0.667)
    assert coh["share"] == pytest.approx(2 / 3) and coh["coherent"] and coh["median_alpha"] == 0.5


# ── Spec v3 ──────────────────────────────────────────────────────────────────


def _doc(**over):
    import copy

    doc = copy.deepcopy(SPEC)
    doc.update(over)
    return doc


def test_spec_v3_keys_power_core_account_and_aliases():
    fam = parse(_doc())
    assert fam.benchmark == "constant_mix_ew" and fam.test["kind"] == "sharpe_vs_benchmark"
    assert parse(_doc(benchmark="buy_and_hold_er")).benchmark == "constant_mix_er"
    assert fam.account == {"kind": "margin", "equity": 10_000.0, "locate_bps": 0.0, "name": "margin_10k"}
    m = mde_alpha(2400, 0.08, families=4)["mde_bp_per_day"]
    power = {"n_days": 2400, "vol_ann": 0.08, "families": 4, "mde_alpha_bp_per_day": round(m, 2),
             "expected_alpha_bp_per_day": 3.5, "diagnostic": False}  # fmt: skip
    v3 = parse(_doc(test={"kind": "overlay_alpha", "sided": "one", "at_cost": 1.0, "n_boot": 500}, power=power,
                    account={"kind": "cash", "equity": 30_000}, benchmark="buy_and_hold"))  # fmt: skip
    assert v3.test["kind"] == "overlay_alpha" and v3.power["mde_alpha_bp_per_day"] == round(m, 2)
    assert v3.account["name"] == "cash_30k" and v3.benchmark == "buy_and_hold"
    with pytest.raises(ValueError, match="needs the `power` block"):
        parse(_doc(test={"kind": "overlay_alpha"}))
    with pytest.raises(ValueError, match="disagrees"):
        parse(_doc(test={"kind": "overlay_alpha"}, power={**power, "mde_alpha_bp_per_day": 1.0}))
    with pytest.raises(ValueError, match="diagnostic"):
        parse(_doc(test={"kind": "overlay_alpha"}, power={**power, "expected_alpha_bp_per_day": 1.0}))
    diag = parse(
        _doc(test={"kind": "overlay_alpha"}, power={**power, "expected_alpha_bp_per_day": 1.0, "diagnostic": True})
    )
    assert diag.power["diagnostic"]
    with pytest.raises(ValueError, match="needs core"):
        parse(_doc(test={"kind": "marginal"}, power=power))
    marg = parse(_doc(test={"kind": "marginal"}, power=power, core={"family": "F1", "k": 1.0},
                      variants=[{"label": "k2", "core.k": 2.0}, {"label": "vix20", "primary.params.vix_max": 20}]))  # fmt: skip
    assert marg.variants[1].core == {"family": "F1", "k": 2.0} and marg.variants[2].core == {"family": "F1", "k": 1.0}
    with pytest.raises(ValueError, match="marginal"):
        parse(_doc(core={"family": "F1"}))
    for bad, match in (({"test": {"kind": "x"}}, "test.kind"), ({"test": {"sided": "left"}}, "test.sided"),
                       ({"test": {"at_cost": -1}}, "at_cost"), ({"account": {"kind": "ira"}}, "account.kind"),
                       ({"state_splits": ["year", "moon"]}, "state split")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            parse(_doc(**bad))


def test_program_caps_scope_to_the_listed_families(tmp_path):
    from experiments.ledger import Ledger
    from families.run import BudgetError, check_budget, program_caps

    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G1, G2]\nmax_families: 2\nmax_trials: 5\n", encoding="utf-8")
    caps = program_caps(prog)
    assert caps["families"] == ["G1", "G2"] and caps["program"] == "P"
    fam = parse(_doc(id="F3_test"))
    with pytest.raises(BudgetError, match="not a family of program"):
        check_budget(fam, [], caps)
    g1 = parse(_doc(id="G1"))
    rows = [
        {
            "stage": "F",
            "status": "ok",
            "kind": "family_variant",
            "cell_hash": "x",
            "base_family": "F5",
            "family": "F5",
            "trial_key": f"F5/{i}",
            "n_trials": 1,
        }
        for i in range(40)
    ]  # PLAN2 rows do not count
    acc = check_budget(g1, rows, caps)
    assert acc["program_trials"] == 3 and acc["program_families"] == 1 and acc["program"] == "P"
    Ledger(tmp_path / "l.jsonl")  # (the ledger type is what run_family passes)


def test_looks_ledger_records_and_bounds(tmp_path):
    from families.looks import bonferroni, looks_total, record_look

    p = tmp_path / "looks.jsonl"
    assert looks_total(p) == 0
    record_look("positions.py", 25, "band and velocity grid", path=p, when="2026-10-10T00:00:00+00:00")
    record_look("velocity.py", 8, path=p)
    assert looks_total(p) == 33 and bonferroni(0.002, 33) == pytest.approx(0.066) and bonferroni(0.5, 33) == 1.0
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["at"] == "2026-10-10T00:00:00+00:00" and rows[1]["n"] == 8
    with pytest.raises(ValueError):
        record_look("x", 0, path=p)


# ── Continuous targets and the next_event exit (SPEC §21) ───────────────────

EVERY_BAR = [f"{(575 + 5 * k) // 60:02d}:{(575 + 5 * k) % 60:02d}" for k in range(77)]  # 09:35 … 15:55


class TargetRule(MechanismPrimary):
    """A continuous target per decision: const (`value`), alternating ±1, or zero (two opposite cells netted).
    Registered as `test_target` by the module fixture below (and removed after, so the registry tests see no test
    primary)."""

    name = "test_target"
    TIMEFRAMES = ("5Min",)
    ALLOW_CONTINUOUS = True
    DEFAULTS: ClassVar[dict] = {"mode": "const", "value": 1.0, "times": "all"}

    def validate(self):
        assert self.params["mode"] in ("const", "alternate", "zero")

    def config_overrides(self, timeframe):
        times = EVERY_BAR if self.params["times"] == "all" else list(self.params["times"])
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": times}, "EXIT_MODEL": "time",
                "EXIT_PARAMS": {"exit_time": "next_event"}, "SIZE_STEP": 0}  # fmt: skip

    def rule(self, df, X, cfg):
        n = len(X)
        if self.params["mode"] == "const":
            t = np.full(n, float(self.params["value"]))
        elif self.params["mode"] == "alternate":
            t = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        else:
            t = (np.ones(n) + -np.ones(n)) / 2  # the mean of two opposite cells
        return np.sign(t), np.abs(t)


@pytest.fixture(autouse=True)
def _register_test_target():
    from primaries.base import REGISTRY

    REGISTRY["test_target"] = TargetRule
    try:
        yield
    finally:
        REGISTRY.pop("test_target", None)


def _continuous(mode, df, c=1e-4, **params):
    cfg = primary_config(
        "test_target", "5Min", {"mode": mode, **params}, META_MODEL="none", COST_MODEL="quotes", VOL_SPAN=10
    )  # fmt: skip (the first 10 bars have no width: the first day's chain starts late)
    sig = rule_signals(df, cfg, symbol="X")
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), side_col="trade_signal",
                                   size_col="bet_size", costs={"X": const_costs(df, c)})  # fmt: skip
    return cfg, sig, eq, tr


def test_a_constant_plus_one_target_is_buy_and_hold_intraday_less_one_entry_and_one_exit_a_day():
    df = intraday(3, seed=7)
    cfg, sig, eq, tr = _continuous("const", df)
    assert exit_params(cfg)["exit_time"] == "next_event" and cfg.SIZE_STEP == 0
    assert len(sig) == 3 * 77 - 10 and (sig["bet_size"] == 1.0).all()
    tr = tr.assign(day=eq.index[tr["entry_b"].to_numpy(int)].normalize()).sort_values("entry_b")
    by_day = tr.groupby("day")
    assert (
        list(by_day.size()) == [67, 77, 77] and tr["rolled"].sum() == 66 + 76 + 76
    )  # one chain a day, rolled untraded
    assert int((tr["entry_cost_bp"] == 0).sum()) == 66 + 76 + 76  # every entry after a rolled exit is free
    assert len(T.position_returns(tr.assign(sym="X"))) == 3  # three positions (U22: chain ids by the rolled flag)
    cash = 10_000.0
    for d, g in df.groupby(df.index.normalize()):
        first = int(by_day.get_group(d)["entry_b"].iloc[0])  # the first decision's entry bar (09:35, or after warm-up)
        o, c = df["open"].iloc[first], g["close"].iloc[-1]  # → the MOC close
        q = cash / (o * (1 + 1e-4))
        cash += q * (c * (1 - 1e-4) - o * (1 + 1e-4))
        assert eq.loc[g.index[-1]] == pytest.approx(cash, rel=1e-12)
    assert eq.attrs["traded_notional"] == pytest.approx(
        sum(
            t
            for t in by_day.apply(
                lambda g: g["qty"].iloc[0] * g["entry_px"].iloc[0] + g["qty"].iloc[-1] * g["exit_px"].iloc[-1]
            )
        )
    )


def test_an_alternating_target_pays_two_costs_per_bar_and_opposite_cells_net_to_no_trade():
    df = intraday(2, seed=8)
    c = 2e-4
    _cfg, _sig, eq, tr = _continuous("alternate", df, c=c)
    # every roll flips the side: the traded shares are the old and the new position, cost = c × (q_old + q_new) × px
    chain = tr.sort_values("entry_b").reset_index(drop=True)
    paid = 0.0
    for i, t in chain.iterrows():
        prev = chain.iloc[i - 1] if i > 0 else None
        if prev is None or not prev["rolled"]:  # a fresh entry pays its own shares
            paid += t["qty"] * t["entry_px"] * c
        else:  # the previous exit rolled into this entry: the flip trades both positions' shares
            assert t["side"] == -prev["side"] and t["entry_px"] == prev["exit_px"]
            paid += (t["qty"] + prev["qty"]) * t["entry_px"] * c
            assert t["entry_cost_bp"] * 1e-4 == pytest.approx(c * (t["qty"] + prev["qty"]) / t["qty"])
        if not t["rolled"]:  # the chain's last exit pays its shares
            paid += t["qty"] * t["exit_px"] * c
    assert eq.attrs["cost_paid"] == pytest.approx(paid, rel=1e-12)
    _, sig0, eq0, tr0 = _continuous("zero", df, c=c)
    assert (sig0["signed_dir"] == 0).all() and len(tr0) == 0 and (eq0 == 10_000).all()
    with pytest.raises(ValueError, match="SIZE_STEP 0"):
        primary_config("test_target", "5Min", {"mode": "const"}, META_MODEL="none", SIZE_STEP=0.1)


def test_next_event_exit_holds_to_the_next_decision_or_the_close():
    df = session_5min("2024-07-02")
    df["open"] = np.arange(len(df)) + 100.0
    df["close"] = df["open"] + 0.5
    cfg = primary_config("test_target", "5Min", {"mode": "const", "times": ["10:00", "11:00"]}, META_MODEL="none")
    ev = sample_events(df, cfg)
    assert [t.strftime("%H:%M") for t in ev.index] == ["09:55", "10:55"]
    lab = exit_frame(df, ev.index, pd.Series(0.01, index=ev.index), cfg)
    assert [df.index[x].strftime("%H:%M") for x in lab["exit_pos"]] == ["11:00", "15:55"]
    assert list(lab["barrier"]) == ["time", "vertical"]
    assert lab["exit_px"].iloc[0] == df.loc[ts("2024-07-02 11:00"), "open"]
    assert lab["exit_px"].iloc[1] == df.loc[ts("2024-07-02 15:55"), "close"]
    for bad in ({"entry_times": ["close"]}, {"entry_times": ["10:00"], "days": "fomc"}):
        with pytest.raises(ValueError, match="next_event"):
            RunConfig.for_timeframe("5Min", EVENT_SAMPLER="schedule", EVENT_PARAMS=bad, EXIT_MODEL="time",
                                    EXIT_PARAMS={"exit_time": "next_event"})  # fmt: skip


# ── Engine-audit items ───────────────────────────────────────────────────────


def _assert_fill_timing(df: pd.DataFrame, cfg: RunConfig, frame: pd.DataFrame, events: pd.DatetimeIndex) -> None:
    """Every entry fills at the bar after its decision (entry_pos == t + 1); every exit fills strictly after the bar
    that decided it: at or after the entry bar, and an open fill of a hysteresis exit at the bar after its trigger."""
    t = df.index.get_indexer(frame.index)
    assert (t >= 0).all() and (frame["entry_pos"].to_numpy() == t + 1).all(), "an entry before its decision bar"
    assert (frame["exit_pos"].to_numpy() >= frame["entry_pos"].to_numpy()).all(), "an exit before its entry"
    open_ = df["open"].to_numpy()
    for b, x, e in zip(frame["barrier"], frame["exit_pos"], frame["entry_pos"]):
        if b == "hysteresis":
            assert x >= e + 1, "a hysteresis exit at its own entry bar"
            sig = EX.exit_signal(df, cfg).to_numpy()
            beta = exit_params(cfg)["beta"]
            side = frame.loc[df.index[x - 1 - (x - 1 - e) - 0 if False else frame.index[0]], "label"] if False else None
            assert side is None
            trig = sig[x - 1]
            assert np.isfinite(trig) and (abs(trig) >= beta), "an open fill not decided at the previous close"
        if b == "time":
            assert frame.loc[frame["exit_pos"] == x, "exit_px"].iloc[0] == open_[x]


@pytest.mark.parametrize(
    "sampler,exit_",
    [
        ("cusum", ("triple_barrier", {})),
        ("dc", ("triple_barrier", {"slip_through": True})),
        ("cusum", ("hysteresis", {"beta": 0.5, "max_bars": 6})),
        ("sched_1000", ("time", {"exit_time": "close"})),
        ("sched_1000", ("time", {"exit_time": "10:30"})),
        ("moc", ("time", {"exit_time": "open"})),
    ],
)
def test_every_sampler_and_exit_model_fills_after_its_decision_bar(sampler, exit_):
    df = intraday(12, seed=21)
    model, params = exit_
    kw = {"EXIT_MODEL": model, "EXIT_PARAMS": params, "VERTICAL_BARS": 8}
    if sampler in ("cusum", "dc"):
        cfg = RunConfig.for_timeframe("5Min", EVENT_SAMPLER=sampler, PRIMARY="bollinger_mr", **kw)
    elif sampler == "moc":
        cfg = moc(exit_params=params, VERTICAL_BARS=8)
    else:
        cfg = sched(["10:00", "15:30"], **kw)
    ev = sample_events(df, cfg).dropna(subset=["width"])
    side = pd.Series(np.where(np.arange(len(ev)) % 2 == 0, 1, -1), index=ev.index)
    frame = exit_frame(df, ev.index, ev["width"], cfg, side=side)
    assert len(frame) >= 8
    _assert_fill_timing(df, cfg, frame, ev.index)


def test_the_fill_timing_test_catches_the_planted_entry_and_hysteresis_mutants(monkeypatch):
    """The engine audit's plant.py ENTRY (enter at the event bar's open) and EXIT_HYST (fill at the trigger bar's
    open) mutants escaped the causality tests; the fill-timing test must fail on both."""
    df = intraday(12, seed=21)
    cfg = RunConfig.for_timeframe("5Min", EVENT_SAMPLER="cusum", PRIMARY="bollinger_mr", VERTICAL_BARS=8)
    ev = sample_events(df, cfg).dropna(subset=["width"])
    side = pd.Series(1, index=ev.index)
    orig_bx = EX.barrier_exits

    def entry_mutant(df_, events, width, side_=None, *, vertical_bars, hold_overnight, slip_through=False):
        ev2 = df_.index[np.maximum(df_.index.get_indexer(events) - 1, 0)]
        w2 = pd.Series(width.reindex(events).to_numpy(), index=ev2)
        s2 = None if side_ is None else pd.Series(side_.reindex(events).to_numpy(), index=ev2)
        out = orig_bx(df_, ev2, w2, s2, vertical_bars=vertical_bars, hold_overnight=hold_overnight)
        out.index = df_.index[df_.index.get_indexer(out.index) + 1]
        return out

    monkeypatch.setattr(EX, "barrier_exits", entry_mutant)
    with pytest.raises(AssertionError, match="before its decision"):
        _assert_fill_timing(df, cfg, exit_frame(df, ev.index, ev["width"], cfg, side=side), ev.index)
    monkeypatch.setattr(EX, "barrier_exits", orig_bx)
    hcfg = RunConfig.for_timeframe("5Min", EVENT_SAMPLER="cusum", PRIMARY="bollinger_mr", EXIT_MODEL="hysteresis",
                                   EXIT_PARAMS={"beta": 0.5, "max_bars": 6})  # fmt: skip
    hev = sample_events(df, hcfg).dropna(subset=["width"])
    hside = pd.Series(np.where(np.arange(len(hev)) % 2 == 0, 1, -1), index=hev.index)
    _assert_fill_timing(df, hcfg, exit_frame(df, hev.index, hev["width"], hcfg, side=hside), hev.index)
    orig_hx = EX.hysteresis_exits

    def hyst_mutant(df_, events, width, signal, side_=None, *, beta, max_bars, hold_overnight):
        out = orig_hx(df_, events, width, signal, side_, beta=beta, max_bars=max_bars, hold_overnight=hold_overnight)
        h = out["barrier"].to_numpy() == "hysteresis"
        x = out["exit_pos"].to_numpy().copy()
        x[h] -= 1  # fill at the OPEN of the trigger bar (decided at its close)
        px = out["exit_px"].to_numpy().copy()
        px[h] = df_["open"].to_numpy()[x[h]]
        out["exit_pos"], out["exit_px"], out["t1"] = x, px, df_.index[x]
        return out

    monkeypatch.setattr(EX, "hysteresis_exits", hyst_mutant)
    with pytest.raises(AssertionError):
        _assert_fill_timing(df, hcfg, exit_frame(df, hev.index, hev["width"], hcfg, side=hside), hev.index)


def test_slip_through_fills_at_the_worse_of_the_barrier_and_the_next_open():
    df = session_5min("2024-07-02")
    df["high"], df["low"] = 100.5, 99.5
    i = 10  # the long entered at open[11] = 100 (event at 10) touches the lower barrier 99 at bar 13
    df.iloc[13, df.columns.get_loc("low")] = 98.5
    df.iloc[14, df.columns.get_loc("open")] = 98.0  # the next open is worse than the barrier
    ev = pd.DatetimeIndex([df.index[i]])
    w = pd.Series(0.01, index=ev)
    side = pd.Series(1, index=ev)
    plain = barrier_exits(df, ev, w, side, vertical_bars=12, hold_overnight=False)
    slip = barrier_exits(df, ev, w, side, vertical_bars=12, hold_overnight=False, slip_through=True)
    assert (plain["exit_pos"].iloc[0], plain["exit_px"].iloc[0]) == (13, pytest.approx(99.0))
    assert (slip["exit_pos"].iloc[0], slip["exit_px"].iloc[0], slip["barrier"].iloc[0]) == (14, 98.0, "lower")
    df.iloc[14, df.columns.get_loc("open")] = 99.4  # a better next open: the barrier price stands
    better = barrier_exits(df, ev, w, side, vertical_bars=12, hold_overnight=False, slip_through=True)
    assert (better["exit_pos"].iloc[0], better["exit_px"].iloc[0]) == (13, pytest.approx(99.0))
    cfg = RunConfig.for_timeframe("5Min", EXIT_PARAMS={"slip_through": True})
    assert exit_params(cfg)["slip_through"] is True
    with pytest.raises(ValueError, match="slip_through"):
        RunConfig.for_timeframe("5Min", EXIT_PARAMS={"slip_through": "yes"})


def test_the_session_clock_counts_a_dropped_session_and_drops_its_fills():
    days = pd.bdate_range("2024-07-01", periods=4)  # Mon … Thu; Tuesday's bars are dropped
    df = pd.concat([session_5min(str(d.date())) for d in days if d.day != 2])
    sess, first, last = session_clock(df.index, days)
    assert (
        list(sess[[0, 78, 156]]) == [0, 2, 3] and list(first) == [0, -1, 78, 156] and list(last) == [77, -1, 155, 233]
    )
    cfg = moc(exit_params={"exit_time": "close", "exit_session": 1})
    ev = schedule_events(df, cfg, sessions=days)
    frame = exit_frame(df, ev, pd.Series(0.01, index=ev), cfg, sessions=days)
    # Monday's MOC entry targets Tuesday's close (no bars: dropped); Wednesday's targets Thursday's (filled)
    assert ts("2024-07-01 15:50") not in frame.index and ts("2024-07-03 15:50") in frame.index
    data_clock = exit_frame(df, ev, pd.Series(0.01, index=ev), cfg)  # the data's sessions: Monday exits Wednesday
    assert ts("2024-07-01 15:50") in data_clock.index
    with pytest.raises(ValueError, match="not exchange sessions"):
        session_clock(df.index, days[1:])


def test_fetch_defaults_to_holdout_start_truncation_logs_and_the_forward_log(tmp_path):
    from data.fetch import parser

    assert parser().get_default("end") == HOLDOUT_START.strftime("%Y-%m-%d")
    assert parser().get_default("timeframes") == ["5Min", "1DayPrint"]
    cache = tmp_path / "cache"
    df = pd.concat([session_5min(d) for d in ("2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02")])
    idx = df.index
    npz, js = cache / "sip" / "all" / "5Min" / "X.npz", cache / "sip" / "all" / "5Min" / "X.json"
    meta = {"symbol": "X", "timeframe": "5Min", "feed": "sip", "adjustment": "all", "tz": NY, "session": "rth",
            "source": "test", "coverage_start": "2026-09-28", "coverage_end": "2026-10-03", "fetched_at": "x"}  # fmt: skip
    save_bars(npz, js, df, meta)
    log = tmp_path / "forward_access.jsonl"
    events = truncate_forward(cache, log)
    assert len(events) == 1 and events[0]["dropped_bars"] == int((idx >= HOLDOUT_START.tz_localize(NY)).sum()) > 0
    from data.store import load_bars_cache

    kept, m2 = load_bars_cache(npz, js)
    assert kept.index.max() < HOLDOUT_START.tz_localize(NY) and m2["coverage_end"] == "2026-10-01"
    assert truncate_forward(cache, log) == []  # idempotent
    log_forward_access({"event": "fetch", "end": "2027-01-01"}, log)
    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["truncate", "fetch"] and all("at" in r for r in rows)


def test_the_runner_records_the_u22_inputs_and_the_cost_ledger(tmp_path):
    """A rule cell through the runner: prints unavailable (counted), no T-bill series (recorded), the as-of table
    requested by COST_TABLE, daily_costs.csv written, the account fields on the family result."""
    from experiments import ledger as L
    from experiments.spec import Cell, normalize
    from tests.test_u17 import OvernightSource

    class Src(OvernightSource):
        asked: ClassVar[list] = []

        def quotes_table(self, symbols, table="year"):
            Src.asked.append(table)
            return super().quotes_table(symbols)

        def prints(self, symbol, start, end, *, allow_holdout):
            df = self.bars(symbol, "5Min", start, end, allow_holdout=allow_holdout)
            return _prints(df, d_close=0.01)

    from experiments.runner import run

    raw = {"symbols": "SPY", "timeframe": "5Min", "start": "2022-01-03", "end": "2022-03-01", "primary": "overnight",
           "model": {"meta": "none"}}  # fmt: skip
    cell = Cell(normalize(raw), "F")
    led = L.Ledger(tmp_path / "ledger.jsonl")
    rows = run([cell], ledger=led, root=tmp_path / "root", source=Src(0.0), jobs=1, family=True)
    r = rows[0]
    assert r["status"] == "ok" and r["auction_fallbacks"] == 0 and "cash_yield" in r["data_notes"]
    assert r["cost_table"] == "asof" and Src.asked[-1] == "asof"
    dc = pd.read_csv(tmp_path / "root" / "cells" / r["cell_hash"] / "daily_costs.csv", index_col=0)
    assert dc["notional_close_auction"].sum() > 0 and dc["notional_open_auction"].sum() > 0
    assert sum(dc[f"cost_{k}"].sum() for k in FILL_CLASSES) == pytest.approx(r["cost_paid"])
    tr = pd.read_csv(tmp_path / "root" / "cells" / r["cell_hash"] / "trades.csv")
    bars = Src(0.0).bars("SPY", "5Min", "2022-01-03", "2022-03-01", allow_holdout=False)
    first = tr.sort_values("entry_b").iloc[0]
    day = pd.Timestamp(first["entry_time"]).tz_convert(NY).normalize()
    assert first["entry_px"] == pytest.approx(bars["close"].groupby(bars.index.normalize()).last().loc[day] + 0.01)
