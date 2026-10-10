"""
U16 (SPEC §16): the mechanism primaries and what they need from the harness — market-on-close entries, periodic
schedules and calendar offsets, multi-session and next-rebalance time exits, scheduled exits that free (and roll into)
a same-fill entry in the portfolio simulator, flat sides, the rule_size sizer, per-slot diagnostics — plus each
primary's causality, synthetic sanity and hand-computed trades on real SPY bars.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wfo.wfo_engine as engine
from data.events import event_series
from features.events import entry_at_close, period_starts, sample_events, schedule_events
from features.exits import exit_frame, time_exits
from features.session import session_frame
from primaries import REGISTRY, check_signal, make_primary
from primaries.diagnostics import primary_diagnostics, slot_diagnostics
from primaries.mechanism import CALENDAR_WINDOWS, MECHANISM, primary_config, session_state
from risk.portfolio import simulate_portfolio
from risk.profiles import get_profile
from sizing import make_sizer
from tests.test_costs import AUCTION, SLIP, bin_bp, const_quotes_table, session_5min
from tests.test_features import random_walk_after
from tests.test_u14 import hhmm, intraday
from utils.config import RunConfig
from wfo.backtest import primary_size_col, run_backtest

NY = "America/New_York"
ts = lambda s: pd.Timestamp(s, tz=NY)


def daily(n: int, seed: int = 1, start: str = "2022-01-03", drift: float = 3e-4, vol: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.004, n))
    idx = pd.bdate_range(start, periods=n, tz=NY)
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                         "low": np.minimum(open_, close) * 0.997, "close": close, "volume": 1e6}, index=idx)  # fmt: skip


def sched(times, tf="5Min", **kw) -> RunConfig:
    ev = {"entry_times": times, **{k: kw.pop(k) for k in ("days", "gate", "every", "day_offset") if k in kw}}
    return RunConfig.for_timeframe(tf, EVENT_SAMPLER="schedule", EVENT_PARAMS=ev, **kw)


def moc(tf="5Min", exit_params=None, **kw) -> RunConfig:
    return sched(["close"], tf, EXIT_MODEL="time", EXIT_PARAMS=exit_params or {"exit_time": "open"}, **kw)


def signals_for(df: pd.DataFrame, cfg: RunConfig, symbol: str | None = None, X: pd.DataFrame | None = None):
    """Fold-free signals of cfg.PRIMARY on every sampled event (rules fit nothing), backtest-ready."""
    ev = sample_events(df, cfg, symbol=symbol).dropna(subset=["width"])
    X = pd.DataFrame(index=ev.index) if X is None else X.reindex(ev.index)
    p = make_primary(cfg)
    fr = check_signal(p.signal(df, X, cfg), X, p)
    return fr.assign(width=ev["width"], trade_signal=fr["signed_dir"], bet_size=fr["magnitude"], meta_prob=1.0)


# ── Schedule sampler: MOC entries, periods, calendar offsets ─────────────────


def test_moc_schedule_decides_at_the_bar_before_the_close_and_fills_at_the_closing_auction():
    regular = session_5min("2024-07-02")
    early = session_5min("2024-07-03", close="13:00")
    holed = session_5min("2024-07-05").drop(ts("2024-07-05 15:55"))  # no closing-auction bar: no MOC entry
    df = pd.concat([regular, early, holed, session_5min("2024-07-08")])
    df["open"] = np.arange(len(df)) + 100.0
    df["close"] = df["open"] + 0.5
    cfg = moc()
    assert entry_at_close(cfg) and not entry_at_close(sched(["15:30"]))
    ev = schedule_events(df, cfg)
    assert [t.strftime("%m-%d %H:%M") for t in ev] == ["07-02 15:50", "07-03 12:50", "07-08 15:50"]
    lab = exit_frame(df, ev[:2], pd.Series(0.01, index=ev[:2]), cfg)
    e, x = lab["entry_pos"].to_numpy(), lab["exit_pos"].to_numpy()
    assert hhmm(df.index[e]) == ["15:55", "12:55"] and hhmm(df.index[x]) == ["09:30", "09:30"]
    np.testing.assert_array_equal(lab["entry_px"], df["close"].to_numpy()[e])  # the entry bar's CLOSE
    np.testing.assert_array_equal(lab["exit_px"], df["open"].to_numpy()[x])
    np.testing.assert_allclose(lab["ret"], df["open"].to_numpy()[x] / df["close"].to_numpy()[e] - 1, rtol=1e-15)


def test_daily_moc_entries_fill_at_the_next_sessions_close():
    d = daily(30)
    cfg = moc("1Day")
    ev = sample_events(d, cfg)
    assert ev.index.equals(d.index[:-1])
    lab = exit_frame(d, ev.index, ev["width"].fillna(0.01), cfg)
    np.testing.assert_array_equal(lab["entry_px"], d["close"].to_numpy()[lab["entry_pos"]])
    assert (lab["exit_pos"] == lab["entry_pos"] + 1).all()


def test_every_keeps_the_first_session_of_each_week_or_month_in_the_data():
    d = daily(70, start="2024-01-03")  # starts on a Wednesday: the data's first session counts as its week's first
    for every, want in (("week", lambda i: (i.weekday == 0) | (i == d.index[0])),
                        ("month", lambda i: (i.day <= 3) & (pd.Series(i.month).diff().fillna(1).to_numpy() != 0))):  # fmt: skip
        entries = d.index[d.index.get_indexer(schedule_events(d, sched(["open"], "1Day", every=every))) + 1]
        first = d.index[want(d.index)]
        assert entries.equals(first[first > d.index[0]]), every
    days = np.asarray(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-12", "2024-02-01"]), "M8[D]")
    assert period_starts(days, "week").tolist() == [True, True, False, True]
    assert period_starts(days, "month").tolist() == [True, False, False, True]


def test_day_offset_selects_the_sessions_before_a_known_calendar_day():
    d = daily(260, start="2024-01-02")
    fomc = set(event_series("calendar", "FOMC").index.tz_localize(None).normalize())
    ev = schedule_events(d, moc("1Day", {"exit_time": "close", "exit_session": 1}, days="fomc", day_offset=1))
    entry = d.index[d.index.get_indexer(ev) + 1]
    nxt = d.index[d.index.get_indexer(ev) + 2]
    assert len(ev) == 8 and all(n.tz_localize(None).normalize() in fomc for n in nxt)  # 2024's eight meetings
    assert not any(e.tz_localize(None).normalize() in fomc for e in entry)
    macro = schedule_events(d, sched(["open"], "1Day", days="macro"))
    union = schedule_events(d, sched(["open"], "1Day", days="fomc")).union(
        schedule_events(d, sched(["open"], "1Day", days="cpi_nfp"))
    )
    assert macro.equals(union)
    me = d.index[d.index.get_indexer(schedule_events(d, sched(["open"], "1Day", days="month_end"))) + 1]
    assert [t.strftime("%m-%d") for t in me[:3]] == ["01-31", "02-29", "03-28"]  # the exchange calendar (Good Friday)
    opex = d.index[d.index.get_indexer(schedule_events(d, sched(["open"], "1Day", days="opex"))) + 1]
    assert [t.strftime("%m-%d") for t in opex[:3]] == ["01-19", "02-16", "03-15"]


def test_day_offset_never_sees_an_unscheduled_meeting_before_it_is_announced():
    """The 2020-03-03 cut was announced at 10:00 that day: the session before it cannot be its eve."""
    d = daily(60, start="2020-02-03")
    ev = schedule_events(d, moc("1Day", {"exit_time": "close", "exit_session": 1}, days="fomc", day_offset=1))
    entry = d.index[d.index.get_indexer(ev) + 1]
    assert ts("2020-03-02") not in entry
    same_day = schedule_events(d, sched(["open"], "1Day", days="fomc"))  # decided at the 03-02 close: not known yet
    assert ts("2020-03-03") not in d.index[d.index.get_indexer(same_day) + 1]


# ── Time exits: exit_session, next rebalance ─────────────────────────────────


def test_exit_session_variants_by_hand():
    df = pd.concat([session_5min(d) for d in ("2024-07-01", "2024-07-02", "2024-07-05", "2024-07-08")])
    df["open"] = np.arange(len(df)) + 100.0
    df["close"] = df["open"] + 0.5
    ev = pd.DatetimeIndex([ts("2024-07-01 10:00")])
    w = pd.Series(0.01, index=ev)
    kw = {"hold_bars": None, "hold_overnight": False, "bar_minutes": 5}
    at = lambda out: out["t1"].dt.strftime("%m-%d %H:%M").tolist()
    assert at(time_exits(df, ev, w, exit_time="close", exit_session=2, **kw)) == ["07-05 15:55"]
    assert at(time_exits(df, ev, w, exit_time="open", exit_session=3, **kw)) == ["07-08 09:30"]
    out = time_exits(df, ev, w, exit_time="14:00", exit_session=1, **kw)
    assert at(out) == ["07-02 14:00"] and out["exit_px"].iloc[0] == df.loc[ts("2024-07-02 14:00"), "open"]
    assert time_exits(df, ev, w, exit_time="close", exit_session=9, **kw).empty  # past the data: dropped
    # same-session HH:MM before the entry is dropped; the next session's is fine
    late = pd.DatetimeIndex([ts("2024-07-01 15:00")])
    assert time_exits(df, late, w.reindex(late, fill_value=0.01), exit_time="14:00", **kw).empty
    assert at(time_exits(df, late, pd.Series(0.01, index=late), exit_time="14:00", exit_session=1, **kw)) == [
        "07-02 14:00"
    ]


def test_next_exit_fills_at_the_next_periods_first_open():
    d = daily(40, start="2024-01-01")
    for every, want in (("session", 1), ("week", None)):
        cfg = sched(["open"], "1Day", every=every, EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "next"})
        ev = sample_events(d, cfg)
        lab = exit_frame(d, ev.index, ev["width"].fillna(0.01), cfg)
        if want == 1:
            assert (lab["exit_pos"] == lab["entry_pos"] + 1).all()
        else:
            assert (d.index[lab["exit_pos"]].weekday == 0).all() and (d.index[lab["entry_pos"]].weekday == 0).all()
            assert (lab["exit_pos"].to_numpy()[:-1] == lab["entry_pos"].to_numpy()[1:]).all()  # back to back
        assert (lab["barrier"] == "time").all()
        np.testing.assert_array_equal(lab["exit_px"], d["open"].to_numpy()[lab["exit_pos"]])


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"EXIT_MODEL": "triple_barrier", "EXIT_PARAMS": {}}, "market-on-close"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "close"}}, "later session"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {"hold_bars": 3}}, "market-on-close"),
        (
            {
                "EXIT_MODEL": "time",
                "EXIT_PARAMS": {"exit_time": "open"},
                "POSITION_MODE": "average",
                "COST_MODEL": "slippage",
            },
            "single",
        ),
    ],
)
def test_runconfig_refuses_unsupported_moc_setups(kw, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.for_timeframe("5Min", EVENT_SAMPLER="schedule", EVENT_PARAMS={"entry_times": ["close"]}, **kw)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"EVENT_PARAMS": {"entry_times": ["close", "15:30"]}}, "cannot be mixed"),
        ({"EVENT_PARAMS": {"entry_times": ["open"], "every": "year"}}, "every"),
        ({"EVENT_PARAMS": {"entry_times": ["open"], "day_offset": -1}}, "day_offset"),
        (
            {
                "EVENT_PARAMS": {"entry_times": ["open"]},
                "EXIT_MODEL": "time",
                "EXIT_PARAMS": {"hold_bars": 2, "exit_session": 1},
            },
            "exit_session",
        ),
        (
            {
                "EVENT_PARAMS": {"entry_times": ["open"]},
                "EXIT_MODEL": "time",
                "EXIT_PARAMS": {"exit_time": "open", "exit_session": 0},
            },
            "exit_session",
        ),
    ],
)
def test_runconfig_validates_the_u16_schedule_and_exit_keys(kw, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.for_timeframe("1Day", EVENT_SAMPLER="schedule", **kw)


def test_next_exit_needs_a_schedule_and_primaries_their_timeframes():
    with pytest.raises(ValueError, match="needs EVENT_SAMPLER='schedule'"):
        RunConfig.for_timeframe("1Day", EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "next"})
    with pytest.raises(ValueError, match="runs on"):
        RunConfig.for_timeframe("5Min", PRIMARY="vol_target")
    with pytest.raises(ValueError, match="runs on"):
        primary_config("gap_fade", "1Day")
    with pytest.raises(ValueError, match="not a mechanism primary"):
        primary_config("sma_cross", "1Day")


# ── Portfolio simulator: MOC fills, freed slots, rolls ───────────────────────

Q = const_quotes_table(["_"])


def _quotes_cfg(cfg: RunConfig) -> RunConfig:
    return cfg.replace(COST_MODEL="quotes", SLIPPAGE_PCT=SLIP)


def _sim(df, sig, cfg, size_col="bet_size"):
    from risk.costs import fill_costs

    costs = None if cfg.COST_MODEL == "slippage" else {"_": fill_costs(df.index, cfg, Q)}
    return simulate_portfolio({"_": df}, {"_": sig}, cfg, get_profile("none"), size_col=size_col, costs=costs)


def _sig(df, pos, sides, sizes=1.0, width=0.01):
    idx = df.index[list(pos)]
    s = np.asarray(sides)
    return pd.DataFrame({"trade_signal": s, "signed_dir": s, "bet_size": sizes, "width": width}, index=idx)


def test_moc_entry_fills_at_the_close_and_pays_the_closing_auction():
    df = pd.concat([session_5min("2024-07-01"), session_5min("2024-07-02")])
    df.loc[ts("2024-07-01 15:55"), "close"] = 101.0
    df.loc[ts("2024-07-02 09:30"), "open"] = 103.0
    cfg = _quotes_cfg(moc())
    eq, tr, _ = _sim(df, _sig(df, [df.index.get_loc(ts("2024-07-01 15:50"))], [1]), cfg)
    t = tr.iloc[0]
    assert (df.index[t["entry_b"]], df.index[t["exit_b"]]) == (ts("2024-07-01 15:55"), ts("2024-07-02 09:30"))
    entry = 101 * (1 + AUCTION["close_auction"] * 1e-4)
    exit_ = 103 * (1 - AUCTION["open_auction"] * 1e-4)
    assert t["entry_fill"] == pytest.approx(entry, rel=1e-14) and t["exit_fill"] == pytest.approx(exit_, rel=1e-14)
    q = 10_000 / entry
    assert eq.loc[ts("2024-07-01 15:55")] == pytest.approx(10_000 + q * (101 - entry), rel=1e-13)  # marked at its close
    assert eq.iloc[-1] == pytest.approx(10_000 + q * (exit_ - entry), rel=1e-13)
    # the same through run_backtest (time exits route to the portfolio simulator even with slippage costs)
    res = run_backtest(df, _sig(df, [df.index.get_loc(ts("2024-07-01 15:50"))], [1]), moc(COST_MODEL="slippage"))
    assert res["Primary only"][1]["entry_px"].iloc[0] == 101.0 and "rolled" in res["Primary only"][1]


def test_an_unchanged_bet_rolls_through_the_rebalance_untraded():
    d = daily(60)
    cfg = _quotes_cfg(sched(["open"], "1Day", EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "next"}))
    sig = _sig(d, range(10, 50), np.ones(40))
    eq, tr, _ = _sim(d, sig, cfg)
    assert len(tr) == 40 and tr["rolled"].sum() == 39 and (tr["entry_cost_bp"].iloc[1:] == 0).all()
    e = int(tr["entry_b"].iloc[0])
    fill = d["open"].iloc[e] * (1 + AUCTION["open_auction"] * 1e-4)
    q = 10_000 / fill
    held = eq.iloc[e:50]  # one position from open[11] through the open of bar 51 (the last bet's next open)
    np.testing.assert_allclose(held, 10_000 + q * (d["close"].iloc[e:50] - fill), rtol=1e-13)
    px = d["open"].iloc[51]
    out = px * (1 - AUCTION["open_auction"] * 1e-4)
    assert eq.attrs["turnover"] == pytest.approx(1 + q * out / (10_000 + q * (px - fill)), rel=1e-12)  # two orders


def test_a_resized_or_flipped_bet_pays_only_the_traded_shares():
    d = daily(30)
    cfg = _quotes_cfg(sched(["open"], "1Day", EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "next"}))
    sig = _sig(d, [5, 6, 7, 8], [1, 1, 1, -1], sizes=[1.0, 0.5, 0.5, 0.5])
    eq, tr, _ = _sim(d, sig, cfg)
    c = AUCTION["open_auction"] * 1e-4
    q0 = tr["qty"].iloc[0]
    b1 = int(tr["entry_b"].iloc[1])
    E1 = eq.iloc[b1 - 1]
    px1 = d["open"].iloc[b1]
    q1 = 0.5 * E1 / (px1 * (1 + c))
    assert tr["qty"].iloc[1] == pytest.approx(q1, rel=1e-13)
    assert tr["entry_fill"].iloc[1] == pytest.approx(px1 * (1 + c * (q0 - q1) / q1), rel=1e-13)
    assert tr["exit_fill"].iloc[0] == px1 and tr["exit_cost_bp"].iloc[0] == 0  # rolled: no exit cost
    assert tr["qty"].iloc[2] == tr["qty"].iloc[1] and tr["entry_cost_bp"].iloc[2] == 0  # unchanged: held
    b3 = int(tr["entry_b"].iloc[3])
    q3 = tr["qty"].iloc[3]
    px3 = d["open"].iloc[b3]
    assert tr["entry_fill"].iloc[3] == pytest.approx(px3 * (1 - c * (q3 + q1) / q3), rel=1e-12)  # a short: −c


def test_a_time_exit_at_the_open_frees_its_symbol_for_an_entry_at_that_open():
    """A MOO exit and a 09:30 entry on the same bar (before U16 the second bet was skipped) are one order."""
    df = pd.concat([session_5min(d) for d in ("2024-07-01", "2024-07-02", "2024-07-05", "2024-07-08")])
    cfg = sched(["open"], EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "open"}, COST_MODEL="slippage")
    ev = schedule_events(df, cfg)
    assert len(ev) == 3  # the last one has no next open: no exit, dropped
    _, tr = run_backtest(df, _sig(df, df.index.get_indexer(ev), [1, 1, 1]), cfg)["Primary only"]
    assert len(tr) == 2 and tr["rolled"].tolist() == [True, False]


def test_a_triple_barrier_gap_exit_does_not_free_the_symbol():
    df = pd.concat([session_5min(d) for d in ("2024-07-01", "2024-07-02")])
    df["open"] = df["close"] = 100.0
    i = df.index.get_loc(ts("2024-07-01 10:00"))
    for col in ("open", "high", "low", "close"):  # gaps through the upper barrier at open[i + 3]
        df.iloc[i + 3 :, df.columns.get_loc(col)] = 103.0
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_PCT=SLIP)
    _, tr, _ = _sim(df, _sig(df, [i, i + 2], [1, 1], width=0.01), cfg)
    assert len(tr) == 1 and tr["exit_reason"].iloc[0] == "upper"  # the entry at open[i + 3] stays blocked


# ── rule_size, flat sides, the primary stream ────────────────────────────────


def test_rule_size_passes_the_magnitude_and_equals_fixed_at_one():
    cfg = RunConfig.for_timeframe("1Day", SIZER="rule_size")
    p = np.array([0.4, 0.5, 0.6, 0.9])
    rs, fx = make_sizer("rule_size", cfg), make_sizer("fixed", cfg)
    np.testing.assert_array_equal(rs.size(p, hint=np.ones(4)), fx.size(p))
    np.testing.assert_array_equal(rs.size(p, hint=[0.3, 0.3, 0.3, 0.7]), [0, 0.3, 0.3, 0.7])
    for bad in (None, [1, 1, 1, 1.5], [np.nan, 1, 1, 1]):
        with pytest.raises(ValueError, match="rule_size"):
            rs.size(p, hint=bad)
    assert primary_size_col(cfg) == "magnitude" and primary_size_col(cfg.replace(SIZER="fixed")) is None


def test_check_signal_allows_flat_only_for_flat_primaries_and_no_short_when_long_only():
    X = pd.DataFrame(index=pd.date_range("2024-01-02", periods=3, tz=NY))
    fr = lambda sides: pd.DataFrame({"clf_prob": np.nan, "direction": 0, "signed_dir": sides,
                                     "magnitude": 1.0, "signal": 0.0, "confidence": 1.0}, index=X.index)  # fmt: skip
    check_signal(fr([1, 0, -1]), X, make_primary(RunConfig.for_timeframe("1Day", PRIMARY="tsmom")))
    with pytest.raises(ValueError, match="outside"):
        check_signal(fr([1, 0, -1]), X, make_primary(RunConfig.for_timeframe("1Day", PRIMARY="sma_cross")))
    long_only = make_primary(RunConfig.for_timeframe("1Day", PRIMARY="tsmom", PRIMARY_PARAMS={"long_only": True}))
    check_signal(fr([1, 0, 0]), X, long_only)
    with pytest.raises(ValueError, match="outside"):
        check_signal(fr([1, 0, -1]), X, long_only)
    assert make_primary(RunConfig.for_timeframe("1Day", PRIMARY="vol_target")).long_only


def test_flat_events_are_not_meta_fitted_and_never_traded(monkeypatch):
    d = daily(700, seed=5, start="2021-01-04", drift=-2e-4)
    cfg = primary_config("tsmom", "1Day", {"long_only": True, "lookback": 21},
                         EVENT_PARAMS={"entry_times": ["open"], "every": "session"}, INITIAL_TRAIN=300, VAL=150,
                         TEST=50, MIN_TRAIN_EVENTS=20, MIN_VAL_EVENTS=10, COST_MODEL="slippage",
                         FEATURE_GROUPS=("wavelet_core",))  # fmt: skip
    seen = []
    real = engine.fit_meta_model

    def spy(X_fit, primary_fit, meta_labels, *a, **k):
        seen.append(primary_fit["signed_dir"].copy())
        return real(X_fit, primary_fit, meta_labels, *a, **k)

    monkeypatch.setattr(engine, "fit_meta_model", spy)
    sig = engine.run_wfo(d, cfg)
    assert seen and all((s == 1).all() for s in seen)  # flat rows dropped before the meta-model
    assert (sig["signed_dir"] == 0).any() and (sig.loc[sig["signed_dir"] == 0, "trade_signal"] == 0).all()
    _, tr = run_backtest(d.loc[sig.index[0] :], sig, cfg)["Primary only"]
    assert len(tr) and (tr["side"] > 0).all()


def test_the_primary_stream_is_sized_by_the_magnitude_under_rule_size():
    d = daily(200)
    cfg = primary_config("vol_target", "1Day", COST_MODEL="slippage", SIZE_STEP=0)
    sig = signals_for(d, cfg)
    _, tr = run_backtest(d, sig, cfg)["Primary only"]
    np.testing.assert_allclose(tr["size"].to_numpy(), sig.loc[tr.index, "magnitude"].to_numpy(), rtol=0)
    assert tr["size"].min() < 1


def test_rule_size_reproduces_fixed_through_the_wfo_when_the_magnitude_is_one():
    df = intraday(60, seed=21)
    kw = {"INITIAL_TRAIN": 30, "VAL": 15, "TEST": 5, "MIN_TRAIN_EVENTS": 20, "MIN_VAL_EVENTS": 10,
          "COST_MODEL": "slippage", "FEATURE_GROUPS": ("wavelet_core",), "META_THRESH": 0.05}  # fmt: skip
    a = engine.run_wfo(df, primary_config("overnight", "5Min", **kw))
    b = engine.run_wfo(df, primary_config("overnight", "5Min", SIZER="fixed", **kw))
    assert (a["magnitude"] == 1).all() and len(a) >= 10 and (a["bet_size"] > 0).any()
    pd.testing.assert_frame_equal(a, b)
    ra, rb = (run_backtest(df.loc[s.index[0] :], s, c) for s, c in
              ((a, primary_config("overnight", "5Min", **kw)), (b, primary_config("overnight", "5Min", SIZER="fixed",
                                                                                **kw))))  # fmt: skip
    for k in ra:
        pd.testing.assert_series_equal(ra[k][0], rb[k][0])


# ── The primaries: causality and synthetic sanity ────────────────────────────


def primary_cases():
    for name in MECHANISM:
        for tf in REGISTRY[name].TIMEFRAMES:
            yield name, tf, {}
    yield "tsmom", "1Day", {"estimator": "modwt"}
    yield "vol_target", "1Day", {"vol_source": "ewm", "band": 0.0}
    yield "gap_fade", "5Min", {"follow_on_news": True}
    yield "gap_fade", "5Min", {"exit": "hysteresis"}
    yield "intraday_momentum", "5Min", {"predictor": "first30", "threshold_sigma": 0.0}
    yield "event_reaction", "5Min", {"release": "cpi_nfp"}
    for w in CALENDAR_WINDOWS:
        yield "calendar_drift", "1Day", {"window": w}


@pytest.mark.parametrize("name,tf,params", list(primary_cases()))
def test_every_mechanism_primary_is_causal(name, tf, params):
    """Bars after c changed: the events up to c and their sides / magnitudes are unchanged (SPEC §8)."""
    if tf == "1Day":
        df = daily(600, seed=3, start="2022-01-03")
        cuts = (400, 520)
    else:
        df = intraday(130, seed=4, start="2024-01-02")
        cuts = (78 * 54 + 30, 78 * 112 + 76)  # inside the 2024-03-20 and 2024-06-12 FOMC weeks
    cfg = primary_config(name, tf, params)
    a = signals_for(df, cfg, symbol="SPY")
    assert len(a) >= 3
    for k, c in enumerate(cuts):
        b = signals_for(random_walk_after(df, c, 10 + k), cfg, symbol="SPY")
        cut = df.index[c]
        pa, pb = a.loc[:cut], b.loc[:cut]
        assert pa.index.equals(pb.index)
        cols = ["signed_dir", "magnitude"]
        pd.testing.assert_frame_equal(pa[cols], pb[cols])
    if name == "gap_fade" and params.get("exit") == "hysteresis":
        score = make_primary(cfg).score(df, cfg)
        b = make_primary(cfg).score(random_walk_after(df, cuts[0], 3), cfg)
        pd.testing.assert_series_equal(score.loc[: df.index[cuts[0]]], b.loc[: df.index[cuts[0]]])


def test_session_state_equals_the_session_frame_without_a_profile():
    df = intraday(30, seed=6)
    cfg = RunConfig.for_timeframe("5Min")
    st, sf = session_state(df, cfg), session_frame(df, cfg)
    for col in ("open_to_now", "first30", "gap_sigma", "sigma_day", "sigma_overnight"):
        pd.testing.assert_series_equal(st[col], sf[col], check_names=False)


def _planted_day(df: pd.DataFrame, day: str, path) -> pd.DataFrame:
    """Overwrite one session's bars with a price path relative to its open (log returns per bar)."""
    out = df.copy()
    m = out.index.normalize() == ts(day)
    o = out.loc[m, "open"].iloc[0]
    c = o * np.exp(np.cumsum(path))
    out.loc[m, "close"] = c
    out.loc[m, "open"] = np.r_[o, c[:-1]]
    out.loc[m, "high"] = np.maximum(out.loc[m, "open"], c) * 1.0002
    out.loc[m, "low"] = np.minimum(out.loc[m, "open"], c) * 0.9998
    return out


def test_intraday_momentum_goes_long_a_morning_rally_and_short_a_selloff():
    df = intraday(40, seed=7, start="2024-03-01")
    up = np.r_[np.full(12, 2e-3), np.zeros(66)]  # +2.4 % in the first hour, flat afterwards
    df = _planted_day(_planted_day(df, "2024-04-10", up), "2024-04-17", -up)
    for pred in ("open_to_now", "first30"):
        cfg = primary_config("intraday_momentum", "5Min", {"predictor": pred})
        sig = signals_for(df, cfg)
        assert (
            sig.loc[ts("2024-04-10 15:25"), "signed_dir"] == 1 and sig.loc[ts("2024-04-17 15:25"), "signed_dir"] == -1
        )
        assert sig.loc[ts("2024-04-10 15:25"), "magnitude"] == 1.0
    quiet = _planted_day(df, "2024-04-24", np.zeros(78))  # no move: the gate drops the day
    assert ts("2024-04-24 15:25") not in signals_for(quiet, primary_config("intraday_momentum", "5Min")).index


def test_gap_fade_shorts_an_up_gap_on_a_quiet_day_and_skips_it_on_fomc_day():
    df = intraday(50, seed=8, start="2024-04-01")
    for day in ("2024-05-07", "2024-05-01"):  # a non-macro Tuesday; the 2024-05-01 FOMC day
        i = df.index.get_loc(ts(f"{day} 09:30"))
        df.iloc[i:, :4] *= 1.03  # a +3 % overnight gap
    fade = signals_for(df, primary_config("gap_fade", "5Min"))
    assert fade.loc[ts("2024-05-07 09:30"), "signed_dir"] == -1 and fade.loc[ts("2024-05-07 09:30"), "magnitude"] == 1
    assert ts("2024-05-01 09:30") not in fade.index
    follow = signals_for(df, primary_config("gap_fade", "5Min", {"follow_on_news": True}))
    assert follow.loc[ts("2024-05-01 09:30"), "signed_dir"] == 1 and ts("2024-05-07 09:30") not in follow.index


def test_gap_fade_hysteresis_exits_when_the_gap_is_filled():
    df = intraday(40, seed=9, start="2024-04-01")
    i = df.index.get_loc(ts("2024-05-07 09:30"))
    prev = df["close"].iloc[i - 1]
    df.iloc[i:, :4] *= 1.03
    # the price falls back below the previous close at the 09:55 bar's close → exit at the 10:00 open
    path = np.full(78, 0.0)
    path[1:6] = np.log(prev * 0.999 / (df["open"].iloc[i])) / 5
    df = _planted_day(df, "2024-05-07", path)
    cfg = primary_config("gap_fade", "5Min", {"exit": "hysteresis"})
    sig = signals_for(df, cfg)
    t = ts("2024-05-07 09:30")
    assert sig.loc[t, "signed_dir"] == -1
    out = exit_frame(df, pd.DatetimeIndex([t]), sig.loc[[t], "width"], cfg, side=sig.loc[[t], "signed_dir"])
    assert df.index[out["exit_pos"].iloc[0]] == ts("2024-05-07 10:00") and out["barrier"].iloc[0] == "hysteresis"
    capped = primary_config("gap_fade", "5Min", {"exit": "hysteresis"}).EXIT_PARAMS["max_bars"]
    assert capped == 11  # 09:35 … 10:25 held: the cap is the 10:25 bar's close (10:30), as the time exit's 10:30 open


def test_event_reaction_follows_the_post_release_move_on_release_days_only():
    df = intraday(50, seed=10, start="2024-04-01")
    k = (14 * 60 - 570) // 5  # the 14:00 bar
    path = np.zeros(78)
    path[k : k + 3] = -3e-3  # −0.9 % from 14:00 to 14:15
    df = _planted_day(df, "2024-05-01", path)
    sig = signals_for(df, primary_config("event_reaction", "5Min"))
    assert list(sig.index) == [ts("2024-05-01 14:10")] and sig["signed_dir"].iloc[0] == -1
    lab = exit_frame(df, sig.index, sig["width"], primary_config("event_reaction", "5Min"))
    assert hhmm(df.index[lab["entry_pos"]]) == ["14:15"] and hhmm(lab["t1"]) == ["15:45"]
    cpi = signals_for(df, primary_config("event_reaction", "5Min", {"release": "cpi_nfp"}))
    days = {t.strftime("%m-%d") for t in cpi.index}
    assert days == {"04-05", "04-10", "05-03", "05-15", "06-07"} and set(hhmm(cpi.index)) == {"09:40"}


def test_overnight_is_long_every_session_and_flat_above_vix_max():
    df = intraday(5, seed=11)
    sig = signals_for(df, primary_config("overnight", "5Min"))
    assert (sig["signed_dir"] == 1).all() and len(sig) == 4  # the last session has no next open: dropped later
    cfg = primary_config("overnight", "5Min", {"vix_max": 25.0})
    assert "vol_state" in cfg.FEATURE_GROUPS
    X = pd.DataFrame({"vix": [20.0, 30.0, np.nan, 25.0]}, index=sig.index)
    fr = make_primary(cfg).signal(df, X, cfg)
    assert fr["signed_dir"].tolist() == [1, 0, 0, 1]
    with pytest.raises(ValueError, match="vol_state"):
        make_primary(cfg).signal(df, pd.DataFrame(index=sig.index), cfg)


def test_vol_target_scales_down_with_volatility_and_holds_within_the_band():
    calm = daily(120, seed=12, vol=0.005)  # ≈ 8 % annualized: m = 1
    wild = daily(120, seed=12, vol=0.03)  # ≈ 48 %: m ≈ 0.31
    for d, lo, hi in ((calm, 1.0, 1.0), (wild, 0.2, 0.45)):
        sig = signals_for(d, primary_config("vol_target", "1Day"))
        m = sig["magnitude"].iloc[30:]
        assert lo <= m.min() and m.max() <= hi and (sig["signed_dir"].iloc[30:] == 1).all()
    sig = signals_for(wild, primary_config("vol_target", "1Day"))
    target = np.minimum(1, 0.15 / (np.log(wild["close"]).diff().rolling(21).std() * np.sqrt(252)))
    m = sig["magnitude"]
    changes = m[m.diff().fillna(1) != 0]
    assert (m - target.reindex(m.index)).abs().max() < 0.10 + 1e-12  # never more than the band from the target
    assert len(changes) < len(m) / 3  # the band holds the exposure most days
    free = signals_for(wild, primary_config("vol_target", "1Day", {"band": 0.0}))
    np.testing.assert_allclose(free["magnitude"].iloc[25:], target.reindex(free.index).iloc[25:], rtol=1e-12)


def test_tsmom_follows_the_trailing_year_and_long_only_is_flat_in_a_downtrend():
    up = daily(400, seed=13, drift=2e-3)
    down = daily(400, seed=13, drift=-2e-3)
    for d, side in ((up, 1), (down, -1)):
        sig = signals_for(d, primary_config("tsmom", "1Day"))
        late = sig.loc[d.index[300] :]
        assert (late["signed_dir"] == side).all()
        assert (sig.loc[: d.index[250], "signed_dir"] == 0).all()  # no trailing year yet: flat
        assert late["magnitude"].between(0.4, 1.0).all()  # 10 % / ≈ 19 % annualized
    lo = signals_for(down, primary_config("tsmom", "1Day", {"long_only": True}))
    assert (lo["signed_dir"] == 0).all()


def test_weekly_reversal_fades_last_weeks_move():
    d = daily(60, seed=14)
    sig = signals_for(d, primary_config("weekly_reversal", "1Day"))
    pos = d.index.get_indexer(sig.index)
    r = np.log(d["close"].to_numpy()[pos] / d["close"].to_numpy()[pos - 5])
    ok = pos >= 5
    np.testing.assert_array_equal(sig["signed_dir"].to_numpy()[ok], np.where(r[ok] >= 0, -1, 1))
    assert (d.index[pos + 1].weekday == 0).all()


def test_calendar_windows_on_2024_by_hand():
    d = daily(330, start="2023-09-01")  # Jan 2024 is past the width warm-up
    cases = {
        "fomc_pre": ("01-30", "01-31"),  # t−1 close → t close (1Day)
        "tom": ("01-30", "02-05"),  # close of session −2 → close of +3 (−1 = 01-31, +1 … +3 = 02-01 … 02-05)
        "month_end": ("01-30", "01-31"),
        "cpi_nfp": ("01-04", "01-05"),
        "opex_week": ("01-12", "01-19"),
    }
    for window, (entry, exit_) in cases.items():
        cfg = primary_config("calendar_drift", "1Day", {"window": window})
        sig = signals_for(d, cfg)
        lab = exit_frame(d, sig.index, sig["width"], cfg)
        first = lab[d.index[lab["entry_pos"]] >= ts("2024-01-01")].iloc[0]
        got = (d.index[first["entry_pos"]].strftime("%m-%d"), d.index[first["exit_pos"]].strftime("%m-%d"))
        assert got == (entry, exit_), window
        assert first["entry_px"] == d["close"].iloc[first["entry_pos"]]
        assert first["exit_px"] == d["close"].iloc[first["exit_pos"]]
    intra = primary_config("calendar_drift", "5Min", {"window": "fomc_pre"})
    assert intra.EXIT_PARAMS == {"exit_time": "14:00", "exit_session": 1}


# ── Hand-computed trades on real SPY 5Min bars (cached data, no network) ─────


class _NoNetwork:
    def __getattr__(self, name):
        raise AssertionError("the test must run on cached bars only")


@pytest.fixture(scope="module")
def spy():
    from data.bars import DEFAULT_CACHE_DIR, load_bars

    if not (Path(DEFAULT_CACHE_DIR) / "sip" / "all" / "5Min" / "SPY.npz").exists():
        pytest.skip("no cached SPY bars")
    return load_bars("SPY", "5Min", "2024-02-01", "2024-04-30", source=_NoNetwork())


def _one_trade(df, cfg, event_ts, symbol="SPY"):
    sig = signals_for(df, cfg, symbol=symbol)
    sig = sig.loc[[event_ts]]
    _, tr, _ = _sim(df, sig, _quotes_cfg(cfg), size_col="magnitude")
    assert len(tr) == 1
    return sig.iloc[0], tr.iloc[0]


def _cost(kind: str, bar: str) -> float:
    return AUCTION[kind] * 1e-4 if kind in AUCTION else SLIP + bin_bp(bar) * 1e-4


def test_real_spy_overnight_trade_by_hand(spy):
    _s, t = _one_trade(spy, primary_config("overnight", "5Min"), ts("2024-03-05 15:50"))
    c_in, px_in = _cost("close_auction", ""), spy.loc[ts("2024-03-05 15:55"), "close"]
    c_out, px_out = _cost("open_auction", ""), spy.loc[ts("2024-03-06 09:30"), "open"]
    assert spy.index[t["entry_b"]] == ts("2024-03-05 15:55") and spy.index[t["exit_b"]] == ts("2024-03-06 09:30")
    assert t["entry_fill"] == pytest.approx(px_in * (1 + c_in), rel=1e-14)
    assert t["exit_fill"] == pytest.approx(px_out * (1 - c_out), rel=1e-14)


def test_real_spy_intraday_momentum_trade_by_hand(spy):
    cfg = primary_config("intraday_momentum", "5Min")
    sig = signals_for(spy, cfg)
    ev = sig.index[0]
    day = ev.strftime("%Y-%m-%d")
    o = spy.loc[ts(f"{day} 09:30"), "open"]
    ret = np.log(spy.loc[ts(f"{day} 15:25"), "close"] / o)
    s, t = _one_trade(spy, cfg, ev)
    assert hhmm([ev]) == ["15:25"] and s["signed_dir"] == np.sign(ret)
    px_in, px_out = spy.loc[ts(f"{day} 15:30"), "open"], spy.loc[ts(f"{day} 15:55"), "close"]
    side = int(np.sign(ret))
    assert t["entry_fill"] == pytest.approx(px_in * (1 + side * _cost("", "15:30")), rel=1e-14)
    assert t["exit_fill"] == pytest.approx(px_out * (1 - side * _cost("close_auction", "")), rel=1e-14)


def test_real_spy_gap_fade_trade_by_hand(spy):
    cfg = primary_config("gap_fade", "5Min")
    sig = signals_for(spy, cfg)
    ev = sig.index[0]
    day = ev.strftime("%Y-%m-%d")
    prev_close = spy["close"].iloc[spy.index.get_loc(ts(f"{day} 09:30")) - 1]
    gap = spy.loc[ts(f"{day} 09:30"), "open"] / prev_close - 1
    s, t = _one_trade(spy, cfg, ev)
    side = -int(np.sign(gap))
    assert hhmm([ev]) == ["09:30"] and s["signed_dir"] == side
    for k in ("FOMC", "CPI", "NFP"):
        assert pd.Timestamp(day) not in event_series("calendar", k).index.tz_localize(None)
    px_in, px_out = spy.loc[ts(f"{day} 09:35"), "open"], spy.loc[ts(f"{day} 10:30"), "open"]
    assert t["entry_fill"] == pytest.approx(px_in * (1 + side * _cost("", "09:30")), rel=1e-14)
    assert t["exit_fill"] == pytest.approx(px_out * (1 - side * _cost("", "10:30")), rel=1e-14)


def test_real_spy_event_reaction_trade_by_hand(spy):
    cfg = primary_config("event_reaction", "5Min")
    react = np.log(spy.loc[ts("2024-03-20 14:10"), "close"] / spy.loc[ts("2024-03-20 14:00"), "open"])
    s, t = _one_trade(spy, cfg, ts("2024-03-20 14:10"))
    side = int(np.sign(react))
    assert s["signed_dir"] == side
    px_in, px_out = spy.loc[ts("2024-03-20 14:15"), "open"], spy.loc[ts("2024-03-20 15:45"), "open"]
    assert t["entry_fill"] == pytest.approx(px_in * (1 + side * _cost("", "14:15")), rel=1e-14)
    assert t["exit_fill"] == pytest.approx(px_out * (1 - side * _cost("", "15:45")), rel=1e-14)


def test_real_spy_pre_fomc_window_by_hand(spy):
    cfg = primary_config("calendar_drift", "5Min", {"window": "fomc_pre"})
    _s, t = _one_trade(spy, cfg, ts("2024-03-19 15:50"))
    px_in, px_out = spy.loc[ts("2024-03-19 15:55"), "close"], spy.loc[ts("2024-03-20 14:00"), "open"]
    assert spy.index[t["exit_b"]] == ts("2024-03-20 14:00")
    assert t["entry_fill"] == pytest.approx(px_in * (1 + _cost("close_auction", "")), rel=1e-14)
    assert t["exit_fill"] == pytest.approx(px_out * (1 - _cost("", "14:00")), rel=1e-14)


# ── Diagnostics and the WFO end to end ───────────────────────────────────────


def test_slot_diagnostics_split_by_entry_slot_and_skip_flat_events():
    df = intraday(30, seed=15)
    cfg = primary_config("intraday_momentum", "5Min", {"threshold_sigma": 0.0},
                         EVENT_PARAMS={"entry_times": ["15:00", "15:30", "15:45"]})  # fmt: skip
    sig = signals_for(df, cfg).assign(fold=1, primary="intraday_momentum")
    tab = slot_diagnostics(df, sig, cfg, symbol="X")
    assert tab["slot"].tolist() == ["15:00", "15:30", "15:45", "all"]
    assert tab["n_events"].iloc[:3].sum() == tab["n_events"].iloc[3] == len(sig)
    span = df.index[(df.index >= sig.index[0]) & (df.index <= sig.index[-1])]
    assert tab["per_session"].iloc[0] == pytest.approx(tab["n_events"].iloc[0] / len(set(span.date)), rel=1e-12)
    flat = sig.copy()
    flat.iloc[:5, flat.columns.get_loc("signed_dir")] = 0
    d = primary_diagnostics(df, flat, cfg)
    assert d["n_events"].iloc[-1] == len(sig) - 5 and d["n_flat"].iloc[-1] == 5
    assert "n_flat" not in primary_diagnostics(df, sig, cfg)


def test_run_wfo_with_mechanism_primaries_keeps_the_signal_schema():
    df = intraday(60, seed=16)
    kw = {"INITIAL_TRAIN": 30, "VAL": 15, "TEST": 5, "MIN_TRAIN_EVENTS": 10, "MIN_VAL_EVENTS": 5,
          "COST_MODEL": "slippage", "FEATURE_GROUPS": ("wavelet_core", "session")}  # fmt: skip
    ref = None
    for name, params in (("intraday_momentum", {"threshold_sigma": 0.0}), ("overnight", {}), ("gap_fade",
                                                                                            {"min_gap_sigma": 0.0})):  # fmt: skip
        sig = engine.run_wfo(df, primary_config(name, "5Min", params, **kw))
        assert len(sig) and set(sig["primary"]) == {name}
        cols = list(sig.columns)
        ref = ref or cols
        assert cols == ref and cols[:6] == ["clf_prob", "direction", "signed_dir", "magnitude", "signal", "confidence"]


def test_an_experiment_cell_records_the_mechanism_primarys_sampler_and_exit():
    from experiments.spec import expand

    doc = {"stage": "U10", "defaults": {"symbols": "SPY", "timeframe": "5Min", "start": "2024-01-02",
                                        "end": "2024-06-28", "sizer": "rule_size"},
           "cells": [{"primary": "overnight"},
                     {"primary": "gap_fade", "overrides": {"EXIT_PARAMS": {"exit_time": "10:00"}}}]}  # fmt: skip
    a, b = expand(doc)
    assert a.spec["overrides"] == {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["close"]},
                                   "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "open"}}  # fmt: skip
    ref = primary_config("overnight", "5Min")
    assert all(getattr(a.config(), k) == getattr(ref, k) for k in ("EVENT_SAMPLER", "EVENT_PARAMS", "EXIT_MODEL",
                                                                     "EXIT_PARAMS", "SIZER"))  # fmt: skip
    assert (
        b.spec["overrides"]["EXIT_PARAMS"] == {"exit_time": "10:00"} and b.config().EXIT_PARAMS["exit_time"] == "10:00"
    )
    with pytest.raises(ValueError, match="runs on"):
        expand({**doc, "cells": [{"primary": "gap_fade", "timeframe": "1Day"}]})


def test_flat_events_stay_out_of_the_rolling_calibration_and_the_meta_diagnostics(monkeypatch):
    from wfo.wfo_metrics import meta_outcomes

    d = daily(700, seed=5, start="2021-01-04", drift=-2e-4)
    cfg = primary_config("tsmom", "1Day", {"long_only": True, "lookback": 21},
                         EVENT_PARAMS={"entry_times": ["open"], "every": "session"}, INITIAL_TRAIN=300, VAL=60,
                         TEST=40, MIN_TRAIN_EVENTS=20, MIN_VAL_EVENTS=10, COST_MODEL="slippage",
                         FEATURE_GROUPS=("wavelet_core",), META_MODEL="logit_l2", META_TRAIN="oof")  # fmt: skip
    added = []
    real = engine.CalHistory.add

    def spy(self, raw, y, spans=None):
        added.append(raw.index)
        return real(self, raw, y, spans)

    monkeypatch.setattr(engine.CalHistory, "add", spy)
    sig = engine.run_wfo(d, cfg)
    flat = sig.index[sig["signed_dir"] == 0]
    assert len(flat) and added and not any(ix.intersection(flat).size for ix in added)
    out = meta_outcomes(d, sig, cfg)
    sided = int((sig["signed_dir"] != 0).sum())
    assert out.index.intersection(flat).empty and sided - 1 <= len(out) <= sided  # the last may have no exit


@pytest.mark.parametrize("name,tf,params", list(primary_cases()))
def test_every_mechanism_primary_is_causal_at_each_event(name, tf, params):
    """Strict cut: for events across the sample, every bar after the EVENT BAR replaced — the event is still sampled
    and its side / magnitude unchanged (catches a one-bar peek that fixed cuts miss on sparse schedules)."""
    df = daily(600, seed=3, start="2022-01-03") if tf == "1Day" else intraday(130, seed=4, start="2024-01-02")
    cfg = primary_config(name, tf, params)
    a = signals_for(df, cfg, symbol="SPY")
    a = a[a["signed_dir"] != 0] if (a["signed_dir"] != 0).sum() >= 3 else a
    picks = a.index[np.unique(np.linspace(0, len(a) - 1, 6).astype(int))]
    for k, t in enumerate(picks):
        c = df.index.get_loc(t)
        if c >= len(df) - 2:
            continue
        b = signals_for(random_walk_after(df, c, 20 + k), cfg, symbol="SPY")
        assert t in b.index, (name, t)
        pd.testing.assert_series_equal(a.loc[t, ["signed_dir", "magnitude"]], b.loc[t, ["signed_dir", "magnitude"]],
                                       check_names=False, obj=f"{name} {t}")  # fmt: skip


def test_a_position_exiting_at_the_close_still_blocks_an_entry_at_that_bars_open():
    df = pd.concat([session_5min("2024-07-01"), session_5min("2024-07-02")])
    cfg = sched(["10:05", "15:55"], EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "close"}, COST_MODEL="slippage")
    ev = schedule_events(df, cfg)
    assert hhmm(ev[:2]) == ["10:00", "15:50"]
    _, tr = run_backtest(df, _sig(df, df.index.get_indexer(ev[:2]), [1, 1]), cfg)["Primary only"]
    assert len(tr) == 1  # the MOC exit of the first fills after the 15:55 open: the second entry is blocked


def test_an_experiment_cell_gets_rule_size_and_refuses_a_missing_feature_group():
    from experiments.spec import expand

    base = {"symbols": "SPY", "timeframe": "1Day", "start": "2016-01-04", "end": "2025-10-01"}
    (cell,) = expand({"stage": "U10", "defaults": base, "cells": [{"primary": "vol_target"}]})
    assert cell.spec["sizer"] == "rule_size" and cell.config().SIZER == "rule_size"
    (fixed,) = expand({"stage": "U10", "defaults": {**base, "sizer": "fixed"}, "cells": [{"primary": "vol_target"}]})
    assert fixed.spec["sizer"] == "fixed"  # an explicit sizer is kept
    with pytest.raises(ValueError, match="vol_state"):
        expand({"stage": "U10", "defaults": base,
                "cells": [{"primary": {"name": "overnight", "params": {"vix_max": 30}}}]})  # fmt: skip
