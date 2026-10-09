"""
U14 (SPEC §13–§14): the time-of-day volatility profile, directional-change and scheduled samplers, the time and
hysteresis exits, the session frame, and their wiring through the WFO engine and the backtests.
"""

import numpy as np
import pandas as pd
import pytest

import wfo.wfo_engine as engine
from features import cache as feature_cache
from features.events import dc_events, sample_events, schedule_events
from features.exits import exit_frame, exit_phase, hysteresis_exits, time_exits
from features.session import session_frame
from features.triple_barrier_labels import triple_barrier_labels
from features.vol_profile import VolProfile, bar_volatility, hold_scale, isom_counts, n_slots
from tests.test_costs import AUCTION, SLIP, bin_bp, const_quotes_table, session_5min, signals_at
from tests.test_features import random_walk_after
from utils.config import RunConfig
from wfo.backtest import run_backtest

NY = "America/New_York"
CFG5 = RunConfig.for_timeframe("5Min")


def intraday(n_sessions: int, seed: int = 0, start: str = "2024-01-02", u_shape: bool = False) -> pd.DataFrame:
    """Synthetic 5Min sessions (78 bars, NY stamps); with u_shape the bar σ is 3× at the open and close slots."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_sessions)
    idx = pd.DatetimeIndex(
        [t for d in days for t in pd.date_range(d + pd.Timedelta("09:30:00"), periods=78, freq="5min")]
    ).tz_localize(NY)
    b = np.tile(np.arange(78), n_sessions)
    u = 1 + 2 * ((b - 38.5) / 38.5) ** 2 if u_shape else np.ones(len(b))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx)) * u))
    o = np.r_[100.0, c[:-1]]
    first = np.r_[True, idx.normalize()[1:] != idx.normalize()[:-1]]
    o[first] *= np.exp(rng.normal(0, 3e-3, first.sum()))  # overnight gaps
    spr = rng.uniform(1e-4, 5e-4, len(idx))
    vol = rng.integers(1e4, 1e6, len(idx)).astype(float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spr), "low": np.minimum(o, c) * (1 - spr),
                         "close": c, "volume": vol}, index=idx)  # fmt: skip


def hhmm(index) -> list[str]:
    return list(pd.DatetimeIndex(index).strftime("%H:%M"))


def assert_prefix_equal(a: pd.DataFrame | pd.Series, b, c_ts, msg=""):
    a, b = a.loc[:c_ts], b.loc[:c_ts]
    assert a.index.equals(b.index), msg
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12, equal_nan=True,
                               err_msg=msg)  # fmt: skip


# ── VolProfile (SPEC §14) ────────────────────────────────────────────────────


def test_profile_recovers_a_planted_u_shape_and_is_flat_without_one():
    s = VolProfile.fit(intraday(250, seed=1, u_shape=True), 5).s
    assert len(s) == 78 == n_slots(5)
    assert 2.4 < s[0] / s[39] < 3.6 and 2.4 < s[77] / s[39] < 3.6  # planted 3×
    assert s[:10].mean() > s[30:48].mean() < s[-10:].mean()
    flat = VolProfile.fit(intraday(250, seed=2), 5).s
    assert np.all(np.abs(flat - 1) < 0.2)
    assert [n_slots(m) for m in (5, 15, 30, 60)] == [78, 26, 13, 7]


def test_tod_sigma_with_a_flat_profile_is_bit_identical_to_plain_sigma():
    close = intraday(20, seed=3)["close"]
    pd.testing.assert_series_equal(bar_volatility(close, 100, VolProfile.flat(5)), bar_volatility(close, 100),
                                   check_exact=True)  # fmt: skip


def test_tod_sigma_scales_with_the_slot_factor():
    df = intraday(40, seed=4, u_shape=True)
    prof = VolProfile.fit(df.iloc[: 78 * 30], 5)
    sig = bar_volatility(df["close"], 100, prof)
    base = (np.log(df["close"]).diff() / prof.factor(df.index)).ewm(span=100, min_periods=100).std()
    np.testing.assert_allclose(sig.to_numpy(), (base * prof.factor(df.index)).to_numpy(), rtol=1e-15)
    assert prof.s.min() >= 0.25


def test_profile_sigma_is_causal_and_a_profile_fit_past_the_cut_is_caught():
    df = intraday(40, seed=5, u_shape=True)
    c = 78 * 32 + 17
    ok = VolProfile.fit(df.iloc[: 78 * 30], 5)  # fit strictly before the cut, as per fold
    for i, other in enumerate([random_walk_after(df, c, 1), df.iloc[: c + 1]]):
        assert_prefix_equal(bar_volatility(df["close"], 100, ok), bar_volatility(other["close"], 100, ok),
                            df.index[c], f"variant {i}")  # fmt: skip
    # mutation: a profile fit one bar past the cut makes σ before the cut depend on later bars
    leak = lambda d: bar_volatility(d["close"], 100, VolProfile.fit(d.iloc[: c + 2], 5))
    with pytest.raises(AssertionError):
        assert_prefix_equal(leak(df), leak(random_walk_after(df, c, 1)), df.index[c])


def test_hold_scale_covers_the_held_slots():
    idx = intraday(2).index
    flat = VolProfile.flat(5)
    np.testing.assert_allclose(hold_scale(idx[[0, 10]], flat, 12, False), np.sqrt(12))
    s = np.ones(78)
    s[0] = 9.0  # the gap slot
    prof = VolProfile(s, 5)
    # 15:20 (slot 70) holds slots 71..77 only, intraday; with overnight holds it wraps into 0..4 of the next session
    assert hold_scale(idx[[70]], prof, 12, False)[0] == pytest.approx(np.sqrt(7))
    assert hold_scale(idx[[70]], prof, 12, True)[0] == pytest.approx(np.sqrt(11 + 81))
    # an event at 09:30 (slot 0) holds slots 1..12: its width ignores the gap slot's s(0)
    assert hold_scale(idx[[0]], prof, 12, False)[0] == pytest.approx(np.sqrt(12))
    # an event on the last bar enters the next session's first bar
    assert hold_scale(idx[[77]], prof, 3, False)[0] == pytest.approx(np.sqrt(81 + 2))


def test_tod_widths_use_the_held_slots_and_flat_tod_widths_match_plain_ones_mid_session():
    df = intraday(30, seed=18, u_shape=True)
    cfg = CFG5.replace(VOL_PROFILE="tod")
    prof = VolProfile.fit(df.iloc[: 78 * 20], 5)
    ev = sample_events(df, cfg, profile=prof)
    base = (np.log(df["close"]).diff() / prof.factor(df.index)).ewm(span=100, min_periods=100).std()
    want = cfg.BARRIER_MULT * base.loc[ev.index].to_numpy() * hold_scale(ev.index, prof, cfg.VERTICAL_BARS, False)
    np.testing.assert_allclose(ev["width"].to_numpy(), want, rtol=1e-14)
    flat = sample_events(df, cfg, profile=VolProfile.flat(5))
    plain = sample_events(df, CFG5)
    assert flat.index.equals(plain.index)
    mid = np.array([m <= 65 for m in (flat.index.hour * 60 + flat.index.minute - 570) // 5])  # a full 12-bar hold
    np.testing.assert_allclose(flat["width"].to_numpy()[mid], plain["width"].to_numpy()[mid], rtol=1e-14)


def test_isom_counts_events_per_slot():
    idx = intraday(2)["close"].index
    ev = idx[[0, 1, 78, 79, 80, 155]]
    counts = isom_counts(ev, 5)
    assert counts[0] == 2 and counts[1] == 2 and counts[2] == 1 and counts[77] == 1 and counts.sum() == 6


# ── Directional change ───────────────────────────────────────────────────────


def test_dc_events_on_a_sawtooth_fire_at_the_confirmation_bar():
    """Log price rises 0.01 for 10 bars then falls 0.01 for 10, repeatedly; δ = 0.025. The first move of 0.03 from
    the start confirms an up run at bar 3; each peak (bar 10, 30, ...) is confirmed 3 bars later, each trough (20, 40)
    likewise. The overshoot (extreme − confirmation price, in δ units) is known only at the next confirmation."""
    leg = np.r_[np.arange(0, 10), np.arange(10, 0, -1)] * 0.01
    x = np.r_[np.tile(leg, 4), 0.0]
    idx = pd.date_range("2024-01-02 09:30", periods=len(x), freq="5min", tz=NY)
    close = pd.Series(np.exp(x), index=idx)
    ev, over = dc_events(close, pd.Series(0.025, index=idx))
    assert list(idx.get_indexer(ev)) == [3, 13, 23, 33, 43, 53, 63, 73]
    o = over.to_numpy()
    assert np.isnan(o[:13]).all()
    np.testing.assert_allclose(o[13:23], (0.10 - 0.03) / 0.025)  # up run: confirmed at 0.03, peak 0.10
    np.testing.assert_allclose(o[23:33], (0.00 - 0.07) / 0.025)  # down run: confirmed at 0.07, trough 0.00


def test_dc_events_and_overshoot_are_causal():
    df = intraday(30, seed=6)
    c = 78 * 20 + 40
    delta = lambda d: 2 * bar_volatility(d["close"], 100)
    ev_a, o_a = dc_events(df["close"], delta(df))
    for other in (random_walk_after(df, c, 2), df.iloc[: c + 1]):
        ev_b, o_b = dc_events(other["close"], delta(other))
        assert list(ev_a[ev_a <= df.index[c]]) == list(ev_b[ev_b <= df.index[c]])
        assert_prefix_equal(o_a, o_b, df.index[c])
    assert len(ev_a) > 100


def test_dc_sampler_uses_the_profile_and_skips_last_bars():
    df = intraday(30, seed=7, u_shape=True)
    cfg = CFG5.replace(EVENT_SAMPLER="dc", EVENT_PARAMS={"dc_mult": 2.0})
    plain = sample_events(df, cfg)
    prof = VolProfile.fit(df.iloc[: 78 * 20], 5)
    tod = sample_events(df, cfg, profile=prof)
    assert not plain.index.equals(tod.index)
    for ev in (plain, tod):
        assert not any(t == "15:55" for t in hhmm(ev.index))
    # tod: thresholds in the open slots scale up, so the share of events in the first hour drops
    first_hour = lambda e: np.mean(np.array(hhmm(e.index)) < "10:30")
    assert first_hour(tod) < first_hour(plain)


# ── Schedule sampler ─────────────────────────────────────────────────────────


def sched(times, days="all", gate=None, tf="5Min", **kw) -> RunConfig:
    return RunConfig.for_timeframe(tf, EVENT_SAMPLER="schedule",
                                   EVENT_PARAMS={"entry_times": times, "days": days, "gate": gate}, **kw)  # fmt: skip


def test_schedule_1530_decides_at_the_1525_close_and_fills_at_the_1530_open():
    df = intraday(3, seed=8)
    ev = schedule_events(df, sched(["15:30"]))
    assert hhmm(ev) == ["15:25"] * 3
    entry = df.index[df.index.get_indexer(ev) + 1]
    assert hhmm(entry) == ["15:30"] * 3 and (entry.normalize() == ev.normalize()).all()


def test_schedule_open_entries_decide_at_the_previous_close():
    df = intraday(3, seed=9)
    ev = schedule_events(df, sched(["09:30", "10:00"]))
    assert hhmm(ev) == ["09:55", "15:55", "09:55", "15:55", "09:55"]  # session 1 has no 09:30 entry (no prior close)


def test_schedule_early_close_maps_to_the_last_bar_and_a_hole_gives_no_event():
    early = session_5min("2024-07-03", close="13:00")
    regular = session_5min("2024-07-05").drop(pd.Timestamp("2024-07-05 15:30", tz=NY))
    full = session_5min("2024-07-08")
    df = pd.concat([session_5min("2024-07-02"), early, regular, full])
    ev = schedule_events(df, sched(["15:30"]))
    assert [str(t) for t in ev.strftime("%m-%d %H:%M")] == ["07-02 15:25", "07-03 12:50", "07-08 15:25"]


def test_schedule_days_use_only_events_known_at_the_decision():
    """2020-03-03: unscheduled FOMC cut at 10:00 (known from then). A 09:35 entry decides at 09:35 → not an FOMC day
    yet; a 15:30 entry decides at 15:25 → an FOMC day. 2024-01-31 (scheduled) is known all day; 2024-01-11 is CPI."""
    df = pd.concat([session_5min(d) for d in ("2020-03-02", "2020-03-03", "2020-03-04")])
    assert len(schedule_events(df, sched(["09:35"], days="fomc"))) == 0
    assert [str(t) for t in schedule_events(df, sched(["15:30"], days="fomc")).date] == ["2020-03-03"]
    assert len(schedule_events(df, sched(["09:35"], days="non_macro"))) == 3
    assert len(schedule_events(df, sched(["15:30"], days="non_macro"))) == 2
    df2 = pd.concat([session_5min(d) for d in ("2024-01-10", "2024-01-11", "2024-01-30", "2024-01-31")])
    assert [str(t) for t in schedule_events(df2, sched(["09:35"], days="fomc")).date] == ["2024-01-31"]
    assert [str(t) for t in schedule_events(df2, sched(["09:35"], days="cpi_nfp")).date] == ["2024-01-11"]
    with pytest.raises(ValueError, match="needs the cell's symbol"):
        schedule_events(df2, sched(["09:35"], days="earnings"))


def test_schedule_gate_is_evaluated_on_the_session_frame_at_the_event_bar():
    df = intraday(40, seed=10)
    cfg = sched(["15:30"], gate="open_to_now > 0.5 * sigma_day")
    ev = schedule_events(df, cfg)
    all_ev = schedule_events(df, sched(["15:30"]))
    frame = session_frame(df, cfg).loc[all_ev]
    want = all_ev[(frame["open_to_now"] > 0.5 * frame["sigma_day"]).to_numpy()]
    assert ev.equals(want) and 0 < len(ev) < len(all_ev)
    with pytest.raises(Exception, match="no_such_column"):
        schedule_events(df, sched(["15:30"], gate="no_such_column > 0"))


def test_schedule_events_are_causal():
    df = intraday(40, seed=11)
    cfg = sched(["10:00", "15:30"], gate="abs(open_to_now) > 0.3 * sigma_day")
    a = sample_events(df, cfg)
    for c in (78 * 30 + 6, 78 * 33 + 71):
        b = sample_events(random_walk_after(df, c, 3), cfg)
        assert_prefix_equal(a, b, df.index[c])


def test_daily_schedule_events_trade_every_next_open():
    from tests.test_features import synthetic_daily

    d = synthetic_daily(30)
    ev = sample_events(d, sched(["open"], tf="1Day"))
    assert ev.index.equals(d.index[:-1])


# ── Exit models ──────────────────────────────────────────────────────────────


def test_time_exit_close_moc_trade_by_hand_with_quotes_costs():
    """Scheduled 15:30 → MOC: decision at the 15:25 close, entry at the 15:30 open (regular bin 15:30: slippage +
    half-spread), exit at the 15:55 bar's close (closing auction proxy only); the label is the fill-to-fill return."""
    df = pd.concat([session_5min("2024-07-01"), session_5min("2024-07-02"), session_5min("2024-07-05")])
    i_in = df.index.get_loc(pd.Timestamp("2024-07-02 15:30", tz=NY))
    i_out = df.index.get_loc(pd.Timestamp("2024-07-02 15:55", tz=NY))
    df.iloc[i_in, df.columns.get_loc("open")] = 101.0
    df.iloc[i_out, df.columns.get_loc("close")] = 103.0
    cfg = sched(
        ["15:30"],
        COST_MODEL="quotes",
        SLIPPAGE_PCT=SLIP,
        EXIT_MODEL="time",
        EXIT_PARAMS={"exit_time": "close"},
        VOL_SPAN=20,
    )
    ev = sample_events(df, cfg)
    lab = triple_barrier_labels(df, ev, cfg)
    r = lab.loc[pd.Timestamp("2024-07-02 15:25", tz=NY)]
    assert (r["entry_pos"], r["exit_pos"], r["entry_px"], r["exit_px"]) == (i_in, i_out, 101.0, 103.0)
    assert r["ret"] == pytest.approx(103 / 101 - 1, rel=1e-15) and r["barrier"] == "vertical"
    sig = signals_at(df, [i_in - 1], [1], width=0.01)
    tr = run_backtest(df, sig, cfg, cost_data=const_quotes_table(["X"]))["Meta-filtered"][1].iloc[0]
    entry_fill = 101 * (1 + SLIP + bin_bp("15:30") * 1e-4)
    exit_fill = 103 * (1 - AUCTION["close_auction"] * 1e-4)
    assert tr["entry_fill"] == pytest.approx(entry_fill, rel=1e-14)
    assert tr["exit_fill"] == pytest.approx(exit_fill, rel=1e-14)
    assert tr["pnl_pct"] == pytest.approx(exit_fill / entry_fill - 1, rel=1e-12)


def test_time_exit_variants_by_hand():
    df = pd.concat([session_5min("2024-07-02"), session_5min("2024-07-03", close="13:00"), session_5min("2024-07-05")])
    df["open"] = np.arange(len(df)) + 100.0
    df["close"] = df["open"] + 0.5
    w = pd.Series(0.01, index=df.index)
    ts = lambda s: pd.Timestamp(s, tz=NY)
    ev = pd.DatetimeIndex([ts("2024-07-02 09:45"), ts("2024-07-02 15:50"), ts("2024-07-03 10:00")])
    kw = {"hold_overnight": False, "bar_minutes": 5}
    hh = time_exits(df, ev, w, exit_time="10:30", hold_bars=None, **kw)
    assert hhmm(hh["t1"]) == ["10:30", "10:30"]  # 07-02 15:50 enters 15:55 > 10:30: dropped; 07-03 has a 10:30 bar
    assert (
        hh["barrier"].tolist() == ["time", "time"] and (hh["exit_px"] == df["open"].iloc[hh["exit_pos"]].values).all()
    )
    late = time_exits(df, ev[2:], w, exit_time="14:00", hold_bars=None, **kw)  # 07-03 closes at 13:00 → MOC
    assert hhmm(late["t1"]) == ["12:55"] and late["barrier"].iloc[0] == "vertical"
    assert late["exit_px"].iloc[0] == df["close"].iloc[late["exit_pos"].iloc[0]]
    moo = time_exits(df, ev, w, exit_time="open", hold_bars=None, **kw)
    assert [str(t) for t in moo["t1"].dt.strftime("%m-%d %H:%M")] == ["07-03 09:30", "07-03 09:30", "07-05 09:30"]
    assert (moo["exit_px"] == df["open"].iloc[moo["exit_pos"]].values).all()
    hb = time_exits(df, ev, w, exit_time=None, hold_bars=4, **kw)
    assert hhmm(hb["t1"]) == ["10:05", "15:55", "10:20"] and (hb["barrier"] == "vertical").all()


def test_time_exit_close_needs_the_closing_bar_and_open_needs_the_opening_bar():
    s1 = session_5min("2024-07-02").drop(pd.Timestamp("2024-07-02 15:55", tz=NY))  # the MOC bar is missing
    s2 = session_5min("2024-07-05").drop(pd.Timestamp("2024-07-05 09:30", tz=NY))  # the MOO bar is missing
    df = pd.concat([s1, s2, session_5min("2024-07-08")])
    w = pd.Series(0.01, index=df.index)
    ev = pd.DatetimeIndex([pd.Timestamp("2024-07-02 10:00", tz=NY), pd.Timestamp("2024-07-05 10:00", tz=NY)])
    kw = {"hold_bars": None, "hold_overnight": False, "bar_minutes": 5}
    assert [str(t.date()) for t in time_exits(df, ev, w, exit_time="close", **kw).index] == ["2024-07-05"]
    assert [str(t.date()) for t in time_exits(df, ev, w, exit_time="open", **kw).index] == ["2024-07-05"]


def test_hysteresis_exit_on_a_synthetic_signal():
    df = session_5min("2024-07-02")
    df["open"] = np.arange(len(df)) + 100.0
    df["close"] = df["open"] + 0.25
    w = pd.Series(0.01, index=df.index)
    sig = pd.Series(1.0, index=df.index)
    sig.iloc[5:] = [0.5, 0.0, -0.1, -0.3, 0.4] + [1.0] * (len(df) - 10)  # ≤ −0.2 at bar 8
    ev = df.index[[3]]
    out = hysteresis_exits(df, ev, w, sig, pd.Series(1, index=ev), beta=0.2, max_bars=10, hold_overnight=False)
    r = out.iloc[0]
    assert (r["entry_pos"], r["exit_pos"], r["barrier"], r["exit_px"]) == (4, 9, "hysteresis", 109.0)  # open[8 + 1]
    # a short exits when the signal is >= +β: bar 4 (1.0) → fills at open[5]
    short = hysteresis_exits(df, ev, w, sig, pd.Series(-1, index=ev), beta=0.2, max_bars=10, hold_overnight=False)
    assert short["exit_pos"].iloc[0] == 5
    # no trigger before the max hold → close of the max_bars-th held bar
    calm = hysteresis_exits(df, ev, w, sig, pd.Series(1, index=ev), beta=0.5, max_bars=4, hold_overnight=False)
    assert (calm["exit_pos"].iloc[0], calm["barrier"].iloc[0], calm["exit_px"].iloc[0]) == (7, "vertical", 107.25)
    # side-less (labels): the side is the sign of the signal at the event
    s2 = sig.copy()
    s2.iloc[3] = -2.0
    lab = hysteresis_exits(df, ev, w, s2, None, beta=0.2, max_bars=10, hold_overnight=False)
    assert lab["exit_pos"].iloc[0] == 5  # treated as a short: bar 4 (1.0 >= 0.2)


def exit_cfgs():
    yield "triple_barrier", CFG5
    yield "time close", CFG5.replace(EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "close"})
    yield "time 11:00", CFG5.replace(EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "11:00"})
    yield "time open", CFG5.replace(EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "open"})
    yield "time hold", CFG5.replace(EXIT_MODEL="time", EXIT_PARAMS={"hold_bars": 6})
    yield "hysteresis", CFG5.replace(PRIMARY="sma_cross", EXIT_MODEL="hysteresis", EXIT_PARAMS={"beta": 0.5,
                                                                                               "max_bars": 12})  # fmt: skip


def labels_causal(label_fn, df: pd.DataFrame, c: int) -> None:
    """Outcomes resolved by bar c (exit_pos <= c) must not change when bars after c change."""
    a = label_fn(df)
    b = label_fn(random_walk_after(df, c, 4))
    done = a.index[a["exit_pos"] <= c]
    assert len(done) > 20
    cols = ["entry_pos", "exit_pos", "entry_px", "exit_px", "ret"]
    pd.testing.assert_frame_equal(a.loc[done, cols], b.loc[done, cols])


@pytest.mark.parametrize("name,cfg", list(exit_cfgs()))
def test_every_exit_model_is_causal(name, cfg):
    df = intraday(40, seed=12)
    ev = sample_events(df, CFG5)
    labels_causal(lambda d: exit_frame(d, ev.index, ev["width"], cfg), df, 78 * 30 + 50)


def test_the_label_causality_check_catches_a_one_bar_look_ahead_in_a_time_exit():
    df = intraday(40, seed=12)
    ev = sample_events(df, CFG5)
    cfg = CFG5.replace(EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "11:00"})

    def leaky(d):  # fills at the NEXT bar's open, but reports the scheduled bar as the exit
        out = exit_frame(d, ev.index, ev["width"], cfg)
        out["exit_px"] = d["open"].to_numpy()[np.minimum(out["exit_pos"].to_numpy() + 1, len(d) - 1)]
        out["ret"] = out["exit_px"] / out["entry_px"] - 1
        return out

    labels_causal(lambda d: exit_frame(d, ev.index, ev["width"], cfg), df, 78 * 30 + 18)  # c = the 11:00 bar
    with pytest.raises(AssertionError):
        labels_causal(leaky, df, 78 * 30 + 18)


def test_exit_phase_books_open_fills_at_the_open():
    barrier = np.array(["vertical", "time", "hysteresis", "upper", "upper", "lower"])
    px = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    open_ = np.array([1.0, 9.0, 9.0, 4.0, 9.0, 9.0])
    assert exit_phase(barrier, px, open_).tolist() == [2, 0, 0, 0, 1, 1]


def test_open_fill_time_exits_pay_the_open_costs_in_the_portfolio():
    df = pd.concat([session_5min(d) for d in ("2024-07-01", "2024-07-02", "2024-07-05")])
    q = const_quotes_table(["X"])
    i_ev = df.index.get_loc(pd.Timestamp("2024-07-01 10:00", tz=NY))
    for exit_time, out_ts, cost in (("10:30", "2024-07-01 10:30", SLIP + bin_bp("10:30") * 1e-4),
                                    ("open", "2024-07-02 09:30", AUCTION["open_auction"] * 1e-4)):  # fmt: skip
        cfg = CFG5.replace(
            COST_MODEL="quotes", SLIPPAGE_PCT=SLIP, EXIT_MODEL="time", EXIT_PARAMS={"exit_time": exit_time}
        )
        tr = run_backtest(df, signals_at(df, [i_ev], [1]), cfg, cost_data=q)["Meta-filtered"][1].iloc[0]
        assert df.index[tr["exit_b"]] == pd.Timestamp(out_ts, tz=NY) and tr["exit_reason"] == "time"
        assert tr["exit_fill"] == pytest.approx(100 * (1 - cost), rel=1e-14)


def test_single_and_average_backtests_agree_on_time_exits():
    from wfo.backtest import simulate_positions, simulate_trades

    df = intraday(10, seed=13)
    cfg = CFG5.replace(COST_MODEL="slippage", EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "11:00"})
    sig = signals_at(df, [78 * k + 3 for k in range(9)], [1, -1] * 4 + [1])
    trades = simulate_trades(df, sig, cfg)
    _, bets = simulate_positions(df, sig, cfg.replace(POSITION_MODE="average", SIZE_STEP=0))
    assert len(trades) == len(bets) == 9 and hhmm(trades["t1"]) == ["11:00"] * 9
    np.testing.assert_allclose(trades["pnl_pct"].to_numpy(), bets["pnl_pct"].to_numpy(), rtol=1e-12)


# ── Session frame ────────────────────────────────────────────────────────────


def test_session_frame_by_hand():
    df = pd.concat([session_5min("2024-07-01", price=100.0), session_5min("2024-07-02", price=101.0),
                    session_5min("2024-07-05", price=102.0)])  # fmt: skip
    df.iloc[78 * 2, df.columns.get_loc("open")] = 104.0  # 07-05 opens at 104 after a 101 close
    df.iloc[78 * 2 + 2, df.columns.get_loc("close")] = 106.0  # 09:40 close
    f = session_frame(df, CFG5)
    r = f.iloc[78 * 2 + 2]  # 07-05 09:40
    assert r["prev_cc"] == pytest.approx(np.log(101 / 100)) and r["prev_oc"] == 0.0
    assert r["open_to_now"] == pytest.approx(np.log(106 / 104)) and r["first30"] == r["open_to_now"]
    assert r["mins_to_close"] == 375.0
    late = f.iloc[78 * 2 + 30]  # after 10:00: first30 stays the 09:55-close return
    assert late["first30"] == pytest.approx(np.log(102 / 104)) and late["open_to_now"] == pytest.approx(
        np.log(102 / 104)
    )
    assert f["mins_to_close"].iloc[77] == 0.0 and np.isnan(f["prev_cc"].iloc[:156]).all()


def test_session_group_is_per_fold_and_causal_with_a_profile():
    from features.registry import REGISTRY

    df = intraday(40, seed=14, u_shape=True)
    grp = REGISTRY["session"].fn
    cfg = CFG5.replace(VOL_PROFILE="tod")
    state = grp.fit(df.iloc[: 78 * 25], cfg)
    assert state[0] is not None and len(state[1]) == 78
    a = grp.transform(df, state, cfg)
    assert a.notna().any().all() and a.columns[0].startswith("session__")
    c = 78 * 30 + 33
    for other in (random_walk_after(df, c, 5), df.iloc[: c + 1]):
        assert_prefix_equal(a, grp.transform(other, state, cfg), df.index[c])


# ── Config, cache, engine ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"VOL_PROFILE": "tod", "_tf": "1Day"}, "intraday"),
        ({"EVENT_SAMPLER": "dc", "EVENT_PARAMS": {"dc_mult": 0}}, "dc_mult"),
        ({"EVENT_SAMPLER": "cusum", "EVENT_PARAMS": {"dc_mult": 2}}, "unknown keys"),
        ({"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["15:32"]}}, "bar open"),
        ({"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": "15:30"}}, "list"),
        ({"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["15:30"], "days": "xmas"}}, "days"),
        ({"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["10:30"]}, "_tf": "1Day"}, "open"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {}}, "exactly one"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "close", "hold_bars": 3}}, "exactly one"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "09:30"}}, "after 09:30"),
        ({"EXIT_MODEL": "time", "EXIT_PARAMS": {"hold_bars": 0}}, "hold_bars"),
        ({"EXIT_MODEL": "hysteresis", "EXIT_PARAMS": {"beta": 0.5}}, "max_bars"),
        ({"EXIT_MODEL": "stop", "EXIT_PARAMS": {}}, "EXIT_MODEL"),
    ],
)
def test_runconfig_validates_the_u14_switches(kw, match):
    tf = kw.pop("_tf", "5Min")
    with pytest.raises(ValueError, match=match):
        RunConfig.for_timeframe(tf, **kw)


def test_new_switches_keep_static_feature_cache_keys():
    base = feature_cache.cfg_hash(CFG5)
    other = CFG5.replace(VOL_PROFILE="tod", EVENT_SAMPLER="dc", EXIT_MODEL="time", EXIT_PARAMS={"hold_bars": 3})
    assert feature_cache.cfg_hash(other) == base


def test_hysteresis_needs_a_rule_primary():
    cfg = CFG5.replace(EXIT_MODEL="hysteresis", EXIT_PARAMS={"beta": 0.5, "max_bars": 6})  # PRIMARY ml_xgb
    with pytest.raises(ValueError, match="rule primary"):
        engine.prepare(intraday(5, seed=15), cfg)


def small_cfg(**kw) -> RunConfig:
    base = {
        "PRIMARY": "sma_cross",
        "INITIAL_TRAIN": 26,
        "VAL": 6,
        "TEST": 4,
        "EMBARGO": 1,
        "MIN_TRAIN_EVENTS": 20,
        "MIN_VAL_EVENTS": 10,
        "COST_MODEL": "slippage",
        "FEATURE_GROUPS": ("wavelet_core", "session"),
    }
    return RunConfig.for_timeframe("5Min", **(base | kw))


def test_run_wfo_with_tod_fits_each_profile_on_train_before_the_embargo(monkeypatch):
    df = intraday(44, seed=16, u_shape=True)
    seen = []
    real = engine.fit_profile

    def spy(train_df, cfg):
        seen.append((train_df.index[0], train_df.index[-1]))
        return real(train_df, cfg)

    monkeypatch.setattr(engine, "fit_profile", spy)
    cfg = small_cfg(VOL_PROFILE="tod")
    sig = engine.run_wfo(df, cfg)
    folds = engine.wfo_folds(df.index, cfg)
    want = [(df.index[0], df.index[f.train_end - f.train_embargo - 1]) for f in folds]
    assert seen == want
    isom = sig.attrs["isom_folds"]
    assert set(isom) == {str(k) for k in range(1, len(folds) + 1)} and all(len(v["s"]) == 78 for v in isom.values())
    # the OOS widths are the fold's tod widths, not the plain ones
    plain = sample_events(df, cfg.replace(VOL_PROFILE="none"))
    common = sig.index.intersection(plain.index)
    assert not np.allclose(sig.loc[common, "width"], plain.loc[common, "width"])


def test_run_wfo_with_a_schedule_and_time_exits_end_to_end():
    df = intraday(44, seed=17)
    cfg = small_cfg(EVENT_SAMPLER="schedule", EVENT_PARAMS={"entry_times": ["10:00", "12:00", "14:00", "15:30"]},
                    EXIT_MODEL="time", EXIT_PARAMS={"exit_time": "close"}, MIN_TRAIN_EVENTS=20, MIN_VAL_EVENTS=10)  # fmt: skip
    sig = engine.run_wfo(df, cfg)
    assert set(hhmm(sig.index)) <= {"09:55", "11:55", "13:55", "15:25"}
    _, trades = run_backtest(df.loc[sig.index[0] :], sig, cfg)["Primary only"]
    assert len(trades) > 0 and set(hhmm(trades["t1"])) == {"15:55"}
