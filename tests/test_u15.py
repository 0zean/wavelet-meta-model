"""
U15: state and context feature groups (SPEC §15): the point-in-time Exo view and its enforcement, causality of every
new group under exo perturbation (bars perturbation: tests/test_features.py runs over every registered group), the
calendar encodings on the real event table, the GARCH fit, cross_asset's sector context and the runner's context.
"""

import numpy as np
import pandas as pd
import pytest

import experiments.runner as R
import features.state as S
from data.alpaca_source import NY_TZ
from data.events import event_series
from experiments.spec import expand
from features import cache as feature_cache
from features.context import exo_range, load_context
from features.exo_align import Exo, bar_stamps_utc
from features.feature_builder import FeatureSet
from features.registry import REGISTRY, check_group_output, context_needs
from tests.test_experiments import FakeSource, doc
from tests.test_features import group_frame, market_for, random_walk_after, synthetic_daily, synthetic_exo
from utils.config import RunConfig
from utils.data_loader import load_ohlcv

CFG5 = RunConfig.for_timeframe("5Min")
CFGD = RunConfig.for_timeframe("1Day")
EXO_GROUPS = sorted(g for g in REGISTRY if "exo" in REGISTRY[g].needs)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_ohlcv("data/data.csv").iloc[:3000]


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=NY_TZ).tz_convert("UTC")


def frame(dates, values, avail, **extra) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame({"value": values, **extra, "available_at": pd.DatetimeIndex(avail)}, index=idx)


def bars_on(*stamps) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(stamps))).tz_localize(NY_TZ)


def intraday_bars(days, times=("09:30", "09:55", "10:00", "13:55", "14:00", "15:55")) -> pd.DataFrame:
    idx = bars_on(*[f"{d} {t}" for d in days for t in times])
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)


# ── The Exo view: the point-in-time rule ──────────────────────────────────────


def test_asof_uses_only_rows_available_at_or_before_the_bar_stamp():
    exo = Exo({"cboe/VIX": frame(["2024-03-04", "2024-03-05"], [13.0, 14.0],
                                 [utc("2024-03-04 16:20"), utc("2024-03-05 16:20")])})  # fmt: skip
    idx = bars_on("2024-03-04 09:30", "2024-03-05 09:30", "2024-03-05 15:55", "2024-03-05 16:20", "2024-03-06 09:30")
    np.testing.assert_array_equal(exo.asof("cboe/VIX", idx), [np.nan, 13.0, 13.0, 14.0, 14.0])
    # a naive index is NY time
    np.testing.assert_array_equal(exo.asof("cboe/VIX", idx.tz_localize(None)), exo.asof("cboe/VIX", idx))


def test_asof_ties_take_the_latest_date_and_change_waits_for_its_later_row():
    # H.10: a week's daily dollar values all become public the next Monday 16:30
    week = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]
    avail = [utc("2024-03-11 16:30")] * 5
    exo = Exo({"fred/DTWEXBGS": frame(week, [100.0, 101.0, 102.0, 103.0, 110.0], avail)})
    idx = bars_on("2024-03-11 16:00", "2024-03-11 16:30")
    np.testing.assert_array_equal(exo.asof("fred/DTWEXBGS", idx), [np.nan, 110.0])
    np.testing.assert_allclose(exo.change("fred/DTWEXBGS", idx, 1), [np.nan, 7.0])
    np.testing.assert_allclose(exo.change("fred/DTWEXBGS", idx, 4, log=True), [np.nan, np.log(110 / 100)])


def test_ratio_is_built_on_common_dates_before_alignment():
    """VIX has rows on exchange holidays from 2022 (SPEC §12); VIX3M does not: the ratio must not pair them."""
    days = ["2024-05-24", "2024-05-27", "2024-05-28"]  # 05-27 Memorial Day: VIX only
    vix = frame(days, [12.0, 30.0, 13.0], [utc(f"{d} 16:20") for d in days])
    v3m = frame([days[0], days[2]], [14.0, 15.0], [utc(f"{days[0]} 16:20"), utc(f"{days[2]} 16:20")])
    exo = Exo({"cboe/VIX": vix, "cboe/VIX3M": v3m})
    idx = bars_on("2024-05-28 09:30", "2024-05-29 09:30")
    np.testing.assert_allclose(exo.ratio("cboe/VIX3M", "cboe/VIX", idx), [14 / 12, 15 / 13])
    np.testing.assert_array_equal(exo.asof("cboe/VIX", idx), [30.0, 13.0])  # the level does use the holiday row


def test_dated_rows_respect_visibility():
    """next / last / on-day: a row counts only once public (scheduled rows long before; unscheduled at their instant)."""
    f = frame(["2024-12-11", "2025-01-15", "2025-03-03"], [1.0, 1.0, 1.0],
              [utc("2024-01-01 00:00"), utc("2025-01-01 00:00"), utc("2025-03-03 10:00")])  # fmt: skip
    exo = Exo({"calendar/CPI": f})
    idx = bars_on("2024-12-10 10:00", "2024-12-20 10:00", "2025-01-02 10:00", "2025-03-03 09:55", "2025-03-03 10:00")
    nxt = exo.next_date("calendar/CPI", idx)
    assert [str(x) for x in nxt] == ["2024-12-11", "NaT", "2025-01-15", "NaT", "2025-03-03"]
    last = exo.last_date("calendar/CPI", idx)
    assert [str(x) for x in last] == ["NaT", "2024-12-11", "2024-12-11", "2025-01-15", "2025-03-03"]
    np.testing.assert_array_equal(exo.on_day("calendar/CPI", idx), [np.nan, np.nan, np.nan, np.nan, 1.0])


def test_a_group_sees_only_its_declared_series():
    exo = Exo(synthetic_exo(bars_on("2024-03-04 09:30")))
    view = exo.restrict(["cboe/VIX"])
    with pytest.raises(KeyError, match="not declared"):
        view.asof("fred/DGS10", bars_on("2024-03-05 09:30"))
    with pytest.raises(ValueError, match="not in the context"):
        exo.restrict(["cboe/NOPE"])
    with pytest.raises(ValueError, match="tz-aware"):
        Exo({"x/y": pd.DataFrame({"value": [1.0], "available_at": [pd.Timestamp("2024-01-01")]})})


def test_feature_set_wraps_exo_and_refuses_missing_series(df):
    ctx = {"exo": synthetic_exo(df.index)}
    fs = FeatureSet(CFG5, ["wavelet_core", "rates_credit", "vol_state"], context=ctx)
    assert isinstance(fs.group_context["rates_credit"]["exo"], Exo)
    assert fs.group_context["rates_credit"]["exo"].names == sorted(REGISTRY["rates_credit"].exo)
    with pytest.raises(ValueError, match="needs context"):
        FeatureSet(CFG5, ["wavelet_core", "vol_state"])
    partial = {"exo": {k: v for k, v in ctx["exo"].items() if k != "fred/BAA10Y"}}
    with pytest.raises(ValueError, match="BAA10Y"):
        FeatureSet(CFG5, ["wavelet_core", "rates_credit"], context=partial)


# ── Enforcement: a planted same-day value, and a leaky alignment the checker catches ──


def _vix_by_date(d: pd.DataFrame, exo: dict) -> pd.DataFrame:
    """A leaky alignment (the bug the point-in-time rule exists for): each bar takes its own date's VIX close."""
    vix = exo["cboe/VIX"]["value"]
    day = d.index.tz_localize(None).normalize() if d.index.tz is not None else d.index.normalize()
    return pd.DataFrame({"leak__vix": vix.reindex(day).to_numpy()}, index=d.index)


def perturb_exo_after(exo: dict, stamp_ns: int, seed: int, drop: bool = False) -> dict:
    """Every row public after `stamp_ns` (UTC ns) gets a random value (or is dropped); earlier rows unchanged."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in exo.items():
        late = pd.DatetimeIndex(f["available_at"]).as_unit("ns").asi8 > stamp_ns
        g = f.copy()
        if drop:
            g = g[~late]
        else:
            g.loc[late, "value"] = rng.uniform(0.5, 50, int(late.sum()))
        out[name] = g
    return out


def assert_exo_causal(fn, d: pd.DataFrame, exo: dict, cuts, label: str) -> None:
    """Rows <= c of fn(d, exo) are unchanged when every exo row public after bar c's stamp is perturbed or dropped."""
    base = fn(d, exo)
    stamps = bar_stamps_utc(d.index)
    for i, c in enumerate(cuts):
        for drop in (False, True):
            other = fn(d, perturb_exo_after(exo, stamps[c], seed=i, drop=drop))
            np.testing.assert_allclose(
                other.iloc[: c + 1].to_numpy(), base.iloc[: c + 1].to_numpy(), rtol=0, atol=1e-12, equal_nan=True,
                err_msg=f"{label}: rows <= {c} changed ({'dropped' if drop else 'perturbed'} exo rows)",
            )  # fmt: skip


def test_planted_same_day_vix_is_not_visible_before_its_available_at(df):
    exo = synthetic_exo(df.index)
    day = pd.Timestamp("2024-02-13")  # a fixture session
    vix = exo["cboe/VIX"].copy()
    vix.loc[day, "value"] = 999.0  # its close, public at 16:20 that day
    exo["cboe/VIX"] = vix
    out = group_frame("vol_state", df, CFG5, None, exo=exo)["vol_state__vix"]
    assert not (out.loc["2024-02-13"] == 999.0).any()  # every bar of the day closes by 16:00
    assert out.loc["2024-02-14 09:30"] == 999.0
    leak = _vix_by_date(df, exo)["leak__vix"]
    assert leak.loc["2024-02-13 09:30"] == 999.0  # what a by-date join would have done


def test_the_exo_checker_catches_a_same_day_leak(df):
    exo = synthetic_exo(df.index)
    with pytest.raises(AssertionError, match="rows <= "):
        assert_exo_causal(_vix_by_date, df, exo, [1600], "leak")


@pytest.mark.parametrize("name", EXO_GROUPS)
def test_exo_groups_are_causal_under_exo_perturbation(df, name):
    cuts = np.sort(np.random.default_rng(7).integers(1600, len(df) - 50, 3))
    exo = synthetic_exo(df.index)
    assert_exo_causal(lambda d, e: group_frame(name, d, CFG5, None, exo=e), df, exo, cuts, name)


@pytest.mark.parametrize("name", EXO_GROUPS)
def test_exo_groups_are_causal_under_exo_perturbation_daily(name):
    d = synthetic_daily(900)
    exo = synthetic_exo(d.index)
    assert_exo_causal(lambda x, e: group_frame(name, x, CFGD, None, fit_end=600, exo=e), d, exo, [700, 850], name)


def test_unscheduled_statement_flips_fomc_day_at_its_instant():
    """2020-03-03: the unscheduled cut was announced 10:00 ET (data/events.py); not known on earlier bars."""
    d = intraday_bars(["2020-03-02", "2020-03-03"])
    exo = {name: event_series("calendar", name.split("/")[1]) for name in S.CALENDAR_EXO}
    cal = group_frame("calendar_events", d, CFG5, None, exo=exo)
    day = cal.loc["2020-03-03"]
    assert list(day["calendar_events__fomc_day"]) == [0, 0, 1, 1, 1, 1]
    assert day.loc["2020-03-03 09:55", "calendar_events__since_fomc"] == 23  # sessions since 2020-01-29
    assert day.loc["2020-03-03 10:00", "calendar_events__since_fomc"] == 0
    assert day.loc["2020-03-03 10:00", "calendar_events__mins_since_release"] == 5  # bar closes 10:05


# ── calendar_events on the real event table ───────────────────────────────────


@pytest.fixture(scope="module")
def real_calendar():
    return {name: event_series("calendar", name.split("/")[1]) for name in S.CALENDAR_EXO}


def test_calendar_hand_cases(real_calendar):
    d = intraday_bars(["2024-01-31", "2024-11-27", "2024-12-20", "2025-01-02", "2025-01-15"])
    cal = group_frame("calendar_events", d, CFG5, None, exo=real_calendar).rename(columns=lambda c: c.split("__")[1])
    fomc = cal.loc["2024-01-31"]  # FOMC 14:00
    assert (fomc["to_fomc"] == 0).all() and (fomc["fomc_day"] == 1).all()
    assert list(fomc["mins_since_release"]) == [-265, -240, -235, 0, 5, 120]
    # Wed before Thanksgiving → NFP Fri 2024-12-06: Wed, Fri, Mon … Thu = 6 sessions (the holiday is skipped)
    assert (cal.loc["2024-11-27", "to_nfp"] == 6).all()
    assert (cal.loc["2024-11-27", "pre_holiday"] == 1).all()
    # late December: 2025's first CPI (01-15) is public from 2025-01-01 only → capped; since the 12-11 CPI: 7
    assert (cal.loc["2024-12-20", "to_cpi"] == S.TO_CAP).all()
    assert (cal.loc["2024-12-20", "since_cpi"] == 7).all()
    assert (cal.loc["2024-12-20", "opex_week"] == 1).all()  # OPEX Fri 2024-12-20
    # 2025-01-02 → 01-15: 01-02, 03, 06 … 10, 13, 14 = 9 (the 01-09 closure is public from 01-06 only)
    assert (cal.loc["2025-01-02", "to_cpi"] == 9).all()
    assert (cal.loc["2025-01-02", "tom"] == 1).all() and (cal.loc["2025-01-15", "tom"] == 0).all()
    cpi = cal.loc["2025-01-15"]  # CPI 08:30: 65 minutes at the 09:30 bar's close
    assert (cpi["cpi_nfp_day"] == 1).all() and cpi["mins_since_release"].iloc[0] == 65


def test_calendar_unknown_closure_is_not_counted_before_it_is_public(real_calendar):
    """The 2025-01-09 closure (PRE_HOLIDAY 01-08, public from Monday 01-06): counted from that day on only."""
    d = intraday_bars(["2025-01-03", "2025-01-06"], times=("09:30",))
    cal = group_frame("calendar_events", d, CFG5, None, exo=real_calendar)["calendar_events__to_cpi"]
    assert list(cal) == [8, 6]  # 01-03: 01-03 … 01-14 with 01-09 still a session = 8; 01-06: 01-06 … 01-14 − 01-09


def test_withdrawn_schedule_rows_are_visible_only_before_their_instant():
    f = frame(["2025-10-03", "2025-11-20"], [1.0, 1.0], [utc("2025-01-01 00:00"), utc("2025-11-20 00:00")],
              withdrawn_at=pd.DatetimeIndex([utc("2025-10-03 08:30"), pd.NaT]))  # fmt: skip
    exo = Exo({"calendar/NFP_SCHEDULE": f})
    idx = bars_on("2025-09-30 10:00", "2025-10-03 00:00", "2025-10-03 09:30", "2025-11-19 10:00", "2025-11-20 09:30")
    nxt = exo.next_date("calendar/NFP_SCHEDULE", idx)
    assert [str(x) for x in nxt] == ["2025-10-03", "2025-10-03", "NaT", "NaT", "2025-11-20"]
    with pytest.raises(ValueError, match="withdrawn"):
        exo.known_dates("calendar/NFP_SCHEDULE", idx)


@pytest.mark.parametrize(
    "day, column, expected",
    [
        ("2020-03-13", "to_fomc", 3),  # the 03-18 meeting was scheduled until the 03-15 statement replaced it
        ("2020-02-03", "to_fomc", 31),  # Presidents Day 02-17 closed
        ("2025-09-10", "to_nfp", 17),  # Oct 3 NFP, postponed by the 2025-10-01 shutdown
        ("2025-09-12", "to_cpi", 23),  # Oct 15 CPI, moved to Oct 24
        ("2025-10-06", "to_nfp", 24),  # after Oct 3 passed: the scheduled Nov 7
        ("2026-01-12", "to_nfp", 18),  # Feb 6 NFP (moved by the 2026-01-31 funding lapse); MLK 01-19 closed
        ("2026-01-02", "to_cpi", 7),  # Jan 13 CPI, held on its scheduled date, public from 2026-01-01
    ],
)
def test_to_star_uses_the_schedule_public_at_the_bar(real_calendar, day, column, expected):
    """U15 review SEVERE: the table holds held releases only; a cancelled / moved release must still count as
    scheduled until its scheduled instant (the cancellation is not public earlier in the data)."""
    d = intraday_bars([day], times=("10:00",))
    cal = group_frame("calendar_events", d, CFG5, None, exo=real_calendar)
    assert cal[f"calendar_events__{column}"].iloc[0] == expected


def test_withdrawn_dates_do_not_flag_event_days(real_calendar):
    d = intraday_bars(["2025-10-03", "2025-10-15", "2020-03-18"], times=("10:00",))
    cal = group_frame("calendar_events", d, CFG5, None, exo=real_calendar)
    assert (cal["calendar_events__cpi_nfp_day"] == 0).all() and (cal["calendar_events__fomc_day"] == 0).all()


def test_calendar_daily_has_no_minutes_column(real_calendar):
    d = synthetic_daily(300)
    d.index = pd.bdate_range("2024-01-02", periods=300, tz=NY_TZ)
    cal = group_frame("calendar_events", d, CFGD, None, exo=real_calendar)
    assert "calendar_events__mins_since_release" not in cal and cal.notna().all().all()


# ── vol_state / GARCH ─────────────────────────────────────────────────────────


def simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        z[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * z[t] ** 2 + beta * h
    return z


def test_garch_fit_recovers_parameters():
    z = simulate_garch(20_000, 1e-6, 0.08, 0.90)
    omega, alpha, beta, vbar = S.fit_garch(z)
    assert abs(alpha - 0.08) < 0.02 and abs(beta - 0.90) < 0.03
    assert omega == pytest.approx(vbar * (1 - alpha - beta))
    h = S._garch_h(z, omega, alpha, beta, vbar)
    assert len(h) == len(z) + 1 and h[1] == pytest.approx(omega + alpha * z[0] ** 2 + beta * vbar)


def test_garch_needs_enough_train_returns():
    with pytest.raises(RuntimeWarning, match="GARCH_MIN_OBS"):
        S.fit_garch(np.random.default_rng(0).normal(size=S.GARCH_MIN_OBS - 1))


def test_vol_state_fit_reads_only_its_train_slice(df):
    """Bars after the train slice (the embargo, val, test) cannot move the fitted state."""
    a = REGISTRY["vol_state"].fn.fit(df.iloc[:1500], CFG5)
    b = REGISTRY["vol_state"].fn.fit(random_walk_after(df, 1499, 5).iloc[:1500], CFG5)
    np.testing.assert_array_equal(a[0].s, b[0].s)
    assert a[1:] == b[1:]


def test_vol_state_values(df):
    exo = synthetic_exo(df.index)
    v = group_frame("vol_state", df, CFG5, None, exo=exo).rename(columns=lambda c: c.split("__")[1])
    idx = df.index
    e = Exo(exo)
    np.testing.assert_allclose(v["vix"], e.asof("cboe/VIX", idx))
    np.testing.assert_allclose(v["vix3m_vix"], e.ratio("cboe/VIX3M", "cboe/VIX", idx))
    # rv21: the 21 sessions before the bar's own, annualized, over VIX / 100
    closes = df["close"].groupby(df.index.normalize()).last()
    r = np.log(closes).diff()
    day = pd.Timestamp("2024-02-14")
    k = closes.index.get_loc(day)
    rv = r.iloc[k - 21 : k].std() * np.sqrt(252)
    np.testing.assert_allclose(v.loc["2024-02-14 11:00", "rv21_vix"], rv / (v.loc["2024-02-14 11:00", "vix"] / 100))
    assert v["rv21_vix"].isna().sum() > 0 and v["rv21_vix"].iloc[-1000:].notna().all()  # warm-up only
    assert (v["garch_sigma"] > 0).all() and abs(v["ret_std_garch"].std() - 1) < 0.5


# ── cross_asset ──────────────────────────────────────────────────────────────


def test_cross_asset_lags_and_sector_columns(df):
    market, sector = market_for(df), market_for(df, seed=11)
    plain = group_frame("cross_asset", df, CFG5, market)
    both = group_frame("cross_asset", df, CFG5, market, sector=sector)
    assert not any("sector" in c or "rel_strength" in c for c in plain)
    pd.testing.assert_frame_equal(both[plain.columns], plain)
    m = np.log(market["close"]).diff()
    np.testing.assert_allclose(plain["cross_asset__mkt_ret_lag3"], m.shift(3), equal_nan=True)
    s = np.log(sector["close"]).diff()
    np.testing.assert_allclose(both["cross_asset__sector_ret_lag6"], s.shift(6), equal_nan=True)
    h = CFG5.VERTICAL_BARS
    rs = np.log(df["close"]).diff(h) - np.log(sector["close"]).diff(h)
    np.testing.assert_allclose(both[f"cross_asset__rel_strength_{h}"], rs, equal_nan=True)


def test_cross_asset_with_sector_is_causal(df):
    market, sector = market_for(df), market_for(df, seed=11)
    base = group_frame("cross_asset", df, CFG5, market, sector=sector)
    c = 2000
    other = group_frame("cross_asset", random_walk_after(df, c, 1), CFG5, random_walk_after(market, c, 2),
                        sector=random_walk_after(sector, c, 3))  # fmt: skip
    np.testing.assert_allclose(other.iloc[: c + 1], base.iloc[: c + 1], rtol=0, atol=1e-12, equal_nan=True)


def test_sector_context_changes_the_cache_key(df):
    k = lambda ctx: feature_cache.cache_key("cross_asset", "AAPL", CFG5, df, ctx)
    m = market_for(df)
    assert k({"market": m}) != k({"market": m, "sector": market_for(df, seed=11)})
    exo = synthetic_exo(df.index)
    bumped = {**exo, "cboe/VIX": exo["cboe/VIX"].assign(value=exo["cboe/VIX"]["value"] + 1)}
    assert feature_cache.context_hash(exo) != feature_cache.context_hash(bumped)
    late = exo["cboe/VIX"].assign(available_at=exo["cboe/VIX"]["available_at"] + pd.Timedelta(hours=1))
    assert feature_cache.context_hash(exo) != feature_cache.context_hash({**exo, "cboe/VIX": late})


# ── Context loading and the runner ────────────────────────────────────────────


class ExoSource(FakeSource):
    """FakeSource plus exo series (synthetic, over each request's business days)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.exo_loads = []

    def exo(self, source, name, start, end, *, allow_holdout):
        self.exo_loads.append((f"{source}/{name}", pd.Timestamp(start), pd.Timestamp(end), allow_holdout))
        full = synthetic_exo(pd.bdate_range(start, end, inclusive="left", tz=NY_TZ))[f"{source}/{name}"]
        return full[(full.index >= start) & (full.index < end)]


def test_context_needs_and_exo_range():
    keys, names = context_needs(["wavelet_core", "cross_asset", "calendar_events"], "1Day")
    assert keys == {"market", "sector", "exo"} and names == set(S.CALENDAR_EXO)
    assert context_needs(["wavelet_core", "trend"], "5Min") == (set(), set())
    a, b = exo_range("cboe/VIX", "2020-01-02", "2020-06-01", False)
    assert (a, b) == (pd.Timestamp("2019-09-04"), pd.Timestamp("2020-06-01"))
    _, b = exo_range("calendar/FOMC", "2020-01-02", "2026-09-01", False)
    assert b == pd.Timestamp("2026-12-30")  # schedules (no outcomes) may extend past HOLDOUT_START (U15 review)


def test_load_context_loads_only_what_the_groups_read():
    src = ExoSource()
    kw = {"bars": src.bars, "exo": src.exo}
    assert load_context("AAPL", "1Day", "2014-01-01", "2015-01-01", ["wavelet_core", "trend"], **kw) == {}
    assert load_context("AAPL", "1Day", "2014-01-01", "2015-01-01", None, **kw) == {}
    ctx = load_context("AAPL", "1Day", "2014-01-01", "2015-01-01", ["wavelet_core", "cross_asset"], **kw)
    assert set(ctx) == {"market", "sector"} and [x[0] for x in src.loads] == ["SPY", "XLK"]
    ctx = load_context("XLE", "1Day", "2014-01-01", "2015-01-01", ["wavelet_core", "cross_asset", "rates_credit"],
                       **kw)  # fmt: skip
    assert set(ctx) == {"market", "exo"} and set(ctx["exo"]) == set(S.RATES_EXO)  # an ETF has no sector
    with pytest.raises(ValueError, match="other than the market"):  # degenerate beta 1 / residual 0 (U15 review)
        load_context("SPY", "1Day", "2014-01-01", "2015-01-01", ["wavelet_core", "cross_asset"], **kw)
    ctx = load_context("XLE", "1Day", "2025-06-02", "2026-09-28", ["wavelet_core", "calendar_events"], **kw)
    assert src.exo_loads[-1][2] > pd.Timestamp("2026-10-01") and src.exo_loads[-1][3]  # calendar past HOLDOUT_START


def test_runner_cell_with_state_groups_passes_context(tmp_path):
    """A 1Day cell on a stock with cross_asset (market + sector), calendar_events, rates_credit and vol_state."""
    from experiments import ledger as L

    groups = ["wavelet_core", "trend", "cross_asset", "calendar_events", "rates_credit", "vol_state"]
    cells = expand(doc(stage="A", grid={"symbols": ["AAPL"]}, feature_groups=groups))
    src = ExoSource()
    cfg = cells[0].config()
    data = R.load_cell_data(cells[0], cfg, src, False)
    ctx = data["context"]["AAPL"]
    assert set(ctx) == {"market", "sector", "exo"} and [x[0] for x in src.loads[1:3]] == ["SPY", "XLK"]
    assert R.signals_key("AAPL", cfg, data["bars"]["AAPL"], ctx) != R.signals_key("AAPL", cfg, data["bars"]["AAPL"])
    other = {**data, "context": {"AAPL": {**ctx, "sector": market_for(ctx["sector"], seed=3)}}}
    assert R.data_hash(other) != R.data_hash(data)
    rows = R.run(cells, ledger=L.Ledger(tmp_path / "ledger.jsonl"), root=tmp_path / "root", source=ExoSource(),
                 feature_cache_dir=tmp_path / "fc")  # fmt: skip
    assert rows[0]["status"] == "ok", rows[0].get("error")
    sig = pd.read_csv(next((tmp_path / "root" / "cells").glob("*/signals_AAPL.csv")))
    assert len(sig) > 0


def test_new_groups_pass_the_stationarity_guard_on_daily_bars():
    d = synthetic_daily(900)
    exo = synthetic_exo(d.index)
    for name in [*EXO_GROUPS, "cross_asset"]:
        f = group_frame(name, d, CFGD, market_for(d), fit_end=600, exo=exo, sector=market_for(d, seed=11))
        check_group_output(name, f, d, REGISTRY[name].level_check)
