"""
U13 exogenous data (SPEC §12): parsers, the available_at rules, the VX roll, the cache, the holdout guard, the
checked-in calendar and earnings tables, and the quotes sampler. No network: fake http callables and quote sources.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import events, exo, quotes
from data.alpaca_source import calendar_from_raw
from data.bars import HOLDOUT_START, HoldoutError
from data.store import CacheError

NY = "America/New_York"
ROOT = Path(__file__).resolve().parent.parent
CBOE_CSV = "DATE,OPEN,HIGH,LOW,CLOSE\n01/04/2016,22.48,23.36,20.67,20.70\n01/05/2016,20.75,21.06,19.25,19.34\n"
FRED_CSV = "observation_date,DGS10\n2024-07-03,4.36\n2024-07-04,\n2024-07-05,4.28\n"


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(scope="module")
def sessions() -> pd.DatetimeIndex:
    """The checked-in exchange calendar (data/cache/calendar.json), read without the refresh logic."""
    rows = json.loads((ROOT / "data" / "cache" / "calendar.json").read_text(encoding="utf-8"))["rows"]
    return calendar_from_raw(rows).index


# ── Parsers and available_at ─────────────────────────────────────────────────


def test_cboe_and_fred_parsers():
    c = exo.parse_cboe_index(CBOE_CSV)
    assert list(c.columns) == ["value", "open", "high", "low"] and c.index[0] == pd.Timestamp("2016-01-04")
    assert c["value"].tolist() == [20.70, 19.34] and c.loc["2016-01-05", "low"] == 19.25
    f = exo.parse_fred(FRED_CSV, "DGS10")
    assert f.index.strftime("%Y-%m-%d").tolist() == ["2024-07-03", "2024-07-05"]  # the holiday's empty value dropped
    with pytest.raises(ValueError, match="FRED columns"):
        exo.parse_fred(FRED_CSV, "DGS2")
    with pytest.raises(ValueError, match="CBOE columns"):
        exo.parse_cboe_index("DATE,CLOSE\n01/04/2016,1\n")


def test_available_at_rules():
    d = pd.DatetimeIndex(["2024-01-05", "2024-07-05"])
    assert list(exo.available_at("cboe", "VIX", d)) == [utc("2024-01-05 21:20"), utc("2024-07-05 20:20")]
    # VIXCLS: next federal business day 09:00 ET (Fri → Mon; Wed 07-03 → Fri 07-05 past Independence Day)
    v = exo.available_at("fred", "VIXCLS", pd.DatetimeIndex(["2024-07-05", "2024-07-03"]))
    assert list(v) == [utc("2024-07-08 13:00"), utc("2024-07-05 13:00")]
    assert exo.available_at("fred", "DGS10", pd.DatetimeIndex(["2024-07-03"]))[0] == utc("2024-07-05 20:30")
    # DTWEXBGS (H.10 weekly): the Monday after the observation's week, 16:30 ET; MLK day 2024-01-15 → Tuesday
    w = exo.available_at("fred", "DTWEXBGS", pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-10"]))
    assert list(w) == [utc("2024-01-08 21:30"), utc("2024-01-16 21:30"), utc("2024-01-16 21:30")]
    # no daily value is usable before its own session's close
    for src, name in (("cboe", "VIX"), ("cboe", "VX1"), *[("fred", n) for n in exo.FRED_SERIES]):
        days = pd.bdate_range("2016-01-04", "2026-09-30")
        close = days.tz_localize(NY) + pd.Timedelta(hours=16)
        assert (exo.available_at(src, name, days) > close.tz_convert("UTC")).all(), name


def test_vx_continuous_rolls_after_the_expiry_row():
    e1, e2, e3 = (pd.Timestamp(x) for x in ("2024-01-17", "2024-02-14", "2024-03-20"))
    days = pd.DatetimeIndex(["2024-01-16", "2024-01-17", "2024-01-18"])
    c = {e1: pd.Series([13.0, 13.5], index=days[:2]), e2: pd.Series([14.0, 14.2, 14.4], index=days),
         e3: pd.Series([15.0, np.nan, 15.4], index=days).dropna()}  # fmt: skip
    vx = exo.vx_continuous(c)
    assert vx["VX1"].tolist() == [13.0, 13.5, 14.4]  # the expiry row (01-17) is still the front month
    assert vx["VX2"].iloc[[0, 2]].tolist() == [14.0, 15.4] and vx["VX2"].iloc[1] == 14.2


def test_cfe_contract_parser_drops_unsettled_rows():
    text = ("Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n"
            "2015-04-20,F (Jan 2016),0,18.95,0,0,0,0,0,0,0\n2016-01-20,F (Jan 2016),25.8,29.6,25.65,28.3,27.4,1.55,4518,0,0\n")  # fmt: skip
    s = exo.parse_cfe_contract(text)
    assert s.to_dict() == {pd.Timestamp("2016-01-20"): 27.4}


# ── Cache, top-ups, holdout ──────────────────────────────────────────────────


class FakeHttp:
    def __init__(self, text: str):
        self.text, self.calls = text, 0

    def __call__(self, url, headers=None):
        self.calls += 1
        return self.text


def test_load_series_caches_tops_up_logs_revisions_and_verifies(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exo, "_today_ny", lambda: pd.Timestamp("2024-07-08"))
    http = FakeHttp(FRED_CSV)
    a = exo.load_series("fred", "DGS10", "2024-07-01", "2024-07-08", cache_dir=tmp_path, http=http)
    assert http.calls == 1 and a["value"].tolist() == [4.36, 4.28] and str(a["available_at"].dt.tz) == "UTC"
    b = exo.load_series("fred", "DGS10", "2024-07-01", "2024-07-06", cache_dir=tmp_path, http=http)
    assert http.calls == 1  # served from the cache (coverage_end = the fetch day)
    pd.testing.assert_frame_equal(a, b)
    monkeypatch.setattr(exo, "_today_ny", lambda: pd.Timestamp("2024-07-10"))
    http.text = FRED_CSV.replace("4.36", "4.37") + "2024-07-08,4.30\n"
    c = exo.load_series("fred", "DGS10", "2024-07-01", "2024-07-10", cache_dir=tmp_path, http=http)
    assert http.calls == 2 and c["value"].tolist() == [4.37, 4.28, 4.30]
    assert "1 cached observation(s) revised" in capsys.readouterr().out
    _, js = exo.cache_paths(tmp_path, "fred", "DGS10")
    meta = json.loads(js.read_text())
    meta["sha256"] = "0" * 64
    js.write_text(json.dumps(meta))
    with pytest.raises(CacheError, match="hash"):
        exo.load_series("fred", "DGS10", "2024-07-01", "2024-07-10", cache_dir=tmp_path, http=http)


def test_load_series_holdout_guard_and_unknown_series(tmp_path):
    end = (HOLDOUT_START + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with pytest.raises(HoldoutError, match="HOLDOUT_START"):
        exo.load_series("cboe", "VIX", "2016-01-01", end, cache_dir=tmp_path, http=FakeHttp(CBOE_CSV))
    out = exo.load_series("cboe", "VIX", "2016-01-01", end, cache_dir=tmp_path, http=FakeHttp(CBOE_CSV),
                          allow_holdout=True)  # fmt: skip
    assert len(out) == 2
    with pytest.raises(ValueError, match="unknown exogenous series"):
        exo.load_series("fred", "SP500", "2016-01-01", "2016-02-01", cache_dir=tmp_path, http=FakeHttp(""))


def test_vx_fetch_caches_expired_contracts(tmp_path, monkeypatch):
    monkeypatch.setattr(exo, "_today_ny", lambda: pd.Timestamp("2024-02-01"))
    listing = {"2024": [{"expire_date": d, "duration_type": "M", "path": f"x/VX/VX_{d}.csv"}
                        for d in ("2024-01-17", "2024-02-14", "2024-03-20")]
               + [{"expire_date": "2024-01-24", "duration_type": "W", "path": "x/VX/VX_2024-01-24.csv"}]}  # fmt: skip
    head = "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n"
    files = {f"VX_{d}.csv": head + "".join(f"{t},F,0,0,0,0,{v},0,0,0,0\n" for t, v in rows) for d, rows in {
        "2024-01-17": [("2024-01-16", 13.0), ("2024-01-17", 13.5)],
        "2024-02-14": [("2024-01-16", 14.0), ("2024-01-17", 14.2), ("2024-01-18", 14.4)],
        "2024-03-20": [("2024-01-16", 15.0), ("2024-01-17", 15.1), ("2024-01-18", 15.4)]}.items()}  # fmt: skip
    seen = []

    def http(url, headers=None):
        seen.append(url.rsplit("/", 1)[-1])
        return json.dumps(listing) if url == exo.CFE_LIST_URL else files[url.rsplit("/", 1)[-1]]

    vx = exo._fetch_vx(tmp_path, http)
    assert vx["VX1"].tolist() == [13.0, 13.5, 14.4] and "VX_2024-01-24.csv" not in seen  # weeklies ignored
    seen.clear()
    exo._fetch_vx(tmp_path, http)
    assert "VX_2024-01-17.csv" not in seen and "VX_2024-02-14.csv" in seen  # expired: from the raw cache


# ── Calendar and earnings tables (checked in) ────────────────────────────────


def test_events_table_integrity(sessions):
    ev = events.read_events()
    ev["d"] = pd.to_datetime(ev["date"])
    fomc = ev[ev["kind"] == "FOMC"]
    sched = fomc[fomc["scheduled"] == 1]
    per_year = sched.groupby(sched["d"].dt.year).size()
    assert per_year.index.tolist() == list(range(2016, 2028))
    assert (per_year.drop(2020) == 8).all() and per_year[2020] == 7  # 2020-03-17/18 cancelled
    assert fomc.loc[fomc["scheduled"] == 0, "date"].tolist() == ["2020-03-03", "2020-03-15"]
    assert (sched["event_time"] == "14:00").all()
    # every scheduled event day inside the calendar is an exchange session, except BLS releases on Good Friday
    inside = ev[(ev["scheduled"] == 1) & (ev["d"] <= sessions[-1])]
    off = inside[~inside["d"].isin(sessions)]
    good_friday = off["d"].map(lambda d: d.dayofweek == 4 and (d + pd.offsets.BDay(1)) in set(sessions)
                               and d.month in (3, 4))  # fmt: skip
    assert set(off["kind"]) <= {"CPI", "NFP"} and good_friday.all(), off[["kind", "date"]].to_string()
    assert len(off) == 5
    for kind in ("CPI", "NFP"):
        k = ev[ev["kind"] == kind]
        n = k.groupby(k["d"].dt.year).size()
        assert (n.drop(2025) == 12).all() and n[2025] == 11, n  # 2025: the shutdown cancelled / merged one release
        assert (k["event_time"] == "08:30").all()
    assert (pd.to_datetime(ev["known_from"]) <= ev["d"]).all()
    opex = ev[ev["kind"] == "OPEX"]["d"]
    assert opex.dt.day.between(14, 21).all() and opex.dt.dayofweek.isin([3, 4]).all()
    assert ev[ev["kind"] == "OPEX"].set_index("date").loc["2019-04-18", "value"] == 1  # Good Friday → Thursday
    tom = ev[ev["kind"] == "TOM"]
    assert sorted(tom["value"].unique()) == [-1, 1, 2, 3] and (tom.groupby("value").size().nunique() == 1)
    pre = ev[ev["kind"] == "PRE_HOLIDAY"]["d"]
    assert (~(pre + pd.offsets.BDay(1)).isin(sessions)).all()
    known = ev.set_index(["kind", "date"])["known_from"]
    assert known[("PRE_HOLIDAY", "2018-12-04")] == "2018-12-03" and known[("CPI", "2025-10-24")] == "2025-10-24"


def test_event_series_point_in_time():
    f = exo.load_series("calendar", "FOMC", "2020-01-01", "2021-01-01")
    jan = f.loc["2020-01-29"]
    assert jan["available_at"] == pd.Timestamp("2020-01-01", tz=NY).tz_convert("UTC")
    assert jan["event_at"] == pd.Timestamp("2020-01-29 14:00", tz=NY).tz_convert("UTC")
    sunday = f.loc["2020-03-15"]
    assert sunday["available_at"] == sunday["event_at"] == pd.Timestamp("2020-03-15 17:00", tz=NY).tz_convert("UTC")
    tom = exo.load_series("calendar", "TOM", "2024-01-01", "2024-02-10")
    assert tom["value"].tolist() == [1, 2, 3, -1, 1, 2, 3] and tom["event_at"].isna().all()
    with pytest.raises(ValueError, match="unknown calendar kind"):
        exo.load_series("calendar", "ECB", "2024-01-01", "2024-02-01")


def test_earnings_table_and_series(sessions):
    ea = events.read_earnings()
    n = ea.groupby("symbol").size()
    assert set(n.index) == set(events.EARNINGS_CIK) and (n >= 43).all()
    assert (pd.to_datetime(ea["date"]).isin(sessions)).all()
    assert (ea.loc[ea["symbol"].isin(events.LATE_FILERS), "timing"] == "bmo").all()
    for sym, conv in events.EARNINGS_CONVENTION.items():
        assert (ea.loc[ea["symbol"] == sym, "timing"] == conv).mean() > 0.85, sym  # XOM: 42 of 47
    jpm = exo.load_series("earnings", "JPM", "2024-01-01", "2025-01-01")
    assert (jpm["timing"] == -1).all() and (jpm["available_at"] < jpm["event_at"]).all()
    nvda = exo.load_series("earnings", "NVDA", "2019-01-01", "2019-02-01")
    assert nvda["timing"].tolist() == [-1] and nvda["available_at"].iloc[0] == nvda["event_at"].iloc[0]  # surprise


def test_earnings_classification():
    def c(sym, t):
        return events.classify_earnings(sym, pd.Timestamp(t, tz=NY).tz_convert("UTC"))

    assert c("JPM", "2024-04-12 10:30") == ("2024-04-12", "bmo", "convention", "2024-04-12")
    assert c("AAPL", "2024-05-02 20:30") == ("2024-05-02", "amc", "acceptance", "2024-05-02")
    assert c("XOM", "2018-02-09 15:24") == ("2018-02-09", "dmh", "acceptance", "")
    assert c("NVDA", "2019-01-28 09:04") == ("2019-01-28", "bmo", "acceptance", "")


def test_fomc_and_bls_parsers():
    hist = "".join(f"<h5 class='x'>{t}</h5>" for t in ("Jan/Feb 31-1 Meeting - 2017", "March 14-15 Meeting - 2017",
                                                      "October 4 (unscheduled) - 2017", "March 19 (notation vote) - 2017",
                                                      "March 17-18 (cancelled) Meeting - 2017"))  # fmt: skip
    sched, unsched = events.parse_fomc_historical(hist, 2017)
    assert sched == [pd.Timestamp("2017-02-01"), pd.Timestamp("2017-03-15")]
    assert unsched == [pd.Timestamp("2017-10-04")]
    cal = ('<h4><a id="1">2025 FOMC Meetings</a></h4>'
           '<div class="fomc-meeting__month c"><strong>Apr/May</strong></div><div class="fomc-meeting__date c">30-1</div>'
           '<div class="fomc-meeting__month c"><strong>August</strong></div><div class="fomc-meeting__date c">22 (notation vote)</div>'
           '<div class="fomc-meeting__month c"><strong>December</strong></div><div class="fomc-meeting__date c">9-10*</div>')  # fmt: skip
    assert events.parse_fomc_calendar(cal) == {2025: [pd.Timestamp("2025-05-01"), pd.Timestamp("2025-12-10")]}
    arch = '<a href="/news.release/archives/cpi_10242025.htm">x</a><a href="/news.release/empsit_2027.htm">y</a>'
    assert events.parse_bls_archive(arch, "cpi") == [pd.Timestamp("2025-10-24")]
    assert events.parse_bls_archive(arch, "empsit") == []
    tab = "<td>Sept. 11, 2026</td><td>Feb. 13, 2026</td><td>May 12, 2026</td><td>September 2026</td>"
    assert events.parse_bls_schedule(tab) == [pd.Timestamp(d) for d in ("2026-02-13", "2026-05-12", "2026-09-11")]


# ── Quotes sampler ───────────────────────────────────────────────────────────


def test_sample_days_avoid_the_turn_of_month_and_expiry(sessions):
    days = quotes.sample_days(sessions, "2016Q1", "2026Q3")
    assert len(days) == 43 * 5 and days[:5] == list(pd.DatetimeIndex(["2016-02-08", "2016-02-09", "2016-02-10",
                                                                       "2016-02-11", "2016-02-12"]))  # fmt: skip
    d = pd.DatetimeIndex(days)
    assert d.month.isin([2, 5, 8, 11]).all() and (d.day <= 14).all()
    first5 = {m: s[:5] for m, s in pd.Series(sessions, index=sessions.to_period("M")).groupby(level=0)}
    assert not any(x in set(first5[x.to_period("M")]) for x in d)
    third_fri = d + pd.offsets.WeekOfMonth(week=2, weekday=4) * 0
    assert not (d.dayofweek == 4)[(d.day >= 15)].any() and len(third_fri) == len(d)


def test_sample_marks_full_and_early_close():
    o = pd.Timestamp("2024-07-02 09:30", tz=NY)
    m = quotes.sample_marks(o, o.replace(hour=16, minute=0))
    labels = [lab for lab, _ in m]
    assert (
        len(m) == 79 and labels[:3] == ["open_auction", "09:35", "09:40"] and labels[-2:] == ["15:55", "close_auction"]
    )
    assert m[0][1].strftime("%H:%M") == "09:31" and m[-1][1].strftime("%H:%M") == "15:59"
    early = quotes.sample_marks(o, o.replace(hour=13, minute=0))
    assert len(early) == 43 and early[-2][0] == "12:55" and early[-1][1].strftime("%H:%M") == "12:59"


class FakeQuotes:
    """A has a quote within 5 s of every mark, B only within 60 s, C never."""

    def __init__(self):
        self.calls = []

    def quotes(self, symbols, start, end):
        w = (end - start).total_seconds()
        self.calls.append((tuple(symbols), w))
        out = {}
        if "A" in symbols:
            out["A"] = [{"t": (end - pd.Timedelta(seconds=2)).isoformat(), "bp": 99.99, "ap": 100.01, "bs": 1, "as": 2}]
        if "B" in symbols and w >= 60:
            out["B"] = [{"t": (end - pd.Timedelta(seconds=30)).isoformat(), "bp": 50.0, "ap": 50.1, "bs": 3, "as": 4}]
        return out


def test_fetch_session_falls_back_and_marks_missing(tmp_path):
    o = pd.Timestamp("2024-07-02 09:30", tz=NY)
    marks = quotes.sample_marks(o, o + pd.Timedelta(minutes=15))[:2]
    src = FakeQuotes()
    df = quotes.fetch_session(src, ["A", "B", "C"], marks)
    assert src.calls[:3] == [(("A", "B", "C"), 5.0), (("B", "C"), 60.0), (("C",), 900.0)]
    a, b, c = (df[df["symbol"] == s].iloc[0] for s in "ABC")
    assert a["window_s"] == 5 and b["window_s"] == 60 and np.isnan(c["bid"]) and pd.isna(c["quote_ts"])
    assert a["quote_ts"] < a["mark"]
    day = pd.Timestamp("2024-07-02")
    quotes.save_samples(tmp_path, day, df)
    back = quotes.load_samples(tmp_path, day)
    pd.testing.assert_frame_equal(back[df.columns.tolist()].reset_index(drop=True), df.reset_index(drop=True),
                                  check_dtype=False)  # fmt: skip
    _, js = quotes._paths(tmp_path, day)
    meta = json.loads(js.read_text())
    meta["sha256"] = "1" * 64
    js.write_text(json.dumps(meta))
    with pytest.raises(CacheError, match="hash"):
        quotes.load_samples(tmp_path, day)


def test_table_medians_bins_and_invalid_quotes():
    assert [quotes.tod_bin(x) for x in ("09:35", "09:40", "09:45", "12:59", "15:55")] == [
        "09:30", "09:30", "09:45", "12:45", "15:45"]  # fmt: skip
    mk = pd.Timestamp("2024-02-08 15:00", tz="UTC")
    rows = [("X", "09:35", 100.0, 100.02), ("X", "09:40", 100.0, 100.04), ("X", "10:05", 100.0, 100.01),
            ("X", "open_auction", 100.0, 100.10), ("X", "close_auction", 100.0, 99.0),  # crossed: dropped
            ("X", "10:10", 0.0, 100.0)]  # one-sided: dropped  # fmt: skip
    s = pd.DataFrame(rows, columns=["symbol", "label", "bid", "ask"]).assign(mark=mk)
    t = quotes.build_table(s).set_index("bin")
    hs = lambda b, a: (a - b) / (a + b) * 1e4
    assert t.loc["09:30", "half_spread_bp"] == pytest.approx(np.median([hs(100, 100.02), hs(100, 100.04)]))
    assert t.loc["09:30", "n"] == 2 and t.loc["10:00", "n"] == 1 and "close_auction" not in t.index
    assert t.loc["open_auction", "half_spread_bp"] == pytest.approx(hs(100, 100.10))
    assert t.loc["day", "n"] == 3  # regular marks only, invalid quotes dropped


# ── Option chain collector ───────────────────────────────────────────────────


def test_option_chain_frame_and_cache_round_trip(tmp_path):
    from data import options

    snaps = {
        "SPY261016P00660000": {"latestQuote": {"bp": 1.0, "ap": 1.1, "t": "2026-10-08T20:14:59.2Z"},
                               "greeks": {"delta": -0.4, "gamma": 0.01, "theta": -0.2, "vega": 0.3},
                               "impliedVolatility": 0.18},
        "SPY261016C00655000": {"latestQuote": {"bp": 9.0, "ap": 9.3, "t": "2026-10-08T20:14:58Z"}},  # no IV / greeks
        "SPY261016C00999000": {"latestQuote": {}},  # no metadata below: dropped
    }  # fmt: skip
    contracts = [
        {"symbol": "SPY261016P00660000", "type": "put", "expiration_date": "2026-10-16", "strike_price": "660",
         "open_interest": "1500", "open_interest_date": "2026-10-07"},
        {"symbol": "SPY261016C00655000", "type": "call", "expiration_date": "2026-10-16", "strike_price": "655",
         "open_interest": None},
    ]  # fmt: skip
    df = options.chain_frame(snaps, contracts)
    assert df["contract"].tolist() == ["SPY261016C00655000", "SPY261016P00660000"]
    put = df.iloc[1]
    assert put["iv"] == 0.18 and put["gamma"] == 0.01 and put["open_interest"] == 1500 and put["strike"] == 660
    assert np.isnan(df.iloc[0]["iv"]) and np.isnan(df.iloc[0]["open_interest"])
    day = pd.Timestamp("2026-10-08")
    options.save_chain(tmp_path, "SPY", day, df, "2026-10-07")
    back = options.load_chain(tmp_path, "SPY", day)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
