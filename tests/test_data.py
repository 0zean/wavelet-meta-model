"""Data layer tests — no network: Alpaca is replaced by FakeSource."""

import json

import numpy as np
import pandas as pd
import pytest

from data import alpaca_source, bars
from data.alpaca_source import BAR_COLUMNS, MissingCredentialsError, bars_from_raw, calendar_from_raw
from data.quality import quality_report
from data.sessions import resample_session, rth_filter
from data.store import CacheError, cache_paths, load_bars_cache, save_bars

NY = "America/New_York"

# Nov 2024: Thanksgiving (28th) closed, 29th closes 13:00; Jul 2024 is in EDT
CAL_ROWS = [
    {"date": d.strftime("%Y-%m-%d"), "open": "09:30", "close": "13:00" if d == pd.Timestamp("2024-11-29") else "16:00"}
    for d in pd.bdate_range("2010-01-01", "2026-12-31")
    if d != pd.Timestamp("2024-11-28")
]


class FakeSource:
    """Deterministic bars 04:00–20:00 NY on weekdays (incl. extended hours and the holiday)."""

    def __init__(self, factor: float = 1.0):
        self.factor = factor
        self.bar_calls: list[tuple] = []
        self.calendar_calls = 0

    def calendar(self, start, end):
        self.calendar_calls += 1
        return calendar_from_raw([r for r in CAL_ROWS if str(start) <= r["date"] <= str(end)])

    def bars(self, symbol, timeframe, start, end, adjustment):
        self.bar_calls.append((symbol, timeframe, start, end))
        days = pd.bdate_range(start.tz_localize(None).normalize(), end.tz_localize(None), inclusive="left")
        m = int(timeframe.removesuffix("Min"))
        offs = pd.timedelta_range("04:00:00", "19:55:00", freq=f"{m}min")
        idx = pd.DatetimeIndex([d + o for d in days for o in offs]).tz_localize(NY)
        idx = idx[(idx >= start) & (idx < end)]
        t = (idx.as_unit("ns").asi8 // 60_000_000_000).astype(float)  # minutes since epoch
        close = (100 + 5 * np.sin(t / 97.0) + (t % 7) * 0.01) * self.factor
        df = pd.DataFrame(
            {
                "open": close - 0.02,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 1000 + t % 13,
                "vwap": close,
                "trade_count": 10.0,
            },
            index=idx.tz_convert("UTC"),
        )
        return df[BAR_COLUMNS]


@pytest.fixture
def cal():
    return calendar_from_raw(CAL_ROWS)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bars, "_today_ny", lambda: pd.Timestamp("2025-01-10"))
    return tmp_path


# ── Source parsing ──────────────────────────────────────────────────────────


def test_bars_from_raw_is_utc_float():
    raw = [{"t": "2024-11-29T14:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100, "n": 3, "vw": 1.2}]
    df = bars_from_raw(raw)
    assert str(df.index.tz) == "UTC" and df.index.name is None
    assert list(df.columns) == BAR_COLUMNS and (df.dtypes == "float64").all()
    assert df.index[0] == pd.Timestamp("2024-11-29 09:30", tz=NY)


def test_missing_credentials_names_vars_without_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("API_KEY=supersecretkey123\n")
    monkeypatch.setattr(alpaca_source, "ENV_PATH", env)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(MissingCredentialsError) as e:
        alpaca_source._credentials()
    assert "SECRET_KEY" in str(e.value) and "supersecretkey123" not in str(e.value)
    import os

    assert "API_KEY" not in os.environ  # .env values are not exported


# ── Sessions ────────────────────────────────────────────────────────────────


def _day_bars(src, day, tf="5Min"):
    d = pd.Timestamp(day, tz=NY)
    return src.bars("X", tf, d, d + pd.Timedelta(days=1), "all")


def test_rth_filter_regular_half_day_and_dst(cal):
    src = FakeSource()
    reg = rth_filter(_day_bars(src, "2024-11-27"), cal, 5)
    assert len(reg) == 78 and reg.index[0].strftime("%H:%M") == "09:30" and reg.index[-1].strftime("%H:%M") == "15:55"
    half = rth_filter(_day_bars(src, "2024-11-29"), cal, 5)
    assert len(half) == 42 and half.index[-1].strftime("%H:%M") == "12:55"
    assert rth_filter(_day_bars(src, "2024-11-28"), cal, 5).empty  # holiday
    summer = rth_filter(_day_bars(src, "2024-07-10"), cal, 5)  # EDT (UTC-4)
    assert len(summer) == 78 and summer.index[0].strftime("%H:%M") == "09:30"
    # a bar must end by the close: a 30-min bar at 15:45 would not fit
    assert rth_filter(_day_bars(src, "2024-11-27", "15Min"), cal, 30).index[-1].strftime("%H:%M") == "15:30"


def test_resample_hour_anchored_at_open_with_stub(cal):
    src = FakeSource()
    five = rth_filter(pd.concat([_day_bars(src, "2024-11-27"), _day_bars(src, "2024-11-29")]), cal, 5)
    hour = resample_session(five, cal, "1Hour")
    d27 = hour[hour.index.day == 27]
    assert [t.strftime("%H:%M") for t in d27.index] == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    d29 = hour[hour.index.day == 29]
    assert [t.strftime("%H:%M") for t in d29.index] == ["09:30", "10:30", "11:30", "12:30"]
    # aggregation of the first bin
    first = five.loc["2024-11-27 09:30":"2024-11-27 10:25"]
    row = hour.iloc[0]
    assert row["open"] == first["open"].iloc[0] and row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max() and row["low"] == first["low"].min()
    assert row["volume"] == first["volume"].sum()
    assert np.isclose(row["vwap"], (first["vwap"] * first["volume"]).sum() / first["volume"].sum())
    stub = five.loc["2024-11-27 15:30":"2024-11-27 15:55"]
    assert d27.iloc[-1]["volume"] == stub["volume"].sum() and len(stub) == 6


# ── Store ───────────────────────────────────────────────────────────────────


def _sample(cal):
    return rth_filter(_day_bars(FakeSource(), "2024-11-27"), cal, 5)


def test_store_round_trip_exact(tmp_path, cal):
    df = _sample(cal)
    npz, js = cache_paths(tmp_path, "sip", "all", "5Min", "spy")
    save_bars(npz, js, df, {"symbol": "SPY", "tz": NY})
    back, meta = load_bars_cache(npz, js, expect={"symbol": "SPY"})
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    assert str(back.index.tz) == NY and meta["n_bars"] == len(df)
    with np.load(npz, allow_pickle=False) as z:
        assert z["ts"].dtype == np.int64 and set(z.files) == {"ts", *BAR_COLUMNS}


def test_store_rejects_tampering_missing_sidecar_and_mismatch(tmp_path, cal):
    df = _sample(cal)
    npz, js = cache_paths(tmp_path, "sip", "all", "5Min", "SPY")
    save_bars(npz, js, df, {"symbol": "SPY", "feed": "sip", "tz": NY})
    with pytest.raises(CacheError, match="feed"):
        load_bars_cache(npz, js, expect={"feed": "iex"})
    arrays = dict(np.load(npz))
    arrays["close"][3] += 0.01
    np.savez(npz, **arrays)
    with pytest.raises(CacheError, match="hash"):
        load_bars_cache(npz, js)
    js.unlink()
    with pytest.raises(CacheError, match="sidecar"):
        load_bars_cache(npz, js)


# ── load_bars ───────────────────────────────────────────────────────────────


def test_load_bars_caches_and_tops_up_without_duplicates(cache):
    src = FakeSource()
    a = bars.load_bars("spy", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=src)
    assert len(src.bar_calls) == 1
    assert (a.index.tz_localize(None).normalize() != pd.Timestamp("2024-11-28")).all()
    again = bars.load_bars("SPY", "5Min", "2024-11-26", "2024-12-02", cache_dir=cache, source=src)
    assert len(src.bar_calls) == 1  # served from cache
    pd.testing.assert_frame_equal(again, a.loc["2024-11-26":"2024-12-01"], check_freq=False)

    ext = bars.load_bars("SPY", "5Min", "2024-11-18", "2024-12-10", cache_dir=cache, source=src)
    assert len(src.bar_calls) == 3  # one prefix + one suffix fetch
    assert not ext.index.has_duplicates and ext.index.is_monotonic_increasing
    full = bars.load_bars("SPY", "5Min", "2024-11-18", "2024-12-10", cache_dir=cache / "fresh", source=FakeSource())
    pd.testing.assert_frame_equal(ext, full, check_freq=False)
    meta = json.loads(cache_paths(cache, "sip", "all", "5Min", "SPY")[1].read_text())
    assert (meta["coverage_start"], meta["coverage_end"]) == ("2024-11-18", "2024-12-10")


def test_load_bars_refetches_all_when_adjustment_changes(cache, capsys):
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-15", cache_dir=cache, source=src)
    src.factor = 0.5  # e.g. a 2:1 split re-adjusts all history
    out = bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-22", cache_dir=cache, source=src)
    assert "re-fetching full range" in capsys.readouterr().out
    fresh = bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-22", cache_dir=cache / "f", source=FakeSource(0.5))
    pd.testing.assert_frame_equal(out, fresh, check_freq=False)


def test_load_bars_clamps_to_today_and_resampled_not_cached(cache):
    src = FakeSource()
    df = bars.load_bars("SPY", "1Hour", "2025-01-06", "2025-02-01", cache_dir=cache, source=src)
    assert df.index[-1] < pd.Timestamp("2025-01-10", tz=NY)
    meta = json.loads(cache_paths(cache, "sip", "all", "5Min", "SPY")[1].read_text())
    assert meta["coverage_end"] == "2025-01-10"
    assert not cache_paths(cache, "sip", "all", "1Hour", "SPY")[0].exists()
    assert (df.index.strftime("%H:%M").isin(["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"])).all()


def test_load_bars_daily_is_rth_resampled(cache):
    df = bars.load_bars("SPY", "1Day", "2024-11-25", "2024-12-03", cache_dir=cache, source=FakeSource())
    assert pd.Timestamp("2024-11-28", tz=NY) not in df.index and len(df) == 5
    five = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=FakeSource())
    d29 = five.loc["2024-11-29"]  # half day: RTH only, closes 13:00
    row = df.loc[pd.Timestamp("2024-11-29", tz=NY)]
    assert row["open"] == d29["open"].iloc[0] and row["close"] == d29["close"].iloc[-1]
    assert row["volume"] == d29["volume"].sum()
    assert not cache_paths(cache, "sip", "all", "1Day", "SPY")[0].exists()


def test_incomplete_sessions_dropped_and_logged(cache, capsys):
    class Holey(FakeSource):
        def bars(self, symbol, timeframe, start, end, adjustment):
            df = super().bars(symbol, timeframe, start, end, adjustment)
            ny = df.index.tz_convert(NY)
            hole = (ny.normalize() == pd.Timestamp("2024-11-26", tz=NY)) & (ny.strftime("%H:%M") != "09:30")
            return df[~hole]  # only the 09:30 bar survives on the 26th

    kw = {"cache_dir": cache, "source": Holey()}
    five = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", **kw)
    assert "dropping 1 incomplete session" in capsys.readouterr().out
    assert five.loc["2024-11-26"].empty and len(five.loc["2024-11-25"]) == 78
    daily = bars.load_bars("SPY", "1Day", "2024-11-25", "2024-12-03", **kw)
    assert pd.Timestamp("2024-11-26", tz=NY) not in daily.index and len(daily) == 4
    raw = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", min_session_coverage=0, **kw)
    assert len(raw.loc["2024-11-26"]) == 1
    rep = quality_report(raw, calendar_from_raw(CAL_ROWS), "5Min")
    assert rep["sessions_incomplete"] == 1 and rep["incomplete_dates"] == "2024-11-26"


def test_load_bars_rejects_intraday_bounds_and_propagates_errors(cache):
    with pytest.raises(ValueError, match="whole days"):
        bars.load_bars("SPY", "5Min", "2024-11-25 10:00", "2024-12-03", cache_dir=cache, source=FakeSource())

    class Unauthorized(FakeSource):
        def bars(self, *a, **k):
            raise PermissionError("subscription does not permit querying recent SIP data")

    with pytest.raises(PermissionError):
        bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=Unauthorized())


def test_load_bars_raises_on_no_data(cache):
    class Empty(FakeSource):
        def bars(self, symbol, timeframe, start, end, adjustment):
            return super().bars(symbol, timeframe, start, end, adjustment).iloc[:0]

    with pytest.raises(ValueError, match="no 5Min bars"):
        bars.load_bars("ZZZZ", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=Empty())


# ── Quality ─────────────────────────────────────────────────────────────────


def test_quality_report_counts_missing_but_not_early_close(cache, cal):
    df = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=FakeSource())
    rep = quality_report(df, cal, "5Min")
    assert rep["missing_bars"] == 0 and rep["early_close_sessions"] == 1 and rep["ohlc_violations"] == 0
    rep2 = quality_report(df.drop(df.index[[5, 100]]), cal, "5Min")
    assert rep2["missing_bars"] == 2
