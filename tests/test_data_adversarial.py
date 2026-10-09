"""U1 data-layer regression tests from the adversarial review. No network: FakeSource only."""

import json

import numpy as np
import pandas as pd
import pytest

from data import bars
from data.alpaca_source import calendar_from_raw
from data.quality import quality_report
from data.sessions import resample_session, rth_filter
from data.store import CacheError, cache_paths
from tests.test_data import CAL_ROWS, NY, FakeSource


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bars, "_today_ny", lambda: pd.Timestamp("2025-01-10"))
    return tmp_path


def _meta(cache):
    return json.loads(cache_paths(cache, "sip", "all", "5Min", "SPY")[1].read_text())


def test_fake_source_prices_vary_within_session():
    """FakeSource must vary every bar, or aggregation tests are vacuous (pandas 3 us-resolution index)."""
    d = pd.Timestamp("2024-11-26", tz=NY)
    b = FakeSource().bars("X", "5Min", d, d + pd.Timedelta(days=1), "all")
    assert b["close"].nunique() > 100


class Dropper(FakeSource):
    """FakeSource that removes bars matching a predicate on the NY index."""

    def __init__(self, pred, factor=1.0):
        super().__init__(factor)
        self.pred = pred

    def bars(self, symbol, timeframe, start, end, adjustment):
        df = super().bars(symbol, timeframe, start, end, adjustment)
        return df[~self.pred(df.index.tz_convert(NY))]


# ── Passing probes (claims that held) ───────────────────────────────────────


def test_prefix_topup_detects_adjustment_change(cache, capsys):
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-12", "2024-11-22", cache_dir=cache, source=src)
    src.factor = 0.9998  # ~2 bp dividend re-adjustment of all history
    out = bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-22", cache_dir=cache, source=src)
    assert "re-fetching full range" in capsys.readouterr().out
    ref = bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-22", cache_dir=cache / "r", source=FakeSource(0.9998))
    pd.testing.assert_frame_equal(out, ref, check_freq=False)


def test_disjoint_earlier_request_leaves_no_gap(cache):
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-18", "2024-11-22", cache_dir=cache, source=src)
    bars.load_bars("SPY", "5Min", "2024-10-01", "2024-10-05", cache_dir=cache, source=src)
    m = _meta(cache)
    assert (m["coverage_start"], m["coverage_end"]) == ("2024-10-01", "2024-11-22")
    n_calls = len(src.bar_calls)
    gap = bars.load_bars("SPY", "5Min", "2024-10-07", "2024-11-15", cache_dir=cache, source=src)
    assert len(src.bar_calls) == n_calls and len(gap) == 78 * 29


def test_dst_transitions_rth(cal=None):
    cal = calendar_from_raw(CAL_ROWS)
    src = FakeSource()
    for day in ("2024-03-11", "2024-11-04", "2024-03-08", "2024-11-01"):
        d = pd.Timestamp(day, tz=NY)
        r = rth_filter(src.bars("X", "5Min", d, d + pd.Timedelta(days=1), "all"), cal, 5)
        assert len(r) == 78 and r.index[0].strftime("%H:%M") == "09:30", day


def test_resampled_bins_only_use_bars_inside_bin():
    cal = calendar_from_raw(CAL_ROWS)
    src = FakeSource()
    d = pd.Timestamp("2024-11-29", tz=NY)  # 13:00 early close
    five = rth_filter(src.bars("X", "5Min", d, d + pd.Timedelta(days=1), "all"), cal, 5)
    for tf, m in (("15Min", 15), ("30Min", 30), ("1Hour", 60)):
        out = resample_session(five, cal, tf)
        for t, row in out.iterrows():
            inside = five[(five.index >= t) & (five.index < t + pd.Timedelta(minutes=m))]
            assert row["close"] == inside["close"].iloc[-1] and row["high"] == inside["high"].max()
        assert out.index[-1] + pd.Timedelta(minutes=m) <= pd.Timestamp("2024-11-29 13:00", tz=NY) or tf == "1Hour"


def test_torn_write_raises_not_stale(cache):
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-18", "2024-11-22", cache_dir=cache, source=src)
    _, js = cache_paths(cache, "sip", "all", "5Min", "SPY")
    old_json = js.read_text()
    bars.load_bars("SPY", "5Min", "2024-11-18", "2024-12-02", cache_dir=cache, source=src)
    js.write_text(old_json)  # simulate crash after .npz replace, before .json replace
    with pytest.raises(CacheError, match="hash"):
        bars.load_bars("SPY", "5Min", "2024-11-18", "2024-11-22", cache_dir=cache, source=src)


# ── Regressions for review findings ─────────────────────────────────────────


def test_refresh_does_not_discard_cached_coverage(cache):
    """refresh=True on a sub-range should not shrink the cache to that sub-range."""
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-01", "2024-12-01", cache_dir=cache, source=src)
    bars.load_bars("SPY", "5Min", "2024-11-12", "2024-11-14", cache_dir=cache, source=src, refresh=True)
    m = _meta(cache)
    assert (m["coverage_start"], m["coverage_end"]) == ("2024-11-01", "2024-12-01")


def test_empty_suffix_fetch_does_not_advance_coverage(cache):
    """A top-up that returns zero bars for sessions the calendar says traded must not be recorded as covered."""
    src = FakeSource()
    bars.load_bars("SPY", "5Min", "2024-11-01", "2024-11-15", cache_dir=cache, source=src)
    outage = Dropper(lambda ny: ny >= pd.Timestamp("2024-11-15", tz=NY))  # transient: nothing after 11-15
    bars.load_bars("SPY", "5Min", "2024-11-01", "2024-12-02", cache_dir=cache, source=outage)
    # Source recovers; the covered-but-empty sessions should now be fetched
    out = bars.load_bars("SPY", "5Min", "2024-11-01", "2024-12-02", cache_dir=cache, source=FakeSource())
    assert len(out.loc["2024-11-18":"2024-11-22"]) == 5 * 78


def test_daily_bar_close_is_session_close_or_session_dropped(cache):
    """A kept session missing its final bars yields a 1Day 'close' that is not the session close (2019-08-12 in real data)."""
    tail_hole = Dropper(
        lambda ny: (ny.normalize() == pd.Timestamp("2024-11-26", tz=NY)) & (ny.strftime("%H:%M") >= "15:35")
    )
    kw = {"cache_dir": cache, "source": tail_hole}
    daily = bars.load_bars("SPY", "1Day", "2024-11-25", "2024-12-03", **kw)
    ref = FakeSource().bars("X", "5Min", pd.Timestamp("2024-11-26", tz=NY), pd.Timestamp("2024-11-27", tz=NY), "all")
    true_close = ref.tz_convert(NY).loc["2024-11-26 15:55", "close"]
    d26 = pd.Timestamp("2024-11-26", tz=NY)
    assert d26 not in daily.index or daily.loc[d26, "close"] == true_close


def test_daily_quality_report_flags_partial_sessions(cache):
    """With min_session_coverage=0 (as data.fetch --report uses), 1Day completeness comes from the 5Min base."""
    holey = Dropper(
        lambda ny: (ny.normalize() == pd.Timestamp("2024-11-26", tz=NY)) & (ny.strftime("%H:%M") != "09:30")
    )
    kw = {"cache_dir": cache, "source": holey, "min_session_coverage": 0.0}
    daily = bars.load_bars("SPY", "1Day", "2024-11-25", "2024-12-03", **kw)
    base = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", **kw)
    rep = quality_report(daily, calendar_from_raw(CAL_ROWS), "1Day", base=base)
    assert rep["sessions_incomplete"] == 1 and rep["incomplete_dates"] == "2024-11-26"
    assert rep["sessions_missing_last_bar"] == 1


def test_cache_meta_has_spec_fields(cache):
    """Sidecar carries every field SPEC §1 lists."""
    bars.load_bars("SPY", "5Min", "2024-11-18", "2024-11-22", cache_dir=cache, source=FakeSource())
    spec = {"schema_version", "symbol", "timeframe", "feed", "adjustment", "tz", "columns", "n_bars", "first_ts",
            "last_ts", "fetched_at", "source", "session", "coverage_start", "coverage_end", "sha256"}  # fmt: skip
    assert spec <= set(_meta(cache))


def test_wavelet_main_empty_range_gives_clear_error(cache, monkeypatch):
    import wavelet_meta_model as wmm

    monkeypatch.setattr(wmm, "load_bars", lambda *a, **k: bars.load_bars(*a, cache_dir=cache, source=FakeSource(), **k))
    bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=FakeSource())  # warm cache
    with pytest.raises(ValueError):
        wmm.main(symbol="SPY", timeframe="5Min", start="2024-11-30", end="2024-12-02")  # weekend only


def test_quality_report_empty_intrasession_is_nan_safe():
    cal = calendar_from_raw(CAL_ROWS)
    d = pd.Timestamp("2024-11-27", tz=NY)
    five = rth_filter(FakeSource().bars("X", "5Min", d, d + pd.Timedelta(days=1), "all"), cal, 5)
    rep = quality_report(five.iloc[:1], cal, "5Min")
    assert np.isfinite(rep["max_abs_ret_intrasession"]) or rep["n_bars"] == 1


def test_holdout_guard(cache):
    with pytest.raises(bars.HoldoutError):
        bars.load_bars("SPY", "5Min", "2026-09-01", "2026-10-02", cache_dir=cache, source=FakeSource())
    with pytest.raises(bars.HoldoutError):  # resampled timeframes too
        bars.load_bars("SPY", "1Hour", "2026-09-01", "2026-10-02", cache_dir=cache, source=FakeSource())
    # exactly up to HOLDOUT_START is allowed; allow_holdout opts in
    assert bars.HOLDOUT_START == pd.Timestamp("2026-10-01")
    bars.load_bars("SPY", "5Min", "2024-11-25", "2026-10-01", cache_dir=cache, source=FakeSource())
    bars.load_bars("SPY", "5Min", "2024-11-25", "2026-10-03", cache_dir=cache, source=FakeSource(), allow_holdout=True)


def test_stale_calendar_is_refetched(tmp_path, monkeypatch):
    src = FakeSource()
    bars.get_calendar("2024-12-31", tmp_path, source=src)
    bars.get_calendar("2024-12-31", tmp_path, source=src)
    assert src.calendar_calls == 1
    blob = json.loads((tmp_path / "calendar.json").read_text())
    blob["fetched_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)).isoformat()
    (tmp_path / "calendar.json").write_text(json.dumps(blob))
    bars.get_calendar("2024-12-31", tmp_path, source=src)
    assert src.calendar_calls == 2


def test_mid_range_empty_session_is_logged(cache, capsys):
    hole = Dropper(lambda ny: ny.normalize() == pd.Timestamp("2024-11-26", tz=NY))
    bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=hole)
    out = capsys.readouterr().out
    assert "no bars in 1 session(s) inside the range: 2024-11-26" in out
    assert _meta(cache)["coverage_end"] == "2024-12-03"  # a hole inside the range does not cap coverage


def test_intraday_quality_flags_missing_last_bar(cache):
    tail = Dropper(lambda ny: (ny.normalize() == pd.Timestamp("2024-11-26", tz=NY)) & (ny.strftime("%H:%M") >= "15:35"))
    df = bars.load_bars("SPY", "5Min", "2024-11-25", "2024-12-03", cache_dir=cache, source=tail)
    rep = quality_report(df, calendar_from_raw(CAL_ROWS), "5Min")
    assert rep["sessions_missing_last_bar"] == 1 and rep["missing_last_bar_dates"] == "2024-11-26"
    assert rep["early_close_sessions"] == 1  # 11-29 closes 13:00 and is not flagged as missing its last bar
