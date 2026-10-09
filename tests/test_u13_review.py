"""
Adversarial review probes for U13 (no network, no Alpaca; reads only checked-in CSVs and, where present, the local
cboe exo cache). Failing tests here are findings, not regressions of the existing suite.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import events
from data.exo import EXO_CACHE_DIR, cache_paths, load_series_cache, vx_continuous
from risk.costs import auction_flags, fill_costs, quotes_half_spread
from utils.config import RunConfig

NY = "America/New_York"


# ── VX continuous: a contract months from expiry must not be labelled the front month ──────────────────────────


def test_vx_continuous_never_labels_a_distant_contract_as_front():
    """
    The CFE listing is read from 2015 on, but the Jan-2015 contract trades from April 2014. On a 2014 date the code
    picks 'the earliest expiry >= d' among the *loaded* contracts, i.e. the Jan-2015 contract 9 months out, as VX1.
    """
    d = pd.bdate_range("2014-04-21", "2015-02-18")
    jan = pd.Series(20.0, index=d[d <= "2015-01-21"])
    feb = pd.Series(21.0, index=d[d <= "2015-02-18"])
    contracts = {pd.Timestamp("2015-01-21"): jan, pd.Timestamp("2015-02-18"): feb}
    out = vx_continuous(contracts)
    exps = np.array(sorted(contracts), dtype="datetime64[ns]")
    front = np.array([exps[np.searchsorted(exps, x)] for x in out.index.to_numpy()])
    days_to_front = (front - out.index.to_numpy()).astype("timedelta64[D]").astype(int)
    # a monthly front contract is never more than ~5 weeks from expiry
    assert days_to_front.max() <= 36, (
        f"VX1 is a contract {days_to_front.max()} days from expiry on {out.index[0].date()}"
    )


def _cached(source, name):
    npz, js = cache_paths(EXO_CACHE_DIR, source, name)
    if not npz.exists():
        pytest.skip(f"no local cache for {source}/{name}")
    return load_series_cache(npz, js, {"source": source, "name": name})[0]


def test_cached_vx1_vix_has_no_pre_2015_rows_from_a_back_month():
    """The real cache: VX1 / VIX on 2014 dates is the Jan-2015 future over spot (median ≈ 1.27 vs ≈ 1.04 after)."""
    r = _cached("cboe", "VX1_VIX")["value"]
    early = r.loc[:"2014-12-17"]
    assert early.empty, f"{len(early)} VX1_VIX rows before 2014-12-18 (median {early.median():.3f}) use a back month"


# ── Earnings: acceptance instants for a fixed-ET-time release must move by 1 h in UTC across DST, not 2 h ───────


@pytest.mark.parametrize("symbol", ["AAPL", "AMZN", "META", "JPM", "UNH", "GOOGL", "MSFT", "NVDA", "XOM"])
def test_earnings_acceptance_times_shift_one_hour_across_dst(symbol):
    ea = events.read_earnings()
    ea = ea[(ea["symbol"] == symbol) & (ea["known_from"] != "")]  # the company's regular release
    t = pd.to_datetime(ea["accepted_at"], utc=True)
    et = t.dt.tz_convert(NY)
    dst = et.map(lambda x: bool(x.dst())).to_numpy()
    utc_min = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    shift = np.median(utc_min[~dst]) - np.median(utc_min[dst])
    # a release at a fixed NY wall-clock time is 60 min later in UTC in winter; 120 min = the NY offset applied twice
    assert abs(shift - 60) <= 30, f"{symbol}: UTC acceptance minute moves {shift:.0f} min between DST and standard time"


# ── Calendar: releases moved by the Jan 31 – Feb 3, 2026 funding lapse ─────────────────────────────────────────


def test_feb_2026_rescheduled_bls_releases_are_not_known_from_january_first():
    """
    Assumes the public record: the January 2026 Employment Situation (scheduled 2026-02-06) was released 2026-02-11
    and January CPI (scheduled 2026-02-11) on 2026-02-13 after the brief early-February 2026 funding lapse. Neither new
    date was public on 2026-01-01.
    """
    ev = events.read_events()
    rows = ev[
        ((ev["kind"] == "NFP") & (ev["date"] == "2026-02-11")) | ((ev["kind"] == "CPI") & (ev["date"] == "2026-02-13"))
    ]
    assert len(rows) == 2
    assert (rows["known_from"] > "2026-01-31").all(), rows[["kind", "date", "known_from"]].to_string()


# ── Cost model sanity probes (expected to pass; they document what was attacked) ─────────────────────────────


def _table(bins_values: dict, years=(2016,)) -> pd.DataFrame:
    rows = [(y, b, v) for y in years for b, v in bins_values.items()]
    return pd.DataFrame(rows, columns=["year", "bin", "half_spread_bp"]).assign(symbol="X", n=1)


def test_one_hour_stub_and_early_close_auction_flags():
    # 1Hour session-anchored bars on a full day (stub 15:30) and an early close (stub 12:30)
    full = pd.date_range("2024-07-02 09:30", periods=7, freq="60min", tz=NY)
    early = pd.date_range("2024-07-03 09:30", periods=4, freq="60min", tz=NY)
    idx = full.append(early)
    o, c = auction_flags(idx, 60)
    assert o.tolist() == [True] + [False] * 6 + [True] + [False] * 3
    assert c.tolist() == [False] * 6 + [True] + [False] * 3 + [True]


def test_quotes_costs_by_kind_hand_values():
    bins = {f"{h:02d}:{m:02d}": 1.0 for h in range(9, 16) for m in (0, 15, 30, 45) if (9, 30) <= (h, m) <= (15, 45)}
    bins |= {"09:30": 3.0, "15:45": 2.0, "open_auction": 5.0, "close_auction": 4.0, "day": 1.5}
    tbl = _table(bins, years=(2016,))
    cfg = RunConfig.for_timeframe("5Min", COST_MODEL="quotes", SLIPPAGE_PCT=1e-4)
    idx = pd.DatetimeIndex([pd.Timestamp("2019-03-04 09:30", tz=NY), pd.Timestamp("2019-03-04 15:50", tz=NY),
                            pd.Timestamp("2019-03-04 15:55", tz=NY)])  # fmt: skip
    fc = fill_costs(idx, cfg, tbl)
    np.testing.assert_allclose(fc["open"], [5e-4, 1e-4 + 2e-4, 1e-4 + 2e-4])
    np.testing.assert_allclose(fc["intra"], [1e-4 + 3e-4, 1e-4 + 2e-4, 1e-4 + 2e-4])
    np.testing.assert_allclose(fc["close"], [1e-4 + 3e-4, 1e-4 + 2e-4, 4e-4])  # 2019 → nearest earlier year 2016


def test_quotes_lookup_refuses_years_before_the_table():
    with pytest.raises(ValueError):
        quotes_half_spread(np.array([2015]), np.array(["day"], dtype=object), _table({"day": 1.0}))


def test_quotes_table_is_checked_in():
    """COST_MODEL defaults to 'quotes', which reads data/costs/quotes_half_spread.csv: it must be tracked by git."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(["git", "ls-files", "data/costs/quotes_half_spread.csv"], cwd=root, capture_output=True,
                         text=True, check=True).stdout  # fmt: skip
    assert out.strip(), "the default cost model's table is not committed"
