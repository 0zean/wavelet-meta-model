"""
U11: fastfracdiff, the in-repo replacement for fracdiff 0.9.0 + statsmodels' adfuller.

`fracdiff_reference.json` was produced by the replaced implementation (fracdiff 0.9.0 FracdiffStat(mode="valid"),
statsmodels 0.15.0 adfuller) on prefixes of data/data.csv's close: every d the binary search tried with its ADF
statistic, p-value and lag, the chosen d, and the log-only diagnostic ADF (maxlag=20) at that d.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastfracdiff import adfuller, fdiff, fdiff_coef, find_d, mackinnonp
from features.fractional_diff import fit_fracdiff_d, fracdiff_transform
from utils.data_loader import load_ohlcv

REF = json.loads((Path(__file__).parent / "fracdiff_reference.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def close() -> np.ndarray:
    return load_ohlcv("data/data.csv")["close"].to_numpy()


def test_fdiff_matches_fracdiff_semantics():
    a = np.array([1, 2, 4, 7, 0])
    np.testing.assert_allclose(fdiff(a, 0.5, window=3), [1.0, 1.5, 2.875, 4.75, -4.0])
    np.testing.assert_allclose(fdiff(a, 0.5, window=3, mode="valid"), [2.875, 4.75, -4.0])
    np.testing.assert_allclose(fdiff_coef(1.5, 4), [1.0, -1.5, 0.375, 0.0625])
    for n in (0, 1, 2):  # an integer order is np.diff whatever the mode
        assert np.array_equal(fdiff(a, float(n), mode="valid"), np.diff(a, n))


def test_mackinnonp_bounds_and_monotone():
    assert mackinnonp(3.0) == 1.0 and mackinnonp(-20.0) == 0.0
    p = [mackinnonp(t) for t in np.linspace(-18, 2.7, 200)]
    assert np.all(np.diff(p) > 0)
    assert mackinnonp(-2.8621) == pytest.approx(0.05, abs=1e-3)  # the 5 % critical value with a constant


@pytest.mark.parametrize("case", REF, ids=[f"n{c['n']}" for c in REF])
def test_d_search_matches_fracdiff_and_statsmodels(case, close):
    x = close[: case["n"]]
    for d, stat, pvalue, lag in case["path"]:
        r = adfuller(fdiff(x, d, window=10, mode="valid"))
        assert r.usedlag == lag
        assert r.stat == pytest.approx(stat, rel=1e-8, abs=1e-10)
        assert r.pvalue == pytest.approx(pvalue, rel=1e-6, abs=1e-12)
    assert find_d(x) == case["d"]
    r = adfuller(fdiff(x, case["d"], window=10, mode="valid"), maxlag=20)
    stat, pvalue, lag = case["diag"]
    assert r.usedlag == lag and r.stat == pytest.approx(stat, rel=1e-8) and r.pvalue == pytest.approx(pvalue, rel=1e-6)


def test_fixed_lag_and_short_series_errors():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=500))
    assert adfuller(x, maxlag=7, autolag=None).usedlag == 7
    assert adfuller(np.diff(x)).pvalue < 0.01 < adfuller(x).pvalue  # white noise vs random walk
    with pytest.raises(ValueError):
        adfuller(x[:20], maxlag=10)


def test_fit_fracdiff_d_state_round_trip(close):
    train = pd.Series(close[:3000])
    fit = fit_fracdiff_d(train)
    assert fit.d == next(c["d"] for c in REF if c["n"] == 3000) and fit.window == 10
    fd = fracdiff_transform(train, fit)
    assert len(fd) == 3000 - 9 and fd.index[0] == 9
    np.testing.assert_array_equal(fd.to_numpy(), fdiff(close[:3000], fit.d, window=10, mode="valid"))
