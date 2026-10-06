"""Minimum stationary order of fractional differencing (fracdiff 0.9.0's `FracdiffStat` search, ADF only)."""

import numpy as np

from fastfracdiff.adf import adfuller
from fastfracdiff.core import fdiff


def find_d(
    x,
    window: int = 10,
    mode: str = "valid",
    pvalue: float = 0.05,
    precision: float = 0.01,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """
    Binary search for the smallest d in [lower, upper] whose `fdiff(x, d, window, mode)` rejects a unit root (ADF
    p-value < `pvalue`), to within `precision`; the upper end of the final bracket is returned. nan if `upper`
    itself is not stationary, `lower` if it already is. Assumes stationarity is monotone in d, as fracdiff did.
    """
    x = np.asarray(x, dtype=np.float64)

    def stationary(d: float) -> bool:
        return adfuller(fdiff(x, d, window=window, mode=mode)).pvalue < pvalue

    if not stationary(upper):
        return float("nan")
    if stationary(lower):
        return float(lower)
    while upper - lower > precision:
        m = (upper + lower) / 2
        if stationary(m):
            upper = m
        else:
            lower = m
    return float(upper)
