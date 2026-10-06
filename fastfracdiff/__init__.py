"""
fastfracdiff — fractional differencing with a fast minimum-stationary-d search.

A self-contained replacement (numpy + scipy only) for the parts of the archived `fracdiff` package (0.9.0, BSD-3,
Shota Imaki) this project uses: `fdiff` (fixed-window fractional difference) and `FracdiffStat`'s binary search for
the minimum d whose fractional difference passes the augmented Dickey–Fuller test. The search's cost was the ADF
test: statsmodels' `adfuller` with AIC lag selection fits ~12·(n/100)^¼ nested OLS regressions (≈ 80 at n = 190k),
each through an SVD pseudo-inverse. Here one Cholesky of the widest (centred) design's Gram matrix gives every nested
model's residual sum of squares, ~170x faster at equal results (same lag, t-statistic and MacKinnon p-value to
rounding). See NOTICE.

    from fastfracdiff import fdiff, adfuller, find_d
    d = find_d(close, window=10)                 # nan when not even d = 1 is stationary
    fd = fdiff(close, d, window=10, mode="valid")
"""

from fastfracdiff.adf import ADFResult, adfuller, mackinnonp
from fastfracdiff.core import fdiff, fdiff_coef
from fastfracdiff.stat import find_d

__all__ = ["ADFResult", "adfuller", "fdiff", "fdiff_coef", "find_d", "mackinnonp"]
