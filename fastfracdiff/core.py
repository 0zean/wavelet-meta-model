"""Fixed-window fractional difference (the semantics of fracdiff 0.9.0's `fdiff` for 1-D input)."""

import numpy as np
from scipy.special import binom


def fdiff_coef(d: float, window: int) -> np.ndarray:
    """The first `window` weights of (1 - B)^d: (-1)^k C(d, k).

    >>> fdiff_coef(0.5, 4)
    array([ 1.    , -0.5   , -0.125 , -0.0625])
    """
    k = np.arange(window)
    return (-1) ** k * binom(d, k)


def fdiff(a, d: float = 1.0, window: int = 10, mode: str = "same") -> np.ndarray:
    """
    d-th fractional difference of a 1-D series, truncated to `window` weights.

    mode "valid": only points where every weight is used (length len(a) - window + 1); "same": length len(a), the
    first window - 1 points using the weights that fit. An integer d is `np.diff(a, d)` (length len(a) - d,
    whatever the mode), as in fracdiff.

    >>> fdiff(np.array([1, 2, 4, 7, 0]), 0.5, window=3, mode="valid")
    array([ 2.875,  4.75 , -4.   ])
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("fdiff takes a 1-D series")
    if float(d).is_integer():
        return np.diff(a, n=int(d))
    dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64
    w = fdiff_coef(d, window).astype(dtype)
    if mode == "valid":
        return np.convolve(a, w, mode="valid")
    if mode == "same":
        return np.convolve(a, w, mode="full")[: a.size]
    raise ValueError(f"invalid mode: {mode!r}")
