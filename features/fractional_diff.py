from dataclasses import dataclass

import numpy as np
import pandas as pd

from fastfracdiff import adfuller, fdiff, find_d

WINDOW = 10  # fixed-window filter length (fracdiff's FracdiffStat default, kept)


@dataclass(frozen=True)
class FracdiffFit:
    d: float
    window: int = WINDOW


def fit_fracdiff_d(train: pd.Series) -> FracdiffFit:
    """
    Fit on TRAIN only: the minimum d whose fixed-window fractional difference passes ADF at 5 %
    (binary search to 0.01, `fastfracdiff.find_d`).

    Raises:
        RuntimeWarning: no d <= 1 is stationary.
    """
    x = train.to_numpy(dtype=np.float64)
    d = find_d(x, window=WINDOW)
    if np.isnan(d):
        raise RuntimeWarning("no d <= 1 makes the series ADF-stationary")
    xt = fdiff(x, d, window=WINDOW, mode="valid")

    # The ADF p-value is a log diagnostic only (find_d chose d); maxlag=20 needs > 44 points
    pval = adfuller(xt, maxlag=20).pvalue if xt.size > 2 * (20 + 2) else np.nan
    corr = np.corrcoef(x[-xt.size :], xt)[0, 1]
    print(f"\n[FRACDIFF] Optimal d  : {d:.4f}")
    print(f"           ADF p-val  : {pval * 100:.4f} %  (n={xt.size})")
    print(f"           Corr (raw) : {corr:.4f}")
    return FracdiffFit(d)


def fracdiff_transform(close: pd.Series, fit: FracdiffFit) -> pd.Series:
    """
    Apply a fitted fixed-window fractional difference to a whole series.

    The filter is causal (bar t uses bars t-window+1 … t), so transforming the
    continuous series gives the same train values as fitting did, and val/test
    bars use the preceding split's prices as history instead of losing their
    first window-1 rows.

    Returns:
        pd.Series: Fractionally differenced series; the leading window-1 bars are dropped.
    """
    arr = fdiff(close.to_numpy(), fit.d, window=fit.window, mode="valid")
    # an integer d is a plain np.diff, so take the length from the output
    return pd.Series(arr, index=close.index[-arr.size :], name="fd_close")
