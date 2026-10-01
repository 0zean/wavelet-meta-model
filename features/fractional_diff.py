import numpy as np
import pandas as pd
from fracdiff import fdiff
from fracdiff.sklearn import FracdiffStat
from statsmodels.tsa.stattools import adfuller


def fit_fracdiff_d(train: pd.Series) -> FracdiffStat:
    """
    Fit FracdiffStat on TRAIN only: the minimum d whose fixed-window fractional
    difference passes ADF at 5 %.

    Raises:
        RuntimeWarning: FracdiffStat raises this when no d <= 1 is stationary.

    Returns:
        FracdiffStat: Fitted transformer (d in `d_[0]`, window in `window`).
    """
    fs = FracdiffStat(mode="valid")
    Xt = fs.fit_transform(train.to_numpy().reshape(-1, 1)).reshape(-1)

    # The ADF p-value is a log diagnostic only (FracdiffStat chose d); adfuller(maxlag=20) needs > 44 points
    pval = adfuller(Xt, maxlag=20, autolag="AIC")[1] if Xt.size > 2 * (20 + 2) else np.nan
    corr = np.corrcoef(train.to_numpy()[-Xt.size :], Xt)[0, 1]
    print(f"\n[FRACDIFF] Optimal d  : {fs.d_[0]:.4f}")
    print(f"           ADF p-val  : {pval * 100:.4f} %  (n={Xt.size})")
    print(f"           Corr (raw) : {corr:.4f}")
    return fs


def fracdiff_transform(close: pd.Series, fs: FracdiffStat) -> pd.Series:
    """
    Apply a fitted fixed-window fractional difference to a whole series.

    The filter is causal (bar t uses bars t-window+1 … t), so transforming the
    continuous series gives the same train values as fitting did, and val/test
    bars use the preceding split's prices as history instead of losing their
    first window-1 rows.

    Args:
        close (pd.Series): Price series.
        fs (FracdiffStat): Transformer fitted on the train split.

    Returns:
        pd.Series: Fractionally differenced series; the leading window-1 bars are dropped.
    """
    arr = fdiff(close.to_numpy(), fs.d_[0], window=fs.window, mode="valid")
    # fdiff falls back to np.diff for integer d, so take the length from the output
    return pd.Series(arr, index=close.index[-arr.size :], name="fd_close")
