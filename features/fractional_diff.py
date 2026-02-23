import numpy as np
import pandas as pd
from fracdiff import fdiff
from fracdiff.sklearn import FracdiffStat
from statsmodels.tsa.stattools import adfuller


def fracdiff_fit_transform(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, FracdiffStat]:
    """
    Fit FracdiffStat on TRAIN only (finds minimum d that passes ADF at 5%),
    then apply the *same* d to val and test — zero look-ahead leakage.

    FracdiffStat uses mode="valid" which already discards early rows where
    the finite-memory filter is unreliable.  We propagate that trimming to
    val/test manually using fdiff().

    Returns aligned Series for each split + the fitted transformer.
    """
    X_train = train.values.reshape(-1, 1)

    fs = FracdiffStat(mode="valid")
    Xt = fs.fit_transform(X_train).reshape(-1)

    d_opt = fs.d_[0]
    _, pval, *_ = adfuller(Xt, maxlag=20, autolag="AIC")
    corr = np.corrcoef(X_train[-Xt.size :, 0], Xt)[0, 1]

    print(f"\n[FRACDIFF] Optimal d  : {d_opt:.4f}")
    print(f"           ADF p-val  : {pval * 100:.4f} %")
    print(f"           Corr (raw) : {corr:.4f}")

    train_fd = pd.Series(Xt, index=train.index[-Xt.size :], name="fd_close")
    val_arr = fdiff(val.values, d_opt, mode="valid")
    test_arr = fdiff(test.values, d_opt, mode="valid")
    val_fd = pd.Series(val_arr, index=val.index[-val_arr.size :], name="fd_close")
    test_fd = pd.Series(test_arr, index=test.index[-test_arr.size :], name="fd_close")

    return train_fd, val_fd, test_fd, fs
