import pandas as pd
from pywddff.filters import scaling_filter

from utils.config import config


def wavelet_ar_features(series: pd.Series) -> pd.DataFrame:
    """
    Lagged MODWT scaling (smooth) coefficients S_J of a price series.

    Uses the MODWT pyramid (Percival & Walden 2000, p.177) without the circular
    wrap:  V_j[t] = Σ_n g[n] · V_{j-1}[t - n·2^(j-1)].  Every coefficient depends
    only on past samples; the first (2^J - 1)(L - 1) rows are NaN, which are
    exactly the boundary coefficients pywddff.modwt(remove_bc=True) drops.
    For db1 / J=4, S4 is a 16-bar trailing mean.

    Args:
        series (pd.Series): Input price series (e.g. close)

    Returns:
        pd.DataFrame: S_J lagged 1..AR_LAGS bars, same index as series.
    """
    g = scaling_filter(config.WAVELET_FILTER, modwt=True)
    smooth = series.astype(float)
    for j in range(config.WAVELET_J):
        smooth = sum(g[n] * smooth.shift(n * 2**j) for n in range(len(g)))

    return pd.DataFrame({f"w_lag_{lag}": smooth.shift(lag) for lag in range(1, config.AR_LAGS + 1)})
