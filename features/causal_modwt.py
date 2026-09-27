import pandas as pd
from pywddff.filters import scaling_filter


def wavelet_ar_features(series: pd.Series, wavelet_filter: str, levels: int, ar_lags: int) -> pd.DataFrame:
    """
    Lagged MODWT scaling (smooth) coefficients S_J of a price series.

    Uses the MODWT pyramid (Percival & Walden 2000, p.177) without the circular
    wrap:  V_j[t] = Σ_n g[n] · V_{j-1}[t - n·2^(j-1)].  Every coefficient depends
    only on past samples; the first (2^J - 1)(L - 1) rows are NaN, which are
    exactly the boundary coefficients pywddff.modwt(remove_bc=True) drops.
    For db1 / J=4, S4 is a 16-bar trailing mean.

    Args:
        series (pd.Series): Input price series (e.g. close)
        wavelet_filter (str): Wavelet filter name (cfg.WAVELET_FILTER).
        levels (int): Decomposition depth J (cfg.WAVELET_J).
        ar_lags (int): Number of lags of S_J (cfg.AR_LAGS).

    Returns:
        pd.DataFrame: S_J lagged 1..ar_lags bars, same index as series.
    """
    g = scaling_filter(wavelet_filter, modwt=True)
    smooth = series.astype(float)
    for j in range(levels):
        smooth = sum(g[n] * smooth.shift(n * 2**j) for n in range(len(g)))

    return pd.DataFrame({f"w_lag_{lag}": smooth.shift(lag) for lag in range(1, ar_lags + 1)})
