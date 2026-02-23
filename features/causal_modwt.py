import pandas as pd
import pywddff.pywddff as pyw

from utils.config import config


def wavelet_ar_features(series: pd.Series) -> pd.DataFrame:
    """
    Apply MODWT and return a dataframe of only the S4 (smoothed) sub-band.
    remove_bc=True ensures each coefficient is derived only from *past* samples

    Args:
        series (pd.Series): Input price series (e.g. close)

    Returns:
        pd.DataFrame: Dataframe of S4 lagged values
    """
    coefs = pyw.modwt(x=series.values, filter=config.WAVELET_FILTER, J=config.WAVELET_J, remove_bc=True)
    # coefs shape: (n_valid, J+1);  J wavelet bands + 1 scaling band
    idx = series.index[-coefs.shape[0] :]
    all_band_names = [f"W{j + 1}" for j in range(config.WAVELET_J)] + [f"S{config.WAVELET_J}"]

    all_coefs = pd.DataFrame(coefs, index=idx, columns=all_band_names)

    s4_column = all_coefs["S4"]
    feat = pd.DataFrame(index=series.index)
    for lag in range(1, config.AR_LAGS + 1):
        feat[f"w_lag_{lag}"] = s4_column.shift(lag)
    return feat
