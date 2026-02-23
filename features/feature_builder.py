import numpy as np
import pandas as pd

from features.causal_modwt import wavelet_ar_features
from features.indicators import compute_log_return, compute_rsi, compute_siegel_slope, compute_vwap


def build_features(df: pd.DataFrame, fd_close: pd.Series) -> pd.DataFrame:
    """
    Assemble the complete feature matrix for a single data split.
    All features are strictly causal.

    Features:
        fd_close        : fractionally differenced close (stationary, memory-preserving)
        vwap_dev        : (close - VWAP) / VWAP — distance from intraday VWAP
        rsi             : RSI(14)
        siegel_slope    : rolling Siegel slope on close(20)
        log_ret         : 1-bar log return
        log_vol         : log(volume)
        w_lag_1..4      : causal MODWT S4 AR features (lags 1-4 of smoothing component)

    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        fd_close (pd.Series): Fractionally differenced close

    Returns:
        pd.DataFrame: DataFrame with features
    """
    feats = pd.DataFrame(index=df.index)

    # Fractionally differenced close (aligned externally, merge on index)
    feats["fd_close"] = fd_close

    # VWAP deviation
    vwap = compute_vwap(df)
    feats["vwap_dev"] = (df["close"] - vwap) / vwap

    # RSI
    feats["rsi"] = compute_rsi(df["close"])

    # Siegel slope
    feats["siegel_slope"] = compute_siegel_slope(df["close"])

    # Log return
    feats["log_ret"] = compute_log_return(df["close"])

    # Log volume
    feats["log_vol"] = np.log(df["volume"].replace(0, np.nan))

    # MODWT S4 AR features (fit on raw close — note: we pass the full split's
    # close here.  The wavelet transform is computed independently per split
    # using only data within that split, so no cross-split leakage.)
    wav = wavelet_ar_features(df["close"])
    feats = feats.join(wav)

    # High-low spread (intrabar volatility proxy)
    feats["hl_spread"] = (df["high"] - df["low"]) / df["close"]

    return feats
