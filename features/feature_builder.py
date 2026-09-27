import numpy as np
import pandas as pd

from features.causal_modwt import wavelet_ar_features
from features.indicators import compute_log_return, compute_rsi, compute_siegel_slope, compute_vwap
from utils.config import RunConfig


def build_features(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    Assemble the causal feature matrix: every column at bar t uses bars <= t only,
    so it can be computed once on the full series and sliced per fold.

    fd_close is not built here: its d is fit per fold on train and added in run_wfo.

    Features:
        vwap_dev        : (close - VWAP) / VWAP — distance from intraday VWAP
                          (for 1Day bars VWAP is the bar's own typical price)
        rsi             : RSI(14)
        siegel_slope    : rolling Siegel slope on close(20)
        log_ret         : 1-bar log return
        log_vol         : log(volume)
        w_lag_1..4      : causal MODWT S4 AR features (lags 1-4 of smoothing component)
        hl_spread       : (high - low) / close

    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        cfg (RunConfig): Run configuration.

    Returns:
        pd.DataFrame: DataFrame with features
    """
    feats = pd.DataFrame(index=df.index)

    # VWAP deviation
    vwap = compute_vwap(df)
    feats["vwap_dev"] = (df["close"] - vwap) / vwap

    # RSI
    feats["rsi"] = compute_rsi(df["close"], cfg.RSI_PERIOD)

    # Siegel slope
    feats["siegel_slope"] = compute_siegel_slope(df["close"], cfg.SIEGEL_WINDOW)

    # Log return
    feats["log_ret"] = compute_log_return(df["close"])

    # Log volume
    feats["log_vol"] = np.log(df["volume"].replace(0, np.nan))

    # MODWT S4 AR features (causal filter; boundary rows are NaN)
    feats = feats.join(wavelet_ar_features(df["close"], cfg.WAVELET_FILTER, cfg.WAVELET_J, cfg.AR_LAGS))

    # High-low spread (intrabar volatility proxy)
    feats["hl_spread"] = (df["high"] - df["low"]) / df["close"]

    return feats
