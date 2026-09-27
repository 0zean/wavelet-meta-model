import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from utils.config import config


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday cumulative VWAP, reset at the start of each trading day.
    At time t, VWAP uses volume and prices up to and including bar t only.
    Causal: no future data.

    VWAP = cumsum(typical_price x volume) / cumsum(volume)

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close', and 'volume' columns

    Returns:
        pd.Series: VWAP values
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"]

    day_key = df.index.normalize()  # date component of DatetimeIndex
    df2 = pd.DataFrame({"tp_vol": tp_vol, "volume": df["volume"], "day": day_key})

    cum_tp_vol = df2.groupby("day")["tp_vol"].cumsum()
    cum_vol = df2.groupby("day")["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol
    vwap.name = "vwap"
    return vwap


def compute_rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """
    Wilder RSI using exponential smoothing.  Causal: no future data.

    Args:
        close (pd.Series): Close prices
        period (int, optional): RSI lookback period. Defaults to config.RSI_PERIOD.

    Returns:
        pd.Series: RSI values
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Use Wilder's smoothing (equivalent to EWM with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "rsi"
    return rsi


def compute_siegel_slope(
    series: pd.Series,
    window: int = config.SIEGEL_WINDOW,
) -> pd.Series:
    """
    Rolling Siegel slope.
    Siegel's repeated median slope estimator is a method for robust
    linear regression using repeated medians with a breakdown point of 50%.

    Args:
        series (pd.Series): Input series
        window (int, optional): Lookback window. Defaults to config.SIEGEL_WINDOW.

    Returns:
        pd.Series: Siegel slope values
    """
    arr = series.to_numpy(dtype=float)
    slopes = np.full(len(arr), np.nan)
    if len(arr) < window:
        return pd.Series(slopes, index=series.index, name="siegel_slope")

    # Repeated medians over x = 0..w-1: slope = median_i( median_{j != i} (y_j - y_i) / (j - i) )
    rows = np.arange(window)[:, None]
    cols = np.array([np.delete(np.arange(window), i) for i in range(window)])  # (w, w-1), j != i
    dx = (cols - rows).astype(float)
    windows = sliding_window_view(arr, window)  # (n - w + 1, w), no copy
    out = slopes[window - 1 :]  # view into slopes; NaN windows stay NaN via np.median

    chunk = 4096
    for start in range(0, len(windows), chunk):
        y = windows[start : start + chunk]
        pair_slopes = (y[:, cols] - y[:, rows]) / dx  # (chunk, w, w-1)
        out[start : start + chunk] = np.median(np.median(pair_slopes, axis=2), axis=1)

    return pd.Series(slopes, index=series.index, name="siegel_slope")


def compute_log_return(close: pd.Series) -> pd.Series:
    """
    Log return of price.

    Args:
        close (pd.Series): Close price of stock.

    Returns:
        pd.Series: Log returns of stock.
    """
    return np.log(close / close.shift(1)).rename("log_ret")
