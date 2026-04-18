import numpy as np
import pandas as pd

from utils.config import config


def triple_barrier_labels(
    df: pd.DataFrame,
    vertical_bars: int = config.VERTICAL_BARS,
    barrier_mult: float = config.BARRIER_MULT,
    vol_lookback: int = config.VOL_LOOKBACK,
) -> pd.Series:
    """
    López de Prado (2018) Triple-Barrier labeling.

    For each bar t:
      • Compute daily σ = rolling std of log-returns over vol_lookback bars
      • Upper barrier = close[t] x (1 + barrier_mult x σ[t])
      • Lower barrier = close[t] x (1 - barrier_mult x σ[t])
      • Vertical barrier = t + vertical_bars

    Touch detection uses HIGH (for upper) and LOW (for lower) within the
    forward window, this is more realistic than close-only detection.

    Args:
        df (pd.DataFrame): DataFrame with columns 'close', 'high', 'low'.
        vertical_bars (int, optional): Number of bars to look ahead for the vertical barrier. Defaults to config.VERTICAL_BARS.
        barrier_mult (float, optional): Multiplier for the volatility to set the barrier levels. Defaults to config.BARRIER_MULT.
        vol_lookback (int, optional): Number of bars to use for calculating the rolling volatility. Defaults to config.VOL_LOOKBACK.

    Returns:
        pd.Series: Series of labels {-1, 0, 1}, same index as df.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    daily_vol = log_ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = daily_vol.values
    n = len(df)
    labels = np.full(n, np.nan)

    for t in range(n - vertical_bars):
        c0 = close[t]
        s = vol[t] if not np.isnan(vol[t]) else 0.001
        ub = c0 * (1.0 + barrier_mult * s)
        lb = c0 * (1.0 - barrier_mult * s)
        lbl = 0  # default: vertical barrier

        for k in range(1, vertical_bars + 1):
            idx = t + k
            if high[idx] >= ub:
                lbl = 1
                break
            if low[idx] <= lb:
                lbl = -1
                break
        labels[t] = lbl

    result = pd.Series(labels, index=df.index, name="label")
    print(
        f"[LABEL] Distribution: "
        f"-1={(result == -1).sum()}  "
        f"0={(result == 0).sum()}  "
        f"1={(result == 1).sum()}  "
        f"NaN={(result.isna()).sum()}"
    )
    return result
