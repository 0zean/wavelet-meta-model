import pandas as pd

from features.indicators import compute_vwap
from utils.config import config


def make_active_day_mask(
    df: pd.DataFrame,
    pctile: float = config.LOW_MOVE_PCTILE,
) -> pd.Series:
    """
    For each trading day, compute |VWAP_last - VWAP_first| as a proxy for
    directional daily movement.  Days in the bottom `pctile` fraction are
    flagged as "low-movement" and excluded from classifier training.

    Returns a boolean Series (True = bar belongs to an *active* day).
    This mask is applied ONLY during classifier training — NOT during
    inference. The meta-model compensates for quiet-period signal degradation.

    Args:
        df (pd.DataFrame): Input dataframe with ohlc data
        pctile (float, optional): Percentile of daily movement to use as threshold. Defaults to config.LOW_MOVE_PCTILE.

    Returns:
        pd.Series: Boolean mask indicating active days
    """
    vwap = compute_vwap(df)
    day_key = df.index.normalize()

    day_vwap = pd.DataFrame({"vwap": vwap, "day": day_key})
    daily_open_v = day_vwap.groupby("day")["vwap"].first()
    daily_close_v = day_vwap.groupby("day")["vwap"].last()
    daily_move = (daily_close_v - daily_open_v).abs()

    threshold = daily_move.quantile(pctile)
    active_days = daily_move[daily_move >= threshold].index

    mask = pd.Series(day_key.isin(active_days), index=df.index, name="active_day")
    pct = mask.mean() * 100
    print(f"[FILTER] Active-day mask: {pct:.1f}% of bars retained for classifier training")
    return mask
