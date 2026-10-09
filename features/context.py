"""
The context a symbol's feature groups read (SPEC §15, U15), loaded only when a group declares it:

    "market"  SPY bars at the run's timeframe (cross_asset)
    "sector"  the mapped sector ETF's bars (data/sectors.py; stocks only)
    "exo"     {"source/name": load_series frame} for the groups' declared series (features.exo_align reads them
              point in time)

Exo rows are requested from EXO_LOOKBACK before the first bar (the as-of value on the first bars, multi-observation
changes, `since_*`). Calendar rows, which are schedules known in advance, are also requested CALENDAR_LOOKAHEAD past
the last bar (`to_*` near the end), past HOLDOUT_START if need be: they carry no market outcome, and the Exo view
still shows a row only from its `known_from` (U15 review: clipping them gave the dev window's last weeks an artificial
"no release scheduled").
"""

import pandas as pd

from data.sectors import MARKET, sector_of
from features.registry import context_needs

EXO_LOOKBACK = pd.Timedelta(days=120)
CALENDAR_LOOKAHEAD = pd.Timedelta(days=120)


def exo_range(name: str, start, end, allow_holdout: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the rows to load for exo series `name` around bars in [start, end)."""
    start, end = pd.Timestamp(start) - EXO_LOOKBACK, pd.Timestamp(end)
    if name.startswith("calendar/"):
        end = end + CALENDAR_LOOKAHEAD
    return start.normalize(), end.normalize()


def load_context(
    symbol: str, timeframe: str, start, end, groups, *, bars, exo, allow_holdout: bool = False, market: str = MARKET
) -> dict:
    """
    The context dict for `symbol`'s `groups` (empty when no group reads one).

    Args:
        bars: callable(symbol, timeframe, start, end, *, allow_holdout) -> bars (e.g. data.bars.load_bars).
        exo: callable(source, name, start, end, *, allow_holdout) -> frame (e.g. data.exo.load_series).
    """
    if groups is None:  # the legacy matrix reads no context
        return {}
    keys, names = context_needs(groups, timeframe)
    ctx = {}
    if "market" in keys:
        if symbol.upper() == market.upper():  # beta 1, residual 0: degenerate columns, not a feature
            raise ValueError(f"'cross_asset' needs a symbol other than the market ({market}); drop the group for it")
        ctx["market"] = bars(market, timeframe, start, end, allow_holdout=allow_holdout)
    sector = sector_of(symbol)
    if "sector" in keys and sector is not None:
        ctx["sector"] = bars(sector, timeframe, start, end, allow_holdout=allow_holdout)
    if names:
        frames = {}
        for name in sorted(names):
            source, series = name.split("/", 1)
            a, b = exo_range(name, start, end, allow_holdout)
            frames[name] = exo(source, series, a, b, allow_holdout=allow_holdout or source == "calendar")
        ctx["exo"] = frames
    return ctx
