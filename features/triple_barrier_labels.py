import numpy as np
import pandas as pd

from utils.config import config


def bar_volatility(close: pd.Series, span: int = config.VOL_SPAN) -> pd.Series:
    """
    Causal EWM standard deviation of 1-bar log returns (σ per bar).

    σ[t] uses returns up to and including bar t, so it is known at the close of t.

    Args:
        close (pd.Series): Close prices.
        span (int, optional): EWM span in bars. Defaults to config.VOL_SPAN.

    Returns:
        pd.Series: Per-bar volatility, NaN during the warm-up period.
    """
    log_ret = np.log(close).diff()
    return log_ret.ewm(span=span, min_periods=span).std().rename("bar_vol")


def barrier_width(vol: pd.Series, mult: float = config.BARRIER_MULT, horizon: int = config.VERTICAL_BARS) -> pd.Series:
    """
    Horizontal barrier half-width as a fraction of the entry price.

    López de Prado sizes barriers with daily σ because his vertical barrier is
    measured in days. Here the vertical barrier is `horizon` intraday bars, so
    σ per bar is scaled by √horizon to express volatility over the holding period.

    Args:
        vol (pd.Series): Per-bar volatility (see bar_volatility).
        mult (float, optional): Barrier multiple of horizon σ. Defaults to config.BARRIER_MULT.
        horizon (int, optional): Vertical barrier in bars. Defaults to config.VERTICAL_BARS.

    Returns:
        pd.Series: Barrier width (e.g. 0.002 = ±0.2 %).
    """
    return (mult * vol * np.sqrt(horizon)).rename("width")


def cusum_events(close: pd.Series, threshold: pd.Series) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter (López de Prado 2018, §2.5.2.1) on log returns.

    An event is emitted at bar t when the cumulative up- or down-move since the
    last reset exceeds threshold[t]. Only returns up to bar t are used, so an
    event at t is known at the close of t and can be traded at the open of t+1.

    Args:
        close (pd.Series): Close prices.
        threshold (pd.Series): Per-bar threshold in log-return units, aligned to close.

    Returns:
        pd.DatetimeIndex: Event timestamps.
    """
    ret = np.log(close).diff().to_numpy()
    h = threshold.reindex(close.index).to_numpy()
    events = []
    s_pos = s_neg = 0.0
    for t in range(1, len(ret)):
        if np.isnan(h[t]) or np.isnan(ret[t]):
            continue
        s_pos = max(0.0, s_pos + ret[t])
        s_neg = min(0.0, s_neg + ret[t])
        if s_pos > h[t]:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -h[t]:
            s_neg = 0.0
            events.append(t)
    return close.index[events]


def barrier_exits(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    width: pd.Series,
    side: pd.Series | None = None,
    vertical_bars: int = config.VERTICAL_BARS,
) -> pd.DataFrame:
    """
    First-touch triple-barrier outcome for each event.

    Execution model (shared by labeling and the backtest):
      • Event at bar t is known at close[t]; the position is entered at open[t+1].
      • Upper / lower barrier = entry x (1 ± width[t]); touches are detected with
        HIGH / LOW from the entry bar onward. A bar that opens beyond a barrier
        exits at that open (gap fill).
      • Vertical barrier = close of the `vertical_bars`-th held bar, truncated at
        the close of the entry session — positions are never held overnight.
      • If both barriers are touched inside the same bar the order is unknown:
        with a known side the adverse barrier is assumed (conservative); without
        a side the bar's close decides the outcome.

    Events whose entry falls in the next session, or whose holding window runs
    past the end of the data, are dropped.

    Args:
        df (pd.DataFrame): OHLC data with a DatetimeIndex.
        events (pd.DatetimeIndex): Event timestamps (subset of df.index).
        width (pd.Series): Barrier half-width per bar (see barrier_width).
        side (pd.Series | None, optional): Position side {-1, +1} per event. Defaults to None.
        vertical_bars (int, optional): Maximum holding period in bars. Defaults to config.VERTICAL_BARS.

    Returns:
        pd.DataFrame: Indexed by event time with columns
            t1        : exit timestamp
            entry_pos : entry bar position in df (t + 1)
            exit_pos  : exit bar position in df
            entry_px  : fill price at open[t+1]
            exit_px   : barrier / vertical exit price
            ret       : exit_px / entry_px - 1 (long perspective, gross of costs)
            label     : sign(ret) ∈ {-1, 0, 1}
            barrier   : "upper", "lower", "tie" (unknown order, no side) or "vertical"
            width     : barrier half-width used
    """
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    day = df.index.normalize()
    session_last = pd.Series(np.arange(n), index=df.index).groupby(day).transform("max").to_numpy()

    t = df.index.get_indexer(events)
    w = width.reindex(events).to_numpy()
    sd = np.zeros(len(t)) if side is None else side.reindex(events).to_numpy(dtype=float)
    e = t + 1

    keep = (t >= 0) & (e < n) & ~np.isnan(w) & ~np.isnan(sd)
    t, e, w, sd = t[keep], e[keep], w[keep], sd[keep]

    # Last held bar: vertical barrier or session close, whichever comes first.
    # The final session may be cut off by the end of the data, so a window that
    # would run past it is incomplete rather than truncated at a real close.
    last = np.minimum(e + vertical_bars - 1, session_last[e])
    keep = (session_last[t] == session_last[e]) & ((last < n - 1) | (e + vertical_bars - 1 <= n - 1))
    t, e, w, sd, last = t[keep], e[keep], w[keep], sd[keep], last[keep]

    entry = open_[e]
    ub = entry * (1.0 + w)
    lb = entry * (1.0 - w)

    # (event, k) → bar e + k for k in [0, vertical_bars); bars past `last` are masked
    idx = np.minimum(e[:, None] + np.arange(vertical_bars), n - 1)
    held = idx <= last[:, None]
    hit_up = (high[idx] >= ub[:, None]) & held
    hit_dn = (low[idx] <= lb[:, None]) & held
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), vertical_bars)
    first_dn = np.where(hit_dn.any(axis=1), hit_dn.argmax(axis=1), vertical_bars)
    first = np.minimum(first_up, first_dn)
    touched = first < vertical_bars
    tie = touched & (first_up == first_dn)

    exit_pos = np.where(touched, e + first, last)
    xp = np.minimum(exit_pos, n - 1)
    up_px = np.maximum(ub, open_[xp])  # gap above the upper barrier fills at the open
    dn_px = np.minimum(lb, open_[xp])  # gap below the lower barrier fills at the open

    goes_up = touched & (first_up < first_dn)
    goes_dn = touched & (first_dn < first_up)
    # Same-bar ties: adverse barrier for a known side, the bar's close otherwise
    goes_up |= tie & (sd < 0)
    goes_dn |= tie & (sd > 0)
    tie_close = tie & (sd == 0)

    exit_px = np.select([goes_up, goes_dn, tie_close], [up_px, dn_px, close[xp]], default=close[xp])
    barrier = np.select([goes_up, goes_dn, tie_close], ["upper", "lower", "tie"], default="vertical")
    ret = exit_px / entry - 1.0

    return pd.DataFrame(
        {
            "t1": df.index[xp],
            "entry_pos": e,
            "exit_pos": xp,
            "entry_px": entry,
            "exit_px": exit_px,
            "ret": ret,
            "label": np.sign(ret).astype(int),
            "barrier": barrier,
            "width": w,
        },
        index=df.index[t],
    )


def sample_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tradeable events and their barrier widths, using only data up to each event.

    σ per bar = causal EWM std of log returns; events come from a symmetric CUSUM
    filter with threshold CUSUM_MULT x σ; barriers are ±BARRIER_MULT x σ x √VERTICAL_BARS.
    Events on a session's last bar are skipped: their entry would be next session.

    Args:
        df (pd.DataFrame): OHLC data with a DatetimeIndex.

    Returns:
        pd.DataFrame: Indexed by event time with column `width`.
    """
    vol = bar_volatility(df["close"])
    events = cusum_events(df["close"], config.CUSUM_MULT * vol)
    session_end = pd.Series(df.index.normalize(), index=df.index).shift(-1) != df.index.normalize()
    events = events[~session_end.loc[events].to_numpy()]
    return barrier_width(vol).loc[events].to_frame()


def triple_barrier_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    López de Prado (2018) triple-barrier labels: first-touch outcome of each
    event without a side (see barrier_exits), label = sign(return).

    These are TARGETS — they use prices after the event and must only be read
    for samples whose exit falls inside the split they are fitted on.

    Args:
        df (pd.DataFrame): OHLC data with a DatetimeIndex.
        events (pd.DataFrame): Output of sample_events.

    Returns:
        pd.DataFrame: One row per event with a complete outcome (see barrier_exits for columns).
    """
    labels = barrier_exits(df, events.index, events["width"])

    counts = labels["barrier"].value_counts().to_dict()
    print(
        f"[LABEL] {len(labels):,} events from {len(df):,} bars  |  "
        f"-1={(labels['label'] == -1).sum()}  0={(labels['label'] == 0).sum()}  1={(labels['label'] == 1).sum()}  |  "
        f"barriers {counts}"
    )
    return labels


def average_uniqueness(labels: pd.DataFrame, n_bars: int) -> pd.Series:
    """
    Average uniqueness of each label (López de Prado 2018, §4.4), used as sample weight.

    Overlapping holding windows share information; a label whose bars are held
    by c other labels at once gets weight mean(1/c) over its window. Compute this
    on the samples actually used for fitting so the weights see no other split.

    Args:
        labels (pd.DataFrame): Label rows with `entry_pos` and `exit_pos`.
        n_bars (int): Length of the underlying bar series.

    Returns:
        pd.Series: Weights in (0, 1], same index as labels.
    """
    start = labels["entry_pos"].to_numpy()
    end = labels["exit_pos"].to_numpy()
    delta = np.zeros(n_bars + 1)
    np.add.at(delta, start, 1)
    np.add.at(delta, end + 1, -1)
    conc = np.cumsum(delta[:-1])
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    return pd.Series((cum[end + 1] - cum[start]) / (end - start + 1), index=labels.index, name="weight")
