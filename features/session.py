"""
Session state for intraday bars (SPEC §14–§15, U14): the `session` feature group and the frame the schedule
sampler's `gate` is evaluated on.

Every column at bar t uses bars <= t only. Previous sessions are complete by then; the current session is read only
up to t. Bar stamps are bar opens in NY time (a naive index is NY time). The session close is taken as 16:00, the same
convention as the `intraday` group: early closes are not known from the bars alone.

Columns (session_frame; the group exposes the first nine):
    gap_sigma       log(open_d / close_{d−1}) / σ_overnight (σ of earlier gaps only)
    prev_cc         log(close_{d−1} / close_{d−2})
    prev_oc         log(close_{d−1} / open_{d−1})
    open_to_now     log(close_t / open_d)
    first30         log(close_k / open_d), k = min(t, the session's last bar closing at or before 10:00; its first
                    bar when none does, e.g. 1Hour): the first-30-minute return, the return so far before 10:00
    mins_to_close   minutes from the bar's close to 16:00
    activity        IAOM of the bar's slot: DC events per session in that slot over the fold's train window (only
                    with a fitted state)
    rvol            session cumulative volume / the median of the previous RVOL_SESSIONS sessions' cumulative
                    volume at the same slot
    dc_overshoot    overshoot of the last completed directional-change run, in δ units (features.events.dc_events)
    sigma_bar       σ_t (bar_volatility with the fold's profile)
    sigma_day       session σ = deseasonalized σ at the previous session's last bar · √Σ_b s(b)² (σ_bar · √n_slots
                    of the previous close without a profile)
    sigma_overnight σ of the close-to-open gaps before today (EWM over VOL_SPAN sessions)
"""

import numpy as np
import pandas as pd

from features.vol_profile import (
    VolProfile,
    bar_slots,
    bar_volatility,
    base_volatility,
    isom_counts,
    n_slots,
    ny_dates,
    ny_minutes,
)

FIRST30_END = 10 * 60
CLOSE_MINUTE = 16 * 60
RVOL_SESSIONS = 20
RVOL_MIN_SESSIONS = 5
OVERNIGHT_MIN_SESSIONS = 20
GROUP_COLUMNS = ("gap_sigma", "prev_cc", "prev_oc", "open_to_now", "first30", "mins_to_close", "activity", "rvol",
                 "dc_overshoot")  # fmt: skip


def _bar_minutes(cfg) -> int:
    from data.timeframes import get_timeframe

    m = get_timeframe(cfg.TIMEFRAME).minutes
    if m is None:
        raise ValueError("session features are intraday only")
    return m


def _lagged(per_session: np.ndarray, sess: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(sess), np.nan)
    ok = sess >= lag
    out[ok] = per_session[sess[ok] - lag]
    return out


def session_frame(
    df: pd.DataFrame, cfg, profile: VolProfile | None = None, iaom: np.ndarray | None = None
) -> pd.DataFrame:
    """The session state at every bar (see the module docstring); `activity` only when `iaom` is given."""
    from features.events import dc_events, dc_mult

    m = _bar_minutes(cfg)
    idx = df.index
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    day = ny_dates(idx)
    mins = ny_minutes(idx)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    sess = np.cumsum(first) - 1
    first_pos, last_pos = np.flatnonzero(first), np.flatnonzero(last)
    o_s, c_s = o[first_pos], c[last_pos]  # c_s of the current (possibly cut) session is never read

    open_d = o_s[sess]
    prev_c, prev2_c, prev_o = _lagged(c_s, sess, 1), _lagged(c_s, sess, 2), _lagged(o_s, sess, 1)
    gap_s = np.log(o_s / np.r_[np.nan, c_s[:-1]])  # gap of session k (needs the k−1 close)
    sig_on_s = pd.Series(gap_s).ewm(span=cfg.VOL_SPAN, min_periods=OVERNIGHT_MIN_SESSIONS).std().shift(1).to_numpy()
    sig_on = sig_on_s[sess]

    # first-30-minute reference bar: the last bar of the session closing by 10:00, else its first bar
    pos = np.arange(n)
    early = np.where(mins + m <= FIRST30_END, pos, -1)
    kstar = np.maximum(pd.Series(early).groupby(sess).transform("max").to_numpy(), first_pos[sess])
    ref = np.minimum(pos, kstar)

    # relative volume: cumulative session volume vs the previous sessions' cumulative volume at the same slot
    slot = bar_slots(idx, m)
    cumv = pd.Series(v).groupby(sess).cumsum().to_numpy()
    grid = pd.DataFrame({"s": sess, "b": slot, "v": cumv}).pivot_table(index="s", columns="b", values="v")
    hist = grid.rolling(RVOL_SESSIONS, min_periods=RVOL_MIN_SESSIONS).median().shift(1)
    hist = hist.reindex(index=range(len(first_pos)), columns=range(n_slots(m)))
    ref_v = hist.to_numpy()[sess, slot]
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = np.where(ref_v > 0, cumv / ref_v, np.nan)

    close = df["close"].astype(float)
    sigma = bar_volatility(close, cfg.VOL_SPAN, profile)
    if profile is None:
        base, scale = sigma.to_numpy(), np.sqrt(n_slots(m))
    else:
        base, scale = base_volatility(close, cfg.VOL_SPAN, profile).to_numpy(), profile.session_scale()
    sigma_day = _lagged(base[last_pos], sess, 1) * scale
    overshoot = dc_events(close, dc_mult(cfg) * sigma)[1].to_numpy()

    cols = {
        "gap_sigma": np.log(open_d / prev_c) / sig_on,
        "prev_cc": np.log(prev_c / prev2_c),
        "prev_oc": np.log(prev_c / prev_o),
        "open_to_now": np.log(c / open_d),
        "first30": np.log(c[ref] / open_d),
        "mins_to_close": np.maximum(CLOSE_MINUTE - (mins + m), 0).astype(float),
    }
    if iaom is not None:
        cols["activity"] = np.asarray(iaom, dtype=float)[slot]
    cols |= {
        "rvol": rvol,
        "dc_overshoot": overshoot,
        "sigma_bar": sigma.to_numpy(),
        "sigma_day": sigma_day,
        "sigma_overnight": sig_on,
    }
    return pd.DataFrame(cols, index=idx)


def train_iaom(train_df: pd.DataFrame, cfg) -> np.ndarray:
    """
    IAOM over a train window: DC events (δ = dc_mult · plain σ_t, so the raw time-of-day seasonality shows) per slot,
    divided by the window's sessions. Plain σ rather than the fold's profile: with a profile the thresholds scale
    with s(b) and the event rate flattens by design.
    """
    from features.events import dc_events, dc_mult

    m = _bar_minutes(cfg)
    close = train_df["close"].astype(float)
    events = dc_events(close, dc_mult(cfg) * bar_volatility(close, cfg.VOL_SPAN))[0]
    sessions = len(np.unique(ny_dates(train_df.index)))
    return isom_counts(events, m) / max(sessions, 1)
