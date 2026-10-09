"""
Intraday volatility structure (SPEC §14, U14; the ISOM idea done properly).

Slot b = the bar's session-relative index from its open stamp: (minutes since 09:30) // bar length, so 13 slots at
30Min, 26 at 15Min, 78 at 5Min and 7 at 1Hour (the 15:30 stub is its own slot). Bars are stamped at their open in NY
time; a naive index is read as NY time.

`VolProfile.fit(train_df)` is per-fold state (fit on train bars before the embargo, like fracdiff):
    s(b) = median over sessions of |r_b| / median over slots of that median,
smoothed by a 3-slot moving median (the two edge slots keep their own value) and floored at PROFILE_FLOOR.
r = 1-bar log close return, the same return bar_volatility uses (slot 0 carries the overnight gap).

`bar_volatility(close, span, profile)`: σ_t = EWM σ(span) of the deseasonalized returns r / s(b) up to t, times s(b_t).
Without a profile it is the plain EWM σ of §3; with a flat profile (s ≡ 1) it is bit-identical to it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PROFILE_FLOOR = 0.25
SESSION_MINUTES = 390  # 09:30–16:00; early closes simply have fewer slots
OPEN_MINUTE = 9 * 60 + 30


def ny_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since midnight of each bar's open stamp in NY time (a naive index is NY time)."""
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    return np.asarray(local.hour * 60 + local.minute, dtype=np.int64)


def ny_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """Session date (NY) of each bar as datetime64[D]."""
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    return np.asarray(local.tz_localize(None).normalize() if local.tz is not None else local.normalize(), "M8[D]")


def n_slots(bar_minutes: int) -> int:
    return -(-SESSION_MINUTES // bar_minutes)


def bar_slots(index: pd.DatetimeIndex, bar_minutes: int) -> np.ndarray:
    """Session-relative slot of each bar, clipped to [0, n_slots − 1]."""
    return np.clip((ny_minutes(index) - OPEN_MINUTE) // bar_minutes, 0, n_slots(bar_minutes) - 1)


def _moving_median3(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    if len(x) >= 3:
        out[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return out


@dataclass(frozen=True)
class VolProfile:
    """
    Multiplicative time-of-day volatility factor s(b) per slot (see module docstring). `s_open` is the first bar's
    factor without the overnight gap (median |log(close / open)| of slot-0 bars on the same normalisation, floored):
    the move a position entered at the session's opening print holds through that bar (hold_scale).
    """

    s: np.ndarray
    bar_minutes: int
    s_open: float = 1.0

    @classmethod
    def flat(cls, bar_minutes: int) -> "VolProfile":
        return cls(np.ones(n_slots(bar_minutes)), bar_minutes)

    @classmethod
    def fit(cls, train_df: pd.DataFrame, bar_minutes: int) -> "VolProfile":
        """Fit s(b) on train bars (median over sessions of |r_b|, normalised by the median slot, smoothed, floored)."""
        k = n_slots(bar_minutes)
        r = np.abs(np.log(train_df["close"].astype(float)).diff().to_numpy())
        slot = bar_slots(train_df.index, bar_minutes)
        ok = ~np.isnan(r)
        med = pd.Series(r[ok]).groupby(slot[ok]).median().reindex(range(k)).to_numpy()
        if np.isnan(med).any() or not np.nanmedian(med) > 0:
            missing = np.flatnonzero(np.isnan(med)).tolist()
            raise ValueError(f"VolProfile.fit: train bars cover no return in slots {missing} (or all moves are 0)")
        s = _moving_median3(med / np.median(med))
        first = slot == 0
        intra = np.abs(np.log(train_df["close"].to_numpy(float)[first] / train_df["open"].to_numpy(float)[first]))
        s_open = max(float(np.nanmedian(intra)) / float(np.median(med)), PROFILE_FLOOR)
        return cls(np.maximum(s, PROFILE_FLOOR), bar_minutes, s_open)

    def factor(self, index: pd.DatetimeIndex) -> np.ndarray:
        """s(b_t) for each bar of `index`."""
        return self.s[bar_slots(index, self.bar_minutes)]

    def transform(self, r: pd.Series) -> pd.Series:
        """Deseasonalized returns r / s(b)."""
        return r / self.factor(r.index)

    def session_scale(self) -> float:
        """√Σ_b s(b)²: σ over a full session in units of the deseasonalized bar σ (√n_slots when flat)."""
        return float(np.sqrt(np.sum(self.s**2)))

    def to_dict(self) -> dict:
        return {"bar_minutes": self.bar_minutes, "s": [round(float(v), 6) for v in self.s],
                "s_open": round(self.s_open, 6)}  # fmt: skip


def bar_volatility(close: pd.Series, span: int, profile: VolProfile | None = None) -> pd.Series:
    """
    Causal EWM standard deviation of 1-bar log returns (σ per bar).

    σ[t] uses returns up to and including bar t, so it is known at the close of t. With a VolProfile (SPEC §14) the
    EWM runs over the deseasonalized returns r / s(b) and σ[t] is scaled back by s(b_t).

    Args:
        close (pd.Series): Close prices.
        span (int): EWM span in bars (cfg.VOL_SPAN).
        profile (VolProfile | None): Time-of-day profile (cfg.VOL_PROFILE="tod"); None = §3 behaviour.

    Returns:
        pd.Series: Per-bar volatility, NaN during the warm-up period.
    """
    log_ret = np.log(close).diff()
    if profile is None:
        return log_ret.ewm(span=span, min_periods=span).std().rename("bar_vol")
    s = profile.factor(close.index)
    base = (log_ret / s).ewm(span=span, min_periods=span).std()
    return (base * s).rename("bar_vol")


def base_volatility(close: pd.Series, span: int, profile: VolProfile) -> pd.Series:
    """EWM σ of the deseasonalized returns (σ per unit of s(b)), known at the close of t."""
    log_ret = np.log(close).diff()
    return (log_ret / profile.factor(close.index)).ewm(span=span, min_periods=span).std()


def hold_scale(index: pd.DatetimeIndex, profile: VolProfile, horizon: int, hold_overnight: bool) -> np.ndarray:
    """
    √Σ s(b)² over the slots a position entered after bar t holds: the `horizon` slots from the slot after b_t (the
    next session's first slot when t is the last slot). Without hold_overnight they stop at the session's last slot
    (16:00; early closes are not known from the stamps), as the vertical barrier does; with it they wrap into the
    next session (slot 0, which carries the overnight gap). A position entered at the opening print (the first held
    slot is 0) does not bear that gap: its first slot counts `s_open`. Flat profile, full hold: √horizon.
    """
    k = len(profile.s)
    start = (bar_slots(index, profile.bar_minutes) + 1) % k
    slots = start[:, None] + np.arange(horizon)
    s2 = np.broadcast_to(profile.s**2, (len(start), k)).copy()
    s2[start == 0, 0] = profile.s_open**2
    rows = np.arange(len(start))[:, None]
    if hold_overnight:
        held = s2[rows, slots % k]
        held[:, 1:] = np.where(slots[:, 1:] % k == 0, profile.s[0] ** 2, held[:, 1:])  # a wrapped slot 0: the gap
        return np.sqrt(held.sum(axis=1))
    return np.sqrt(np.where(slots < k, s2[rows, np.minimum(slots, k - 1)], 0.0).sum(axis=1))


def isom_counts(events: pd.DatetimeIndex, bar_minutes: int) -> np.ndarray:
    """Events per session slot (Kablan 2009's ISOM): an int array of length n_slots."""
    k = n_slots(bar_minutes)
    return np.bincount(bar_slots(pd.DatetimeIndex(events), bar_minutes), minlength=k)[:k]


def profile_for(cfg) -> int | None:
    """Bar minutes of an intraday cfg (None for 1Day)."""
    from data.timeframes import get_timeframe

    return get_timeframe(cfg.TIMEFRAME).minutes


def fit_profile(train_df: pd.DataFrame, cfg) -> VolProfile | None:
    """The fold's VolProfile for cfg.VOL_PROFILE ("tod": fit on train_df; "none": None)."""
    if cfg.VOL_PROFILE == "none":
        return None
    return VolProfile.fit(train_df, profile_for(cfg))
