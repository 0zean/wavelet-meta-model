"""
Intraday estimator kernels (SPEC §23, U23; PLAN3 §3 G1): every estimator of the region primary for every lookback in
one pass over a 5Min series, with session-bounded windows.

Conventions (numba, structure of arrays, float64 throughout, no `fastmath`):
- Bars are stamped at their open in NY time (a naive index is NY time); a session is a NY date of the data.
- A window never holds bars of two sessions (the RMV repo's gap-contamination finding): at the p-th bar of a session
  (p = 0 the first) an N-bar estimator is NaN for p < N − 1 and reads bars p − N + 1 … p of that session only. The
  first bar's close is in the window; the previous session's close is not.
- Velocities are slopes of LOG close per bar. Storage is float64 (PLAN3 §6 allowed float32 storage; at 190k bars × 5
  lookbacks the saving is 4 MB, and float64 keeps a threshold comparison off the rounding boundary the RMV repo
  documents for float32).

    rmedv_all(x, start, ns)            Siegel's repeated-median slope (hierarchical; scipy.stats.siegelslopes)
    sg_velocity_all(x, start, ns, d)   Meyers' next-bar velocity of a degree-d least-squares polynomial (Savitzky–Golay)
    rmedv_normalized(raw, ns, sess, k) RMedV · √N · xmult (Meyers 2025), xmult refit over the k previous sessions
    poly_normalized(raw, sess, k)      velocity / its SD per (degree, N) (Meyers 2026), refit over the k previous sessions
    prior_sd(values, sess, k)          SD of each row's values over the k previous sessions
    band_state(o, c, sess, slot, L)    Zarattini band: (gap-adjusted distance in band units, band)
    session_vwap(c, v, sess)           cumulative Σ close·volume / Σ volume within the session
    session_layout(index, minutes)     (sess, start, pos, slot) of every bar

Every output at bar t uses bars ≤ t only (SPEC §8): the session open, the previous session's close and the previous
sessions' moves are known at the open; the current session is read up to t.
"""

import numba
import numpy as np
import pandas as pd

from features.vol_profile import bar_slots, ny_dates


def session_layout(index: pd.DatetimeIndex, bar_minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per bar: session number `sess` (0, 1, … in data order), the position `start` of its session's first bar, its
    position `pos` within the session and its time-of-day `slot` (features.vol_profile.bar_slots)."""
    n = len(index)
    day = ny_dates(index)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    sess = np.cumsum(first) - 1
    firsts = np.flatnonzero(first)
    start = firsts[sess] if n else np.zeros(0, np.int64)
    return sess.astype(np.int64), start.astype(np.int64), np.arange(n) - start, bar_slots(index, bar_minutes)


def _check_ns(ns, lo: int) -> np.ndarray:
    ns = np.asarray(ns)
    if ns.ndim != 1 or ns.size == 0 or not np.all(ns == np.floor(ns)) or np.any(ns < lo):
        raise ValueError(f"lookbacks must be a non-empty list of integers >= {lo}, got {ns.tolist()}")
    return ns.astype(np.int64)


def _check_x(x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"the series must be 1-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        # numba's np.median does not propagate NaN: a NaN inside a window would come back as a finite slope
        raise ValueError("the series contains NaN or inf: a window over it would return a finite lie")
    return x


# ── Repeated-median velocity ─────────────────────────────────────────────────


@numba.njit(cache=True)
def _median(buf: np.ndarray, m: int) -> float:
    """Median of buf[:m] (sorted in place; insertion sort, m ≤ the largest lookback); numpy's mean of the two
    middle values for an even m."""
    for i in range(1, m):
        v = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > v:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = v
    h = m // 2
    if m % 2 == 1:
        return buf[h]
    return (buf[h - 1] + buf[h]) / 2.0


@numba.njit(cache=True)
def _rmedv_kernel(x: np.ndarray, start: np.ndarray, ns: np.ndarray, out: np.ndarray) -> None:
    total = x.shape[0]
    nmax = 0
    for a in range(ns.shape[0]):
        nmax = max(nmax, ns[a])
    pairs = np.empty(nmax)
    inner = np.empty(nmax)
    for a in range(ns.shape[0]):
        n = ns[a]
        for t in range(total):
            if t - start[t] < n - 1:
                out[a, t] = np.nan
                continue
            b = t - n + 1
            for i in range(n):
                k = 0
                for j in range(n):
                    if j != i:
                        pairs[k] = (x[b + j] - x[b + i]) / (j - i)
                        k += 1
                inner[i] = _median(pairs, k)
            out[a, t] = _median(inner, n)


def rmedv_all(x, start, ns) -> np.ndarray:
    """
    Siegel's (1982) repeated-median slope of x over the last N bars of the session, for every N in `ns`: for each
    point the median of its N − 1 pairwise slopes, then the median of those N medians (scipy.stats.siegelslopes,
    method "hierarchical"). float64[len(ns), len(x)], NaN for the first N − 1 bars of every session.

    Args:
        x: the series (log close), finite.
        start: per bar, the position of its session's first bar (session_layout).
        ns: lookbacks (integers >= 3).
    """
    x = _check_x(x)
    ns = _check_ns(ns, 3)
    out = np.empty((len(ns), len(x)))
    _rmedv_kernel(x, np.ascontiguousarray(start, dtype=np.int64), ns, out)
    return out


# ── Polynomial (Savitzky–Golay) velocity ─────────────────────────────────────


def sg_weights(n: int, degree: int, offset: int = 1) -> np.ndarray:
    """
    FIR weights w (length n, oldest first) of the velocity of the least-squares polynomial of `degree` through the last
    n points, evaluated `offset` bars after the last point: velocity = Σ_k w_k x_{t−n+1+k}. offset 1 is Meyers' nth-order
    fixed-memory polynomial velocity, the derivative of the next bar's forecast, Velocity(T+1) (Meyers 2026, Appendix I;
    Morrison 1969 ch. 7); offset 0 is the derivative at the last point. Degree 1 is the least-squares slope at any offset.
    Fit on centred, scaled abscissae (conditioning); Σ w = 0 to rounding.
    """
    if not (isinstance(degree, int | np.integer) and 1 <= degree < n):
        raise ValueError(f"degree must be an integer in [1, n), got {degree} for n={n}")
    u = np.arange(n, dtype=float)
    c, h = (n - 1) / 2.0, (n - 1) / 2.0
    s = (u - c) / h  # in [−1, 1]; the last point is s = 1
    A = np.vander(s, degree + 1, increasing=True)
    coef = np.linalg.pinv(A)  # row j: the weights of the s**j coefficient
    s0 = (n - 1 + offset - c) / h
    j = np.arange(1, degree + 1)
    # d/du Σ_j a_j s**j at s0 = Σ_j j a_j s0**(j − 1) / h
    return ((j * s0 ** (j - 1))[:, None] * coef[1:]).sum(axis=0) / h


@numba.njit(cache=True)
def _fir_kernel(x: np.ndarray, start: np.ndarray, ns: np.ndarray, w: np.ndarray, out: np.ndarray) -> None:
    total = x.shape[0]
    for a in range(ns.shape[0]):
        n = ns[a]
        for t in range(total):
            if t - start[t] < n - 1:
                out[a, t] = np.nan
                continue
            b = t - n + 1
            ref = x[t]
            acc = 0.0
            for k in range(n):
                acc += w[a, k] * (x[b + k] - ref)  # Σ w = 0: differencing first keeps the sum at the slope's scale
            out[a, t] = acc


def sg_velocity_all(x, start, ns, degree: int = 1, offset: int = 1) -> np.ndarray:
    """The polynomial velocity of a degree-`degree` fit over the last N bars of the session, `offset` bars past the
    last (sg_weights; 1 = Meyers' next-bar velocity), for every N in `ns`: float64[len(ns), len(x)], NaN for the first
    N − 1 bars of every session."""
    x = _check_x(x)
    ns = _check_ns(ns, 2)
    w = np.zeros((len(ns), int(ns.max())))
    for a, n in enumerate(ns):
        w[a, :n] = sg_weights(int(n), degree, offset)
    out = np.empty((len(ns), len(x)))
    _fir_kernel(x, np.ascontiguousarray(start, dtype=np.int64), ns, w, out)
    return out


# ── Normalization (Meyers' multipliers, refit every session) ─────────────────


@numba.njit(cache=True)
def _prior_sd_kernel(v: np.ndarray, firsts: np.ndarray, k: int, ddof: int, out: np.ndarray) -> None:
    n_sess = firsts.shape[0] - 1  # firsts holds a sentinel at the end
    for a in range(v.shape[0]):
        for d in range(n_sess):
            out[a, d] = np.nan
            if d < k:
                continue
            lo, hi = firsts[d - k], firsts[d]
            m = 0
            acc = 0.0
            for t in range(lo, hi):
                if v[a, t] == v[a, t]:
                    acc += v[a, t]
                    m += 1
            if m <= ddof + 1:
                continue
            mean = acc / m
            ss = 0.0
            for t in range(lo, hi):
                if v[a, t] == v[a, t]:
                    dv = v[a, t] - mean
                    ss += dv * dv
            out[a, d] = np.sqrt(ss / (m - ddof))


def prior_sd(values, sess, k: int, ddof: int = 1) -> np.ndarray:
    """
    Per row of `values` (float64[rows, bars]; NaN = undefined, skipped) and per bar: the standard deviation of the
    row's defined values over the k sessions before the bar's session (NaN for the first k sessions or with fewer than
    ddof + 2 values). Known at the session's open; never reads the current session.
    """
    v = np.ascontiguousarray(np.atleast_2d(values), dtype=np.float64)
    sess = np.ascontiguousarray(sess, dtype=np.int64)
    if not (isinstance(k, int | np.integer) and k >= 1):
        raise ValueError(f"k must be an integer >= 1, got {k}")
    if v.shape[1] != len(sess):
        raise ValueError(f"values have {v.shape[1]} bars, sess {len(sess)}")
    firsts = np.r_[np.flatnonzero(np.r_[True, sess[1:] != sess[:-1]]), len(sess)] if len(sess) else np.zeros(1, int)
    per = np.empty((v.shape[0], len(firsts) - 1))
    _prior_sd_kernel(v, firsts.astype(np.int64), int(k), int(ddof), per)
    return per[:, sess] if len(sess) else np.zeros((v.shape[0], 0))


def rmedv_normalized(raw, ns, sess, k: int) -> np.ndarray:
    """
    Meyers' RMedV normalization (Meyers 2025, Appendix III; the RMV repo's §1.2): RMedV_N · √N · xmult, where
    xmult = mean over the N in `ns` of 1 / sd(RMedV_N · √N). sd(RMedV_N) falls about as 1 / √N, so √N equalizes the
    lookbacks and the one scalar xmult brings every N to unit standard deviation; thresholds are then in SD units.
    Meyers calibrates xmult once on a long sample; a frozen scale fails across volatility regimes (the RMV repo: a 6.45×
    swing), so it is refit at every session over the k previous sessions (`prior_sd`; the RMV repo refit it per
    21-session window).
    """
    ns = np.asarray(ns, dtype=np.int64)
    scaled = np.asarray(raw, dtype=np.float64) * np.sqrt(ns)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xmult = np.mean(1.0 / prior_sd(scaled, sess, k), axis=0)
        return scaled * xmult[None, :]


def poly_normalized(raw, sess, k: int) -> np.ndarray:
    """
    Meyers' polynomial-velocity normalization (Meyers 2026, Appendix III, "The Normalization Multiplier"): the velocity
    times Mult = 1 / SD for its own (degree, N), so every lookback and degree has unit standard deviation (his
    multiplier is a surface fitted to that SD table; there is no √N). Refit at every session over the k previous
    sessions, as rmedv_normalized.
    """
    raw = np.asarray(raw, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return raw / prior_sd(raw, sess, k)


# ── Noise band (Zarattini, Aziz & Barbon) and VWAP ───────────────────────────


def band_state(open_, close, sess, slot, lookback: int = 14, min_count: int | None = None):
    """
    The time-of-day noise band and the gap-adjusted distance from it, per bar.

    band(d, b) = mean over the `lookback` sessions before d of |close at slot b / that session's open − 1| (sessions
    without a bar at slot b skipped; NaN with fewer than `min_count` values, default lookback // 2). With
    hi = max(open_d, close_{d−1}) and lo = min(open_d, close_{d−1}) (the gap adjustment), the distance is
    (close / hi − 1) / band above hi, (close / lo − 1) / band below lo, else 0: a long breakout of the VM-band
    (close > hi · (1 + VM · band)) is distance > VM. NaN without a band or a previous close (the first session).

    Returns:
        (distance, band): float64 arrays aligned to the bars.
    """
    o = np.asarray(open_, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    sess = np.asarray(sess, dtype=np.int64)
    slot = np.asarray(slot, dtype=np.int64)
    min_count = lookback // 2 if min_count is None else min_count
    if not (isinstance(lookback, int | np.integer) and lookback >= 1 and 1 <= min_count <= lookback):
        raise ValueError(f"lookback must be >= 1 and min_count in [1, lookback], got {lookback} / {min_count}")
    n = len(c)
    first = np.r_[True, sess[1:] != sess[:-1]] if n else np.zeros(0, bool)
    last = np.r_[sess[1:] != sess[:-1], True] if n else np.zeros(0, bool)
    o_s, c_s = o[first], c[last]
    n_sess, k = len(o_s), int(slot.max()) + 1 if n else 0
    move = np.full((n_sess, k), np.nan)
    move[sess, slot] = np.abs(c / o_s[sess] - 1.0)
    have = ~np.isnan(move)
    # trailing sums over the `lookback` previous sessions (row d reads rows d − lookback … d − 1)
    s = np.vstack([np.zeros((1, k)), np.cumsum(np.where(have, move, 0.0), axis=0)])
    m = np.vstack([np.zeros((1, k), int), np.cumsum(have, axis=0)])
    d = np.arange(n_sess)
    lo_row = np.maximum(d - lookback, 0)
    tot, cnt = s[d] - s[lo_row], m[d] - m[lo_row]
    full = (d >= lookback)[:, None] & (cnt >= min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        band_s = np.where(full, tot / cnt, np.nan)
    band = band_s[sess, slot]
    prev_c = np.r_[np.nan, c_s[:-1]][sess]
    hi, lo = np.maximum(o_s[sess], prev_c), np.minimum(o_s[sess], prev_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(c > hi, (c / hi - 1.0) / band, np.where(c < lo, (c / lo - 1.0) / band, 0.0))
    dist = np.where(np.isnan(band) | np.isnan(prev_c), np.nan, dist)
    return dist, band


def session_vwap(close, volume, sess) -> np.ndarray:
    """Cumulative Σ close·volume / Σ volume from the session's first bar to t (NaN while the session has traded no
    volume)."""
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    pv = pd.Series(c * v).groupby(np.asarray(sess)).cumsum().to_numpy()
    cv = pd.Series(v).groupby(np.asarray(sess)).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cv > 0, pv / cv, np.nan)
