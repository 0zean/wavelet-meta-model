"""
Time–frequency state (SPEC §24, U26; PLAN3 §3 G4): per session, known at its open, from the 5Min log returns of the
previous sessions. A state, never a forecast (PLAN3 §1.1 closed the cycle forecaster): reported as splits of G1 and kept
as inputs for a later overlay (U31); it cannot be selected on.

Returns: the close → close 5Min log returns inside each session (the overnight gap and the first bar's open → close
excluded, as features.kernels.session_sigma), concatenated over sessions in data order. Session d's window is the
returns of sessions d − TF_SESSIONS … d − 1 (M of them, demeaned; NaN for the first TF_SESSIONS sessions or with fewer
than TF_MIN_RETURNS).

    pow_hf     mean periodogram over the Fourier bins k / M with period 2–6 bars, over the mean over every bin
               k = 1 … ⌊(M − 1) / 2⌋ (white noise: 1); the reversal / bid-ask bounce band
    pow_lf     the same over periods 12–78 bars (one hour to one session); the continuation band
    tilt       log(pow_hf / pow_lf): > 0 reversal-dominated, < 0 continuation-dominated microstructure
    flatness   spectral flatness (Wiener entropy), geometric / arithmetic mean of the periodogram over every bin; white
               noise ≈ e^−γ ≈ 0.56, a peaked spectrum → 0
    modwt_e1 … modwt_e5   log(2^j · ν̂_j / σ̂²): the Haar MODWT wavelet variance at level j (scales 2^(j−1) … 2^j bars,
               features.groups.causal_modwt on the concatenated returns, so every coefficient reads past returns only)
               over the window, relative to white noise (ν_j = σ² / 2^j for any MODWT filter); > 0 = more variance at
               that scale than a random walk has
    ac1        the lag-1 autocorrelation of the window's returns (features.kernels.prior_lag1_autocorr over the same
               sessions; PLAN3 §3 G3's revealed-gamma proxy: < 0 dealers long gamma, ≥ 0 short gamma)
    vov        realized vol-of-vol: the SD of log session σ (features.kernels.session_sigma) over the VOV_SESSIONS
               previous sessions (NaN with fewer than half of them)

The periodogram is Goertzel's (fastgoertzel, C++): I_k = |Σ_t x_t e^(−2πi k t / M)|² / M. fastgoertzel 0.1.9 returns
(amplitude 2|X| / M, phase) of the Fourier bin int(f · M): the fractional part of f · M is dropped, so f = k / M
itself lands on bin k − 1 whenever k / M · M rounds below k (18 of the 192 bins of a 385-return window). It is
therefore called at the bin's centre f = (k + 0.5) / M (`goertzel_power`; parity with a numpy DFT to 1e-10 in
tests/test_u26.py). 0.1.9 is pinned: fastgoertzel 1.x requires numpy ≥ 2.2 and the repo's bit-exact pins run on
numpy 1.26.
"""

import fastgoertzel
import numpy as np
import pandas as pd

from features.kernels import prior_lag1_autocorr, session_layout, session_sigma

TF_SESSIONS = 5
TF_MIN_RETURNS = 100
HF_PERIODS = (2, 6)  # bars, inclusive
LF_PERIODS = (12, 78)
MODWT_LEVELS = 5
MODWT_FILTER = "db1"
VOV_SESSIONS = 21
COLUMNS = ("pow_hf", "pow_lf", "tilt", "flatness", *(f"modwt_e{j}" for j in range(1, MODWT_LEVELS + 1)), "ac1", "vov")


def goertzel_power(x: np.ndarray, k: int) -> float:
    """The periodogram ordinate |X_k|² / M of x at Fourier bin k (0 < k < M / 2), by fastgoertzel."""
    m = len(x)
    if not (isinstance(k, int | np.integer) and 0 < k < m / 2):
        raise ValueError(f"bin k must be an integer in (0, M/2), got {k} for M={m}")
    amp, _ = fastgoertzel.goertzel(np.ascontiguousarray(x, dtype=np.float64), (k + 0.5) / m)  # int(f · M) = k
    return amp * amp * m / 4.0


def periodogram(x: np.ndarray) -> np.ndarray:
    """I_k for k = 1 … ⌊(M − 1) / 2⌋ (DC and Nyquist excluded), one Goertzel per bin."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    return np.array([goertzel_power(x, k) for k in range(1, (len(x) - 1) // 2 + 1)])


def _band(m: int, periods: tuple[int, int]) -> slice:
    """Index range into periodogram(x) (bin k at index k − 1) of the bins with period in [lo, hi] bars."""
    lo, hi = periods
    k_lo, k_hi = int(np.ceil(m / hi)), min(int(np.floor(m / lo)), (m - 1) // 2)
    return slice(max(k_lo, 1) - 1, k_hi)


def spectral_state(x: np.ndarray) -> tuple[float, float, float, float]:
    """(pow_hf, pow_lf, tilt, flatness) of one window of returns (demeaned here)."""
    x = np.asarray(x, dtype=np.float64)
    p = periodogram(x - x.mean())
    mean = p.mean()
    if not mean > 0:
        return np.nan, np.nan, np.nan, np.nan
    hf, lf = p[_band(len(x), HF_PERIODS)].mean() / mean, p[_band(len(x), LF_PERIODS)].mean() / mean
    flat = np.exp(np.mean(np.log(p))) / mean if np.all(p > 0) else 0.0
    return hf, lf, np.log(hf / lf), flat


def _session_returns(close, sess) -> tuple[np.ndarray, np.ndarray]:
    """(r, day): the finite inside-session 5Min log returns, concatenated, and the session of each."""
    c = np.asarray(close, dtype=np.float64)
    sess = np.asarray(sess, dtype=np.int64)
    first = np.r_[True, sess[1:] != sess[:-1]] if len(c) else np.zeros(0, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(c / np.r_[np.nan, c[:-1]])
    ok = ~first & np.isfinite(r)
    return r[ok], (np.cumsum(first) - 1)[ok]


def tf_state_sessions(close, sess) -> pd.DataFrame:
    """Per session (rows in data order, 0 … n_sessions − 1): the COLUMNS above, known at the session's open."""
    from features.groups import causal_modwt

    sess = np.asarray(sess, dtype=np.int64)
    n_sess = int(sess[-1]) + 1 if len(sess) else 0
    r, day = _session_returns(close, sess)
    bounds = np.searchsorted(day, np.arange(n_sess + 1))  # returns of session d: r[bounds[d]:bounds[d + 1]]
    out = {c: np.full(n_sess, np.nan) for c in COLUMNS}

    details, _ = causal_modwt(pd.Series(r), MODWT_FILTER, MODWT_LEVELS)
    w2 = [np.nan_to_num(w.to_numpy() ** 2) for w in details]
    have = [np.isfinite(w.to_numpy()).astype(float) for w in details]
    csum = lambda v: np.r_[0.0, np.cumsum(v)]
    cr2, cw2, cn = csum(r * r), [csum(v) for v in w2], [csum(v) for v in have]

    for d in range(TF_SESSIONS, n_sess):
        a, b = bounds[d - TF_SESSIONS], bounds[d]
        m = b - a
        if m < TF_MIN_RETURNS:
            continue
        x = r[a:b]
        out["pow_hf"][d], out["pow_lf"][d], out["tilt"][d], out["flatness"][d] = spectral_state(x)
        var = (cr2[b] - cr2[a]) / m
        for j in range(MODWT_LEVELS):
            n = cn[j][b] - cn[j][a]
            if n >= m // 2 and var > 0:
                nu = (cw2[j][b] - cw2[j][a]) / n
                out[f"modwt_e{j + 1}"][d] = np.log(2 ** (j + 1) * nu / var) if nu > 0 else np.nan

    out["ac1"] = prior_lag1_autocorr(close, sess, TF_SESSIONS)
    with np.errstate(divide="ignore"):
        ls = np.log(session_sigma(close, sess))
    ls[~np.isfinite(ls)] = np.nan
    vov = pd.Series(ls).rolling(VOV_SESSIONS, min_periods=VOV_SESSIONS // 2 + 1).std().shift(1)
    out["vov"] = vov.to_numpy()
    return pd.DataFrame(out)


def tf_state_frame(df: pd.DataFrame, bar_minutes: int) -> dict[str, np.ndarray]:
    """Per bar: its session's tf_state (every bar of a session carries the values known at its open)."""
    sess = session_layout(df.index, bar_minutes)[0]
    per = tf_state_sessions(df["close"].to_numpy(dtype=float), sess)
    return {c: per[c].to_numpy()[sess] for c in COLUMNS}


def tf_state_days(df: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """tf_state_sessions indexed by session date (tz-naive NY): the reported splits' input (families.run)."""
    from features.vol_profile import ny_dates

    sess = session_layout(df.index, bar_minutes)[0]
    per = tf_state_sessions(df["close"].to_numpy(dtype=float), sess)
    day = pd.DatetimeIndex(pd.unique(ny_dates(df.index)))
    return per.set_index(day)
