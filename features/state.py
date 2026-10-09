"""
State and context feature groups (SPEC §15, U15): the computations behind `vol_state`, `calendar_events` and
`rates_credit` (registered in features/groups.py). Exogenous values are read only through features.exo_align.Exo, so
every exo column obeys the point-in-time rule (available_at ≤ bar stamp).

vol_state (per fold; needs exo):
    vix             VIX close (CBOE), as of the bar
    vix9d_vix       VIX9D / VIX, vix3m_vix  VIX3M / VIX (ratios built on common observation dates)
    vx1_vix         front VIX future / VIX, vx2_vx1  second / front future (contango > 1)
    rv21_vix        realized σ of the last 21 completed sessions' close-to-close returns (annualized) / (VIX / 100);
                    intraday bars use the sessions before the bar's own (it is not complete), 1Day bars include it
    garch_sigma     GARCH(1,1) one-step conditional σ for the next bar, in session units (intraday: bar returns
                    deseasonalized by the fold's time-of-day VolProfile, σ × √Σ s(b)²; 1Day: daily returns)
    ret_std_garch   the bar's (deseasonalized) return / its conditional σ given the bars before it (the residual)
    The GARCH parameters (variance targeting: ω = v̄(1 − α − β), v̄ the train variance) are fit by Gaussian maximum
    likelihood on the fold's train bars only; fewer than GARCH_MIN_OBS train returns raises RuntimeWarning (the fold
    is skipped, as for fracdiff).

calendar_events (static; needs exo; deterministic encodings, so no level check):
    to_fomc, to_cpi, to_nfp           sessions from the bar's session to the next visible event day (0 on the day);
                                      TO_CAP where none is visible yet (scheduled rows are public from 1 January of
                                      their year, so the next year's first CPI / NFP / FOMC is unknown in late
                                      December) and clipped at TO_CAP
    since_fomc, since_cpi, since_nfp  sessions since the last visible event day (0 on the day); NaN before the table
    fomc_day, cpi_nfp_day             the bar's day is an (already announced) FOMC / CPI or NFP day
    opex_week                         the bar's Mon–Fri week holds a monthly options expiry
    tom                               turn-of-month position: −1 on the month's last session, +1 … +3 on the next
                                      month's first three, 0 otherwise
    pre_holiday                       a session followed by a weekday closure
    mins_since_release                (intraday) minutes from the day's FOMC / CPI / NFP release to the bar's close;
                                      negative before a release later that day; RELEASE_CAP on days without one
    Sessions are counted as weekdays minus the exchange closures known at the bar (the weekday after each visible
    PRE_HOLIDAY session; no closure in 2016–2027 spans two weekdays).

rates_credit (static; needs exo):
    d_dgs10     change of the 10-year Treasury yield over its last observation (percentage points)
    t10y2y      10-year minus 2-year Treasury spread (percentage points)
    d_credit    change of Moody's Baa − 10-year spread (FRED BAA10Y) over its last observation; HY OAS
                (BAMLH0A0HYM2) only has the last three years on FRED and cannot cover the development window
    d_dollar    log change of the broad dollar index (DTWEXBGS) over its last 5 observations (H.10 is weekly)
"""

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features.vol_profile import VolProfile, ny_dates, profile_for

RV_SESSIONS = 21
TRADING_DAYS = 252
GARCH_MIN_OBS = 500
TO_CAP = 63
RELEASE_CAP = 480  # minutes; > 08:30 → 16:00

VOL_STATE_EXO = ("cboe/VIX", "cboe/VIX9D", "cboe/VIX3M", "cboe/VX1_VIX", "cboe/VX2_VX1")
CALENDAR_EXO = ("calendar/FOMC", "calendar/CPI", "calendar/NFP", "calendar/OPEX", "calendar/TOM",
                "calendar/PRE_HOLIDAY")  # fmt: skip
RATES_EXO = ("fred/DGS10", "fred/T10Y2Y", "fred/BAA10Y", "fred/DTWEXBGS")
RELEASE_KINDS = ("calendar/FOMC", "calendar/CPI", "calendar/NFP")


def _session_closes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(session number of each bar, close of each session's last bar)."""
    day = ny_dates(df.index)
    n = len(day)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    return np.cumsum(first) - 1, df["close"].to_numpy(float)[last]


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _garch_h(z: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """h[t] = conditional variance of z[t] given z[<t] (h[0] = h0); h[n] = the forecast after the last value. A NaN
    value (warm-up, gaps) carries the variance forward."""
    n = len(z)
    h = np.empty(n + 1)
    h[0] = h0
    for t in range(n):
        x = z[t]
        h[t + 1] = h[t] if np.isnan(x) else omega + alpha * x * x + beta * h[t]
    return h


@numba.njit(cache=True)
def _garch_nll(z: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> float:
    h = _garch_h(z, omega, alpha, beta, h0)
    s = 0.0
    for t in range(len(z)):
        if not np.isnan(z[t]):
            s += np.log(h[t]) + z[t] * z[t] / h[t]
    return 0.5 * s


def fit_garch(z: np.ndarray) -> tuple[float, float, float, float]:
    """
    Gaussian GARCH(1,1) with variance targeting on z (zero mean): (ω, α, β, v̄). Parametrised by persistence
    p = α + β ∈ [0.01, 0.998] and share a = α / p ∈ [0.001, 0.999] (stationary by construction).

    Raises:
        RuntimeWarning: Fewer than GARCH_MIN_OBS finite values, or the optimiser fails.
    """
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(z)
    if ok.sum() < GARCH_MIN_OBS:
        raise RuntimeWarning(f"GARCH: {int(ok.sum())} train returns < GARCH_MIN_OBS {GARCH_MIN_OBS}")
    z = np.where(ok, z, np.nan)
    vbar = float(np.nanmean(z**2))
    if not vbar > 0:
        raise RuntimeWarning("GARCH: train returns are all zero")

    def nll(x):
        p, a = x
        return _garch_nll(z, vbar * (1 - p), a * p, (1 - a) * p, vbar) / ok.sum()

    res = minimize(nll, x0=np.array([0.95, 0.1]), method="L-BFGS-B", bounds=[(0.01, 0.998), (0.001, 0.999)])
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeWarning(f"GARCH: the likelihood fit failed ({res.message})")
    p, a = res.x
    return vbar * (1 - p), a * p, (1 - a) * p, vbar


def garch_returns(df: pd.DataFrame, profile: VolProfile | None) -> np.ndarray:
    """1-bar log close returns, divided by s(b) of the bar's slot under a profile (intraday)."""
    r = np.log(df["close"].astype(float)).diff().to_numpy()
    return r if profile is None else r / profile.factor(df.index)


def fit_vol_state(train_df: pd.DataFrame, cfg) -> tuple:
    """Per-fold state: (VolProfile | None, ω, α, β, v̄) fit on train bars only."""
    m = profile_for(cfg)
    profile = None if m is None else VolProfile.fit(train_df, m)
    return (profile, *fit_garch(garch_returns(train_df, profile)))


def vol_state_frame(df: pd.DataFrame, state: tuple, cfg, exo) -> dict[str, np.ndarray]:
    profile, omega, alpha, beta, vbar = state
    idx = df.index
    vix = exo.asof("cboe/VIX", idx)

    sess, c_s = _session_closes(df)
    r_s = np.r_[np.nan, np.log(c_s[1:] / c_s[:-1])]
    rv_s = pd.Series(r_s).rolling(RV_SESSIONS, min_periods=RV_SESSIONS).std().to_numpy() * np.sqrt(TRADING_DAYS)
    lag = 0 if profile_for(cfg) is None else 1  # an intraday bar's own session is not complete
    known = sess - lag
    rv = np.full(len(idx), np.nan)
    rv[known >= 0] = rv_s[known[known >= 0]]

    z = garch_returns(df, profile)
    h = _garch_h(z, omega, alpha, beta, vbar)
    scale = 1.0 if profile is None else profile.session_scale()
    with np.errstate(divide="ignore", invalid="ignore"):
        rv_vix = np.where(vix > 0, rv / (vix / 100), np.nan)
    return {
        "vix": vix,
        "vix9d_vix": exo.ratio("cboe/VIX9D", "cboe/VIX", idx),
        "vix3m_vix": exo.ratio("cboe/VIX3M", "cboe/VIX", idx),
        "vx1_vix": exo.asof("cboe/VX1_VIX", idx),
        "vx2_vx1": exo.asof("cboe/VX2_VX1", idx),
        "rv21_vix": rv_vix,
        "garch_sigma": np.sqrt(h[1:]) * scale,
        "ret_std_garch": z / np.sqrt(h[:-1]),
    }


# ── Calendar events ──────────────────────────────────────────────────────────


def _sessions_between(a: np.ndarray, b: np.ndarray, k: np.ndarray, pre_holidays: np.ndarray) -> np.ndarray:
    """np.busday_count(a, b) (weekdays in [a, b)) minus the closures known at each bar: the weekday after each of the
    first k[j] PRE_HOLIDAY dates. NaN where a or b is NaT."""
    out = np.full(len(a), np.nan)
    ok = ~(np.isnat(a) | np.isnat(b))
    closures = np.busday_offset(pre_holidays, 1, roll="forward")
    for kk in np.unique(k[ok]):
        m = ok & (k == kk)
        out[m] = np.busday_count(a[m], b[m], holidays=closures[:kk])
    return out


def calendar_events_frame(df: pd.DataFrame, cfg, exo) -> dict[str, np.ndarray]:
    idx = df.index
    day = ny_dates(idx)
    one = np.timedelta64(1, "D")
    k_hol, pre_hol = exo.known_dates("calendar/PRE_HOLIDAY", idx)
    cols = {}
    for kind in ("FOMC", "CPI", "NFP"):
        name = f"calendar/{kind}"
        to = _sessions_between(day, exo.next_date(name, idx), k_hol, pre_hol)
        cols[f"to_{kind.lower()}"] = np.minimum(np.where(np.isnan(to), TO_CAP, to), TO_CAP)
        cols[f"since_{kind.lower()}"] = _sessions_between(exo.last_date(name, idx) + one, day + one, k_hol, pre_hol)
    on = {kind: ~np.isnan(exo.on_day(f"calendar/{kind}", idx)) for kind in ("FOMC", "CPI", "NFP", "PRE_HOLIDAY")}
    opex = exo.next_date("calendar/OPEX", idx)
    monday = np.busday_offset(day, 0, roll="backward", weekmask="Mon")
    cols |= {
        "fomc_day": on["FOMC"].astype(float),
        "cpi_nfp_day": (on["CPI"] | on["NFP"]).astype(float),
        "opex_week": (~np.isnat(opex) & (opex - monday < np.timedelta64(5, "D"))).astype(float),
        "tom": np.nan_to_num(exo.on_day("calendar/TOM", idx), nan=0.0),
        "pre_holiday": on["PRE_HOLIDAY"].astype(float),
    }
    m = profile_for(cfg)
    if m is not None:
        cols["mins_since_release"] = _mins_since_release(idx, m, exo)
    return cols


def _mins_since_release(idx: pd.DatetimeIndex, bar_minutes: int, exo) -> np.ndarray:
    """Minutes from the day's latest release at or before the bar's close (else the next one that day, negative) to
    the bar's close; RELEASE_CAP on days without a visible release."""
    from features.exo_align import bar_stamps_utc

    close = bar_stamps_utc(idx) + bar_minutes * 60 * 10**9
    none = np.iinfo(np.int64).min
    since = np.full(len(idx), np.inf)  # smallest non-negative gap
    until = np.full(len(idx), -np.inf)  # largest negative gap
    for name in RELEASE_KINDS:
        ev = exo.on_day(name, idx, "event_at")
        ok = ev != none
        gap = np.full(len(idx), np.nan)
        gap[ok] = (close[ok] - ev[ok]) / 6e10
        since = np.where(ok & (gap >= 0), np.minimum(since, gap), since)
        until = np.where(ok & (gap < 0), np.maximum(until, gap), until)
    out = np.where(np.isfinite(since), since, np.where(np.isfinite(until), until, RELEASE_CAP))
    return np.clip(out, -RELEASE_CAP, RELEASE_CAP)


# ── Rates and credit ─────────────────────────────────────────────────────────


def rates_credit_frame(df: pd.DataFrame, cfg, exo) -> dict[str, np.ndarray]:
    idx = df.index
    return {
        "d_dgs10": exo.change("fred/DGS10", idx, 1),
        "t10y2y": exo.asof("fred/T10Y2Y", idx),
        "d_credit": exo.change("fred/BAA10Y", idx, 1),
        "d_dollar": exo.change("fred/DTWEXBGS", idx, 5, log=True),
    }
