"""
The feature zoo (SPEC §3). Every group is causal: row t uses bars <= t only, so a group
can be computed once on the full series and sliced per fold. Warm-up rows are NaN.

Only stationary transforms: returns, ratios, z-scores, oscillators, volatilities,
cyclical calendar encodings, and (per fold) fracdiff. Window lengths not taken from
`cfg` are module constants below (they are part of the feature-cache code hash).
"""

import numba
import numpy as np
import pandas as pd
from pywddff.filters import scaling_filter, wavelet_filter

from features.causal_modwt import wavelet_ar_features
from features.fractional_diff import fit_fracdiff_d, fracdiff_transform
from features.indicators import compute_rsi, compute_siegel_slope, compute_vwap
from features.registry import feature_group
from features.triple_barrier_labels import bar_volatility

WAVELET_EXT_FILTERS = ("db1", "db2", "la8")
ENERGY_WINDOW_MULT = 4  # detail-energy window = ENERGY_WINDOW_MULT · 2^J bars
SMA_WINDOWS = (20, 50)
EMA_SPAN = 20
ADX_PERIOD = 14
BB_WINDOW, BB_K = 20, 2.0
ZSCORE_WINDOW = 50
RANGE_VOL_WINDOW = 20  # Parkinson / Garman–Klass
SHORT_VOL_SPAN = 20
VOL_OF_VOL_WINDOW = 50
SPREAD_WINDOW = 20  # Amihud / Roll / Corwin–Schultz
SADF_WINDOWS = (50, 100, 200)
BETA_WINDOW = 50


# ── helpers ──────────────────────────────────────────────────────────────────


def _div(num, den) -> np.ndarray:
    """num / den with 0 where den == 0 (NaN stays NaN)."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, 0.0, num / den)


def _prefixed(name: str, cols: dict, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({f"{name}__{k}": np.asarray(v, dtype=float) for k, v in cols.items()}, index=index)


def _zscore(x: pd.Series, window: int) -> np.ndarray:
    roll = x.rolling(window, min_periods=window)
    return _div(x - roll.mean(), roll.std())


def causal_modwt(x: pd.Series, filt: str, levels: int) -> tuple[list[pd.Series], pd.Series]:
    """
    Causal MODWT pyramid (Percival & Walden 2000, p.177) without circular wrap:
        W_j[t] = Σ_n h[n] V_{j-1}[t - n·2^(j-1)],  V_j[t] = Σ_n g[n] V_{j-1}[t - n·2^(j-1)].

    Returns:
        (details W_1..W_J, smooth V_J), each aligned to x, NaN over the boundary.
    """
    g = scaling_filter(filt, modwt=True)
    h = wavelet_filter(filt, modwt=True)
    v = x.astype(float)
    details = []
    for j in range(levels):
        lagged = [v.shift(n * 2**j) for n in range(len(g))]
        details.append(sum(h[n] * lagged[n] for n in range(len(h))))
        v = sum(g[n] * lagged[n] for n in range(len(g)))
    return details, v


def _intraday_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since 09:30 of each bar's open stamp (NY time for tz-aware indexes)."""
    return np.asarray(index.hour * 60 + index.minute - 570, dtype=float)


TIMEFRAME_MINUTES = {"1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60}


# ── groups ───────────────────────────────────────────────────────────────────


@feature_group("wavelet_core", required=True)
def wavelet_core(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """
    Causal MODWT smooth S_J of close, lagged 1..AR_LAGS bars, expressed relative to the
    current close: log(S_J[t-k] / close[t]). (The legacy feature set uses the raw S_J
    levels, which are ~1.0 correlated with price; this is the stationary equivalent.)
    """
    lags = wavelet_ar_features(df["close"], cfg.WAVELET_FILTER, cfg.WAVELET_J, cfg.AR_LAGS)
    close = df["close"].to_numpy(dtype=float)
    return _prefixed(
        "wavelet_core",
        {f"s{cfg.WAVELET_J}_lag_{k}": np.log(lags[f"w_lag_{k}"].to_numpy() / close) for k in range(1, cfg.AR_LAGS + 1)},
        df.index,
    )


@feature_group("wavelet_ext")
def wavelet_ext(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """
    Per filter (db1, db2, la8) on log close: relative detail energies e_j = E_j / Σ_k E_k
    (E_j = trailing mean of W_j² over ENERGY_WINDOW_MULT·2^J bars), the multiscale variance
    ratio log(E_J / E_1), and the S_J slope V_J[t] − V_J[t−1] in σ-units of the bar return.
    """
    J = cfg.WAVELET_J
    logp = np.log(df["close"].astype(float))
    window = ENERGY_WINDOW_MULT * 2**J
    sigma = bar_volatility(df["close"], cfg.VOL_SPAN).to_numpy()
    cols = {}
    for filt in WAVELET_EXT_FILTERS:
        details, smooth = causal_modwt(logp, filt, J)
        energy = [(w**2).rolling(window, min_periods=window).mean().to_numpy() for w in details]
        total = np.sum(energy, axis=0)
        for j, e in enumerate(energy, start=1):
            cols[f"{filt}_e{j}"] = _div(e, total)
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.log(_div(energy[-1], energy[0]))
        cols[f"{filt}_vr"] = np.where(np.isfinite(vr) | np.isnan(vr), vr, 0.0)
        cols[f"{filt}_slope"] = _div(smooth.diff().to_numpy(), sigma)
    return _prefixed("wavelet_ext", cols, df.index)


def _adx(df: pd.DataFrame, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Wilder ADX and (+DI − −DI)/100."""
    high, low, close = df["high"], df["low"], df["close"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm[up.isna()] = np.nan
    minus_dm[up.isna()] = np.nan
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    tr[close.shift().isna()] = np.nan

    def wilder(s):
        return s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    atr = wilder(tr)
    pdi, mdi = _div(wilder(plus_dm), atr), _div(wilder(minus_dm), atr)
    dx = pd.Series(_div(np.abs(pdi - mdi), pdi + mdi), index=df.index)
    return wilder(dx).to_numpy(), pdi - mdi


@feature_group("trend")
def trend(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """1-bar and holding-horizon momentum, Siegel slope / close, SMA/EMA distance, ADX and DI spread."""
    close = df["close"].astype(float)
    logp = np.log(close)
    h = cfg.VERTICAL_BARS
    adx, di = _adx(df, ADX_PERIOD)
    cols = {
        "ret_1": logp.diff(),
        f"mom_{h}": logp.diff(h),
        f"mom_{4 * h}": logp.diff(4 * h),
        "siegel_slope": compute_siegel_slope(close, cfg.SIEGEL_WINDOW) / close,
        **{f"sma_dist_{n}": logp - np.log(close.rolling(n, min_periods=n).mean()) for n in SMA_WINDOWS},
        f"ema_dist_{EMA_SPAN}": logp - np.log(close.ewm(span=EMA_SPAN, min_periods=EMA_SPAN, adjust=False).mean()),
        f"adx_{ADX_PERIOD}": adx,
        "di_spread": di,
    }
    return _prefixed("trend", cols, df.index)


@feature_group("mean_reversion")
def mean_reversion(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """RSI, Bollinger %b, and z-score of close vs its rolling mean."""
    close = df["close"].astype(float)
    roll = close.rolling(BB_WINDOW, min_periods=BB_WINDOW)
    mid, sd = roll.mean(), roll.std()
    cols = {
        "rsi": compute_rsi(close, cfg.RSI_PERIOD),
        f"bb_pctb_{BB_WINDOW}": _div(close - (mid - BB_K * sd), 2 * BB_K * sd),
        f"zscore_{ZSCORE_WINDOW}": _zscore(close, ZSCORE_WINDOW),
    }
    return _prefixed("mean_reversion", cols, df.index)


@feature_group("volatility")
def volatility(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """EWM σ, Parkinson and Garman–Klass range σ, high-low range, short/long σ ratio, vol-of-vol."""
    o, hi, lo, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    hl = np.log(hi / lo)
    co = np.log(c / o)
    sigma = bar_volatility(c, cfg.VOL_SPAN)
    sigma_s = bar_volatility(c, SHORT_VOL_SPAN)
    w = RANGE_VOL_WINDOW
    park = np.sqrt((hl**2).rolling(w, min_periods=w).mean() / (4 * np.log(2)))
    gk = np.sqrt((0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(w, min_periods=w).mean().clip(lower=0))
    with np.errstate(divide="ignore"):
        log_sig = pd.Series(np.log(sigma_s.to_numpy()), index=df.index).replace(-np.inf, np.nan)
    cols = {
        "ewm_sigma": sigma,
        f"parkinson_{w}": park,
        f"garman_klass_{w}": gk,
        "hl_range": hl,
        "sigma_ratio": _div(sigma_s, sigma),
        "vol_of_vol": log_sig.rolling(VOL_OF_VOL_WINDOW, min_periods=VOL_OF_VOL_WINDOW).std(),
    }
    return _prefixed("volatility", cols, df.index)


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Two-bar Corwin–Schultz (2012) spread estimate, floored at 0."""
    k = 3 - 2 * np.sqrt(2)
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift()
    gamma = np.log(np.maximum(high, high.shift()) / np.minimum(low, low.shift())) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    return (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)


@feature_group("microstructure")
def microstructure(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """log(1+volume), volume z-score (≈ one-session window), Amihud illiquidity, Roll and Corwin–Schultz spreads."""
    c, v = df["close"].astype(float), df["volume"].astype(float)
    ret = np.log(c).diff()
    logv = np.log1p(v)
    w = SPREAD_WINDOW
    amihud = pd.Series(_div(ret.abs(), c * v), index=df.index).rolling(w, min_periods=w).mean()
    cov = ret.rolling(w, min_periods=w).cov(ret.shift())
    cols = {
        "log_volume": logv,
        "volume_z": _zscore(logv, max(20, cfg.BARS_PER_DAY)),
        "amihud": np.log(amihud.clip(lower=1e-20)),
        "roll_spread": 2 * np.sqrt((-cov).clip(lower=0)),
        "cs_spread": corwin_schultz(df["high"].astype(float), df["low"].astype(float)).rolling(w, min_periods=w).mean(),
    }
    return _prefixed("microstructure", cols, df.index)


def cusum_state(close: pd.Series, threshold: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The symmetric CUSUM filter's running S+ / S− (reset on crossing), in units of the threshold."""
    ret = np.log(close.astype(float)).diff().to_numpy()
    return _cusum_state_kernel(ret, threshold.to_numpy(dtype=float))


@numba.njit(cache=True)
def _cusum_state_kernel(ret: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cusum_state's recursion, compiled (U12; the loop's float operations in the same order, checked in
    tests/test_features.py). NaN on bars with a NaN return or a NaN / zero threshold, which leave the state as is."""
    pos, neg = np.full(len(ret), np.nan), np.full(len(ret), np.nan)
    s_pos = 0.0
    s_neg = 0.0
    for t in range(1, len(ret)):
        if np.isnan(h[t]) or np.isnan(ret[t]) or h[t] == 0:
            continue
        s_pos = max(0.0, s_pos + ret[t])
        s_neg = min(0.0, s_neg + ret[t])
        pos[t], neg[t] = s_pos / h[t], s_neg / h[t]
        if s_pos > h[t]:
            s_pos = 0.0
        elif s_neg < -h[t]:
            s_neg = 0.0
    return pos, neg


def rolling_df_tstat(y: pd.Series, window: int) -> np.ndarray:
    """Dickey–Fuller t-stat (Δy_t = a + b·y_{t−1}, no lags) over the trailing `window` pairs."""
    x, z = y.shift(), y.diff()
    n = window
    var_x = x.rolling(n, min_periods=n).var()
    var_z = z.rolling(n, min_periods=n).var()
    cov = z.rolling(n, min_periods=n).cov(x)
    b = _div(cov, var_x)
    resid = np.clip((var_z - b * cov) * (n - 1) / (n - 2), 0, None)
    se = np.sqrt(_div(resid, (n - 1) * var_x))
    return _div(b, se)


@feature_group("structural")
def structural(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """CUSUM filter state (S+, S− in threshold units) and SADF-lite: max rolling DF t-stat over SADF_WINDOWS."""
    close = df["close"].astype(float)
    pos, neg = cusum_state(close, cfg.CUSUM_MULT * bar_volatility(close, cfg.VOL_SPAN))
    logp = np.log(close)
    stats = np.vstack([rolling_df_tstat(logp, w) for w in SADF_WINDOWS])
    cols = {
        "cusum_pos": pos,
        "cusum_neg": neg,
        f"df_t_{SADF_WINDOWS[0]}": stats[0],
        "sadf_lite": np.where(np.isnan(stats).any(axis=0), np.nan, np.max(stats, axis=0)),
    }
    return _prefixed("structural", cols, df.index)


@feature_group("calendar", level_check=False)
def calendar(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """Day-of-week and month-of-year, cyclically encoded (known in advance, so trivially causal)."""
    idx = df.index
    dow, month = np.asarray(idx.dayofweek, dtype=float), np.asarray(idx.month - 1, dtype=float)
    cols = {
        "dow_sin": np.sin(2 * np.pi * dow / 5),
        "dow_cos": np.cos(2 * np.pi * dow / 5),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }
    return _prefixed("calendar", cols, idx)


@feature_group("intraday", intraday_only=True)
def intraday(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """
    Session VWAP deviation log(close/VWAP), time of day (cyclical, of the bar's close), and
    the bar's scheduled length as a fraction of the timeframe (the 1Hour 15:30 bar = 0.5).
    Bar length assumes a 16:00 close (early closes are not known from the bars alone).
    """
    c = df["close"].astype(float)
    tf_min = TIMEFRAME_MINUTES[cfg.TIMEFRAME]
    start = _intraday_minutes(df.index)
    end = np.minimum(start + tf_min, 390.0)
    cols = {
        "vwap_dev": np.log(c / compute_vwap(df)),
        "tod_sin": np.sin(2 * np.pi * end / 390),
        "tod_cos": np.cos(2 * np.pi * end / 390),
        "bar_frac": (end - start) / tf_min,
    }
    return _prefixed("intraday", cols, df.index)


@feature_group("cross_asset", needs=("market",))
def cross_asset(df: pd.DataFrame, cfg, context) -> pd.DataFrame:
    """
    Market (SPY) context for another symbol: market return and σ, rolling beta and
    correlation, residual return, relative momentum. The market close is aligned
    causally: bar t takes the last market bar stamped <= t.
    """
    market = context["market"]
    mc = market["close"].astype(float)
    mc = mc.reindex(df.index.union(mc.index)).ffill().reindex(df.index)
    r = np.log(df["close"].astype(float)).diff()
    m = np.log(mc).diff()
    w, h = BETA_WINDOW, cfg.VERTICAL_BARS
    beta = pd.Series(_div(r.rolling(w, min_periods=w).cov(m), m.rolling(w, min_periods=w).var()), index=df.index)
    cols = {
        "mkt_ret_1": m,
        "mkt_sigma": bar_volatility(mc, cfg.VOL_SPAN),
        f"beta_{w}": beta,
        f"corr_{w}": r.rolling(w, min_periods=w).corr(m).clip(-1, 1),
        "resid_ret": r - beta * m,
        f"rel_mom_{h}": np.log(df["close"].astype(float)).diff(h) - np.log(mc).diff(h),
    }
    return _prefixed("cross_asset", cols, df.index)


@feature_group("fracdiff", per_fold=True)
class Fracdiff:
    """Fixed-window fractional difference of close; d = the minimum ADF-stationary d fit on train only."""

    column = "fracdiff__close"

    @staticmethod
    def fit(train_df: pd.DataFrame, cfg):
        return fit_fracdiff_d(train_df["close"])

    @staticmethod
    def transform(df: pd.DataFrame, state, cfg) -> pd.DataFrame:
        fd = fracdiff_transform(df["close"], state).reindex(df.index)
        return fd.rename(Fracdiff.column).to_frame()
