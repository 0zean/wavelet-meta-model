import time

import numpy as np
import pandas as pd

from features import cache as feature_cache
from features.causal_modwt import wavelet_ar_features
from features.fractional_diff import fit_fracdiff_d, fracdiff_transform
from features.indicators import compute_log_return, compute_rsi, compute_siegel_slope, compute_vwap
from features.registry import FeatureGroup, check_group_output, resolve_groups
from utils.config import RunConfig


def legacy_features(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    The pre-U3 feature matrix, kept verbatim for the legacy regression invariant
    (RunConfig.legacy_5min(), FEATURE_GROUPS=None). Not a registry group: w_lag_* are raw
    S_J price levels, which the registry's stationarity check rejects.

    Features:
        vwap_dev        : (close - VWAP) / VWAP — distance from intraday VWAP
                          (for 1Day bars VWAP is the bar's own typical price)
        rsi             : RSI(14)
        siegel_slope    : rolling Siegel slope on close(20)
        log_ret         : 1-bar log return
        log_vol         : log(volume)
        w_lag_1..4      : causal MODWT S4 AR features (lags 1-4 of smoothing component)
        hl_spread       : (high - low) / close
    fd_close is added per fold (see FeatureSet).
    """
    feats = pd.DataFrame(index=df.index)
    vwap = compute_vwap(df)
    feats["vwap_dev"] = (df["close"] - vwap) / vwap
    feats["rsi"] = compute_rsi(df["close"], cfg.RSI_PERIOD)
    feats["siegel_slope"] = compute_siegel_slope(df["close"], cfg.SIEGEL_WINDOW)
    feats["log_ret"] = compute_log_return(df["close"])
    feats["log_vol"] = np.log(df["volume"].replace(0, np.nan))
    feats = feats.join(wavelet_ar_features(df["close"], cfg.WAVELET_FILTER, cfg.WAVELET_J, cfg.AR_LAGS))
    feats["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    return feats


class _LegacyFracdiff:
    column = "fd_close"

    @staticmethod
    def fit(train_df, cfg):
        return fit_fracdiff_d(train_df["close"])

    @staticmethod
    def transform(df, state, cfg):
        return fracdiff_transform(df["close"], state).reindex(df.index).rename("fd_close").to_frame()


class FeatureSet:
    """
    The resolved feature groups for one run.

    `build(df)` computes the static groups once on the full series (causal, so it is
    sliced per fold); `fit(train_df)` fits the per-fold groups on train only and
    `transform(df, states)` applies them to a (prefix of the) series.

    cfg.FEATURE_GROUPS = None selects the legacy matrix (regression invariant only).
    """

    def __init__(
        self,
        cfg: RunConfig,
        groups=None,
        *,
        context: dict[str, pd.DataFrame] | None = None,
        symbol: str | None = None,
        cache_dir=None,
    ):
        self.cfg = cfg
        self.context = context or {}
        self.symbol = symbol
        self.cache_dir = cache_dir
        groups = cfg.FEATURE_GROUPS if groups is None else groups
        self.legacy = groups is None
        if self.legacy:
            self.static: list[FeatureGroup] = []
            self.per_fold = [_LegacyFracdiff]
            return
        specs = resolve_groups(groups, cfg.TIMEFRAME)
        for spec in specs:
            missing = [k for k in spec.needs if k not in self.context]
            if missing:
                raise ValueError(f"feature group {spec.name!r} needs context {missing} (e.g. market bars)")
        self.static = [s for s in specs if not s.per_fold]
        self.per_fold = [s.fn for s in specs if s.per_fold]
        self.names = [s.name for s in specs]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.legacy:
            return legacy_features(df, self.cfg)
        parts = []
        for spec in self.static:
            t0 = time.time()
            feats = self._cached(spec, df)
            print(f"[FEAT]  {spec.name:<15} {feats.shape[1]:>3} cols  {time.time() - t0:6.2f}s")
            parts.append(feats)
        return pd.concat(parts, axis=1)

    def _cached(self, spec: FeatureGroup, df: pd.DataFrame) -> pd.DataFrame:
        ctx = {k: self.context[k] for k in spec.needs}
        use_cache = self.symbol is not None and self.cache_dir is not None
        if use_cache:
            key = feature_cache.cache_key(spec.name, self.symbol, self.cfg, df, ctx)
            hit = feature_cache.load(self.cache_dir, self.symbol, self.cfg.TIMEFRAME, spec.name, key, df.index)
            if hit is not None:
                return hit
        feats = spec.fn(df, self.cfg, ctx)
        check_group_output(spec.name, feats, df, spec.level_check)
        if use_cache:
            feature_cache.save(self.cache_dir, self.symbol, self.cfg.TIMEFRAME, spec.name, key, feats)
        return feats

    def fit(self, train_df: pd.DataFrame) -> list:
        """Per-fold states fit on `train_df` only (fracdiff raises RuntimeWarning if no d is stationary)."""
        return [g.fit(train_df, self.cfg) for g in self.per_fold]

    def transform(self, df: pd.DataFrame, states: list) -> pd.DataFrame:
        cols = [g.transform(df, s, self.cfg) for g, s in zip(self.per_fold, states, strict=True)]
        return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=df.index)


def build_features(
    df: pd.DataFrame,
    cfg: RunConfig,
    groups=None,
    *,
    context: dict[str, pd.DataFrame] | None = None,
    symbol: str | None = None,
    cache_dir=None,
) -> pd.DataFrame:
    """
    Static (non-per-fold) causal features: every column at bar t uses bars <= t only,
    so it can be computed once on the full series and sliced per fold.

    Args:
        df (pd.DataFrame): Bars with at least OHLCV.
        cfg (RunConfig): Run configuration.
        groups (list[str] | None): Feature groups; None = cfg.FEATURE_GROUPS (None there = legacy matrix).
            Must include 'wavelet_core'; intraday-only groups are skipped on 1Day.
        context (dict | None): Extra aligned inputs, e.g. {"market": SPY bars} for 'cross_asset'.
        symbol (str | None): With cache_dir, enables the feature cache.
        cache_dir (path | None): Feature cache root (see features.cache).

    Returns:
        pd.DataFrame: Features indexed like df, columns '{group}__{feature}'.
    """
    return FeatureSet(cfg, groups, context=context, symbol=symbol, cache_dir=cache_dir).build(df)
