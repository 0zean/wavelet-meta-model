"""
Rule-based primaries (SPEC §4). Each is a causal bar-level score whose sign is the side and
whose magnitude (in σ-units or channel units) is the strength passed to the meta-model.

Parameters are fixed per experiment (PRIMARY_PARAMS) and never tuned: fit is a no-op, so a
rule's side can depend on neither train nor test outcomes. Windows are in bars.
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from features.groups import causal_modwt
from features.vol_profile import bar_volatility
from primaries.base import primary, rule_frame
from utils.config import RunConfig


class RulePrimary:
    """Base for fixed rules: subclasses define DEFAULTS and score(df, cfg) -> bar-level Series."""

    DEFAULTS: ClassVar[dict] = {}
    name: str

    def __init__(self, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(
                f"primary {self.name!r}: unknown params {sorted(unknown)}; allowed {sorted(self.DEFAULTS)}"
            )
        self.params = {**self.DEFAULTS, **params}
        self.validate()

    def validate(self) -> None:
        # Integer defaults are bar windows: require a true int >= 2 (JSON 20.0 or a bool must fail here, not in fold 1)
        for k, v in self.params.items():
            if isinstance(self.DEFAULTS[k], int) and (not isinstance(v, int) or isinstance(v, bool) or v < 2):
                raise ValueError(f"primary {self.name!r}: {k} must be an integer >= 2, got {v!r}")

    def fit(self, df, X, labels, weights, cfg, *, val=None) -> "RulePrimary":
        return self

    def score(self, df: pd.DataFrame, cfg: RunConfig) -> pd.Series:
        raise NotImplementedError

    def signal(self, df: pd.DataFrame, X: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
        return rule_frame(self.score(df, cfg), X.index, self.name)


def _sigma(df: pd.DataFrame, cfg: RunConfig) -> np.ndarray:
    return bar_volatility(df["close"], cfg.VOL_SPAN).to_numpy()


def _ratio(num, den, index: pd.Index) -> pd.Series:
    """num / den, keeping num's sign when den == 0 (a flat window must not erase the direction); NaN stays NaN."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.sign(num))
    out[np.isnan(num) | np.isnan(den)] = np.nan
    return pd.Series(out, index=index)


@primary("sma_cross")
class SmaCross(RulePrimary):
    """Trend: long when the fast SMA is above the slow SMA. Score = log(SMA_fast / SMA_slow) / σ_bar."""

    DEFAULTS: ClassVar[dict] = {"fast": 20, "slow": 50}

    def validate(self) -> None:
        super().validate()
        if self.params["fast"] >= self.params["slow"]:
            raise ValueError("sma_cross: fast must be < slow")

    def score(self, df, cfg):
        c = df["close"].astype(float)
        f = c.rolling(self.params["fast"], min_periods=self.params["fast"]).mean()
        s = c.rolling(self.params["slow"], min_periods=self.params["slow"]).mean()
        return _ratio(np.log(f / s), _sigma(df, cfg), df.index)


@primary("bollinger_mr")
class BollingerMr(RulePrimary):
    """Mean reversion: fade the close's z-score vs its rolling mean. Score = −(close − mean_w) / std_w."""

    DEFAULTS: ClassVar[dict] = {"window": 20}

    def score(self, df, cfg):
        c = df["close"].astype(float)
        roll = c.rolling(self.params["window"], min_periods=self.params["window"])
        return _ratio(roll.mean() - c, roll.std(), df.index)


@primary("wavelet_trend")
class WaveletTrend(RulePrimary):
    """
    Trend from the causal MODWT smooth S_J of log close (cfg.WAVELET_FILTER, cfg.WAVELET_J).
    mode="slope": score = (S_J[t] − S_J[t−1]) / σ_bar; mode="level": score = (log close[t] − S_J[t]) / σ_bar.
    """

    DEFAULTS: ClassVar[dict] = {"mode": "slope"}

    def validate(self) -> None:
        if self.params["mode"] not in ("slope", "level"):
            raise ValueError("wavelet_trend: mode must be 'slope' or 'level'")

    def score(self, df, cfg):
        logp = np.log(df["close"].astype(float))
        _, smooth = causal_modwt(logp, cfg.WAVELET_FILTER, cfg.WAVELET_J)
        x = smooth.diff() if self.params["mode"] == "slope" else logp - smooth
        return _ratio(x, _sigma(df, cfg), df.index)


@primary("donchian_breakout")
class DonchianBreakout(RulePrimary):
    """
    Channel of the previous `window` bars (high max / low min, excluding bar t). Score = (close − mid) / half-width:
    > 1 is an upside breakout (long), < −1 a downside breakout (short); inside the channel the side follows
    the close's half of the channel.
    """

    DEFAULTS: ClassVar[dict] = {"window": 20}

    def score(self, df, cfg):
        w = self.params["window"]
        upper = df["high"].rolling(w, min_periods=w).max().shift(1)
        lower = df["low"].rolling(w, min_periods=w).min().shift(1)
        return _ratio(df["close"] - (upper + lower) / 2, (upper - lower) / 2, df.index)
