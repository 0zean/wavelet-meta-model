"""
Run configuration (SPEC §2).

`RunConfig` is passed explicitly (`cfg`) to every function that needs a parameter;
there is no module-level config instance. Build one with:

    RunConfig.for_timeframe("1Hour")          # timeframe-scaled defaults
    RunConfig.for_timeframe("1Day", SEED=7)   # ... with overrides
    RunConfig.legacy_5min()                   # pre-U2 behaviour (bar-based WFO windows, no embargo)
"""

from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd

SEED = 42
SLIPPAGE_PCT = 0.0001  # 1 basis point (1-way)


# random_state is not set here: RunConfig.__post_init__ stamps SEED into every model's params
_XGB_BASE = {
    "n_estimators": 500,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
}
_PARAM_FIELDS = ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")


@dataclass(frozen=True)
class TimeframeDefaults:
    """Timeframe-dependent defaults: barrier/vol scaled in time units, WFO windows in trading days."""

    bars_per_day: int
    vertical_bars: int
    vol_span: int
    hold_overnight: bool
    initial_train_days: int
    val_days: int
    test_days: int


# SPEC §2 table. Holding periods are fixed in clock time (1Min 30 min, 5Min 1 h, 15Min 2 h, 30Min 3 h,
# 1Hour ≈ a session, 1Day 2 weeks). WFO windows grow with the bar length so each split holds
# enough CUSUM events for MIN_TRAIN_EVENTS / MIN_VAL_EVENTS (event rates measured on SPY, SPEC §2).
TIMEFRAME_DEFAULTS: dict[str, TimeframeDefaults] = {
    "1Min": TimeframeDefaults(390, 30, 390, False, 21, 10, 5),
    "5Min": TimeframeDefaults(78, 12, 100, False, 42, 21, 10),
    "15Min": TimeframeDefaults(26, 8, 78, False, 126, 42, 21),
    "30Min": TimeframeDefaults(13, 6, 65, False, 189, 63, 21),
    "1Hour": TimeframeDefaults(7, 6, 70, False, 252, 126, 21),  # 6 full hours + the 15:30 stub bar
    "1Day": TimeframeDefaults(1, 10, 50, True, 1008, 504, 63),
}


@dataclass(frozen=True)
class RunConfig:
    TIMEFRAME: str = "5Min"
    BARS_PER_DAY: int = 78  # 09:30–16:00 in 5-min bars; used for annualisation and plots only

    # Triple Barrier (López de Prado 2018)
    VERTICAL_BARS: int = 12  # max hold = 12 × 5 min = 1 hour
    # False: the vertical barrier is truncated at the entry session's close and events on a
    # session's last bar are skipped. True (daily bars): positions carry across sessions.
    HOLD_OVERNIGHT: bool = False
    BARRIER_MULT: float = 1.0  # horizontal barriers = ±N × σ_bar × √VERTICAL_BARS
    VOL_SPAN: int = 100  # EWM span (bars) for σ_bar
    CUSUM_MULT: float = 1.5  # CUSUM event threshold = N × σ_bar

    # Wavelet (db1 = Haar)
    WAVELET_J: int = 4  # decomposition levels → S4 smoothing
    AR_LAGS: int = 4  # AR features: S_J(t-1) … S_J(t-AR_LAGS)
    WAVELET_FILTER: str = "db1"

    # Technical indicators
    RSI_PERIOD: int = 14
    SIEGEL_WINDOW: int = 20  # Siegel slope rolling window

    # Low-movement day filter (classifier training only)
    # Bars whose trading day has |VWAP_close − VWAP_open| in the bottom
    # LOW_MOVE_PCTILE are excluded from classifier training.
    LOW_MOVE_PCTILE: float = 0.20

    # The single source of randomness: copied into every *_PARAMS["random_state"] on construction,
    # overriding any random_state passed in the params.
    SEED: int = SEED

    # Primary: XGBoost Classifier (direction, binary)
    CLF_PARAMS: dict = field(default_factory=lambda: {**_XGB_BASE, "learning_rate": 0.02, "eval_metric": "auc"})
    # Every event gets a side, so the threshold only splits long from short;
    # 0.5 keeps it unbiased and avoids tuning it on the meta-model's training data.
    CLF_THRESH: float = 0.50

    # Primary: XGBoost Regressor (magnitude = |realised return at the barrier exit|)
    # Trained on absolute error: robust to the fat-tailed exit returns.
    REG_PARAMS: dict = field(
        default_factory=lambda: {
            **_XGB_BASE,
            "learning_rate": 0.02,
            "eval_metric": "mae",
            "objective": "reg:absoluteerror",
        }
    )

    # Meta-label XGBoost Classifier
    META_PARAMS: dict = field(default_factory=lambda: {**_XGB_BASE, "learning_rate": 0.03, "eval_metric": "auc"})
    META_THRESH: float = 0.50  # probability cutoff to trade
    # A trade counts as a success only if it beats round-trip slippage
    META_MIN_RET: float = 2 * SLIPPAGE_PCT

    # Walk-Forward Optimisation
    # Expanding-window WFO.  Each fold:
    #   train = all data up to fold cutoff (grows each fold), purged of events whose exit reaches val
    #   val   = VAL  (primary eval + meta-model training), purged of events whose exit reaches test
    #   test  = TEST (OOS, produces the backtest signals)
    # Window lengths are counted in WINDOW_UNIT: trading sessions ("days") or bars ("bars", legacy).
    WINDOW_UNIT: Literal["days", "bars"] = "days"
    INITIAL_TRAIN: int = 42  # minimum history before the first fold
    VAL: int = 21
    TEST: int = 10
    # Gap (in WINDOW_UNIT) at the end of each fitting split: train/val samples whose exit falls in the
    # last EMBARGO units before the next split are dropped (SPEC §5).
    EMBARGO: int = 1
    MIN_TRAIN_EVENTS: int = 200
    MIN_VAL_EVENTS: int = 100

    # Backtest
    # Zero commission, slippage adverse on every fill (buys up, sells down).
    # SPY @ ~$500, 1 bp ≈ $0.05 ≈ one-way spread / 2
    SLIPPAGE_PCT: float = SLIPPAGE_PCT
    INIT_CASH: float = 10_000
    SIZE: float = 1.0  # fraction of equity per trade

    # Holdout (SPEC §9): the WFO refuses data on/after data.bars.HOLDOUT_START unless True
    ALLOW_HOLDOUT: bool = False

    def __post_init__(self):
        # Own copies of the param dicts (copies made by replace() never share them) with the seed stamped in
        for name in _PARAM_FIELDS:
            object.__setattr__(self, name, {**getattr(self, name), "random_state": self.SEED})
        if self.WINDOW_UNIT not in ("days", "bars"):
            raise ValueError(f"WINDOW_UNIT must be 'days' or 'bars', got {self.WINDOW_UNIT!r}")
        for name in ("INITIAL_TRAIN", "VAL", "TEST", "VERTICAL_BARS", "BARS_PER_DAY"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.EMBARGO < min(self.INITIAL_TRAIN, self.VAL):
            raise ValueError("EMBARGO must be >= 0 and shorter than the train and val windows")
        if self.TIMEFRAME == "1Day" and not self.HOLD_OVERNIGHT:
            raise ValueError("1Day bars are one per session: HOLD_OVERNIGHT must be True")

    @property
    def bars_per_year(self) -> int:
        return 252 * self.BARS_PER_DAY

    @classmethod
    def for_timeframe(cls, timeframe: str, **overrides) -> "RunConfig":
        """Timeframe-scaled defaults (SPEC §2 table), then `overrides` (field names as keys)."""
        if overrides.get("WINDOW_UNIT") == "bars":
            missing = {"INITIAL_TRAIN", "VAL", "TEST", "EMBARGO"} - overrides.keys()
            if missing:
                raise ValueError(f"WINDOW_UNIT='bars' needs explicit bar counts for {sorted(missing)}")
        try:
            d = TIMEFRAME_DEFAULTS[timeframe]
        except KeyError:
            raise ValueError(f"Unknown timeframe {timeframe!r}; expected one of {list(TIMEFRAME_DEFAULTS)}") from None
        base = cls(
            TIMEFRAME=timeframe,
            BARS_PER_DAY=d.bars_per_day,
            VERTICAL_BARS=d.vertical_bars,
            VOL_SPAN=d.vol_span,
            HOLD_OVERNIGHT=d.hold_overnight,
            INITIAL_TRAIN=d.initial_train_days,
            VAL=d.val_days,
            TEST=d.test_days,
        )
        return base.replace(**overrides)

    @classmethod
    def legacy_5min(cls, **overrides) -> "RunConfig":
        """Pre-U2 behaviour: 5Min, bar-based windows 2000/1000/500, no embargo (regression invariant)."""
        return cls.for_timeframe(
            "5Min", WINDOW_UNIT="bars", INITIAL_TRAIN=2000, VAL=1000, TEST=500, EMBARGO=0, **overrides
        )

    def replace(self, **overrides) -> "RunConfig":
        """Copy with fields replaced (validated, and the XGBoost params reseeded, like any construction)."""
        return replace(self, **overrides)

    def holdout_guard(self, index: pd.DatetimeIndex) -> None:
        """Raise if `index` reaches the holdout and ALLOW_HOLDOUT is off (SPEC §9)."""
        from data.bars import HOLDOUT_START, HoldoutError

        if self.ALLOW_HOLDOUT or len(index) == 0:
            return
        start = HOLDOUT_START if index.tz is None else HOLDOUT_START.tz_localize(index.tz)
        if index[-1] >= start:
            raise HoldoutError(
                f"data runs to {index[-1]}, past HOLDOUT_START {HOLDOUT_START.date()}; "
                "labels would resolve inside the holdout (set ALLOW_HOLDOUT only for the final evaluation)"
            )
