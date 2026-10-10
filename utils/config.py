"""
Run configuration (SPEC §2).

`RunConfig` is passed explicitly (`cfg`) to every function that needs a parameter;
there is no module-level config instance. Build one with:

    RunConfig.for_timeframe("1Hour")          # timeframe-scaled defaults
    RunConfig.for_timeframe("1Day", SEED=7)   # ... with overrides
    RunConfig.legacy_5min()                   # pre-U2 behaviour (bar windows, no embargo, legacy features)
"""

import copy
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

# SPEC §3: the default feature zoo (features/groups.py). 'wavelet_core' is required in every set;
# 'intraday' is skipped on 1Day; 'wavelet_ext', 'structural', 'calendar' and 'cross_asset' are opt-in.
DEFAULT_FEATURE_GROUPS = (
    "wavelet_core",
    "trend",
    "mean_reversion",
    "volatility",
    "microstructure",
    "intraday",
    "fracdiff",
)
# SPEC §11.2 (U12): the hyper-parameters U11's purged-CV searches chose most often, fixed (one fit per window)
DEFAULT_ZOO_FIXED_PARAMS = {
    "rf_ldp_fast": {"max_features": 1},
    "logit_l2": {"C": 0.1},
    "xgb": {"max_depth": 2},
    "lightgbm": {"num_leaves": 7},
}


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
    pwfo_is_grid: tuple[int, ...] = (504, 756)  # SPEC §11.2: U11 Stage C found no cadence effect; OOS 21 for all


# SPEC §2 table. Holding periods are fixed in clock time (1Min 30 min, 5Min 1 h, 15Min 2 h, 30Min 3 h,
# 1Hour ≈ a session, 1Day 2 weeks). WFO windows grow with the bar length so each split holds
# enough CUSUM events for MIN_TRAIN_EVENTS / MIN_VAL_EVENTS (event rates measured on SPY, SPEC §2).
TIMEFRAME_DEFAULTS: dict[str, TimeframeDefaults] = {
    "1Min": TimeframeDefaults(390, 30, 390, False, 21, 10, 5),
    "5Min": TimeframeDefaults(78, 12, 100, False, 42, 21, 10),
    "15Min": TimeframeDefaults(26, 8, 78, False, 126, 42, 21),
    "30Min": TimeframeDefaults(13, 6, 65, False, 189, 63, 21),
    "1Hour": TimeframeDefaults(7, 6, 70, False, 252, 126, 21),  # 6 full hours + the 15:30 stub bar
    "1Day": TimeframeDefaults(1, 10, 50, True, 1008, 504, 63, (1260, 1512)),
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

    # Event samplers, exit models, intraday volatility structure (SPEC §13–§14, U14; features/events.py, exits.py,
    # vol_profile.py). VOL_PROFILE "tod": σ_bar is time-of-day aware (a VolProfile fit per fold on train bars), so
    # CUSUM / DC thresholds and barrier widths in slot b scale with s(b); intraday only. EVENT_SAMPLER: "cusum" (§3),
    # "dc" (directional change, EVENT_PARAMS dc_mult), "schedule" (EVENT_PARAMS entry_times, days, gate).
    # EXIT_MODEL: "triple_barrier" (§3), "time" (EXIT_PARAMS exit_time | hold_bars), "hysteresis" (beta, max_bars).
    VOL_PROFILE: Literal["none", "tod"] = "none"
    EVENT_SAMPLER: Literal["cusum", "dc", "schedule"] = "cusum"
    EVENT_PARAMS: dict = field(default_factory=dict)
    EXIT_MODEL: Literal["triple_barrier", "time", "hysteresis"] = "triple_barrier"
    EXIT_PARAMS: dict = field(default_factory=dict)

    # Wavelet (db1 = Haar)
    WAVELET_J: int = 4  # decomposition levels → S4 smoothing
    AR_LAGS: int = 4  # AR features: S_J(t-1) … S_J(t-AR_LAGS)
    WAVELET_FILTER: str = "db1"

    # Technical indicators
    RSI_PERIOD: int = 14
    SIEGEL_WINDOW: int = 20  # Siegel slope rolling window

    # Feature zoo (SPEC §3). None = the legacy pre-U3 feature matrix (legacy_5min regression only).
    FEATURE_GROUPS: tuple[str, ...] | None = DEFAULT_FEATURE_GROUPS
    # "cmda": clustered MDA on each fold's purged train events keeps a feature subset
    # (wavelet_core always kept); "none": use every column.
    FEATURE_SELECTION: Literal["none", "cmda"] = "none"
    CMDA_TREES: int = 100  # random-forest size used to score permutation importance
    CMDA_SPLITS: int = 4  # purged k-fold splits inside the train window
    CV_EMBARGO_PCT: float = 0.01  # purged-CV embargo as a fraction of the train bars (SPEC §5)

    # Validation toolkit (SPEC §5–6, U5)
    CPCV_GROUPS: int = 10  # CPCV N contiguous groups
    CPCV_TEST_GROUPS: int = 2  # CPCV k test groups per split → C(N,k) splits, C(N−1,k−1) paths
    PBO_BLOCKS: int = 16  # CSCV blocks S
    SELECTION_METRIC: Literal["neg_log_loss", "brier"] = "neg_log_loss"

    # Model zoo (SPEC §4, U6; models/zoo.py). "legacy" = the pre-U6 fixed-parameter XGBoost (no search, no
    # calibration); any other zoo model runs its purged-CV HP search + calibration inside its fitting rows.
    META_MODEL: str = "legacy"
    # Direction classifier of the ml_xgb primary; must stay "legacy" for rule primaries (they fit nothing).
    PRIMARY_MODEL: str = "legacy"
    ZOO_CV_SPLITS: int = 4  # inner PurgedKFold splits for the zoo's HP search and calibration choice
    # Meta-model training rows: "val" = the val split with the train-fit primary (pre-U6); "oof" = train+val
    # events with purged k-fold out-of-fold primary signals (primary refit on train+val for test).
    META_TRAIN: Literal["val", "oof"] = "val"
    # SPEC §11.2 (U12). Zoo model name → fixed hyper-parameters (keys = the model's grid keys): a listed model skips
    # its purged-CV grid search. {} = every model searches its grid (U6–U11).
    ZOO_FIXED_PARAMS: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_ZOO_FIXED_PARAMS.items()})
    # Meta-model probability calibration (zoo models; `legacy` is uncalibrated): "crossfit" = sigmoid vs isotonic
    # cross-fitted on the purged-CV OOF predictions (U6–U11); "rolling" = Platt map fit on the previous windows' OOS
    # (raw p, meta-label) pairs resolved inside the current fitting span, "crossfit" while fewer than MIN_VAL_EVENTS
    # pairs (or one class) are available; "none" = raw probabilities.
    CALIBRATION: Literal["crossfit", "rolling", "none"] = "rolling"
    # Train-window OOF meta-probabilities for the ecdf / kelly_capped sizers: "refit" = a fresh meta-model per purged
    # split (U7–U11); "reuse" = the fitted zoo meta-model's own OOF predictions through its calibrator.
    OOF_META: Literal["refit", "reuse"] = "reuse"

    # Low-movement day filter (classifier training only)
    # Bars whose trading day has |VWAP_close − VWAP_open| in the bottom
    # LOW_MOVE_PCTILE are excluded from classifier training.
    LOW_MOVE_PCTILE: float = 0.20

    # The single source of randomness: copied into every *_PARAMS["random_state"] on construction,
    # overriding any random_state passed in the params.
    SEED: int = SEED

    # Primary signal (SPEC §4, primaries/): "ml_xgb" (the classifier + regressor below), "sma_cross",
    # "bollinger_mr", "wavelet_trend", "donchian_breakout". PRIMARY_PARAMS are the rule's fixed
    # parameters (constructor kwargs; never tuned); ml_xgb takes none and uses CLF_/REG_PARAMS.
    PRIMARY: str = "ml_xgb"
    PRIMARY_PARAMS: dict = field(default_factory=dict)

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

    # Power Walk-Forward (SPEC §6, U9; wfo/pwfo.py). Windows in WINDOW_UNIT (exchange-calendar sessions in the
    # runner): IS ∈ PWFO_IS_GRID × OOS ∈ PWFO_OOS_GRID, retraining every OOS. Each IS window splits into train and a
    # final val of round(PWFO_VAL_FRAC · IS) units (None → VAL / (INITIAL_TRAIN + VAL), the WFO's own ratio).
    # Defaults SPEC §11.2 (for_timeframe: 1Day IS (1260, 1512)); U9–U11 ran (63, 126, 252, 504) × (5, 10, 21, 63)
    PWFO_IS_GRID: tuple[int, ...] = (504, 756)
    PWFO_OOS_GRID: tuple[int, ...] = (21,)
    PWFO_EXPANDING: bool = False  # True: every IS window starts at the first unit
    PWFO_PARTIAL_LAST: bool = False  # True: a last, shorter OOS window runs to the end of the data (U11 Stage E)
    PWFO_VAL_FRAC: float | None = None
    PWFO_DEFAULT: tuple[int, int] = (504, 21)  # (IS, OOS) used during the nested-selection burn-in (U9–U11: (252, 10))
    PWFO_MIN_WINDOWS: int = 50  # fewer OOS windows flags a combo as statistically weak (Meyers)
    PWFO_WFE_MIN_T: float = 2.0  # WFE is reported only when the mean IS figure is > 0 with this t-statistic
    # The PWFO stream (SPEC §11.2): "nested" = walk-forward selection of the combo (U9–U11); "average" = equal-weight
    # mean of the run combos' daily OOS returns on their common span (no burn-in, no selection)
    PWFO_COMBINE: Literal["nested", "average"] = "average"
    SELECT_EVERY: int = 10  # nested selection: re-pick the combo every SELECT_EVERY trading days ...
    SELECT_LOOKBACK: int = 126  # ... by its Sharpe over the prior SELECT_LOOKBACK days of OOS returns

    # Backtest
    # Zero commission, slippage adverse on every fill (buys up, sells down).
    # SPY @ ~$500, 1 bp ≈ $0.05 ≈ one-way spread / 2
    SLIPPAGE_PCT: float = SLIPPAGE_PCT
    INIT_CASH: float = 10_000
    SIZE: float = 1.0  # fraction of equity per trade at bet size m = 1 (notional = SIZE · m · equity)

    # Bet sizing (SPEC §7, U7; sizing/). SIZER maps the calibrated meta-probability to m ∈ [0, 1]: "fixed" (m = 1,
    # pre-U7), "linear", "ldp_sigmoid", "ecdf", "kelly_capped" (the last two fit on train-window OOF meta-probs).
    SIZER: str = "fixed"
    SIZE_STEP: float = 0.1  # sizes are discretized to multiples of this (0 = off)
    KELLY_FRACTION: float = 0.25  # λ of kelly_capped
    # "single": one position at a time, events arriving while it is open are skipped (pre-U7);
    # "average": every approved bet is live over its own barrier window, position = discretized mean signed
    # size of the active bets (López de Prado §10.4), resized at each bet's entry / exit.
    POSITION_MODE: Literal["single", "average"] = "single"

    # Risk layer (SPEC §7, U8; risk/): a named profile in risk.profiles.PROFILES. "none" = the pre-U8 backtest;
    # any other routes the backtest through the portfolio simulator (vol target, caps, drawdown throttle, daily
    # loss gate, spread costs), which needs POSITION_MODE="single".
    RISK_PROFILE: str = "none"
    # Transaction costs (SPEC §19, U13; risk.costs.fill_costs): "quotes" = half-spread from the quotes table
    # (data/costs/quotes_half_spread.csv) by year and time of day, auction proxies for open / close auction fills,
    # + SLIPPAGE_PCT off-auction; "cs" = SLIPPAGE_PCT + trailing Corwin–Schultz half-spread (U8–U11 "standard");
    # "slippage" = SLIPPAGE_PCT only (U7–U11 default). Any but "slippage" backtests through the portfolio simulator.
    COST_MODEL: Literal["slippage", "cs", "quotes"] = "quotes"

    # Holdout (SPEC §9): the WFO refuses data on/after data.bars.HOLDOUT_START unless True
    ALLOW_HOLDOUT: bool = False

    def __post_init__(self):
        # Own copies of the param dicts (copies made by replace() never share them) with the seed stamped in
        for name in _PARAM_FIELDS:
            object.__setattr__(self, name, {**getattr(self, name), "random_state": self.SEED})
        object.__setattr__(self, "PRIMARY_PARAMS", dict(self.PRIMARY_PARAMS))
        for name in ("EVENT_PARAMS", "EXIT_PARAMS"):
            object.__setattr__(self, name, copy.deepcopy(dict(getattr(self, name))))
        if self.FEATURE_GROUPS is not None:
            object.__setattr__(self, "FEATURE_GROUPS", tuple(self.FEATURE_GROUPS))
        if self.FEATURE_SELECTION not in ("none", "cmda"):
            raise ValueError(f"FEATURE_SELECTION must be 'none' or 'cmda', got {self.FEATURE_SELECTION!r}")
        if self.FEATURE_SELECTION == "cmda" and self.FEATURE_GROUPS is None:
            raise ValueError("FEATURE_SELECTION='cmda' needs FEATURE_GROUPS (the legacy matrix has no wavelet_core)")
        if not (self.CPCV_GROUPS >= 2 and 1 <= self.CPCV_TEST_GROUPS < self.CPCV_GROUPS):
            raise ValueError(
                f"need CPCV_GROUPS >= 2 and 1 <= CPCV_TEST_GROUPS < CPCV_GROUPS; got {self.CPCV_GROUPS}, {self.CPCV_TEST_GROUPS}"
            )
        if self.PBO_BLOCKS < 2 or self.PBO_BLOCKS % 2:
            raise ValueError(f"PBO_BLOCKS must be an even integer >= 2; got {self.PBO_BLOCKS}")
        if self.SELECTION_METRIC not in ("neg_log_loss", "brier"):
            raise ValueError(f"SELECTION_METRIC must be 'neg_log_loss' or 'brier', got {self.SELECTION_METRIC!r}")
        from models.zoo import REGISTRY as ZOO

        for name in ("META_MODEL", "PRIMARY_MODEL"):
            if getattr(self, name) not in ZOO:
                raise ValueError(f"{name} must be one of {sorted(ZOO)}, got {getattr(self, name)!r}")
        if self.PRIMARY_MODEL != "legacy" and self.PRIMARY != "ml_xgb":
            raise ValueError(f"PRIMARY_MODEL applies only to the ml_xgb primary (PRIMARY={self.PRIMARY!r})")
        if self.META_TRAIN not in ("val", "oof"):
            raise ValueError(f"META_TRAIN must be 'val' or 'oof', got {self.META_TRAIN!r}")
        object.__setattr__(self, "ZOO_FIXED_PARAMS", {k: dict(v) for k, v in dict(self.ZOO_FIXED_PARAMS).items()})
        for name, params in self.ZOO_FIXED_PARAMS.items():
            grid = getattr(ZOO.get(name), "grid", None)
            if grid is None:
                raise ValueError(f"ZOO_FIXED_PARAMS: {name!r} is not a zoo model with a grid; expected one of "
                                 f"{sorted(n for n, c in ZOO.items() if hasattr(c, 'grid'))}")  # fmt: skip
            if set(params) != set(grid):
                raise ValueError(f"ZOO_FIXED_PARAMS[{name!r}] must set exactly {sorted(grid)}, got {sorted(params)}")
        for name, allowed in (("CALIBRATION", ("crossfit", "rolling", "none")), ("OOF_META", ("refit", "reuse")),
                              ("PWFO_COMBINE", ("nested", "average"))):  # fmt: skip
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        from sizing import REGISTRY as SIZERS

        if self.SIZER not in SIZERS:
            raise ValueError(f"SIZER must be one of {sorted(SIZERS)}, got {self.SIZER!r}")
        if not (
            self.SIZE_STEP == 0
            or (0 < self.SIZE_STEP <= 1 and abs(1 / self.SIZE_STEP - round(1 / self.SIZE_STEP)) < 1e-9)
        ):
            raise ValueError(f"SIZE_STEP must be 0 (off) or 1/k for an integer k (m = 1 stays 1), got {self.SIZE_STEP}")
        if not 0 < self.KELLY_FRACTION <= 1:
            raise ValueError(f"KELLY_FRACTION must be in (0, 1], got {self.KELLY_FRACTION}")
        if self.POSITION_MODE not in ("single", "average"):
            raise ValueError(f"POSITION_MODE must be 'single' or 'average', got {self.POSITION_MODE!r}")
        from risk.profiles import get_profile

        if get_profile(self.RISK_PROFILE).active and self.POSITION_MODE != "single":
            raise ValueError(f"RISK_PROFILE={self.RISK_PROFILE!r} needs POSITION_MODE='single'")
        if self.COST_MODEL not in ("slippage", "cs", "quotes"):
            raise ValueError(f"COST_MODEL must be one of ('slippage', 'cs', 'quotes'), got {self.COST_MODEL!r}")
        if self.COST_MODEL != "slippage" and self.POSITION_MODE != "single":
            raise ValueError(f"COST_MODEL={self.COST_MODEL!r} needs POSITION_MODE='single' (the portfolio simulator)")
        if not 0 < self.META_THRESH < 1:
            raise ValueError(f"META_THRESH must be in (0, 1), got {self.META_THRESH}")
        if self.ZOO_CV_SPLITS < 2:
            raise ValueError("ZOO_CV_SPLITS must be >= 2")
        if self.WINDOW_UNIT not in ("days", "bars"):
            raise ValueError(f"WINDOW_UNIT must be 'days' or 'bars', got {self.WINDOW_UNIT!r}")
        for name in ("INITIAL_TRAIN", "VAL", "TEST", "VERTICAL_BARS", "BARS_PER_DAY"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.EMBARGO < min(self.INITIAL_TRAIN, self.VAL):
            raise ValueError("EMBARGO must be >= 0 and shorter than the train and val windows")
        for name in ("PWFO_IS_GRID", "PWFO_OOS_GRID"):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
            if not getattr(self, name) or min(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a non-empty tuple of positive lengths")
        object.__setattr__(self, "PWFO_DEFAULT", tuple(int(v) for v in self.PWFO_DEFAULT))
        if self.PWFO_VAL_FRAC is not None and not 0 < self.PWFO_VAL_FRAC < 1:
            raise ValueError(f"PWFO_VAL_FRAC must be in (0, 1) or None, got {self.PWFO_VAL_FRAC}")
        if not self.PWFO_WFE_MIN_T >= 0:
            raise ValueError(f"PWFO_WFE_MIN_T must be >= 0, got {self.PWFO_WFE_MIN_T}")
        for name in ("SELECT_EVERY", "SELECT_LOOKBACK", "PWFO_MIN_WINDOWS"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.TIMEFRAME == "1Day" and not self.HOLD_OVERNIGHT:
            raise ValueError("1Day bars are one per session: HOLD_OVERNIGHT must be True")
        if self.VOL_PROFILE not in ("none", "tod"):
            raise ValueError(f"VOL_PROFILE must be 'none' or 'tod', got {self.VOL_PROFILE!r}")
        if self.VOL_PROFILE == "tod" and self.TIMEFRAME == "1Day":
            raise ValueError("VOL_PROFILE='tod' is a time-of-day profile: intraday timeframes only")
        from features.events import event_params
        from features.exits import exit_params

        ev = event_params(self)
        xp = exit_params(self)
        if self.EVENT_SAMPLER == "schedule" and ev["entry_times"] == ["close"]:  # SPEC §16: market-on-close entries
            if self.EXIT_MODEL != "time" or xp["hold_bars"] is not None:
                raise ValueError("a market-on-close entry (entry_times ['close']) needs EXIT_MODEL='time' with an "
                                 "exit_time")  # fmt: skip
            if xp["exit_time"] not in ("open", "next") and not xp["exit_session"]:
                raise ValueError("a market-on-close entry exits in a later session: set EXIT_PARAMS exit_session >= 1")
            if self.POSITION_MODE != "single":
                raise ValueError("market-on-close entries are simulated with POSITION_MODE='single' only")
        from primaries import REGISTRY as PRIMARIES

        allowed_tf = getattr(PRIMARIES.get(self.PRIMARY), "TIMEFRAMES", None)
        if allowed_tf is not None and self.TIMEFRAME not in allowed_tf:
            raise ValueError(f"primary {self.PRIMARY!r} runs on {list(allowed_tf)}, not {self.TIMEFRAME!r}")
        if (self.EVENT_SAMPLER == "schedule" and self.TIMEFRAME != "1Day" and not self.HOLD_OVERNIGHT
                and self.EXIT_MODEL != "time" and any(t in ("open", "09:30") for t in ev["entry_times"])):  # fmt: skip
            raise ValueError(f"a 09:30 schedule entry decides at the previous session's close: EXIT_MODEL="
                             f"{self.EXIT_MODEL!r} drops cross-session entries unless HOLD_OVERNIGHT (use 'time')")  # fmt: skip

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
            PWFO_IS_GRID=d.pwfo_is_grid,
            PWFO_DEFAULT=(d.pwfo_is_grid[0], 21),
        )
        return base.replace(**overrides)

    @classmethod
    def legacy_5min(cls, **overrides) -> "RunConfig":
        """Pre-U2 behaviour: 5Min, bar windows 2000/1000/500, no embargo, legacy features (regression invariant)."""
        return cls.for_timeframe(
            "5Min",
            **{"WINDOW_UNIT": "bars", "INITIAL_TRAIN": 2000, "VAL": 1000, "TEST": 500, "EMBARGO": 0}
            | {"FEATURE_GROUPS": None, "COST_MODEL": "slippage"}
            | overrides,
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
