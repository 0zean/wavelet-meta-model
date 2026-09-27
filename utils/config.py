from dataclasses import dataclass, field

SEED = 42
SLIPPAGE_PCT = 0.0001  # 1 basis point (1-way)
_XGB_BASE = {
    "n_estimators": 500,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "tree_method": "hist",
    "device": "cpu",
}


@dataclass(frozen=True)
class Config:
    BARS_PER_DAY: int = 78  # 09:30–16:00 in 5-min bars

    # Triple Barrier (López de Prado 2018)
    VERTICAL_BARS: int = 12  # max hold = 12 × 5 min = 1 hour (truncated at the session close)
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
    #   val   = VAL_BARS  (primary eval + meta-model training), purged of events whose exit reaches test
    #   test  = TEST_BARS (OOS, produces the backtest signals)
    INITIAL_TRAIN_BARS: int = 2000  # minimum bars before first fold
    VAL_BARS: int = 1000
    TEST_BARS: int = 500
    MIN_TRAIN_EVENTS: int = 200
    MIN_VAL_EVENTS: int = 100

    # Backtest
    # Zero commission, slippage adverse on every fill (buys up, sells down).
    # SPY @ ~$500, 1 bp ≈ $0.05 ≈ one-way spread / 2
    SLIPPAGE_PCT: float = SLIPPAGE_PCT
    INIT_CASH: float = 10_000
    SIZE: float = 1.0  # fraction of equity per trade


config = Config()
