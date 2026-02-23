class Config:
    # Triple Barrier
    VERTICAL_BARS = 12  # max hold = 12 × 5 min = 1 hour
    BARRIER_MULT = 1.0  # horizontal barriers = ±N × rolling σ
    VOL_LOOKBACK = 78   # rolling vol window ≈ 1 trading day (78×5m)

    # Wavelet (db1 = Haar)
    WAVELET_J = 4       # decomposition levels → S4 smoothing
    AR_LAGS = 4         # AR features: S4(t-1) … S4(t-4)
    WAVELET_FILTER = "db1"

    # Technical indicators
    RSI_PERIOD = 14
    SIEGEL_WINDOW = 20  # Siegel slope rolling window

    # Low-movement day filter (classifier training only)
    # Bars whose trading day has |VWAP_close − VWAP_open| in the bottom
    # LOW_MOVE_PCTILE are excluded from classifier training.
    LOW_MOVE_PCTILE = 0.20

    # Primary: XGBoost Classifier (direction, binary)
    CLF_PARAMS = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        device="cpu",
    )
    # Threshold tuned on val to achieve ≥ target recall before meta-labeling
    CLF_RECALL_TARGET = 0.70

    # Primary: XGBoost Regressor (magnitude = |expected % move|)
    # NOTE: "optimised on directional accuracy" is non-differentiable so we
    # train on MAE (a good proxy: underestimates tend to flip direction) and
    # report directional accuracy as an evaluation metric post-hoc. This is the
    # principled choice — do not use a custom discrete loss inside XGBoost as
    # gradient approximations break for 0/1 step functions.
    REG_PARAMS = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mae",
        random_state=42,
        tree_method="hist",
        device="cpu",
        objective="reg:squarederror",
    )

    # Meta-label XGBoost Classifier
    META_PARAMS = dict(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        device="cpu",
    )
    META_THRESH = 0.50  # probability cutoff to trade

    # Walk-Forward Optimisation
    # Expanding-window WFO.  Each fold:
    #   in_sample  = all data up to fold cutoff  (grows each fold)
    #   val        = VAL_BARS  (for early-stopping + meta-model training)
    #   test       = TEST_BARS (OOS, produces the backtest signals)
    INITIAL_TRAIN_BARS = 2000  # minimum bars before first fold
    VAL_BARS = 500
    TEST_BARS = 500

    # Backtest
    # Zero commission, realistic slippage modelled as:
    # entry at next-bar OPEN ± SLIPPAGE_PCT × price
    # SPY @ ~$500, 1 bp ≈ $0.05 ≈ one-way spread / 2
    SLIPPAGE_PCT = 0.0001  # 1 basis point (1-way)
    INIT_CASH = 100_000
    SIZE = 1.0  # fraction of equity per trade


config = Config()
