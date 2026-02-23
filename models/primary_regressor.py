import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import config


def fit_primary_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> xgb.XGBRegressor:
    """
    Train a magnitude regressor on the FULL dataset (all labels, not filtered).

    Target: the *realised* log-return magnitude over the actual exit bar
    (whichever barrier was hit first in triple-barrier labeling).

    We train on |log_return| so the regressor estimates position size
    irrespective of direction; direction comes from the classifier.

    Directional accuracy is evaluated post-hoc.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels (triple-barrier labels {-1, 0, 1})
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        df_train (pd.DataFrame): Training data (Raw OHLCV data).
        df_val (pd.DataFrame): Validation data.

    Returns:
        xgb.XGBRegressor: Trained regressor.
    """
    # Realised log return at the triple-barrier exit (approximated here as
    # vertical_bars-ahead log return; for a production system, store the actual
    # exit bar from the labeling loop above).
    V = config.VERTICAL_BARS
    close_tr = df_train["close"]
    close_vl = df_val["close"]

    # Forward log return (absolute magnitude) — strictly look-ahead for TARGET
    # only, never used as a feature.
    fwd_tr = np.log(close_tr.shift(-V) / close_tr).abs().reindex(y_train.index)
    fwd_vl = np.log(close_vl.shift(-V) / close_vl).abs().reindex(y_val.index)

    valid_tr = fwd_tr.dropna().index
    valid_vl = fwd_vl.dropna().index

    reg = xgb.XGBRegressor(**config.REG_PARAMS)
    reg.fit(
        X_train.loc[valid_tr],
        fwd_tr.loc[valid_tr],
        eval_set=[(X_val.loc[valid_vl], fwd_vl.loc[valid_vl])],
        verbose=False,
    )

    # Evaluate directional accuracy: sign(predicted log-ret × label) == 1
    val_pred = reg.predict(X_val.loc[valid_vl])
    val_lbl = y_val.reindex(valid_vl)
    dir_acc = (np.sign(val_lbl) == np.sign(val_pred)).mean()
    print(f"[REG]  Val dir-acc={dir_acc:.4f}")
    return reg
