import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import RunConfig


def fit_primary_regressor(
    X_train: pd.DataFrame,
    ret_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame | None,
    ret_val: pd.Series | None,
    cfg: RunConfig,
) -> xgb.XGBRegressor:
    """
    Train a magnitude regressor on all training events (not day-filtered).

    Target: |realised return| at the triple-barrier exit, so the regressor
    estimates the size of the move irrespective of direction; direction comes
    from the classifier.

    Args:
        X_train (pd.DataFrame): Training features (purged event rows).
        ret_train (pd.Series): Realised barrier-exit returns for the training events.
        w_train (pd.Series): Sample weights (average uniqueness).
        X_val (pd.DataFrame | None): Validation features (logging only; None skips it).
        ret_val (pd.Series | None): Realised barrier-exit returns for the validation events.
        cfg (RunConfig): Run configuration.

    Returns:
        xgb.XGBRegressor: Trained regressor.
    """
    reg = xgb.XGBRegressor(**cfg.REG_PARAMS)
    reg.fit(X_train, ret_train.abs(), sample_weight=w_train, verbose=False)

    if X_val is None:
        return reg
    val_pred = reg.predict(X_val)
    mae = np.abs(val_pred - ret_val.abs()).mean()
    naive = np.abs(ret_train.abs().median() - ret_val.abs()).mean()
    print(f"[REG]  Val MAE={mae:.6f}  (train-median baseline={naive:.6f})")
    return reg
