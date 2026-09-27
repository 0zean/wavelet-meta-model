import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from utils.config import config


def fit_primary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    active_mask_train: pd.Series,
) -> xgb.XGBClassifier:
    """
    Train the direction classifier (XGBoost) on active-day events only.
    Labels: 1 = long signal, 0 = short signal  (triple-barrier ±1 → 0/1 binary;
    events with a zero return carry no direction and are excluded).

    Args:
        X_train (pd.DataFrame): Training features (purged event rows)
        y_train (pd.Series): Training labels {-1, 0, 1}
        w_train (pd.Series): Sample weights (average uniqueness)
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels {-1, 0, 1}
        active_mask_train (pd.Series): Active-day mask for training rows

    Returns:
        xgb.XGBClassifier: Fitted classifier
    """
    keep = y_train.isin([-1, 1]) & active_mask_train
    y_tr = y_train.loc[keep].map({-1: 0, 1: 1})

    clf = xgb.XGBClassifier(**config.CLF_PARAMS)
    clf.fit(X_train.loc[keep], y_tr, sample_weight=w_train.loc[keep], verbose=False)

    keep_vl = y_val.isin([-1, 1])
    y_vl = y_val.loc[keep_vl].map({-1: 0, 1: 1})
    if y_vl.nunique() == 2:
        auc = roc_auc_score(y_vl, clf.predict_proba(X_val.loc[keep_vl])[:, 1])
        print(f"[CLF]  Val AUC={auc:.4f}  (train events={len(y_tr)})")
    return clf
