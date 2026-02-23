import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score

from utils.config import config


def threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = config.CLF_RECALL_TARGET,
) -> float:
    """
    Sweep probability thresholds and return the highest threshold that still
    achieves recall ≥ target_recall on the validation set.
    Higher threshold = more precise but lower recall; we prefer recall.

    Args:
        y_true (np.ndarray): True labels
        y_prob (np.ndarray): Predicted probabilities
        target_recall (float, optional): Target recall. Defaults to config.CLF_RECALL_TARGET.

    Returns:
        float: Threshold for recall
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays ordered by decreasing threshold;
    # recall increases as threshold decreases.
    # We want the *highest* threshold where recall >= target.
    valid = thresholds[recall[:-1] >= target_recall]
    if len(valid) == 0:
        return 0.5  # fallback
    return float(valid[-1])


def fit_primary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    active_mask_train: pd.Series,
) -> tuple[xgb.XGBClassifier, float]:
    """
    Train the direction classifier (XGBoost) on active-day bars only.
    Labels: 1 = long signal, 0 = short signal  (triple-barrier ±1 → 0/1 binary;
    label=0 from TB is excluded - we only train on confirmed directional moves).

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        active_mask_train (pd.Series): Active mask for training

    Returns:
        tuple[xgb.XGBClassifier, float]: Fitted classifier and recall-optimised threshold
    """
    # Map {-1→0, 1→1} and exclude TB label=0 (vertical barrier / no trend)
    mask_dir = y_train.isin([-1, 1])
    y_bin = y_train[mask_dir].map({-1: 0, 1: 1})
    X_tr = X_train.loc[mask_dir & active_mask_train]
    y_tr = y_bin.loc[mask_dir & active_mask_train]

    mask_val = y_val.isin([-1, 1])
    y_val_b = y_val[mask_val].map({-1: 0, 1: 1})
    X_vl = X_val.loc[mask_val]

    clf = xgb.XGBClassifier(**config.CLF_PARAMS)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_vl, y_val_b)],
        verbose=False,
    )

    val_prob = clf.predict_proba(X_vl)[:, 1]
    auc = roc_auc_score(y_val_b, val_prob)
    thresh = threshold_for_recall(y_val_b.values, val_prob)
    print(f"[CLF]  Val AUC={auc:.4f}  Recall-thresh={thresh:.4f}")
    return clf, thresh
