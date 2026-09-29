import pandas as pd
from sklearn.metrics import roc_auc_score

from models.zoo import ZooModel, describe, inner_cv, make_model
from utils.config import RunConfig


def fit_primary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    active_mask_train: pd.Series,
    cfg: RunConfig,
    spans: pd.DataFrame,
) -> ZooModel:
    """
    Train the direction classifier (cfg.PRIMARY_MODEL; "legacy" = XGBoost CLF_PARAMS) on active-day events only.
    Labels: 1 = long signal, 0 = short signal  (triple-barrier ±1 → 0/1 binary;
    events with a zero return carry no direction and are excluded).

    Args:
        X_train (pd.DataFrame): Training features (purged event rows)
        y_train (pd.Series): Training labels {-1, 0, 1}
        w_train (pd.Series): Sample weights (average uniqueness)
        X_val (pd.DataFrame | None): Validation features (logging only; None skips it)
        y_val (pd.Series | None): Validation labels {-1, 0, 1}
        active_mask_train (pd.Series): Active-day mask for training rows
        cfg (RunConfig): Run configuration.
        spans (pd.DataFrame): Training label rows (`entry_pos`, `exit_pos`) for the zoo's inner purged CV.

    Returns:
        ZooModel: Fitted classifier; predict_proba gives P(long).
    """
    keep = y_train.isin([-1, 1]) & active_mask_train
    y_tr = y_train.loc[keep].map({-1: 0, 1: 1})

    clf = make_model(cfg.PRIMARY_MODEL, cfg, role="primary")
    cv = None if cfg.PRIMARY_MODEL == "legacy" else inner_cv(spans.loc[y_tr.index], cfg)
    clf.fit(X_train.loc[keep], y_tr, w_train.loc[keep], cv)
    if cfg.PRIMARY_MODEL != "legacy":
        print(f"[CLF]  {describe(clf)}")

    if X_val is not None:
        keep_vl = y_val.isin([-1, 1])
        y_vl = y_val.loc[keep_vl].map({-1: 0, 1: 1})
        if y_vl.nunique() == 2:
            auc = roc_auc_score(y_vl, clf.predict_proba(X_val.loc[keep_vl]))
            print(f"[CLF]  Val AUC={auc:.4f}  (train events={len(y_tr)})")
    return clf
