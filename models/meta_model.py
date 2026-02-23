import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

from utils.config import config


def make_meta_labels(primary_preds: pd.DataFrame, y_true: pd.Series, thresh: float) -> pd.Series:
    """
    Meta-label = 1 if the primary model's direction prediction matches the
    actual triple-barrier label, else 0.

    We define "correct" as:
    sign(primary_direction) == sign(true_label)  for non-zero labels.
    Bars where true_label == 0 (vertical barrier hit) are excluded — the
    meta-model should only learn from cases where there was a clear outcome.

    NOTE: Meta-labels are computed on the VALIDATION split using primary model
    predictions from a model trained on TRAIN — no leakage.

    Args:
        primary_preds (pd.DataFrame): Primary model predictions
        y_true (pd.Series): Triple-barrier labels {-1, 0, 1}
        thresh (float): Threshold for meta-label

    Returns:
        pd.Series: Meta-labels
    """
    signed = primary_preds["signed_dir"]  # {-1, +1}
    mask = y_true.isin([-1, 1])  # exclude TB label=0
    meta = (signed == y_true).astype(int)
    meta[~mask] = np.nan
    meta.name = "meta_label"
    return meta


def fit_meta_model(
    X_val: pd.DataFrame,
    primary_val: pd.DataFrame,
    meta_labels: pd.Series,
) -> xgb.XGBClassifier:
    """
    Train the meta-label classifier.

    Feature set = original features ∪ primary signal features.
    Concatenating the primary signal gives the meta-model visibility into the
    primary model's confidence, allowing it to learn when to trust it.

    Crucially, this is trained entirely on VAL-fold data using primary
    predictions from a model that never saw val — clean OOF setup.

    Args:
        X_val (pd.DataFrame): Validation features
        primary_val (pd.DataFrame): Primary model predictions
        meta_labels (pd.Series): Meta-labels

    Returns:
        xgb.XGBClassifier: Trained meta-model
    """
    # Build meta feature set
    X_meta = pd.concat([X_val, primary_val], axis=1)

    valid = meta_labels.dropna().index
    X_m = X_meta.loc[valid].dropna()
    y_m = meta_labels.loc[X_m.index].astype(int)

    if y_m.nunique() < 2:
        print("[META]  Warning: only one class in meta-labels — skipping fit")
        return None

    # Small internal train/val split within the val fold for early stopping
    n_split = int(len(X_m) * 0.75)
    X_mt, X_mv = X_m.iloc[:n_split], X_m.iloc[n_split:]
    y_mt, y_mv = y_m.iloc[:n_split], y_m.iloc[n_split:]

    meta = xgb.XGBClassifier(**config.META_PARAMS)
    meta.fit(
        X_mt,
        y_mt,
        eval_set=[(X_mv, y_mv)],
        verbose=False,
    )

    meta_prob = meta.predict_proba(X_mv)[:, 1]
    if y_mv.nunique() == 2:
        auc = roc_auc_score(y_mv, meta_prob)
        f1 = f1_score(y_mv, (meta_prob >= config.META_THRESH).astype(int))
        print(f"[META]  Val AUC={auc:.4f}  F1={f1:.4f}")
    return meta


def meta_predict(
    meta_model: xgb.XGBClassifier | None,
    X_test: pd.DataFrame,
    primary_test: pd.DataFrame,
    thresh: float = config.META_THRESH,
) -> pd.DataFrame:
    """
    Generate meta-model predictions for the test fold.
    Returns a DataFrame with meta_prob and the final trade signal.

    Args:
        meta_model (xgb.XGBClassifier | None): The meta-model. Fallback to primary if None.
        X_test (pd.DataFrame): Test features.
        primary_test (pd.DataFrame): Primary model predictions.
        thresh (float, optional): Threshold for meta-label. Defaults to config.META_THRESH.

    Returns:
        pd.DataFrame: DataFrame with meta_prob and trade_signal.
    """
    if meta_model is None:
        # Fallback: use primary signal directly
        result = primary_test.copy()
        result["meta_prob"] = 0.0
        result["trade_signal"] = 0
        return result

    X_meta = pd.concat([X_test, primary_test], axis=1).dropna()
    meta_prob = pd.Series(meta_model.predict_proba(X_meta)[:, 1], index=X_meta.index, name="meta_prob")

    # Final trade signal: only take positions where meta-model approves
    take_trade = meta_prob >= thresh
    trade_signal = (primary_test["signed_dir"] * take_trade).reindex(X_test.index).fillna(0)
    trade_signal.name = "trade_signal"

    result = primary_test.copy()
    result["meta_prob"] = meta_prob
    result["trade_signal"] = trade_signal
    return result
