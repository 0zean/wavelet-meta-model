import pandas as pd
import xgboost as xgb

from features.triple_barrier_labels import barrier_exits
from utils.config import RunConfig


def make_meta_labels(df: pd.DataFrame, events: pd.DataFrame, primary_preds: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """
    Meta-label = 1 if taking the primary model's side on the event would have
    earned more than META_MIN_RET (round-trip slippage), else 0.

    The barriers are re-evaluated with the side known (López de Prado 2018, §3.6),
    so same-bar barrier ties resolve against the position, exactly as in the backtest.

    NOTE: Meta-labels are computed on the VALIDATION split using primary model
    predictions from a model trained on TRAIN — no leakage. The caller must pass
    only events whose exit falls inside the validation split (purged).

    Args:
        df (pd.DataFrame): OHLC data.
        events (pd.DataFrame): Validation events with a `width` column.
        primary_preds (pd.DataFrame): Primary model predictions for those events.
        cfg (RunConfig): Run configuration.

    Returns:
        pd.Series: Meta-labels {0, 1} indexed by event time.
    """
    side = primary_preds["signed_dir"]
    out = barrier_exits(
        df,
        events.index,
        events["width"],
        side=side,
        vertical_bars=cfg.VERTICAL_BARS,
        hold_overnight=cfg.HOLD_OVERNIGHT,
    )
    return (side.loc[out.index] * out["ret"] > cfg.META_MIN_RET).astype(int).rename("meta_label")


def fit_meta_model(
    X_val: pd.DataFrame,
    primary_val: pd.DataFrame,
    meta_labels: pd.Series,
    weights: pd.Series,
    cfg: RunConfig,
) -> xgb.XGBClassifier | None:
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
        weights (pd.Series): Sample weights (average uniqueness)
        cfg (RunConfig): Run configuration.

    Returns:
        xgb.XGBClassifier | None: Trained meta-model, or None if the labels have a single class.
    """
    X_m = pd.concat([X_val, primary_val], axis=1).loc[meta_labels.index]

    if meta_labels.nunique() < 2:
        print("[META]  Warning: only one class in meta-labels — skipping fit")
        return None

    meta = xgb.XGBClassifier(**cfg.META_PARAMS)
    meta.fit(X_m, meta_labels, sample_weight=weights.loc[meta_labels.index], verbose=False)
    print(f"[META]  Trained on {len(meta_labels)} val events  (success rate={meta_labels.mean():.3f})")
    return meta


def meta_predict(
    meta_model: xgb.XGBClassifier | None,
    X_test: pd.DataFrame,
    primary_test: pd.DataFrame,
    thresh: float,
) -> pd.DataFrame:
    """
    Generate meta-model predictions for the test fold.
    Returns a DataFrame with meta_prob and the final trade signal.

    Args:
        meta_model (xgb.XGBClassifier | None): The meta-model. If None, no trades are taken.
        X_test (pd.DataFrame): Test features.
        primary_test (pd.DataFrame): Primary model predictions.
        thresh (float): Probability cutoff to trade (cfg.META_THRESH).

    Returns:
        pd.DataFrame: DataFrame with meta_prob and trade_signal.
    """
    result = primary_test.copy()
    if meta_model is None:
        # No meta-model (single-class val fold): take no trades this fold
        result["meta_prob"] = 0.0
        result["trade_signal"] = 0
        return result

    X_meta = pd.concat([X_test, primary_test], axis=1)
    result["meta_prob"] = meta_model.predict_proba(X_meta)[:, 1]

    # Final trade signal: only take positions where meta-model approves
    result["trade_signal"] = result["signed_dir"].where(result["meta_prob"] >= thresh, 0)
    return result
