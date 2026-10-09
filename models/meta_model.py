import numpy as np
import pandas as pd

from features.triple_barrier_labels import barrier_exits
from models.zoo import ZooFitError, ZooModel, _clip, _Sigmoid, _Tuned, describe, inner_cv, make_model
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
    return (side_returns(df, events, primary_preds, cfg) > cfg.META_MIN_RET).astype(int).rename("meta_label")


def side_returns(df: pd.DataFrame, events: pd.DataFrame, primary_preds: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """Gross barrier-exit return of taking the primary's side on each event (side-aware ties, as in the backtest)."""
    side = primary_preds["signed_dir"]
    out = barrier_exits(
        df,
        events.index,
        events["width"],
        side=side,
        vertical_bars=cfg.VERTICAL_BARS,
        hold_overnight=cfg.HOLD_OVERNIGHT,
    )
    return (side.loc[out.index] * out["ret"]).rename("side_ret")


def fit_meta_model(
    X_fit: pd.DataFrame,
    primary_fit: pd.DataFrame,
    meta_labels: pd.Series,
    weights: pd.Series,
    cfg: RunConfig,
    spans: pd.DataFrame | None = None,
    cal_pairs: pd.DataFrame | None = None,
) -> ZooModel | None:
    """
    Train the meta-label classifier (cfg.META_MODEL, models/zoo.py).

    Feature set = original features ∪ primary signal features.
    Concatenating the primary signal gives the meta-model visibility into the
    primary model's confidence, allowing it to learn when to trust it.

    The primary frame must come from primary models that never saw these events' labels:
    the train-fit primary on VAL events (META_TRAIN="val"), or purged out-of-fold primaries on
    train+val events (META_TRAIN="oof"). A zoo model's HP search and calibration use only these rows.

    Args:
        X_fit (pd.DataFrame): Meta-training features
        primary_fit (pd.DataFrame): Primary predictions for those events
        meta_labels (pd.Series): Meta-labels
        weights (pd.Series): Sample weights (average uniqueness)
        cfg (RunConfig): Run configuration.
        spans (pd.DataFrame | None): Label rows (`entry_pos`, `exit_pos`) covering meta_labels' events,
            for the inner purged CV; required unless META_MODEL is "legacy".
        cal_pairs (pd.DataFrame | None): CALIBRATION="rolling" only: earlier windows' OOS events resolved before
            this fit (columns `raw` = the predicting window's uncalibrated meta P(y=1), `y` = meta-label, `w` =
            average uniqueness). With >= MIN_VAL_EVENTS rows of both classes the model is fit uncalibrated and
            a weighted Platt map on these pairs becomes its calibrator; otherwise (the first window, or a caller
            without a walk-forward) the model cross-fits its own calibration.

    Returns:
        ZooModel | None: Trained meta-model, or None if the labels (or an inner purged train split) have a
            single class — the fold then takes no trades.
    """
    X_m = pd.concat([X_fit, primary_fit], axis=1).loc[meta_labels.index]

    if meta_labels.nunique() < 2:
        print("[META]  Warning: only one class in meta-labels — skipping fit")
        return None

    rolling = (
        cfg.CALIBRATION == "rolling"
        and cal_pairs is not None
        and len(cal_pairs) >= cfg.MIN_VAL_EVENTS
        and cal_pairs["y"].nunique() == 2
    )
    calibration = "none" if rolling or cfg.CALIBRATION == "none" else "crossfit"
    meta = make_model(cfg.META_MODEL, cfg, role="meta", calibration=calibration)
    if cfg.META_MODEL == "legacy":
        meta.fit(X_m, meta_labels, weights.loc[meta_labels.index])
    else:
        if spans is None:
            raise ValueError(f"META_MODEL={cfg.META_MODEL!r} needs the label spans for its inner purged CV")
        try:
            meta.fit(X_m, meta_labels, weights.loc[meta_labels.index], inner_cv(spans.loc[meta_labels.index], cfg))
        except ZooFitError as e:
            print(f"[META]  Warning: {e} — skipping fit")
            return None
        if rolling:
            meta.cal = _Sigmoid().fit(cal_pairs["raw"].to_numpy(), cal_pairs["y"].to_numpy(), cal_pairs["w"].to_numpy())
            meta.calibration_ = f"rolling ({len(cal_pairs)} pairs)"
        elif cfg.CALIBRATION == "rolling":
            n = 0 if cal_pairs is None else len(cal_pairs)
            print(f"[META]  rolling calibration: {n} resolved pair(s), need {cfg.MIN_VAL_EVENTS} of both classes")
        print(f"[META]  {describe(meta)}")
    print(f"[META]  Trained on {len(meta_labels)} events  (success rate={meta_labels.mean():.3f})")
    return meta


def oof_meta_prob(
    X_fit: pd.DataFrame,
    primary_fit: pd.DataFrame,
    meta_labels: pd.Series,
    weights: pd.Series,
    cfg: RunConfig,
    spans: pd.DataFrame,
    meta: ZooModel | None = None,
) -> pd.Series:
    """
    Out-of-fold meta-probabilities of the meta-model's own fitting events (the train-window inputs of the `ecdf`
    and `kelly_capped` sizers, SPEC §7).

    OOF_META="reuse" with a zoo `meta` (the fitted fit_meta_model result): its purged-CV OOF raw predictions
    (`oof_raw_`, same rows) through its calibrator — no fit. A meta-model without OOF predictions raises.
    OOF_META="refit" (or a legacy meta-model): over a purged k-fold of their spans (ZOO_CV_SPLITS, CV_EMBARGO_PCT) a
    fresh cfg.META_MODEL (with its own inner HP search, and its own calibration under CALIBRATION="crossfit") is fit
    on each purged train split and predicts its test split; under "rolling" the raw predictions go through `meta`'s
    calibrator, under "none" they stay raw. Same rows and features as fit_meta_model. A single-class train split
    raises ZooFitError.
    """
    if cfg.OOF_META == "reuse" and isinstance(meta, _Tuned):
        if meta.oof_raw_ is None or len(meta.oof_raw_) != len(meta_labels):
            raise RuntimeError(
                f"OOF_META='reuse': the {meta.name} meta-model holds no OOF predictions for these "
                f"{len(meta_labels)} events (one grid point fit without cross-fitting); use OOF_META='refit'"
            )
        return pd.Series(_clip(meta.cal.predict(meta.oof_raw_)), index=meta_labels.index, name="oof_meta_prob")
    rolling = cfg.CALIBRATION == "rolling" and cfg.META_MODEL != "legacy"
    if rolling and meta is None:
        raise ValueError("CALIBRATION='rolling' maps the OOF predictions through the fitted meta-model's calibrator")
    X_m = pd.concat([X_fit, primary_fit], axis=1).loc[meta_labels.index]
    spans = spans.loc[meta_labels.index]
    y, w = meta_labels.to_numpy(), weights.loc[meta_labels.index].to_numpy()
    out = pd.Series(np.nan, index=meta_labels.index, name="oof_meta_prob")
    calibration = "crossfit" if cfg.CALIBRATION == "crossfit" else "none"
    for i, (train, test) in enumerate(inner_cv(spans, cfg).split(X_m)):
        if np.unique(y[train]).size < 2:
            raise ZooFitError(f"OOF meta split {i}: single-class purged train set ({train.size} rows)")
        m = make_model(cfg.META_MODEL, cfg, role="meta", calibration=calibration)
        if cfg.META_MODEL == "legacy":
            m.fit(X_m.iloc[train], y[train], w[train])
        else:
            m.fit(X_m.iloc[train], y[train], w[train], inner_cv(spans.iloc[train], cfg))
        if rolling:
            out.iloc[test] = _clip(meta.cal.predict(m.predict_raw(X_m.iloc[test])))
        else:
            out.iloc[test] = m.predict_proba(X_m.iloc[test])
    if out.isna().any():
        raise RuntimeError(f"OOF meta: {int(out.isna().sum())} events without a prediction")
    return out


def meta_predict(
    meta_model: ZooModel | None,
    X_test: pd.DataFrame,
    primary_test: pd.DataFrame,
    thresh: float,
) -> pd.DataFrame:
    """
    Generate meta-model predictions for the test fold.
    Returns a DataFrame with meta_prob and the final trade signal.

    Args:
        meta_model (ZooModel | None): The meta-model. If None, no trades are taken.
        X_test (pd.DataFrame): Test features.
        primary_test (pd.DataFrame): Primary model predictions.
        thresh (float): Probability cutoff to trade (cfg.META_THRESH).

    Returns:
        pd.DataFrame: DataFrame with meta_prob and trade_signal.
    """
    result = primary_test.copy()
    if meta_model is None:
        # No meta-model (single-class meta-labels): take no trades this fold
        result["meta_prob"] = 0.0
        result["trade_signal"] = 0
        return result

    X_meta = pd.concat([X_test, primary_test], axis=1)
    result["meta_prob"] = meta_model.predict_proba(X_meta)

    # Final trade signal: only take positions where meta-model approves
    result["trade_signal"] = result["signed_dir"].where(result["meta_prob"] >= thresh, 0)
    return result
