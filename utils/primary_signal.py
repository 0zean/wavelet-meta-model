import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import config


def primary_signal(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    X: pd.DataFrame,
    thresh: float = config.CLF_THRESH,
) -> pd.DataFrame:
    """
    Produce the combined primary signal from classifier + regressor.

    direction  = clf.predict_proba > thresh → {0, 1}
    signed_dir = direction x 2 - 1          → {-1, +1}
    magnitude  = |reg.predict|              → ≥ 0

    signal     = signed_dir x magnitude     (signed expected move)
    confidence = clf_prob x (1 + magnitude) (magnitude-weighted probability)

    The confidence score is a single monotonic ranking passed to the meta-model
    as a feature, allowing it to learn when the combined signal is reliable.

    Args:
        clf (xgb.XGBClassifier): The primary direction classifier.
        reg (xgb.XGBRegressor): The primary magnitude regressor.
        X (pd.DataFrame): The feature matrix.
        thresh (float, optional): Probability threshold for a long side. Defaults to config.CLF_THRESH.

    Returns:
        pd.DataFrame: Dataframe containing model outputs and combined signal + confidence level.
    """
    clf_prob = clf.predict_proba(X)[:, 1]
    direction = (clf_prob >= thresh).astype(int)
    signed_dir = direction * 2 - 1
    magnitude = np.abs(reg.predict(X))

    signal = signed_dir * magnitude
    confidence = clf_prob * (1.0 + magnitude)

    return pd.DataFrame(
        {
            "clf_prob": clf_prob,
            "direction": direction,
            "signed_dir": signed_dir,
            "magnitude": magnitude,
            "signal": signal,
            "confidence": confidence,
        },
        index=X.index,
    )
