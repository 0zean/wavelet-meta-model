"""The ML primary: XGBoost direction classifier + magnitude regressor (the pre-U4 pipeline)."""

import pandas as pd

from models.primary_classifier import fit_primary_classifier
from models.primary_regressor import fit_primary_regressor
from primaries.base import primary
from utils.config import RunConfig
from utils.movement_filter import make_active_day_mask
from utils.primary_signal import primary_signal


@primary("ml_xgb")
class MlXgb:
    """
    Classifier on active-day train events (side), regressor on all train events (|move|).
    The classifier is cfg.PRIMARY_MODEL (models/zoo.py; "legacy" = XGBoost CLF_PARAMS); the regressor uses
    REG_PARAMS and the side threshold CLF_THRESH, so it takes no PRIMARY_PARAMS.
    """

    def __init__(self, **params):
        if params:
            raise ValueError(f"ml_xgb takes no PRIMARY_PARAMS (use CLF_PARAMS / REG_PARAMS), got {sorted(params)}")
        self.clf = self.reg = None

    def fit(self, df, X, labels, weights, cfg: RunConfig, *, val=None) -> "MlXgb":
        X_vl, lab_vl = val if val is not None else (None, None)  # val is for logging only
        y_vl, r_vl = (None, None) if lab_vl is None else (lab_vl["label"], lab_vl["ret"])
        # Active-day mask from the fitting bars only (df ends at the fitting split's end)
        active = make_active_day_mask(df, cfg.LOW_MOVE_PCTILE).loc[X.index]
        self.clf = fit_primary_classifier(X, labels["label"], weights, X_vl, y_vl, active, cfg, labels)
        self.reg = fit_primary_regressor(X, labels["ret"], weights, X_vl, r_vl, cfg)
        return self

    def signal(self, df, X: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
        if self.clf is None:
            raise RuntimeError("ml_xgb.signal called before fit")
        return primary_signal(self.clf, self.reg, X, cfg.CLF_THRESH)
