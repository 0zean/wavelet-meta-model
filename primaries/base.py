"""
Primary-signal protocol and registry (SPEC §4).

A primary gives every event a side {-1, +1}; the meta-model then decides whether to take it. A primary whose class
sets ALLOW_FLAT (the mechanism primaries, SPEC §16) may also give 0 = flat: no position for that holding period (a
long-only rule's short signal, a gate, a warm-up); flat events are never traded nor fitted on. A primary whose
`long_only` attribute is true must give no short side.
`signal` returns the full primary frame (PRIMARY_COLUMNS), which is also the primary's
input to the meta-model; `side` is its `signed_dir` column.

    @primary("sma_cross")
    class SmaCross(RulePrimary): ...

    make_primary(cfg)  # a fresh, unfitted instance of cfg.PRIMARY with cfg.PRIMARY_PARAMS
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from utils.config import RunConfig

# The primary frame. For the ML primary (utils.primary_signal): clf_prob = P(long), direction = 1 if long,
# signed_dir = side, magnitude = predicted |move|, signal = side · magnitude, confidence = clf_prob · (1 + magnitude).
# Rule primaries have no probability (clf_prob NaN) and use their rule strength as magnitude and confidence.
PRIMARY_COLUMNS = ("clf_prob", "direction", "signed_dir", "magnitude", "signal", "confidence")


class Primary(Protocol):
    name: str

    def fit(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        labels: pd.DataFrame,
        weights: pd.Series,
        cfg: RunConfig,
        *,
        val: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> "Primary":
        """Fit on purged train events (X, labels, weights). `df` ends at the train split's end;
        `val` = (X_val, labels_val) is for logging only. A no-op for fixed rules."""
        ...

    def signal(self, df: pd.DataFrame, X: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
        """Primary frame for the events X.index; the row for event t uses bars <= t only."""
        ...


REGISTRY: dict[str, Callable[..., Primary]] = {}


def primary(name: str) -> Callable:
    """Register a primary class under `name`; the class is constructed with PRIMARY_PARAMS as kwargs."""

    def register(cls):
        if name in REGISTRY:
            raise ValueError(f"primary {name!r} registered twice")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return register


def make_primary(cfg: RunConfig) -> Primary:
    """A fresh, unfitted instance of cfg.PRIMARY built with cfg.PRIMARY_PARAMS."""
    try:
        cls = REGISTRY[cfg.PRIMARY]
    except KeyError:
        raise ValueError(f"unknown primary {cfg.PRIMARY!r}; expected one of {sorted(REGISTRY)}") from None
    return cls(**cfg.PRIMARY_PARAMS)


def side(p: Primary, df: pd.DataFrame, X: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """The primary's side {-1, +1} per event (SPEC §4)."""
    return p.signal(df, X, cfg)["signed_dir"]


def check_signal(frame: pd.DataFrame, X: pd.DataFrame, primary: "str | Primary") -> pd.DataFrame:
    """Raise unless `frame` is a complete primary frame for X's events with sides in {-1, +1} ({-1, 0, +1} for an
    ALLOW_FLAT primary, {0, +1} for a long-only one). `primary` = the primary instance (or its name: no flat sides)."""
    name = primary if isinstance(primary, str) else primary.name
    allowed = (-1, 0, 1) if getattr(primary, "ALLOW_FLAT", False) else (-1, 1)
    if getattr(primary, "long_only", False):
        allowed = tuple(v for v in allowed if v >= 0)
    if tuple(frame.columns) != PRIMARY_COLUMNS:
        raise ValueError(f"primary {name!r} returned columns {list(frame.columns)}, expected {list(PRIMARY_COLUMNS)}")
    if not frame.index.equals(X.index):
        raise ValueError(f"primary {name!r} returned a frame not indexed like its events")
    sides = frame["signed_dir"].to_numpy()
    if not np.isin(sides, allowed).all():
        bad = frame.index[~np.isin(sides, allowed)]
        raise ValueError(f"primary {name!r} gave {len(bad)} events a side outside {set(allowed)} (first {bad[0]})")
    return frame


def rule_frame(score: pd.Series, index: pd.Index, name: str) -> pd.DataFrame:
    """
    Primary frame of a rule from its signed score (sign = side, |score| = strength).

    A score of exactly 0 goes long (a tie has no direction). A NaN score at an event raises:
    rule windows are shorter than the feature warm-up, so it means missing bars, not warm-up.
    """
    s = score.reindex(index)
    if s.isna().any():
        raise ValueError(
            f"primary {name!r} is undefined at {int(s.isna().sum())} events (first {s.index[s.isna()][0]})"
        )
    sd = np.where(s.to_numpy() >= 0, 1, -1)
    strength = np.abs(s.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "clf_prob": np.nan,
            "direction": (sd > 0).astype(int),
            "signed_dir": sd,
            "magnitude": strength,
            "signal": sd * strength,
            "confidence": strength,
        },
        index=index,
    )
