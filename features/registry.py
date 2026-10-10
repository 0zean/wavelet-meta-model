"""
Feature-group registry (SPEC §3).

A group is a named, causal block of feature columns. Static groups are functions
`(df, cfg, context) -> DataFrame` computed once on the full series; per-fold groups
(e.g. fracdiff, whose d is fit on train) are classes with `fit(train_df, cfg) -> state`
and `transform(df, state, cfg) -> DataFrame` (`transform(df, state, cfg, context)` when the
group declares `needs`). Every column is prefixed `{group}__`. `context` holds the keys the
group declares in `needs` (required) and `optional` (when supplied); "exo" is a
features.exo_align.Exo view restricted to the group's declared `exo` series.

    @feature_group("trend")
    def trend(df, cfg, context): ...

    @feature_group("fracdiff", per_fold=True)
    class Fracdiff:
        @staticmethod
        def fit(train_df, cfg): ...
        @staticmethod
        def transform(df, state, cfg): ...
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_GROUP = "wavelet_core"
STATIONARITY_MAX_CORR = 0.99  # |corr(column, close)| at or above this marks a raw price level


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    fn: Any  # static: callable(df, cfg, context); per_fold: class with fit/transform
    required: bool = False
    intraday_only: bool = False
    per_fold: bool = False
    needs: tuple[str, ...] = ()  # context keys the group reads (e.g. "market"); "exo" = point-in-time series
    level_check: bool = True  # False only for deterministic encodings (calendar) that can trend with a short sample
    optional: tuple[str, ...] = ()  # context keys read when present (e.g. "sector")
    exo: tuple[str, ...] = ()  # "source/name" exo series the group reads (needs "exo"; see features.exo_align)


REGISTRY: dict[str, FeatureGroup] = {}


def feature_group(
    name: str,
    required: bool = False,
    intraday_only: bool = False,
    per_fold: bool = False,
    needs: tuple[str, ...] = (),
    level_check: bool = True,
    optional: tuple[str, ...] = (),
    exo: tuple[str, ...] = (),
) -> Callable:
    """Register a feature group under `name` (see module docstring)."""

    def register(fn):
        if name in REGISTRY:
            raise ValueError(f"feature group {name!r} registered twice")
        if per_fold and not (hasattr(fn, "fit") and hasattr(fn, "transform")):
            raise TypeError(f"per_fold group {name!r} must define fit() and transform()")
        if bool(exo) != ("exo" in needs):
            raise ValueError(f"feature group {name!r}: `exo` series and needs=('exo',) go together")
        REGISTRY[name] = FeatureGroup(
            name, fn, required, intraday_only, per_fold, tuple(needs), level_check, tuple(optional), tuple(exo)
        )
        return fn

    return register


def resolve_groups(groups, timeframe: str, require_core: bool = True) -> list[FeatureGroup]:
    """
    Validate a group list and drop intraday-only groups on daily bars (with a log line).

    Raises:
        ValueError: If `wavelet_core` is missing (unless not `require_core`: the rule pass builds only the groups a
            fixed rule reads), a name is unknown, or a name repeats.
    """
    import features.groups  # noqa: F401  (registers the groups)

    groups = list(groups)
    if require_core and REQUIRED_GROUP not in groups:
        raise ValueError(f"feature set {groups} lacks the required group {REQUIRED_GROUP!r}")
    unknown = [g for g in groups if g not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; registered: {sorted(REGISTRY)}")
    if len(set(groups)) != len(groups):
        raise ValueError(f"duplicate feature groups in {groups}")

    out = []
    for g in groups:
        spec = REGISTRY[g]
        if spec.intraday_only and timeframe == "1Day":
            print(f"[FEAT]  Skipping intraday-only group {g!r} on 1Day bars")
            continue
        out.append(spec)
    return out


def context_needs(groups, timeframe: str) -> tuple[set[str], set[str]]:
    """(context keys, exo series) the resolved groups read: needs ∪ optional, and the union of their `exo`."""
    keys, exo = set(), set()
    for spec in resolve_groups(groups, timeframe):
        keys |= set(spec.needs) | set(spec.optional)
        exo |= set(spec.exo)
    return keys, exo


def check_group_output(name: str, feats: pd.DataFrame, df: pd.DataFrame, level_check: bool = True) -> None:
    """
    Enforce the group contract: aligned index, `{name}__` prefix, float columns, no ±inf,
    and no raw price level (|corr with close| ≥ STATIONARITY_MAX_CORR).

    Raises:
        ValueError: On any violation.
    """
    if not feats.index.equals(df.index):
        raise ValueError(f"group {name!r} returned an index not aligned to the bars")
    bad = [c for c in feats.columns if not c.startswith(f"{name}__")]
    if bad:
        raise ValueError(f"group {name!r} returned columns without the '{name}__' prefix: {bad}")
    vals = feats.to_numpy(dtype=float)
    if np.isinf(vals).any():
        cols = feats.columns[np.isinf(vals).any(axis=0)].tolist()
        raise ValueError(f"group {name!r} produced ±inf in {cols}")
    if not level_check:
        return
    close = df["close"].to_numpy(dtype=float)
    for j, col in enumerate(feats.columns):
        x = vals[:, j]
        ok = ~np.isnan(x)
        if ok.sum() < 100 or np.std(x[ok]) == 0 or np.std(close[ok]) == 0:
            continue
        rho = np.corrcoef(x[ok], close[ok])[0, 1]
        if abs(rho) >= STATIONARITY_MAX_CORR:
            raise ValueError(
                f"{col}: |corr with close| = {abs(rho):.4f} ≥ {STATIONARITY_MAX_CORR} — looks like a raw price "
                "level; only returns, ratios, z-scores or fracdiff are allowed"
            )
