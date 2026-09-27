"""
Clustered MDA feature selection (López de Prado 2020, ch.6; SPEC §3).

Run on ONE fold's purged training events only: features are clustered on 1 − |ρ|
(hierarchical, average linkage, k chosen by silhouette), then each cluster's columns are
permuted jointly inside a purged k-fold and the drop in weighted neg-log-loss is its
importance. Clusters whose mean importance exceeds 0 by one standard error are kept;
'wavelet_core' columns are always kept.
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, silhouette_score

from features.registry import REQUIRED_GROUP
from utils.config import RunConfig
from validation.purged_cv import PurgedKFold

MAX_CLUSTERS = 10


class Selection(NamedTuple):
    kept: list[str]  # selected columns, in X's column order
    clusters: dict[int, list[str]]
    importance: pd.DataFrame  # per cluster: mean, se, n_features, kept


def cluster_features(X: pd.DataFrame, max_clusters: int = MAX_CLUSTERS) -> dict[int, list[str]]:
    """Hierarchical clustering of the columns on 1 − |corr|; k in [2, max] with the best silhouette."""
    cols = list(X.columns)
    if len(cols) < 3:
        return {i: [c] for i, c in enumerate(cols)}
    corr = X.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - np.abs(corr), 0.0, None)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    best, best_score = None, -np.inf
    for k in range(2, min(max_clusters, len(cols) - 1) + 1):
        lab = fcluster(link, k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        score = silhouette_score(dist, lab, metric="precomputed")
        if score > best_score:
            best, best_score = lab, score
    if best is None:
        best = np.ones(len(cols), dtype=int)
    return {int(k): [c for c, lab in zip(cols, best) if lab == k] for k in np.unique(best)}


def clustered_mda(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    weights: pd.Series,
    cfg: RunConfig,
    n_bars: int,
) -> Selection:
    """
    Clustered MDA on training events.

    Args:
        X (pd.DataFrame): Train features (event rows, no NaN).
        labels (pd.DataFrame): Train label rows aligned to X (`label` ∈ {-1,0,1}, `entry_pos`, `exit_pos`).
        weights (pd.Series): Sample weights aligned to X.
        cfg (RunConfig): CMDA_TREES, CMDA_SPLITS, CV_EMBARGO_PCT, SEED.
        n_bars (int): Bars in the training window (embargo = ceil(CV_EMBARGO_PCT · n_bars)).

    Returns:
        Selection: kept columns, clusters and per-cluster importance.
    """
    keep = labels["label"].isin([-1, 1])
    X, lab, w = X.loc[keep], labels.loc[keep], weights.loc[keep]
    order = np.argsort((lab["entry_pos"] - 1).to_numpy(), kind="stable")
    X, lab, w = X.iloc[order], lab.iloc[order], w.iloc[order]
    y = (lab["label"] == 1).astype(int).to_numpy()
    t0, t1 = (lab["entry_pos"] - 1).to_numpy(), lab["exit_pos"].to_numpy()

    core = [c for c in X.columns if c.startswith(f"{REQUIRED_GROUP}__")]
    if not core:
        raise ValueError(f"clustered MDA needs '{REQUIRED_GROUP}' columns to keep; got {list(X.columns)[:5]}…")
    varying = [c for c in X.columns if X[c].std() > 0]
    clusters = cluster_features(X[varying])
    rng = np.random.default_rng(cfg.SEED)
    cv = PurgedKFold(cfg.CMDA_SPLITS, embargo=math.ceil(cfg.CV_EMBARGO_PCT * n_bars))

    scores = {k: [] for k in clusters}
    Xv, wv = X.to_numpy(), w.to_numpy()
    col_pos = {c: i for i, c in enumerate(X.columns)}
    for train, test in cv.split(t0, t1):
        if len(np.unique(y[train])) < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=cfg.CMDA_TREES,
            max_features=1,
            class_weight="balanced_subsample",
            min_weight_fraction_leaf=0.05,
            random_state=cfg.SEED,
            n_jobs=1,  # multi-threaded fits differ in the last bits, which can flip a knife-edge keep
        )
        rf.fit(Xv[train], y[train], sample_weight=wv[train])
        base = -log_loss(y[test], rf.predict_proba(Xv[test]), sample_weight=wv[test], labels=[0, 1])
        for k, cols in clusters.items():
            Xp = Xv[test].copy()
            perm = rng.permutation(len(test))
            idx = [col_pos[c] for c in cols]
            Xp[:, idx] = Xp[perm][:, idx]
            scores[k].append(base + log_loss(y[test], rf.predict_proba(Xp), sample_weight=wv[test], labels=[0, 1]))

    rows = []
    for k, cols in clusters.items():
        s = np.asarray(scores[k])
        mean = s.mean() if s.size else np.nan
        se = s.std(ddof=1) / np.sqrt(s.size) if s.size > 1 else np.nan
        rows.append({"cluster": k, "mean": mean, "se": se, "n_features": len(cols), "kept": bool(mean - se > 0)})
    imp = pd.DataFrame(rows).set_index("cluster")
    chosen = set(core).union(*(clusters[k] for k in imp.index[imp["kept"]]))
    kept = [c for c in X.columns if c in chosen]
    print(
        f"[CMDA]  {len(clusters)} clusters, {int(imp['kept'].sum())} kept → {len(kept)}/{X.shape[1]} features "
        f"({len(core)} wavelet_core always kept)"
    )
    return Selection(kept, clusters, imp)
