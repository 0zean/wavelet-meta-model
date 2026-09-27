import time
from typing import NamedTuple

import numpy as np
import pandas as pd

from features.feature_builder import FeatureSet
from features.selection import clustered_mda
from features.triple_barrier_labels import average_uniqueness, sample_events, triple_barrier_labels
from models.meta_model import fit_meta_model, make_meta_labels, meta_predict
from models.primary_classifier import fit_primary_classifier
from models.primary_regressor import fit_primary_regressor
from utils.config import RunConfig
from utils.movement_filter import make_active_day_mask
from utils.primary_signal import primary_signal


def purged(labels: pd.DataFrame, start: int, end: int, embargo: int = 0) -> pd.DataFrame:
    """
    Label rows whose event bar t lies in [start, end) AND whose exit is observed
    before `end - embargo` — a label that is resolved by prices in the next split
    is purged, so no fitting sample peeks into a later split (López de Prado 2018, §7.4).

    Embargo (SPEC §5): the splits are walked forward, so the only fitting samples
    adjacent to a later split are those at the END of this one. Their outcomes
    are serially correlated with the next split's first outcomes, so samples
    whose exit falls in the last `embargo` bars before `end` are dropped too.
    (Train samples never follow a test split here, so no post-test embargo is needed.)

    Args:
        labels (pd.DataFrame): Label rows with `entry_pos` and `exit_pos`.
        start (int): First bar of the split.
        end (int): First bar of the next split.
        embargo (int, optional): Bars before `end` in which no exit may fall. Defaults to 0.

    Returns:
        pd.DataFrame: The usable label rows.
    """
    t = labels["entry_pos"] - 1  # an event at bar t enters at open[t+1]
    return labels[(t >= start) & (t < end) & (labels["exit_pos"] < end - embargo)]


def build_features(df: pd.DataFrame, cfg: RunConfig, fset: FeatureSet) -> pd.DataFrame:
    """Static features for the whole series (module-level so tests can inject a leak)."""
    return fset.build(df)


def select_features(X_tr, lab_tr, w_tr, cfg: RunConfig, n_bars: int):
    """Clustered MDA on the fold's purged train events (module-level so tests can spy on its inputs)."""
    return clustered_mda(X_tr, lab_tr, w_tr, cfg, n_bars)


class Fold(NamedTuple):
    train_end: int  # train = [0, train_end)
    val_end: int  # val = [train_end, val_end)
    test_end: int  # test = [val_end, test_end)
    train_embargo: int  # bars before train_end excluded from train exits
    val_embargo: int  # bars before val_end excluded from val exits


def wfo_folds(index: pd.DatetimeIndex, cfg: RunConfig) -> list[Fold]:
    """
    Expanding-window fold boundaries as bar positions.

    WINDOW_UNIT="bars": INITIAL_TRAIN/VAL/TEST/EMBARGO are bar counts (legacy).
    WINDOW_UNIT="days": they count trading sessions present in the data, and every
    boundary falls on a session's first bar, so a split holds whole sessions
    (half-days contribute their shorter session). Sessions the data layer dropped
    (Alpaca holes) do not count as days.

    Args:
        index (pd.DatetimeIndex): Bar timestamps.
        cfg (RunConfig): Run configuration.

    Returns:
        list[Fold]: One entry per fold, in order.
    """
    T0, V, TS, E = cfg.INITIAL_TRAIN, cfg.VAL, cfg.TEST, cfg.EMBARGO
    if cfg.WINDOW_UNIT == "bars":
        n = len(index)
        return [Fold(tr, tr + V, tr + V + TS, E, E) for tr in range(T0, n - V - TS + 1, TS)]

    day = index.normalize()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    bounds = np.r_[starts, len(index)]  # bounds[k] = first bar of session k; bounds[-1] = end of data
    n_sess = len(starts)
    folds = []
    for tr in range(T0, n_sess - V - TS + 1, TS):
        vl, te = tr + V, tr + V + TS
        folds.append(
            Fold(
                int(bounds[tr]),
                int(bounds[vl]),
                int(bounds[te]),
                int(bounds[tr] - bounds[tr - E]),
                int(bounds[vl] - bounds[vl - E]),
            )
        )
    return folds


def run_wfo(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    context: dict[str, pd.DataFrame] | None = None,
    symbol: str | None = None,
    feature_cache_dir=None,
) -> pd.DataFrame:
    """
    Expanding-window Walk-Forward Optimisation on CUSUM events.

    Fold structure (see wfo_folds; windows in trading sessions or bars):
    ┌────────────────────────────────────────────────────────┐
    │ TRAIN (expanding) │  VAL (fixed)  │  TEST (OOS/fixed)  │
    └────────────────────────────────────────────────────────┘
    ↑                   ↑               ↑                    ↑
    0             train_end         val_end              test_end

    Once, on the full series (all causal — bar t uses bars <= t):
      • CUSUM events + barrier widths, and the static feature groups (cfg.FEATURE_GROUPS)
      • Triple-barrier outcomes (TARGETS, read only through `purged`)

    At each fold:
      1. Fit the per-fold feature groups (fracdiff d) on train bars before its embargo → transform the series
      2. Purge train/val events whose barrier exit falls in the next split
         or in the embargo before it
      3. Optional clustered-MDA feature selection on the purged train events only
         (cfg.FEATURE_SELECTION = "cmda"); fit primary classifier (active days) and regressor on train events
      4. Primary signal on val → side-aware meta-labels → fit meta-model on val
      5. Final trade signal on test (OOS) events → store

    Args:
        df (pd.DataFrame): Full dataset to perform WFO.
        cfg (RunConfig): Run configuration.
        context (dict | None): Extra aligned inputs for feature groups, e.g. {"market": SPY bars}.
        symbol (str | None): Symbol name; with feature_cache_dir enables the feature cache.
        feature_cache_dir (path | None): Feature cache root (features.cache).

    Raises:
        HoldoutError: If df reaches HOLDOUT_START and cfg.ALLOW_HOLDOUT is off.
        RuntimeError: If no WFO folds are completed.

    Returns:
        pd.DataFrame: OOS predictions for every test event, concatenated across folds.
    """
    cfg.holdout_guard(df.index)
    N = len(df)

    print("\n" + "═" * 60)
    print("  Sampling events, labels and causal features on the full dataset")
    print("═" * 60)
    events = sample_events(df, cfg)
    labels = triple_barrier_labels(df, events, cfg)
    fset = FeatureSet(cfg, context=context, symbol=symbol, cache_dir=feature_cache_dir)
    base_feats = build_features(df, cfg, fset)
    event_pos = pd.Series(df.index.get_indexer(events.index), index=events.index)

    all_results = []
    for fold, (train_end, val_end, test_end, emb_tr, emb_vl) in enumerate(wfo_folds(df.index, cfg), start=1):
        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold}: train[0:{train_end}]  val[{train_end}:{val_end}]  test[{val_end}:{test_end}]")
        print(f"{'─' * 60}")

        t0 = time.time()

        # ── Features: per-fold groups (fracdiff d) fit on train bars before the embargo ──
        # (fitting on the embargo bars would make every train feature depend on them)
        try:
            states = fset.fit(df.iloc[: train_end - emb_tr])
        except RuntimeWarning as e:  # FracdiffStat raises this when no d <= 1 is stationary
            print(f"[WFO]  Fold {fold}: fracdiff failed ({e}) — skipping")
            continue
        X_all = base_feats.iloc[:test_end].join(fset.transform(df.iloc[:test_end], states))

        # ── Samples: events with complete features; fitting splits purged ─────
        lab_tr = purged(labels, 0, train_end, emb_tr)
        lab_vl = purged(labels, train_end, val_end, emb_vl)
        ev_ts = events[(event_pos >= val_end) & (event_pos < test_end)]

        X_tr = X_all.reindex(lab_tr.index).dropna()
        X_vl = X_all.reindex(lab_vl.index).dropna()
        X_ts = X_all.reindex(ev_ts.index).dropna()
        n_nan = len(lab_tr) - len(X_tr) + len(lab_vl) - len(X_vl) + len(ev_ts) - len(X_ts)
        if n_nan:
            print(f"[WFO]  Fold {fold}: {n_nan} events dropped for NaN features (warm-up or gaps)")
        lab_tr, lab_vl = lab_tr.loc[X_tr.index], lab_vl.loc[X_vl.index]

        if len(X_tr) < cfg.MIN_TRAIN_EVENTS or len(X_vl) < cfg.MIN_VAL_EVENTS or X_ts.empty:
            print(f"[WFO]  Fold {fold}: insufficient events (train={len(X_tr)}, val={len(X_vl)}) — skipping")
            continue

        w_tr = average_uniqueness(lab_tr, N)
        w_vl = average_uniqueness(lab_vl, N)

        # ── Feature selection on purged train events only ─────────────────────
        if cfg.FEATURE_SELECTION == "cmda":
            kept = select_features(X_tr, lab_tr, w_tr, cfg, train_end).kept
            X_tr, X_vl, X_ts = X_tr[kept], X_vl[kept], X_ts[kept]

        # ── Active-day mask (train only) ──────────────────────────────────────
        active_mask = make_active_day_mask(df.iloc[:train_end], cfg.LOW_MOVE_PCTILE).loc[X_tr.index]

        # ── Fit primary models ────────────────────────────────────────────────
        clf = fit_primary_classifier(X_tr, lab_tr["label"], w_tr, X_vl, lab_vl["label"], active_mask, cfg)
        reg = fit_primary_regressor(X_tr, lab_tr["ret"], w_tr, X_vl, lab_vl["ret"], cfg)

        # ── Primary signal on val → meta-labels ──────────────────────────────
        prim_val = primary_signal(clf, reg, X_vl, cfg.CLF_THRESH)
        meta_lbl = make_meta_labels(df, events.loc[X_vl.index], prim_val, cfg)
        meta_mdl = fit_meta_model(X_vl, prim_val, meta_lbl, w_vl, cfg)

        # ── Final prediction on OOS test fold ─────────────────────────────────
        prim_ts = primary_signal(clf, reg, X_ts, cfg.CLF_THRESH)
        result_ts = meta_predict(meta_mdl, X_ts, prim_ts, cfg.META_THRESH)
        result_ts["width"] = ev_ts.loc[X_ts.index, "width"]
        result_ts["fold"] = fold
        all_results.append(result_ts)

        elapsed = time.time() - t0
        trade_ct = (result_ts["trade_signal"] != 0).sum()
        print(f"[WFO]  Fold {fold} done in {elapsed:.1f}s  |  OOS events={len(result_ts)}  approved={trade_ct}")

    if not all_results:
        raise RuntimeError("No WFO folds completed — check INITIAL_TRAIN/VAL/TEST vs data length.")

    combined = pd.concat(all_results).sort_index()
    print(f"\n[WFO]  Total OOS events: {len(combined):,}  |  Approved trades: {(combined['trade_signal'] != 0).sum()}")
    return combined
