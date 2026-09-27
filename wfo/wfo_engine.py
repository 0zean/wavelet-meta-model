import time

import pandas as pd

from features.feature_builder import build_features
from features.fractional_diff import fit_fracdiff_d, fracdiff_transform
from features.triple_barrier_labels import average_uniqueness, sample_events, triple_barrier_labels
from models.meta_model import fit_meta_model, make_meta_labels, meta_predict
from models.primary_classifier import fit_primary_classifier
from models.primary_regressor import fit_primary_regressor
from utils.config import config
from utils.movement_filter import make_active_day_mask
from utils.primary_signal import primary_signal


def purged(labels: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Label rows whose event bar t lies in [start, end) AND whose exit is observed
    before `end` — a label that is resolved by prices in the next split is
    purged, so no fitting sample peeks into a later split (López de Prado 2018, §7.4).
    """
    t = labels["entry_pos"] - 1  # an event at bar t enters at open[t+1]
    return labels[(t >= start) & (t < end) & (labels["exit_pos"] < end)]


def run_wfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding-window Walk-Forward Optimisation on CUSUM events.

    Fold structure (bars, not calendar time):
    ┌────────────────────────────────────────────────────────┐
    │ TRAIN (expanding) │  VAL (fixed)  │  TEST (OOS/fixed)  │
    └────────────────────────────────────────────────────────┘
    ↑                   ↑               ↑                    ↑
    0             train_end         val_end              test_end

    Once, on the full series (all causal — bar t uses bars <= t):
      • CUSUM events + barrier widths, and the non-fracdiff features
      • Triple-barrier outcomes (TARGETS, read only through `purged`)

    At each fold:
      1. Fit FracDiffStat d on train close → transform the continuous series
      2. Purge train/val events whose barrier exit falls in the next split
      3. Fit primary classifier (active days) and regressor on train events
      4. Primary signal on val → side-aware meta-labels → fit meta-model on val
      5. Final trade signal on test (OOS) events → store

    Args:
        df (pd.DataFrame): Full dataset to perform WFO.

    Raises:
        RuntimeError: If no WFO folds are completed.

    Returns:
        pd.DataFrame: OOS predictions for every test event, concatenated across folds.
    """
    N = len(df)
    T0 = config.INITIAL_TRAIN_BARS
    V = config.VAL_BARS
    TS = config.TEST_BARS

    print("\n" + "═" * 60)
    print("  Sampling events, labels and causal features on the full dataset")
    print("═" * 60)
    events = sample_events(df)
    labels = triple_barrier_labels(df, events)
    base_feats = build_features(df)
    event_pos = pd.Series(df.index.get_indexer(events.index), index=events.index)

    all_results = []
    for fold, train_end in enumerate(range(T0, N - V - TS + 1, TS), start=1):
        val_end = train_end + V
        test_end = val_end + TS

        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold}: train[0:{train_end}]  val[{train_end}:{val_end}]  test[{val_end}:{test_end}]")
        print(f"{'─' * 60}")

        t0 = time.time()

        # ── Features: fracdiff d fit on train, applied causally to the series ──
        try:
            fs = fit_fracdiff_d(df["close"].iloc[:train_end])
        except RuntimeWarning as e:  # FracdiffStat raises this when no d <= 1 is stationary
            print(f"[WFO]  Fold {fold}: fracdiff failed ({e}) — skipping")
            continue
        X_all = base_feats.iloc[:test_end].assign(fd_close=fracdiff_transform(df["close"].iloc[:test_end], fs))

        # ── Samples: events with complete features; fitting splits purged ─────
        lab_tr = purged(labels, 0, train_end)
        lab_vl = purged(labels, train_end, val_end)
        ev_ts = events[(event_pos >= val_end) & (event_pos < test_end)]

        X_tr = X_all.reindex(lab_tr.index).dropna()
        X_vl = X_all.reindex(lab_vl.index).dropna()
        X_ts = X_all.reindex(ev_ts.index).dropna()
        lab_tr, lab_vl = lab_tr.loc[X_tr.index], lab_vl.loc[X_vl.index]

        if len(X_tr) < config.MIN_TRAIN_EVENTS or len(X_vl) < config.MIN_VAL_EVENTS or X_ts.empty:
            print(f"[WFO]  Fold {fold}: insufficient events (train={len(X_tr)}, val={len(X_vl)}) — skipping")
            continue

        w_tr = average_uniqueness(lab_tr, N)
        w_vl = average_uniqueness(lab_vl, N)

        # ── Active-day mask (train only) ──────────────────────────────────────
        active_mask = make_active_day_mask(df.iloc[:train_end]).loc[X_tr.index]

        # ── Fit primary models ────────────────────────────────────────────────
        clf = fit_primary_classifier(X_tr, lab_tr["label"], w_tr, X_vl, lab_vl["label"], active_mask)
        reg = fit_primary_regressor(X_tr, lab_tr["ret"], w_tr, X_vl, lab_vl["ret"])

        # ── Primary signal on val → meta-labels ──────────────────────────────
        prim_val = primary_signal(clf, reg, X_vl)
        meta_lbl = make_meta_labels(df, events.loc[X_vl.index], prim_val)
        meta_mdl = fit_meta_model(X_vl, prim_val, meta_lbl, w_vl)

        # ── Final prediction on OOS test fold ─────────────────────────────────
        prim_ts = primary_signal(clf, reg, X_ts)
        result_ts = meta_predict(meta_mdl, X_ts, prim_ts)
        result_ts["width"] = ev_ts.loc[X_ts.index, "width"]
        result_ts["fold"] = fold
        all_results.append(result_ts)

        elapsed = time.time() - t0
        trade_ct = (result_ts["trade_signal"] != 0).sum()
        print(f"[WFO]  Fold {fold} done in {elapsed:.1f}s  |  OOS events={len(result_ts)}  approved={trade_ct}")

    if not all_results:
        raise RuntimeError("No WFO folds completed — check INITIAL_TRAIN_BARS vs data length.")

    combined = pd.concat(all_results).sort_index()
    print(f"\n[WFO]  Total OOS events: {len(combined):,}  |  Approved trades: {(combined['trade_signal'] != 0).sum()}")
    return combined
