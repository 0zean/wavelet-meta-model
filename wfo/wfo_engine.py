import time

import pandas as pd

from features.feature_builder import build_features
from features.fractional_diff import fracdiff_fit_transform
from features.triple_barrier_labels import triple_barrier_labels
from models.meta_model import fit_meta_model, make_meta_labels, meta_predict
from models.primary_classifier import fit_primary_classifier
from models.primary_regressor import fit_primary_regressor
from utils.config import config
from utils.movement_filter import make_active_day_mask
from utils.primary_signal import primary_signal


def run_wfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding-window Walk-Forward Optimisation.

    Fold structure (bars, not calendar time):
    ┌────────────────────────────────────────────────────────┐
    │ TRAIN (expanding) │  VAL (fixed)  │  TEST (OOS/fixed)  │
    └────────────────────────────────────────────────────────┘
    ↑                   ↑               ↑                    ↑
    0             train_end         val_end              test_end

    At each fold:
      1. Compute triple-barrier labels on FULL split (targets only, no leakage)
      2. Fit FracDiffStat on train close → transform train/val/test
      3. Build feature matrix for all splits
      4. Fit primary classifier on filtered train bars
      5. Fit primary regressor on all train bars
      6. Generate OOF primary signal on val → compute meta-labels
      7. Fit meta-model on val
      8. Generate final trade signal on test (OOS) → store

    Args:
        df (pd.DataFrame): Full dataset to perform WFO.

    Raises:
        RuntimeError: If no WFO folds are completed.

    Returns:
        pd.DataFrame: OOS predictions concatenated across all folds,
        aligned to df's index.  Used directly by the VectorBT backtest engine.
    """
    N = len(df)
    T0 = config.INITIAL_TRAIN_BARS
    V = config.VAL_BARS
    TS = config.TEST_BARS

    all_results = []
    fold = 0

    # Compute labels once for entire dataset (uses future data → targets only)
    print("\n" + "═" * 60)
    print("  Computing triple-barrier labels for full dataset")
    print("═" * 60)
    labels_full = triple_barrier_labels(df)

    train_end = T0
    while train_end + V + TS <= N:
        fold += 1
        val_end = train_end + V
        test_end = val_end + TS

        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold}: train[0:{train_end}]  val[{train_end}:{val_end}]  test[{val_end}:{test_end}]")
        print(f"{'─' * 60}")

        t0 = time.time()

        # ── Split data ───────────────────────────────────────────────────────
        df_tr = df.iloc[:train_end].copy()
        df_vl = df.iloc[train_end:val_end].copy()
        df_ts = df.iloc[val_end:test_end].copy()

        y_tr = labels_full.iloc[:train_end]
        y_vl = labels_full.iloc[train_end:val_end]
        y_ts = labels_full.iloc[val_end:test_end]

        # ── Fractional differencing ───────────────────────────────────────────
        fd_tr, fd_vl, fd_ts, _ = fracdiff_fit_transform(df_tr["close"], df_vl["close"], df_ts["close"])

        # ── Build features ────────────────────────────────────────────────────
        # IMPORTANT: wavelet transform is computed independently per split
        # using ONLY that split's close prices → no cross-split leakage.
        X_tr_raw = build_features(df_tr, fd_tr)
        X_vl_raw = build_features(df_vl, fd_vl)
        X_ts_raw = build_features(df_ts, fd_ts)

        # Align indices (fracdiff + wavelet both trim leading rows)
        def align(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
            common = X.dropna().index.intersection(y.dropna().index)
            return X.loc[common], y.loc[common]

        X_tr, y_tr_ = align(X_tr_raw, y_tr)
        X_vl, y_vl_ = align(X_vl_raw, y_vl)
        X_ts, y_ts_ = align(X_ts_raw, y_ts)

        if len(X_tr) < 200 or len(X_vl) < 50:
            print(f"[WFO]  Fold {fold}: insufficient data after alignment — skipping")
            train_end += TS
            continue

        feature_cols = X_tr.columns.tolist()

        # ── Active-day mask (train only) ──────────────────────────────────────
        active_mask = make_active_day_mask(df_tr).reindex(X_tr.index).fillna(False)

        # ── Fit primary models ────────────────────────────────────────────────
        clf, clf_thresh = fit_primary_classifier(X_tr, y_tr_, X_vl, y_vl_, active_mask)
        reg = fit_primary_regressor(X_tr, y_tr_, X_vl, y_vl_, df_tr, df_vl)

        # ── Primary signal on val → meta-labels ──────────────────────────────
        prim_val = primary_signal(clf, reg, X_vl, clf_thresh)
        meta_lbl = make_meta_labels(prim_val, y_vl_, clf_thresh)
        meta_mdl = fit_meta_model(X_vl, prim_val, meta_lbl)

        # ── Final prediction on OOS test fold ─────────────────────────────────
        prim_ts = primary_signal(clf, reg, X_ts, clf_thresh)
        result_ts = meta_predict(meta_mdl, X_ts, prim_ts)
        result_ts["fold"] = fold

        all_results.append(result_ts)

        elapsed = time.time() - t0
        trade_ct = (result_ts["trade_signal"] != 0).sum()
        print(f"[WFO]  Fold {fold} done in {elapsed:.1f}s  |  OOS trades={trade_ct}")

        train_end += TS  # expand by one TEST window each fold

    if not all_results:
        raise RuntimeError("No WFO folds completed — check INITIAL_TRAIN_BARS vs data length.")

    combined = pd.concat(all_results).sort_index()
    print(f"\n[WFO]  Total OOS bars: {len(combined):,}  |  Total signals: {(combined['trade_signal'] != 0).sum()}")
    return combined
