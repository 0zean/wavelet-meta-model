import time
from typing import NamedTuple

import numpy as np
import pandas as pd

from features.feature_builder import FeatureSet
from features.selection import clustered_mda
from features.triple_barrier_labels import average_uniqueness, sample_events, triple_barrier_labels
from models.meta_model import fit_meta_model, make_meta_labels, meta_predict, oof_meta_prob, side_returns
from models.zoo import ZooFitError, inner_cv
from primaries import check_signal, make_primary
from sizing import SizerFitError, make_sizer
from utils.config import RunConfig

ONE_SIDED_SHARE = 0.10  # warn when a split's long share falls outside [10%, 90%]


class NoFitError(RuntimeError):
    """No walk-forward window could be fit (too little data or too few events): a counted no-fit, not a crash."""


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


def oof_primary(
    df: pd.DataFrame,
    X: pd.DataFrame,
    labels: pd.DataFrame,
    weights: pd.Series,
    cfg: RunConfig,
    select=None,
    fit_start: int = 0,
) -> pd.DataFrame:
    """
    Out-of-fold primary frame for the fitting events (META_TRAIN="oof"): over a purged k-fold of their spans
    (ZOO_CV_SPLITS, CV_EMBARGO_PCT), a fresh primary is fit on each purged train split and signals its test
    split, so no event's primary signal comes from a model that saw its label. `select(X, labels, weights)`
    (clustered MDA) → kept columns is run on each train split too, so feature selection never sees a test
    split's labels either. `df` ends at the fitting split's end; the primaries are fit on df.iloc[fit_start:] (a
    rolling window) and signal with the whole prefix as history. Fixed rules ignore the fit, so their frame equals
    the plain signal.
    """
    cv = inner_cv(labels, cfg)
    frames = []
    for train, test in cv.split(X):
        Xtr, lab, w = X.iloc[train], labels.iloc[train], weights.iloc[train]
        cols = list(X.columns) if select is None else select(Xtr, lab, w)
        p = make_primary(cfg).fit(df.iloc[fit_start:], Xtr[cols], lab, w, cfg)
        Xte = X.iloc[test][cols]
        frames.append(check_signal(p.signal(df, Xte, cfg), Xte, p.name))
    return pd.concat(frames).loc[X.index]


def fold_sizers(names, meta_mdl, df, events, X_fit, prim_fit, meta_lbl, w_fit, lab_fit, cfg: RunConfig) -> dict:
    """
    Fit each named sizer (SPEC §7) on the fold's train-window inputs: out-of-fold meta-probabilities of the
    meta-model's fitting events and their side returns (computed once, only if a sizer needs them). Returns
    name → fitted sizer, or None where it cannot be fit (no meta-model, a single-class OOF split, SizerFitError):
    that sizer then takes no trades in the fold.
    """
    sizers = {n: make_sizer(n, cfg) for n in names}
    if meta_mdl is None:
        return dict.fromkeys(names)
    p_oof = ret = None
    if any(sz.needs_train for sz in sizers.values()):
        try:
            p_oof = oof_meta_prob(X_fit, prim_fit, meta_lbl, w_fit, cfg, lab_fit)
        except ZooFitError as e:
            print(f"[SIZE]  OOF meta-probabilities failed ({e}) — train-fit sizers skip this fold")
            return {n: (sz if not sz.needs_train else None) for n, sz in sizers.items()}
        ret = side_returns(df, events.loc[p_oof.index], prim_fit.loc[p_oof.index], cfg).loc[p_oof.index]
    out = {}
    for n, sz in sizers.items():
        try:
            out[n] = sz.fit(p_oof, ret) if sz.needs_train else sz
        except SizerFitError as e:
            print(f"[SIZE]  {e} — no trades for this sizer in the fold")
            out[n] = None
    return out


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


class Prepared(NamedTuple):
    """Whole-series inputs shared by every fold / window (all causal: bar t uses bars <= t)."""

    events: pd.DataFrame  # CUSUM events + barrier widths
    labels: pd.DataFrame  # triple-barrier outcomes (read only through `purged`)
    fset: FeatureSet
    base_feats: pd.DataFrame  # static feature groups
    event_pos: pd.Series  # bar position of each event
    size_names: tuple[str, ...]  # cfg.SIZER first, then the extra sizers


class WindowFit(NamedTuple):
    status: str  # "ok" | "fracdiff_failed" | "insufficient_events" | "primary_failed"
    oos: pd.DataFrame | None  # OOS predictions of the test events (run_wfo's per-fold frame)
    in_sample: pd.DataFrame | None  # the same models' predictions on their own fitting events (if asked)
    meta_skipped: bool  # no meta-model could be fit (no trades)
    sizer_skipped: tuple[str, ...]  # sizers that could not be fit while a meta-model exists
    n_fit_events: int  # meta-model fitting rows


def prepare(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    context: dict[str, pd.DataFrame] | None = None,
    symbol: str | None = None,
    feature_cache_dir=None,
    sizers: tuple[str, ...] = (),
) -> Prepared:
    """Events, labels and static features on the full series, once per run (see run_wfo)."""
    cfg.holdout_guard(df.index)
    make_primary(cfg)  # fail fast on an unknown primary or bad PRIMARY_PARAMS

    print("\n" + "═" * 60)
    print("  Sampling events, labels and causal features on the full dataset")
    print("═" * 60)
    events = sample_events(df, cfg)
    labels = triple_barrier_labels(df, events, cfg)
    fset = FeatureSet(cfg, context=context, symbol=symbol, cache_dir=feature_cache_dir)
    base_feats = build_features(df, cfg, fset)
    event_pos = pd.Series(df.index.get_indexer(events.index), index=events.index)
    return Prepared(events, labels, fset, base_feats, event_pos, tuple(dict.fromkeys((cfg.SIZER, *sizers))))


def fit_window(
    df: pd.DataFrame,
    cfg: RunConfig,
    prep: Prepared,
    fold: int,
    fit_start: int,
    train_end: int,
    val_end: int,
    test_end: int,
    emb_tr: int,
    emb_vl: int,
    *,
    in_sample: bool = False,
) -> WindowFit:
    """
    One walk-forward step: fit on train = [fit_start, train_end) and val = [train_end, val_end), predict the test
    events in [val_end, test_end). fit_start = 0 is the expanding WFO (run_wfo); a rolling window (wfo/pwfo.py)
    starts later, and then every fitted state (fracdiff d, feature selection, primary, meta-model, sizers) sees
    only bars and events from fit_start on. Steps 1–6 of run_wfo.

    in_sample=True also returns the fitted models' predictions on their own fitting events (train + val purged
    at val_end), the in-sample side of the walk-forward efficiency (SPEC §6).
    """
    events, labels, fset, base_feats, event_pos, size_names = prep
    N = len(df)

    # ── Features: per-fold groups (fracdiff d) fit on train bars before the embargo ──
    # (fitting on the embargo bars would make every train feature depend on them)
    try:
        states = fset.fit(df.iloc[fit_start : train_end - emb_tr])
    except RuntimeWarning as e:  # FracdiffStat raises this when no d <= 1 is stationary
        print(f"[WFO]  Fold {fold}: fracdiff failed ({e}) — skipping")
        return WindowFit("fracdiff_failed", None, None, False, (), 0)
    X_all = base_feats.iloc[:test_end].join(fset.transform(df.iloc[:test_end], states))

    # ── Samples: events with complete features; fitting splits purged ─────
    lab_tr = purged(labels, fit_start, train_end, emb_tr)
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
        return WindowFit("insufficient_events", None, None, False, (), 0)

    w_tr = average_uniqueness(lab_tr, N)
    w_vl = average_uniqueness(lab_vl, N)

    cmda = cfg.FEATURE_SELECTION == "cmda"
    try:
        # ── Primary + meta-model training rows (cfg.META_TRAIN) ──────────
        if cfg.META_TRAIN == "val":
            # Feature selection on purged train events only; primary fit on them (bars up to the train end);
            # meta on val events
            if cmda:
                kept = select_features(X_tr, lab_tr, w_tr, cfg, train_end - fit_start).kept
                X_tr, X_vl, X_ts = X_tr[kept], X_vl[kept], X_ts[kept]
            prim = make_primary(cfg).fit(df.iloc[fit_start:train_end], X_tr, lab_tr, w_tr, cfg, val=(X_vl, lab_vl))
            prim_fit = check_signal(prim.signal(df.iloc[:val_end], X_vl, cfg), X_vl, prim.name)
            X_fit, lab_fit, w_fit = X_vl, lab_vl, w_vl
        else:
            # Train+val events purged at the val end; out-of-fold primary signals (feature selection inside
            # each OOF split); then selection and the primary refit on all of them
            lab_fit = purged(labels, fit_start, val_end, emb_vl)
            X_fit = X_all.reindex(lab_fit.index).dropna()
            lab_fit = lab_fit.loc[X_fit.index]
            w_fit = average_uniqueness(lab_fit, N)
            n_sel = val_end - fit_start
            select = (lambda X, lab, w, n=n_sel: select_features(X, lab, w, cfg, n).kept) if cmda else None
            prim_fit = oof_primary(df.iloc[:val_end], X_fit, lab_fit, w_fit, cfg, select=select, fit_start=fit_start)
            if cmda:
                kept = select_features(X_fit, lab_fit, w_fit, cfg, n_sel).kept
                X_fit, X_ts = X_fit[kept], X_ts[kept]
            prim = make_primary(cfg).fit(df.iloc[fit_start:val_end], X_fit, lab_fit, w_fit, cfg)
    except ZooFitError as e:  # a non-legacy PRIMARY_MODEL with a single-class (inner) train split
        print(f"[WFO]  Fold {fold}: primary could not be fit ({e}) — skipping")
        return WindowFit("primary_failed", None, None, False, (), 0)

    # ── Side-aware meta-labels → meta-model ──────────────────────────────
    meta_lbl = make_meta_labels(df, events.loc[X_fit.index], prim_fit, cfg)
    meta_mdl = fit_meta_model(X_fit, prim_fit, meta_lbl, w_fit, cfg, lab_fit)

    # ── Final prediction on OOS test fold ─────────────────────────────────
    fitted = fold_sizers(size_names, meta_mdl, df, events, X_fit, prim_fit, meta_lbl, w_fit, lab_fit, cfg)

    def predict(X: pd.DataFrame, end: int) -> pd.DataFrame:
        prim_x = check_signal(prim.signal(df.iloc[:end], X, cfg), X, prim.name)
        res = meta_predict(meta_mdl, X, prim_x, cfg.META_THRESH)
        res["width"] = events.loc[X.index, "width"]
        res["fold"] = fold
        res["primary"] = prim.name
        for n, sz in fitted.items():
            col = "bet_size" if n == cfg.SIZER else f"bet_size:{n}"
            res[col] = 0.0 if sz is None else np.where(res["trade_signal"] != 0, sz.size(res["meta_prob"]), 0.0)
        return res

    result_ts = predict(X_ts, test_end)
    is_frame = None
    if in_sample:
        X_is = X_fit if cfg.META_TRAIN == "oof" else pd.concat([X_tr, X_vl]).sort_index()
        is_frame = predict(X_is, val_end)
    shares = {cfg.META_TRAIN: (prim_fit["signed_dir"] > 0).mean(), "test": (result_ts["signed_dir"] > 0).mean()}
    print(f"[PRIM]  {prim.name}: long share " + "  ".join(f"{k}={v:.3f}" for k, v in shares.items()))
    for split, sh in shares.items():
        if not ONE_SIDED_SHARE <= sh <= 1 - ONE_SIDED_SHARE:
            print(f"[PRIM]  Warning: fold {fold} {split} sides are {sh:.1%} long (one-sided primary)")
    skipped = tuple(n for n, sz in fitted.items() if sz is None and meta_mdl is not None)
    return WindowFit("ok", result_ts, is_frame, meta_mdl is None, skipped, len(X_fit))


def run_wfo(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    context: dict[str, pd.DataFrame] | None = None,
    symbol: str | None = None,
    feature_cache_dir=None,
    sizers: tuple[str, ...] = (),
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
         (cfg.FEATURE_SELECTION = "cmda"); fit the primary (cfg.PRIMARY, primaries/) on train events —
         a no-op for fixed rules; ml_xgb fits its classifier (active days) and regressor
      4. Meta-model training rows (cfg.META_TRAIN): "val" = primary signal on val events → side-aware
         meta-labels → fit cfg.META_MODEL on val; "oof" = purged out-of-fold primary signals on train+val
         events → meta-model on all of them, primary refit on train+val. A zoo meta/primary model runs its
         HP search + calibration by purged CV inside its own fitting rows (models/zoo.py).
      5. Bet sizers (cfg.SIZER, plus `sizers`) fit on the fitting events' OOF meta-probabilities (sizing/)
      6. Final trade signal and bet size on test (OOS) events → store

    Args:
        df (pd.DataFrame): Full dataset to perform WFO.
        cfg (RunConfig): Run configuration.
        context (dict | None): Extra aligned inputs for feature groups, e.g. {"market": SPY bars}.
        symbol (str | None): Symbol name; with feature_cache_dir enables the feature cache.
        feature_cache_dir (path | None): Feature cache root (features.cache).
        sizers (tuple[str, ...]): Extra sizers to evaluate alongside cfg.SIZER (column `bet_size:<name>`).

    Raises:
        HoldoutError: If df reaches HOLDOUT_START and cfg.ALLOW_HOLDOUT is off.
        NoFitError: If no WFO folds are completed.

    Returns:
        pd.DataFrame: OOS predictions for every test event, concatenated across folds
            (primary frame, meta_prob, trade_signal, width, fold, primary, bet_size). `bet_size` ∈ [0, 1] is
            cfg.SIZER's raw size (0 where no trade), before active-bet averaging and discretization (backtest).
    """
    prep = prepare(df, cfg, context=context, symbol=symbol, feature_cache_dir=feature_cache_dir, sizers=sizers)
    size_names = prep.size_names
    all_results, meta_skipped, primary_skipped = [], [], []
    sizer_skipped = {n: [] for n in size_names}
    for fold, (train_end, val_end, test_end, emb_tr, emb_vl) in enumerate(wfo_folds(df.index, cfg), start=1):
        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold}: train[0:{train_end}]  val[{train_end}:{val_end}]  test[{val_end}:{test_end}]")
        print(f"{'─' * 60}")

        t0 = time.time()
        res = fit_window(df, cfg, prep, fold, 0, train_end, val_end, test_end, emb_tr, emb_vl)
        if res.status == "primary_failed":
            primary_skipped.append(fold)
        if res.status != "ok":
            continue
        if res.meta_skipped:
            meta_skipped.append(fold)
        for n in res.sizer_skipped:
            sizer_skipped[n].append(fold)
        result_ts = res.oos
        all_results.append(result_ts)

        elapsed = time.time() - t0
        trade_ct = (result_ts["trade_signal"] != 0).sum()
        print(f"[WFO]  Fold {fold} done in {elapsed:.1f}s  |  OOS events={len(result_ts)}  approved={trade_ct}")

    if not all_results:
        raise NoFitError("No WFO folds completed — check INITIAL_TRAIN/VAL/TEST vs data length.")

    combined = pd.concat(all_results).sort_index()
    combined.attrs["meta_skipped_folds"] = meta_skipped  # folds whose meta-model could not be fit (no trades)
    combined.attrs["primary_skipped_folds"] = primary_skipped  # folds dropped: primary could not be fit
    combined.attrs["sizer_skipped_folds"] = sizer_skipped[cfg.SIZER]  # sizer could not be fit (no trades)
    for n in size_names[1:]:
        combined.attrs[f"sizer_skipped_folds:{n}"] = sizer_skipped[n]
    if primary_skipped:
        print(f"[WFO]  Primary skipped in {len(primary_skipped)} fold(s): {primary_skipped} (no OOS rows)")
    if sizer_skipped[cfg.SIZER]:
        print(f"[WFO]  Sizer {cfg.SIZER} skipped in folds {sizer_skipped[cfg.SIZER]} (no trades there)")
    if meta_skipped:
        print(f"[WFO]  Meta-model skipped in {len(meta_skipped)} fold(s): {meta_skipped} (no trades there)")
    print(f"\n[WFO]  Total OOS events: {len(combined):,}  |  Approved trades: {(combined['trade_signal'] != 0).sum()}")
    return combined
