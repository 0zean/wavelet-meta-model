"""
Rule pass (SPEC §17, U17): the signals of a fixed rule primary over a whole span, with no meta-model and no
walk-forward fitting (RunConfig.META_MODEL = "none"; the Phase-1 family cells, PLAN2 U18: "for pure rules there is no
fit and the WFO is a single pass").

Every sampled event (cfg.EVENT_SAMPLER) with a barrier width gets the rule's side and magnitude; flat sides stay in the
frame and are never traded. The frame has the WFO's signal columns (wfo.wfo_engine.run_wfo), so the backtest reads it
unchanged: trade_signal = signed_dir (nothing filters the rule), bet_size = the rule's magnitude under the `rule_size`
sizer, 1 under `fixed` (0 on flat events), meta_prob NaN.

Only the feature groups the rule reads (primary.needs_groups()) are built; an event with a NaN in them is dropped (as
the WFO drops events with NaN features). State that must be fit — a time-of-day volatility profile (VOL_PROFILE
"tod": the sampler's σ) or a per-fold feature group (vol_state's GARCH) — is fit causally in segments: the test
windows of wfo.wfo_engine.wfo_folds (TEST sessions each, the first after INITIAL_TRAIN + VAL sessions of warm-up), each
segment's state fit on every bar before it less the EMBARGO sessions, its events sampled on the full series with that
state (causal: an event uses bars <= it) and kept inside the segment. Without such state the pass is one segment over
the whole span. `attrs["live_start"]` = the first bar a segment covers: the stream starts there.
"""

import numpy as np
import pandas as pd

from features.events import sample_events
from features.feature_builder import FeatureSet
from features.vol_profile import fit_profile
from primaries import check_signal, make_primary
from utils.config import RunConfig

SIGNAL_COLUMNS = [
    "clf_prob",
    "direction",
    "signed_dir",
    "magnitude",
    "signal",
    "confidence",
    "meta_prob",
    "trade_signal",
    "width",
    "fold",
    "primary",
    "bet_size",
]  # fmt: skip  (run_wfo's columns)


def _segments(index: pd.DatetimeIndex, cfg: RunConfig, stateful: bool) -> list[tuple[int, int, int]]:
    """(start, end, fit_end) bar positions: events in [start, end), state fit on bars [0, fit_end)."""
    if not stateful:
        return [(0, len(index), 0)]
    from wfo.wfo_engine import wfo_folds

    return [(f.val_end, f.test_end, f.val_end - f.val_embargo) for f in wfo_folds(index, cfg)]


def rule_signals(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    context: dict | None = None,
    symbol: str | None = None,
    feature_cache_dir=None,
) -> pd.DataFrame:
    """The rule pass of cfg.PRIMARY over df (module docstring). Raises if cfg.META_MODEL is not "none"."""
    if cfg.META_MODEL != "none":
        raise ValueError(f"the rule pass needs META_MODEL='none', got {cfg.META_MODEL!r}")
    cfg.holdout_guard(df.index)
    prim = make_primary(cfg)
    needs = tuple(getattr(prim, "needs_groups", lambda: ())())
    fset = FeatureSet(cfg, needs, context=context, symbol=symbol, cache_dir=feature_cache_dir, require_core=False)
    base = fset.build(df)
    stateful = cfg.VOL_PROFILE == "tod" or bool(fset.per_fold)
    segments = _segments(df.index, cfg, stateful)
    frames, skipped = [], []
    events = sample_events(df, cfg, symbol=symbol) if not stateful else None
    for k, (start, end, fit_end) in enumerate(segments, start=1):
        if stateful:
            try:
                states = fset.fit(df.iloc[:fit_end])
            except RuntimeWarning as e:  # a per-fold group could not be fit (too few returns, no convergence)
                print(f"[RULE]  segment {k}: state fit failed ({e}); no events there")
                skipped.append(k)
                continue
            X_all = base.iloc[:end].join(fset.transform(df.iloc[:end], states))
            ev = sample_events(df, cfg, profile=fit_profile(df.iloc[:fit_end], cfg), symbol=symbol)
        else:
            X_all, ev = base, events
        pos = df.index.get_indexer(ev.index)
        ev = ev[(pos >= start) & (pos < end)].dropna(subset=["width"])
        X = X_all.reindex(ev.index)
        X = X.dropna() if X.shape[1] else X  # (dropna drops every row of a frame without columns)
        if not len(X):  # (not X.empty: a frame without columns is "empty" whatever its rows)
            continue
        fr = check_signal(prim.signal(df.iloc[:end], X, cfg), X, prim)
        frames.append(fr.assign(width=ev.loc[X.index, "width"], fold=k))
    if frames:
        out = pd.concat(frames).sort_index()
    else:
        out = pd.DataFrame(columns=["clf_prob", "direction", "signed_dir", "magnitude", "signal", "confidence",
                                    "width", "fold"], index=df.index[:0], dtype=float)  # fmt: skip
    sided = out["signed_dir"].to_numpy(dtype=float) != 0
    mag = out["magnitude"].to_numpy(dtype=float)
    if cfg.SIZER == "rule_size" and sided.any() and (np.isnan(mag[sided]).any() or (mag[sided] < 0).any()
                                                     or (mag[sided] > 1).any()):  # fmt: skip
        raise ValueError(f"primary {cfg.PRIMARY!r}: magnitudes of sided events must lie in [0, 1] for rule_size")
    out["meta_prob"] = np.nan
    out["trade_signal"] = out["signed_dir"]
    out["primary"] = prim.name
    out["bet_size"] = np.where(sided, mag if cfg.SIZER == "rule_size" else 1.0, 0.0)
    out = out[SIGNAL_COLUMNS]
    out.attrs["live_start"] = df.index[segments[0][0]] if segments and segments[0][0] < len(df) else None
    out.attrs["n_segments"] = len(segments)
    out.attrs["skipped_segments"] = skipped
    print(f"[RULE]  {prim.name}: {len(out)} events, {int(sided.sum())} sided, {len(segments)} segment(s)")
    return out
