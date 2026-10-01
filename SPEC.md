# SPEC — interfaces, definitions and defaults

Companion to [PLAN.md](PLAN.md). Section numbers are referenced from the plan. Anything marked
**default** is a starting value, tunable in config; anything marked **invariant** must hold and be tested.

---

## §1 Data (U1)

**Source.** Alpaca Market Data v2 via `alpaca-py` `StockHistoricalDataClient`. Credentials: `API_KEY`,
`SECRET_KEY` from `.env` (loaded with `python-dotenv` inside `data/alpaca_source.py` only).

| Setting | Default | Notes |
|---|---|---|
| feed | `sip` | Historical SIP is allowed on the free plan except the latest 15 min. If the request is unauthorized, **raise** — never silently fall back to `iex` (IEX ≈ 2–3% of volume; different bars). |
| adjustment | `all` | Splits + dividends. Cache path includes adjustment so mixes are impossible. |
| history start | 2016-01-01 | Alpaca's practical history start. |
| timezone | `America/New_York` | Index is tz-aware in NY time. |

**Invariants.**
- Bar timestamp = **bar open time** (Alpaca convention). A 5Min bar stamped 09:30 covers [09:30, 09:35) and
  is known at 09:35. The pipeline treats "bar t" as known at the end of its interval; execution at `open[t+1]`.
- Intraday bars restricted to regular trading hours using the Alpaca market calendar (`get_calendar`),
  including early closes. No extended-hours bars enter features, labels or fills.
- Timeframes above 5Min (15Min, 30Min, 1Hour, **1Day**) are **resampled locally from RTH 5Min bars anchored at
  09:30**. Alpaca's native hourly bars are clock-aligned at :00 and include pre-market minutes; its native daily
  bars include extended-hours trades (U1 measured volume 1.1–1.3× RTH and closes up to 5.5% off the last RTH
  trade, e.g. XOM 2020-03-12), so they are not used. 1Min and 5Min are native. 1Day bars are stamped at NY midnight
  of the session date and are known at that session's close.
- Sessions with < `min_session_coverage` (default 0.5) of their expected base bars are dropped with a log line
  (Alpaca has holes: Nasdaq-listed names have only the 09:30 bar on 2018-05-02/03; several NYSE names have partial
  sessions on 2021-04-19, 2021-10-25, 2022-01-24, 2022-03-08). For 1Day, sessions missing their **last** 5Min bar
  are also dropped (their daily close would not be the session close; e.g. 11 symbols end at 15:20–15:30 on 2019-08-12).
- A fetch whose trailing sessions return no bars does not mark them as cached (coverage stops after the last
  session with bars, logged); empty sessions inside a range are logged.
- `load_bars(..., allow_holdout=False)` **raises** `HoldoutError` if `end > HOLDOUT_START` (§9). Only caching
  (`data.fetch`) and the final evaluation pass `allow_holdout=True`; the quality report covers the development window only.
- Schema: columns `open, high, low, close, volume, vwap, trade_count` (float64), unique sorted index.
  `load_ohlcv` still requires only OHLCV so the CSV path keeps working.

**Cache format.** `data/cache/{feed}/{adjustment}/{timeframe}/{SYMBOL}.npz` + `{SYMBOL}.json`.
- `.npz` (via `np.savez`, uncompressed for fast loads): `ts` = int64 nanoseconds since epoch, **UTC**; one
  float64 array per schema column. Loaded with `np.load(..., allow_pickle=False)` — no object arrays ever.
- `.json` sidecar: `schema_version, symbol, timeframe, feed, adjustment, tz ("America/New_York"), columns,
  n_bars, first_ts, last_ts (ISO), fetched_at, source ("alpaca"), session ("rth"), coverage_start, coverage_end,
  sha256` (hash over the `.npz` array bytes in column order).
- **Invariants:** writes are atomic (temp file + `os.replace`, `.npz` before `.json`); on load, a missing
  sidecar, sha256 mismatch, schema_version mismatch or feed/adjustment mismatch with the path **raises**;
  top-ups merge on `ts`, keep the newer fetch for duplicates, and re-verify sort/uniqueness.
- Only native timeframes (1Min, 5Min, 1Day) are cached; 15Min/30Min/1Hour are resampled from cached 5Min on
  every load (cheap, never stale). The exchange calendar is cached as `data/cache/calendar.json` (atomic write,
  re-fetched when older than 30 days so late-announced closures are picked up).
- Top-ups re-fetch a 7-day overlap; if overlapping closes differ (history re-adjusted by a new split/dividend)
  the whole covered range is re-fetched and replaced, with a log line.
- Ranges are whole NY trading days `[start, end)`; `end` is clamped to today so coverage never claims unfinished sessions.

**Default universe** (liquid, long history, shortable; survivorship-biased by construction — documented):
- ETFs: SPY, QQQ, IWM, DIA, XLF, XLK, XLE, XLV, TLT, GLD
- Stocks: AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, UNH

**Timeframes:** `1Min` (restricted to ≤ 3 symbols for data volume), `5Min`, `15Min`, `30Min`, `1Hour`, `1Day`.

---

## §2 RunConfig and timeframe scaling (U2)

`RunConfig` is a frozen dataclass passed explicitly (`cfg`) to every function that previously read the
module-level `config`. `RunConfig.for_timeframe(tf, **overrides)` fills the timeframe-dependent fields:

| tf | bars/day | vertical barrier (default) | vol span (bars) | hold overnight | WFO train₀ / val / test (trading days) |
|---|---|---|---|---|---|
| 1Min | 390 | 30 bars (30 min) | 390 | no | 21 / 10 / 5 |
| 5Min | 78 | 12 bars (1 h) | 100 | no | 42 / 21 / 10 |
| 15Min | 26 | 8 bars (2 h) | 78 | no | 126 / 42 / 21 |
| 30Min | 13 | 6 bars (3 h) | 65 | no | 189 / 63 / 21 |
| 1Hour | 7* | 6 bars (≈ session) | 70 | no | 252 / 126 / 21 |
| 1Day | 1 | 10 bars (2 wk) | 50 | yes | 1008 / 504 / 63 |

\* 09:30–16:00 anchored at 09:30 gives six full hours + a 30-min stub. The stub is kept as an ordinary bar
(it is the session close, which the vertical barrier truncates at); its shorter duration is left for a U3
`calendar` feature to expose rather than a flag column in the bar schema.

WFO windows (U2): the train window grows with bar length so every split clears `MIN_TRAIN_EVENTS = 200` /
`MIN_VAL_EVENTS = 100`. Measured on SPY 2016-01-04→2025-09-30 (CUSUM events per session: 5Min 15.7, 15Min 5.1,
30Min 2.5, 1Hour 1.3, 1Day 0.25), the smallest val split is 246 / 168 / 122 / 145 / 126 events and the first
train split 616 / 654 / 466 / 320 / 233; fold counts 238 / 108 / 104 / 98 / 14. 1Min was not measured (no
cache yet). A "trading day" is a session **present in the data**: the few sessions the data layer drops
(§1) are not counted, so a window spanning one is one calendar session longer. The PWFO (§6, U9) counts exchange-calendar sessions instead; `run_wfo` keeps data sessions (regression).

Timeframe-independent fields (existing semantics): `BARRIER_MULT`, `CUSUM_MULT`, `SLIPPAGE_PCT`,
`META_MIN_RET = 2·SLIPPAGE_PCT`, `SEED`, model params. WFO windows (`INITIAL_TRAIN`, `VAL`, `TEST`,
`EMBARGO`) are counted in `WINDOW_UNIT = "days"` (trading sessions; every split boundary is a session's
first bar) — `"bars"` exists only for the legacy regression. `RunConfig.replace(SEED=…)` also reseeds the
XGBoost params. `ALLOW_HOLDOUT` (default False): `run_wfo` raises `HoldoutError` if the data reaches
`HOLDOUT_START`; since the loader also stops there, no label can resolve inside the holdout during development.

**Overnight policy invariant.** `hold_overnight=False` → vertical barrier truncated at the entry session's
last bar, events on a session's last bar skipped (current behaviour). `True` → no truncation; the next
session's open is a normal bar (gap-through-barrier fills at that open, as today).

**Regression invariant.** `RunConfig.legacy_5min()` (5Min, bar windows 2000/1000/500, `EMBARGO = 0`) + CSV
input reproduces the pre-U2 `wfo_signals.csv` exactly (sha1 `402ef202…`, 754 rows).

---

## §3 Feature registry (U3)

```python
@feature_group(name, required=False, intraday_only=False, per_fold=False, needs=(), level_check=True)
def group(df: pd.DataFrame, cfg: RunConfig, context: dict) -> pd.DataFrame: ...


@feature_group(name, per_fold=True)
class Group:
    def fit(train_df, cfg) -> state: ...
    def transform(df, state, cfg) -> pd.DataFrame: ...
```
- Registry in `features/registry.py`; groups in `features/groups.py`; `FeatureSet` / `build_features(df, cfg, groups,
  *, context, symbol, cache_dir)` in `features/feature_builder.py`.
- Returns columns prefixed `{name}__` indexed like `df`; NaN during warm-up (≤ 200 bars for the whole zoo at 5Min–1Day; max(200, VOL_SPAN) — 390 at 1Min).
  Events whose features are NaN are dropped by `run_wfo` with a per-fold count logged.
- `per_fold=True` groups (fracdiff) are fit on each fold's train bars **before the train embargo**
  (`[0, train_end − embargo)`) and applied to the continuous series.
- `needs=("market",)` groups read `context["market"]` (bars of the market symbol, default SPY).
- `resolve_groups` **raises** if `wavelet_core` is absent, a name is unknown or repeated; skips `intraday_only`
  groups on 1Day with a log line. A group needing a context key that was not supplied raises.
- **Invariant:** only stationary transforms. Every static group's output is checked at build time: aligned index,
  prefix, no ±inf, and no column with |corr(column, close)| ≥ 0.99 (raw price level). Exempt: per-fold fracdiff
  (by design it keeps memory; LdP reports corr ≈ 0.99 at the minimum stationary d) and `calendar` (deterministic
  encodings can trend with a short sample). The guard only catches near-verbatim levels (e.g. SPY 1Day `log(close)` has
  |ρ| = 0.985 and passes): it is a backstop; stationarity of a new group is a review item.
- `cfg.FEATURE_GROUPS` default = `wavelet_core, trend, mean_reversion, volatility, microstructure, intraday,
  fracdiff` (covers the legacy matrix's information). `None` = the legacy matrix (`legacy_features` + `fd_close`), used
  only by `RunConfig.legacy_5min()` for the regression invariant: its `w_lag_*` are raw S_J levels.

| group | columns | notes |
|---|---|---|
| `wavelet_core` (required) | `s{J}_lag_k = log(S_J[t−k] / close[t])`, k = 1..AR_LAGS | causal MODWT smooth of close (legacy lags, made stationary) |
| `wavelet_ext` | per filter db1/db2/la8 on log close: relative detail energies `e1..eJ`, `vr = log(E_J/E_1)`, `slope = ΔS_J / σ_bar` | energy window 4·2^J bars |
| `trend` | `ret_1`, `mom_h`, `mom_4h` (h = VERTICAL_BARS), Siegel slope / close, `sma_dist_20/50`, `ema_dist_20` (log), ADX(14), (+DI−−DI)/100 | |
| `mean_reversion` | RSI, Bollinger %b(20, 2), z-score of close vs 50-bar mean | |
| `volatility` | EWM σ (VOL_SPAN), Parkinson / Garman–Klass (20), log(high/low), σ₂₀/σ_VOL_SPAN, vol-of-vol (50-bar std of log σ₂₀) | |
| `microstructure` | log(1+volume), volume z-score (window max(20, bars/day)), log Amihud (20), Roll spread (20), Corwin–Schultz (20-bar mean) | |
| `structural` | CUSUM filter S+/S− in threshold units, DF t-stat (50), SADF-lite = max DF t over windows 50/100/200 | entropy not implemented |
| `calendar` | day-of-week, month (sin/cos) | |
| `intraday` (intraday_only) | log(close/session VWAP), time of day of the bar close (sin/cos), `bar_frac` = scheduled bar length / timeframe (1Hour 15:30 stub = 0.5) | assumes a 16:00 close; early closes not known from bars |
| `cross_asset` (needs market) | market return and σ, 50-bar beta and corr, residual return, relative h-bar momentum | market close = last market bar stamped ≤ t |
| `fracdiff` (per_fold) | `fracdiff__close`: fixed-window (10) fracdiff, minimum ADF-stationary d (5 %, binary search to 0.01) fit on train — `fastfracdiff` (in-repo; replaces the archived fracdiff 0.9.0 and statsmodels' adfuller with equal results: one QR serves every AIC lag candidate) | |

**Feature cache** (`features/cache.py`, root `data/cache/features/{symbol}/{tf}/{group}/{key}.npz`, used when the WFO
has a symbol — Alpaca input). Key = sha256 over (group, symbol, timeframe, source hash of every `features/*.py`,
numpy/pandas/scipy/pywddff versions, all RunConfig fields except a listed set of model/WFO/backtest fields, data hash of the bars' timestamps + OHLCV(+vwap),
and the data hash of any context frame). The data hash subsumes range and adjustment. Plain numpy arrays, atomic
write, index re-verified on load. Per-fold groups are not cached.

**Clustered MDA** (López de Prado 2020, ch.6; `features/selection.py`, `cfg.FEATURE_SELECTION = "cmda"`): on each
fold's purged, embargoed train events only (directional labels ±1). Cluster features by `1 − |ρ|` (average-linkage
hierarchical, k ∈ [2, 10] by silhouette); `PurgedKFold(CMDA_SPLITS=4, embargo = ceil(CV_EMBARGO_PCT · train bars))`;
single-threaded random forest (`CMDA_TREES=100`, `max_features=1`, balanced_subsample, min_weight_fraction_leaf 0.05); importance =
drop in weighted neg-log-loss when a cluster's columns are permuted jointly; keep clusters with mean − 1 s.e. > 0.
`wavelet_core` columns are always kept; `cmda` with the legacy matrix (`FEATURE_GROUPS=None`) raises. Selection is applied to the train, val and test matrices of that fold.

## §4 Primary and model protocols (U4, U6)

```python
class Primary(Protocol):  # primaries/base.py; registered with @primary(name)
    name: str

    def fit(self, df, X, labels, weights, cfg, *, val=None) -> "Primary": ...  # no-op for fixed rules
    def signal(self, df, X, cfg) -> pd.DataFrame: ...  # PRIMARY_COLUMNS, index = X.index (the events)


side(p, df, X, cfg) = p.signal(df, X, cfg)["signed_dir"]  # {-1,+1}
```
- `cfg.PRIMARY` (default `ml_xgb`) selects the primary; `cfg.PRIMARY_PARAMS` are its constructor kwargs (fixed per
  experiment, unknown keys raise). CLI: `--primary NAME --primary-params '{"k": v}'`.
- In each WFO fold a fresh instance is fit on the purged train events with `df = bars[:train_end]`
  (`val` is for logging only), then `signal` is called on val events with `bars[:val_end]` and on test events with
  `bars[:test_end]`. `check_signal` raises unless the frame has exactly `PRIMARY_COLUMNS`, the events' index and
  sides in {−1, +1}. The frame is also the primary's meta-model input.
- `PRIMARY_COLUMNS = clf_prob, direction, signed_dir, magnitude, signal, confidence`. Rules: `clf_prob` = NaN,
  `magnitude` = `confidence` = |score|, `signal` = side·|score|; score 0 → long; a NaN score at an event raises.
- WFO output = primary frame + `meta_prob, trade_signal, width, fold, primary, bet_size` (§7). With `primary` and
  `bet_size` dropped, the legacy run still reproduces sha1 `402ef202…` (§2).

| primary | params (default) | score (sign = side) |
|---|---|---|
| `ml_xgb` | — (CLF_/REG_PARAMS) | XGB classifier P(long) ≥ CLF_THRESH; magnitude from the \|ret\| regressor (pre-U4 pipeline) |
| `sma_cross` | fast 20, slow 50 | log(SMA_fast / SMA_slow) / σ_bar |
| `bollinger_mr` | window 20 | −(close − mean_w) / std_w |
| `wavelet_trend` | mode `slope` \| `level` | (S_J[t] − S_J[t−1]) / σ_bar, or (log close − S_J) / σ_bar; S_J = causal MODWT smooth of log close |
| `donchian_breakout` | window 20 | (close − mid) / half-width of the previous `window` bars' channel (> 1 = breakout) |

σ_bar = EWM σ (VOL_SPAN); a zero denominator keeps the numerator's sign. Rule parameters are never tuned.

**Diagnostics** (`primaries/diagnostics.py`, post hoc on OOS events, per fold + all; `primary_diagnostics.csv` per run):
`n_events`, `long_share`, `precision` = P(side·ret > META_MIN_RET), `opportunity` = P(either side would), `recall` =
precision | opportunity, `turnover` = share of consecutive events whose side flips, `net_bp` = mean side-return net of
round-trip slippage. Returns use the backtest's barrier rules with each side (adverse same-bar ties).

Invariant: the primary frame at event t uses only bars ≤ t (causality test §8).

```python
class ZooModel(Protocol):
    name: str

    def fit(self, X, y, sample_weight, cv: PurgedKFold) -> "ZooModel": ...  # HP search + calibration inside
    def predict_proba(self, X) -> np.ndarray: ...  # calibrated P(y=1)
```
`models/zoo.py`; `make_model(name, cfg, role)` (role `meta` | `primary`), `cv = inner_cv(labels, cfg)` =
`PurgedKFold(ZOO_CV_SPLITS=4, ceil(CV_EMBARGO_PCT · bars spanned))` bound to the fitting rows' spans
[event bar, exit bar]. Inside `fit`, on the given rows only:
1. HP search: for each grid point, purged-CV OOF P(y=1) (`purged_cv_predict`: fresh clone per split, train-row
   weights), mean per-split `selection_score(SELECTION_METRIC)` with test-row weights; best wins, ties → first point.
2. Calibration: sigmoid (Platt on logit p) and isotonic are each cross-fitted on the winner's OOF predictions over the
   same splits; lower mean weighted Brier wins, ties → sigmoid.
3. Refit the winner on all rows; fit the chosen calibrator on all OOF predictions. `predict_proba` = calibrator(raw),
   clipped to [`PROB_CLIP` = 1e-3, 1 − 1e-3]. Columns all-NaN in the fitting rows (a rule's `clf_prob`) are dropped;
   any other NaN raises. A single-class fitting set or inner train split raises `ZooFitError`; the meta-model then
   skips the fold (no trades, listed in `signals.attrs["meta_skipped_folds"]`, excluded from OOS meta scoring).

| model | grid | fixed |
|---|---|---|
| `logit_l1` / `logit_l2` | C ∈ {0.01, 0.1, 1} | weighted StandardScaler fit inside `fit`; saga (L1) / lbfgs (L2) |
| `rf_ldp` | max_features ∈ {1, sqrt} | 500 trees, `balanced_subsample`, `max_samples` = mean weight (= avg uniqueness), `min_weight_fraction_leaf=0.05` |
| `extra_trees` | min_weight_fraction_leaf ∈ {0.01, 0.05} | 500 trees, sqrt, `balanced_subsample` |
| `xgb` | max_depth ∈ {2, 4} | META_PARAMS (meta) / CLF_PARAMS (primary) |
| `lightgbm` | num_leaves ∈ {7, 31} | 300 trees, lr 0.03, bagging 0.8, colsample 0.8, deterministic |
| `legacy` | — | pre-U6 XGBoost (META_PARAMS / CLF_PARAMS), no search, no calibration (regression path) |

Forests predict single-threaded (threaded prediction sums trees in completion order → last-bit nondeterminism).

**Config switches** (CLI `--meta-model`, `--primary-model`, `--meta-train`): `META_MODEL` (default `legacy`),
`PRIMARY_MODEL` (ml_xgb's direction classifier, default `legacy`; any other value with a rule primary raises),
`META_TRAIN`: `val` (pre-U6: primary fit on train, meta on val) or `oof` (fitting events = train+val purged at the
val end with its embargo; primary frames are purged k-fold OOF over those events (`wfo_engine.oof_primary`, df ends at
the val end); the meta-model trains on all of them; the primary is refit on all of them for test).
**OOS scoring** (`wfo_metrics.signal_diagnostics`, `meta_outcomes`, `calibration_table`): meta log-loss (p clipped to
`PROB_CLIP`), Brier and AUC on scored folds; `meta_calibration.csv` = 10-bin reliability curve.
**Comparison** (`python -m models.compare`): per model a WFO row + CPCV path-Sharpe distribution, appended to
`trials.jsonl` (stage `U6`).

---

## §5 Purging, embargo and cross-validation (U2, U5)

Each sample i has span `[t0_i, t1_i]` = `[event bar, exit bar]`.
- **Purge:** drop train sample i if its span overlaps any test span.
- **Embargo:** additionally drop train samples with `t0_i ∈ (max test t1, max test t1 + h]`, `h = ceil(embargo_pct · N_bars)`;
  default `embargo_pct = 0.01` (k-fold CV, U5).
- **Embargo in the forward WFO (U2):** train precedes val precedes test, so no fitting sample ever follows the
  split it is scored against; the embargo instead sits at the **end** of each fitting split: train (val) samples
  whose exit falls in the last `EMBARGO` trading days before val (test) starts are dropped, i.e. keep
  `exit_pos < split_end − h`. This removes the fitting outcomes most serially correlated with the next split's
  first outcomes. Default `EMBARGO = 1` day (h = that session's bar count, shorter after a half-day).
- **Splitters** (`validation/purged_cv.py`, U5): spans are integer bar positions, samples sorted by t0.
  `purged_train(t0, t1, test, h)` splits `test` into runs of consecutive indices; for each run with hull [a, b]
  (a = min t0, b = max t1) a train sample is kept only if `t1_i < a` or `t0_i > b + h`. Because samples are sorted by
  t0, overlapping a run's hull ⇔ overlapping one of its spans, so this equals the per-sample definition (property-tested
  against brute force). `embargo_bars(pct, n_bars) = ceil(pct · n_bars)` (product rounded to 1e-9 first; also used by clustered MDA). `splitter.bind(t0, t1)` is a scikit-learn
  `cv=` object (rows of X in span order).
- **PurgedKFold(n_splits, embargo):** contiguous, unshuffled folds (`np.array_split` sizes). Used by clustered MDA (U3).
- **CombinatorialPurgedCV(N, k, embargo):** N contiguous groups, all C(N,k) test combinations in `itertools.combinations`
  order; number of backtest paths φ = C(N,k)·k/N = C(N−1,k−1); each group is a test group in exactly φ splits. Path p
  takes group g from the p-th split (in enumeration order) that tests g (`path_splits()`; `assemble_paths` stitches
  per-split OOS outputs into φ full-length paths). Defaults `CPCV_GROUPS=10`, `CPCV_TEST_GROUPS=2` → 45 splits, 9 paths;
  `from_cfg(cfg, n_bars)` embargo = `ceil(CV_EMBARGO_PCT · n_bars)`.
- **Selection metric** (`validation/scoring.py`, `cfg.SELECTION_METRIC`): sample-weighted neg-log-loss (default) or
  −Brier, higher is better. AUC/F1/accuracy (`score_report`) are reported only. `purged_cv_predict/score` fit a fresh
  clone per split with the train rows' weights and score with the test rows' weights; a single-class train split raises.

---

## §6 PWFO definitions and statistics (U5, U9)

**Windows** (`wfo/pwfo.py`, U9). In trading days on the **exchange calendar** (`unit_bounds`: calendar sessions between
the first and last data session; a session the data layer dropped counts as a day with no bars, a half-day is one
session with fewer bars; a data session missing from the calendar raises). `unit="bars"` exists for the legacy
regression. For combo (IS, OOS, val), window w (units):
IS = [a_w, b_w) with b_w = IS + w·OOS and a_w = w·OOS (rolling) or 0 (`PWFO_EXPANDING`); train = [a_w, b_w − val),
val = [b_w − val, b_w); OOS = [b_w, b_w + OOS). Retraining cadence = OOS length; the OOS windows tile
[IS, IS + W·OOS) contiguously; only full OOS windows are run (a final partial one is dropped, as in the WFO).
val = round(`PWFO_VAL_FRAC`·IS) (default VAL/(INITIAL_TRAIN+VAL), ⅓ on 1Hour). Default grid IS ∈ {63, 126, 252, 504},
OOS ∈ {5, 10, 21, 63}.
- *Refinement — embargo.* There is no gap between IS and OOS; the embargo is the WFO's (§5): train (val) samples
  whose exit falls in the last `EMBARGO` sessions before val (OOS) starts are purged. Bars in a gap would never be
  traded, and the purge already keeps every fitting label resolved before the OOS starts.
- *Rolling isolation.* Everything fitted in window w (fracdiff d, cMDA selection, primary, meta-model, sizers) sees only
  events with t ≥ a_w and bars from a_w on; primary *signals* still read the bar prefix as warm-up history (causal).
- The legacy expanding WFO is the single combo (INITIAL_TRAIN+VAL, TEST, VAL) with `PWFO_EXPANDING` (tested in both
  units; `legacy_5min` reproduces the `402ef202…` CSV through `run_combo`).
- A window that cannot be fit is a counted skip (`status` = `insufficient_events` / `fracdiff_failed` /
  `primary_failed` / `empty_oos`): its OOS is flat. A combo with no fitted window is reported (`n_ok_windows` = 0),
  left out of selection and PBO, and still counts as a trial for the DSR.

**Per-combo statistics** (`combo_stats`). The combo's stitched OOS signals are backtested once (`run_backtest`,
meta-filtered, cfg sizing / risk profile) from its first OOS bar; daily close-to-close returns over its OOS sessions.
Window w's OOS return = compounded daily return over its sessions. IS: window w's models predict their own fitting
events (train+val purged at the IS end; `fit_window(in_sample=True)`), backtested on the IS bars → annualized return
and Sharpe.
- `n_oos_windows`, `n_ok_windows` (fitted); flag `weak` if `n_ok_windows < PWFO_MIN_WINDOWS` = 50, per Meyers.
- `WFE = annualized stitched OOS return / mean annualized IS return` over **all** windows (an unfitted window is flat
  on both sides, as in the OOS stream). *Refinement:* **NaN unless that mean is > 0 with t-statistic ≥
  `PWFO_WFE_MIN_T` = 2** across windows (`is_ret_t` reported): a ratio to an IS loss is meaningless and a ratio to an IS
  figure indistinguishable from 0 explodes (review: SPY IS504_OOS5 IS mean 0.0029, t 0.85 → WFE −6.5).
  `n_is_nonpos` = fitted windows whose IS return ≤ 0. `WFE_sharpe` the same with annualized Sharpes (`is_sharpe_t`).
- `pct_profitable_oos` (fitted windows; one without an OOS trade counts as not profitable), `pct_windows_traded`,
  `pct_profitable_traded` (among windows that traded), `oos_sharpe`, `oos_sortino` (daily, ×√252), `max_dd`, PSR(0),
  `is_oos_spearman` (IS annualized vs OOS window return across fitted windows), trades, turnover.
- PBO over the combo × day matrix of daily OOS returns of the run combos, on the days all of them cover
  (CSCV, `PBO_BLOCKS` S=16; `validation/pbo.py`): rows cut into S contiguous
  blocks; for each of the C(S, S/2) IS block sets, n* = best IS column (ties → lowest index), ω = OOS rank of n*
  (1 = worst, ties averaged)/(N+1), λ = log(ω/(1−ω)); PBO = P(λ < 0) + ½·P(λ = 0) (the exact-median case, odd N or
  ties, counts half, so noise gives ≈ 0.5 for any N). Default metric = per-period Sharpe from block moments of each column
  shifted by its first value (a flat column scores exactly 0, or ±inf with non-zero mean). `PBO_BLOCKS` must be even. Also reported: P(OOS loss of n*) and the IS→OOS degradation slope.

**Nested selection (walk-forward of the walk-forward)** (`nested_select`). Each run combo yields a causal daily OOS
return stream. Every `SELECT_EVERY = 10` trading days from the default combo's first OOS session, at decision session d
pick the combo with the highest Sharpe over its last `SELECT_LOOKBACK = 126` returns strictly before d (a flat
stream scores 0). Ties → higher WFE_sharpe over the lookback (that Sharpe / mean IS Sharpe of the windows whose OOS
started in it; a window's IS Sharpe is known when its OOS starts) → shorter IS → grid order. Until every run combo
has a full lookback, use the default combo `PWFO_DEFAULT` = (IS=252, OOS=10); that period is labeled burn-in and
excluded from headline stats. The PWFO stream = the chosen combo's returns over [d, d+SELECT_EVERY) (switching between
the combos' paper return streams); it ends where the earliest run combo's OOS ends. Invariant (tested): perturbing
returns or IS Sharpes from d on cannot change the combo chosen at d or before.
**DSR of the PWFO** = DSR of the live (post-burn-in) PWFO daily returns with N = every grid combo and V[SR_n] = variance of
the run combos' per-period Sharpes over the same days.

**Probabilistic Sharpe** PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ),
γ₄ = raw (non-excess) kurtosis, SR per-period (not annualized).
**Deflated Sharpe** DSR = PSR(SR₀) with SR₀ = √V[SR_n]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))), γ ≈ 0.5772,
N = number of trials from the ledger (§9), V[SR_n] = variance of trial Sharpes (per period). N = 1 → SR₀ = 0.
**MinTRL**(SR*, α) = 1 + (1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²)·(Φ⁻¹(1−α)/(SR̂ − SR*))²; ∞ if SR̂ ≤ SR*.
Moments (`validation/stats.py`): SR = mean/std (ddof 1), skew and raw kurtosis are the biased sample estimators.
Invariant (tested): the Bailey & López de Prado (2014) example — SR 2.5/√250, T = 1250, N = 100, V = ½/250, γ₃ = −3,
γ₄ = 10 — gives SR₀ = 0.1132 and DSR = 0.9004 (N = 46 → 0.9505).

---

## §7 Bet sizing and risk (U7, U8)

p = calibrated meta-probability, τ = `META_THRESH` ∈ (0, 1). `sizing/sizers.py`; every sizer returns m ∈ [0, 1],
non-decreasing in p, with m = 0 for p < τ; the sign comes from the primary side. `linear` and `ecdf` also start at 0 at
p = τ; `fixed`, `kelly_capped` and (for τ > ½) `ldp_sigmoid` jump at τ.
- `fixed`: m = 1 (pre-U7).
- `linear`: m = clip((p − τ)/(1 − τ), 0, 1).
- `ldp_sigmoid`: z = (p − 1/2)/√(p(1−p)), m = max(0, 2Φ(z) − 1).
- `ecdf`: m = F̂(p), F̂ = right-continuous ECDF of the train-window OOF meta-probabilities **that reach τ** (so m is
  the bet's rank among the train fold's approved bets; ranking against all OOF p would floor every size at F̂(τ)).
- `kelly_capped`: f* = p − (1−p)/b, m = clip(λ·f*, 0, 1), λ = `KELLY_FRACTION` = 0.25; b = mean win / mean loss of
  the net returns (side barrier return − 2·`SLIPPAGE_PCT`; win ⇔ net > 0, the meta-label at the default
  `META_MIN_RET`) of the train-window OOF events with p ≥ τ.
- **Train-window inputs** (`ecdf`, `kelly_capped`): per WFO fold, `meta_model.oof_meta_prob` = purged k-fold
  (`ZOO_CV_SPLITS`, `CV_EMBARGO_PCT`) over the meta-model's own fitting events (val, or train+val for
  `META_TRAIN="oof"`); a fresh `META_MODEL` (with its inner search + calibration) per split. Test events are never
  used. Computed only when a sizer needs it. No approved OOF event (ecdf), no approved win or loss (kelly) or a
  single-class OOF split → the sizer is unfit: the fold takes no trades, listed in `signals.attrs["sizer_skipped_folds"]`
  (and `wfo_run.json`).
- WFO column `bet_size` = raw m of `SIZER` on approved test events (0 elsewhere, and in skipped folds);
  `run_wfo(..., sizers=(...))` adds `bet_size:<name>` for extra sizers on the same signals (sizer comparison).
- **Post-processing** in the backtest (`wfo/backtest.py`), per `POSITION_MODE`:
  - `single` (default, pre-U7): one position at a time; each bet's m is discretized, m ← round(m/step)·step
    (`SIZE_STEP` = 0.1, half to even; 0 = off, otherwise 1/k for an integer k so m = 1 stays whole); m = 0 is no bet and does not block later events. A trade commits
    `SIZE`·m of equity at the entry fill. With `fixed` this is bit-identical to the U6 backtest.
  - `average` (López de Prado 10.4): every bet with raw m > 0 is live from open[t+1] to its own side-aware barrier
    exit; target exposure f = discretize(mean side·m over live bets). Changes are evaluated at each entry / exit
    (per bar: the open (entries, gap exits), then intrabar barrier exits in bet order, then vertical exits at the
    close; exits sharing a phase and price are one change; the running mean is rounded to 1e-12 so float residue
    never leaves a phantom position at step 0) and traded
    only when f changes: shares = f·`SIZE`·equity / fill, equity marked at the reference price, fill with adverse
    slippage, so every notional change (resizes included) pays `SLIPPAGE_PCT`. Non-overlapping bets reproduce
    `single`.
- Metrics add `Avg Bet Size` = mean |size| over the closes at which a position is held (same definition in both
  modes) and `Turnover (x/yr)` = Σ |traded notional| / equity at the fill, annualized. The approval threshold in
  `Sizer.size` is compared in p's own dtype, exactly as `trade_signal` (legacy meta-probabilities are float32).
  CLI `--sizer`, `--size-step`, `--position-mode`; comparison `python -m sizing.compare` (one WFO, every
  sizer × mode → `trials.jsonl`, stage `U7`).

**Risk layer** (U8, `risk/`). `RISK_PROFILE` names a `risk.profiles.RiskProfile`: `none` (default) = the pre-U8
backtest above, bit-identical; `standard` = the defaults below. An active profile routes the backtest through
`risk.portfolio.simulate_portfolio` — one or many symbols on one **union timeline** and one equity curve, one position
per symbol (`POSITION_MODE="single"` required; an event arriving while its symbol holds a position is skipped, a bet
whose final size is 0 does not block). A symbol without a bar at a union timestamp neither fills nor re-marks there.
**Timing:** every decision filled at open[b] is taken at the close of union bar b−1 with information ≤ that close:
equity E, drawdown, session P&L, held positions marked at their symbol's last close, the event's width, the
half-spread. Per bar: gap exits at the open → gate flattening → drift trims → entries (notional f·E(close b−1),
shares = notional / fill) → intrabar barrier exits (bet order) → vertical exits at the close → marks.
1. Vol target: f = SIZE·m·min(1, σ_target / σ_hold), σ_hold = width / BARRIER_MULT = σ_bar,t·√VERTICAL_BARS; σ_target 0.5%.
2. Caps: |f| ≤ min(per-position 20%, per-symbol 25%) (one position per symbol); entries at one bar, ranked by |f| (then
   symbol), fill the free slots of max concurrent 10; one pro-rata factor k ∈ [0, 1] then keeps gross ≤ 100% (no
   leverage) and **each side's** gross (long total, short total) ≤ the net cap 100% given the held positions on decision
   marks — so |net| stays within the cap whichever positions exit (a cap on the net itself is broken as soon as the
   offsetting side exits). Held positions are not resized, except a
   **drift trim** at the next open when marks take a position past cap·(1 + 10%) (back to the cap), or gross / a side's
   gross past theirs (the positions that can trade scaled pro-rata back). A held symbol with no bar at b cannot trade and is
   trimmed at its next bar. Positions that gap through a barrier at open[b] are unknown at close[b−1] and still take
   their slot and exposure in the decision for that open. Invariant (tested): at every entry bar gross, the position cap and
   the side(s) receiving entries hold exactly on decision marks; after every open, positions whose symbol has a bar
   there are within the 10% drift band.
3. Drawdown throttle: × 1.0 / 0.5 / 0.0 for DD < 10% / [10%, 20%) / ≥ 20% (close-marked equity vs its running high;
   resets at a new high), applied after the position cap. The 0 tier is absorbing once flat (no position can make a
   new high): it is a kill switch.
4. Daily loss gate: session P&L = equity at close / equity at the previous session's last close − 1. At ≤ −2% every
   position is flattened at its symbol's next open and entries are blocked for the rest of that session (for 1Day,
   the next session: its open is the next decision).

**Costs:** every fill (entry, exit, trim, flattening) pays c = `SLIPPAGE_PCT` + half-spread adversely on the fill price;
`standard` uses the Corwin–Schultz (2012) half-spread (`risk/costs.py`), floored at 0.5 bp; `none` charges
`SLIPPAGE_PCT` only (U7). **Refinement:** CS grows with the bar length (2016–2025 SPY mean half-spread 14 bp from 1Day
bars, 4.9 bp from 1Hour, 1.5 bp from 5Min, vs ≈ 0.1 bp quoted), so it is estimated from **5Min** bars for every
traded timeframe: per session the mean pair estimate (pairs straddling sessions excluded, CS overnight adjustment,
negative estimates 0), and a fill uses the trailing 21-session mean of the sessions **before** its own (causal). No
quoted-spread fallback (no quote data cached); a fill without an estimate raises. Optional short borrow (annual bps,
0 default) is charged at exit on the entry notional for the bars held. There is no internal crossing between symbols.

**Metrics** (all over the OOS span): the U7 table plus Sortino (RMS of negative per-bar returns), max drawdown duration
(longest run of closes below the running peak, trading days), PSR(0) of the per-bar returns, profit factor (Σ winning /
Σ losing trade P&L; cash P&L under the risk layer), tail ratio (|q95 / q5| of the non-zero per-bar returns, NaN below
20). Portfolio runner `python -m risk.run` (one WFO per symbol, every profile, equal-weight buy-and-hold benchmark →
`trials.jsonl`, stage `U8`).

---

## §8 Causality test (all units)

For a function f(df) → frame aligned to df, and cut points c (≥ 3, random, seeded):
`f(df)[:c+1] == f(df')[:c+1]` (atol 1e-12, NaN-equal) where `df'` = df with rows > c replaced by an
independent random walk and also a variant truncated at c. Applies to features, event sampling, widths,
primary `side`, sizers (given fixed probabilities) and the WFO signal frame (for rows before the first
test window touching c).

---

## §9 Experiments, ledger and holdout (U10, U11)

**Holdout.** `HOLDOUT_START = 2025-10-01` (last ~12 months to data end). Development window = 2016-01-01 → 2025-09-30.
The loader enforces it: `load_bars(..., allow_holdout=False)` raises if `end > HOLDOUT_START` (U1). U2/U10 must
additionally stop labels from resolving inside the holdout (last event ≤ HOLDOUT_START minus the max vertical
barrier). Only `experiments run --final` sets it True, and the ledger records that event.

**Holdout in the runner** (U10, `experiments/runner.py`). A cell whose `end` is after HOLDOUT_START is refused
before any data is loaded unless `run --final`; a final run needs every cell in stage E **and** ending after
HOLDOUT_START (a batch reading no holdout data must not use up the one access), appends a `holdout_access` event
(batch id = hash of its cells and code) to the ledger and to `data/cache/holdout_access.jsonl` (outside any ledger, so
`--ledger elsewhere` still sees it) before loading data, passes `allow_holdout` / `ALLOW_HOLDOUT`, and is refused if
either already holds a holdout access of a different batch (resuming the same batch is allowed). The check and the
event are under the ledger's run lock.
Labels cannot resolve inside the holdout in a development cell: its bars end before HOLDOUT_START and
`RunConfig.holdout_guard` raises on any bar on or after it, so a barrier still open at the data end is cut there.

**Cell spec** (YAML, `experiments/spec.py`): `stage`, `name`, `defaults`, `grid` (cartesian; dotted keys such as
`primary.name`, `model.meta`) and optional explicit `cells` (each merged over the defaults, then crossed with the
grid). Cell fields: `symbols` (a string = one symbol; a list = one portfolio cell), `timeframe, start, end`
([start, end) NY dates), `feature_groups` ("default" or a list), `primary{name, params}`, `model{meta, primary}`,
`meta_train`, `sizer`, `risk_profile`, `pwfo{is_grid, oos_grid, expanding}` (null = one expanding WFO; one symbol
only), `seed`, `overrides` (any other RunConfig field). Cells are normalized (defaults filled, symbols upper-cased and
sorted, dates ISO) and their RunConfig built at load time, so an invalid spec fails before anything runs; identical
cells are run once. Stages, in counting order: `U6, U7, U8, U9, U10, A, B, C, D, E`.

**Cell hash** = sha256(canonical cell, every resolved RunConfig field, code hash, data hash)[:16]. Code hash = the
source of every `*.py` in data, fastfracdiff, features, primaries, models, sizing, risk, validation, wfo, utils and experiments
(except `experiments/report.py`, line endings normalized to LF so a CRLF checkout hashes the same) plus the Python and
numeric-library versions and the OS and CPU architecture (`platform.system()`, `platform.machine()`: equal library
versions on arm64 macOS and x86-64 Windows need not give bit-identical floats, so a result is reused only on the
platform that computed it); data hash = each symbol's bars, the 5Min
spread bars of a spread-charging risk profile and, for a PWFO cell, the exchange sessions **between its first and last data session** (the only ones
`unit_bounds` reads; the cached calendar runs a year past today and is refreshed monthly). The stage is not hashed.
Artifacts: `results/experiments/cells/<hash>/` (spec, log, daily returns, signals or PWFO outputs, result, traceback).
Per-symbol WFO signals are also cached (`results/experiments/signals/<key>.pkl`, key = the same inputs minus the
backtest-only fields RISK_PROFILE, POSITION_MODE, SIZE_STEP, INIT_CASH, SIZE and the PWFO / selection fields, which
`run_wfo` never reads — tested), so a risk or position-mode ablation refits nothing.
**Resume / cache:** a cell whose (hash, stage) is in the ledger is skipped (an `error` row too, unless
`--retry-errors`); a hash recorded in another stage is re-recorded for this stage without fitting (`cache_hit`).
**Execution:** one run per ledger (an exclusive run lock; a second run raises). Data is loaded and hashed in the
parent; the distinct per-symbol WFOs are fitted first, once each; then cells run in joblib processes with BLAS pinned
to one thread, and the parent appends each row when its cell finishes (one write + fsync under a lock; rows are
validated: hash, known stage, known status), so a killed run loses only in-flight cells. A truncated last line is
skipped (reported) by the reader and isolated by the next append. A worker that dies (OOM kill, segfault) breaks the
pool: the cells not yet recorded are re-run one per fresh process, and one whose process dies again is an `error` row
("WorkerDied"). With `--jobs 1` cells run in-process (no crash isolation).
**Status:** `ok`; `no_fit` (`NoFitError`: no window could be fit — counted, as in §6); `error` (any exception,
including a data-load failure: full traceback under the cell's `traceback.txt`, path in `error_path`; never dropped).

**Ledger row** (`results/ledger.jsonl`, append-only): `cell_hash, stage, status, kind (wfo | portfolio | pwfo |
legacy), label, git_sha, code_hash, data_hash, run_id, final, spec_json, started_at, runtime_s, n_trials,
n_oos_events, n_trades, meta_auc, meta_logloss, brier, ret_ann, vol_ann, sharpe, sortino, calmar, max_dd, turnover,
sr_skew, sr_kurt, n_obs, psr, wfe, n_oos_windows, error, error_path` (+ `sharpe_primary` for one symbol;
`per_symbol` for a portfolio; `pbo, pwfo_dsr, picks, combos, n_decisions` for PWFO; `cache_hit, cache_from`).
*Refinement — return statistics are daily:* `ret_ann, vol_ann, sharpe, sortino, sr_skew, sr_kurt, psr, n_obs` come
from the daily close-to-close returns of the meta-filtered equity (the PWFO's live stream for a PWFO cell), so they are
comparable across timeframes and the DSR's per-period unit is one day; `max_dd` is the bar-level drawdown, `calmar` =
ret_ann / |max_dd|; meta AUC / log-loss / Brier from `signal_diagnostics` (the mean AUC over symbols for a portfolio).
*Refinement — trial count:* `n_trials` for the DSR of a stage = Σ `n_trials` over the **distinct** cell hashes with
status `ok` or `no_fit` in that stage or an earlier one (a WFO / portfolio cell = 1, a PWFO cell = its grid size).
A cache hit shares its hash and is counted once; a configuration re-run under new code is a new hash and counts again
(over-counting is the safe side); errors are not counted (listed in the report). V[SR_n] = variance (ddof 1) of the
per-period Sharpes (annualized / √252) of those trials, a PWFO cell contributing its combos' OOS Sharpes.
*Legacy trials* (U6–U9 `trials.jsonl`, `experiments import-legacy`, every `results/**/trials.jsonl` incl.
`pre_review/`): one `kind="legacy"` row per line, hash = "legacy-" + hash of the line (re-import is a no-op),
`n_trials` = 1 except the U9 nested-PWFO row (0: a selection among the counted combos); they count in N and V but get
no DSR in the report (their moments are per bar). A malformed (partial) source line is reported and skipped; a row
with an unknown stage or status is refused.
**Report** (`experiments report` → `results/experiments/report/report.{md,html}`, `leaderboard.csv`): stage funnel
(cells, ok / no_fit / error, cache hits, trials in stage, cumulative N), PBO per stage (CSCV over the daily returns of
the stage's runner cells on their common days, identical streams — one configuration under two code versions — counted
once), leaderboard by DSR (N and V through the row's stage) plus `dsr_all` (N and V over every ledger trial: the
deflation a pick made from the whole table faces), errors, holdout accesses.

---

## §10 Dependencies

Add: `alpaca-py`, `python-dotenv`, `lightgbm` (added U6), `joblib` (added U10), `scipy` (explicit). `pyyaml` becomes used (U10).
