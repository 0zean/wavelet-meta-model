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
*(Amended by §19, U13: costs are chosen by `RunConfig.COST_MODEL`, not by the profile; `standard` + `COST_MODEL="cs"`
is this paragraph.)*

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

---

## §11 Amendments for the hypothesis-family program (PLAN2, 2026-10-08)

Everything in §1–§10 stands unless amended here. Section references below are to this file.

### §11.1 Windows, holdout, stages

| item | was (U1–U11) | now |
|---|---|---|
| development window | 2016-01-04 → 2025-09-30 | unchanged (selection and headline tests) |
| holdout | 2025-10-01 → 2026-09-27, evaluated once (done, batch `33b8b08e02e9413c`) | that window is the **quasi-holdout**: reported per family as a robustness slice with a contamination note, never a gate |
| `HOLDOUT_START` | 2025-10-01 | **2026-10-01**; research code may not read bars on or after it; only `live/` (§18) reads past it, with an audit record |
| ledger stages | U6 … U10, A … E | + `F` (family dev test), `G` (overlay), `H` (forward test) |
| trial unit | a cell (a PWFO cell = its grid) | a family **variant** = 1 trial whatever its instrument count (the test is pooled); an overlay configuration = 1 trial |
| trial budget | none | `TRIAL_BUDGET` per family spec (default 12 + 2 overlay), program cap 8 families / 112 trials, enforced by the runner |
| verdict | per cell: Holm over 14 holdout streams; DSR over 1,402 | per family: Holm over families on the pooled headline test + floors + coherence (§17); DSR reported with N = program trials |

### §11.2 Engine defaults (U12)

| field | default | notes |
|---|---|---|
| `ZOO_FIXED_PARAMS` | `{rf_ldp_fast: {max_features: 1}, logit_l2: {C: 0.1}, xgb: {max_depth: 2}, lightgbm: {num_leaves: 7}}` | a model with an entry skips its purged-CV grid (one fit per window); `{}` restores the grid |
| `CALIBRATION` | `"rolling"` | `rolling`: Platt map fit on the previous windows' OOS (raw p, outcome) pairs inside the current rolling IS span; first window uses `crossfit`; `crossfit` = §4 behaviour; `none` = raw p |
| `OOF_META` | `"reuse"` | `oof_meta_prob` returns `cal.predict(oof_raw_)` of the fitted `_Tuned`; `"refit"` = §7 behaviour |
| `PWFO_COMBINE` | `"average"` | the PWFO stream = equal-weight mean of the run combos' daily returns on their common span; no burn-in, no selection; PBO across combos still reported; `"nested"` = §6 behaviour |
| `PWFO_IS_GRID` / `PWFO_OOS_GRID` | intraday `(504, 756)` / `(21,)`; 1Day `(1260, 1512)` / `(21,)` | U11 Stage C: cadence made no difference |
| `META_MODEL` for screens | `logit_l2`, `rf_ldp_fast` | `catboost` stays registered, never a default |
| parallelism | flat loky pool over (cell, combo) tasks | no nested pools (U11 Stage E crashes) |
| `COST_MODEL` | `"quotes"` (§19) | `"cs"` and `"slippage"` kept for regression |
| `VOL_PROFILE` | `"none"` globally; `"tod"` for every intraday family (§14) | |

Invariant: with every new switch at its old value (`ZOO_FIXED_PARAMS={}`, `CALIBRATION=crossfit`,
`OOF_META=refit`, `PWFO_COMBINE=nested`, `COST_MODEL=slippage`, `VOL_PROFILE=none`, `EVENT_SAMPLER=cusum`,
`EXIT_MODEL=triple_barrier`) the U11 Stage C `daily_returns.csv` of the parity cells
(`tests/fixtures/u12_parity.json`) and the legacy hash `402ef202…` reproduce exactly.

Implementation (U12, as built):
- `ZOO_FIXED_PARAMS[name]` must set exactly the model's grid keys. With `CALIBRATION=crossfit` a fixed model still
  runs its purged CV once (the OOF predictions the cross-fitted calibrator needs: k fits + 1 refit); with
  `rolling` / `none` and one grid point it is one fit and holds no OOF predictions (`oof_raw_ = None`). Applies to
  the primary role too (a zoo `PRIMARY_MODEL`), whose calibration is always cross-fitted.
- `rolling` (`wfo_engine.CalHistory`, one per combo / per WFO run, walked in window order): after a window predicts
  its OOS events, their raw meta P(y=1) and side-aware meta-labels are stored. A later fit on `[fit_start, val_end)`
  may use the pairs whose event bar is ≥ `fit_start` and whose barrier exit is before `val_end − val embargo` (the
  `purged` rule of the fit's own events), weighted by their average uniqueness. With ≥ `MIN_VAL_EVENTS` such pairs of
  both classes the meta-model is fit uncalibrated and a weighted Platt map on the pairs is its calibrator; otherwise
  (always the first window) it cross-fits. A window without a meta-model adds no pairs. `legacy` meta-models are
  never calibrated (the switch does not apply).
- `OOF_META=reuse` applies to zoo meta-models (`legacy` refits). When a run's sizers (`cfg.SIZER` or extra ones)
  include a train-fit sizer (`ecdf`, `kelly_capped`), each window's meta-model keeps its purged-CV OOF predictions
  (`need_oof`: k fits + 1 refit even with one grid point and rolling calibration); otherwise it is one fit. Under
  `refit` + `rolling` the per-split refits' raw predictions go through the window's calibrator.
- Each fitted window records its meta-model's calibration (`rolling`, `sigmoid` / `isotonic` = cross-fit, `none`,
  `legacy`): PWFO window table column `calibration`, stats / ledger `calibration_windows`; WFO signals attr and ledger
  `calibration_folds`. At 1Day with `MIN_VAL_EVENTS = 100` roughly a third of windows cross-fit (too few resolved
  pairs); intraday, the first 2–4 windows of each combo.
- `average`: the common span is every day on which all run combos (those with ≥ 1 fitted window) have an OOS return;
  `n_trials` / DSR still count every grid combo. The ledger row's `n_oos_windows` is the sum over combos (nested: the
  default combo's). `PWFO_DEFAULT` (only read by `nested`) defaults to the first IS length × OOS 21.
- Flat pool (`--jobs ≥ 2`): a PWFO cell's `runtime_s` is its combos' summed task seconds plus assembly (its serial
  cost); a failed combo makes the cell an `error` row with every combo traceback; a dead worker re-runs the unfinished
  cells one per process (serially). `--jobs 1` runs each cell in-process.
- `cusum_events` / `cusum_state` are numba kernels with the loops' float operations in the same order (identical
  output, tested against the loops); numba is in the code hash's library list.

---

## §12 Exogenous data (U13)

`data/exo.py: load_series(source, name, start, end, *, allow_holdout=False) -> DataFrame` with columns
`value` (float) and `available_at` (tz-aware UTC instant from which the value may be used; the
point-in-time rule), indexed by the observation date (NY, tz-naive midnight). Cache
`data/cache/exo/{source}/{name}.npz` + `.json` sidecar (`schema_version`, `source`, `name`, `columns`,
`n_rows`, `coverage_start/end`, `fetched_at`, `url`, `sha256`), atomic writes, hash verified on load,
top-ups with a 30-day overlap (re-fetch all on any revision). `end > HOLDOUT_START` raises unless
`allow_holdout`.

| source | names | `available_at` | notes |
|---|---|---|---|
| `cboe` | VIX, VIX9D, VIX3M, VIX6M (daily OHLC) | session date 16:20 ET | `https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{NAME}_History.csv` for all four (the `cdn.cboe.com` URL answers 307 to it); `value` = close, plus `open`/`high`/`low` |
| `cboe` | VX1, VX2 (continuous front/second VIX futures settlements), `VX1/VIX`, `VX2/VX1` | settlement date 16:20 ET | built from Cboe CFE per-expiry files; roll on the expiry date; the expiry row's settle is the final price |
| `fred` | VIXCLS, DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2, DTWEXBGS | VIXCLS: next federal business day 09:00 ET; DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2: next federal business day 16:30 ET (H.15 daily update 16:15 ET); DTWEXBGS: the Monday after the observation's week 16:30 ET, next business day if a holiday (H.10 is weekly) | VIXCLS is a cross-check of the CBOE file only; BAMLH0A0HYM2 starts 2023-10-09 on FRED (ICE licence: last 3 years) |
| `calendar` | FOMC statement days (14:00 ET), CPI days, NFP days (08:30 ET), OPEX (third Friday), month-end, turn-of-month (−1 … +3), pre-holiday | known in advance | checked-in CSV `data/calendar/events.csv` with a provenance column; 2016 → 2027 |
| `earnings` | per cached stock: date, `timing ∈ {bmo, amc}` | known in advance once announced; stored with the announcement date | Alpaca corporate actions / news first; checked-in CSV where missing |
| `alpaca_options` | daily SPY/QQQ chain snapshot: strike, expiry, bid/ask, IV, Greeks | snapshot time | 2024-02 → ; free indicative feed; open interest only if the schema has it (U13 confirms) |
| `quotes` | per symbol: median half-spread (bp) by time-of-day bin × year; open and close auction proxies | — | one trading week per quarter of NBBO at 5-minute marks, 2016 → 2026 |

**Invariants.** No value enters a feature on a bar stamped before its `available_at` (enforced in the
feature builder, tested with a planted same-day value). Daily exo values align to intraday bars by the
last `available_at` ≤ bar stamp. Revisions overwrite (no vintages kept); the sidecar records the fetch time.

Implementation (U13, as built):
- Fetches always download the whole published history (the files are small) and replace the cache; changed or
  removed observations on dates already cached are counted in a log line (this replaces the 30-day overlap).
  `coverage_end` = the NY date of the fetch; a request ending after it re-fetches. Index unit ns, `available_at` UTC.
- VX1 / VX2 / ratios are named `VX1`, `VX2`, `VX1_VIX`, `VX2_VX1` (file-safe). CFE list
  `https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/`, files
  `https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{expiry}.csv`; monthly contracts only
  (weeklies ignored), `Settle` column, rows with settle ≤ 0 dropped; VX1 on date d = earliest expiry ≥ d (the expiry
  row is its final settlement), VX2 the next; only dates on or after the first loaded expiry (2015-01-21) are kept,
  since before it the true front month was not loaded (U13 review); expired contract files are cached raw under
  `cboe/vx_contracts/`.
- CBOE has VIX (not VIX3M etc.) values on exchange holidays from 2022-05-30 on (Cboe global trading hours), and so does
  VIXCLS; the as-of alignment puts them on the next session. A cross-series ratio (VIX/VIX3M) must be built on common
  observation dates before alignment, or it pairs a holiday VIX with the previous session's VIX3M (U15).
- Live smoke (`scripts/u13_smoke.py exo`, `results/u13/exo_smoke.{json,png}`): CBOE VIX vs FRED VIXCLS max |Δ| =
  0.00 on 2,734 common days; VX1/VIX median 1.043 over 2016-01-04 → 2025-09-30 (> 1 on 76 % of days), VX2/VX1
  median 1.068. Coverage 2016-01-04 → 2026-09-30 for every series except BAMLH0A0HYM2 (from 2023-10-09): HY OAS
  cannot serve the development window; the credit-spread state needs another source (e.g. HYG vs IEF, U15).
- `calendar` / `earnings` (`data/events.py`, rebuilt by `uv run python -m data.events`; not cached in `exo/`):
  `data/calendar/events.csv` (kind, date, event_time, value, known_from, scheduled, source) 2016 → 2027 — FOMC from
  the Fed's historical pages (2016–2020) and `fomccalendars.htm` (2021–2027), 8 scheduled per year, 7 in 2020 (March
  17–18 cancelled) plus the unscheduled 2020-03-03 10:00 and 2020-03-15 (Sunday) 17:00 statements (the 2019-10-04
  meeting issued none); CPI and NFP from BLS's release archives (`/bls/news-release/{cpi,empsit}.htm`) and
  schedules (`/schedule/news_release/…`; bls.gov refuses python-requests, so the builder uses curl), 131 each to
  2026-12 (October 2025 CPI cancelled, October/November 2025 NFP merged); BLS releases on Good Friday (5 times since
  2016), so a release day is not always a session. OPEX = third Friday or the session before; TOM = −1, +1 … +3;
  PRE_HOLIDAY = a session followed by a weekday closure. `known_from` (→ `available_at` at its NY midnight): 1 January
  of the event year for scheduled kinds; the release date for CPI/NFP moved by the 2025 shutdown (2025-10-01 …
  2026-02-28, including the January 2026 CPI / NFP moved by the early-February 2026 funding lapse); the Monday of the
  closure week for PRE_HOLIDAY before the unscheduled closures 2018-12-05 and
  2025-01-09; the statement instant for unscheduled FOMC. `event_at` = the release instant (NaT for day flags).
- `earnings`: `data/calendar/earnings.csv` from SEC EDGAR (Form 8-K with Item 2.02), 395 filings 2016-01 → 2026-08
  for the 9 stocks. The acceptance time is the "Accepted" field (ET) of each filing's index page: the submissions
  JSON's `acceptanceDateTime` carries a second NY offset for AAPL, AMZN, META, JPM and UNH (AAPL's 16:30:26 ET
  filing of 2024-08-01 appears as 2024-08-02T00:30:26Z; found by the U13 review). Date = the acceptance's NY date;
  timing: < 09:30 ET `bmo`, ≥ 16:00 `amc`, otherwise `dmh` (JPM files about 06:46, UNH about 06:01 ET). Announcement
  dates are not in EDGAR: a filing with the company's usual timing (amc: AAPL MSFT NVDA AMZN GOOGL META; bmo: JPM UNH
  XOM) is available from its day's midnight, any other (NVDA's two pre-open preliminary results, XOM's mid-quarter
  "earnings considerations", one late AAPL filing) from its acceptance time. Series columns: `value` = 1, `timing`
  (−1 bmo, 0 dmh, +1 amc), `event_at` = acceptance time (an upper bound on the release time).
- `alpaca_options` (checked 2026-10-09, free plan): the snapshots endpoint serves the current chain only (IV and
  Greeks on contracts with a usable quote, indicative feed); open interest is on the trading API's option contracts
  (`open_interest` as of `open_interest_date`, the previous session), also current only; historical option daily
  bars exist from 2024-02. No historical chain or open interest can be built: `data/options.py` collects the SPY/QQQ
  chain forward (`uv run python -m data.options`), and the U15 gamma proxy uses the VIX/skew fallback for the
  development window.
- `quotes` (`data/quotes.py`; not a `load_series` source): sessions 6–10 of Feb/May/Aug/Nov 2016Q1 → 2026Q3 (215
  sessions), 30 symbols, marks open + 5 min … close − 5 min plus `open_auction` (open + 1 min) and `close_auction`
  (close − 1 min); prevailing quote = last in [m − 5 s, m), then 60 s, then 15 min (windows clipped at the open).
  Table `data/costs/quotes_half_spread.csv` (+ `.json`): median half-spread per symbol × year × {26 15-minute bins,
  open_auction, close_auction, day}, 9,570 rows from 509,550 samples (82 missing: Alpaca's 2019-08-12 SIP hole; 49
  crossed / one-sided; 3 pre-open). The raw samples are cached under `data/cache/exo/quotes/` (git-ignored; about 3 h
  to re-fetch at 200 requests / min). SPY 10:00–15:30 median 0.155 bp over 2016–2025 vs Corwin–Schultz 1.53 bp
  (`results/u13/quotes_smoke.json`); the 0.25 bp floor binds for SPY in every year and for QQQ (2020–2026 except 2022) and IWM (2021, 2024–2026). Stocks' open
  proxy (the 09:31 quote) is wide (UNH 8.9 bp, XOM 3.9 bp median): conservative for a single-price auction fill.
  Sampling bias (U13 review): the base week holds almost no FOMC / NFP / OPEX / month-end session and no stress
  session. A separate sample (`fetch --plan events`: one FOMC, NFP, OPEX, month-end session per year + 7 stress
  sessions, 50 sessions, `data/cache/exo/quotes/event_samples/`, not in the table) measures the gap in each kind's
  trade bins (event half-spread / table value, median across symbols): FOMC 14:00–16:00 1.03 (max UNH 1.48), NFP
  open 1.01, OPEX close 1.01, month-end close 0.99, stress sessions (2018-02-06, 2020-03-16 … 20, 2024-08-05, all
  bins) 2.10 (IWM 2.9, AAPL 3.5, TLT 4.0, USO 4.7). Scheduled events cost about what the table says; crisis sessions
  cost 2–4× more, which the table does not model (a volatility-scaled cost is a family-spec decision, U17).
- Universe: `DEFAULT_UNIVERSE` += IEF, LQD, HYG, DBC, USO, UUP, EFA, EEM, VNQ, SLV, SVXY (5Min from 2016-01-04; all
  listed before 2016). SVXY changed from −1× to −0.5× the VIX short-term futures index on 2018-02-28 after the
  2018-02-05 loss: its history is two products, and it is the survivor of that event (XIV was terminated);
  `data.fetch.REGIME_BREAKS` records the date. USO's 1:8 reverse split (2020-04-29) is in the adjusted bars.
  Quality (`results/data_quality_u13_etfs.csv`, 2016-01-04 → 2026-09-30): 2,701 sessions each, none missing, no OHLC
  violation; UUP misses 5.0 % of 5Min bars (no-trade bars), DBC 0.7 %, the rest ≤ 0.13 %. Largest moves are real:
  SVXY −84 % on 2018-02-06, USO 2020-03/04, SLV −29 % on 2026-01-30.

---

## §13 Event samplers and exit models (U14)

`RunConfig.EVENT_SAMPLER ∈ {"cusum", "dc", "schedule"}`, `EVENT_PARAMS: dict`;
`RunConfig.EXIT_MODEL ∈ {"triple_barrier", "time", "hysteresis"}`, `EXIT_PARAMS: dict`. The exit models
and `schedule` never change a feature value, so they are excluded from the feature-cache key; `dc` and
`cusum` with `VOL_PROFILE` change `session.dc_overshoot`, so the sampler stays in the key.

**Samplers** (`features/events.py`; all return the event frame of `sample_events`: index = event bar,
`width`, plus `entry_pos`, the bar whose open is the fill):
- `cusum`: §3/§4 behaviour; threshold `CUSUM_MULT · σ_t` with σ_t from §14.
- `dc`: directional change (Guillaume et al. 1997; the ISOM paper's event definition): a run is upward
  until log price falls `δ` below its running maximum, then an event is emitted at the confirmation bar
  and a downward run starts; `δ = DC_MULT · σ_session` (`EVENT_PARAMS.dc_mult`, default 2). `overshoot`
  = log move from the confirmation bar to the next reversal, exposed (lagged) by the `session` group.
- `schedule`: `EVENT_PARAMS = {entry_times: ["15:30"], days: "all" | "fomc" | "cpi_nfp" | "tom" |
  "non_macro" | "earnings", gate: "<expr>"}`. The event bar is the last bar whose close precedes the
  entry time; `entry_pos` = the first bar stamped ≥ entry time (its open is the fill). `days` is a
  calendar predicate from §12; `gate` is a boolean expression over `session` columns evaluated at the
  event bar (e.g. `abs(open_to_now) > 0.5 * sigma_day`). Early closes: times past the close map to the
  last bar; `"close"` always means the session's last bar.

**Exit models** (`features/exits.py`; all return the `barrier_exits` frame):
- `triple_barrier`: §3 behaviour.
- `time`: `EXIT_PARAMS = {exit_time: "close" | "HH:MM" | null, hold_bars: int | null}`; `"close"` =
  fill at the session's last bar close (modelled MOC); `"HH:MM"` = fill at that bar's open; `hold_bars`
  = close of the k-th held bar. Overnight (`overnight` primary): entry MOC, exit next session MOO
  (`exit_time: "open"`).
- `hysteresis`: `EXIT_PARAMS = {beta: float, max_bars: int}` on the primary's continuous `signal`:
  a long exits at the first bar whose signal ≤ −β (short: ≥ +β) or at `max_bars`; fill at that bar's
  close. The entry threshold α is the primary's own gate.

**Fills.** Auction fills (`close`/`open`) pay the §19 half-spread only; all other fills pay half-spread +
`SLIPPAGE_PCT`. The same-bar tie rules of §3 apply to `triple_barrier` only.

**Invariants (tested).** The decision bar strictly precedes the fill bar; no exit before entry; labels
for scheduled events are the fill-to-fill return; `average_uniqueness` and the purge use the same spans.

---

## §14 Intraday volatility structure (U14; the ISOM idea done properly)

References: Andersen & Bollerslev (1997) intraday periodicity; Dacorogna et al. (2001) ϑ-time; the
Kablan (2009) ISOM/IAOM counts are the event-rate view of the same seasonality.

- `features/vol_profile.VolProfile` (per-fold state, fit on train bars before the embargo like
  fracdiff): slots b = session-relative bar index (13 at 30Min, 26 at 15Min, 78 at 5Min, 7 at 1Hour
  with the stub as its own slot). `s(b) = median_sessions |r_b| / median_b median_sessions |r_b|`,
  smoothed by a 3-slot moving median, floored at 0.25. `transform(r) = r / s(b)`.
- `bar_volatility(close, span, profile)`: σ_t = EWM σ(span) of the deseasonalized returns up to t,
  multiplied by `s(b_t)`. With `VOL_PROFILE="none"` the profile is identically 1 (bit-identical to §3).
- `σ_day` (session σ) = σ_t at the first slot × √(bars per session); `σ_overnight` = EWM σ of
  close-to-open log returns (span `VOL_SPAN` sessions).
- ISOM diagnostics: `isom_counts(events, n_slots)` and `iaom = isom / n_sessions` per fold in
  `wfo_run.json`; the `session` group exposes `iaom[b_t]` (train-window value, causal) as `activity`.
- Invariant: barrier widths and CUSUM thresholds in slot b scale with `s(b)`; the event share in the
  first hour drops relative to `none` (reported, not gated).

Implementation (U14, as built; §13 and §14):
- Modules: `features/vol_profile.py` (VolProfile, `bar_volatility` moved here from `triple_barrier_labels`, slots,
  `hold_scale`, `isom_counts`), `features/events.py` (samplers; `sample_events` moved here), `features/exits.py`
  (exit models, the `exit_frame` dispatcher, `exit_phase`), `features/session.py` (the session frame). Every caller
  of `barrier_exits` (labels, meta-labels and side returns, both backtests, the portfolio simulator, `meta_outcomes`,
  the primary diagnostics, `models/compare`) now calls `exit_frame(df, events, width, cfg, side)`, which is
  `barrier_exits` itself under `triple_barrier`.
- Execution model kept: decision at the close of event bar t, fill at the open of the next bar in the data
  (`entry_pos = t + 1`, which `purged` relies on). The schedule sampler picks t as the bar before its entry bar, so
  the frame has no separate `entry_pos` column. Bar timestamps are treated as the calendar (known in advance), the
  convention of the §3 last-bar filter and `barrier_exits`' session truncation.
- Switches: `VOL_PROFILE` ("none"), `EVENT_SAMPLER` ("cusum"), `EVENT_PARAMS` ({}), `EXIT_MODEL` ("triple_barrier"),
  `EXIT_PARAMS` ({}); parameters are validated in `RunConfig` (`events.event_params`, `exits.exit_params`). The
  five are `_NON_FEATURE_FIELDS` of the static feature cache: no static group reads them (the `session` group is
  per fold and so uncached), and existing cache keys are unchanged. They are not `BACKTEST_ONLY` (they change the
  WFO signals). CLI: `--vol-profile --event-sampler --event-params --exit-model --exit-params`.
- `dc`: δ_t = `dc_mult` · σ_t, the profile-aware bar σ (the same scale as the CUSUM threshold). "σ_session" above is
  read this way: δ = 2 · σ_day would give about 0.1 events per session. SPY 5Min 2016–2025: 5.6 events / session
  (7.2 with `tod`). Before the first confirmation the run direction is unknown and the first δ move either way
  confirms it. The overshoot is exposed in δ units of its confirmation bar, signed by the run.
- `schedule`: entry times must be bar opens of the timeframe between 09:30 and 16:00 (1Day: `open` only). T = 09:30
  decides at the previous session's last bar; a later T needs the bar before the entry bar in the same session (a
  hole there gives no event that day); an early close (the session's last bar spans 13:00, the `auction_flags` rule,
  so a regular session missing its last bar is not one) maps a T at or past the close to the session's last bar.
  A 09:30 entry needs `EXIT_MODEL="time"` (or HOLD_OVERNIGHT): the other exits drop cross-session entries, and
  RunConfig refuses the combination rather than yield zero labels. `days`: the entry session must be a day of the kind (FOMC; CPI or NFP; TOM;
  non_macro = none of FOMC/CPI/NFP; earnings: the release date for bmo/dmh, the next session of the data for amc
  if it is at most 4 calendar days later, else no session) whose
  `available_at` is at or before the decision time (the event bar's close, 16:00 for 1Day). The unscheduled FOMC
  cut of 2020-03-03 10:00 is not an FOMC day for a 09:35 entry but is for a 15:30 entry; a Sunday FOMC date matches
  no session. `gate`: `DataFrame.eval` over `session_frame` at the event bar (columns of the module docstring
  incl. `sigma_day`, `sigma_bar`, `sigma_overnight`; `activity` is not available there); a NaN comparison is False.
- `time`: exactly one of `exit_time` / `hold_bars`. `"close"` fills at the entry session's last bar close and
  `"open"` at the next session's first bar open, each only if that bar is the auction bar (`auction_flags`): a
  session whose last bar (MOC) or next session whose first bar (MOO) is missing drops the event. `"HH:MM"` fills at
  the open of the first bar of the entry session stamped at or after it (a bar open after 09:30, before 16:00);
  when the session ends first (early close) it fills MOC; when the entry bar is not before it the event is dropped.
  Time exits keep cross-session entries (scheduled 09:30 entries); `triple_barrier` and `hysteresis` still drop
  them unless HOLD_OVERNIGHT.
- `hysteresis`: the bar-level signal is the rule primary's `score` (a primary without one is refused at
  `prepare`). Deviation: a trigger at bar k fills at the OPEN of k + 1, not the close of k, so the decision bar
  strictly precedes the fill (the §13 invariant); a trigger on the last held bar is moot (the position closes there
  at the close anyway). Labels without a side take the sign of the score at the event (>= 0 long, as `rule_frame`).
- Barrier names and fills: `vertical` = a close fill (vertical barrier, MOC, `hold_bars`, hysteresis max hold);
  `time` / `hysteresis` = an open fill. `exit_phase` books open fills at the bar's open (phase 0, cost kind `open`:
  the opening-auction proxy at 09:30, else slippage + the bin's half-spread) and close fills at the close (phase 2,
  `close`: the closing-auction proxy on the session's last bar). Old barrier names keep their phases.
  `width` stays the sampler's barrier width under every exit model: under `time` / `hysteresis` it is not the σ of
  the actual hold (an `exit_time: "close"` from 10:00 holds ~70 bars), so the risk layer's vol target
  (σ_hold = width / BARRIER_MULT) misstates σ_hold by about √(hold / VERTICAL_BARS). Deferred to U16, where each
  primary fixes its own hold and sizing.
- Profile: `r` = 1-bar close-to-close log return, so slot 0 carries the overnight gap; edge slots of the 3-slot
  moving median keep their own value. SPY 5Min 2016-01-04 → 2025-09-30 (`scripts/u14_isom.py`,
  `results/u14/isom_spy_5min.json`): s(0) = 8.7 (the gap bar), s(1) = 1.8, lunch ≈ 0.85, close ≈ 1.6–1.8; median
  |r| by 30-minute slot 6.5 bp at 09:30 vs 2.9 bp at 12:30 (2.2×; 1.8× without the gap bar).
- Barrier widths under a profile (deviation): `BARRIER_MULT · σ_base_t · √Σ s(b)²` over the slots the position
  holds (`hold_scale`: the VERTICAL_BARS slots after b_t, stopping at the 16:00 slot unless HOLD_OVERNIGHT; the next
  session's first slots for an event on the last bar), instead of `σ_t · √VERTICAL_BARS`. With slot 0 at 8.7 the
  literal rule would make every event on the 09:30 bar about 6× too wide for the hour it holds. A position entered
  at the opening print does not bear the gap: its first slot counts `s_open` (median |log(close / open)| of the
  first bar on the same normalisation; SPY 2.2), and only a wrapped overnight hold counts s(0). With a flat profile
  and a full hold it equals the §3 width; near the close it is narrower (fewer held slots). CUSUM / DC thresholds use
  σ_t of the event slot as specified. `VOL_PROFILE="none"` is the §3 code path, bit for bit.
- Per window (`wfo_engine.window_events`): with `tod` the profile is fit on [fit_start, train_end − embargo) and the
  events, widths and labels are re-sampled on the full series for that window (≈ 0.2 s on SPY 5Min); the latest
  window is cached by (fit_start, fit_end). `prepare` then samples nothing. Rolling calibration (`CalHistory`)
  purges each OOS pair on the label span of the window that scored it (`add(..., spans=labels)`), not on the later
  window's re-sampled labels (U14 review: those may lack the event, ~10 % of pairs, or resolve it earlier than the
  meta-label, a leak once a hold can exceed the embargo); without a profile the spans are the run's labels. Each fold's train-window ISOM
  (`isom`, `n_sessions`, and `s` under `tod`) goes to `signals.attrs["isom_folds"]` → `wfo_run.json` (intraday).
- Hashing: `experiments.runner.code_hash` also hashes `data/calendar/*.csv` (the `days` tables); the feature
  cache's code hash covers `features/*.py`, so this unit's files invalidate every cached static feature once.
  `models/compare` (one whole-window CPCV, no fold) refuses `VOL_PROFILE="tod"`.
- `session` group: per fold (the fold's profile for `dc_overshoot`, and `activity` = IAOM of DC events with the
  plain σ over the train window, so the raw seasonality shows; with the profile the rate flattens by design).
  Deviations: `first30` before 10:00 is the return so far (a NaN there would drop every event before 10:00 and
  every 1Hour event, since `fit_window` drops events with a NaN feature); `mins_to_close` assumes 16:00 (as the
  `intraday` group); `gap_sigma` divides by the σ of earlier gaps (EWM over VOL_SPAN sessions, min 20). It costs
  ≈ 0.7 s per window on SPY 5Min (not cached).
- ISOM (SPY 5Min, CUSUM_MULT 1.5, dc_mult 2, profile fit on the whole window): CUSUM 15.7 events / session, 21.8 % in
  the first hour; with `tod` 20.1 / session, 16.9 %. DC 5.6 / session, 23.4 %; with `tod` 7.2, 17.5 %. With `tod`
  the IAOM is flat from 10:00 to 15:00 and lowest in the last 30 minutes.

---

## §15 State feature groups (U15)

All groups prefixed as in §3; exo-backed groups declare `needs=("exo",)` and read `context["exo"]`.

| group | columns | alignment |
|---|---|---|
| `session` (intraday) | `gap_sigma` (overnight gap / σ_overnight), `prev_cc`, `prev_oc`, `open_to_now`, `first30` (NaN before 10:00), `mins_to_close`, `activity` (IAOM at slot), `rvol` (cumulative volume / its time-of-day profile), `dc_overshoot` (lagged) | bars only |
| `vol_state` | `vix`, `vix9d_vix`, `vix3m_vix`, `vx1_vix`, `vx2_vx1`, `rv21_vix`, `garch_sigma` (GARCH(1,1) fit on train, per fold), `ret_std_garch` | `available_at` |
| `calendar_events` | `to_fomc`, `since_fomc`, `to_cpi`, `to_nfp`, flags `fomc_day`, `cpi_nfp_day`, `opex_week`, `tom`, `pre_holiday`, `mins_since_release` (intraday, on release days) | known in advance |
| `cross_asset` | existing columns + `mkt_ret_lag{1,3,6}`, `sector_ret_lag{1,3,6}`, `rel_strength` (symbol − sector, h bars) | market/sector bars aligned by stamp (≤ t) |
| `rates_credit` | `d_dgs10`, `t10y2y`, `d_hy_oas`, `d_dollar` (daily changes) | `available_at` |
| `gamma_proxy` (optional, 2024-02 →) | `skew25` (put − call IV at 25Δ), `atm_iv`, `gex_sign` if open interest exists | snapshot time |

Runner: `experiments/runner.load_cell_data` supplies `context["market"]` (SPY at the cell's timeframe),
`context["sector"]` (mapping in `data/sectors.py`) and `context["exo"]` (the series the cell's groups
need, loaded through §12). The feature cache key already includes each context frame's data hash.

Implementation (U15, as built):
- Point-in-time access (`features/exo_align.py`): a group declaring `needs=("exo",)` lists its series in
  `FeatureGroup.exo` ("source/name") and receives an `Exo` view restricted to them, never the raw frames. Every
  accessor uses a row on a bar only when `available_at` ≤ the bar's stamp (its open; a naive index is NY time):
  `asof` (the visible row with the greatest `available_at`, ties → latest date), `ratio` (built on the two series'
  common observation dates, available when both are, so a 2022+ VIX holiday row is not paired with the previous
  session's VIX3M), `change` (over N observations, available from the cumulative maximum `available_at`), and for
  calendar kinds `next_date` / `last_date` / `on_day` / `known_dates`. 1Day bars are stamped at NY midnight, so a
  daily bar sees the previous session's VIX and the FRED value released the session before (conservative: the
  decision is at the close). Optional context keys (`FeatureGroup.optional`, e.g. "sector") are passed when present.
  Per-fold groups with `needs` get their context as `transform(df, state, cfg, context)`.
- `vol_state` is per fold (GARCH): exo columns as specified; `vix9d_vix` / `vix3m_vix` are VIX9D / VIX and VIX3M /
  VIX; `rv21_vix` = σ of the last 21 completed sessions' close-to-close log returns × √252 / (VIX / 100) (an intraday
  bar's own session is excluded; a 1Day bar includes itself). GARCH(1,1), Gaussian, zero mean, variance targeting
  (ω = v̄(1 − α − β), v̄ = mean z² on train), fit by L-BFGS-B on persistence ∈ [0.01, 0.998] and α share ∈
  [0.001, 0.999], numpy/scipy (no `arch`). Intraday it runs on bar returns deseasonalized by a VolProfile fit on the
  same train bars (Andersen–Bollerslev): 5Min WFO folds start with 42 sessions, too few for a daily GARCH.
  `garch_sigma` = √h(t+1) × √Σ s(b)² (the next bar's conditional σ in session units; 1Day: the next day's σ);
  `ret_std_garch` = z_t / √h_t. Fewer than 500 train returns or a failed fit raise RuntimeWarning: the fold is skipped
  (`WindowFit` status `feature_fit_failed`, formerly `fracdiff_failed`). SPY 5Min first fold (3,197 bars): α 0.044,
  β 0.951; 1Day first fold (1,007): α 0.193, β 0.743.
- `calendar_events` (static, `level_check=False`): `to_/since_{fomc,cpi,nfp}` (since_cpi, since_nfp added), flags
  as specified, `tom` = the TOM value (−1, +1 … +3; 0 otherwise) rather than a 0/1 flag, `mins_since_release`
  intraday only (minutes from the day's FOMC / CPI / NFP release to the bar's close, negative before a later release
  that day, RELEASE_CAP = 480 on days without one, so no NaN drops ordinary days). Sessions = weekdays minus the
  closures known at the bar (the weekday after each visible PRE_HOLIDAY row; no closure in 2016–2027 spans two
  weekdays). Scheduled rows are public from 1 January of their year (§12), so in late December (and after an
  early-December NFP) the next CPI / NFP / FOMC is not yet known: `to_*` = TO_CAP = 63 then (and is clipped at 63),
  not NaN. `to_*` reads `calendar/{FOMC,CPI,NFP}_SCHEDULE` (`data/events.schedule_series`): the held releases plus
  `ORIGINAL_SCHEDULE`, the originally published dates that were cancelled or moved (FOMC 2020-03-18; CPI 2025-10-15,
  11-13, 12-10, 2026-02-11; NFP 2025-10-03, 11-07, 12-05, 2026-02-06; from the Fed's 2020 calendar and Wayback
  snapshots of BLS's schedule pages), public from 1 January of their year and withdrawn (`withdrawn_at`, honoured by
  the Exo view) at their own scheduled instant. A cancellation is thus never visible before the release time it
  replaced; between its announcement and that time the feature is stale (the announcement dates are not recorded),
  never early. Held 2026-01-09 / 01-13 releases, conservatively public only on the day in the U13 table, are public
  from 2026-01-01 in the schedule series. The event-day flags, `since_*` and `mins_since_release` read the held
  releases only. A release on a non-session day (the Sunday 2020-03-15 statement, Good Friday CPI / NFP) flags no
  session; `since_*` = 1 on the next one.
- `rates_credit`: `d_dgs10` (1 observation), `t10y2y` (level), `d_credit` = change of FRED `BAA10Y` (Moody's Baa −
  10-year; daily from 1986, H.15 `available_at` rule; added to `data/exo.py`) instead of HY OAS, which FRED has only
  from 2023-10-09; `d_dollar` = log change over 5 DTWEXBGS observations (one H.10 week).
- `gamma_proxy`: not built. No historical chain or open interest exists (U13: Alpaca serves the current chain only;
  `data/options.py` collects forward from 2026-10), so the group would be NaN over the whole development window.
  Revisit once a year of forward snapshots exists (U19 / U20).
- `cross_asset`: + `mkt_ret_lag{1,3,6}` (the market's 1-bar return k bars ago); with "sector": `sector_ret_lag{1,3,6}`,
  `sector_resid_ret` (beta to the sector over BETA_WINDOW) and `rel_strength_h`. `data/sectors.py`: AAPL MSFT NVDA →
  XLK, JPM → XLF, XOM → XLE, UNH → XLV, AMZN GOOGL META → QQQ (no cached XLY / XLC covering 2016 →); ETFs, sector ETFs
  included, get the market only.
- Context loading (`features/context.py: load_context`): only the keys and series the cell's groups read
  (`registry.context_needs`), so cells without them keep their data hash. Exo rows from start − 120 days; calendar
  rows to end + 120 days, past HOLDOUT_START if need be (schedules carry no outcome; the Exo view still applies
  `known_from`; clipping gave the dev window's last weeks a false "nothing scheduled"). `cross_asset` on the market
  symbol itself (beta 1, residual 0) is refused. The runner puts the per-symbol context
  in `data["context"]`, in the data hash, the signals key, `run_wfo` and every PWFO combo; the CLI loads it for an
  Alpaca `--symbol`. Exo frames are hashed in full (`cache.context_hash`: values, `available_at`, `event_at`).
- Warm-ups (SPY 5Min 2016-01-04 → 2025-09-30, `results/u15/state_features.json`): `since_*` NaN until the table's
  first event (2016-01-07 NFP, 01-19 CPI, 01-26 FOMC); `rv21_vix` 22 sessions; `ret_std_garch` the first bar; every
  other new column is never NaN. Largest |corr with close|: `t10y2y` 0.39.

---

## §16 Mechanism primaries (U16)

All are `RulePrimary` subclasses: no fit, fixed parameters, frame = `PRIMARY_COLUMNS`; `magnitude` is the
rule's size hint in [0, 1] (the `rule_size` sizer passes it through; `fixed` ignores it). Each declares
`SAMPLER`, `EXIT`, `TIMEFRAMES`, `LONG_ONLY`.

| primary | sampler / exit | side | magnitude | params (defaults) |
|---|---|---|---|---|
| `vol_target` | schedule daily open / time (next rebalance) | +1 | min(1, σ*/σ̂) | `sigma_target` 0.15, `window` 21, `band` 0.10, `vol_source` rv\|ewm\|vix |
| `tsmom` | schedule weekly (Monday open) / time | sign(trailing return) or sign(MODWT slope) | min(1, 0.10/σ̂_asset) | `lookback` 252, `vol_window` 63, `estimator` ret\|modwt, `long_only` false, `gross_cap` 1.0 |
| `overnight` | schedule last bar / time `open` | +1 | 1 | `vix_max` null |
| `calendar_drift` | schedule by predicate / time | +1 | 1 | `windows` [fomc_pre, tom], `tlt_month_end` true |
| `intraday_momentum` | schedule `15:30` / time `close` | sign(predictor) | min(1, \|predictor\|/σ_day) | `entry` 15:30, `predictor` open_to_now\|first30, `threshold_sigma` 0.5 |
| `gap_fade` | schedule `09:35`, days non_macro / time `10:30` or hysteresis | −sign(gap) | min(1, \|gap_sigma\|/2) | `min_gap_sigma` 1.0, `exit` 10:30, `follow_on_news` false |
| `event_reaction` | schedule release + 15 min, days fomc\|cpi_nfp / time + 90 min | sign(post-release return) | min(1, \|react\|/σ_day) | `observe_min` 15, `hold_min` 90 |
| `weekly_reversal` | schedule weekly / time | −sign(last-week return), rank within set | 1/n | `set` sector_etfs |
| `vix_carry` (U21) | schedule daily / time | −1 on SVXY when `vx1_vix > 1.05 and vix < 25` else flat | 1 | hard rules in the family spec |

Diagnostics add per-slot precision and ISOM counts. Invariant: every primary passes §8 causality and a
synthetic sanity test; `rule_size` with magnitude ≡ 1 equals `fixed` bit for bit.

---

## §17 Family tests (U17)

### §17.1 Family spec (`families/<id>.yaml`)

```yaml
id: F5_intraday_momentum
mechanism: "<two sentences>"
registered: {sha: <git sha>, date: 2026-xx-xx}     # written by `families register`
instruments: [SPY, QQQ, IWM, DIA]        # or basket: {symbols: [...], weighting: equal_risk}
timeframe: 5Min
headline:                                 # the only tested variant
  primary: {name: intraday_momentum, params: {entry: "15:30", predictor: open_to_now, threshold_sigma: 0.5}}
  exit: {model: time, params: {exit_time: close}}
  cost_model: quotes
  risk_profile: standard
variants:                                 # reported only; count <= TRIAL_BUDGET - 1
  - {label: entry_1500, primary.params.entry: "15:00"}
  - {label: first30, primary.params.predictor: first30}
state_splits: [vix_tercile, macro_day, abs_move_tercile, gamma_sign]
response: [sharpe, mean_per_trade_bp, hit_rate, path_monotone]
sample_splits: [{label: sector_etfs, instruments: [XLE, XLF, XLK, XLV]}]
benchmark: buy_and_hold_ew
floors: {min_net_ret: 0.02, min_edge_to_cost: 3.0, max_dd: null}
test: {alpha: 0.05, block_days: 21, n_boot: 2000, coherence_share: 0.667}
overlay: {models: [logit_l2, rf_ldp_fast], sizers: [linear, rule_size],
          features: [wavelet_core, session, vol_state, calendar_events, cross_asset]}
TRIAL_BUDGET: 12
```

The runner refuses a family whose `registered.sha` is not an ancestor of `HEAD`, whose spec differs
from the committed file, or whose variant count exceeds the budget.

### §17.2 Pooled streams and the headline test

- Variant stream = equal-risk mean of the instruments' daily returns (weights ∝ 1/σ of each
  instrument's buy-and-hold over the dev window, fixed ex ante); a basket family's stream is its
  portfolio cell's.
- Headline statistics: annualized Sharpe with PSR(0) (§6); Sharpe difference vs benchmark by
  Ledoit & Wolf (2008): studentized circular block bootstrap, block 21 sessions, 2,000 resamples,
  two-sided p; alpha vs benchmark by OLS with Newey–West (lag 5) SE; max drawdown; net return.
- Floors: `net_ret ≥ min_net_ret` (overlays: 2 %/yr; long-only families: `0.8 × benchmark`);
  `gross_edge_per_turnover ≥ min_edge_to_cost × cost_per_turnover`, where gross edge = Σ gross trade
  P&L / Σ traded notional and cost = the §19 half-spread (+ slippage where it applies) averaged over the
  fills; `max_dd` if set.
- Coherence: ≥ `coherence_share` of the variants have the headline's sign on the Sharpe difference and
  the **median** variant clears the floors. The specification curve (every variant's statistic with its
  CI, sorted) is published; no variant replaces the headline.
- Splits (state / response / sample) and the quasi-holdout slice are reported with CIs, never tested.
- Shrinkage: per-instrument Sharpes with James–Stein shrinkage toward the family mean (reported).

### §17.3 Program verdict

Holm over the families' headline Sharpe-difference p-values at α = 0.05. A family **passes** iff its
Holm-adjusted p < 0.05, every floor holds and it is coherent. The DSR of each headline is reported with
N = counted trials of stages F and G, V from their per-period Sharpes.

### §17.4 Overlay test (stage G)

Overlay stream vs the family's headline stream: paired Ledoit–Wolf Sharpe difference (same block
bootstrap), floor `overlay net_ret ≥ headline net_ret`, Holm over the overlay tests, plus a procedure
bootstrap: the choice between the ≤ 2 overlay configurations is re-run on 100 block-bootstrapped label
sets and the adopted configuration's advantage must exceed the 95th percentile of the bootstrapped
advantages. Adopted overlays replace the headline in the registry; others are discarded.

---

## §18 Forward test (U20)

- `families/registry.yaml`: per adopted family — cell hash(es), code hash, freeze date, registered
  expectation (dev Sharpe and its 90 % CI, benchmark), verdict rule.
- `live/`: signal recomputation from cached bars plus an intraday top-up (Alpaca latest bars) under the
  frozen code hash (a mismatch refuses to trade); orders by `time_in_force` `opg` / `cls` for auction
  fills, limit-at-touch for intraday entries, bracket exits for `time` exits; reconciliation of modelled vs
  actual fills (slippage ledger); BOCPD monitor on the daily P&L (Adams & MacKay 2007; Student-t
  predictive; hazard 1/λ with λ = the registered expected run length; prune at log-prob −10; erosion
  trigger = expected run length < λ/4 for 5 consecutive days; shock trigger = P(changepoint) > 0.5) →
  flatten and alert; stage `H` ledger rows from forward sessions.
- Verdict after ≥ 252 forward sessions: PSR(0) ≥ 0.95 after Holm across forward families, and
  "consistent with registration" iff the forward Sharpe lies inside the registered CI. Interim reports at
  60 and 126 sessions are descriptive only.
- `HOLDOUT_START = 2026-10-01`; `live/` reads past it with `allow_holdout=True` and an audit line in
  `data/cache/forward_access.jsonl`; research code cannot.

---

## §19 Cost model (U13)

`COST_MODEL ∈ {"slippage", "cs", "quotes"}`:
- `slippage`: `SLIPPAGE_PCT` only (U7).
- `cs`: `SLIPPAGE_PCT` + Corwin–Schultz half-spread from 5Min bars (U8); known to over-estimate by ≈ 10×
  on SPY (1.5 bp vs ≈ 0.3 bp quoted); regression only.
- `quotes`: half-spread from the §12 quotes table (symbol, year, time-of-day bin; nearest earlier year
  when missing; floored at 0.25 bp) + `SLIPPAGE_PCT` for non-auction fills; auction fills (`open`/`close`
  entries and exits) pay the auction proxy half-spread only. Short borrow `borrow_bps` as in §7.
- Every fill records its modelled cost in the trade frame (`cost_bp`); the family report shows cost as a
  share of gross P&L. Invariant: the magnitude floor of §17.2 uses the same table.

Implementation (U13, as built):
- `RunConfig.COST_MODEL` (default `"quotes"`); `RiskProfile.spread` is gone (the profile no longer chooses costs; it
  keeps `spread_floor` / `spread_window_days` for `cs`). The U8–U11 `standard` profile = `RISK_PROFILE="standard"` +
  `COST_MODEL="cs"`. `legacy_5min()` pins `"slippage"`. `COST_MODEL` is backtest-only (WFO signals cache key) and not
  a feature field.
- `risk.costs.fill_costs(index, cfg, cost_data)` returns the one-way cost fraction of a fill at each bar by kind:
  `open` (entries, gap exits through a barrier at the open, gate flattening, drift trims), `intra` (intrabar
  barrier exits), `close` (vertical exits at the close); None for `"slippage"`. `cs`: the same SLIPPAGE_PCT + CS
  half-spread on every kind (bit-identical to U8). `quotes`: an open fill is an opening-auction fill when the bar
  starts at 09:30 (every daily bar), a close fill a closing-auction fill when the bar ends at or after 16:00 (every
  daily bar, session-anchored stubs included) or is the session's last bar in the data and spans 13:00 (early
  close); auction fills pay the `open_auction` / `close_auction` half-spread only, others SLIPPAGE_PCT + the
  half-spread of the 15-minute bin of the fill time (open: bar start, close: bar end, intrabar: bar start; daily
  intrabar: `day`). Year = the bar's year, else the nearest earlier year in the table; a year before the table's
  first, or a bin missing in every earlier year, raises.
- Any `COST_MODEL` other than `"slippage"` routes `run_backtest` through `simulate_portfolio` (one symbol; with profile
  `none` it applies no risk layer and, with SLIPPAGE_PCT cost frames, equals `simulate_trades` bit for bit), so it
  needs `POSITION_MODE="single"` (RunConfig raises otherwise). Portfolio trades carry `entry_cost_bp`,
  `exit_cost_bp` and `cost_bp` (round trip; a trimmed position's is its final exit's).
- `cost_data` replaces `spread_bars` in `run_backtest`, `wfo.pwfo` and the runner: `cs` the symbol's 5Min bars,
  `quotes` its rows of `data/costs/quotes_half_spread.csv` (the runner's data source exposes `quotes_table(symbols)`;
  a symbol without rows raises). The data hash covers each symbol's cost inputs (`cost:{symbol}`).
- Not converted (diagnostics only): `META_MIN_RET = 2 × SLIPPAGE_PCT`, `models/compare.py` and
  `primaries/diagnostics.py` net-of-cost figures.
- The per-year medians use the whole year's sample, so a fill early in a year is priced with that year's later
  sessions too: cost-model look-ahead within a year (not a signal input).
- Parity: with `COST_MODEL="slippage"` added to the old switches (`scripts/u12_parity.py`), the four U11 Stage C
  parity cells reproduce their `daily_returns.csv` and `returns_pwfo.csv` exactly (`parity: PASS`, U13 code).
