# PLAN — Zoo-scale meta-labeling research with Power Walk-Forward retraining

## Thesis

The wavelet meta-labeling pipeline has sound plumbing (causal features, purged WFO, barrier-consistent
backtest), but on SPY 5-min it shows **no OOS skill** (meta AUC 0.50, Sharpe −3.6). The question is not
"does it work on SPY 5-min" but **where, if anywhere, it works**. We answer that by:

1. running it across a **stock zoo × timeframe zoo × feature zoo × primary zoo × model zoo**, on real Alpaca data;
2. turning meta-probabilities into positions with **bet sizing** and a **risk layer**;
3. choosing the retraining cadence and in-sample length with a **Power Walk-Forward (PWFO)** grid
   (Meyers Analytics style: many IS/OOS combos, ≥50 OOS windows, walk-forward efficiency);
4. guarding every selection step against multiple-testing bias with **CPCV, PBO and the Deflated Sharpe
   Ratio**, and scoring the final pick **once** on a locked holdout.

A null result ("no cell survives DSR on the holdout") is a valid outcome and must be reportable as such.

Interfaces, formulas and defaults live in [SPEC.md](SPEC.md). This file is the work breakdown.

## Working protocol (applies to every unit)

1. Implement the unit on a branch `unit/NN-slug` off `main`.
2. Done-gate: `uv run pytest -q` green, `uvx ruff check . && uvx ruff format --check .` clean, and every
   unit-specific criterion below demonstrably met (command + output recorded in the unit's status note).
3. Deploy **one** `adversarial-reviewer` subagent scoped to that unit's diff and claims.
4. Fix every BREAKING/SEVERE finding; fix or explicitly defer (with reason, in this file) each MINOR.
5. Update the unit's **Status** line below, then **stop** and report to the user. The user decides on
   commit/merge and on starting the next unit.

Cross-cutting rules:
- **No network in tests.** Alpaca is mocked; tests use fixtures or synthetic data.
- **No silent fallbacks.** Missing data, wrong feed, failed fits, empty folds → raise or log a counted skip.
- **Causality is tested, not asserted.** Every new feature/primary/label passes the future-perturbation test.
- **Holdout is locked** (SPEC §9). Nothing before U11 may read bars on/after `HOLDOUT_START` (enforced by `load_bars` since U1).
- **Every experiment is a counted trial** in the ledger (U10), because DSR/PBO depend on the trial count.

## Dependency graph

```
U1 data ──► U2 config/timeframes ──► U3 feature zoo ──► U4 primary zoo ─┐
                                    └► U5 validation toolkit ──► U6 model zoo ──┤
                                                                   U7 bet sizing ──► U8 risk + portfolio bt
                                                                                        │
                                                         U9 PWFO engine ◄───────────────┘
                                                                │
                                                         U10 experiment runner + ledger
                                                                │
                                                         U11 staged zoo study + holdout report
```
U3/U4 and U5 can proceed in either order after U2; U7 needs U6's calibrated probabilities.

---

## U1 — Alpaca data layer and cache

**Goal.** Replace `data/data.csv` / synthetic data with reproducible, cached Alpaca bars for any symbol and timeframe.

**Scope.**
- `data/alpaca_source.py`: `fetch_bars(symbol, timeframe, start, end, feed, adjustment)` via `alpaca-py`
  `StockHistoricalDataClient`; credentials from `.env` via `python-dotenv` (`API_KEY`, `SECRET_KEY`); pagination handled by SDK.
- `data/store.py`: cache as native numpy arrays — `data/cache/{feed}/{adjustment}/{timeframe}/{SYMBOL}.npz`
  (one array per column + int64 UTC-ns timestamps) with a sidecar `{SYMBOL}.json` of metadata (SPEC §1);
  atomic writes, incremental top-up, `data/cache/` gitignored.
- Session handling: convert UTC → `America/New_York`; regular-hours filter (09:30–16:00) for intraday using the
  Alpaca **market calendar** (handles half-days); bar timestamps = bar **open** time (Alpaca convention) — document it.
- `load_bars(symbol, timeframe, start, end) -> DataFrame` — the one entry point the pipeline uses.
- Data-quality report per (symbol, timeframe): missing bars vs calendar, zero-volume bars, duplicate stamps, OHLC
  consistency (`low ≤ open,close ≤ high`), split/dividend adjustment flag.
- CLI: `uv run python -m data.fetch --symbols SPY QQQ --timeframes 5Min 1Day --start 2016-01-01`.
- `wavelet_meta_model.py` accepts `--symbol/--timeframe/--start/--end`; CSV path still supported for tests.

**Done when.**
- [x] Unit tests (mocked client) cover: tz conversion, RTH filter incl. a half-day, `.npz`/`.json` cache round-trip
      (exact arrays, tz-aware index restored, metadata/array hash mismatch raises),
      incremental top-up without duplicates, credential-missing error message (no secret echoed).
- [x] Live smoke (manual, recorded): SPY 5Min 2024-01-02→2024-05-15 fetched; overlap with `data/data.csv`
      has close correlation > 0.9999 and bar-count difference explained (feed/adjustment). *Explained, not met
      literally:* overall corr 0.99953 due to a defect in the CSV (see notes); 0.99999999997 outside it.
- [x] Quality report runs for the default zoo on 5Min and 1Day and is saved to `results/data_quality.csv`.
- [x] Secrets never logged; `.env` never read outside `alpaca_source.py`.
- [x] Existing 14 tests still pass.

**Reviewer focus.** Timestamp convention (bar-open vs bar-close → off-by-one lookahead), RTH/half-day
filtering, adjustment consistency across cache top-ups, survivorship bias of the default universe (document, not fix).

**Status.** ✅ Complete (adversarial review done, findings fixed). 45 tests pass. Notes:
- Smoke vs `data/data.csv` (SPY 5Min, raw adjustment): 7,332 = 7,332 bars, identical stamps. Closes match
  to the CSV's 2-decimal rounding (max |Δ| 0.005) except 2024-03-01→03-14, where the CSV is scaled by a constant 0.99690 (±6e-7) —
  the CSV was partially dividend-adjusted for SPY's $1.59 ex-div on 2024-03-15. Overall corr 0.99953 because of
  that CSV defect, not the loader.
- Design changes found on real data: (1) 1Day is resampled from RTH 5Min — Alpaca daily bars include
  extended hours; (2) sessions with < 50% of bars are dropped + logged (Alpaca holes, see SPEC §1);
  (3) 1Day also drops sessions missing their last bar (2019-08-12 for 11/19 symbols).
- Review fixes: vacuous test fixture (pandas 3 µs index) fixed and mutation-checked; empty trailing fetches no
  longer advance cache coverage; `refresh` keeps prior coverage; atomic, 30-day-refreshed calendar cache;
  holdout guard added early (`HoldoutError`); 1Day quality completeness measured on the 5Min base; clear
  errors on empty ranges. Reviewer regression tests kept in `tests/test_data_adversarial.py`.
- Quality report (`results/data_quality.csv`) covers the development window 2016-01-04→2025-09-30 only.
- 19 symbols × 5Min 2016-01-04→2026-09-25: 209.4–209.7k bars each, 2,698 sessions, 0 missing sessions,
  0 OHLC violations, 0 zero-volume bars. Full fetch ≈ 1 min/symbol.
- Pipeline runs end-to-end on Alpaca SPY 5Min (tz-aware index): meta −3.9%, AUC 0.51 — consistent with the CSV run.

---

## U2 — Injected run configuration and timeframe generalization

**Goal.** One pipeline that runs on 1Min…1Day without editing globals.

**Scope.**
- Replace module-level `config` defaults in function signatures with an explicit `RunConfig` passed down
  (SPEC §2). `RunConfig.for_timeframe(tf)` derives `bars_per_day`, vol span, vertical barrier, CUSUM, and
  window lengths from **time units** (SPEC §2 table), so "1 hour hold" means the same thing at every timeframe.
- Overnight policy: `hold_overnight=False` for intraday (current session truncation), `True` for daily
  (vertical barrier in days, no session truncation, gap-at-open fills still apply).
- WFO windows expressed in **trading days** (not bars), mapped to bar positions per timeframe.
- Add the deferred **embargo** (SPEC §5) to `purged()`.
- Add the deferred **stronger end-to-end causality test**: inject a 1-bar look-ahead into a feature and
  confirm the WFO-level test now fails; then remove it.

**Done when.**
- [x] 5Min run on the legacy CSV with legacy parameters reproduces the pre-refactor `wfo_signals.csv`
      bit-for-bit (embargo = 0) — proves the refactor is behavior-preserving.
- [x] Pipeline completes end-to-end on SPY at 5Min, 15Min, 1Hour, 1Day from Alpaca cache.
- [x] Tests: daily bars hold overnight and exit at a later session; intraday never does; embargo removes the
      expected samples; the injected-lookahead WFO test fails when the leak is present.
- [x] `grep -r "from utils.config import config"` only hits the entry point / defaults module (no hits at all).

**Reviewer focus.** Behavior drift hidden in the refactor, daily-bar barrier logic (gap fills, weekend gaps),
trading-day→bar mapping at half-days, embargo direction.

**Status.** ✅ Complete (adversarial review done, findings fixed). 70 tests pass; ruff clean. Notes:
- `RunConfig` (utils/config.py) is frozen and passed as `cfg`; no module-level config remains. Low-level
  functions take explicit arguments (`bar_volatility(close, span)`, `barrier_exits(..., vertical_bars=, hold_overnight=)`).
  `for_timeframe(tf)` fills the SPEC §2 table incl. WFO windows in trading days; `legacy_5min()` = bar windows
  2000/1000/500, `EMBARGO=0`. `SEED` is stamped into all XGBoost params on every construction.
- Regression: `wavelet_meta_model.py data/data.csv` → `wfo_signals.csv` sha1 `402ef202…` on `main` and on this
  branch (also re-checked after the review fixes; the reviewer reproduced it independently from `git archive main`).
- `wfo_folds`: split boundaries on session first bars; half-days count as one day; embargo = bars of the last
  `EMBARGO` sessions (42 after a 13:00 close). Embargo is a gap at the END of each fitting split (SPEC §5 explains
  why a post-test embargo is a no-op in a forward WFO); reviewer challenged and agreed.
- Holdout: `run_wfo` raises `HoldoutError` if data reaches `HOLDOUT_START` (`ALLOW_HOLDOUT` to override); with the
  loader guard, no development label can resolve inside the holdout.
- Deferred U1 questions resolved: sessions dropped by the data layer are simply absent (not counted as days;
  an overnight hold steps over them); the 1Hour 15:30 stub is kept as a normal bar (SPEC §2).
- Tests: `tests/test_runconfig.py` (14) and `tests/test_u2_adversarial.py` (11, reviewer). The WFO causality check
  now cuts right after an OOS event and is shown to fail with a monkeypatched 1-bar look-ahead feature; a
  fitting-side test shows train/val targets and meta inputs are unchanged when every bar from the embargo on
  is perturbed, and that it fails if the embargo is dropped.
- Review fixes: SEED ignored on direct construction / `dataclasses.replace` and param dicts shared between copies
  (now copied + stamped in `__post_init__`); `WINDOW_UNIT="bars"` requires explicit bar counts; CSV input with a
  non-5Min `--timeframe` raises; pre-existing tie bug fixed — a bar that opens beyond a barrier exits there
  even if it also touches the other barrier (0 cases on real SPY; legacy sha unchanged).
- End-to-end SPY runs (Alpaca cache, dev window; `--out` to scratch): no fold skipped at any timeframe.

  | tf | range | folds | OOS events | meta Sharpe | primary Sharpe | meta AUC | runtime |
  |---|---|---|---|---|---|---|---|
  | 1Day | 2016-01→2025-10 | 14 | 234 | 0.18 | −0.02 | 0.572 | 34 s |
  | 1Hour | 2016-01→2025-10 | 98 | 2,734 | −0.11 | −0.53 | 0.526 | 82 min* |
  | 15Min | 2016-01→2025-10 | 108 | 11,547 | −1.06 | −1.27 | 0.495 | 49 min* |
  | 5Min | 2023-01→2025-10 | 62 | 9,741 | −1.90 | −2.85 | 0.513 | 16 min |

  \* under CPU contention from the test suite / reviewer. 5Min used a shorter range because 238 expanding folds
  over 10 years would take hours. These are plumbing checks, not findings (single symbol, no DSR, 1Day has
  only 234 events); buy-and-hold beats every cell.
- Cost: a full-window run is slow (expanding train, 3 × 500-tree XGBoost + fracdiff ADF search per fold) —
  1Hour 98 folds took 82 min under CPU contention. U9/U10 will need rolling IS windows and/or caching.
- Added `--out` to `wavelet_meta_model.py` so runs don't overwrite `results/`.

---

## U3 — Feature zoo (wavelet core kept)

**Goal.** A registry of causal feature groups that can be mixed per experiment; wavelet features are always present.

**Scope.**
- `features/registry.py`: `@feature_group(name, required=False)` registry (SPEC §3). `wavelet_core` is `required=True`.
- Groups (initial):
  - `wavelet_core` — existing causal MODWT S_J AR lags.
  - `wavelet_ext` — causal detail energies D1..DJ, multiple filters (db1, db2, la8), S_J slope, multiscale variance ratios.
  - `trend` — Siegel slope, SMA/EMA distance, ADX-style strength.
  - `mean_reversion` — RSI, Bollinger %b, z-score of close vs rolling mean, VWAP deviation (intraday only).
  - `volatility` — EWM σ, Parkinson/Garman–Klass, hl_spread, vol-of-vol.
  - `microstructure` — log volume, volume z-score, Amihud illiquidity, Roll/Corwin–Schultz spread estimate.
  - `structural` — fracdiff close (d refit per fold), rolling SADF-lite / CUSUM statistic, entropy (optional).
  - `calendar` — time-of-day / day-of-week (cyclical encoding).
  - `cross_asset` (optional) — SPY return/vol as market context for non-SPY symbols (aligned causally).
- Feature selection **inside the training window only**: clustered MDA (López de Prado 2020) with purged CV; returns a kept-feature list per fold.
- Stationarity guard: non-stationary raw levels are rejected by the registry (only returns/ratios/fracdiff).

**Done when.**
- [x] Parametrized causality test runs over **every registered feature** (future-perturbation, SPEC §8).
- [x] Any feature set lacking `wavelet_core` raises.
- [x] `build_features(df, cfg, groups=[...])` works at 5Min and 1Day (intraday-only groups auto-skipped on 1Day with a log line).
- [x] Clustered MDA runs on one fold and its selection uses no val/test rows (test asserts index subset).
- [x] Feature build time for 10y of 5Min on one symbol recorded; cache features keyed by (symbol, tf, group, code hash).

**Reviewer focus.** Hidden lookahead in rolling/EWM/normalization, cross-asset alignment, selection leakage, NaN warm-up handling.

**Status.** ✅ Complete (adversarial review done, findings fixed). 108 tests pass; ruff clean. Notes:
- Registry `features/registry.py`, groups `features/groups.py` (11: `wavelet_core` (required), `wavelet_ext`, `trend`,
  `mean_reversion`, `volatility`, `microstructure`, `structural`, `calendar`, `intraday` (intraday-only), `cross_asset`
  (needs market bars), `fracdiff` (per fold)), `FeatureSet`/`build_features` in `features/feature_builder.py`. Column
  list and formulas in SPEC §3. `cfg.FEATURE_GROUPS` default = wavelet_core, trend, mean_reversion, volatility,
  microstructure, intraday, fracdiff; CLI `--features a,b,c` and `--select cmda`.
- **Decision:** `wavelet_core` = `log(S_J[t−k]/close[t])`, not the raw S_J lags — the raw lags are price levels
  (|ρ| ≈ 1 with close) and violate the SPEC §3 stationarity invariant. The pre-U3 matrix survives only as
  `FEATURE_GROUPS=None` (`legacy_5min()`), so the regression holds: sha1 `402ef202…` re-checked after the review fixes.
- VWAP deviation / time of day / bar length are a separate `intraday` group (so "skip on 1Day" is per group); the 1Hour
  15:30 stub gets `bar_frac = 0.5` (U2 deferral closed). Entropy (structural) not implemented.
- Tests `tests/test_features.py` (49): SPEC §8 causality (random-walk + truncation, 3 seeded cuts) parametrized over every
  registered group at 5Min and every non-intraday group at 1Day; a sensitivity check; registry errors; stationarity
  guard; MODWT details vs pywddff (db1/db2/la8); Roll/Parkinson/EWM σ on known processes; cross-asset alignment;
  cache hit/miss/invalidation; PurgedKFold property test; CMDA keeps the informative cluster; CMDA in the WFO sees only
  purged train rows and its inputs/importances are identical when bars from the train embargo on are perturbed
  (4 seeds + an embargo-only ramp; mutation-checked).
- Build time, SPY 5Min 2016-01-04→2025-09-30 (190,402 bars, all 10 static groups, 63 cols): 1.8 s cold, 0.1 s cached
  (111 MB). Cache key = (group, symbol, tf, `features/*.py` source + numpy/pandas/scipy/pywddff/fracdiff versions,
  feature-relevant cfg fields, data hash, context data hash); used for Alpaca input.
- End-to-end: SPY 1Day default zoo 14 folds / 234 OOS events (meta AUC 0.499); AAPL 1Day `wavelet_core,trend,
  cross_asset,fracdiff` + CMDA completes (5 folds, 6–18 of 20 features kept per fold).
- Review fixes: (BREAKING) fracdiff d was fit on train bars incl. the embargo, so train features — and CMDA — depended
  on embargo bars (no val/test leak); now fit on `[0, train_end − embargo)`, and the test that passed by seed luck now
  uses several seeds + a ramp. (SEVERE) `--select cmda` on the legacy matrix dropped every column and crashed; `cmda`
  now requires `FEATURE_GROUPS` and CMDA raises without `wavelet_core` columns. (MINOR, fixed) per-fold count of events
  dropped for NaN features logged; library versions in the cache key; CMDA forest single-threaded (bit-reproducible);
  SPEC warm-up statement corrected (390 bars at 1Min).
- Deferred MINORs: the |ρ| ≥ 0.99 guard only catches near-verbatim levels (SPY 1Day `log(close)` has ρ = 0.985) — it is a
  backstop, stationarity of new groups stays a review item; zero-volume session-start bars / a market session missing
  for ≥ 50 bars would NaN `vwap_dev` / `corr_50` (0 cases in the cache; now logged as dropped events, not silent).
- Deferred (U9/U10): fracdiff's ADF search dominates per-fold cost — fitting d on a 5-year 5Min train took 18 min.
  Full-window 5Min WFO needs rolling IS windows or a cached/coarser d search.
- `validation/purged_cv.py` is a minimal PurgedKFold for CMDA; U5 extends it (CPCV, sklearn adapter).

---

## U4 — Primary signal zoo

**Goal.** Meta-labeling adds most on top of a **high-recall** primary (Hudson & Thames; López de Prado ch.3).
Test rule-based primaries alongside the current ML primary.

**Scope.**
- `primaries/` with a common `Primary` protocol (SPEC §4): `side(df, feats, events) -> {-1,+1}` per event.
- Implementations: `ml_xgb` (current classifier+regressor), `sma_cross` (fast/slow), `bollinger_mr`,
  `wavelet_trend` (sign of S_J slope / S_J vs close), `donchian_breakout`.
- Primary diagnostics per fold: event count, precision (side-correct rate net of costs), recall of profitable
  moves, turnover. Written into the WFO signal frame.
- Rule primaries with parameters get **no tuning on test**; parameters either fixed or tuned in train only.

**Done when.**
- [x] Each primary passes the causality test and a synthetic-data sanity test (e.g., `sma_cross` is long in a
      monotone uptrend, `bollinger_mr` fades a spike).
- [x] WFO runs with any primary via config switch; output schema unchanged apart from `primary` column.
- [x] Diagnostics table for SPY 5Min / 1Day × all primaries saved (development window only). *1Day saved; 5Min
      deferred to U9/U10 by user decision (runtime, see notes).*

**Reviewer focus.** Side determined with information after event time, primaries silently producing all-one-side, tuning leakage.

**Status.** ✅ Complete (adversarial review done: no BREAKING/SEVERE; MINORs fixed). 5Min diagnostics deferred to U9/U10. 143 tests pass; ruff clean. Notes:
- `primaries/`: `base.py` (protocol, `@primary` registry, `make_primary(cfg)`, `check_signal`, `rule_frame`), `ml_xgb.py`
  (pre-U4 classifier + regressor), `rules.py` (`sma_cross`, `bollinger_mr`, `wavelet_trend` slope/level,
  `donchian_breakout`), `diagnostics.py`. Config `PRIMARY` (default `ml_xgb`) / `PRIMARY_PARAMS`; CLI `--primary`,
  `--primary-params`. Formulas and contract in SPEC §4.
- **Decision:** the protocol returns the full primary frame via `signal()` (it is also the meta-model's input);
  `side()` is a helper over its `signed_dir`. Rule frames have `clf_prob` = NaN and |score| as magnitude/confidence.
- **Decision:** diagnostics (n_events, long_share, precision net of costs, opportunity, recall, turnover, net_bp; per fold
  + all) are computed post hoc from the signal frame into `primary_diagnostics.csv` per run, not added as signal
  columns — the schema criterion allows only `primary`, and outcomes should not sit next to the signals.
- Regression: legacy CSV run → `wfo_signals.csv` with the `primary` column stripped = sha1 `402ef202…`. SPY 1Day
  `ml_xgb` output is bit-identical to `main` (sha1 `5b82e1d2…`, meta AUC 0.5065 — the U3 note's 0.499 was stale).
- Tests `tests/test_primaries.py` (35): SPEC §8 causality per rule (random walk + truncation, 3 cuts) and for ml_xgb;
  a leaky rule is caught; WFO-level causality with a rule primary (mutation-checked: Donchian `shift(-1)` fails both);
  rules ignore labels; synthetic sanity (trend rules follow monotone/noisy trends, Donchian breakout strength > 1,
  Bollinger fades spikes, ml_xgb learns a planted side); hand formulas per rule; registry/param errors; WFO with every
  primary (schema + rule sides passed through); hand-computed diagnostics (tie bar, sub-cost exit, mixed sides).
- Review fixes (all MINOR): diagnostics test now catches tie/cost/net_bp mutations; rule window params must be
  ints ≥ 2 (JSON `20.0` failed only inside fold 1); exact-formula tests; one-sided warning at < 10% / > 90% long on
  val or test (was only at exactly 0/100%).
- SPY 1Day dev window (14 folds, 234 OOS events; `results/primary_diagnostics.csv`):

  | primary | long share | precision | recall | turnover | net bp/event | meta AUC | meta Sharpe | primary Sharpe |
  |---|---|---|---|---|---|---|---|---|
  | ml_xgb | 0.75 | 0.530 | 0.535 | 0.24 | +0.6 | 0.507 | 0.12 | −0.12 |
  | sma_cross | 0.56 | 0.457 | 0.461 | 0.09 | −38.3 | 0.474 | −0.18 | −0.50 |
  | bollinger_mr | 0.48 | 0.432 | 0.435 | 0.22 | −39.5 | 0.588 | −0.21 | −0.96 |
  | wavelet_trend | 0.50 | 0.551 | 0.556 | 0.17 | +30.7 | 0.545 | 0.94 | 0.64 |
  | donchian_breakout | 0.58 | 0.547 | 0.552 | 0.22 | +29.0 | 0.527 | −0.07 | 0.81 |

  Buy-and-hold Sharpe 0.69. 234 events, single symbol, no DSR: plumbing evidence, not findings.
- 5Min (SPY 2023-01→2025-10, 62 folds) was stopped at 9–17 folds after ~2 h of contended CPU: 5 parallel runs each
  refit fracdiff + meta XGB per fold. Rule-primary diagnostics do not need the WFO fits and could be computed directly;
  `ml_xgb` needs the full WFO (≈ 16 min alone, U2). **Deferred to U9/U10** (user decision): rerun once rolling IS
  windows / cached fracdiff d make 5Min WFO affordable.

---

## U5 — Validation toolkit (CV, overfitting statistics)

**Goal.** Quant-appropriate model selection and honest performance statistics.

**Scope.** `validation/` (SPEC §5–6):
- `PurgedKFold(n_splits, embargo)` — sklearn-compatible splitter using event `t1`.
- `CombinatorialPurgedCV(N, k, embargo)` — CPCV; reconstructs φ = C(N,k)·k/N backtest paths.
- `pbo(perf_matrix)` — CSCV Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu 2017).
- `psr`, `dsr(sr, n_trials, var_trials, T, skew, kurt)` — Probabilistic & Deflated Sharpe (Bailey & López de Prado 2014).
- `min_track_record_length`.
- Scoring: weighted neg-log-loss / Brier (probabilistic) as primary selection metrics; F1/AUC reported, not selected on.

**Done when.**
- [x] Splitters: no train sample's `[t0, t1]` overlaps any test sample's span or embargo (property test over random event sets).
- [x] CPCV: split count = C(N,k), each group is tested in exactly φ = C(N−1,k−1) splits, and φ full paths are reconstructed (test).
- [x] PBO on pure-noise strategy matrix ≈ ≥0.5; on a matrix with one genuinely dominant strategy → low (tests).
- [x] DSR reproduces the worked example in Bailey & López de Prado (2014) to 3 decimals.

**Reviewer focus.** Off-by-one in purging/embargo, PSR/DSR formula (kurtosis convention: raw vs excess), PBO rank logic.

**Status.** ✅ Complete (adversarial review done, findings fixed). 216 tests pass; ruff clean. Notes:
- Modules: `validation/purged_cv.py` (`purged_train`, `PurgedKFold`, `CombinatorialPurgedCV` with `path_splits` /
  `assemble_paths` / `from_cfg`, `bind()` → scikit-learn `cv=` adapter, `embargo_bars`), `validation/stats.py`
  (`sharpe_ratio`, `return_moments`, `psr`, `expected_max_sharpe`, `dsr`, `min_track_record_length`), `validation/pbo.py`
  (CSCV PBO, vectorized via block moments; also P(OOS loss) and the IS→OOS degradation slope), `validation/scoring.py`
  (`neg_log_loss`, `brier`, `selection_score` (higher = better), report-only `score_report`, `purged_cv_predict/score`).
  Config: `CPCV_GROUPS=10`, `CPCV_TEST_GROUPS=2`, `PBO_BLOCKS=16`, `SELECTION_METRIC="neg_log_loss"` (validated; excluded
  from the feature-cache key). Conventions in SPEC §5–6.
- Purge is per contiguous test run (CPCV test sets are non-contiguous), with the embargo after each run; exact vs a
  per-sample brute force (40 seeded property cases in the suite; reviewer: 1,500 configs, 0 mismatches). `PurgedKFold`
  output is identical to `main` (reviewer: 3,000 configs); SPY 1Day `--select cmda` `wfo_signals.csv` is bit-identical to
  `main` (sha1 `79bf64c2…`); legacy CSV run still sha1 `402ef202…`.
- DSR example (Bailey & López de Prado 2014): SR₀ = 0.1132, DSR = 0.9004, N = 46 → 0.9505 — matched to 4 decimals.
  Raw kurtosis; excess would give 0.9018. PSR calibration under the null (5.0% ± 1.2% false positives), E[max SR] vs
  Monte Carlo (≤ 3%), MinTRL inverts PSR.
- PBO: noise ≈ 0.5 for N = 2, 3, 5, 9, 20; one dominant strategy < 0.05; reverting IS winners > 0.9; hand-computed
  2-block examples pin rank direction, tie averaging and the median rule.
- Tests: `tests/test_validation.py` (73).
- Review (no BREAKING/SEVERE) — MINORs fixed: (1) odd-N PBO noise baseline was (N+1)/(2N) because λ = 0 counted as
  overfit → λ = 0 now counts ½ (pinned by test); (2) `ceil(0.07·100)` = 8 from float error → `embargo_bars` rounds first
  and clustered MDA uses it (identical to the old `ceil` for every n ≤ 400,000 at the default 0.01); (3) a flat non-zero
  column scored ~1e7 instead of ±inf (cancellation) → block moments are shifted by each column's first value;
  (4) CPCV / PBO config fields now validated. `PBO_BLOCKS` / `SELECTION_METRIC` are consumed from U6/U9 on.
- For U6: `purged_cv_predict` passes `sample_weight=` to `fit`, which an sklearn `Pipeline` rejects (loud error) — U6's
  scaled logit models must route weights (`logisticregression__sample_weight`) or implement ZooModel.fit directly.

---

## U6 — Model zoo and selection

**Goal.** Test whether something other than XGBoost is better for the primary/meta models, selected with purged CV.

**Scope.**
- `models/zoo.py` behind one protocol (SPEC §4): `logit_l1`, `logit_l2` (standardized), `rf_ldp`
  (López de Prado settings: `max_features=1`/sqrt, `class_weight="balanced_subsample"`, `max_samples≈avg uniqueness`),
  `extra_trees`, `xgb` (current), `lightgbm`, optional `sequential_bootstrap_bagging`.
- Hyper-parameter search: small grids per model, **inside train** with `PurgedKFold` (not the WFO val/test).
- Probability calibration (isotonic or Platt, chosen by purged-CV Brier) — required by U7.
- Meta-model training data: replace the single VAL block with purged-CV OOF primary predictions on train+val
  when it increases meta sample size (A/B vs current scheme, recorded).
- Out of scope unless a cheap screen shows promise: deep sequence models (TCN/LSTM). Noted as stretch.

**Done when.**
- [x] All zoo models run through the same WFO with a config switch; calibration curve + Brier reported OOS.
- [x] Model comparison table on development window for SPY 5Min/1Day: OOS log-loss, Brier, AUC, meta precision, and
      CPCV Sharpe distribution; each row logged as a trial. *(5Min CPCV on a shorter window — see notes.)*
- [x] Deterministic given seed (two runs identical).

**Reviewer focus.** Hyper-parameter tuning touching val/test, calibration fit on the same rows it's evaluated on,
class-imbalance handling, scaling fit leakage for linear models.

**Status.** ✅ Complete (adversarial review done, findings fixed). 250 tests pass (3 min 40 s); ruff clean. Notes:
- Modules: `models/zoo.py` (registry, `make_model(name, cfg, role)`, `inner_cv`, `_Tuned` purged-CV HP search +
  sigmoid/isotonic calibration by cross-fitted Brier, `ScaledLogit`, serial-predict forests, `legacy`),
  `models/compare.py` (WFO row + CPCV path Sharpes → `results/model_zoo/comparison_*.csv`, `calibration_*.csv`,
  `trials.jsonl`). Config `META_MODEL`, `PRIMARY_MODEL` (ml_xgb's classifier), `ZOO_CV_SPLITS=4`, `META_TRAIN`
  (`val`|`oof`); all default to pre-U6 behaviour; excluded from the feature-cache key. CLI `--meta-model`,
  `--primary-model`, `--meta-train`. OOS meta log-loss / Brier / AUC + `meta_calibration.csv`; skipped folds in
  `signals.attrs` and `wfo_run.json`. Dependency: `lightgbm` 4.7. Conventions in SPEC §4.
- Regression: legacy CSV run still sha1 `402ef202…`. Determinism: two SPY 1Day comparisons (rf_ldp, lightgbm) identical
  except timestamps; `test_wfo_runs_with_zoo_models_and_is_deterministic` reruns 4 WFO configs.
- Tests `tests/test_zoo.py` (34): every model learns/bounded/deterministic; legacy == pre-U6 XGB; HP-search fits see only
  the fitting rows and pick the best (ties → first); calibration fit on OOF (memorizing forest on noise stays at the
  base rate); stored CV scores, calibration Briers, refit and calibrator recomputed independently with unequal weights;
  output clipping; weighted scaler inside fit; weight routing; all-NaN column dropped; single-class → skip; WFO switch +
  causality with tuned meta and OOF; OOF primary purged/partitioned; CMDA inside each OOF split; primary-fit skip.
- Review (no BREAKING) — SEVERE fixed: `META_TRAIN="oof"` + `cmda` + ml_xgb selected features on train labels before
  the OOF primary scored those same events → selection now runs inside each OOF split (test + mutant check). MINORs fixed:
  five weighting/calibration/clip mutants survived the tests → new recomputation tests kill all five; skipped folds
  only in `attrs` → `wfo_run.json`; a primary `ZooFitError` aborted the WFO → fold skipped + recorded; a CPCV failure
  marked valid WFO metrics as an error → `cpcv_error`. Reviewer simulation: the cross-fitted calibration choice shows no
  optimism (fresh − CV Brier within 1 SE).
- SPY 1Day, 2016-01→2025-10, 14 folds, 234 OOS events, `wavelet_trend` primary. CPCV (45 splits, 9 paths) does not depend
  on META_TRAIN. Buy-and-hold Sharpe 0.69; primary-only 0.64.

  | meta model | log-loss val / oof | Brier val / oof | AUC val / oof | precision val / oof | Sharpe val / oof | CPCV SR mean ± sd (min) |
  |---|---|---|---|---|---|---|
  | legacy | 0.940 / 0.866 | 0.316 / 0.298 | 0.545 / 0.531 | 0.580 / 0.596 | 0.94 / 0.60 | 0.47 ± 0.17 (0.26) |
  | logit_l1 | 0.701 / 0.681 | 0.254 / 0.244 | 0.433 / 0.575 | 0.556 / 0.600 | 0.59 / 0.88 | 0.74 ± 0.23 (0.31) |
  | logit_l2 | 0.873 / 0.682 | 0.270 / 0.244 | 0.529 / 0.576 | 0.571 / 0.597 | 0.46 / 1.07 | 0.49 ± 0.16 (0.32) |
  | rf_ldp | 0.695 / 0.678 | 0.251 / 0.242 | 0.542 / 0.579 | 0.565 / 0.620 | −0.15 / 0.60 | 0.82 ± 0.17 (0.54) |
  | extra_trees | 0.707 / 0.682 | 0.256 / 0.245 | 0.529 / 0.568 | 0.544 / 0.603 | −0.23 / 0.32 | 0.88 ± 0.15 (0.62) |
  | xgb | 0.711 / 0.693 | 0.258 / 0.250 | 0.475 / 0.540 | 0.534 / 0.552 | −0.02 / −0.07 | 0.20 ± 0.11 (0.03) |
  | lightgbm | 0.766 / 0.695 | 0.272 / 0.251 | 0.470 / 0.522 | 0.549 / 0.535 | −0.08 / 0.45 | 0.14 ± 0.08 (−0.00) |

  A/B (recorded): `oof` (≈ 4× the meta rows) improves OOS log-loss and Brier for every model and AUC for every tuned
  model. Only rf_ldp / extra_trees / logit_* reach log-loss < ln 2 = 0.693, and only barely. `legacy` is badly
  overconfident (0.87–0.94). 234 events, one symbol, no DSR (21 trials so far): plumbing evidence, not findings.
- Primary-model zoo (ml_xgb, `legacy` meta, `val`): every calibrated classifier goes long on every test event in all 84 zoo fold-tests (6 models × 14; P(long)
  sits at the up-move base rate > CLF_THRESH = 0.5), so CPCV path Sharpes collapse to ≈ 1.05 ± 0.0–0.1, i.e. SPY
  buy-and-hold on event days. `legacy` (uncalibrated) keeps mixed sides: CPCV 0.05 ± 0.27. A calibrated primary needs a
  side threshold relative to the train base rate — **deferred to U7/U11** (sizing/threshold choice; not tuned here).
- SPY 5Min, `wavelet_trend` primary, `val`. Full CPCV on 2023-01→2025-10 (~12k events × 45 splits × inner search) did not
  finish one model in 2.5 h and was stopped, so the WFO table covers 2023-01→2025-10 and CPCV a shorter 2025-01→2025-10
  window (its own WFO row is in `trials.jsonl`). The full-window 5Min reliability curves were overwritten by the short run
  (filenames lacked dates, now fixed); its log-loss / Brier are in `trials.jsonl`.

  | meta model | WFO 2023-01→2025-10 (62 folds, 9,741 events): log-loss | Brier | AUC | precision | Sharpe | CPCV 2025 (1,825 WFO events): SR mean ± sd (min) |
  |---|---|---|---|---|---|---|
  | legacy | 0.891 | 0.315 | 0.502 | 0.463 | −2.67 | −2.49 ± 0.62 (−3.07) |
  | logit_l1 | 0.698 | 0.252 | 0.509 | 0.489 | −0.76 | −0.14 ± 0.81 (−1.90) |
  | logit_l2 | 0.698 | 0.252 | 0.498 | 0.469 | −1.21 | −0.12 ± 0.73 (−0.91) |
  | rf_ldp | 0.698 | 0.252 | 0.500 | 0.477 | −1.17 | −1.26 ± 1.87 (−3.75) |
  | extra_trees | 0.696 | 0.251 | 0.508 | 0.481 | −1.54 | −1.23 ± 1.52 (−2.52) |
  | xgb | 0.697 | 0.252 | 0.500 | 0.462 | −2.07 | −1.79 ± 1.51 (−4.98) |
  | lightgbm | 0.697 | 0.252 | 0.499 | 0.467 | −1.49 | −1.78 ± 1.51 (−4.93) |

  Primary-only Sharpe −2.65, buy-and-hold 1.47. No 5Min meta-model beats a constant (log-loss ≥ ln 2); calibration mostly
  just shrinks P toward the base rate. It approves fewer trades than `legacy` (27–32% vs 42%), which lifts Sharpe from −2.7
  to between −0.8 and −2.1. That comes from abstaining, not from skill (AUC ≈ 0.50). 35 trials logged (stage U6). The
  `oof` A/B was not run at 5Min (cost).
- Stretch not done: `sequential_bootstrap_bagging`, deep sequence models.

---

## U7 — Bet sizing

**Goal.** Size positions from calibrated meta-probabilities instead of all-in/none (H&T "next steps"; López de Prado ch.10).

**Scope.** `sizing/` (SPEC §7):
- `fixed` (current behaviour, baseline), `ldp_sigmoid` (z = (p−1/K)/√(p(1−p)) → m = 2Φ(z)−1),
  `ecdf` (rank of p against the train-fold p distribution), `kelly_capped` (fractional Kelly from p and
  realized win/loss ratio in train), `linear` (clip((p−τ)/(1−τ))).
- Active-bet averaging for overlapping signals and size discretization (step size) to limit turnover.
- Backtest supports **fractional** position size and (for U8) concurrent positions.

**Done when.**
- [x] Unit tests: each sizer is monotone in p, bounded in [0, 1], and zero below threshold (`linear`/`ecdf` also at it);
      ECDF uses train-fold p only.
- [x] Backtest with `fixed` sizer reproduces U6 results exactly; fractional sizing P&L verified by a hand-computed 3-trade example.
- [x] Sizer comparison (dev window, a few cells) logged as trials.

**Reviewer focus.** ECDF/Kelly inputs drawn from test data, size applied at the wrong bar, turnover cost from resizing not charged.

**Status.** ✅ Complete (adversarial review done, findings fixed). 293 tests pass (4 min 6 s); ruff clean. Notes:
- Modules: `sizing/sizers.py` (registry, `make_sizer`, `discretize`, `SizerFitError`; `fixed`, `linear`, `ldp_sigmoid`,
  `ecdf`, `kelly_capped`), `models/meta_model.py` (`side_returns`, `oof_meta_prob`: purged k-fold OOF meta-probs of the
  meta-model's own fitting events, computed only when a sizer needs them), `wfo_engine.fold_sizers` (WFO column
  `bet_size`; `run_wfo(..., sizers=)` adds `bet_size:<name>`; unfit sizer → no-trade fold in
  `attrs["sizer_skipped_folds"]` / `wfo_run.json`), `wfo/backtest.py` (fractional sizes in `single` mode;
  `simulate_positions` = active-bet averaging with concurrent bets, costs on every notional change; turnover /
  exposure / avg-position attrs), `sizing/compare.py` (one WFO per cell, every sizer × mode → `results/sizing/`).
  Config `SIZER="fixed"`, `SIZE_STEP=0.1` (0 or 1/k), `KELLY_FRACTION=0.25`, `POSITION_MODE="single"`; all default to
  pre-U7 behaviour; excluded from the feature-cache key. CLI `--sizer`, `--size-step`, `--position-mode`. SPEC §7.
- SPEC refinement: `ecdf` ranks p against the train OOF probabilities **≥ τ** (ranking against all of them would floor
  every approved size at F̂(τ) ≈ ½).
- Regression: legacy CSV run → `wfo_signals.csv` with `primary,bet_size` stripped = sha1 `402ef202…`; with `fixed`, the
  new backtest's equity and trades are **bit-identical** to `git show main:wfo/backtest.py` on those signals (195 / 345
  trades), re-checked after the review fixes.
- Tests `tests/test_sizing.py` (43): monotone/bounded/zero-below-τ for 3 values of τ; formulas; unfit sizers raise; ECDF
  a function of train inputs; discretize; hand-computed 3-trade fractional example (equity at 8 bars, rel 1e-12);
  zero-size bets skipped without blocking; fixed == verbatim pre-U7 equity loop (bitwise); averaging hand example
  (resize at open, exit at close, every change charged, turnover, exposure); opposite sides net to flat; sub-step
  changes do not trade; sized backtest causal in both modes and a size at event t first moves equity at t+1; WFO size
  columns, determinism, adding sizers changes nothing else; sizer inputs = purged fitting events of the fold (both
  META_TRAIN), OOF models never trained on the event or overlapping spans; sizer fit failure = counted no-trade fold;
  WFO + sizer causal; 5 review regressions.
- Review (no BREAKING / SEVERE). MINORs fixed: (1) same-bar exits filled at the close before an earlier intrabar barrier
  (infeasible order) → per-bar phases open → intrabar barriers → vertical closes (reviewer repro 10,077.91 → 10,018.01,
  test); (2) `SIZE_STEP=0` left a float-residue phantom position (0.3 − 0.1 − 0.2) → mean rounded to 1e-12, sum reset
  when no bet is live (test); (3) a step not dividing 1 silently de-levered `fixed` and the primary-only benchmark →
  `SIZE_STEP` must be 0 or 1/k (test); (4) `Avg Bet Size` meant different things per mode → mean |size| over held closes
  in both (test); (5) float32 legacy `meta_prob` tied at a non-default τ: `trade_signal` set but `bet_size` 0 → threshold
  compared in p's own dtype (test). Reviewer checks that held: slippage cost = slip × turnover to 0.1% in log equity;
  averaging == single on 40 random non-overlapping configurations.
- Sizer comparison: `wavelet_trend` primary, 50 trials (stage U7) in `results/sizing/trials.jsonl` (5 cells × 5 sizers × 2
  modes); 50 pre-review trials (same cells, pre-fix engine) kept in `results/sizing/pre_review/` — both count for DSR.
  OOS Sharpe (single / average); `fixed`/single = the U6 row:

  | cell (meta, train) | fixed | linear | ldp_sigmoid | ecdf | kelly_capped | primary-only | B&H |
  |---|---|---|---|---|---|---|---|
  | SPY 1Day (logit_l2, oof) | 1.07 / 0.88 | 0.69 / 0.46 | 0.51 / 0.51 | 0.77 / 0.67 | 0.49 / 0.28 | 0.64 | 0.69 |
  | SPY 1Day (rf_ldp, oof) | 0.60 / 0.75 | 0.75 / 0.60 | 0.61 / 0.55 | 0.70 / 0.57 | 0.14 / −0.28 | 0.64 | 0.69 |
  | QQQ 1Day (logit_l2, oof) | −0.28 / 0.39 | 0.06 / 0.60 | 0.08 / 0.73 | 0.32 / 0.72 | 0.53 / 0.45 | 0.28 | 0.64 |
  | IWM 1Day (logit_l2, oof) | −0.43 / −0.39 | 0.43* / 0.28* | 0.43* / 0.19* | −0.62 / −0.23 | none | −0.12 | 0.30 |
  | SPY 5Min 2025-01→10 (logit_l1, val) | −2.11 / −1.65 | −3.10 / −1.98 | −1.85 / −0.47 | −0.41 / −0.19 | none | −1.64 | 2.55 |

  \* 3 trades at size 0.1. "none" = every Kelly size rounds to 0 (no trades). Sizes are small because calibrated p sits
  near τ: mean size 0.19–0.26 for `linear`/`ldp_sigmoid`, 0.54–0.66 for `ecdf`, ≤ 0.1 for `kelly_capped` (λ = ¼ of
  an edge of a few %). At 5Min `ecdf`/`kelly_capped` are unfit in 4 of 12 folds (no approved OOF event or no win/loss),
  counted and traded flat. Sizing lowers vol and turnover roughly in proportion to size; it does not
  create skill where the meta-model has none (IWM, 5Min: AUC ≤ 0.5). No cell/sizer is significant: 50 trials, a few
  hundred events per 1Day cell, no DSR yet. Averaging helps QQQ, hurts SPY logit_l2: noise at this sample size.
- Deferred to U11 (not sizing): a side threshold for calibrated zoo *primaries* relative to the train base rate (U6).
- Out of scope here: vol targeting, caps and the risk layer (U8).

---

## U8 — Risk management and portfolio backtest

**Goal.** Realistic, risk-controlled P&L across multiple symbols at once.

**Scope.** `risk/` + extension of `wfo/backtest.py` (SPEC §7):
- Volatility targeting (per-trade notional scaled to target σ contribution).
- Caps: per-position, per-symbol, gross and net exposure; max concurrent positions.
- Loss controls: daily loss limit (flat until next session), drawdown throttle (scale size by DD tier).
- Cost model: per-symbol half-spread estimate (Corwin–Schultz or quoted-spread fallback) + slippage floor; optional short-borrow bps.
- Portfolio simulator: event-driven over a union timeline of all symbols in a cell, one equity curve.
- Metrics: annualized return/vol, Sharpe, Sortino, Calmar, max DD & duration, turnover, exposure, hit rate,
  profit factor, tail ratio, plus PSR — all OOS only.

**Done when.**
- [ ] Single-symbol, no-risk-layer backtest reproduces U7 output exactly.
- [ ] Tests: caps never exceeded (property test), daily stop flattens and blocks re-entry that session, vol target
      scales inversely with σ, costs are charged on every notional change.
- [ ] Portfolio run over ≥5 symbols at 1Hour completes; metrics table saved.

**Reviewer focus.** Using same-bar close for sizing decisions executed at that bar's open, cap enforcement after
fills vs before, cross-symbol timestamp alignment, cost double-counting.

**Status.** Not started.

---

## U9 — Power Walk-Forward engine (PWFO)

**Goal.** Choose IS length and retraining cadence robustly, Meyers-style, then walk forward the *choice itself*.

**Scope.** `wfo/pwfo.py` (SPEC §6):
- Calendar-based rolling (and expanding) windows in trading days; retrain cadence = OOS length.
- Grid: IS ∈ {63, 126, 252, 504} trading days × OOS ∈ {5, 10, 21, 63} trading days (bi-weekly = 10), configurable.
- Per (IS, OOS) combo: full WFO (inner purged-CV model/HP/calibration selection inside each IS window), then
  per-window IS and OOS metrics, **walk-forward efficiency** (annualized OOS / annualized IS return), % profitable
  OOS windows, IS↔OOS rank correlation, OOS Sharpe, max DD, number of OOS windows (flag < 50).
- **Nested selection**: combo is chosen on a rolling "selection lookback" of prior OOS windows only, then applied to
  the next OOS window (walk-forward of the walk-forward), so the reported PWFO equity curve contains no
  selection look-ahead.
- Outputs: combo heat-maps (WFE, OOS Sharpe), per-window table, stitched OOS equity, PBO across combos.

**Done when.**
- [ ] Window generator tests: no IS/OOS overlap, OOS windows tile the span contiguously, embargo respected, half-days.
- [ ] Nested-selection test: perturbing any OOS window's returns cannot change the combo chosen *for that same window*.
- [ ] Legacy fixed-bar WFO is reproducible as a single PWFO combo.
- [ ] PWFO on SPY 1Hour completes with heat-maps + PBO + DSR (n_trials = combos) reported.

**Reviewer focus.** Selection look-ahead in the nested step, WFE definition with negative IS returns, window counts
too small to be meaningful, inner CV leaking across window boundaries.

**Status.** Not started.

---

## U10 — Experiment runner, trial ledger and caching

**Goal.** Run the zoo reproducibly and cheaply, and count every trial.

**Scope.** `experiments/` (SPEC §9):
- Experiment spec (YAML) → cartesian/staged grid of cells (symbol, timeframe, feature groups, primary, model, sizer,
  risk profile, PWFO grid).
- Cell hash = hash(spec fields + code version); results & artifacts keyed by hash; re-runs are cache hits.
- Ledger (`results/ledger.jsonl`, append-only): one row per trial with metrics, n_obs, SR moments, timings, git SHA, stage.
- Parallel execution with `joblib` (process-level), per-cell error capture (failures recorded, never dropped).
- `report` command: leaderboard with DSR (n_trials from ledger), PBO per stage, and Markdown/HTML summary.

**Done when.**
- [ ] Re-running an identical spec performs zero model fits (cache test).
- [ ] Killing a run mid-way and restarting resumes without duplicate ledger rows.
- [ ] A failing cell produces a ledger row with `status="error"` and traceback path.
- [ ] Holdout guard: the runner refuses any cell whose data range crosses `HOLDOUT_START` unless `--final` is given.

**Reviewer focus.** Hash omitting a field that changes results, ledger trial count undercounting (DSR too generous), cache staleness after code change.

**Status.** Not started.

---

## U11 — Staged zoo study and holdout evaluation

**Goal.** Answer the thesis: which (symbol, timeframe, features, primary, model, sizing, risk, cadence) cells, if any, have robust OOS edge.

**Stages** (all on development window; every cell is a ledger trial):
- **A — Screen:** full stock × timeframe × primary zoo, default features (`wavelet_core`+defaults), `xgb`/`rf_ldp`,
  fixed sizing, bi-weekly cadence. Keep cells with meta OOS AUC > 0.52 and positive PSR, max top-K.
- **B — Refine:** feature groups × model zoo × sizers on survivors.
- **C — PWFO:** IS × cadence grid on finalists; nested selection; PBO across the stage.
- **D — Portfolio:** combine finalists into a risk-managed portfolio (U8).
- **E — Holdout (once):** run the frozen finalists + portfolio on `[HOLDOUT_START, end]`. Report DSR using the
  **total** ledger trial count.

**Done when.**
- [ ] Report (published as an Artifact) with: stage funnel counts, leaderboard, heat-maps (symbol × timeframe,
      IS × cadence), sizing/risk ablations, PBO per stage, DSR on holdout, and an explicit verdict (edge / no edge) per finalist.
- [ ] Holdout accessed exactly once (ledger shows a single `stage=E` batch).
- [ ] Recommended retraining cadence and IS length stated with its WFE and OOS window count.

**Reviewer focus.** Any post-holdout tuning, cherry-picked reporting, trial undercount, survivorship bias in the universe.

**Status.** Not started.

---

## Deferred / out of scope (revisit after U11)
- Live/paper trading loop against Alpaca (order routing, reconciliation).
- Alternative bars (dollar/volume/tick) — Alpaca trades endpoint makes this feasible; big data cost.
- Deep sequence models (TCN/LSTM/transformers) beyond a cheap screen.
- Point-in-time universe (delisted names) to remove survivorship bias.
- (U9/U10) SPY 5Min × all primaries diagnostics deferred from U4 — full-window 5Min WFO too slow before rolling IS / d caching.
- Removal of unused deps (`dill`, `pyfts`, `pyyaml`, `requests`) — `pyyaml` becomes used in U10.
