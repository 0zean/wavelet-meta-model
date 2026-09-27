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
- [ ] 5Min run on the legacy CSV with legacy parameters reproduces the pre-refactor `wfo_signals.csv`
      bit-for-bit (embargo = 0) — proves the refactor is behavior-preserving.
- [ ] Pipeline completes end-to-end on SPY at 5Min, 15Min, 1Hour, 1Day from Alpaca cache.
- [ ] Tests: daily bars hold overnight and exit at a later session; intraday never does; embargo removes the
      expected samples; the injected-lookahead WFO test fails when the leak is present.
- [ ] `grep -r "from utils.config import config"` only hits the entry point / defaults module.

**Reviewer focus.** Behavior drift hidden in the refactor, daily-bar barrier logic (gap fills, weekend gaps),
trading-day→bar mapping at half-days, embargo direction.

**Status.** Not started.

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
- [ ] Parametrized causality test runs over **every registered feature** (future-perturbation, SPEC §8).
- [ ] Any feature set lacking `wavelet_core` raises.
- [ ] `build_features(df, cfg, groups=[...])` works at 5Min and 1Day (intraday-only groups auto-skipped on 1Day with a log line).
- [ ] Clustered MDA runs on one fold and its selection uses no val/test rows (test asserts index subset).
- [ ] Feature build time for 10y of 5Min on one symbol recorded; cache features keyed by (symbol, tf, group, code hash).

**Reviewer focus.** Hidden lookahead in rolling/EWM/normalization, cross-asset alignment, selection leakage, NaN warm-up handling.

**Status.** Not started.

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
- [ ] Each primary passes the causality test and a synthetic-data sanity test (e.g., `sma_cross` is long in a
      monotone uptrend, `bollinger_mr` fades a spike).
- [ ] WFO runs with any primary via config switch; output schema unchanged apart from `primary` column.
- [ ] Diagnostics table for SPY 5Min / 1Day × all primaries saved (development window only).

**Reviewer focus.** Side determined with information after event time, primaries silently producing all-one-side, tuning leakage.

**Status.** Not started.

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
- [ ] Splitters: no train sample's `[t0, t1]` overlaps any test sample's span or embargo (property test over random event sets).
- [ ] CPCV: split count = C(N,k), each group is tested in exactly φ = C(N−1,k−1) splits, and φ full paths are reconstructed (test).
- [ ] PBO on pure-noise strategy matrix ≈ ≥0.5; on a matrix with one genuinely dominant strategy → low (tests).
- [ ] DSR reproduces the worked example in Bailey & López de Prado (2014) to 3 decimals.

**Reviewer focus.** Off-by-one in purging/embargo, PSR/DSR formula (kurtosis convention: raw vs excess), PBO rank logic.

**Status.** Not started.

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
- [ ] All zoo models run through the same WFO with a config switch; calibration curve + Brier reported OOS.
- [ ] Model comparison table on development window for SPY 5Min/1Day: OOS log-loss, Brier, AUC, meta precision, and
      CPCV Sharpe distribution; each row logged as a trial.
- [ ] Deterministic given seed (two runs identical).

**Reviewer focus.** Hyper-parameter tuning touching val/test, calibration fit on the same rows it's evaluated on,
class-imbalance handling, scaling fit leakage for linear models.

**Status.** Not started.

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
- [ ] Unit tests: each sizer is monotone in p, bounded in [0, 1], and zero at/below threshold; ECDF uses train-fold p only.
- [ ] Backtest with `fixed` sizer reproduces U6 results exactly; fractional sizing P&L verified by a hand-computed 3-trade example.
- [ ] Sizer comparison (dev window, a few cells) logged as trials.

**Reviewer focus.** ECDF/Kelly inputs drawn from test data, size applied at the wrong bar, turnover cost from resizing not charged.

**Status.** Not started.

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
- Removal of unused deps (`dill`, `pyfts`, `pyyaml`, `requests`) — `pyyaml` becomes used in U10.
