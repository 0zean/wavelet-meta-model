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

| tf | bars/day | vertical barrier (default) | vol span (bars) | hold overnight |
|---|---|---|---|---|
| 1Min | 390 | 30 bars (30 min) | 390 | no |
| 5Min | 78 | 12 bars (1 h) | 100 | no |
| 15Min | 26 | 8 bars (2 h) | 78 | no |
| 30Min | 13 | 6 bars (3 h) | 65 | no |
| 1Hour | 7* | 6 bars (≈ session) | 70 | no |
| 1Day | 1 | 10 bars (2 wk) | 50 | yes |

\* 09:30–16:00 anchored at 09:30 gives six full hours + a 30-min stub; the stub bar is kept and flagged.

Timeframe-independent fields (existing semantics): `BARRIER_MULT`, `CUSUM_MULT`, `SLIPPAGE_PCT`,
`META_MIN_RET = 2·SLIPPAGE_PCT`, `SEED`, model params. WFO windows are in **trading days** (§6).

**Overnight policy invariant.** `hold_overnight=False` → vertical barrier truncated at the entry session's
last bar, events on a session's last bar skipped (current behaviour). `True` → no truncation; the next
session's open is a normal bar (gap-through-barrier fills at that open, as today).

**Regression invariant.** `RunConfig.legacy_5min()` + CSV input reproduces the pre-U2 `wfo_signals.csv` exactly.

---

## §3 Feature registry (U3)

```python
@feature_group(name: str, required: bool = False, intraday_only: bool = False, per_fold: bool = False)
def group(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame: ...
```
- Returns columns prefixed `{name}__` indexed like `df`; NaN during warm-up.
- `per_fold=True` groups (e.g. fracdiff) receive a `fit(train_df) -> state` / `transform(df, state)` pair instead.
- `build_features(df, cfg, groups)` raises if `wavelet_core` is absent; skips `intraday_only` groups on 1Day with a log line.
- **Invariant:** only stationary transforms (returns, ratios, z-scores, fracdiff); raw price levels are rejected
  by a check that the column is not ≥ 0.99 correlated with `close` on the fixture.
- Feature cache key: `(symbol, tf, group, adjustment, source-hash of the group function)`.

**Clustered MDA** (López de Prado 2020, ch.6): cluster features by `1 − |ρ|` (ONC or hierarchical with silhouette),
permute cluster-wise, score = drop in purged-CV neg-log-loss. Run on the training window only; keep clusters with
mean importance > 0 at 1 s.e.

---

## §4 Primary and model protocols (U4, U6)

```python
class Primary(Protocol):
    name: str

    def fit(self, df, feats, events, labels, weights, cfg) -> "Primary": ...  # no-op for fixed rules
    def side(self, df, feats, events, cfg) -> pd.Series: ...  # {-1,+1}, index = events
```
Invariant: `side` at event t uses only rows ≤ t (causality test §8).

```python
class ZooModel(Protocol):
    name: str

    def fit(self, X, y, sample_weight, cv: PurgedKFold) -> "ZooModel": ...  # HP search + calibration inside
    def predict_proba(self, X) -> np.ndarray: ...  # calibrated P(y=1)
```
Model defaults: `logit_l1/l2` (StandardScaler fit inside `fit`), `rf_ldp` (`n_estimators=500`,
`max_features="sqrt"`, `class_weight="balanced_subsample"`, `max_samples=avg_uniqueness`, `min_weight_fraction_leaf=0.05`),
`extra_trees`, `xgb` (current params), `lightgbm`. Calibration: isotonic vs sigmoid chosen by purged-CV Brier.

---

## §5 Purging, embargo and cross-validation (U2, U5)

Each sample i has span `[t0_i, t1_i]` = `[event bar, exit bar]`.
- **Purge:** drop train sample i if its span overlaps any test span.
- **Embargo:** additionally drop train samples with `t0_i ∈ (max test t1, max test t1 + h]`, `h = ceil(embargo_pct · N_bars)`;
  default `embargo_pct = 0.01`, or `embargo_days = 1` for WFO splits.
- **PurgedKFold:** contiguous, unshuffled folds; purge + embargo per test fold.
- **CPCV(N, k):** N contiguous groups, all C(N,k) test combinations; number of backtest paths
  φ = C(N,k)·k/N = C(N−1,k−1); each group is a test group in exactly φ splits. Default N=10, k=2 → 45 splits, 9 paths.
- **Selection metric:** sample-weighted neg-log-loss (default) or Brier. AUC/F1/accuracy reported only.

---

## §6 PWFO definitions and statistics (U5, U9)

**Windows.** In trading days on the market calendar. For combo (IS, OOS): window w has
IS = [s_w, s_w+IS), OOS = [s_w+IS+embargo, s_w+IS+embargo+OOS), s_{w+1} = s_w + OOS (rolling; `expanding=True` fixes s_w = start).
Retraining cadence = OOS length. Default grid IS ∈ {63, 126, 252, 504}, OOS ∈ {5, 10, 21, 63}.
Inside each IS window, the existing train/val split and inner purged CV apply.

**Per-combo statistics.**
- `n_oos_windows` (flag `< 50` as statistically weak, per Meyers).
- `WFE = annualized OOS return / annualized IS return` (IS return = in-sample fit performance of the same
  model on its IS window's val+train events). Reported NaN when IS return ≤ 0, with the count of such windows.
  Also `WFE_sharpe` (ratio of Sharpes).
- `pct_profitable_oos`, `oos_sharpe`, `oos_sortino`, `max_dd`, `IS↔OOS Spearman` across windows, trades, turnover.
- PBO over the combo × window performance matrix (CSCV, S=16 blocks).

**Nested selection (walk-forward of the walk-forward).** Each combo yields a causal daily OOS return stream.
Every `SELECT_EVERY = 10` trading days at decision date d, pick the combo with the highest trailing
`SELECT_LOOKBACK = 126`-day Sharpe computed from returns strictly before d (ties → higher WFE_sharpe → shorter IS).
The PWFO equity curve = the chosen combo's returns over [d, d+SELECT_EVERY). Before every combo has a full
lookback, use the default combo (IS=252, OOS=10); that period is labeled burn-in and excluded from headline stats.

**Probabilistic Sharpe** PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ),
γ₄ = raw (non-excess) kurtosis, SR per-period (not annualized).
**Deflated Sharpe** DSR = PSR(SR₀) with SR₀ = √V[SR_n]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))), γ ≈ 0.5772,
N = number of trials from the ledger (§9), V[SR_n] = variance of trial Sharpes.

---

## §7 Bet sizing and risk (U7, U8)

p = calibrated meta-probability, τ = `META_THRESH`. All sizers return m ∈ [0,1], m = 0 if p < τ; sign from the primary side.
- `fixed`: m = 1.
- `linear`: m = clip((p − τ)/(1 − τ), 0, 1).
- `ldp_sigmoid`: z = (p − 1/2)/√(p(1−p)), m = max(0, 2Φ(z) − 1).
- `ecdf`: m = F̂_train(p), F̂ = ECDF of the train-window OOF meta-probabilities.
- `kelly_capped`: f* = p − (1−p)/b, b = mean win / mean loss of train-window OOF approved trades; m = clip(λ·f*, 0, 1), λ = 0.25.
- Post-processing: average over active bets (López de Prado 10.4), then discretize m ← round(m/step)·step, step = 0.1.

**Risk layer**, applied in order at decision time t (info ≤ close[t], fill at open[t+1]):
1. Vol target: notional = m · equity · min(1, σ_target / σ_hold,t), σ_hold = bar σ · √vertical_bars; σ_target default 0.5% per trade.
2. Caps: per-position ≤ 20% equity, per-symbol ≤ 25%, gross ≤ 100% (no leverage default), |net| ≤ 100%, max concurrent 10 — scale down pro-rata.
3. Drawdown throttle: multiplier 1.0 / 0.5 / 0.0 at DD < 10% / 10–20% / > 20% (resets at new high).
4. Daily loss gate: if session P&L ≤ −2% equity, flatten at next open and block entries for the rest of that session (next day for 1Day).

**Costs:** every notional change pays `half_spread_sym + SLIPPAGE_PCT` on the traded notional; `half_spread_sym`
from the trailing (train-window) Corwin–Schultz estimate, floored at 0.5 bp; optional short borrow 0 bp default.

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

**Cell spec** (YAML): `symbols, timeframe, feature_groups, primary{name,params}, model{meta,primary}, sizer,
risk_profile, pwfo{is_grid, oos_grid, expanding}, seed, stage`.

**Ledger row:** `cell_hash, stage, status, git_sha, spec_json, started_at, runtime_s, n_oos_events, n_trades,
meta_auc, meta_logloss, brier, ret_ann, vol_ann, sharpe, sortino, calmar, max_dd, turnover, sr_skew, sr_kurt,
n_obs, psr, wfe, n_oos_windows, error_path`. Stored as append-only `results/ledger.jsonl`. `n_trials` for DSR = count of `status="ok"` rows up to and including the stage.

---

## §10 Dependencies

Add: `alpaca-py`, `python-dotenv`, `lightgbm`, `joblib`, `scipy` (explicit). `pyyaml` becomes used (U10).
