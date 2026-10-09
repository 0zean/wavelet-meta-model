# PLAN2 — Hypothesis-family research program (U12 → U21)

Supersedes [PLAN.md](PLAN.md) for all forward work. PLAN.md remains the record of U1–U11 (complete, null
result). The reasons for the change are in [REVIEW.md](REVIEW.md); this file does not repeat them.
Interfaces, formulas and defaults live in [SPEC.md](SPEC.md) §11–§19 (added with this plan). Everything
in SPEC §1–§10 that §11 does not amend still holds.

## Thesis

U11 showed that the harness is sound and that a mechanism-free search over single-symbol technical
features has no edge after costs. The new thesis is:

> A retail Alpaca account can earn a risk-adjusted return above buy-and-hold by trading a small number of
> **economically motivated strategy families** (daily and intraday, nothing below 5Min), each tested
> **once as a family** with a pre-registered headline variant, a magnitude floor and a coherence check
> over its specification variants; machine learning enters only as a **conditioning overlay** on a family
> that has already passed; the true holdout is a **pre-registered forward paper-trading test**.

A null result for a family is a valid, reportable outcome. A null result for the whole program is too.

## What carries over, what is retired

Kept unchanged: the data layer (§1), `RunConfig` (§2), the feature registry and causality tests (§3, §8),
the primary and model protocols (§4), purge/embargo/CPCV/PBO/PSR/DSR (§5–§6), sizers and the risk layer
(§7), the runner, ledger and cell hashing (§9), `fastfracdiff`, the feature cache, `scripts/run_specs.ps1`.

Retired as defaults (still available behind a flag, for regression): nested PWFO selection as a gate
(replaced by combo averaging), per-window hyper-parameter search (frozen parameters), per-window
calibration cross-fitting (rolling calibration), `catboost` in screens, the `ecdf`/`kelly` refits, the
U11 A → B → C selection ladder, per-cell Holm/DSR verdicts on a one-year holdout, Corwin–Schultz spreads
as the cost model, `ml_xgb` as a research primary (kept for the legacy regression only), 1Min bars.

## Working protocol (unchanged from PLAN.md, with three additions)

1. One unit per branch `unit/NN-slug` off `main`; done-gate = `uv run pytest -q` green, `ruff` clean,
   every unit criterion demonstrated with command + output in the status note; one adversarial-reviewer
   pass per unit; BREAKING/SEVERE fixed, MINOR fixed or deferred here with a reason; stop and report.
2. **Pre-registration is a file, not a paragraph.** A family's spec (`families/<id>.yaml`, SPEC §17) is
   committed, with its git SHA recorded in the ledger, before any cell of that family runs. Amendments are
   new files (`families/<id>.v2.yaml`) with a stated reason; the runner refuses a family spec whose SHA is
   not in git history.
3. **Trial budget is enforced, not hoped for.** `TRIAL_BUDGET` in each family spec (default 12 variants
   + 2 overlay configurations); the runner refuses a cell that would exceed it. Program-wide cap: 8
   families, 112 counted trials (vs 1,402 in U11).
4. **No hashed code changes during a run** (as before). Runs are now short enough (hours) that this is
   no longer a development blocker.

Cross-cutting rules carried over: no network in tests; no silent fallbacks; causality tested, not
asserted; every experiment is a counted ledger trial; holdout (now the forward test) is locked.

## Data windows and holdout (SPEC §11)

- Development window: **2016-01-04 → 2025-09-30** (unchanged) for selection and headline tests.
- Quasi-holdout: **2025-10-01 → 2026-09-27**. Reported for every family as a robustness slice, never a
  gate. It is contaminated: the U11 holdout exposed the 14 symbols' buy-and-hold outcomes and 14 strategy
  outcomes on it, and new hypotheses were chosen afterwards. The report must say so.
- Forward test (the holdout that nobody can have seen): paper trading on Alpaca from each family's
  **freeze date**, minimum 12 months (U20). `HOLDOUT_START` moves to `2026-10-01`; research code may not
  read bars on or after it except through the forward-test scorer.

## Dependency graph

```
U12 harness speed ─┬─► U13 exogenous data + quotes cost model ─┬─► U15 state features ─┐
                   ├─► U14 samplers, exits, intraday vol (ISOM) ┴─► U16 mechanism primaries ─┼─► U17 family-test tooling
                   │                                                                           │
                   └───────────────────────────────────────────────────────────────────────────┘
U17 ─► U18 Phase 1: family dev tests (no ML) ─► U19 Phase 2: ML overlay ─► U20 forward test (paper trading)
U13 ─► U21 VRP / VIX term-structure family (optional, gated on VX data)
```

U13, U14 can run in parallel after U12; U15 needs U13; U16 needs U14 (and U13 for the calendar
primaries); U17 needs U14–U16 interfaces; U18 needs all of U12–U17.

---

## Hypothesis families

Each family states: the mechanism; the core claim (what the signature is and in which instruments); the
**headline variant** (the a priori natural specification, the only one that is tested with multiplicity
control); auxiliary predictions; specification variants along Beckman's three axes (state, response,
sample) that are *reported*, not selected among; expected magnitude; evidence and caveats; costs; data;
trial budget. Formal definitions of the primaries are in SPEC §16; of the family test in SPEC §17.

Instruments. Core ETFs (cached): SPY, QQQ, IWM, DIA, XLE, XLF, XLK, XLV, TLT, GLD. Added in U13 for F2:
IEF, LQD, HYG, DBC, USO, UUP, EFA, EEM, VNQ, SLV (5Min from 2016 where listed). Single stocks (AAPL, MSFT,
NVDA, AMZN, GOOGL, META, JPM, XOM, UNH) are used only where a family's mechanism says so (F7, F8 samples).

### F1 — Volatility-managed exposure (daily)

- **Mechanism.** Variance is highly predictable at a one-month horizon; the equity premium is not
  proportionally higher when variance is high (Moreira & Muir 2017). Scaling exposure by σ*/σ̂ raises
  the Sharpe ratio and cuts drawdowns; with the retail leverage cap of 1.0 it works by de-risking.
- **Core claim.** On SPY, QQQ, IWM, DIA (and TLT, GLD as a non-equity check): Sharpe(vol-managed) >
  Sharpe(B&H) and max drawdown < B&H, with annual return ≥ 0.8 × B&H. Signature strongest in 2018Q4,
  2020Q1, 2022.
- **Headline variant.** σ̂ = 21-session realized vol of close-to-close returns; σ* = 15 % annualized;
  exposure = min(1, σ*/σ̂); rebalance when the target exposure moves by ≥ 10 points; fills at the next
  open; long-only, cash earns nothing (conservative).
- **Auxiliary predictions.** (a) Exposure is lowest in the worst drawdown months. (b) The gain comes
  from the left tail (lower kurtosis), not from timing the mean. (c) Works on QQQ and IWM too; weaker or
  absent on TLT and GLD (variance premium is an equity phenomenon).
- **Variants (≤ 10).** State: σ̂ ∈ {21d RV, 63d RV, EWM span 33, VIX}; σ* ∈ {10 %, 15 %, 20 %}.
  Response: Sharpe, max DD, Calmar, lower partial moment. Sample: equity ETFs vs TLT/GLD; 2016–2019 vs
  2020–2025.
- **Expected magnitude.** Sharpe +0.1 to +0.3 over B&H, DD −20 to −40 %, return 0.8–1.0 × B&H.
- **Evidence and caveats.** Strong academic and practitioner literature; results depend on the
  leverage cap (without leverage the return falls), and 2016–2025 is a bull sample where B&H is hard to
  beat on return.
- **Costs.** ~1–3 fills/month; negligible.
- **Data.** Cached bars; VIX from U13 for one variant.
- **Budget.** 1 headline + 9 variants = 10 trials.

### F2 — Time-series momentum across ETFs (daily/weekly)

- **Mechanism.** Slow information diffusion, under-reaction and herding create 1–12 month return
  continuation in asset classes; vol-scaling equalizes risk across assets (Moskowitz, Ooi & Pedersen
  2012; Hurst, Ooi & Pedersen 2017).
- **Core claim.** An equal-risk basket of 15–20 ETFs across equities, bonds, commodities and currencies,
  long (short) each asset whose trailing 12-month excess return is positive (negative), vol-targeted at
  10 % per asset, rebalanced weekly, has Sharpe 0.5–0.8 and positive return in 2020Q1 and 2022 when
  equities fall ("crisis alpha").
- **Headline variant.** Signal = sign of the 252-session return; vol scaling with 63-session RV;
  weekly rebalance (Monday open); long/short with shorts allowed only in ETFs Alpaca marks shortable;
  gross leverage cap 1.0.
- **Auxiliary predictions.** (a) Long-only version keeps most of the Sharpe in 2016–2025 (bull sample).
  (b) Returns are positively skewed. (c) Correlation to SPY near zero over the full window, negative in
  drawdowns.
- **Variants (≤ 12).** State: lookback ∈ {63, 126, 252, blended}; the MODWT smooth slope (J = 6, 7) as
  the trend estimator (re-uses `wavelet_core`'s machinery). Response: Sharpe, crisis-period return, skew.
  Sample: long-only vs long/short; equity-only vs full basket; weekly vs monthly rebalance.
- **Expected magnitude.** Sharpe 0.4–0.8, return 4–8 %/yr at 10 % vol, correlation to SPY ≈ 0.
- **Evidence and caveats.** Decades of cross-asset evidence; 2016–2025 contains the 2018, 2020 and 2022
  trend reversals, which hurt short lookbacks. Short ETF borrow is charged at `borrow_bps`.
- **Costs.** Weekly rebalance of a 15–20 asset basket, turnover ~2–4×/yr; low.
- **Data.** U13 adds the extra ETFs; no exogenous data.
- **Budget.** 12 trials.

### F3 — Overnight vs intraday return (daily, SPY and QQQ only)

- **Mechanism.** Almost all of the index return accrues close-to-open; intraday is ≈ 0 or negative
  (Lou, Polk & Skouras 2019; Knuteson 2022). Candidate drivers: overnight information arrival with
  intraday liquidity provision, retail/institutional clientele differences, close-auction pressure.
- **Core claim.** Holding SPY/QQQ only overnight (buy MOC, sell MOO) has a higher Sharpe than B&H and
  B&H's return, net of two auction fills a day; the intraday leg has mean ≈ 0.
- **Headline variant.** Long overnight every session, flat intraday; fills at the close and open
  auctions (SPEC §19 cost model: half-spread, no slippage premium beyond it).
- **Auxiliary predictions.** (a) The effect is larger after down days and in high-VIX regimes.
  (b) The intraday leg's mean is not significantly positive. (c) Fails in IWM/DIA only through cost.
- **Variants (≤ 8).** State: all days vs VIX above/below median vs prior-day return sign. Response:
  Sharpe, mean per leg with Newey–West SE. Sample: SPY, QQQ; 2016–2019 vs 2020–2025.
- **Expected magnitude.** Gross overnight Sharpe ≈ 1.2–1.5 vs B&H 0.9–1.2 on this sample; net of
  ~1.5 %/yr of fills, return ≈ 0.8–1.0 × B&H at ~0.7 × its vol.
- **Evidence and caveats.** Robust for decades; costs decide it; auction fills on SPY/QQQ are ~0.3 bp.
- **Costs.** 504 fills/yr: the family lives or dies on the quotes-based cost model (U13).
- **Data.** Cached 5Min bars give the session open and close; VIX for one variant.
- **Budget.** 8 trials.

### F4 — Calendar and event drift (daily, SPY/QQQ/IWM/TLT)

- **Mechanism.** Scheduled flows and scheduled information: pre-FOMC announcement drift (Lucca & Moench
  2015: the 24 hours before the statement carry a large share of the equity premium), turn-of-month
  pension and payroll flows (Etula, Rinne, Suominen & Vaittinen 2020), options-expiration week, month-end
  duration extension in Treasuries.
- **Core claim.** A calendar overlay that is long SPY from the close before an FOMC day to 14:00 on the
  day, long over the last session and first three sessions of each month, and flat otherwise, has
  positive net return with Sharpe above 0.8 on ~40 trades/yr, and a much higher return per day held than
  B&H.
- **Headline variant.** FOMC window + turn-of-month window on SPY; TLT month-end window as a separate
  instrument leg; fills at the open and close auctions.
- **Auxiliary predictions.** (a) Return per day in the window > 3× the unconditional daily mean.
  (b) The FOMC leg's return accrues before 14:00, not after. (c) Effect weaker after 2016 (known), but
  still positive.
- **Variants (≤ 10).** Window definitions (FOMC t−1 close → t 14:00 vs t−1 close → t close; TOM
  −1/+3 vs −2/+2), OPEX-week leg on/off, CPI/NFP day leg on/off. Response: return per window day, hit
  rate. Sample: SPY vs QQQ vs IWM; TLT.
- **Expected magnitude.** 2–5 %/yr on ~15 % of days held; Sharpe 0.6–1.0 because exposure is sparse.
- **Evidence and caveats.** Pre-FOMC drift has weakened since 2016; turn-of-month is persistent but
  small; low trade counts mean wide intervals, so this family pools across instruments and windows.
- **Costs.** ~80 fills/yr; low.
- **Data.** U13 calendar (FOMC dates, CPI/NFP schedule, OPEX, month-ends).
- **Budget.** 10 trials.

### F5 — Market intraday momentum and close-flow continuation (intraday, 5Min bars)

- **Mechanism.** Two related flows push the last 30 minutes in the direction of the day: (i) gamma
  hedging by option market makers and leveraged-ETF rebalancing, both of which buy into rises and sell
  into falls near the close (Baltussen, Da, Lammers & Martens 2021; Cheng & Madhavan 2009); (ii)
  late-informed traders who act on the morning's information (Gao, Han, Li & Zhou 2018: the first
  half-hour return predicts the last half-hour).
- **Core claim.** On SPY, QQQ, IWM, DIA: the last-30-minute return has the sign of the return from the
  open to 15:30, with a larger effect on high-volatility days, on macro-announcement days and when the
  day's move is large; an overlay long or short from 15:30 to the close has positive net return.
- **Headline variant.** Entry at 15:30 (fill at the 15:30 bar's open, i.e. the first print after the
  decision at the 15:25 bar's close), direction = sign(open → 15:25 close return), exit MOC; trade only
  when |open-to-entry return| > 0.5 × the day's expected σ (SPEC §14 time-of-day σ); instruments
  SPY, QQQ, IWM, DIA, equal risk.
- **Auxiliary predictions.** (a) The first 30 minutes alone also predicts (weaker). (b) The effect
  reverts over the next 1–3 days (hedging demand, not information). (c) Larger on FOMC/CPI days and
  when VIX > 20. (d) Absent on sector ETFs and single stocks (index-level flows).
- **Variants (≤ 12).** Predictor ∈ {first-30 return, open-to-entry return}; entry ∈ {15:00, 15:30,
  15:45}; threshold ∈ {0, 0.5σ, 1σ}. State splits (reported): VIX tercile, macro day, |day move|
  tercile, gamma-proxy sign (2024–2026 only, SPEC §15). Response: last-leg return, hit rate, path
  monotonicity. Sample: index ETFs vs sector ETFs.
- **Expected magnitude.** Gross 4–7 %/yr on the index ETFs (literature 6 %/yr gross), net 2–4 %/yr
  after ~500 fills/yr at 0.3–0.5 bp; Sharpe 0.6–1.0 on an overlay with zero overnight exposure.
- **Evidence and caveats.** Documented 1993–2013 with R² ≈ 2–3.5 %; weaker after 2013 in some
  replications, stronger in high-vol regimes; the gamma mechanism is the most credible driver, so the
  state splits are the real test. This is the family where the ISOM-style time-of-day volatility
  structure (U14) is load-bearing: the threshold and the vol scaling are in time-of-day units.
- **Costs.** Two fills a day on SPY/QQQ: the quotes-based cost model decides; the magnitude floor
  (SPEC §17: gross edge per unit turnover ≥ 3× cost) is binding here.
- **Data.** Cached 5Min bars; VIX, calendar (U13); gamma proxy optional (U15).
- **Budget.** 12 trials.

### F6 — folded into F5 as the "open-to-entry" predictor variants (same signature; see §17 coherence)

### F7 — Overnight gap continuation or reversal in the first hour (intraday, 5Min)

- **Mechanism.** Large overnight gaps mix information (continues) with liquidity-driven overnight
  moves and opening-auction imbalances (revert). On days without scheduled news the liquidity component
  dominates and part of the gap reverts within the first hour; on news days (earnings for stocks,
  macro releases for indices) the information component dominates and the gap continues.
- **Core claim.** On SPY/QQQ/IWM (index leg) and the 9 cached single stocks (stock leg): conditional
  on |gap| > 1σ_overnight and no scheduled news, the 09:35 → 10:30 return has the opposite sign to the
  gap; conditional on scheduled news it has the same sign. Pooled across instruments the two conditional
  means are significantly different from zero and from each other.
- **Headline variant.** Index leg only (fewer confounders): fade gaps > 1σ on non-macro days, enter at
  the 09:35 bar open, exit at 10:30, size by 1/σ; the stock leg and the news-continuation leg are
  auxiliary.
- **Auxiliary predictions.** (a) Reversal share rises with |gap|. (b) Reversal is larger when the
  overnight move was against the prior-day trend. (c) Stock-leg continuation on earnings days.
- **Variants (≤ 10).** Gap threshold ∈ {0.5σ, 1σ, 1.5σ}; exit ∈ {10:00, 10:30, hysteresis β}; state
  splits: VIX tercile, day-of-week, news flag. Sample: index vs stocks.
- **Expected magnitude.** Small per trade (5–15 bp gross), ~60–100 trades/yr per instrument; net
  1–3 %/yr on the index leg; Sharpe 0.4–0.8.
- **Evidence and caveats.** Mostly practitioner evidence; academic support for opening-auction
  imbalance reversal is at minutes, so the 5Min bar and the 09:35 entry (skipping the auction) is a
  deliberate, conservative choice. Thin evidence → modest budget and a strict floor.
- **Costs.** Two fills per trade; stocks have 1–2 bp half-spreads, which is why the headline is index-only.
- **Data.** Cached bars; calendar and earnings dates (U13).
- **Budget.** 10 trials.

### F8 — Macro-announcement reaction continuation (intraday, 5Min; exploratory)

- **Mechanism.** After scheduled macro releases (FOMC 14:00; CPI, NFP 08:30), prices under-react for
  one to two hours as information is absorbed across asset classes (equities, Treasuries, gold, dollar
  react sequentially; dealers' inventory constraints slow the adjustment).
- **Core claim.** The 15-minute post-release return in SPY, QQQ, IWM, TLT, GLD predicts the sign of the
  following 90-minute return on release days; pooled across instruments and releases (≈ 400 events over
  the dev window) the continuation is positive and significant.
- **Headline variant.** FOMC: observe 14:00–14:15, enter 14:15 in that direction, exit 15:45. CPI/NFP:
  observe 09:30–09:45 (first RTH 15 minutes, since 08:30 is pre-market), enter 09:45, exit 11:15.
  Instruments SPY, TLT (the two with the cleanest macro loading); QQQ, IWM, GLD auxiliary.
- **Auxiliary predictions.** (a) Continuation larger when the 15-minute reaction is large relative to
  the day's σ. (b) TLT and SPY reactions have consistent signs on CPI days (inflation surprise).
- **Variants (≤ 8).** Observation window 15 vs 30 min; hold 60 vs 90 min; instruments.
- **Expected magnitude.** 10–20 bp gross per event on ~40 events/yr → 4–8 %/yr on event days, Sharpe
  on the sparse stream 0.5–0.9 if real; uncertain.
- **Evidence and caveats.** Mixed; some studies find the reaction is complete within minutes.
  Treated as exploratory: pre-registered, budgeted, and a null is expected to be likely.
- **Costs.** ~80 fills/yr; low.
- **Data.** FOMC, CPI, NFP dates (U13).
- **Budget.** 8 trials.

### F9 — Volatility risk premium via VIX term structure (daily; optional, U21)

- **Mechanism.** Implied volatility exceeds realized on average and the VIX futures curve is in contango
  most of the time; a short-volatility position harvests the roll and the premium but carries crash risk.
- **Core claim.** Short VIX exposure (via SVXY, −0.5× since 2018) when VX1/VIX > 1.05 and VIX < 25, flat
  otherwise, has Sharpe 0.6–1.0 with a hard stop on backwardation.
- **Headline, variants, budget:** defined in U21 after the VX data unit confirms availability; the
  risk layer (daily loss gate, drawdown kill switch) is mandatory. Budget 8 trials.

### F10 — Weekly reversal in sector ETFs (daily; optional, low priority)

- Short-term reversal at the 1-week horizon in XLE, XLF, XLK, XLV and siblings; a rank-based long/short
  of last week's losers vs winners, held one week. Documented but modest; run only if the program budget
  has room after F1–F8. Budget 6 trials.

### F11 — Re-test of the U11 intraday primaries under deseasonalized volatility (optional, closure)

- The ISOM-style time-of-day σ (U14) changes CUSUM event timing and barrier widths at 15Min–1Hour.
  A single pre-registered re-run of the best-evidenced U11 cell family (donchian_breakout / rf_ldp_fast
  on QQQ, SPY, XLK at 30Min and 1Hour, 4 cells) with `VOL_PROFILE="tod"` closes the question of whether
  the U11 null was partly a labeling artefact. Expected answer: no. Budget 4 trials.

---

## Units of work

Cost estimates are for the i9 PC (`LOKY_MAX_CPU_COUNT=1`, `scripts/run_specs.ps1`) unless stated.
"Session" = one working session of implementation.

### U12 — Harness speed and simplification

**Goal.** Make a family test cost hours, not days, without changing any result the science depends on.

**Scope** (SPEC §11.2).
- `models/zoo.py`: `ZOO_FIXED_PARAMS: dict[str, dict]` in `RunConfig`; when a model has an entry its
  grid search is skipped (one fit); default entries from the U11 evidence: `rf_ldp_fast`
  `{max_features: 1}`, `logit_l2` `{C: 0.1}`, `xgb` `{max_depth: 2}`, `lightgbm` `{num_leaves: 7}`.
  Keep the grid path behind `ZOO_FIXED_PARAMS = {}` for regression.
- `CALIBRATION ∈ {"crossfit", "rolling", "none"}` (default `rolling`): `rolling` fits the Platt map on the
  previous windows' OOS (raw p, outcome) pairs within the current rolling IS span (causal, no refit);
  the first window of a run uses `crossfit`. `none` = raw probabilities (for screening).
- `models/meta_model.oof_meta_prob`: when the fitted meta model is a `_Tuned` with `oof_raw_`, return
  `cal.predict(oof_raw_)` instead of refitting; the refit path stays behind `OOF_META="refit"`.
- `wfo/pwfo.py`: `PWFO_COMBINE ∈ {"average", "nested"}` (default `average`): the PWFO stream is the
  equal-weight mean of the run combos' daily returns on their common span (no burn-in, no selection);
  `stats["pbo"]` is still computed across combos and reported; `nested` keeps U9/U11 behaviour.
  Default grid: intraday `PWFO_IS_GRID=(504, 756)`, daily `(1260, 1512)`, `PWFO_OOS_GRID=(21,)`.
- `experiments/runner.py`: flat parallelism — a PWFO cell's combos are submitted to the *same* loky pool
  as the cells (task = (cell, combo)); the parent assembles `run_pwfo`'s post-processing. No nested pools.
- `features/triple_barrier_labels.cusum_events` and `features/groups.cusum_state`: vectorised or numba
  (numba added as a dependency only if the pure-numpy reset recursion is not within 2× of it).
- `scripts/run_specs.ps1`: `-Jobs` defaults to 24 (P+E cores), inner jobs removed.

**Done when.**
- [ ] Parity: the four U11 Stage C cells listed in `tests/fixtures/u12_parity.json` re-run with
      `ZOO_FIXED_PARAMS` equal to the parameters U11 actually chose in ≥ 95 % of windows, `CALIBRATION=
      crossfit`, `PWFO_COMBINE=nested`, `OOF_META=refit` reproduce their U11 `daily_returns.csv` exactly
      (hash equality) — proves the refactor is behaviour-preserving when the new switches are off.
- [ ] Cost table (recorded in the status note): per-window fits and seconds for `logit_l2`,
      `rf_ldp_fast`, `xgb` at 30Min, 1Hour, 1Day with defaults on vs off; target ≥ 10× fewer fits per
      window and ≥ 8× fewer windows per PWFO cell.
- [ ] `PWFO_COMBINE=average` tested: equal to the mean of the combo columns; causal; PBO still reported;
      nested path bit-identical to U11.
- [ ] Flat pool: a scratch-ledger run of 2 PWFO cells × 2 combos with `--jobs 4` records 2 `ok` rows;
      parallel = serial (test).
- [ ] `cusum_events` vectorised version equals the loop version on 10 seeded series and on SPY 5Min.

**Reviewer focus.** Rolling calibration leaking the current window's OOS; averaging combos with
different OOS spans; the parity fixture actually exercising the old code paths.

**Cost.** 2 sessions. Depends on nothing.

### U13 — Exogenous data layer and quotes-based cost model

**Goal.** Point-in-time exogenous series and a cost model from real quotes, cached like bars.

**Scope** (SPEC §12, §19).
- `data/exo.py`: `load_series(source, name, start, end) -> pd.DataFrame` with columns `value`,
  `available_at` (tz-aware UTC instant from which the value may be used); cache
  `data/cache/exo/<source>/<name>.npz` + sidecar (same atomic/hash scheme as bars; `schema_version`).
  Sources:
  - `cboe`: VIX, VIX9D, VIX3M, VIX6M daily OHLC from
    `https://cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv` (confirmed for VIX3M;
    the unit verifies the other three and records the exact URLs in SPEC §12), `available_at` = session
    date 16:15 ET + 5 min. VX futures settlements per expiry from Cboe CFE historical files → a
    continuous `VX1`, `VX2` series and `VX1/VIX`, `VX2/VX1` ratios (roll on expiry).
  - `fred`: VIXCLS (cross-check), DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2 (HY OAS), DTWEXBGS (broad dollar);
    `available_at` = next business day 09:00 ET (FRED posting lag) unless the series' own release
    schedule is known.
  - `calendar`: FOMC statement days and times (2016–2027, from the Fed's published schedule; 14:00 ET),
    CPI and NFP release days (BLS schedules; 08:30 ET), monthly OPEX (third Friday), month-end and
    turn-of-month flags, early closes (already in the exchange calendar). Stored as a table, not fetched
    at run time; a checked-in CSV under `data/calendar/` with a provenance note.
  - `earnings`: earnings dates for the 9 cached stocks 2016–2026 (Alpaca corporate actions / news as
    first source; a checked-in CSV if the API is incomplete), with before-open / after-close flag.
  - `alpaca_options`: daily snapshot of the SPY/QQQ chain (IV, Greeks) from 2024-02 on (free indicative
    feed; OPRA requires the paid plan); stored per day as a compact table; used only by the optional
    gamma proxy (U15). Open interest is not in the snapshot schema as far as the docs show: the unit
    confirms and, if absent, the gamma proxy falls back to a VIX/skew proxy.
  - `quotes`: a quotes sampler — for each cached symbol, NBBO at 5-minute marks for one trading week per
    quarter 2016–2026 (Alpaca historical quotes endpoint), reduced to a per-symbol table of median
    half-spread (bp) by time-of-day bin and year, plus the open/close auction half-spread proxies.
- `risk/costs.py`: `half_spread_quotes(index, table)` returns the half-spread for each bar from the
  table (nearest year, time-of-day bin); profile `standard` gains `spread="quotes"`; `cs` kept for
  regression. `COST_MODEL ∈ {"slippage", "cs", "quotes"}` on `RunConfig`.
- Holdout guard: `load_series` raises on `end > HOLDOUT_START` unless `allow_holdout`; `available_at`
  is enforced by the feature builder (U15), not here.
- `data/fetch.py`: `DEFAULT_UNIVERSE` += IEF, LQD, HYG, DBC, USO, UUP, EFA, EEM, VNQ, SLV, SVXY; cache
  them (5Min, 2016 →, ~1 min each).

**Done when.**
- [ ] Unit tests (mocked HTTP): CSV parsing, `available_at` rules, cache round-trip, holdout guard,
      calendar table integrity (every FOMC day is an exchange session; 8 per year), VX roll logic.
- [ ] Live smoke (recorded): the four CBOE series fetched and plotted against FRED VIXCLS (max |Δ| < 0.01
      on common days); VX1/VIX ratio has median > 1 (contango) over 2016–2025.
- [ ] Quotes table for SPY, QQQ, IWM, AAPL, NVDA with the time-of-day profile; SPY median half-spread at
      10:00–15:30 reported in bp and compared with Corwin–Schultz (expected ≈ 0.3 bp vs ≈ 1.5 bp).
- [ ] `COST_MODEL=quotes` produces fills whose cost equals half-spread + `SLIPPAGE_PCT` (hand test).
- [ ] New ETFs cached with the quality report; listing-date gaps logged, not filled.

**Reviewer focus.** Point-in-time violations (a daily value used on its own session), survivorship in
the added ETFs (SVXY's 2018 leverage change), quotes sampling bias (a week per quarter).

**Cost.** 1–2 sessions; data fetch < 1 h. Depends on nothing (can start with U12).

### U14 — Event samplers, exit models and intraday volatility structure (ISOM)

**Goal.** Let a primary define *when* it trades (scheduled times, directional-change events, CUSUM) and
*how* it exits (barriers, scheduled time, hysteresis), and make intraday σ time-of-day aware.

**Scope** (SPEC §13, §14).
- `features/vol_profile.py`: per-fold state `VolProfile.fit(train_df)` = multiplicative time-of-day
  factor s(b) for each bar slot b of the session (13 slots at 30Min … 78 at 5Min; the 1Hour stub is its
  own slot): s(b) = median over train sessions of |r_b| / median over all slots, smoothed with a 3-slot
  moving median; `transform` returns the deseasonalized return series r_b / s(b).
  `bar_volatility(close, span, profile=None)`: with a profile, σ_t = EWM σ of deseasonalized returns ×
  s(b_t) (so barriers and CUSUM thresholds in slot b are scaled to that slot's typical move).
  `RunConfig.VOL_PROFILE ∈ {"none", "tod"}` (default `none` for regression; `tod` for every intraday
  family). ISOM diagnostic: `isom_counts(events, n_slots)` = CUSUM/DC events per slot over the train
  window (the paper's ISOM), reported per fold in `wfo_run.json`; `iaom` = counts / sessions.
- `features/events.py` (moved from `triple_barrier_labels.sample_events`): `EVENT_SAMPLER ∈ {"cusum",
  "dc", "schedule"}`, `EVENT_PARAMS`:
  - `cusum`: as today (threshold = `CUSUM_MULT` × σ_t, now profile-aware).
  - `dc`: directional-change intrinsic time (Guillaume et al. 1997): an event when log price moves δ
    from the last extreme in the opposite direction; δ = `DC_MULT` × σ_session; the DC *overshoot* is
    exposed as a feature.
  - `schedule`: events at fixed session-relative times (`entry_times`, e.g. `["15:30"]`), restricted by
    a calendar predicate (`days ∈ {"all", "fomc", "cpi_nfp", "tom", "non_macro"}`) and an optional gate
    expression on session features (e.g. `abs(open_to_now) > 0.5 * sigma_day`). The event bar is the
    bar whose close precedes the entry fill (decision at 15:25 close, fill at the 15:30 open).
- `features/exits.py` (generalises `barrier_exits`): `EXIT_MODEL ∈ {"triple_barrier", "time",
  "hysteresis"}`, `EXIT_PARAMS`:
  - `triple_barrier`: as today.
  - `time`: exit at a scheduled bar (`exit_time`, e.g. `"close"` → the session's last bar close, modelled
    as an MOC fill; `"10:30"` → that bar's open) or after `hold_bars`; gap-through rules not needed.
  - `hysteresis` (STTS, Toulson & Toulson): for primaries exposing a continuous `signal`, exit when the
    signal crosses −β (long) / +β (short), with a maximum hold; β < α where α is the entry threshold.
  All exit models return the same frame as `barrier_exits` (`t1, entry_pos, exit_pos, entry_px, exit_px,
  ret, label, barrier, width`), so labels, meta-labels, the backtest and `average_uniqueness` are
  unchanged downstream. Auction fills: `entry_px`/`exit_px` at the session open/close with the
  `COST_MODEL` half-spread; no further slippage.
- `features/groups.py` new group `session` (intraday only): overnight gap (log open / prior close in
  σ_day units), prior-session close-to-close and open-to-close returns, open-to-now return, first-30-min
  return (NaN before 10:00), time-to-close (minutes), expected activity at this slot (IAOM), relative
  volume (session cumulative volume / its time-of-day profile), DC overshoot. All causal; `available_at`
  not needed (bars only).
- `wfo/wfo_engine.prepare` / `fit_window`: per-fold fitting of `VolProfile` (like fracdiff, on train
  before the embargo); events and labels are recomputed per window when the sampler or σ depends on
  fold state (scheduled events do not; CUSUM/DC with `tod` do, so `Prepared` becomes per-window where
  needed, cached by `(fit_start, train_end)`).

**Done when.**
- [ ] Causality (SPEC §8) for `VolProfile`, `dc` and `schedule` events, every exit model, the `session`
      group; mutation checks (a one-bar look-ahead in the profile and in `time` exits must fail).
- [ ] Hand-computed tests: a scheduled 15:30 → MOC trade's entry/exit fills and return; hysteresis exit
      on a synthetic signal; DC events on a sawtooth; the profile's U-shape on synthetic data with a
      planted U-shaped σ; `tod` σ equals plain σ when the profile is flat.
- [ ] Regression: `VOL_PROFILE="none"`, `EVENT_SAMPLER="cusum"`, `EXIT_MODEL="triple_barrier"`
      reproduce the legacy CSV hash `402ef202…` and a U11 cell's `daily_returns.csv` exactly.
- [ ] Diagnostic (recorded): SPY 5Min 2016–2025 ISOM event counts by 30-minute slot and the fitted s(b):
      expected U-shape, open slot ≈ 2–3× the lunch slot; the share of CUSUM events in the first hour
      with and without `tod`.

**Reviewer focus.** Scheduled events whose decision bar is after the fill; MOC fills using the close
of a bar that is not the session's last; profile fit on embargo bars; DC events emitted at the extreme
rather than at the confirmation bar.

**Cost.** 2 sessions. Depends on U12 (engine structure).

### U15 — State and context feature groups

**Goal.** Give primaries and the overlay the market-state inputs the review found missing.

**Scope** (SPEC §15).
- `vol_state` (`needs=("exo",)`): VIX, VIX9D/VIX, VIX3M/VIX, VX1/VIX, VX2/VX1, 21-day RV/VIX (realized
  vs implied), GARCH(1,1) conditional σ fitted per fold on train (`arch` dependency, or a 20-line
  numpy fit), the return standardized by it; all aligned by `available_at` (a value is used only on bars
  whose stamp is after it).
- `calendar_events` (`needs=("exo",)`): sessions to next / since last FOMC, CPI, NFP; FOMC-day,
  CPI/NFP-day, OPEX-week, turn-of-month, pre-holiday flags; minutes since the announcement on the day.
- `cross_asset`: enable in the runner — `experiments/runner.load_cell_data` passes `context["market"]`
  (SPY bars at the cell's timeframe) and, for sector ETFs and stocks, `context["sector"]` (the mapped
  sector ETF); the group adds the lagged market and sector returns at 1, 3 and 6 bars (the lead-lag
  family's features), beta-residual returns and the relative-strength spread.
- `rates_credit` (`needs=("exo",)`): DGS10 change, 2s10s, HY OAS change, broad dollar change (daily,
  lagged by `available_at`).
- `gamma_proxy` (optional, `needs=("exo",)`, 2024-02 →): from the daily SPY/QQQ snapshot, the
  put–call IV skew at 25Δ and the ratio of near-the-money gamma-weighted IV; if open interest is
  available, a crude dealer gamma exposure sign. Documented as experimental; used only in F5 state splits.
- `features/feature_builder.FeatureSet`: `context["exo"]` = a dict of series frames; the builder
  enforces `available_at` ≤ bar stamp for every exo column (a test plants a same-day value and must fail).

**Done when.**
- [ ] Causality tests for every new group (bars perturbed after c; exo series perturbed after c), plus
      the `available_at` enforcement test.
- [ ] `cross_asset` runs in a runner cell (smoke on a scratch ledger) with market and sector context.
- [ ] Stationarity guard passes for all new columns; NaN warm-ups documented.

**Reviewer focus.** Point-in-time alignment of daily exo values to intraday bars (VIX close at 16:15 ET
must not be visible at 15:30 the same day), GARCH fit leaking across the embargo.

**Cost.** 1 session. Depends on U13.

### U16 — Primaries with a mechanism

**Goal.** One registered primary per family, each a thin rule over the new samplers/exits/features.

**Scope** (SPEC §16). All are `RulePrimary` subclasses (no fit); each declares its default
`EVENT_SAMPLER`, `EXIT_MODEL`, allowed timeframes and whether it is long-only. The frame columns are
unchanged (`PRIMARY_COLUMNS`); `magnitude` carries the rule's size hint (used by the `rule_size` sizer).
- `vol_target` (F1; 1Day; schedule daily at the open; exit next rebalance): side +1, magnitude =
  min(1, σ*/σ̂); a new sizer `rule_size` passes magnitude through as the bet size.
- `tsmom` (F2; 1Day; schedule weekly): side = sign(trailing return or MODWT slope), magnitude =
  min(1, 10 %/σ̂_asset); long-only variant clips side ≥ 0.
- `overnight` (F3; 5Min; schedule entry at the last bar → MOC, exit at next open → MOO): side +1.
- `calendar_drift` (F4; 1Day/5Min; schedule by calendar predicate): side +1 in the window.
- `intraday_momentum` (F5; 5Min; schedule entry `15:30`, exit `close`, gate on |open-to-now| vs σ):
  side = sign(open-to-entry return) or sign(first-30 return) per `predictor`.
- `gap_fade` (F7; 5Min; schedule entry `09:35`, exit `10:30`/hysteresis; days `non_macro`): side =
  −sign(gap) when |gap| > k·σ_overnight; `gap_follow` variant for news days.
- `event_reaction` (F8; 5Min; schedule entry at announcement + 15 min, exit + 90 min; days `fomc` /
  `cpi_nfp`): side = sign(post-release return).
- `vix_carry` (F9; 1Day; U21).
- `weekly_reversal` (F10; 1Day; weekly schedule): side = −sign(last-week return), cross-sectional rank
  within the sector-ETF set.
- Primary diagnostics (`primaries/diagnostics.py`) extended with per-slot (time-of-day) precision and
  the ISOM counts.

**Done when.**
- [ ] Each primary: causality test, a synthetic sanity test (e.g. `intraday_momentum` goes long on a
      planted morning rally; `gap_fade` shorts a planted up-gap on a non-macro day and not on an FOMC
      day), and a hand-computed trade on real SPY bars with the fills checked against the 5Min OHLC.
- [ ] A WFO run with each primary completes on SPY (and TLT for `calendar_drift`/`event_reaction`) with
      the schema unchanged; the `rule_size` sizer reproduces `fixed` when magnitude ≡ 1.
- [ ] Per-slot diagnostics table for F5 and F7 on SPY (dev window) saved under `results/families/`.

**Reviewer focus.** Fill timing of scheduled entries (decision bar strictly before the fill bar),
MOC/MOO modelling, long-only flags respected by the backtest, calendar predicates off-by-one around
early closes.

**Cost.** 2 sessions. Depends on U13, U14.

### U17 — Family-test tooling and report

**Goal.** One pre-registered, pooled, magnitude-floored, coherence-checked test per family, with
multiplicity control across families and an honest report.

**Scope** (SPEC §17). `families/` package:
- `families/spec.py`: the family YAML schema (id, mechanism text, registered SHA, instruments,
  timeframe, headline primary + params, ≤ `TRIAL_BUDGET` variants as override dicts, state splits
  (reported), response measures, sample splits, benchmark, floors, test settings, overlay settings).
  Loading validates every variant's `RunConfig` and refuses a budget overrun.
- `families/run.py`: expands a family into runner cells (stage `F`), one per variant × instrument (an
  instrument set is one *portfolio* cell when the family is a basket: F2, F10), runs them through
  `experiments.runner` (signals cache, ledger rows, `n_trials` per variant = 1 regardless of instruments,
  because the test is pooled), and writes `results/families/<id>/`.
- `families/test.py`:
  - pooled stream per variant = equal-risk mean of the instruments' daily returns (or the portfolio
    cell's stream);
  - headline test: PSR(0) of the headline stream and the Ledoit–Wolf (2008) Sharpe-difference test vs
    the registered benchmark (buy-and-hold EW of the family's instruments) using a circular block
    bootstrap (21-session blocks, 2,000 resamples); alpha vs benchmark by OLS with Newey–West SE;
  - magnitude floors: annual net return ≥ `min_net_ret` (default 2 % for overlays, 0.8 × benchmark
    for long-only families); gross edge per unit turnover ≥ `min_edge_to_cost` × the quotes-based cost
    (default 3×); max drawdown ≤ `max_dd` if set;
  - coherence: the sign of the headline statistic is shared by ≥ 2/3 of the variants and the median
    variant also clears the floors (Beckman's "coherent signature"); the full specification curve is
    plotted (all variants, sorted, with CIs) and no variant is selected;
  - state/response/sample splits reported with CIs (no test);
  - quasi-holdout slice (2025-10 → 2026-09) reported with its contamination note;
  - shrinkage: per-instrument Sharpes James–Stein-shrunk toward the family mean, reported next to raw;
  - program-level: Holm over the families' headline p-values at α = 0.05; a family **passes** iff its
    Holm-adjusted p < 0.05, it clears every floor, and it is coherent.
- `families/report.py`: Markdown + HTML per family and a program summary (the funnel of families, not
  of cells); the trial accounting (budget used, N for the ledger); the registered-vs-run spec diff.
- Ledger: stage `F` (family dev), `G` (overlay), `H` (forward) added to `STAGES`; `TRIAL_BUDGET`
  enforcement in the runner (a family's counted trials ≤ budget; the program cap in
  `families/program.yaml`).

**Done when.**
- [ ] Tests: the Sharpe-difference test has the right size under the null (two independent noise
      streams: 5 % ± 1.5 % rejections over 1,000 simulations) and power on a planted 0.3 Sharpe gap;
      block bootstrap preserves autocorrelation (test on an AR(1)); floors and coherence on synthetic
      variants; Holm on a planted set; budget overrun refused; registered-SHA check refused on an
      uncommitted spec.
- [ ] End-to-end on synthetic bars with a planted overnight drift: F3's family test passes and a
      no-drift control fails.
- [ ] The report renders for a dry-run family (F1 on SPY only, scratch ledger).

**Reviewer focus.** Any path by which a variant's result influences the headline (selection creeping
back); the bootstrap's handling of overlapping holding periods; pooled streams that double-count an
instrument.

**Cost.** 2 sessions. Depends on U14–U16 interfaces (can start on the statistics in parallel).

### U18 — Phase 1: family development tests (no machine learning)

**Goal.** Run F1, F2, F3, F4, F5, F7, F8 (and F10, F11 if budget allows) once each, as registered.

**Scope.**
- Write and commit `families/F1.yaml` … `families/F8.yaml` (headline, variants, floors) *before* any run;
  the PLAN2 status note records each SHA.
- Order: F1, F3, F4 (cheap, daily) → F2 (basket) → F5, F7, F8 (intraday) → F10, F11 (optional).
- Each family's cells run with `COST_MODEL=quotes`, `RISK_PROFILE=standard` (vol target off for F1/F2,
  which size themselves; on for the overlays), `VOL_PROFILE=tod` for intraday, `PWFO_COMBINE=average`
  with the default 2-combo grid where a fitted component exists (for pure rules there is no fit and the
  "WFO" is a single pass).
- Pre-registered decision rule (SPEC §17): a family passes on the Holm-adjusted headline, floors and
  coherence; passing families go to U19; failing families are reported and closed (no re-specification
  on the dev window).
- Expected outcome stated before running: F1 and F3 likely pass on Sharpe but F3 may fail the cost
  floor; F2 passes or fails on the 2016–2025 sample's trend reversals; F4, F5 marginal; F7, F8 likely
  null.

**Done when.**
- [ ] Every family spec committed before its first ledger row (checked by the runner).
- [ ] All families run; `results/families/program_summary.md` published as an artifact with the funnel,
      per-family verdicts, specification curves, quasi-holdout slices, trial accounting (≤ 112).
- [ ] Status note lists, per family: headline Sharpe and CI, Sharpe vs benchmark with p, floors, coherence
      share, verdict.

**Reviewer focus.** Post-hoc changes to any spec, floors or benchmark; quasi-holdout slipping into a
decision; cost model under-estimation for the two-fills-a-day families.

**Cost.** Compute: F1/F3/F4 minutes; F2 < 1 h; F5/F7/F8 ≈ 1–3 h each (5Min bars, 4–14 instruments, no
model fits). 1 session of orchestration.

### U19 — Phase 2: meta-labeling overlay on passing families

**Goal.** Answer one question per passing family: does a meta-model that decides which of the family's
trades to take (or how much) raise the family's pooled Sharpe net of the exposure it gives up?

**Scope** (SPEC §17.4).
- Overlay cell = the family's headline primary + `META_MODEL ∈ {logit_l2, rf_ldp_fast}` (frozen
  parameters), `META_TRAIN=oof`, `SIZER ∈ {linear, rule_size}`, features = `wavelet_core`, `session`,
  `vol_state`, `calendar_events`, `cross_asset` (+ `gamma_proxy` for F5 where available), two averaged
  PWFO combos; ≤ 2 configurations per family (stage `G`).
- Test: Ledoit–Wolf Sharpe-difference between the overlay stream and the family's headline stream
  (paired, block bootstrap), magnitude floor on the overlay's net return (≥ the headline's), Holm over
  the overlay tests; the overlay is adopted only if it passes, otherwise the family proceeds without it.
- Procedure bootstrap for the overlay (the one place where model selection exists): the two
  configurations' selection is re-run on 100 block-bootstrapped label sets; the headline overlay's
  advantage must exceed the 95th percentile of the bootstrapped advantages.

**Done when.**
- [ ] Overlay specs committed before running; ≤ 2 trials per family.
- [ ] Report: per family, overlay vs headline Sharpe with CI, exposure, turnover, calibration curve of
      the meta-model, adopted / not adopted.

**Reviewer focus.** Meta-model fit rows overlapping scheduled trades (purging with `time` exits);
overlay comparisons on different day sets.

**Cost.** Compute ≈ 1–2 h per family after U12. 1 session.

### U20 — Forward test: paper trading loop, monitor, registry

**Goal.** Trade the frozen families on an Alpaca paper account from a registered freeze date and score
them exactly as a holdout, with a kill switch.

**Scope** (SPEC §18).
- `families/registry.yaml`: for each adopted family, the exact cell hash(es), code hash, freeze date,
  registered expectation (dev Sharpe and CI, benchmark), verdict rule (PSR(0) on forward days ≥ 0.95
  after ≥ 252 sessions, Holm across the forward families; "consistent with registration" if the forward
  Sharpe lies inside the registered CI).
- `live/` package: `signals.py` (recompute today's events and sides from cached bars + an intraday
  top-up through Alpaca's latest-bars endpoint, using the frozen code hash — the runner refuses a
  different hash), `orders.py` (MOO/MOC via `time_in_force` `opg`/`cls`, limit-at-touch for intraday
  entries, bracket exits for `time` exits), `reconcile.py` (fills vs modelled fills; slippage ledger),
  `monitor.py` (BOCPD on the daily P&L stream, Adams & MacKay 2007 with a Student-t predictive; hazard
  from the registered expected run length; erosion and shock triggers → flatten and alert), `score.py`
  (stage `H` ledger rows from forward days; the forward report).
- Alpaca paper account only; the loop is a Windows scheduled task (09:20, 15:20, 16:10 ET) using
  `scripts/run_live.ps1`; every decision and fill logged to `results/live/`.
- `HOLDOUT_START = 2026-10-01`; `data/bars.load_bars` keeps refusing research reads past it; `live/`
  reads through `allow_holdout=True` with an audit record like `holdout_access.jsonl`.

**Done when.**
- [ ] Dry run against the Alpaca paper API with a one-share size for one week: orders, fills,
      reconciliation and the scorer all produce rows; modelled vs actual fill cost reported.
- [ ] Monitor tests: BOCPD flags a planted mean shift within the registered delay; no flag on a
      stationary stream over 1,000 simulations at the chosen hazard.
- [ ] Registry enforced: a hash mismatch refuses to trade; the forward scorer can reproduce every
      day's signal from cached data (determinism check weekly).
- [ ] Forward report after 60 sessions (interim, descriptive only) and the verdict after ≥ 252 sessions.

**Reviewer focus.** Any research path that can read forward bars; signal recomputation drift between
live and research code; order timing vs the modelled fill.

**Cost.** 2 sessions of code; ≥ 12 months of calendar time. Start the moment U19 adopts a family; do
not wait for the others.

### U21 — Volatility risk premium family (optional, gated)

**Goal.** F9 if, and only if, U13 delivers a clean VX1/VX2 settlement history and SVXY is tradable in
the paper account with its post-2018 leverage.

**Scope.** `vix_carry` primary; family spec with the hard risk rules (flat on VX1/VIX < 1.0; daily loss
gate 3 %; kill switch at 15 % drawdown); the same family test; budget 8.

**Done when.** As U18 for one family; the status note includes the February 2018 and March 2020 paths.

**Cost.** 1 session.

---

## Status

- U12 — not started.
- U13 — not started.
- U14 — not started.
- U15 — not started.
- U16 — not started.
- U17 — not started.
- U18 — not started (family specs not yet registered).
- U19 — not started.
- U20 — not started.
- U21 — not started (gated).

## Deferred / out of scope

- Anything below 5Min, including 1Min bars and any liquidity-provision (maker) strategy.
- Deep sequence models; a learnable wavelet layer (WEAPON) — not before a family passes without it.
- Point-in-time universe for single stocks (only needed if F7's stock leg or F10 is pursued at scale).
- Dollar / imbalance bars — only if an intraday family passes and its signal is activity-driven.
- Real-money trading: not in this plan; a passed forward test is the precondition for discussing it.
