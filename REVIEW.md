# Critical review of the U1–U11 study, and a path to a tradable edge

Date: 2026-10-08. Read for this review: `PLAN.md`, `SPEC.md`, `handoff.md`, every pipeline package
(`data`, `features`, `primaries`, `models`, `sizing`, `risk`, `validation`, `wfo`, `experiments`), the
ledger (1,068 rows), the Stage A–E selection reports, the artifacts of the 14 Stage E cells, the two
papers (Toulson & Toulson, WEAPON/STTS; Ahrabian, Cheong Took & Mandic, phase synchronization), nine
Quant Beckman posts (public previews only; the bodies are paywalled) and that blog's sitemap (131 posts).

---

## 1. Verdict

1. **None of the 14 finalists is usable, even on paper.** Two independent reasons, neither statistical:
   - The cells whose holdout Sharpe looks good earn nothing. QQQ 1Hour (holdout Sharpe 2.31) made
     1.0 %/yr at 0.4 % volatility and was flat on 81 % of days; in development it made 0.4 %/yr. GOOGL
     1Day: 1.8 %/yr at 2.6 % vol. AMZN 1Hour: 0.2 %/yr, flat 96 % of days. A Sharpe here is a ratio of two
     tiny numbers. Retail leverage cannot turn 1 %/yr into a strategy.
   - The cells that did earn money in development reversed: NVDA 1Day 36.6 %/yr → −5.0 %, SPY 1Day
     13.4 %/yr → −12.4 %, XLE 1Day −7.7 % → +12.0 %. Development Sharpe did not predict holdout Sharpe
     (Spearman −0.15), the equal-weight portfolio of all 14 holdout streams has Sharpe −0.35, and the 14
     streams are nearly independent (mean pairwise correlation 0.04, effective N 12.3): fourteen draws of
     zero-mean noise, as the U11 report already concluded.
2. **The null is correct and the experiment was well run.** The harness (data layer, causality tests,
   purge/embargo, OOF meta-training, calibration, sizers, risk layer, ledger, pre-registration, locked
   holdout) is sound and is the genuinely valuable output of U1–U11. The defect is upstream of it.
3. **The root defect is the hypothesis, not the machinery.** A mechanism-free search over single-symbol
   price/volume transforms at 15Min–1Day, labelled by the sign of a one-session return, on 19 liquid
   survivors in a bull market, has an a priori expected edge of about zero after costs. Meta-labeling,
   PWFO and a model zoo cannot create edge the primary does not have; they can only decide when to
   abstain, which is exactly what the results show (primary-only Sharpe −0.6 to −1.1, meta-filtered ≈ −0.1).
4. **The selection criteria were not too strict; the search was too large.** The 3.5 Sharpe bar is the
   arithmetic consequence of 1,402 trials followed by a one-year, per-cell test. Loosening the criteria
   would manufacture false positives. Shrinking the search and pooling across symbols lowers the bar
   legitimately.
5. **Speed.** About 95 % of compute goes into nested cross-validation that buys nothing (2–3 point
   hyper-parameter grids re-searched every window, calibration cross-fitting, ecdf refits, a 16-combo
   PWFO grid along an axis shown not to matter). A 20–50× reduction is available with no loss of rigour
   (§6).
6. **The way forward** is hypothesis-first: three or four strategy families with a documented economic
   mechanism that a retail Alpaca account can actually trade, each tested once as a family with the
   existing harness, with meta-labeling used as a conditioning overlay on a primary that has a mechanism,
   and a forward paper-trading test as the new holdout (§8, §11).

---

## 2. What the ledger says

Counted trials through Stage E: 1,402. Holdout accessed once (batch `33b8b08e02e9413c`).

| stage | ok cells | mean Sharpe | median | share > 0 | note |
|---|---|---|---|---|---|
| A (screen, 475 xgb + 83 rf) | 558 | −0.18 | −0.19 | 31 % | 54 pass AUC > 0.515 and PSR > 0.5 |
| B (B1+B2+B3) | 297 | 0.23 | 0.26 | 75 % | survivors of A: regression to the mean visible in every sub-stage |
| C (rolling PWFO, 14 finalists) | 14 | 0.22 | 0.06 | 64 % | 0 of 14 pass the C→D gate |
| E (holdout, 248 days) | 14 | 0.02 | 0.05 | 50 % | 0 edge, 13 not demonstrated, 1 negative |

Stage A by timeframe (xgb meta): mean OOS meta AUC 0.506 (15Min), 0.502 (30Min), 0.503 (1Hour), 0.493
(1Day, sd 0.05), 0.512 (5Min, sd 0.008). Only 5Min shows an AUC consistently above 0.5, and 5Min is
where costs destroy it (mean Sharpe −0.23, 22 % positive, median turnover 238×/yr). By primary, the
primary-only Sharpe is −0.63 (ml_xgb) to −1.09 (bollinger_mr): every primary loses money after 1 bp.

Economic magnitude of the finalists (dev = Stage C live stream; holdout = 248 days):

| cell | dev ret/yr | dev vol | dev Sharpe | dev max DD | holdout ret/yr | holdout vol | holdout Sharpe | flat days |
|---|---|---|---|---|---|---|---|---|
| NVDA 1Day ml_xgb/logit_l2/ecdf | 36.6 % | 24.0 % | 1.41 | −34 % | −5.0 % | 11.1 % | −0.41 | 49 % |
| SPY 1Day wavelet_trend/logit_l2/fixed | 13.4 % | 12.7 % | 1.06 | −9 % | −12.4 % | 10.3 % | −1.22 | 25 % |
| GOOGL 1Day bollinger_mr/rf/linear | 1.5 % | 3.3 % | 0.49 | −6 % | 1.8 % | 2.6 % | 0.72 | 66 % |
| XLK 30Min donchian/rf/linear | 0.5 % | 1.2 % | 0.42 | −2 % | 0.1 % | 0.6 % | 0.14 | 81 % |
| QQQ 1Hour donchian/xgb/ldp_sigmoid | 0.4 % | 0.9 % | 0.40 | −2 % | 1.0 % | 0.4 % | 2.31 | 81 % |
| XLE 1Hour donchian/catboost/ecdf | 2.2 % | 7.5 % | 0.33 | −16 % | −1.9 % | 1.5 % | −1.27 | 95 % |
| SPY 30Min donchian/rf/fixed | 0.2 % | 9.1 % | 0.07 | −30 % | 1.0 % | 5.1 % | 0.23 | 41 % |
| XLK 1Hour donchian/catboost/fixed | −0.1 % | 10.9 % | 0.04 | −18 % | 7.1 % | 9.5 % | 0.77 | 55 % |
| QQQ 30Min donchian/rf/fixed | −1.0 % | 11.4 % | −0.03 | −31 % | −11.0 % | 8.0 % | −1.42 | 33 % |
| AMZN 15Min ml_xgb/catboost/linear | −0.1 % | 1.7 % | −0.04 | −4 % | 0.0 % | 0.7 % | −0.05 | 83 % |
| MSFT 1Hour donchian/rf/fixed | −3.6 % | 11.6 % | −0.25 | −24 % | −2.1 % | 8.7 % | −0.20 | 54 % |
| XLE 1Day ml_xgb/catboost/fixed | −7.7 % | 17.8 % | −0.36 | −41 % | 12.0 % | 16.3 % | 0.78 | 40 % |
| GLD 1Day bollinger_mr/rf/ldp_sigmoid | −0.2 % | 0.6 % | −0.28 | −2 % | −2.9 % | 3.2 % | −0.88 | 48 % |
| AMZN 1Hour wavelet_trend/rf/ldp_sigmoid | −0.3 % | 0.9 % | −0.37 | −3 % | 0.2 % | 0.3 % | 0.72 | 96 % |

Buy-and-hold Sharpe over the same holdout days: 0.18 (MSFT) to 1.76 (XLE), median 1.24.

Three structural facts fall out of the per-combo tables in the Stage C/E cells:

- **Turnover eats the 30Min cells.** SPY 30Min and QQQ 30Min turn over 450–640× a year; at 1 bp per
  side that is roughly 5–6 %/yr of cost, which is the whole of their −3 %/yr. Nothing about them is a
  signal problem that a better model would fix at this horizon.
- **The sized intraday cells barely trade.** `ldp_sigmoid`/`linear` with calibrated probabilities that sit
  near τ = 0.5 give mean sizes of 0.1–0.3, and `single` position mode skips overlapping events. The
  result is the 80–96 % flat-day share above.
- **Cadence never mattered.** Within every cell the 8–16 combos have nearly the same Sharpe; the
  nested selector's switching among them is what produced PBO 0.6–0.8 (§4.7).

---

## 3. Can anything be paper traded?

As a candidate for capital: no. Continuing to *score* the 14 frozen procedures forward costs nothing,
extends the holdout without any re-selection, and is the only way to add statistical power to those
fourteen tests. It is legitimate, but it is a continuation of Stage E, not a strategy. What it would
most likely show after another year is what §2 already shows: the return is near zero.

Minimum track record lengths make the point. For PSR(0) ≥ 0.95 a true annualized Sharpe of 0.4 needs
about 17 years of daily returns; 1.0 needs about 2.7 years; 2.0 needs under a year. The development
Sharpes of the plausible cells were 0.3–0.5 on sub-1 % returns. No holdout length rescues that.

If you want to paper trade something, paper trade the *next* hypothesis family (§11), pre-registered,
from day one. That forward test is the honest replacement for the holdout you have now spent (§7).

---

## 4. Where the design is defective

Ranked by how much of the null result each one explains.

### 4.1 Mechanism-free search (root cause)

PLAN's thesis was "where, if anywhere, does the pipeline work", answered by a 950-cell screen and a
selection ladder. With 737 trials through Stage A the expected maximum Sharpe of pure noise is 1.8
annualized; every Stage A cell was below it. The ladder then selected on PSR three more times. The
B → C collapse (finalists with B3 PSR 0.80–0.998, of which 5 of 14 went negative and 4 more to ≈ 0
under rolling retraining) is the textbook signature of selection on noise. The Quant Beckman
"hypothesis families" lecture (§10) describes exactly the alternative: fix a mechanism, then test
specification variants for coherence, instead of screening cells.

### 4.2 The primaries have no edge, and meta-labeling cannot supply one

López de Prado's precondition for meta-labeling is a high-recall primary with real but noisy edge.
Here every primary (four rules and the ML classifier) has primary-only Sharpe −0.6 to −1.1 after 1 bp.
The meta-model's job then reduces to abstaining; approved shares of 27–42 % lift Sharpe from −0.7 to
≈ −0.1, and AUCs of 0.50–0.54 are what you get when the label is a coin flip. The ML primary
(`ml_xgb`: XGBoost on the same features, plus a magnitude regressor) is no better than the rules,
which tells you the features carry no directional information at these horizons.

### 4.3 The label is the least informative one available

Symmetric barriers at ±1σ·√h with a vertical barrier of one session (intraday) or two weeks (1Day)
make the label almost exactly the sign of a short-horizon return: base rate ≈ 50 %, minimum
information. `META_MIN_RET` = 2 bp does not change that. Nothing ties the label horizon or the feature
windows (20–50 bars) to any mechanism's holding period. The Beckman transformation post's one good
point applies: match the transformation and the horizon to the economic mechanism, or you measure
"false statistical stability".

### 4.4 Features describe the symbol's own past, not the state of the market

Every group in `features/groups.py` is a transform of the traded symbol's OHLCV at the trading
timeframe. There is no volatility state (VIX level, VIX term structure, realized vs implied), no
session structure (overnight gap, prior-session intraday return, time-since-open return), no breadth,
no cross-asset context (the `cross_asset` group exists but was never run in U11 because the runner
passes no market bars), no calendar events (FOMC, CPI, OPEX, month-end, earnings), no flows or
positioning. Those are where the documented, retail-tradable anomalies live (§8). Your instinct in the
question is right, with one caveat: adding those features to *this* label and *these* primaries will
not help; they need to drive the primary, not decorate the meta-model.

### 4.5 Universe and period

Nineteen large, liquid, surviving names from 2016 to 2025: a bull market with two short shocks. A
market-neutral intraday stream (β ≈ 0 for all 14) is then compared with buy-and-hold Sharpes of
1.0–1.8. "Beat buy-and-hold" over such a period is a very high bar for anything that is not long
equities. The right target for a retail account is "beat buy-and-hold risk-adjusted, or add to it as an
overlay" (§8.1).

### 4.6 The selection ladder optimised the wrong thing at each rung

B1 picked the meta-model by OOS AUC; within a cell AUC ranked the models' Sharpes with mean Spearman
0.23. B2 picked the feature arm by AUC on margins of 0.001–0.018. B3 picked the sizer by PSR, which is
the best of four on the same trades. Each step carried a near-random choice forward and inflated the
finalists' PSRs. The ladder also never selected on economic magnitude: a cell earning 0.4 %/yr
with Sharpe 0.4 outranked cells earning 10 %/yr with Sharpe 0.3.

### 4.7 Nested PWFO selection tests a choice that should not exist

The nested selector re-picks a combo every 10 days on trailing 126-day Sharpe among 8–16 combos that
are, by construction, near-identical strategies. That choice is noise, so its PBO is 0.6–0.8 even in
cells where every combo is positive (NVDA 8/8, SPY 1Day 8/8, QQQ 1Hour 16/16). The C → D rule then
failed everything on the PBO of the cadence choice, as the plan itself noted. Replace selection with
equal-weight averaging of the combos' returns (an ensemble over window lengths, which cannot overfit
by construction), or run one combo. The Stage C evidence is that cadence does not matter at all.

### 4.8 The statistics were right for the question, but the question had no power

A 248-day holdout gives an annualized Sharpe a standard error of about 1.0, so the first Holm step
over 14 needs Sharpe ≈ 2.7 and the full-search DSR needs ≈ 3.5. The DSR's N = 1,402 also over-deflates:
the trials are highly dependent (same symbols, same events, similar models), so the expected maximum
is lower than the independent-trial formula gives. None of this changes the verdict here (the streams
have no return), but for future work the honest null is a *procedure bootstrap*: re-run the whole
selection ladder on block-bootstrapped or sign-flipped returns and read the null distribution of the
final pick directly. That is affordable once the search is small (§6, §7).

### 4.9 Costs

1 bp one-way, no spread under risk profile `none` (all U11 cells). Generous for SPY/QQQ, about right
for AMZN/NVDA at 1Hour, optimistic at 15Min. Not the cause of the null (development Sharpes were ≈ 0
before costs mattered), but any future intraday candidate must be run with `standard` and a half-spread
taken from quotes, not Corwin–Schultz on 5Min bars, which over-estimates by about 10×.

### 4.10 Operational

Nested loky pools crashed three times in Stage E (memory corruption inside forest code); the hashed-code
freeze during multi-day runs blocks all development; `results/ledger.jsonl` and the artifacts are
gitignored, so the trial history that every DSR depends on lives on one disk.

---

## 5. What was done right, and should be kept unchanged

- Data layer: SIP feed, resampling everything from RTH 5Min bars with the exchange calendar, atomic
  cache with hashes, re-adjustment detection, dropped-session logging.
- Causality as a test, not a claim: the future-perturbation test over every feature, primary, sizer and
  the WFO frame; the injected-lookahead mutation checks.
- Purge and embargo, out-of-fold primary signals for the meta-model, inner purged CV, cross-fitted
  calibration, uniqueness weights.
- Barrier-consistent backtest and the risk/portfolio simulator with its property tests.
- The ledger, cell hashing, cache hits, resumability, and the once-only holdout with an audit record.
- Pre-registration of every rule before results were seen, and candid amendments when a rule was
  degenerate.
- `fastfracdiff` (170× on the d search), the feature cache, the EcoQoS fix.

This is a research harness most retail quants never build. The next project should change the
hypotheses and keep the harness.

---

## 6. Speed: where the time goes and how to cut it 20–50×

Cost anatomy of one fitted window today, `rf_ldp_fast` meta, `oof`, `ecdf` sizer, rule primary:

| step | model fits per window |
|---|---|
| meta HP search: grid 1 × 4 purged splits | 4 forests |
| calibration cross-fit (sigmoid and isotonic on the OOF) | cheap |
| refit on all rows | 1 forest |
| `ecdf`/`kelly` train-window inputs: `oof_meta_prob` refits a *tuned* meta per split | 4 × 5 = 20 forests |
| `ml_xgb` primary with `oof` (if used): 4 OOF fits + 1 refit, each classifier + regressor | 10 boosters |
| per-fold `cmda` (if used): 4 forests of 100 trees | 4 forests |

So an `ecdf` window costs 25 forest fits, and a PWFO cell runs ~2,865 windows across 16 combos (about
70,000 forest fits per cell). Catboost adds a fixed ~30 s per fit regardless of window size.

Changes, ranked by payoff, none of which weakens the science:

1. **Freeze hyper-parameters per (model, timeframe).** The grids are 2–3 points; `rf_ldp` chose
   `max_features=1` in 81 % of 21,948 fits. Choose once in a dedicated study, then fit once per window.
   Saves 4–8× on every cell.
2. **Reuse the OOF you already have for the sizers.** `_Tuned` stores `oof_raw_`; apply the chosen
   calibrator to it instead of refitting 4 tuned meta-models in `oof_meta_prob`. Saves 3.5× on
   `ecdf`/`kelly` cells, and makes those sizers free.
3. **Calibrate from the past, not from a refit.** With no inner CV left, calibrate each window with Platt
   on the previous windows' OOS predictions (causal, zero extra fits) rather than cross-fitting inside
   the window.
4. **Cut the PWFO grid to two combos** (IS 504 and 756 intraday, 1260 and 1512 daily, OOS 21) and
   average them instead of nested-selecting. 8× on Stage C-type runs and the PBO problem disappears.
5. **Drop catboost** (no demonstrated gain, dominates wall-clock in B1, B2, C and E) and keep
   `logit_l2` + `rf_ldp_fast` as the zoo for screening; bring xgb back only for a finalist.
6. **Retrain less often and warm-start.** At OOS 5 and IS 756 consecutive windows share 99 % of their
   rows. OOS 21 or 63 is indistinguishable in results; XGBoost supports `xgb_model=` warm starts.
7. **Flatten the parallelism.** One loky pool over (cell × combo) tasks instead of pools inside pools.
   This is also the likely fix for the Stage E crashes.
8. **Screen with a proxy.** For hypothesis screening use standardized logistic regression, no
   calibration, no sizer, `fixed`, one expanding WFO: seconds per cell. Rank by pooled log-loss
   improvement over the base rate and by net return per unit turnover, not by Sharpe.
9. Minor: `cusum_events`/`cusum_state` are Python loops over every bar (numba or a vectorised reset
   would take them to milliseconds); `compute_siegel_slope` is O(n·w²) and is 20 × 19 pair medians per
   bar.

Estimated effect: a Stage A-sized screen falls from ~860 CPU-hours to under 30; a Stage C-sized run
from ~110 CPU-hours to about 5; a Stage E-sized holdout to under an hour.

---

## 7. Data limitations and selection criteria

**The limitation is information content, not length.** Ten years of 5Min bars is plenty for intraday
tests. What is missing is exogenous state. Available to a retail account, mostly free:

| data | source | cost |
|---|---|---|
| VIX, VIX9D, VIX3M, VIX6M daily closes (term structure, RV/IV) | CBOE CSVs; FRED (VIXCLS) | free |
| VIX futures settlements (contango/backwardation) | CBOE | free (delayed) |
| SPY/QQQ option chains, implied vol, open interest | Alpaca options data (snapshots), CBOE delayed | free with Alpaca; gamma exposure needs OI snapshots, crude daily GEX is computable |
| Treasury yields, breakeven inflation, dollar index, credit spreads | FRED | free |
| Breadth (advance/decline, % above 50/200-day) | compute from a broader Alpaca universe | free, needs more symbols |
| FOMC, CPI, NFP, OPEX, month-end, holiday calendar | Fed, BLS schedules; OPEX is deterministic | free |
| Earnings dates | Alpaca corporate actions / news, or free vendors | free |
| Overnight gap, prior-session intraday return, session structure | already in your 5Min cache | free |
| Quoted spreads for the cost model | Alpaca quotes endpoint (sample, do not cache everything) | free |

**Selection criteria to change:**

1. Test *families*, not cells. A family is one mechanism applied to a set of symbols; its test is one
   pooled statistic (stack the symbols' trade returns, or trade the family as an equal-weight portfolio).
   One test per family, pre-registered, instead of one per symbol × timeframe × model × sizer.
2. Require economic magnitude: a floor on net return per year and on net return per unit of turnover,
   alongside Sharpe. A 0.4 %/yr stream is rejected at the gate whatever its Sharpe.
3. Shrink. Per-symbol estimates within a family should be reported as James–Stein-shrunk toward the
   family mean; "do the 14 look like noise" becomes a number (the shrinkage factor).
4. Replace the DSR over a correlated 1,400-trial ledger with a procedure bootstrap for the small search
   you will now run: rerun the entire selection on block-bootstrapped (or sign-flipped) return streams
   and read off the null of the final pick. Affordable after §6.
5. Keep PBO for comparing genuinely different strategies; never gate on the PBO of a cadence or
   window-length choice. Average those choices instead.
6. Holdout: the 2025-10 → 2026-09 window is now mildly contaminated for *new* hypotheses (you have seen
   buy-and-hold Sharpes of 14 symbols and 14 strategy outcomes on it). The honest replacement is a
   pre-registered forward test: paper trading on Alpaca from the day a family is frozen, minimum 12
   months, with the development window extended to 2026-09. A frozen procedure scored on live data is a
   holdout nobody can have peeked at.

The "very high DSR and Sharpe needed" problem is therefore not something to address by relaxing the
bar. It is addressed by (1) making the search small enough that N is tens, not thousands, (2) pooling
so that T is thousands of trade-days, not 248, and (3) requiring returns large enough that a real edge is
detectable in two or three years at all.

---

## 8. Where edge plausibly exists for a retail Alpaca account

Ranked by evidence quality × tradability at retail × fit with the harness. All are daily-or-slower
except where noted; all are testable with your current data plus the free sources in §7.

1. **Volatility-managed exposure (Moreira & Muir, 2017).** Scale SPY/QQQ (or an ETF basket) exposure to
   a target volatility using trailing realized vol (or VIX). Historically the same return as
   buy-and-hold at lower drawdown, Sharpe higher by ~0.1–0.3, a few trades a month. Zero machine
   learning. This is the baseline every overlay must beat, and it directly answers "beat buy-and-hold in
   a retail scenario" on a risk-adjusted basis. Your `standard` risk profile already contains the
   mechanism (vol target); it was only ever applied to signals with no edge.
2. **Time-series momentum / trend across uncorrelated ETFs** (Moskowitz, Ooi & Pedersen 2012; Hurst,
   Ooi & Pedersen): 1–12 month lookbacks, vol-scaled, weekly or monthly rebalance, across SPY, QQQ, IWM,
   TLT, GLD, XLE, XLF and ideally DBC/USO/UUP/EFA/EEM. Long-sample Sharpe 0.6–0.9 with crisis
   convexity. `wavelet_trend` is a trend rule applied at the wrong horizon (1Hour, one-session hold);
   the MODWT smooth at J = 6–8 on daily bars is a perfectly good trend estimator for this family.
3. **Overnight versus intraday decomposition** (Lou, Polk & Skouras 2019; Knuteson 2022). In SPY/QQQ
   essentially all of the drift accrues close-to-open; intraday is ≈ 0 or negative. "Long overnight, flat
   or vol-scaled intraday" and overnight-gap-conditional intraday rules are testable today with the 5Min
   cache. Costs: two fills a day at ~0.3 bp on SPY ≈ 1.5 %/yr, so SPY/QQQ only.
4. **Calendar and event drift**: pre-FOMC announcement drift (Lucca & Moench 2015; weaker since 2016
   but documented), turn-of-month and month-end rebalancing pressure, OPEX week, pre-holiday. Few
   trades, cheap, needs only a calendar.
5. **Volatility risk premium / VIX term structure**: harvest contango (short VIX futures exposure via
   SVXY-type products, or sell index options; Alpaca supports options for retail) when VIX is low and the
   curve is in contango, exit or hedge on backwardation. Sharpe 0.8–1.2 historically with severe tail
   risk (Feb 2018), so this is the one family where your risk layer earns its keep. Data: CBOE, free.
6. **Short-term reversal in liquid ETFs and large caps** at a 1–5 day horizon, conditioned on volume and
   volatility. Documented but modest and cost-sensitive; `bollinger_mr` at 1Day is a crude version.
   Worth a family test at 1Day only.
7. **Post-earnings-announcement drift / earnings-gap continuation** in single stocks. Needs an earnings
   calendar; effect is strongest outside mega-caps, so the current universe is the worst place to look.
8. **Cross-asset lead-lag** (the phase-synchronization paper's idea, §9). At daily frequency in US
   large caps the effect is small; at 1–5 minutes it exists (futures → ETF → constituents) but is
   high-frequency territory. Cheap to test as *features* (sector ETF lagged returns, leader residual
   spread) through the existing `cross_asset` group; expect little.
9. **Dealer gamma regime** as a conditioning variable (positive gamma → intraday mean reversion,
   negative gamma → trend). Growing literature and widely used by retail vol traders; needs option OI,
   which Alpaca's options snapshots or CBOE delayed data can supply for a crude daily GEX. Use as a
   regime feature for families 3 and 6, never standalone.

Where machine learning belongs in that list: as the *conditioning* layer (when to scale up or down)
and the *combiner*, which is to say meta-labeling on a primary that has a mechanism. That is precisely
the design you built, pointed at primaries that had none.

---

## 9. The two papers

**Toulson & Toulson, "Intra-day trading of the FTSE-100 futures contract using neural networks with
wavelet encodings" (WEAPON / STTS).** A four-layer MLP whose first hidden layer is a bank of Mexican-hat
wavelet neurons with *learnable* dilation and translation, trained by backprop with an orthogonality
penalty and a Laplacian weight prior ("marginalise and murder" pruning), on 240 lagged 1-minute returns,
predicting 15/30/60/90-minute returns; committees of five networks per horizon; a Signal Thresholded
Trading System that sums the horizons' forecasts (S(t) = Σ ωᵢ pᵢ/τᵢ) and trades with entry threshold α
and exit threshold β < α (hysteresis). Evidence: 18 months of 1995–96 LIFFE data, thresholds tuned by
simulated annealing on a 6-month "optimisation" set, a 6-month test: 53 ticks/month net, 18 round
trips/month, daily Sharpe 0.136 (monthly 0.48, ≈ 1.7 annualized), with 8 ticks of round-trip cost and a
3-minute fill delay. No multiplicity control, one asset, one 6-month window, 1996 microstructure. Sign
accuracy 53–55 %.

Usable here: (a) the STTS hysteresis exit (enter above α, exit below β) as an alternative exit model to
the triple barrier for continuous-signal primaries; it cuts turnover, which is where the 30Min cells
died; (b) the multi-horizon committee as a way to combine forecasts rather than pick one; (c) a
learnable wavelet layer is today a small 1-D CNN with wavelet-initialised filters. Given AUC ≈ 0.50 for
every learner on these inputs, do not expect a different outcome from a different learner on the same
inputs. Verdict: not a primary candidate; keep the exit logic in mind.

**Ahrabian, Cheong Took & Mandic, "Algorithmic trading using phase synchronization" (IEEE JSTSP 2012).**
De-trend two related stocks, band-pass each with a windowed synchrosqueezed transform, compute the
instantaneous phase difference and an entropy-based synchrony score; when synchrony exceeds a threshold,
identify the leading asset from the sign of the phase lag and trade the lagging asset on the leader's
confirmed extrema. Evidence: three UK pairs, 2003–2011, results shown for a 500-day high-volatility
window (2008–10): +20–28 % versus negative for MA-crossover and MLMS, no costs, no multiplicity control,
window and band chosen by hand, SST is a block transform run on sliding windows. The surrogate test
establishes that synchrony exists, not that it is tradable.

Usable here: relative-value inputs between a leader and a lagger (phase difference, lagged leader
returns, residual spread) are exactly the cross-asset family you deferred. At 1Hour–1Day in US
mega-caps the lead-lag is small and well arbitraged; at minutes it is real but needs execution you do
not have. Verdict: test as features once, cheaply; not a standalone system at retail.

---

## 10. The Quant Beckman posts

All seven linked posts expose only a preview; the methods, code and results are behind the paywall,
and the sample PDFs were not read. What the visible text supports:

| post | usable here |
|---|---|
| Feature selection: wrapper / embedded / filter | A catalogue of standard methods. The one transferable point: score features by *net economic* value (turnover, cost, subset size), not by log-loss. Your cMDA already showed selection changes nothing on these features (median AUC −0.007). |
| Data transformations: data shape | Benchmark-neutralised residual returns, cointegration spreads, state-space "surprise" (local-linear-trend forecast errors), bounded rolling min-max. These are the relative-value and state inputs you lack; they belong to the cross-asset family. |
| Data transformations: preprocessing | Fractional differencing (you have it), GARCH-standardised returns (cheap; a better vol normaliser than EWM σ), rank-Gaussian transforms, dollar bars (deferred in PLAN; small gain at 15Min+ on liquid ETFs), Takens embedding (no). The "60 % of memory lost" claim is unverified. |
| Switch-off: Bayesian online changepoint detection | Directly useful for the paper-trading phase: a run-length posterior on the live P&L stream with a Student-t predictive, hazard set from the expected regime length, pruning for speed. It is the principled version of your drawdown throttle. Adams & MacKay (2007) is public; the paywalled "dual trigger" is easy to design. |
| Data: building micro-trading machines | Cache-line, NUMA and TLB engineering for sub-millisecond loops. Irrelevant at any horizon you will trade. |
| Ordered Random Forest (the "parametric bias" post) | Ordinal-target forests; not relevant, and the public code appears to drop the top class in `predict_proba`. |
| Hypothesis families lecture; Event-driven alpha | The methodological correction this project needs: fix a mechanism, enumerate auxiliary and nested hypotheses, vary state / response / sample definitions and look for coherence. The event catalysts listed (earnings, index flows, macro releases) overlap with §8 items 4 and 7. |
| Validation framework (older) | Timing control groups, stationary block bootstrap, stratified synthetic data: the "procedure bootstrap" of §7 in spirit. Your CPCV/PBO/DSR implementations are stronger than what the preview shows. |
| Rest of the archive (by slug) | Options income strategies (iron condor, butterfly, straddles: §8 item 5), bars (tick/dollar/imbalance), risk-engine sizing posts, changepoint variants (conformal, RANSAC), an intraday-trading overview, calibration and scoring, alpha/factor decomposition lectures. All previews; nothing is a ready edge. Treat the blog as a checklist, not a source. |

---

## 11. Proposed next unit (U12), hypothesis-first

**Phase 0, harness changes (mostly code, little compute).**
- Speed items 1–5 and 7 of §6; flatten the pools.
- Exogenous data loaders: CBOE VIX family, FRED series, FOMC/CPI/OPEX/month-end calendar, earnings
  dates; a quotes sampler for spreads.
- New feature groups: `session` (overnight gap, prior intraday return, open-to-now), `vol_state` (VIX,
  VIX3M/VIX, RV/IV, GARCH-standardised return), `calendar_events`, and run the existing `cross_asset`.
- New primaries with a mechanism: `vol_target` (long-only, exposure = σ*/σ̂), `tsmom` (multi-lookback,
  daily), `overnight` (long close-to-open in SPY/QQQ), `event_drift`. New exit model `hysteresis`
  (STTS) for continuous signals, alongside the triple barrier.
- Selection tooling: family-level pooled test, magnitude floors, shrinkage report, procedure bootstrap.

**Phase 1, baselines without ML on the development window (now 2016-01 → 2026-09).** Buy-and-hold,
vol-managed SPY/QQQ, TSMOM ETF basket, overnight-only SPY/QQQ, each with the `standard` risk profile and
quote-based costs. Pre-register one metric per family: net return and Sharpe versus buy-and-hold, alpha
versus buy-and-hold, max drawdown. If no family beats buy-and-hold risk-adjusted here, stop; the ML
phase is moot.

**Phase 2, meta-labeling as an overlay.** For each family that passed, ask one question: does a
meta-model (logit_l2 and rf_ldp_fast only, frozen hyper-parameters, linear sizer, two averaged PWFO
combos) improve the family's pooled Sharpe net of its extra turnover? One pre-registered test per
family. Features: the new state groups plus `wavelet_core`.

**Phase 3, forward test as the holdout.** Freeze the passing families, register them, and paper trade
them on Alpaca from the freeze date for at least 12 months, with the BOCPD monitor as the kill switch.
Score them exactly as Stage E scored the 14; Holm across the families.

**Phase 4, deferred items that now matter:** the paper-trading loop (orders at open/close,
reconciliation), a point-in-time universe if family 7 is pursued, dollar bars only if an intraday family
survives Phase 1.

Compute: with §6 applied, Phases 1 and 2 are hours of CPU, not days. Phase 3 costs calendar time, which
is the one resource no amount of CPU replaces, so start it early and let it run while Phases 1–2 of the
next families proceed.
