# PLAN3 — Intraday continuation program (U22 → U31)

Supersedes [PLAN2.md](PLAN2.md) for all forward work. PLAN2 remains the record of U12–U18 (complete; every
family failed). [PLAN.md](PLAN.md) remains the record of U1–U11. [SPEC.md](SPEC.md) §1–§19 still describes the
as-built engine; this plan adds SPEC §20–§24 (written in U22). Date: 2026-10-10.

## 0. Diagnosis: why the PLAN2 program ended in a null, and what this plan changes

The PLAN2 program was run honestly, and its machinery is sound in the ways that matter most: pre-registration
with a git SHA, trial budgets, causality tests, point-in-time exogenous data, a quotes-based cost model, a
locked forward holdout. Two audits were run for this plan (scratchpad `audit_engine/` and `audit_stats/`;
nothing in the repo was modified by either). Three things, in this order, made a null the only possible
outcome.

### 0.1 The test could not pass for the effects it hypothesized (statistics audit)

Every family was judged on Sharpe against equal-risk buy-and-hold of its own instruments over 2016-01 →
2025-09 (SPY Sharpe 0.83, 15 %/yr), plus a net-return floor of 0.8 × buy-and-hold for long-only families. F1
(vol-managed) delivered what the literature promises (Δ Sharpe +0.09, drawdown −18 % vs −34 %) and failed,
because the 95 % interval on a Sharpe difference over 2,400 days is ± 0.3 and the return floor punishes
de-risking in a bull decade. An overlay exposed 15–40 % of the time cannot earn 0.8 × buy-and-hold without
leverage, and its economic value (zero beta, positive alpha, convexity) was never the quantity tested.

The audit (`power.py`, `mde.py`, `lwcheck.py`, `dsrcheck.py`) reproduced every reported p and DSR exactly and
found the implementation correct: the Ledoit–Wolf gradient, studentization, circular block bootstrap, PSR/DSR
and Holm all match the papers. It then measured what the design could detect. Simulating each family's real
stream with its mean shifted to the **top** of PLAN2's expected range, the probability of passing the
registered verdict was 22 % (F1), 10 % (F3), 1.5 % (F4) and ≤ 0.5 % (F2, F5, F7, F8). For an overlay
uncorrelated with its benchmark the Holm-level minimum detectable Δ Sharpe was 1.65–1.86, i.e. a standalone
Sharpe of 2.5–2.8 against hypotheses of 0.4–1.0; F2 and F7 were registered with expected magnitudes that are
*negative* Δs against their own benchmark. Two further findings: cash earned nothing while the benchmark's
Sharpe includes the risk-free rate (a 0.1–0.3 Sharpe tilt against cash-heavy overlays), and the 0.8 ×
buy-and-hold floor, not the sign share, decided coherence in seven of eight families. Pooling
SPY/QQQ/IWM/DIA gave 1.0–1.7 effective instruments.

The null is therefore evidence of no edge only where a stream's own statistics are negative (F5, F7, F8,
F11); for F1–F4 it is evidence of nothing. Even the appropriate test (alpha of the overlay against zero)
needs an appraisal ratio of about 1.1–1.2 at Holm over eight families on 2,400 days. That is why §4 tests
one-sided, keeps the number of simultaneous families at four, and treats the forward test, not the
development window, as the place where an edge of 0.9–1.3 is confirmed.

### 0.2 The hypotheses were the textbook forms of widely known effects, several of which have decayed

Last-30-minute index momentum (F5) is documented on 1993–2013 and weaker since; pre-FOMC drift (F4) weakened
after 2016; the overnight premium (F3) is unstable after 2020 (SPY overnight mean by year on our own bars:
2022 −5.9 bp/day, 2023 +2.5 vs intraday +6.9, 2024 +9.7). Gap fade (F7) and macro continuation (F8) had thin
evidence and were expected nulls. None of the eight was a place where a retail account has an edge over the
firms that published them.

### 0.3 Two assumptions decide intraday verdicts and were never measured (engine audit)

The engine audit found **no BREAKING or SEVERE defect**. Every fill path decides strictly before it fills; cost
is charged exactly on traded notional through rolls and flips (an adversarial run with 1,784 trades matched an
independent share-path computation to the cent); an always-long rule reproduces buy-and-hold to 2e-16; a
random-sign 5Min rule has gross +0.24 bp per trade (t 1.1) and loses exactly the booked 2.39 bp round trip;
flipping every side negates gross P&L exactly; overnight × intraday complements rebuild close-to-close on 2,444
of 2,446 days; the stored F3, F5 and F7 headline cells rebuild to 1e-16; 772 tests pass. The verdict: "sound
and robust" is a fair description of the backtest engine, and it is more careful than most retail backtests.

What it does not cover is executability, and two of the nine minor findings decide intraday verdicts:
- **Auction prints.** Closing fills are priced at the 15:55 bar's last trade, not the auction print. Checked
  against Alpaca 1Day bars (whose close is the official close): the two differ by 2–6 cents, 0.4–1.2 bp, on
  each of five sampled sessions, with no obvious sign. Opening fills likewise use the first 5Min open.
- **Slippage.** 1 bp per side is assumed on top of the quoted half-spread: 2.3 bp per round trip on SPY. The
  audit measured |open[t+1] / close[t]| at a median of 0.17 bp on SPY (bid-ask-bounce scale), and the
  published measurement on SPY market orders is ≈ 0.02 bp (Zarattini, Aziz & Barbon 2025, 1,000 live fills).
  §1.2 shows an intraday family's verdict flips between 0.3 and 2.3 bp.

The other minor findings, all taken into U22: no generic test asserts same-bar fill timing (two planted leaks
escaped the causality machinery and were caught only by hand tests); one causality test uses a single cut and
does not bite on a planted one-bar look-ahead in `bar_volatility`; eleven ETF caches hold bars past
HOLDOUT_START because `data/fetch.py` defaults its end date to today (the loader still refuses them); the
cost table's within-year medians are ± 25 % look-ahead on tenths of a bp; stress sessions (VIX ≥ 30) cost
1.5–3.5× the table (F3 +13–36 bp/yr); equal-risk pooling weights use full-window σ and the benchmark is a
daily-rebalanced constant mix rather than drifting buy-and-hold; adjustment vintages differ by cents-rounding
noise only; barrier touches fill at the barrier, MOC entries size shares from the fill close, and schedule
events require the entry bar to exist. Not modelled at all: the pattern-day-trader rule (F5 as registered
makes ≈ 2 day trades a session on $10k, which is illegal in a margin account under $25k), settlement, margin
and locate fees, partial fills, latency, and the live data plan (free accounts see IEX in real time).

### 0.4 What is kept, what is retired

Kept unchanged: the data layer (SIP 5Min RTH bars, exchange calendar, atomic cache), the feature registry and
causality tests, the point-in-time exogenous layer, family registration and budgets, the ledger, the rule pass,
the risk layer, the forward-test design. Retired: the Sharpe-vs-buy-and-hold headline for overlays, the 0.8 ×
buy-and-hold floor, DSR in the program verdict, point-parameter headlines (replaced by region portfolios), and
every PLAN2 family except F1 (which becomes the portfolio core) and F9 (optional, G6).

## 1. Evidence gathered before writing this plan

Everything below is a **diagnostic, not a test**: parameters were natural or taken from published work whose
samples overlap ours, and about 60 configurations were examined on SPY and QQQ 2016-01-04 → 2025-09-30. The
looks are recorded in `families/looks.jsonl` (U22) as K₀ = 60. **No bar on or after 2025-10-01 was read.**
Scripts: session scratchpad `diag/` (`periodicity.py`, `periodicity2.py`, `goertzel_fc.py`, `velocity.py`,
`positions.py`, `splits.py`, `costs.py`, `close_check.py`); U22 copies them to `scripts/plan3_diagnostics/`.

### 1.1 Market structure on our own bars (SIP, 5Min, 2,425 full sessions)

| fact | SPY | QQQ | what it means for a strategy |
|---|---|---|---|
| lag-1 / lag-2 autocorrelation of 5Min returns | −0.014 / −0.015 | −0.009 / −0.012 | 5–10 minute mean reversion (bounce, dealer hedging): never trade the raw 5Min sign |
| lag-12 (one hour) | +0.013 | +0.011 | hourly-scale continuation: the horizon a trend rule should live on |
| same-slot lags 77 / 154 / 385 (1, 2, 5 sessions) | +0.005 / +0.008 / +0.009 | +0.004 / +0.007 / +0.004 | the Heston–Korajczyk–Sadka periodicity exists but at ≈ 0.5 % correlation: not tradable alone |
| slot-mean variance / sampling variance | 1.34 | 1.50 | a small deterministic time-of-day mean pattern; largest slot 09:50–09:55 (+0.75 bp, t 3.6) |
| Goertzel power at the 1-day period vs a session-shift null | 0 % of 40-session windows significant | 5 % | **no detectable daily cycle in returns**; weekly period 8–12 % of windows (barely above 5 %) |
| Meyers adaptive n-cycle Goertzel next-bar forecast (EPF 1,560 bars, periods 4–390, top 1/3/5/10 cycles) | corr with next bar −0.012; next 6 bars −0.007 | −0.014; −0.004 | the cycle forecaster has **no** predictive power on these bars; its published results come from the 200k–345k-combination filter search, not the forecaster |

### 1.2 Intraday continuation after a noise-exceeding move (gross, 10:00 → 15:55, flat overnight)

Two rules, each evaluated on bar closes, holding from the next bar's open, re-evaluated at every decision:

- **Band** (Zarattini, Aziz & Barbon 2025): long while the move from the open exceeds the 14-session mean
  absolute move to that time of day, short while below the mirror band, flat inside; decisions at HH:00/HH:30.
- **SAR velocity**: least-squares slope over 6 bars in units of the prior 5 sessions' 5Min σ; enter when |v| >
  0.75 and **hold until the opposite signal** (stop-and-reverse, Meyers' RMedV form). The "flat inside"
  version of the same signal is ≤ 0 everywhere: the edge is in holding through, not in the burst.

| rule | gross bp/day | t | Sharpe | round trips/day | years > 0 | 2024-05-13 → 2025-09-30 (post-publication) |
|---|---|---|---|---|---|---|
| SPY band, half-hourly | 2.95 | 2.7 | 0.88 | 1.33 | 8 / 10 | +3.0 bp/day (t 1.0, 343 d) |
| QQQ band, half-hourly | 4.84 | 3.6 | 1.17 | 1.24 | 9 / 10 | +5.1 bp/day (t 1.35) |
| SPY SAR N=6, 0.75σ | 4.12 | 3.1 | 1.01 | 1.75 | 8 / 10 | +0.1 bp/day |
| QQQ SAR N=6, 0.75σ | 6.10 | 3.8 | 1.22 | 1.77 | 8 / 10 | +1.6 bp/day |
| control: long 10:00 → close | 2.05 / 2.53 | 1.3 | 0.41 | 1.0 | — | — |

- Correlation with close-to-close returns: −0.02 to 0.00 (zero beta). Band vs SAR daily P&L: 0.33 (SPY), 0.28
  (QQQ): two expressions of one mechanism that diversify each other. SPY vs QQQ for the same rule: 0.75 / 0.71.
- By prior trailing-vol quintile (bp/day): band SPY −0.1, +0.2, +3.7, +8.0, +2.9; SAR QQQ −1.6, +4.9, +2.4,
  +5.6, +20.2. **Long volatility**: 2018 and 2022 are the best years, 2016–2017 the worst.
- Net of costs, 4-leg equal-weight portfolio (SPY, QQQ × band, SAR), 1× notional, 7.9 % vol:

| round-trip cost | net bp/day | Sharpe | ann. return | max DD |
|---|---|---|---|---|
| 0.3 bp (2 × quoted half-spread) | 4.1 | **1.29** | 10.2 % | −6.2 % |
| 1.0 bp | 3.0 | 0.95 | 7.5 % | −8.6 % |
| 2.3 bp (SPEC §19 today: 1 bp slippage/side + half-spread) | 1.0 | 0.32 | 2.6 % | −16.0 % |

The cost number is the verdict. That is why U27 (paper trading with fill reconciliation) is on the critical
path, and why the family report must carry the cost curve rather than one cost.

### 1.3 External evidence, read for this plan

- **Zarattini, Aziz & Barbon, "Beat the Market" (SFI 24-97, v. 2025-02-03).** SPY, 1-minute IQFeed, 2007 →
  2024-04. Band breakout with the opposite band as stop: Sharpe 0.61, 6.2 %/yr. With the current band and the
  session VWAP as trailing stops: Sharpe 1.24, 9.7 %/yr at 7.7 % vol, 7,668 trades in 17 years, hit rate 43 %,
  skew +1.3, alpha 9.9 %/yr, beta −0.03. Sharpe rises monotonically with VIX at the open; the 5-day RSI (their
  dealer-gamma proxy) predicts profitability with β −3.25 (p 0.001). Costs $0.0035 + $0.001 per share. Authors
  have a commercial interest; the sample overlaps ours; no independent out-of-sample replication was found;
  the post-publication slice in §1.2 is the only clean evidence.
- **Repeated-Median-Velocity repo (0zean, branch `dev`), Units 9–10.** The one careful retail replication of a
  Meyers strategy we know of. Findings: (i) weekly parameter selection has **negative** expected value: the
  in-sample → out-of-sample rank correlation across 4,312 combos is t = −0.45, and picking the in-sample best
  returns t = −2.0; (ii) an equal-weight **region** (N ≥ 5, thresholds 0.75–2.75 σ, no filter) earns net t
  = 2.4 (SPY) / 2.8 (QQQ) over 525 weeks, long and short sides symmetric, skew +2.0, 2018 +13.6 %, 2022
  +12.5 %, 2017 −0.2 %; (iii) a 26-week withheld tail, pre-registered, came in at +240 bp (t 0.8), consistent
  with the pre-tail rate; (iv) IEX bars are unusable (57 % of 5-minute buckets missing), live needs the paid
  SIP feed, and the pattern-day-trader rule sets a $25k floor. Everything in (ii) transfers directly: **the
  headline of an intraday rule family is a region, never a point.**
- **Meyers Analytics working papers** (RMedV DIA 5m 2025; nth-order polynomial velocity QQQ 15m 2026;
  Goertzel cycle DIA 5m 2021). Mechanically sound estimators (Siegel's repeated median; Legendre-polynomial
  velocity, which is a Savitzky–Golay derivative filter; Goertzel). The published profits come from choosing
  one filter among 199,919 / 264,599 / 345,600 input-filter combinations on the full out-of-sample set, then
  a 13-week or 60-day withheld check; the "mirror random filter" bootstrap is a null for random switching, not
  for a fixed strategy. Take the estimators, not the selection method.
- **Market intraday momentum** (Gao, Han, Li & Zhou 2018; Baltussen, Da, Lammers & Martens 2021; Rosa 2022):
  the last-half-hour effect is driven by hedging demand and has weakened since 2013; Rosa's Markov-switching
  evidence and our vol-quintile split agree that it lives in high-volatility regimes. F5's test of the textbook
  form was right to fail.
- **Dealer gamma** (Barbon & Buraschi 2020; practitioner data): sign-dependent momentum/reversal is best
  documented in less liquid stocks; index-level evidence is mixed (Cboe 2023 finds de minimis net gamma). A
  free full-history gamma series does not exist (SqueezeMetrics $720/month; FlashAlpha API since 2018, paid;
  Alpaca option snapshots since 2024-02 without open interest). Treated as a reported state split with proxies.
- **Retail infrastructure.** Alpaca Level-3 multi-leg options live since 2025-02; historical option data since
  2024-02 only (indicative feed); real-time SIP via Algo Trader Plus (≈ $99/month), needed for any intraday
  loop (free-tier keys are refused SIP requests ending inside the last 15 minutes); Alpaca returns no bars for
  most inactive (delisted) assets, so a survivorship-free single-stock universe needs a third-party source.

## 2. Thesis

> A retail Alpaca account can add a **zero-beta, long-volatility intraday overlay** to a **vol-managed
> index core** such that the portfolio's Sharpe and drawdown are both better than the core's, with the
> overlay's edge coming from one mechanism — intraday trend continuation in index ETFs after a move that
> exceeds the day's time-of-day noise, driven by sticky order imbalance and hedging feedback — expressed as a
> pre-registered **region** of robust velocity and band estimators (no parameter selection), decided at
> ≥ 30-minute cadence (the lag-1 reversal is never traded), flat overnight, and **costed at measured fills**.

Three consequences shape the program:

- **The quantity tested is alpha, not "beats buy-and-hold".** An overlay is judged on its own excess-return
  stream against zero (block bootstrap, Newey–West), on the marginal Sharpe it adds to the core, and on
  economic floors; the minimum detectable effect is computed and written into the spec before registration.
- **DSP is for estimation and state, not forecasting.** Robust slopes (repeated median), polynomial-derivative
  filters, wavelet smooths and Goertzel band power are *estimators of velocity and of the noise floor*; the §1.1
  results close the "cycle forecast" idea on these bars. The time–frequency domain enters as **state**
  (spectral tilt, flatness, vol-of-vol), reported as splits and later as overlay features, never selected on.
- **Compute is cheap by construction.** Rule families need no model fits and no PWFO: one structure-of-arrays
  pass computes every estimator for every lookback; a region's position is the mean of a grid of positions;
  the backtest of a position series is a vector product. A family run is minutes, the program an afternoon,
  and the paper-trading loop needs < 50 ms per decision.

## 3. Hypothesis families

Each family states mechanism, core claim, **headline** (the one tested statistic), variants (reported on a
specification curve, never selected), state splits (reported), expected magnitude with its source, costs,
data, budget. Instruments are the cached SIP 5Min ETFs; single stocks only in the gated G5.

### G1 — Intraday trend continuation in index ETFs (5Min bars, decisions every 30 minutes)

- **Mechanism.** Within a session, a move that exceeds the typical move-from-open for that time of day
  reveals a persistent demand/supply imbalance (institutional parent orders executed through the day,
  leveraged-ETF and option-hedging flows that buy into rises and sell into falls). The imbalance continues at
  the 30–90-minute horizon (lag-12 autocorrelation +0.013) while the 5–10-minute horizon reverts (lag-1
  −0.014), so the signal must be a smoothed velocity or a band distance, decided no more often than every 30
  minutes, and held until it reverses rather than exited on the first adverse bar. The payoff is long
  volatility.
- **Core claim.** On SPY and QQQ, an equal-weight portfolio over the registered region of band and velocity
  rules, flat overnight, has positive alpha against zero and raises the Sharpe of the vol-managed core, with
  the return concentrated in the top three trailing-volatility quintiles, symmetric long/short contributions,
  and positive skew. IWM and DIA show the same sign with lower magnitude (sample split).
- **Headline (one trial).** The region R₀, equal weight across every cell, per instrument, 50/50 SPY/QQQ:
  - estimators: (a) band distance d = (close − open) / band(slot), band = mean |move from the open to this
    slot| over the prior 14 sessions, gap-adjusted as in Zarattini et al.; (b) repeated-median velocity over N
    bars, normalized by √N and the prior 5 sessions' 5Min σ (the RMV repo's normalization, refit per session);
    (c) least-squares (Savitzky–Golay degree-1) velocity over N bars, same normalization;
  - lookbacks N ∈ {6, 9, 12, 18, 24} for (b) and (c); thresholds θ ∈ {0.75, 1.0, 1.5, 2.0} σ for (b) and
    (c); volatility multipliers VM ∈ {1.0, 1.25, 1.5} for (a);
  - decisions at the closes of the bars ending at HH:00 and HH:30, first decision 10:00; exit at the closing
    auction (U22's official-close fill); position held until the opposite signal (stop-and-reverse) for (b)
    and (c); flat inside the band for (a);
  - trailing stop: for (a) the current band and the session VWAP; for (b) and (c) none in the headline;
  - size: 1× notional per instrument; no vol-targeting inside the family (it would cut the profitable weeks);
    the risk layer's daily loss gate (2 % of equity) is the only control; the account profile (U22) checks
    buying power and the day-trade count.
  The region has 3 + 2 × 5 × 4 = 43 cells per instrument, 86 in the headline portfolio. **Nothing in R₀ may
  change after registration; the specification curve shows every cell.**
- **Variants (≤ 11, reported).** Cadence 15 minutes; cadence every bar; first decision 09:35 (opening session
  included); exit 15:30 (avoids the close-flow period); VWAP stop on (b) and (c); "flat inside" on (b) and
  (c); the MODWT smooth slope (J = 3, 4) as a fourth estimator; a vol-targeted size (σ* = 10 %/√252 per day,
  cap 2×); R₀ on IWM and DIA; R₀ with the 15:55 fill instead of the auction fill; R₀ under a VIX ≥ 30 stress
  multiplier of 2.1× on half-spreads. The cost curve (0.3 / 1.0 / 2.3 bp and `measured`) is a response, not
  a trial.
- **State splits (reported).** Trailing-vol quintile (prior-session 5Min σ vs its causal 252-session
  distribution); VIX tercile at the open; day of week; FOMC / CPI / NFP days; OPEX days; revealed-gamma proxy
  (sign of the trailing 5-session lag-1 autocorrelation of 5Min returns); CBOE SKEW and VVIX terciles (after
  U26); year.
- **Expected magnitude (gross, from §1.2 and the RMV repo).** 2.5–5 bp per instrument-day; portfolio Sharpe
  0.9–1.3 net at 0.3–1.0 bp per round trip, ≈ 0.3 at 2.3 bp; vol ≈ 8 % at 1× notional; worst day ≈ −4 %;
  2016–2017-type years ≈ 0. The MDE line in the spec will show this is at the edge of what 2,400 days can
  confirm at Holm over four families; the forward test is where it is confirmed.
- **Costs.** 0.8–1.8 round trips per instrument-day. The verdict depends on the round-trip cost between 0.3
  and 2.3 bp; the dev-window report carries the curve and the registered floor is evaluated at the **measured**
  cost once U27 has 60 sessions of fills (until then at 1.0 bp).
- **Data.** Cached SPY, QQQ, IWM, DIA 5Min; 1Day bars for auction fills (U22); VIX, calendar (cached).
- **Budget.** 1 headline + 11 variants = 12 trials. The K₀ = 60 diagnostic looks are recorded against G1.

### G2 — Vol-managed core plus overlay (the portfolio claim; daily core, intraday overlay)

- **Mechanism.** F1's result stands as the core: scaling index exposure by σ*/σ̂ cuts drawdowns at about the
  same return (Moreira & Muir 2017); the F1 test showed +0.09 Sharpe and −16 points of drawdown but could not
  pass a test with a ± 0.3 interval. Adding a zero-beta, long-volatility overlay to a long-only core is the
  classic diversification: the overlay earns most when the core loses (2018, 2020, 2022).
- **Core claim.** Sharpe(core + k × G1) > Sharpe(core) at k = 1 (overlay notional = core notional, inside
  Reg-T intraday buying power), with a lower maximum drawdown, and the improvement not explained by leverage
  (the comparison at matched volatility is reported).
- **Headline.** Core = F1 headline as registered, restricted to SPY/QQQ 50/50 (σ̂ 21-day RV, σ* 15 %,
  exposure ≤ 1, rebalance on 10-point moves, re-run under U22's fills and the T-bill cash yield); overlay = G1
  headline at k = 1; paired Ledoit–Wolf Sharpe difference vs the core, one-sided; floors: overlay net return
  ≥ 3 %/yr at the registered cost, max DD(core + overlay) ≤ max DD(core).
- **Variants (≤ 5).** k ∈ {0.5, 2}; core = plain buy-and-hold; core = F1 on SPY/QQQ/IWM/DIA; overlay
  vol-targeted at 10 %.
- **Expected magnitude.** Core Sharpe ≈ 0.9; portfolio 1.2–1.5 at 1.0 bp cost; drawdown −18 % → ≈ −12 %.
- **Budget.** 6 trials. Runs only if G1's headline alpha is positive at the registered cost.

### G3 — Volatility-state gating of G1 (a second, pre-registered hypothesis on G1's stream)

- **Mechanism.** Hedging flows and imbalances scale with volatility; in calm regimes the band is narrow in
  dollar terms and the continuation is below cost. Realized volatility is persistent (unlike parameter rank,
  which the RMV repo showed is not), so a *state* filter can work where a *parameter* filter cannot. The RMV
  repo's deferred result: weekly net by trailing-vol quintile −3.3, +2.5, +6.3, +12.2, +11.2 bp; our band/SAR
  splits agree (§1.2).
- **Core claim.** Skipping sessions whose prior-session trailing 5Min σ lies in the bottom quintile of its
  causal trailing-252-session distribution raises G1's net Sharpe without lowering its net return by more
  than 10 %.
- **Headline.** G1 headline × gate (bottom quintile → flat); paired Ledoit–Wolf vs the G1 headline on the
  same days, one-sided; floor: net return ≥ 0.9 × G1's.
- **Variants (≤ 5).** Bottom two quintiles; VIX-at-open < 13 as the gate; gate on the band's dollar width
  < 2 × round-trip cost (the mechanism stated directly); continuous size ∝ vol quintile; gate on the
  revealed-gamma proxy (trade only when the trailing lag-1 autocorrelation is ≥ 0).
- **Expected magnitude.** +0.1 to +0.3 Sharpe; return −10 % to +5 %.
- **Budget.** 6 trials. Depends on G1.

### G4 — Time–frequency state features (no trial; a feature unit feeding G1/G3 splits)

- Goertzel band power of 5Min returns over the trailing 5 sessions at periods 2–6 bars vs 12–78 bars
  (spectral tilt: reversal-dominated vs continuation-dominated microstructure), spectral flatness, MODWT
  energy by scale (J = 1–5), vol-of-vol (VVIX; GARCH(1,1) innovation variance). All causal, per session,
  computed with `fastgoertzel` (C++) in one pass. Used as **reported** state splits of G1 and as inputs to a
  later meta-labeling overlay (stage G, only after G1 passes its forward test). Zero trials; it cannot be
  selected on.

### G5 — Opening-range breakout in "stocks in play" (gated on a point-in-time universe)

- **Mechanism.** Stocks with abnormal opening volume relative to their average carry news or attention; the
  first-five-minute range breakout continues through the day (Zarattini, Barbon & Aziz 2024, 7,000+ stocks
  2016–2023; QuantConnect's 2016 replication Sharpe 2.4). Capacity-limited, attention-driven, many small
  names: exactly where a retail account competes.
- **Why gated.** It needs a survivorship-free universe with delisted names and point-in-time listing, which
  Alpaca does not provide; a vendor (Polygon/Massive, EODHD, Norgate, Sharadar) costs money and a data unit
  (U29). Small-cap half-spreads of 5–20 bp and stop-order fills make the cost model, not the signal, the risk.
- **Budget.** 10 trials, after the user decides to buy the data. Headline and variants are specified in U29's
  spec file before any run.

### G6 — Volatility risk premium via SVXY (daily; optional; data already cached)

- PLAN2's F9 / U21 unchanged in substance: long SVXY (−0.5× VIX short-term futures since 2018-02-28) when
  VX1/VIX > 1.05 and VIX < 25, flat otherwise; hard rules (flat on backwardation, daily loss gate 3 %, kill at
  15 % drawdown). SVXY, VIX and VX1/VX2 are cached. Tested as an overlay (alpha vs zero, marginal to the core).
- **Budget.** 8 trials. Runs any time after U22; low priority because its tail risk sits outside what a
  twelve-month paper test can reveal.

### Closed (no further trials)

F2 TSMOM basket at gross ≤ 1, F3 overnight, F4 calendar, F5 textbook last-30-minute momentum, F7 gap fade,
F8 macro reaction, F11, the Goertzel cycle forecaster, same-slot return periodicity as a standalone rule.

## 4. Statistical protocol (v3; SPEC §22 in U22)

1. **Headline = region.** A rule family's headline is an equal-weight region fixed in the spec; cells are
   reported on a specification curve with intervals; the median cell's alpha and the share of cells with the
   headline's sign are the coherence statistic. No cell replaces the region.
2. **Overlay test.** Alpha of the overlay's daily net excess-return stream against zero, one-sided: circular
   block bootstrap (21 days, 5,000 resamples) of the mean, Newey–West t alongside; PSR(0). Pass at p < 0.05
   after Holm across the program's families (at most four at once).
3. **Portfolio test.** Ledoit–Wolf paired Sharpe difference, core + overlay vs core, one-sided, same
   bootstrap; the drawdown difference is reported with its interval.
4. **Floors.** Net return ≥ 3 %/yr at 1× notional (overlays); gross edge per round trip ≥ 2 × the round-trip
   cost at the measured cost (1.0 bp until U27 reports); max DD if set. The dev-window report shows every
   statistic at 0.3 / 1.0 / 2.3 bp and at `measured` once it exists.
5. **Power first.** Before registration the spec records the minimum detectable alpha at 80 % power for the
   family's n and the realized overlay volatility (`families/power.py`, U22); a family whose expected
   magnitude is below its MDE is registered as a diagnostic, not as a test, and says so.
6. **Looks, not trials.** `families/looks.jsonl` counts every configuration examined on the development
   window, starting with K₀ = 60; the report prints K and a Bonferroni bound next to the bootstrap p. DSR
   leaves the program verdict (it double-counted with Holm and deflated pre-registered headlines with the
   variance of unrelated cells); it stays a per-cell ledger diagnostic with N = the headline count.
7. **Cash and Sharpe.** Idle cash earns the 3-month T-bill; every Sharpe, alpha and PSR is computed on
   excess returns, benchmark included.
8. **Holdouts.** The quasi-holdout 2025-10-01 → 2026-09-30 stays a reported slice (contaminated by U11/U18
   buy-and-hold outcomes, not by this session's diagnostics, which stopped at 2025-09-30). The forward paper
   test (U27) is the holdout: a family is "consistent" after 26 weeks if its sign and t > 0 hold, "confirmed"
   only when PSR(0) ≥ 0.95 on forward days (≈ 150 weeks at Sharpe 0.9).
9. **Amendments** are new files with reasons, as in PLAN2; an amended family keeps its budget.

## 5. Units of work

Working protocol as PLAN2 (one unit per branch `unit/NN-slug`, done-gate = tests green + ruff clean + every
criterion demonstrated with command and output in the status note, one adversarial-reviewer pass, BREAKING /
SEVERE fixed, MINOR fixed or deferred with a reason, stop and report). Three additions: every unit states a
**compute budget** that is a done-when criterion; every look at the development window is appended to
`families/looks.jsonl` before its result is read; no family spec is registered without its MDE line (§4.5).

Dependency graph:

```
U22 engine corrections + protocol v3 ─┬─► U23 intraday kernels + region primary ─► U24 G1 dev test ─► U25 G2 + G3 ─► U27 forward test (paper)
                                      ├─► U26 time–frequency state + gamma proxies (splits for U24/U25; parallel with U23)
                                      ├─► U28 G6 SVXY (any time)
                                      └─► U29 point-in-time universe + G5 (gated on a data purchase)
U27 ─► U30 forward verdicts, registry, program summary ─► U31 (optional) meta-labeling overlay on a confirmed family
```

Cost estimates are for the i9 PC (`LOKY_MAX_CPU_COUNT=1`, flat pool). "Session" = one working session.

### U22 — Engine corrections and protocol v3

**Goal.** Fix what the two audits found, price the two assumptions that decide intraday verdicts, model the
account constraints that decide tradability, and replace the headline test with §4 — before any new family
runs.

**Scope** (SPEC §20 fills, costs and account; §21 region primaries; §22 protocol v3).
- **Auction fills.** `data/bars.load_bars(..., timeframe="1Day")` fetches and caches Alpaca 1Day bars (whose
  open and close are the auction prints; the closing print lies outside the 15:55 RTH bar). `COST_MODEL=quotes`
  prices `close`-kind fills at the 1Day close and `open`-kind fills at the 1Day open when the bar exists, else
  the 15:55 close / 09:30 open as today with a counted warning; `FILL_AUCTION ∈ {"print", "last_bar"}`
  (default `print`; `last_bar` keeps U18 behaviour for regression). The five-session check in §0.3 becomes a
  test on cached bars. MOC entries size shares from the 15:50 close, not the fill close.
- **Cost curve.** `SLIPPAGE_PCT` becomes `SLIPPAGE_BP: dict[kind, float]` with the U18 value (1.0 bp per side,
  every kind) as the regression default and a `measured` profile written by U27's reconciliation; the family
  report evaluates every statistic at round-trip costs {0.3, 1.0, 2.3} bp and at `measured` once it exists;
  a `STRESS_MULT` (default 1.0; variant 2.1 on VIX ≥ 30 sessions, the U13/audit measurement) multiplies the
  half-spread; the within-year cost look-ahead is removed by pricing each fill with the latest *completed*
  quarter's table.
- **Account profile** (`risk/account.py`, read by the simulator and the family report): equity, Reg-T buying
  power (4× intraday, 2× overnight for ≥ $25k; 1× cash), the pattern-day-trader counter (a margin account under
  $25k may make ≤ 3 day trades per 5 sessions), settlement for cash accounts, locate fees for shorts. A family
  report states whether its headline is tradable on the registered account (`INIT_CASH` ≥ $25k for intraday
  families) and what it would cost otherwise. Nothing in the profile changes a P&L; it refuses or flags.
- **Protocol v3** in `families/test.py`: `test: {kind: overlay_alpha | marginal | sharpe_vs_benchmark,
  sided: one | two}`; `overlay_alpha` (bootstrap mean vs zero + Newey–West t + PSR(0)); `marginal` (paired
  Ledoit–Wolf core + overlay vs core); excess returns everywhere (FRED DTB3 added to the exo layer,
  `available_at` next business day 09:00 ET; the simulator credits cash daily); `families/power.py` (ports the
  audit's `power.py` / `mde.py`: analytic MDE plus a 200-simulation check) and the spec keys
  `mde_alpha_bp_per_day`, `power` (required); floors as §4.4; coherence over region cells by alpha sign and
  the median cell's alpha; `families/looks.jsonl` (`python -m families look <label> <n>`, printed K); DSR out
  of `program_verdict`; `n_boot` 5,000 for headlines; `program.yaml` → `max_families: 5`, `max_trials: 42`
  (G1 12, G2 6, G3 6, G6 8, G5 10 reserved).
- **Pooling and benchmarks.** Equal-risk weights from the trailing 252-session σ at the family's start (ex
  ante); the report prints N_eff (the audit's formula); `buy_and_hold_er` renamed `constant_mix_er` and a
  drifting `buy_and_hold` added.
- **Region primaries.** `RulePrimary` gains `ALLOW_CONTINUOUS`: the primary frame carries a *target position*
  in [−1, +1] re-evaluated at every event; the portfolio simulator's roll path trades only the change (cost on
  the traded notional only, as rolled exits already do); `SIZE_STEP = 0` for such primaries; `rule_size` passes
  the target through; a region primary's target is the mean of its cells' positions.
- **Engine-audit items.** A generic fill-timing test over every sampler × exit model (`entry_pos == t + 1`
  for next-bar entries; every exit fills strictly after its decision bar; the audit's `plant.py` ENTRY and
  EXIT_HYST mutants must fail it); `test_events_and_widths_are_causal` re-written with per-event cuts (the
  VOL mutant must fail it); `data/fetch.py` defaults `--end` to HOLDOUT_START − 1 day and writes
  `data/cache/forward_access.jsonl` when told otherwise (the eleven ETF caches that run to 2026-10-08 are
  truncated on the next top-up and the event logged); schedule and MOC events use the exchange calendar, not
  bar existence, as the session clock; the barrier fill model gets a `slip_through` option (fill at the worse
  of barrier and next open) for any future triple-barrier family, off by default.
- **Diagnostics ledger.** `families/looks.jsonl` seeded with this session's K₀ = 60 (the §1 scripts, by name
  and configuration count); the scripts copied to `scripts/plan3_diagnostics/` unchanged.

**Done when.** (status note, 2026-10-10: every criterion with its command; SPEC §20–§22 describe the as-built)
- [x] Hand test: a 15:55-decided MOC fill on SPY 2024-03-04 is priced at the 1Day close (512.30) under
      `print` and at 512.25 under `last_bar` — `tests/test_u22.py::test_real_spy_2024_03_04_moc_fill…` on the cached
      bars (adjusted prices today: print 496.04 vs 15:55 close 496.00, +0.81 bp; the five sessions 03-04 → 03-08
      differ by 0.6–1.2 bp with no sign). F3's and F5's registered headlines re-run under `print` with every other
      switch at its old value, on a scratch ledger (`scripts/u22_checks.py print-rerun`): F3 pooled Sharpe 0.725 →
      0.707 (Δ −0.018; SPY 0.671 → 0.641, QQQ 0.761 → 0.756), F5 −0.122 → −0.164 (Δ −0.042; IWM −0.464 → −0.587, SPY
      0.062 → 0.035); the direction is the finding: the auction print costs these two overlays a little, within the
      expected ± 0.05. Every daily return changes (largest on the March 2020 closes, where the print sits up to 288 bp
      from the 15:55 bar).
- [x] Cost curve: `families/run.py` writes `cost_curve` (registered / 0.3 / 1.0 / 2.3 bp / measured) and the report
      prints it (tests in `tests/test_u22.py`, `tests/test_u17.py` end to end); `SLIPPAGE_BP` 1.0 with `last_bar`,
      `COST_TABLE=year`, `CASH_YIELD=none`, `STRESS_MULT` 1.0, `MOC_SIZE_FROM=fill` and `SESSION_CLOCK=data`
      reproduces U18's F5 AND F3 streams bit for bit: `scripts/u22_checks.py f5-parity --family F5 | F3` → the six
      headline cells' `daily_returns.csv` identical by sha256 (PASS; the adversarial review found the first version
      unpinned the MOC sizing and the session clock, so F3 did not reproduce until both got switches).
- [x] Account profile: `scripts/u22_checks.py account` — F5's registered headline on $10k is a PDT violation from
      its first session (2016-04-06: four day trades, one per instrument; 98.7 % of its day trades would be
      blocked); on $30k it is tradable; a cash account refuses the shorts, and `tests/test_u22.py` shows a cash
      account refusing an unsettled re-entry (good-faith violation) on synthetic trades.
- [x] `overlay_alpha` size and power (`scripts/u22_checks.py power --sims 1000 --n-boot 999`, 2,400 days, 8 % vol):
      size 0.050 (iid) and 0.045 (GARCH(1,1)-t) over 1,000 null simulations; power 0.905 on a planted 3 bp/day
      alpha (400 sims) and 0.795 at the analytic MDE of 2.56 bp/day (target 0.80: within 10 %). `marginal`
      reproduces the paired Ledoit–Wolf test when the overlay is a − core (`tests/test_u22.py`). Sharpe on excess
      returns (`scripts/u22_checks.py tilt`, FRED DTB3, mean 2.17 %/yr): the U18 benchmarks' Sharpe falls by
      0.10–0.29 (F1 0.114, F3 0.110, F5 0.114, F2 0.286, F4 0.206, F8 0.207), the audit's 0.11–0.28; the overlays
      fall further (F5 −0.12 → −0.97) because their simulated cash earned nothing — what `CASH_YIELD=tbill` now
      credits. MDE at Holm over four families: 3.17 bp/day = 8.0 %/yr = Sharpe 1.0 (`families power`).
- [x] A continuous-position rule that targets +1 every bar equals buy-and-hold intraday less one entry and one exit
      a day; a rule alternating ±1 every bar pays exactly c · (q_old + q_new) · px per bar; a region of two cells with
      opposite positions nets to zero trades (`tests/test_u22.py`, the `test_target` primary with `next_event` exits).
- [x] The two fill-timing mutants (ENTRY, EXIT_HYST) fail `tests/test_u22.py::test_the_fill_timing_test_catches…`
      and the VOL mutant fails the per-event causality test (`tests/test_methodology.py`); the eleven forward-window
      caches (DBC, EEM, EFA, HYG, IEF, LQD, SLV, SVXY, USO, UUP, VNQ) were truncated to HOLDOUT_START with one
      `truncate` record each in `data/cache/forward_access.jsonl` (`python -m data.fetch --truncate-forward`).
- [x] U12 parity PASS with the new switches at their old values (`scripts/u12_parity.py` on the final code:
      the four Stage C PWFO cells reproduce daily_returns.csv and returns_pwfo.csv by sha256, parity: PASS); `uv run pytest -q`: 804 passed (the full suite on the final code, 6 min); ruff clean.
- [x] Adversarial review (one pass): two BREAKING (the account module's chain convention and a flip inside a chain;
      the edge floor counted the cash interest as edge), two SEVERE (F3 not reproducible under the old switches: MOC
      sizing and the session clock were unpinned; the legacy backtest path silently dropped prints and the yield) and
      six MINOR findings — all fixed (`MOC_SIZE_FROM`, `SESSION_CLOCK`, `portfolio_path`, floors on excess returns,
      trading P&L without interest, hysteresis open fills at the print, slip-through leaving gap fills alone, the
      first-table counter by session, the counters surfaced in the report, exits before entries at one instant); the
      looks ledger is a protocol, not a lock (documented).

**Reviewer focus.** Auction fills that peek (the 1Day close is known only after 16:00: a `close` fill needs a
decision bar ≤ 15:55 and the next session's `open` fill must use that session's 1Day open); cost on netted
position changes; `looks.jsonl` bypasses; the account profile silently changing a P&L.

**Cost.** 2–3 sessions; compute minutes. Depends on nothing (the audits are done).

### U23 — Intraday kernels and the region primary

**Goal.** Compute every G1 estimator for every lookback in one pass, build region positions without a per-cell
backtest, and reproduce the harness's backtest on a sample — fast enough that a family run is minutes.

**Scope** (SPEC §21).
- `features/kernels.py` (numba, structure-of-arrays, float64 accumulators, no `fastmath`):
  - `rmedv_all(close, N_list)`: Siegel's repeated-median slope for every N in one pass (the RMV repo's kernel,
    pairwise medians over the last N closes; oracle `scipy.stats.siegelslopes(method="hierarchical")`);
  - `sg_velocity_all(close, N_list, degree)`: Savitzky–Golay derivative filters (Meyers' nth-order
    fixed-memory polynomial velocity is the degree-n endpoint derivative; one FIR per (N, degree));
  - `band_state(open, close, slot, lookback)`: Zarattini band distance with the gap adjustment, per slot;
  - `session_vwap(close, volume)`; `tod_sigma` from `features/vol_profile.py` reused for normalization;
  - all computed on the full RTH series with session boundaries respected (a window never straddles a
    session: the RMV repo's gap-contamination finding), NaN warm-up exactly N − 1 bars after each open.
- `primaries/region.py`: `region_trend` primary (G1): parameters = the region spec (estimator list, N list,
  θ list, VM list, cadence, first decision, exit, stop rule, mode); `rule()` returns the mean cell position in
  [−1, 1] at each scheduled decision bar; `cells()` returns every cell's position series for the specification
  curve; `needs_groups()` = the session group only.
- `wfo/position_backtest.py`: vectorized daily P&L of a position series (Σ pos × forward return − cost ×
  |Δpos|, auction fills via U22), used for the specification curve and the cost curve; **parity test** against
  `simulate_portfolio` through the rule pass on SPY 2024 (max |Δ daily P&L| < 1e-10 at `SIZE_STEP = 0`).
- `scripts/u23_kernels.py`: timings and the RMV-repo parity numbers (worked examples from Meyers 2005 p.2 and
  2025 p.2 = 1.0 exactly; 4,400 random (N, t) pairs vs scipy within float64).

**Done when.** (status note, 2026-10-10: every criterion with its command; SPEC §23 describes the as-built)
- [x] Oracles (`scripts/u23_kernels.py parity`, `tests/test_u23.py`): `rmedv_all` equals scipy's siegelslopes
      (hierarchical) on 4,400 random (N, t) pairs over N ∈ {6, 9, 12, 18, 24}, all 4,400 bit-equal (max |Δ| = 0);
      Meyers 2005 p.2 and 2025 p.2 worked examples = 1.0 exactly; `sg_velocity_all` degree 1 = the least-squares
      slope to 7.2e-16 (≤ 1e-12), degrees 2–3 exact on polynomials; the band state, VWAP and σ₅ match pandas
      references on the last 20 of 40 sessions with a missing bar, a 13:00 early close and a missing last bar.
- [x] Causality: every kernel at four cuts (mid-session, a session's first and last bar) with a planted peek per
      kernel caught (one bar; σ₅'s plant reads the current session, since a one-bar shift of a per-session value is
      known at the next open); `region_trend`'s cells at eight decisions with every bar after the decision bar
      replaced, and a planted one-bar peek in the repeated median caught; the U22 fill-timing invariant on the region's
      events (entries at the bar after the decision, exits at the next decision's fill or the 15:55 auction).
- [x] Session boundaries: a planted +5 % overnight gap leaves the first N − 1 velocities of the session NaN and the
      N-th on unchanged (rmedv, sgv degrees 1 and 2); windows allowed to straddle the open would read it (the test
      bites); SAR state resets every session (by hand).
- [x] Parity: `position_backtest` = `simulate_portfolio` through the rule pass, max |Δ daily return| = 0 (bit-exact)
      on cached SPY 2024 for three cells (rmedv N 12 θ 1.0; sgv N 24 θ 0.75; band VM 1.25) and the region, under
      profile `none` and a 2 % loss gate (quotes as-of costs, closing prints, T-bill yield); on a synthetic series
      where the gate fires; and over SPY and QQQ 2016-01-04 → 2025-09-30 with the calendar clock and the gate
      (`scripts/u23_kernels.py budgets`).
- [x] **Budgets** (i9, one core, `scripts/u23_kernels.py budgets`, timings only — no look): all estimators for SPY
      2016–2025 (190,402 bars, five lookbacks) 1.2 s warm (1.5 s first call; ≤ 5 s); 43-cell region positions at
      29,304 decisions 1.3 s (≤ 10 s); specification curve 43 cells × 4 costs 0.13 s (≤ 5 s); the G1 headline on
      SPY + QQQ including the 5,000-resample bootstrap 14 s (≤ 2 min).
- [x] Tests and lint: `uv run pytest -q` — 819 passed on the final code (after the review fixes, 5 min); ruff clean.
- [x] Adversarial review (one pass): no BREAKING or SEVERE finding; parity held bit-exact in every configuration the
      reviewer built (early closes, missing 09:30 / 10:00 / 15:30 / 15:55 bars, a dropped session under both clocks,
      cadence 5 / 15 / 60, `first` 09:35, `exit` 15:30, sign flips, targets of 5e-324, fractions 5e-13 apart, gates
      firing at decision bars, partial yields and prints). Four MINOR findings, all fixed: the `exit` "HH:MM" variant
      held an early-close (or missing-exit-bar) session into its auction (now flat from the last decision's fill);
      NaN targets were read as flat by `position_backtest` (now raise); numpy integers were refused as lookbacks;
      the √N decision was recorded in SPEC only (now §7 item 6).

**Reviewer focus.** Normalization leaking (σ from the current session), stop-and-reverse state carried across
sessions, decisions at HH:00/HH:30 using the bar that *ends* at that time (decision at the 09:55 bar's close
for the 10:00 mark, fill at the 10:00 bar's open).

**Cost.** 2 sessions. Depends on U22.

### U24 — G1: registration and development-window test

**Goal.** Register R₀ exactly as §3 G1 states it, run it once, report it with the cost curve, splits and the
specification curve, and decide whether G2/G3 run.

**Scope.** `families/G1.yaml` (headline = region; 11 variants; splits; `test: {kind: overlay_alpha, sided:
one}`; MDE line; floors at 1.0 bp pending the measured cost; `INIT_CASH` 30,000); a status-only smoke on
2016-01 → 2016-06 after the spec commit; registration; the run; the report; an adversarial review of
registration, fills and the bootstrap.

**Done when.**
- [ ] Spec committed before any ledger row; MDE recorded; `looks.jsonl` unchanged by the run.
- [ ] Report: headline alpha with its interval at four costs; the specification curve over 86 cells; splits;
      the quasi-holdout slice with its note; per-instrument results with N_eff; trade statistics (round
      trips/day, hit rate, skew, worst day, longest flat run); the account check.
- [ ] Decision rule applied as registered: G1 proceeds to U25 iff the headline passes `overlay_alpha` at 1.0 bp
      and the sign holds on ≥ 2/3 of cells; otherwise the family is closed and the program moves to U28/U29.

**Reviewer focus.** Any dependence of R₀ on the §1 diagnostics beyond what the spec states; the cost model's
fill kinds on half-hour fills (non-auction) vs the close (auction); the PDT counter on a 2-instrument region.

**Cost.** 1 session; compute ≤ 10 min. Depends on U23.

### U25 — G2 portfolio test and G3 volatility gate

**Goal.** Test the two claims that give G1 its economic meaning: it improves the core, and a volatility state
filter improves it.

**Scope.** `families/G2.yaml`, `families/G3.yaml` (as §3); the `marginal` kind for G2 (core stream = the F1
headline on SPY/QQQ re-run under U22's fills and cash yield, overlay = G1 headline at k = 1); G3 as a paired
one-sided test on G1's stream; a combined report with the portfolio equity curve, drawdowns, the matched-
volatility comparison and the buying-power check (overlay notional + core notional ≤ 2 × equity at every
decision).

**Done when.**
- [ ] Both specs registered after U24's decision; ≤ 6 trials each.
- [ ] Report: Sharpe(core + k × G1) vs Sharpe(core) with intervals at k ∈ {0.5, 1, 2}; the drawdown
      difference; G3's paired Δ Sharpe and Δ return at every cost.
- [ ] A `families/registry.yaml` entry for each passing claim with the exact cell hashes, code hash and freeze
      date (U27's input).

**Cost.** 1 session; compute ≤ 10 min. Depends on U24.

### U26 — Time–frequency state and gamma proxies (G4; splits only)

**Goal.** Add the state variables the mechanism points to, as reported splits and as inputs for a later
overlay — never as selection.

**Scope** (SPEC §15 additions). `features/groups.py`: `tf_state` (per session, causal, from the prior 5
sessions' 5Min returns): Goertzel band power via `fastgoertzel` at periods 2–6 vs 12–78 bars and their ratio
(spectral tilt), spectral flatness, MODWT energy by scale J = 1–5, the trailing lag-1 autocorrelation
(revealed-gamma proxy), realized vol-of-vol; `data/exo.py`: CBOE SKEW and VVIX histories (the VIX endpoint,
`available_at` 16:20 ET); `vol_state` gains `skew`, `vvix`, `vvix_vix`. `fastgoertzel` added as a pinned
dependency (parity with a numpy Goertzel on 1,000 random windows).

**Done when.**
- [ ] Causality and `available_at` tests for every new column; the stationarity guard passes.
- [ ] Diagnostic recorded: G1 headline P&L by tercile of each new state on the dev window (a split table in the
      G1 report, re-rendered); the §1.1 session-shift null test reproduced from the group's own code.
- [ ] Budget: the whole `tf_state` group for SPY 2016–2025 ≤ 10 s.

**Cost.** 1 session. Depends on U22; parallel with U23.

### U27 — Forward test: paper-trading loop with fill reconciliation (rewrite of PLAN2's U20)

**Goal.** Trade the registered G1/G2 (and G3 if adopted) on an Alpaca paper account from a freeze date,
measure fills, and score forward days as the holdout.

**Prerequisites (user decisions, recorded in the status note).** The Algo Trader Plus subscription
(real-time SIP; free-tier keys are refused SIP requests ending inside the last 15 minutes, and IEX bars are
unusable — RMV repo Unit 1); a paper account funded at the registered `INIT_CASH`; the PDT floor noted for any
later live account.

**Scope** (SPEC §18 amended). `live/`: `bars.py` (real-time 5Min bars from the SIP trade stream or the
latest-bars endpoint, with the exchange calendar as the session clock); `signals.py` (the frozen region
primary at the frozen code hash; refuses a mismatch); `orders.py` (marketable limit at the HH:00/HH:30 marks,
MOC via `cls` submitted by 15:50, flatten on failure); `reconcile.py` (every fill vs the modelled fill; the
`measured` slippage profile by kind written to `data/costs/measured_slippage.csv` with its sample size);
`monitor.py` (BOCPD on daily P&L, Adams & MacKay 2007 with a Student-t predictive; daily loss gate 2 %; kill
at −8 % from the registered expectation's lower band); `score.py` (stage H rows; the forward report). A
scheduled task on the i9 at 09:20, every half hour, and 16:10 ET. All forward-day reads through
`allow_holdout` with an audit record.

**Done when.**
- [ ] One week of one-share dry runs: orders, fills, reconciliation and scorer rows exist; modelled vs actual
      fill cost reported per kind.
- [ ] After 60 sessions: the `measured` cost profile is written and U24's report is re-rendered at it (the
      registered floor is then evaluated at the measured cost; the decision is recorded, not re-selected).
- [ ] Monitor tests: BOCPD flags a planted mean shift within 20 sessions; < 1 false flag per 1,000 stationary
      simulations at the chosen hazard.
- [ ] Determinism: a weekly job recomputes every forward day's signal from cached bars and matches the live
      log.
- [ ] Budget: per-decision compute + order submission ≤ 50 ms; the loop never blocks on a hung request
      (timeouts on every call); the external flatten job (the RMV repo's Unit 12b design) runs every 5 minutes.

**Cost.** 2 sessions of code; ≥ 12 months of calendar time. Starts the day U25 adopts a family.

### U28 — G6: SVXY volatility risk premium (optional, daily)

As PLAN2's U21 with protocol v3 (overlay alpha + marginal to the core), 8 trials, the February 2018 and March
2020 paths in the report. 1 session. Depends on U22 only.

### U29 — Point-in-time stock universe and G5 (gated)

**Goal.** Only if the user buys a survivorship-free source: `data/universe.py` with per-date membership
(listing, delisting, ticker changes; the "Q" ticker-reuse case as a test), the ORB primary (first-5-minute
range, relative-volume rank, ATR stop, close exit), a small-cap cost model from sampled quotes, and the G5
family spec with its MDE line. 2–3 sessions plus the data cost. Not scheduled until the decision is taken.

### U30 — Forward verdicts, registry and program summary

**Goal.** After ≥ 26 forward weeks: the interim forward report (sign, t, consistency with the registered
interval); after ≥ 252 sessions: the verdict (PSR(0) ≥ 0.95, Holm across forward families); the program
summary as a private artifact; the decision on whether to discuss real money. 1 session.

### U31 — Meta-labeling overlay on a confirmed family (optional)

PLAN2's U19 unchanged in design (stage G, ≤ 2 configurations, procedure bootstrap), with `tf_state`,
`vol_state`, `session` and `calendar_events` as features, applied only to a family that passed U30. Not before.

## 6. Computational performance (budgets are done-when criteria)

| stage | design | budget (i9, one core unless stated) |
|---|---|---|
| estimators (U23) | numba SoA kernels, one pass per estimator over all N; float32 storage, float64 accumulators; session-bounded windows | SPY 2016–2025, 5 lookbacks: ≤ 5 s |
| region positions (U23) | mean over cells at decision marks only (13 per session), not per bar | 43 cells, one instrument: ≤ 10 s |
| backtest (U23) | vector product per position series; the portfolio simulator only for the headline (parity-tested) | specification curve 43 × 4 costs: ≤ 5 s |
| family test (U22) | bootstrap on daily streams, vectorized resamples | 5,000 resamples: ≤ 10 s |
| state features (U26) | `fastgoertzel` (C++) batch over windows; MODWT from `features/causal_modwt.py` | `tf_state` SPY 2016–2025: ≤ 10 s |
| whole G1 run (U24) | rule pass, no model fits, no PWFO; flat pool over instruments × variants | SPY + QQQ + 11 variants: ≤ 10 min wall-clock |
| paper loop (U27) | session state precomputed at the open; per-decision update O(N) | ≤ 50 ms per decision |

Rules: no nested pools; no per-cell backtests for regions; no pandas below the reporting layer in kernels; no
`fastmath`; every budget measured in the status note with the command that measured it.

## 7. Open decisions (the user's)

1. **Algo Trader Plus** (≈ $99/month) before U27; without real-time SIP there is no intraday paper test.
2. **G5 data purchase** (a point-in-time universe with delisted names) before U29; otherwise G5 stays closed.
3. **Whether to run G6** (SVXY) at all; it costs one session and 8 trials and its tail risk is not learnable
   from a one-year paper test.
4. **`INIT_CASH` for the registered specs** ($30k proposed: above the PDT floor, so the paper account mirrors
   a tradable live account).
5. **The eleven forward-window caches**: truncate on the next top-up (proposed) or keep with a logged record.
6. **The velocity normalization R₀ registers (U24; raised by U23).** §3 G1 states v = slope · √N / σ₅ (the RMV
   repo's form, as built); §1.2's diagnostic SAR rule used slope / σ₅ without √N
   (`scripts/plan3_diagnostics/positions.py`). The diagnostic's "N = 6, 0.75 σ" cell is θ = 0.75 · √6 = 1.84 under
   √N, and R₀'s θ = 0.75 at N = 6 is 0.31 in the diagnostic's units, so §1.2's magnitudes do not describe R₀'s
   velocity cells as written. Options: register √N as stated (the θ grid then means "the window's move in σ₅ units
   of a √N-bar random walk"), or drop √N so the grid matches the diagnostic. Either is one parameter in
   `region_trend`'s normalization; it must be settled before the G1 spec is committed.

## 8. Deferred / out of scope

- Anything below 5Min decisions, maker strategies, futures (Alpaca has none), options strategies beyond G6
  (option history costs money and Alpaca's starts 2024-02).
- A learnable wavelet layer / deep sequence models: not before G1 passes its forward test.
- Real-money trading: a passed forward test is the precondition for the discussion.
- A point-in-time single-stock universe: only with G5 (U29).

## 9. Status

- U22 — built on `unit/22-engine-protocol-v3` (2026-10-10); SPEC §20–§22 written; every done-when criterion above
  demonstrated (U12 parity and the full suite re-run on the final code after the adversarial review's fixes). Data: auction prints cached for the 30-symbol universe through 2026-09-30
  (`1DayPrint`), FRED DTB3 cached, the as-of quotes table built, the eleven forward caches truncated and logged.
  `families/looks.jsonl` seeded with K₀ = 60. `families/program.yaml` scopes the caps to G1, G2, G3, G5, G6 (5 / 42).
  User decisions received 2026-10-10: Algo Trader Plus will be bought before U27; G5's data purchase still under
  consideration; G6 will run; `INIT_CASH` $30k agreed (account profile `margin_30k`).
- U23 — built on `unit/23-intraday-kernels` (2026-10-10); SPEC §23 written; every done-when criterion above
  demonstrated (`scripts/u23_kernels.py parity | budgets`, `tests/test_u23.py`); budgets 1.2 s / 1.3 s / 0.13 s / 14 s
  against 5 / 10 / 5 / 120 s; `position_backtest` bit-exact with the portfolio simulator over SPY and QQQ
  2016–2025. No look taken (the scripts print timings and parity residuals only; `families/looks.jsonl` unchanged).
  Open for U24: §7 item 6 (the √N normalization) before G1's registration.
- U24–U31 — not started.
