# Family G1

**Mechanism.** Within a session, a move that exceeds the typical move from the open for that time of day reveals a persistent order imbalance (institutional parent orders worked through the day, leveraged-ETF and option-hedging flows that buy into rises and sell into falls), which continues at the 30–90-minute horizon while the 5–10-minute horizon reverts; a smoothed velocity or a band breakout decided every 30 minutes and held until it reverses earns that continuation, long volatility and flat overnight (Zarattini, Aziz & Barbon 2024; Meyers 2025, 2026).

- Registered: c1689612eef33fd87f87ec99bc793cd3038bbc7b (2026-10-11); run at 2026-10-11T04:30:44.047948+00:00, code 2994b5e83c6e, git d0a31c0192af
- Instruments: QQQ, SPY (pooled, weights QQQ 0.50, SPY 0.50, equal weights); window 2016-01-04 → 2025-10-01; benchmark constant_mix_ew
- N_eff 1.06 of 2 instruments (the pooled stream's effective count)
- Test: overlay_alpha, one-sided, at cost 1.0; excess returns: FRED DTB3 as of each session's open, ACT/360
- MDE line: 3.17 bp/day at 80% power over 2400 days, vol 8.0%, 4 families; expected 3.00 bp/day → a DIAGNOSTIC (below its MDE)
- MDE at the realized excess volatility 6.6% over 2436 days: 2.58 bp/day (Sharpe 0.99; reported, the registered line is the one above)
- Trials: family 11 / budget 12; program 11 / 42 trials, 1 / 5 families

## Headline test

- Alpha 5.3 %/yr (2.09 bp/day) on 2436 days, excess of the T-bill; bootstrap t 2.59, Newey–West t 2.70; one-sided p = 0.0034 (block 21 sessions); interval 2.0 % … —; Sharpe 0.80 (benchmark 0.83)
- Looks: K = 158 configurations examined on the development window (families/looks.jsonl); Bonferroni bound p × K = 0.5371
- PSR(0) 0.995; DSR 0.476 (N = 55 program trials, V = 0.000987) — a ledger diagnostic, not part of the verdict
- Alpha 5.3 %/yr (NW t 2.74, p 0.0061), beta -0.00
- Floors: min_edge_to_cost 2.990 vs 2.000 → ok; min_net_ret 0.052 vs 0.030 → ok
- Coherence (cells, the verdict's): 86 (instrument, cell) points, share with the headline's sign 0.965 (needs 0.667), median cell 1.77 bp/day → coherent
- Coherence (variants, reported): 10 variants, share with the headline's sign 1.00, median variant vel_vwap_stop (floors yes) → coherent
- Before Holm across families: p 0.0034, floors yes, coherent yes, positive yes (at cost 1.0)

## Cost curve (every variant re-priced from its cost ledger; the verdict reads one column)

| variant | 0.3 | 1.0 | 2.3 | registered |
|---|---|---|---|---|
| cadence15 | SR 1.58, ret 11.0 %, floors yes | SR 1.24, ret 8.5 %, floors yes | SR 0.60, ret 3.9 %, floors no | SR 0.64, ret 4.2 %, floors no |
| cadence5 | SR 1.41, ret 10.1 %, floors yes | SR 1.02, ret 7.2 %, floors yes | SR 0.30, ret 1.9 %, floors no | SR 0.32, ret 2.0 %, floors no |
| exit1530 | SR 1.20, ret 7.0 %, floors yes | SR 0.91, ret 5.2 %, floors no | SR 0.39, ret 2.1 %, floors no | SR 0.28, ret 1.5 %, floors no |
| fill_1555 | SR 1.41, ret 9.4 %, floors yes | SR 1.14, ret 7.5 %, floors yes | SR 0.65, ret 4.1 %, floors no | SR 0.71, ret 4.5 %, floors no |
| first0935 | SR 1.40, ret 9.4 %, floors yes | SR 1.13, ret 7.5 %, floors yes | SR 0.64, ret 4.1 %, floors no | SR 0.70, ret 4.5 %, floors no |
| headline | 0.070 (p 0.000); SR 1.40, ret 9.4 %, floors yes | 0.053 (p 0.003); SR 1.13, ret 7.5 %, floors yes | 0.020 (p 0.152); SR 0.64, ret 4.1 %, floors no | 0.024 (p 0.108); SR 0.70, ret 4.5 %, floors no |
| modwt | SR 1.37, ret 9.2 %, floors yes | SR 1.11, ret 7.3 %, floors yes | SR 0.62, ret 3.9 %, floors no | SR 0.68, ret 4.3 %, floors no |
| stress_2x | SR 1.41, ret 9.5 %, floors yes | SR 1.15, ret 7.6 %, floors yes | SR 0.65, ret 4.1 %, floors no | SR 0.70, ret 4.4 %, floors no |
| vel_flat_inside | SR 1.24, ret 5.3 %, floors yes | SR 0.81, ret 3.4 %, floors no | SR 0.00, ret -0.1 %, floors no | SR -0.07, ret -0.4 %, floors no |
| vel_vwap_stop | SR 1.28, ret 9.2 %, floors yes | SR 1.02, ret 7.2 %, floors yes | SR 0.54, ret 3.6 %, floors no | SR 0.60, ret 4.1 %, floors no |
| vol_target | SR 1.55, ret 7.4 %, floors yes | SR 1.20, ret 5.6 %, floors yes | SR 0.54, ret 2.5 %, floors no | SR 0.62, ret 2.8 %, floors no |

Columns: `registered` = the simulated costs (quotes half-spread + slippage per side); a number = that round-trip cost in bp on every fill; `measured` = the forward test's fill reconciliation. The headline column shows its statistic and p first.

## Account check (risk/account.py; nothing here changes a P&L)

- margin_30k (margin account, equity 30,000): tradable
- Day trades: 10549 in total, at most 44 in 5 sessions; PDT rule applies: no; first flag None
- Gross exposure: 1.00× intraday (limit 4×), 0.00× overnight (limit 2×); first breach None
- What the PDT rule would cost on an account under the floor: 0.0 % of the day trades blocked (0)

## Data notes

- Headline members: 92 session(s) priced with the first as-of cost table (2016Q1), 0 bar(s) in stress sessions, 0 session(s) without a cash-yield rate

## Region cells (the specification curve over every cell; reported, the region is the headline)

- Parity of the re-simulation (wfo.position_backtest) with the headline members at the booked costs: QQQ max |Δ daily return| 0 over 2432 days; SPY max |Δ daily return| 0 over 2436 days

Region re-simulated at each cost: alpha bp/day (excess Sharpe). The headline test re-prices the members' cost ledgers instead (first order); the two agree to the re-pricing error.

| region | registered | 0.3 | 1.0 | 2.3 |
|---|---|---|---|---|
| QQQ | 1.35 bp (0.45) | 3.24 bp (1.09) | 2.54 bp (0.85) | 1.24 bp (0.41) |
| SPY | 0.58 bp (0.24) | 2.30 bp (0.97) | 1.64 bp (0.69) | 0.37 bp (0.15) |
| pooled | 0.96 bp (0.37) | 2.77 bp (1.06) | 2.09 bp (0.80) | 0.80 bp (0.31) |

- 86 (instrument, cell) points at cost 1.0: share with the headline's sign 0.965, median cell 1.77 bp/day; positive 83 / 86
- By estimator: band 6/6 positive, median 2.26 bp; rmedv 38/40 positive, median 1.42 bp; sgv 39/40 positive, median 1.89 bp

![region cells](cells_curve.png)

Every cell at cost 1.0, alpha bp/day (excess), sorted by the pooled cell:

| cell | QQQ | SPY | pooled | CI (pooled) | Sharpe (pooled) |
|---|---|---|---|---|---|
| sgv_n12_t1 | 3.89 | 3.24 | 3.56 | 0.86 … 6.26 | 0.88 |
| sgv_n12_t0.75 | 3.93 | 3.06 | 3.49 | 0.81 … 6.18 | 0.83 |
| sgv_n12_t1.5 | 3.36 | 2.87 | 3.11 | 0.83 … 5.39 | 0.86 |
| band_vm1 | 4.34 | 1.81 | 3.07 | 1.53 … 4.61 | 1.12 |
| sgv_n6_t1.5 | 3.44 | 1.87 | 2.65 | 0.51 … 4.79 | 0.71 |
| rmedv_n12_t1.5 | 3.30 | 2.00 | 2.65 | 0.38 … 4.91 | 0.74 |
| sgv_n9_t1 | 2.74 | 2.29 | 2.51 | 0.22 … 4.80 | 0.64 |
| band_vm1.25 | 2.96 | 1.94 | 2.45 | 0.96 … 3.93 | 0.98 |
| sgv_n18_t1.5 | 2.99 | 1.91 | 2.45 | 0.41 … 4.48 | 0.68 |
| rmedv_n24_t1.5 | 2.63 | 2.12 | 2.37 | 0.45 … 4.29 | 0.81 |
| sgv_n24_t1.5 | 2.43 | 2.09 | 2.26 | 0.33 … 4.18 | 0.76 |
| rmedv_n6_t1.5 | 3.22 | 1.24 | 2.23 | -0.05 … 4.50 | 0.59 |
| rmedv_n9_t1.5 | 2.41 | 2.01 | 2.21 | 0.09 … 4.33 | 0.61 |
| band_vm1.5 | 2.58 | 1.67 | 2.12 | 0.84 … 3.40 | 0.99 |
| rmedv_n18_t0.75 | 2.14 | 2.07 | 2.10 | -0.32 … 4.53 | 0.52 |
| sgv_n6_t1 | 3.63 | 0.56 | 2.09 | -0.44 … 4.62 | 0.48 |
| rmedv_n18_t1.5 | 2.79 | 1.09 | 1.94 | 0.07 … 3.81 | 0.59 |
| rmedv_n6_t1 | 3.35 | 0.44 | 1.89 | -0.63 … 4.41 | 0.44 |
| sgv_n9_t1.5 | 1.18 | 2.36 | 1.77 | -0.29 … 3.83 | 0.50 |
| sgv_n12_t2 | 2.51 | 1.04 | 1.77 | -0.08 … 3.63 | 0.56 |
| rmedv_n9_t2 | 1.39 | 2.02 | 1.70 | -0.21 … 3.61 | 0.56 |
| rmedv_n12_t2 | 2.44 | 0.85 | 1.64 | -0.36 … 3.64 | 0.52 |
| sgv_n6_t2 | 1.31 | 1.94 | 1.62 | -0.41 … 3.66 | 0.50 |
| sgv_n24_t0.75 | 2.65 | 0.55 | 1.60 | -0.74 … 3.93 | 0.41 |
| rmedv_n12_t1 | 2.02 | 1.15 | 1.58 | -1.19 … 4.36 | 0.38 |
| sgv_n18_t0.75 | 1.43 | 1.71 | 1.56 | -0.77 … 3.90 | 0.38 |
| sgv_n18_t2 | 1.63 | 1.49 | 1.56 | -0.15 … 3.26 | 0.55 |
| rmedv_n24_t0.75 | 1.60 | 1.44 | 1.51 | -0.75 … 3.78 | 0.39 |
| sgv_n24_t2 | 2.09 | 0.93 | 1.51 | -0.09 … 3.11 | 0.60 |
| sgv_n18_t1 | 1.55 | 1.46 | 1.51 | -0.76 … 3.77 | 0.38 |
| rmedv_n24_t2 | 1.87 | 1.12 | 1.49 | 0.01 … 2.98 | 0.63 |
| rmedv_n18_t1 | 2.04 | 0.94 | 1.49 | -0.74 … 3.71 | 0.38 |
| rmedv_n12_t0.75 | 1.07 | 1.72 | 1.39 | -1.31 … 4.09 | 0.34 |
| sgv_n24_t1 | 1.51 | 1.16 | 1.33 | -0.94 … 3.61 | 0.36 |
| sgv_n9_t2 | 1.42 | 1.11 | 1.26 | -0.64 … 3.16 | 0.41 |
| rmedv_n6_t0.75 | 2.64 | -0.30 | 1.17 | -1.29 … 3.63 | 0.26 |
| rmedv_n18_t2 | 1.41 | 0.74 | 1.07 | -0.59 … 2.73 | 0.41 |
| rmedv_n24_t1 | 1.25 | 0.83 | 1.04 | -1.15 … 3.23 | 0.29 |
| rmedv_n6_t2 | 0.95 | 0.72 | 0.83 | -1.02 … 2.69 | 0.26 |
| sgv_n6_t0.75 | 1.91 | -0.39 | 0.76 | -1.86 … 3.38 | 0.16 |
| sgv_n9_t0.75 | 0.03 | 0.39 | 0.21 | -2.22 … 2.64 | 0.05 |
| rmedv_n9_t1 | 0.06 | 0.18 | 0.12 | -2.23 … 2.47 | 0.03 |
| rmedv_n9_t0.75 | 0.48 | -0.44 | 0.02 | -2.46 … 2.51 | 0.01 |

*each cell re-simulated by wfo.position_backtest on the member's inputs; `measured` prices decision fills at the profile's `open` class and the closing fill at `close_auction` (`close` under FILL_AUCTION last_bar)*

## Trade statistics (headline; booked costs)

| instrument | round_trips_per_day | sessions_traded | long_pnl_bp_per_day | short_pnl_bp_per_day | long / short trades |
|---|---|---|---|---|---|
| QQQ | 1.002 | 0.995 | 1.041 | 0.449 | 15049 / 12497 |
| SPY | 0.977 | 0.992 | 0.562 | 0.000 | 14755 / 12195 |
| pooled | 0.989 | 0.994 | 0.802 | 0.224 |  |

Round trips per day: traded notional / the session's starting equity / 2, averaged over sessions. Long / short: trading P&L by side (booked costs) per day of the member's starting cash.

## Per-instrument headline alpha (cost 1.0; excess of the T-bill)

| instrument | days | alpha_bp | CI (NW 95 %) | sharpe |
|---|---|---|---|---|
| QQQ | 2432 | 2.54 | 0.82 … 4.26 | 0.85 |
| SPY | 2436 | 1.64 | 0.22 … 3.05 | 0.69 |

## Specification curve (reported; no variant is selected)

![spec curve](spec_curve.png)

| variant | status | start | days | sharpe | Δ vs bench | CI | p | Δ on headline days | net ret | max dd | floors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cadence15 | ok | 2016-01-25 | 2436 | 0.91 | 0.06 | 0.03 … — | 0.0006 | 0.06 | 8.5 % | -5.5 % | yes |
| cadence5 | ok | 2016-01-25 | 2436 | 0.71 | 0.05 | 0.02 … — | 0.0034 | 0.05 | 7.2 % | -10.6 % | yes |
| exit1530 | ok | 2016-01-26 | 2435 | 0.54 | 0.03 | -0.00 … — | 0.0546 | 0.03 | 5.2 % | -12.0 % | no |
| fill_1555 | ok | 2016-01-25 | 2436 | 0.81 | 0.05 | 0.02 … — | 0.0032 | 0.05 | 7.5 % | -9.2 % | yes |
| first0935 | ok | 2016-01-25 | 2436 | 0.80 | 0.05 | 0.02 … — | 0.0034 | 0.05 | 7.5 % | -9.7 % | yes |
| headline | ok | 2016-01-25 | 2436 | 0.80 | 0.05 | 0.02 … — | 0.0034 | — | 7.5 % | -9.7 % | yes |
| modwt | ok | 2016-01-25 | 2436 | 0.77 | 0.05 | 0.02 … — | 0.0050 | 0.05 | 7.3 % | -10.4 % | yes |
| stress_2x | ok | 2016-01-25 | 2436 | 0.81 | 0.05 | 0.02 … — | 0.0030 | 0.05 | 7.6 % | -9.3 % | yes |
| vel_flat_inside | ok | 2016-01-25 | 2436 | 0.30 | 0.01 | -0.01 … — | 0.1564 | 0.01 | 3.4 % | -6.1 % | no |
| vel_vwap_stop | ok | 2016-01-25 | 2436 | 0.71 | 0.05 | 0.02 … — | 0.0070 | 0.05 | 7.2 % | -9.8 % | yes |
| vol_target | ok | 2016-02-03 | 2429 | 0.73 | 0.03 | 0.01 … — | 0.0064 | 0.03 | 5.6 % | -5.4 % | yes |

Coherence reads each variant's Δ on the days it shares with the headline (column 'Δ on headline days'); the curve and the p-values are each variant's own sample.

## Responses

| variant | crisis_return | exposure | hit_rate | longest_flat_run | max_dd | mean_per_trade_bp | ret_ann | sharpe | skew | vol_ann | worst_day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cadence15 | 2020-02-19..2020-03-24: -2.1 %; 2022-01-03..2022-10-13: 20.8 % | 0.998 | 0.261 | 2 | -0.068 | -9.488 | 0.062 | 0.914 | 1.824 | 0.068 | -0.026 |
| cadence5 | 2020-02-19..2020-03-24: -5.8 %; 2022-01-03..2022-10-13: 19.8 % | 0.998 | 0.236 | 2 | -0.128 | -10.122 | 0.049 | 0.710 | 2.491 | 0.070 | -0.028 |
| exit1530 | 2020-02-19..2020-03-24: -9.3 %; 2022-01-03..2022-10-13: 9.7 % | 0.997 | 0.285 | 2 | -0.121 | -11.075 | 0.030 | 0.535 | 1.269 | 0.058 | -0.025 |
| fill_1555 | 2020-02-19..2020-03-24: -4.4 %; 2022-01-03..2022-10-13: 19.4 % | 0.997 | 0.291 | 2 | -0.092 | -10.346 | 0.052 | 0.808 | 1.640 | 0.065 | -0.026 |
| first0935 | 2020-02-19..2020-03-24: -5.0 %; 2022-01-03..2022-10-13: 19.4 % | 0.997 | 0.290 | 2 | -0.097 | -10.383 | 0.052 | 0.801 | 1.845 | 0.066 | -0.026 |
| headline | 2020-02-19..2020-03-24: -4.9 %; 2022-01-03..2022-10-13: 19.5 % | 0.997 | 0.291 | 2 | -0.097 | -10.369 | 0.052 | 0.802 | 1.845 | 0.066 | -0.026 |
| modwt | 2020-02-19..2020-03-24: -5.7 %; 2022-01-03..2022-10-13: 19.0 % | 0.998 | 0.286 | 1 | -0.105 | -10.786 | 0.050 | 0.775 | 1.694 | 0.066 | -0.024 |
| stress_2x | 2020-02-19..2020-03-24: -4.5 %; 2022-01-03..2022-10-13: 19.5 % | 0.997 | 0.291 | 2 | -0.093 | -10.402 | 0.052 | 0.813 | 1.917 | 0.065 | -0.025 |
| vel_flat_inside | 2020-02-19..2020-03-24: -3.2 %; 2022-01-03..2022-10-13: 6.1 % | 0.997 | 0.281 | 2 | -0.062 | -8.389 | 0.012 | 0.295 | 2.024 | 0.043 | -0.020 |
| vel_vwap_stop | 2020-02-19..2020-03-24: -5.6 %; 2022-01-03..2022-10-13: 12.8 % | 0.997 | 0.247 | 2 | -0.098 | -10.130 | 0.049 | 0.714 | 1.924 | 0.070 | -0.023 |
| vol_target | 2020-02-19..2020-03-24: 1.3 %; 2022-01-03..2022-10-13: 11.0 % | 0.998 | 0.291 | 2 | -0.063 | -10.335 | 0.033 | 0.728 | 1.462 | 0.047 | -0.021 |

Responses, splits and slices below: excess of the T-bill, at cost 1.0; mean_per_trade_bp and hit_rate are per position (a chain of rolled legs, ended at a side flip: Σ cash pnl / its largest leg notional) from the booked-cost trades.

## State splits (headline; reported with 95 % Newey–West intervals, not tested)

**day_of_week**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| Friday | 490 | 3.03 | -0.56 … 6.62 | 1.17 |
| Monday | 454 | 2.52 | -1.10 … 6.14 | 1.07 |
| Thursday | 492 | 2.31 | -1.75 … 6.37 | 0.87 |
| Tuesday | 502 | -0.81 | -3.85 … 2.23 | -0.33 |
| Wednesday | 498 | 3.47 | -0.58 … 7.52 | 1.21 |

- gamma_sign: unavailable

**macro_day**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| macro | 299 | 0.28 | -4.57 … 5.13 | 0.10 |
| other | 2137 | 2.34 | 0.69 … 3.99 | 0.91 |

**opex_day**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| opex | 116 | 4.29 | -1.70 … 10.29 | 1.74 |
| other | 2320 | 1.98 | 0.38 … 3.57 | 0.76 |

**vix_tercile**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| high | 812 | 4.20 | 0.21 … 8.18 | 1.09 |
| low | 812 | 0.18 | -1.15 … 1.52 | 0.14 |
| mid | 812 | 1.88 | -0.08 … 3.85 | 0.96 |

**vol_quintile**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| n/a | 363 | -0.46 | -3.01 … 2.08 | -0.30 |
| q1 low | 497 | 0.49 | -1.78 … 2.76 | 0.28 |
| q2 | 362 | 0.58 | -2.12 … 3.28 | 0.31 |
| q3 | 365 | 0.71 | -2.35 … 3.77 | 0.36 |
| q4 | 404 | 4.25 | 0.16 … 8.33 | 1.63 |
| q5 high | 445 | 6.35 | 0.45 … 12.26 | 1.44 |

**year**

| group | n_days | mean_bp | ci_bp | sharpe |
|---|---|---|---|---|
| 2016 | 238 | -0.62 | -4.22 … 2.98 | -0.36 |
| 2017 | 251 | 0.12 | -1.87 … 2.10 | 0.10 |
| 2018 | 251 | 7.69 | 2.32 … 13.07 | 2.48 |
| 2019 | 252 | -0.58 | -2.91 … 1.76 | -0.43 |
| 2020 | 253 | 2.30 | -3.69 … 8.29 | 0.68 |
| 2021 | 252 | 1.27 | -1.74 … 4.28 | 0.67 |
| 2022 | 251 | 7.15 | -0.89 … 15.19 | 1.75 |
| 2023 | 250 | 1.67 | -1.41 … 4.75 | 0.80 |
| 2024 | 252 | 2.38 | -1.34 … 6.11 | 1.07 |
| 2025 | 186 | -1.59 | -8.31 … 5.13 | -0.46 |

## Sample splits (headline; reported)

| split | n_days | sharpe | bench_sharpe | ret | max_dd | mean_bp |
|---|---|---|---|---|---|---|
| 2016_2019 | 992 | 0.84 | 1.17 | 17.6 % | -4.5 % | 1.68 |
| 2020_2025 | 1444 | 0.80 | 0.72 | 38.5 % | -9.7 % | 2.37 |
| iwm_dia | 2436 | -0.08 | 0.59 | -6.2 % | -16.8 % | -0.20 |

## Quasi-holdout slice (reported; never a gate)

*quasi-holdout 2025-10-01 → 2026-09-30: contaminated (the U11 holdout exposed these days' buy-and-hold outcomes before the families were chosen); a robustness slice, never a gate*

| slice | n_days | sharpe | bench_sharpe | ret | max_dd | mean_bp |
|---|---|---|---|---|---|---|
| 2025-10 → 2026-09 | 250 | -0.61 | 1.18 | -3.4 % | -5.3 % | -1.34 |

## Per-instrument headline Sharpe (raw and James–Stein shrunk toward the family mean)

| instrument | sharpe | sharpe_js |
|---|---|---|
| QQQ | 0.85 | 0.85 |
| SPY | 0.69 | 0.69 |

## Registered vs run spec

```
+registered: {sha: 'c1689612eef33fd87f87ec99bc793cd3038bbc7b', date: '2026-10-11'}
```
