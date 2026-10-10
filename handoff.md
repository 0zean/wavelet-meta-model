# Session Handoff — U18 Phase 1 family tests (merged): all eight families fail

## Where it started
The user asked me to review `handoff.md` (U17 merged as PR #19; the XGBoost thread fix merged as PR #20) and begin U18
per PLAN2.md. U18 is complete and merged as PR #21 (merge commit aa922fb). Every pre-registered hypothesis family was
tested once on the development window (2016-01-04 → 2025-09-30). **None passes**, so U19 (ML overlay) and U20 (paper
trading) have nothing to work on.

## Results (program summary: `results/families/program_summary.md`)

| family | Sharpe vs equal-risk B&H | Δ (95 % CI) | p (Holm) | fails on |
|---|---|---|---|---|
| F1 vol-managed | 0.92 vs 0.83 | +0.09 (−0.22 … 0.40) | 0.55 (1.00) | p; net-return floor (11.5 % vs 12.0 %) |
| F2 TSMOM basket | 0.23 vs 0.86 | −0.63 (−1.65 … 0.38) | 0.21 (0.83) | p; floor; coherence |
| F3 overnight | 0.72 vs 0.90 | −0.18 (−0.72 … 0.36) | 0.50 (1.00) | p; floor; coherence |
| F4 calendar | 0.57 vs 0.67 | −0.10 (−0.91 … 0.71) | 0.81 (1.00) | p; coherence |
| F5 intraday momentum | −0.12 vs 0.83 | −0.95 (−2.00 … 0.10) | 0.07 (0.43) | p; floors (gross edge 0.8× cost) |
| F7 gap fade | −0.69 vs 0.87 | −1.55 (−2.55 … −0.56) | 0.006 (0.044) | significantly worse than B&H (sign verified by hand) |
| F8 macro reaction | −0.29 vs 0.71 | −0.99 (−2.06 … 0.07) | 0.07 (0.43) | p; floors (gross edge negative) |
| F11 U11 cells, tod σ | −0.30 vs 0.99 | −1.29 (−2.34 … −0.24) | 0.02 (0.13) | p; floors; coherence |

- Program: 8 / 8 families, 44 / 112 trials. F10 was not run (its cross-sectional rank is not built).
- F11 answers its closure question: deseasonalized volatility reveals no edge in the best U11 cells. The paired tod −
  plain σ difference is −0.40 (p 0.29) at 30Min and −0.11 (p 0.77) at 1Hour. It is not significantly worse either, so
  do not say "tod made it worse".
- Published as a private artifact: https://claude.ai/artifact/5ca2XK2Zc4uGoUdcbLMF5v (version 2, includes F11). To
  update it, regenerate it with the session scratchpad script `make_summary_page.py`, which is not in the repo.

## Decisions locked (user)
- **Benchmark** `buy_and_hold_er` for every family (PLAN2 said EW): it matches the pooled streams' equal-risk weights.
- **Risk profile.** `none` for single-instrument families, `basket` for F2. PLAN2 named `standard`, whose 20 %
  position cap, drawdown tiers and loss gate would have replaced the rules under test.
- **F2 not amended.** The review found that its construction caps low-vol assets at m = 1, so the bond / credit /
  dollar sleeve carries ~0.6–0.7 of the other assets' risk, which is not PLAN2's equal-risk basket. The user chose to
  keep F2 closed as registered.
- **Approval before registration.** Each spec batch was approved by the user before registering: F1–F4 at e19b44f,
  F5/F7/F8 at dd8228d, F11 at ce5dcd2.
- **Smoke runs.** From the second round on, they ran only after the spec commit, on short early windows, status only.

## What shipped (SPEC §17.6 has the as-built detail)
- **calendar_drift** takes a window list: one position over the union of windows (schedule `windows`, daily MOC
  entries rolled, never stacked).
- **`basket` risk profile**: gross ≤ 1, each side ≤ 1, 50 bp/yr borrow. A basket family's sample split runs its own
  cell against its own benchmark.
- **Family spec keys**:
  - `per_instrument` patches;
  - per-variant `timeframe`;
  - `legs`: one cell per instrument and leg, summed per instrument, and the run refuses legs that overlap in time on
    any bar;
  - model cells (`model`, `feature_groups`, `pwfo`, `meta_train`, `seed`).
- **Stage F** may hold model cells. Every family-stage cell counts 0 trials; the variant row counts 1.
- **Edge-floor totals**: WFO and `average` PWFO rows carry pnl / cost_paid / traded_notional / init_cash, and rule
  cells' trades.csv carry entry_time / exit_time.
- **Fixes**:
  - registration decoded `git show` with cp1252, which refused a spec containing non-ASCII text;
  - the summary DSR is recomputed with the program's trials at summary time;
  - the coherence column shows the verdict;
  - a chain of rolled trades counts as one position;
  - families sort naturally.
- `tests/test_u18.py` has 23 tests.

## Key files
- `families/F1.yaml` … `F5.yaml`, `F7.yaml`, `F8.yaml`, `F11.yaml`. Each header lists its choices and its deviations
  from PLAN2.
- `results/families/<id>/` (report.md / .html, result.json, streams.csv, spec_curve.png) and `program_summary.*`.
- `results/ledger.jsonl`, the canonical program ledger, holds every family variant row and its member cells.
- PLAN2.md U18 status note: three rounds, each with its review findings.
- Memory files touched: none.

## Running state
- `main` = aa922fb (PR #21 merged). The local branch `unit/18-family-dev-tests` and its remote are kept.
- Background processes: none.
- Data caches were topped up to 2026-09-30 by the quasi-holdout cells and committed: DIA, IWM, QQQ, GLD, TLT, XLK.
  TLT's whole history was re-adjusted for a dividend partway through the program (×0.996), so F1's tlt_gld split and
  F4's dev TLT leg ran on the old vintage.
- Scratch (disposable) in the session scratchpad:
  - smoke runs: `smoke18/`, `smoke18b/`, `smoke18c/`;
  - reviewer probes: `review18/`, `review18b/`, `review18c/`;
  - U12 parity: `parity18/`;
  - the artifact page and its generator.

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q`: 772 passed (~7 min). Run it in
  the foreground or into a log file; piping a background run through `tail` lost its output twice.
- `uvx ruff check . && uvx ruff format --check .`: clean.
- U12 parity: `uv run python scripts/u12_parity.py spec P/parity.yaml`, then
  `python -m experiments --root P --ledger P/ledger.jsonl run P/parity.yaml --jobs 8`, then
  `scripts/u12_parity.py check P P/ledger.jsonl`. PASS (8/8).
- Re-verify a family's registration:
  `uv run python -c "from families.spec import check_registered as c; print(c('families/F11.yaml'))"`.
- Regenerate the summary: `PYTHONIOENCODING=utf-8 MPLBACKEND=Agg uv run python -m families summary`.

## Deferred and known caveats (all disclosed in the PLAN2 U18 note)
- **Close fills.** "MOC" fills are priced at the last 5Min bar's close (the last continuous trade before 16:00), not
  the closing-auction print. This affects F3, F4 and F5. U20's reconciliation would measure it.
- **Size rounding.** SIZE_STEP 0.1 rounds rule magnitudes, so m < 0.05 is never traded: about 12 % of F8's events,
  and F5's thr0 behaves as an implicit 0.05σ threshold.
- **`abs_move_tercile`** uses the same day's |benchmark move|, so it is outcome-conditioned for intraday legs.
- **PWFO cost totals** cover each combo's full OOS span while the stream covers only the common span (F11 edge/cost
  ≈ 0.70 vs 0.78).
- **XLK** carries about 4 % of F11's pooled risk.
- **Quasi-holdout benchmark** drops its first day (2025-10-01) in every family; reported slices only.
- **Borrow** is charged for bars held + 1 (≈ 20 % over on weekly shorts, ~0.03 %/yr).
- **Not built**: F10's cross-sectional rank, `path_monotone`, the gamma proxy, stage G (U19).

## Open (the user's call)
- **The program's next step.** All family slots are used and nothing passed. Continuing means amending the program
  (e.g. raising `families/program.yaml`'s family cap for new hypotheses, which needs new PLAN2 text) rather than
  running U19 / U20 as planned. The alternative is to close the program with the null as its result.
- **The handoff commit is unpushed.** This handoff update is committed on local `main` and not pushed.
- **Remote branches.** Whether to delete the remote `origin/unit/*` branches (carried over).

## Pick up here
Ask the user how to proceed with the program, given that every family failed. Do not start U19 or U20: there is no
passing family to overlay or paper-trade.
