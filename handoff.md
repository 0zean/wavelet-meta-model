# Session Handoff — U13 exogenous data + quotes cost model (complete)

## Where it started
U12 had been merged into `main` (PR #14, 0116b75). The user asked me to review this file and begin U13. U13 was built,
reviewed and verified on `unit/13-exo-data-costs` (branched off `main`).

## What shipped (commits on `unit/13-exo-data-costs`)
- 8c5770e — exogenous data layer, event tables, quotes sampler, per-fill cost model.
- 7a21cfb — `HOLDOUT_START` moves to 2026-10-01 (SPEC §11.1; separate so it can be reverted alone).
- 44a59b7 — review fixes.
- a1df672 — data: quotes half-spread table, exo series cache, smoke results, new-ETF quality report.
- ec706e8 — PLAN2 status note, SPEC notes, event-session cost comparison.

## Decisions locked
- **One cost knob.** `RunConfig.COST_MODEL` ∈ {slippage, cs, quotes}, default `quotes`.
  - `RiskProfile.spread` was removed. The old U8–U11 `standard` = `RISK_PROFILE="standard"` + `COST_MODEL="cs"`.
  - Any non-slippage model backtests through `simulate_portfolio` and needs `POSITION_MODE="single"`.
  - Tests on synthetic data pin `COST_MODEL="slippage"`.
- **Auction fills.** They pay the auction proxy only: the 09:31 / 15:59 quoted half-spread, which is conservative for stocks' open.
- **Quotes table.**
  - It is the cost of an ordinary session. Event sessions were measured at about 1.0× the table and stress sessions at 2–4×.
  - Not adjusted. Whether to scale by volatility is a U17 family-spec decision.
- **Earnings acceptance times** come from EDGAR filing index pages. The submissions JSON is wrong by one NY offset for AAPL, AMZN, META, JPM and UNH.
- **Calendar.** The CSVs are checked in and rebuilt with `uv run python -m data.events` (network: Fed, BLS via curl, SEC).
- **Parity.** The U12 parity re-run with `COST_MODEL=slippage` passes (`scripts/u12_parity.py` OLD_SWITCHES updated).

## Key files
- `data/exo.py`, `data/events.py`, `data/quotes.py`, `data/options.py`, `risk/costs.py` (`fill_costs`, `auction_flags`).
- `data/calendar/{events,earnings}.csv`, `data/costs/quotes_half_spread.{csv,json}`.
- `scripts/u13_smoke.py exo|quotes` → `results/u13/`.
- `tests/test_exo.py`, `tests/test_costs.py`, `tests/test_u13_review.py`.
- PLAN2 U13 status note; SPEC §12 / §19 "Implementation (U13, as built)".

## Running state
- Background processes: none.
- Worktrees: the main checkout is on `unit/13-exo-data-costs`. `wavelet-meta-model-stage-b` is carried over, untouched.
- Nothing pushed.
- **Uncommitted, the user's call:**
  - The 11 new ETFs' bar caches (`data/cache/sip/all/5Min/{IEF,…,SVXY}.{npz,json}`, 141 MB). The earlier bar cache was committed once "for continuity between runs". Add them, or leave them local: `python -m data.fetch` re-creates them in ~10 min.
  - The raw quote samples (`data/cache/exo/quotes/`, git-ignored). Re-fetching takes about 3 h at Alpaca's 200 req/min; the table is the committed product.

## Verification
- Tests: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q` → 508 passed.
- Lint: `uvx ruff check . && uvx ruff format --check .` → clean.
- Exo smoke: `uv run python scripts/u13_smoke.py exo`. VIX vs VIXCLS max |Δ| 0.00; VX1/VIX median 1.043.
- Quotes smoke: `uv run python scripts/u13_smoke.py quotes`. SPY 0.155 bp vs CS 1.53 bp; `event_vs_base` ratios.
- Parity: `uv run python scripts/u12_parity.py spec P.yaml`, then `python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 24` (with `LOKY_MAX_CPU_COUNT=1`), then `scripts/u12_parity.py check R R/ledger.jsonl` → PASS.

## Deferred + open questions
- **Stress-session costs (2–4× the table):** whether to scale with volatility (U17).
- **No usable HY OAS** (FRED from 2023-10 only): the credit state needs e.g. HYG vs IEF (U15).
- **VIX holiday rows from 2022:** build ratios on common dates (U15).
- **Exo `coverage_end` = fetch day:** a same-day morning fetch isn't refreshed (U20).
- **Option chain:** forward-only (`python -m data.options`). The U15 gamma proxy needs the VIX/skew fallback; a daily collection job could start with U20.
- **SVXY regime break 2018-02-28** (`data.fetch.REGIME_BREAKS`): F9/U21 must start after it or model it.
- **Carried over:**
  - Removing the stage-b worktree, and pushing / merging branches (the user's call).
  - The U12 cost-target tuning.
  - The 1Day `MIN_VAL_EVENTS` calibration question (U17/U18).

## Pick up here
Review, then PR / merge `unit/13-exo-data-costs` (the user's call), and decide whether to commit the new ETF bar caches. Next: U14 (samplers, exits, ISOM vol profile), on its own branch off `main`. U15 (state features) needs U13 and can follow.
