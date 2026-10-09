# Session Handoff — U13 exogenous data layer + quotes cost model (merged to main)

## Where it started
The user asked me to review `handoff.md` (U12 already merged as PR #14) and begin U13 per PLAN2.md and SPEC §12 / §19.
U13 was built, adversarially reviewed and verified on `unit/13-exo-data-costs`. Then, at the user's request, the
branch was pushed with the new ETF bar caches, opened as PR #15 and merged into `main` (merge commit 724eff3).

## Decisions locked + what shipped
- **Exogenous series** (`C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\data\exo.py`):
  - `load_series` serves cboe (the VIX family; continuous VX1/VX2 and the ratios VX1_VIX, VX2_VX1) and fred (VIXCLS, DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2, DTWEXBGS).
  - `available_at` rules: CBOE 16:20 ET same day; VIXCLS next federal business day 09:00; H.15 series and HY OAS next business day 16:30; DTWEXBGS the following Monday 16:30.
  - Each fetch downloads the full history and replaces the cache; revisions are logged.
  - VX rows start at the first loaded expiry (2015-01-21).
- **Event tables** (`...\data\events.py`; CSVs in `...\data\calendar\events.csv` and `earnings.csv`):
  - FOMC, CPI, NFP, OPEX, MONTH_END, TOM and PRE_HOLIDAY for 2016–2027, with `known_from` rules.
  - Earnings come from SEC 8-K Item 2.02, with acceptance times read from the EDGAR index pages. The submissions JSON is off by one NY offset for AAPL, AMZN, META, JPM and UNH.
  - Rebuild with `uv run python -m data.events`; bls.gov needs curl.
- **Quotes cost model:**
  - Sampler (`...\data\quotes.py`): sessions 6–10 of Feb/May/Aug/Nov, 5-minute marks plus `open_auction` (open+1 min) and `close_auction` (close−1 min).
  - Table: `...\data\costs\quotes_half_spread.csv` (+ `.json`), 9,570 rows covering 30 symbols × 2016–2026.
  - Per-fill pricing: `risk\costs.py` `fill_costs` / `auction_flags` / `quotes_half_spread`, selected by `RunConfig.COST_MODEL` (slippage | cs | quotes, default quotes).
- **Cost wiring:**
  - `RiskProfile.spread` is removed. The old U8–U11 `standard` = `RISK_PROFILE="standard"` + `COST_MODEL="cs"`.
  - Any non-slippage model backtests through `simulate_portfolio` and needs `POSITION_MODE="single"`.
  - `cost_data` replaces `spread_bars` in `wfo\backtest.py`, `wfo\pwfo.py`, `experiments\runner.py` (data hash `cost:{sym}`), `risk\run.py` and `wavelet_meta_model.py`.
- **Holdout:** `HOLDOUT_START` is now 2026-10-01 (`data\bars.py`, commit 7a21cfb).
- **Universe:** `DEFAULT_UNIVERSE` += IEF LQD HYG DBC USO UUP EFA EEM VNQ SLV SVXY (`data\fetch.py`, with `REGIME_BREAKS` for SVXY at 2018-02-28). Their 5Min caches are committed (46abeca, 141 MB).
- **Option chain** (`data\options.py`): forward-only collector. Alpaca has no historical chain or OI, so the U15 gamma proxy must use the VIX/skew fallback.
- **Measured results:**
  - VIX vs VIXCLS max |Δ| 0.00. VX1/VIX median 1.043 (2016–2025).
  - SPY quoted half-spread 0.155 bp vs Corwin–Schultz 1.53 bp.
  - Event sessions cost 0.99–1.03× the table; stress sessions 2.1× (up to 4.7×). Not adjusted.
  - BAMLH0A0HYM2 starts 2023-10-09, so it can't serve the development window.
- **Parity:** U12 parity re-run with `COST_MODEL="slippage"` → PASS. `scripts\u12_parity.py` OLD_SWITCHES now include it.
- **Records:** the full record is the PLAN2.md U13 status note plus SPEC §12 / §19 "Implementation (U13, as built)".

## Key files for next session
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN2.md` — the U13 status note; U14 and U15 are next.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` — §12 / §19 as-built notes (sources, URLs, rules, cost semantics).
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\risk\costs.py` — `fill_costs`, the contract every backtest path now uses.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\data\exo.py`, `data\events.py`, `data\quotes.py` — inputs for the U15 state features.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\scripts\u13_smoke.py` — `exo` / `quotes` smoke, writing to `results\u13\`.
- Plan file: none (PLAN2.md drives the program).
- Memory files touched: `C:\Users\Nick\.claude\projects\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\memory\windows-pc-run-setup.md` (Alpaca 200 req/min shared limit; quotes fetch ~3 h).

## Running state
- Background processes: none. The three this session (quotes fetch, event-session fetch, parity run) all exited 0.
- Dev servers / ports: none.
- Open worktrees / branches:
  - Main checkout `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model` on `main` at 724eff3, in sync with origin except this handoff commit.
  - `origin/unit/13-exo-data-costs` is merged; it can be deleted.
  - Carried over, untouched: worktree `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b`.
- Local-only data (git-ignored):
  - `data\cache\exo\quotes\samples\` (215 sessions) and `event_samples\` (50 sessions). Re-fetching takes about 3 h plus 45 min at Alpaca's 200 req/min.
  - Scratch parity root in the session scratchpad (disposable).

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q` — 508 passed.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `uv run python scripts/u13_smoke.py exo` — `max_abs_diff` 0.0; `vx1_vix_dev.median` ≈ 1.043.
- `uv run python scripts/u13_smoke.py quotes` — `spy_quotes_median_bp_dev` ≈ 0.155, `spy_cs_mean_bp_dev` ≈ 1.53, and `event_vs_base` ratios.
- Parity, about 17 min:
  1. `uv run python scripts/u12_parity.py spec P.yaml`
  2. `LOKY_MAX_CPU_COUNT=1 python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 24`
  3. `uv run python scripts/u12_parity.py check R R/ledger.jsonl` → `parity: PASS`.

## Deferred + open questions
- Deferred: stress-session costs are 2–4× the table. Whether to scale costs with volatility is for the U17 family specs; the §17.2 floor uses the table as is.
- Deferred: a credit-spread state needs a source other than FRED HY OAS (e.g. HYG vs IEF) — U15.
- Deferred: VIX has exchange-holiday rows from 2022 that the other CBOE indexes lack, so build ratio features on common dates — U15.
- Deferred: exo `coverage_end` = the fetch day, so a same-day morning fetch isn't refreshed after the close — U20.
- Deferred: per-year quote medians price a fill with its whole year's sample (cost-model look-ahead within a year) — accepted.
- Deferred: SVXY regime break (2018-02-28) — F9 / U21 must start after it or model it.
- Deferred, carried from U12: cost-target tuning; the 1Day `MIN_VAL_EVENTS` calibration question (U17 / U18).
- Open: delete the merged `unit/13-exo-data-costs` branch and the stage-b worktree? (user's call).
- Open: push this handoff commit to `origin/main`? (user's call).

## Pick up here
Start U14 (event samplers, exit models, ISOM `VOL_PROFILE="tod"`; PLAN2 U14, SPEC §13–§14) on a new branch `unit/14-...` off `main`. Add `VOL_PROFILE`, `EVENT_SAMPLER` and `EXIT_MODEL` to `OLD_SWITCHES` in `scripts/u12_parity.py`, then re-check parity.
