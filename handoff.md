# Session Handoff — U15 state and context feature groups (merged to main)

## Where it started
The user asked me to review `handoff.md` (U14 merged as PR #16) and begin U15 per PLAN2.md and SPEC §15. U15 was built
on `unit/15-state-features`, adversarially reviewed, fixed, re-verified (suite, parity, legacy run), and then (at the
user's request) pushed, opened as PR #17 and merged into `main` (merge commit 6b4808c).

## Decisions locked + what shipped
- Point-in-time exo access — `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\features\exo_align.py`:
  - Groups never see raw frames; they get an `Exo` view restricted to their declared series.
  - Rule: a row is used only when `available_at` ≤ bar stamp (bar open; a naive index is NY time).
  - Accessors: `asof`, `ratio` (built on common observation dates), `change`, `next_date` / `last_date` / `on_day`,
    `known_dates`. `withdrawn_at` is honoured.
- Registry: `FeatureGroup.optional` and `FeatureGroup.exo`, plus `context_needs`
  (`...\features\registry.py`).
- `FeatureSet` (`...\features\feature_builder.py`):
  - builds each group's context;
  - per-fold groups that declare `needs` get it as `transform(df, state, cfg, context)`.
- `features\cache.py`: `context_hash` covers the full exo frames.
- Groups — `...\features\state.py` (computations), registered in `...\features\groups.py`:
  - `vol_state` (per fold): VIX, VIX9D/VIX3M ratios, VX1/VIX, VX2/VX1, rv21/VIX; GARCH(1,1) with variance targeting,
    numpy/scipy, fit on train bars only. Intraday it runs on returns deseasonalized by a train-fit VolProfile,
    because 5Min folds start with 42 sessions. < 500 returns → RuntimeWarning → fold skipped
    (`WindowFit` status renamed `fracdiff_failed` → `feature_fit_failed` in `...\wfo\wfo_engine.py`).
  - `calendar_events`:
    - `to_` / `since_` FOMC/CPI/NFP in sessions; `to_*` = 63 cap when no next event is visible (late December);
    - flags: `fomc_day`, `cpi_nfp_day`, `opex_week`, `tom` (value −1/+1..+3), `pre_holiday`;
    - `mins_since_release` (intraday, 480 on non-release days).
  - `rates_credit`: `d_dgs10`, `t10y2y`, `d_credit` (FRED BAA10Y), `d_dollar` (5-obs log change of DTWEXBGS).
  - `cross_asset` now also has `mkt_ret_lag{1,3,6}`. With a "sector" context it adds `sector_ret_lag{1,3,6}`,
    `sector_resid_ret` and `rel_strength_h`.
  - `gamma_proxy` NOT built: no historical chain.
- BAA10Y added to `...\data\exo.py` `FRED_SERIES` (H.15 available_at rule) and cached in
  `...\data\cache\exo\fred\BAA10Y.{npz,json}` (committed). It replaces HY OAS as the credit spread, because HY OAS
  only covers 2023-10 →.
- Review SEVERE fix — `...\data\events.py`:
  - `ORIGINAL_SCHEDULE` and `schedule_series`, served as `calendar/{FOMC,CPI,NFP}_SCHEDULE`.
  - Contents: the originally published dates of cancelled/moved releases (FOMC 2020-03-18; BLS 2025-10..2026-02),
    checked against Wayback BLS snapshots, public from Jan 1 and withdrawn at their own scheduled instant.
  - `to_*` reads these. The flags and `since_*` read held releases only.
- `...\data\sectors.py`: AAPL/MSFT/NVDA → XLK, JPM → XLF, XOM → XLE, UNH → XLV, AMZN/GOOGL/META → QQQ. ETFs get
  market context only.
- `...\features\context.py` `load_context`:
  - market SPY, sector, exo rows from start − 120 d;
  - calendar rows to end + 120 d, allowed past HOLDOUT_START (schedules only);
  - refuses `cross_asset` on the market symbol itself.
- Runner (`...\experiments\runner.py`):
  - `load_cell_data` returns `data["context"][sym]` only for cells whose groups need it;
  - the context enters `data_hash` and `signals_key` and is passed to `run_wfo`, `run_pwfo` and `combo_job`;
  - `CachedBars.exo` added; `experiments\report.py` pilot rule passes context.
- CLI (`...\wavelet_meta_model.py`) uses `load_context` for an Alpaca `--symbol`.
- Records: PLAN2.md U15 status note and Status list; SPEC.md §15 "Implementation (U15, as built)".
- Smoke spec and diagnostic:
  - `...\experiments\specs\u15_smoke.yaml` (stage U10; 4 cells ok on a scratch ledger);
  - `...\scripts\u15_state.py` → `...\results\u15\state_features.json`.

## Key files for next session
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN2.md` — U15 status note (deferred items); U16
  (mechanism primaries, SPEC §16) is next.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` — §15 as-built notes; §16 for U16.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\features\exo_align.py`, `features\state.py`,
  `features\context.py` — the APIs U16 primaries (calendar_drift, event_reaction, gap_fade days) will read.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\tests\test_u15.py` — 38 tests: point-in-time,
  exo-perturbation causality, the leaky-join mutation check, calendar hand cases, schedule-leak regressions, GARCH,
  runner context.
- Plan file: none (PLAN2.md drives the program).
- Memory files touched: none.

## Running state
- Background processes: none (suite, parity, smoke, diagnostic and reviewer all finished).
- Dev servers / ports: none.
- Open worktrees / branches:
  - `unit/15-state-features` is merged; it can be deleted, as can `unit/13-…` and `unit/14-…`.
  - Carried over, untouched: `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b`.
  - The scratch worktree of `main` used for the legacy comparison was removed.
- Scratch (disposable), in the session scratchpad: `parity15\`, `smoke\`, `legacy\`, `legacy_main\`, `wb\` (Wayback
  BLS pages), `review\` (reviewer probes).
- Pending task chip: "Investigate legacy CSV regression hash drift" (spawned this session).

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q` — 614 passed.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- Parity (~17 min) — `parity: PASS` (8/8), as at 7a52fdd:
  1. `uv run python scripts/u12_parity.py spec P.yaml`
  2. `LOKY_MAX_CPU_COUNT=1 python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 24`
  3. `uv run python scripts/u12_parity.py check R R/ledger.jsonl`
- `PYTHONIOENCODING=utf-8 MPLBACKEND=Agg uv run python wavelet_meta_model.py data/data.csv --out R` —
  `R/wfo_signals.csv` sha1 `f8e0617e…` (identical on `main` before U15; see open question).
- `uv run python scripts/u15_state.py` — reproduces `results\u15\state_features.json` (cached bars and exo, no
  network).
- `LOKY_MAX_CPU_COUNT=1 uv run python -m experiments --root <scratch> --ledger <scratch>/ledger.jsonl run experiments/specs/u15_smoke.yaml --jobs 4`
  — 4 ok.

## Deferred + open questions
- Deferred: releases on non-session days (the Sunday 2020-03-15 FOMC statement, Good Friday CPI/NFP) flag no session
  — a family-spec choice for F8 (U16/U17).
- Deferred: sector ETFs get no "sector" context (PLAN asked for it); AMZN/GOOGL/META map to QQQ.
- Deferred: `gamma_proxy` — no historical chain or open interest; `data/options.py` collects forward (U19/U20).
  Candidate sources were discussed at the end of the session (SqueezeMetrics GEX/DIX CSV, CBOE SKEW/VVIX, ThetaData /
  ORATS / CBOE DataShop, a bar-based realized-gamma proxy).
- Deferred: BAA10Y `available_at` (next business day 16:30 ET) assumed, not verified against FRED's publication lag.
- Deferred: between a cancellation's announcement and its original scheduled instant, `to_*` is stale (announcement
  dates not recorded); rescheduled BLS rows in 2025-10..2026-02 stay public only on their release day (U13 rule).
- Carried over: MOC entries and `width` under time/hysteresis exits (U16); stress-session costs (U17); exo
  `coverage_end` (U20); SVXY's 2018 break (F9/U21).
- Open: the legacy CSV hash on `main` is `f8e0617e…`, not the `551d8074…` recorded after U12–U14; U15 does not change
  it. Either the reference value or the environment drifted (task chip spawned).
- Open: delete the merged `unit/13-…` / `unit/14-…` / `unit/15-…` branches and the stage-b worktree? (the user's call)
- Open: this handoff.md is written to the working tree on `main` but not committed.

## Pick up here
Start U16 (mechanism primaries; PLAN2 U16, SPEC §16) on a new branch `unit/16-...` off `main`. It builds on the U14
samplers/exits and the U15 `session` / `calendar_events` / `vol_state` groups.
