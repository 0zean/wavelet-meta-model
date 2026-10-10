# Session Handoff — U16 mechanism primaries (merged to main)

## Where it started
The user asked me to review `handoff.md` (U15 merged as PR #17) and begin U16 per PLAN2.md and SPEC §16. U16 was built
on `unit/16-mechanism-primaries`, adversarially reviewed, fixed, and re-verified (suite, parity twice, legacy run,
smoke spec), then (at the user's request) pushed, opened as PR #18 and merged into `main` (merge commit b338211).

## Decisions locked + what shipped
- Primaries — `primaries/mechanism.py` (`MechanismPrimary`, `primary_config`, `session_state`, `CALENDAR_WINDOWS`):
  `vol_target`, `tsmom`, `overnight`, `calendar_drift` (one window per cell), `intraday_momentum`, `gap_fade`,
  `event_reaction`, `weekly_reversal` (time-series form; the sector rank is U17). `vix_carry` is U21.
- MOC modelling decided (the U14 deferral): entry-at-close fill model. Schedule `entry_times: ["close"]` = decide at
  the bar before the closing-auction bar and fill at its close (`features/events.entry_at_close`).
- Sampler / exit keys (`features/events.py`, `features/exits.py`): `every` (session / week / month), `day_offset`,
  predicates `macro` / `month_end` / `opex`; time exit `exit_session`, `exit_time: next`.
- Portfolio simulator (`risk/portfolio.py`):
  - MOC entries fill at the close.
  - A known scheduled exit frees its symbol; an exit at the entry's own fill rolls into it (cost on the traded
    shares only; an unchanged bet keeps its shares).
  - `wfo/backtest.portfolio_path` sends time / hysteresis exits through the portfolio simulator.
  - Behaviour change: a MOO exit no longer blocks the same open's entry.
- Flat sides (0) for mechanism primaries (`primaries/base.check_signal(frame, X, primary)`). They are excluded from:
  meta fitting, CalHistory, `meta_outcomes`, CPCV (refused), and trading.
- `rule_size` sizer (`sizing/sizers.py`, `size(p, hint)`). Under it the primary stream is sized by `magnitude`
  (`wfo/backtest.primary_size_col`).
- `experiments/spec.normalize`: a mechanism primary's sampler / exit go into the cell's `overrides`; `sizer`
  defaults to `rule_size`; a needed feature group must be listed.
- Diagnostics: `primaries/diagnostics.slot_diagnostics`; `n_flat`.
- Records:
  - SPEC §16 "Implementation (U16, as built)".
  - PLAN2 U16 status note and Status line.
  - `experiments/specs/u16_smoke.yaml`.
  - `scripts/u16_primaries.py` → `results/families/u16_slot_diagnostics.{csv,json}`, `results/u16/smoke_cells.json`.

## Key files for next session
- `PLAN2.md` — U16 status note (deferred items); U17 (family-test tooling, SPEC §17) is next.
- `SPEC.md` §16 as built; §17 for U17.
- `primaries/mechanism.py`, `risk/portfolio.py`, `features/events.py`, `features/exits.py` — what U17's family cells
  run.
- `tests/test_u16.py` — 94 tests:
  - harness: MOC, periods / offsets, exit_session / next, rolls, freed slots, rule_size, flats;
  - per-primary causality (fixed cuts + strict per-event), synthetic sanity, real SPY hand trades, spec layer.
- Memory files touched: none.

## Running state
- Background processes: none.
- Open worktrees / branches: none besides `main`.
  - Merged local branches `unit/11-*` … `unit/16-*` deleted.
  - The stage-b worktree and the scratch `main` worktree removed (both clean; their branches were merged).
  - Remote `origin/unit/*` branches are kept.
- Hung processes left by the review subagent (a stdin-waiting `python -` and a wait loop) stopped.
- Scratch (disposable) in the session scratchpad: `smoke16run/`, `parity16/`, `parity16b/`, `legacy16/`,
  `legacy_main16/`, `review16/` (reviewer probes).
- Pending task chip from U15 ("Investigate legacy CSV regression hash drift") is now moot: see below.

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q` — 712 passed.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- Parity (~17 min) — PASS (8/8):
  1. `uv run python scripts/u12_parity.py spec P.yaml`
  2. `LOKY_MAX_CPU_COUNT=1 python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 20`
  3. `uv run python scripts/u12_parity.py check R R/ledger.jsonl`
- `PYTHONIOENCODING=utf-8 MPLBACKEND=Agg uv run python wavelet_meta_model.py data/data.csv --out R` —
  `wfo_signals.csv` sha1 `551d8074…`; all five outputs identical to `main`.
- `LOKY_MAX_CPU_COUNT=1 uv run python -m experiments --root S --ledger S/ledger.jsonl run experiments/specs/u16_smoke.yaml --jobs 8`
  — 14 ok. Then `uv run python scripts/u16_primaries.py --smoke S`.

## Deferred + open questions
- Deferred (PLAN2 U16 list):
  - Calendar windows and weekly rebalances count data sessions, not exchange sessions (SPY 1Day drops 2019-08-12).
  - weekly_reversal's rank needs a basket cell (U17).
  - vol_target's `vix` band path restarts per window.
  - MOC at 15:55 is after NYSE's 15:50 cutoff for single stocks.
  - `n_trades` counts rolls.
  - Width warm-up drops the first VOL_SPAN bars' events.
  - event_reaction trades 2020-03-03 at the scheduled times.
- Closed: the legacy hash. `main` and this branch both give `551d8074…` (the U12–U14 reference); U15's `f8e0617e…`
  was that session's environment. The U15 task chip can be dismissed.
- Open: delete the remote `origin/unit/*` branches? (the user's call)
- Noted for later units: the smoke spec lowers MIN_*_EVENTS and widens 5Min windows. U17 family cells should run
  the rules without the meta-model's event minimums (Phase 1 has no ML).

## Pick up here
Start U17 (family-test tooling; PLAN2 U17, SPEC §17) on `unit/17-...` off
`main`. Primary inputs:
- `primary_config` / spec cells;
- the primary-only stream sized by `rule_size`;
- portfolio cells for F2 / F10.
