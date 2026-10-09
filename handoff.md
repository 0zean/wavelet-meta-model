# Session Handoff — U12 harness speed (complete)

## Where it started
The user asked me to review `handoff.md`, commit the PLAN2/REVIEW/SPEC doc changes to `main`, and start U12 on a new
branch. The docs went to `main` as 632867b (not pushed). U12 was then built, reviewed and verified on
`unit/12-harness-speed`.

## Decisions locked + what shipped
- U12 per PLAN2 / SPEC §11.2. The full status note, with the cost table, is in `PLAN2.md` under U12; the as-built
  choices are in SPEC §11.2 "Implementation". Commits on `unit/12-harness-speed`: 7906389 (implementation), 02c4fca
  (review fixes), plus the docs / status commit.
- Switches, now defaulting to the SPEC §11.2 values:
  - `ZOO_FIXED_PARAMS`, `CALIBRATION` (crossfit | rolling | none), `OOF_META` (refit | reuse), `PWFO_COMBINE`
    (nested | average).
  - Grids: IS (504, 756) intraday, (1260, 1512) on 1Day, each × OOS 21.
  - Flat loky pool in the runner; `--inner-jobs` removed; `run_specs.ps1 -Jobs` defaults to 24.
  - numba CUSUM kernels; numba 0.68 added (numpy unchanged).
- Parity criterion = the SPEC invariant (old switches pinned), not PLAN2's "≥ 95 % chosen params" wording. Result:
  PASS on all four fixture cells, before and after the review fixes. A baseline of unchanged `main` code also
  reproduces U11 on this PC.
- Legacy CSV regression: byte-identical to `main` on Windows (sha1 `551d8074…`). The `402ef202…` reference is the Mac
  value.
- Review fixes: OOF on demand (`need_oof`) instead of refusing train-fit sizers; calibration mode recorded per window.
- Not met: "≥ 10× fewer fits per window" for xgb (9×) and rf_ldp_fast (5×); "≥ 8× fewer windows" on 1Day (7.7×).
  Accepted by the user; tuning deferred.

## Key files for next session
- `PLAN2.md` — U12 status note; U13 / U14 are next (parallel).
- `SPEC.md` §11.2 — switch semantics as built.
- `scripts/u12_parity.py`, `tests/fixtures/u12_parity.json` — parity re-check (add the U13 / U14 switches to
  `OLD_SWITCHES`).
- `scripts/u12_cost.py` — cost table.
- `tests/test_harness_speed.py`, `tests/test_u12_review.py` — U12 tests.

## Running state
- Background processes: none.
- Worktrees: the main checkout on `unit/12-harness-speed`; `wavelet-meta-model-stage-b` (carried over, untouched).
  The U12 baseline / parity worktrees were removed.
- Nothing pushed; `main` is 1 commit ahead of origin (the docs commit).

## Verification
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q` — 463 passed. `uvx ruff check . && uvx ruff format --check .` — clean.
- Parity: `uv run python scripts/u12_parity.py spec P.yaml`, then
  `python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 24`, then
  `scripts/u12_parity.py check R R/ledger.jsonl` → `parity: PASS` (about 17 min).

## Deferred + open questions
- Deferred: tuning the partly met cost targets (accepted by the user 2026-10-09).
- Open: at 1Day about a third of windows fall back to cross-fit calibration (`MIN_VAL_EVENTS = 100` resolved
  pairs) — revisit in the family specs (U17 / U18).
- Deferred MINORs: `ZOO_FIXED_PARAMS` value validation; a minimum length for the averaged stream; serial batch
  completion after a dead worker.
- Carried over: `HOLDOUT_START` move (U13 / U20); the VIX URLs and Alpaca open interest (U13); the extended ETF
  universe and F9; removing the stage-b worktree; pushing / merging branches (user's call).

## Pick up here
Merge or PR `unit/12-harness-speed` (user's call), then start U13 (exogenous data + quotes cost model) and/or U14
(samplers, exits, ISOM vol profile) on their own branches off `main`.
