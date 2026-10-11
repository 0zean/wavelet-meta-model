# Session Handoff — U28: G6, the SVXY volatility risk premium (branch `unit/28-g6-svxy`)

## Where it started
U26 (PR #27) was merged to `main` (da05171). U28 was built, reviewed, registered and run on `unit/28-g6-svxy`:
- SPEC §27 describes the as-built code, the review and the result;
- PLAN3 §5 U28 lists every done-when with its evidence;
- PLAN3 §9 is the status line.

## User decisions (2026-10-11, before any G6 run)
- The headline is a region of 15 cells (VX1 / VIX > θ ∈ {1.00, 1.025, 1.05, 1.075, 1.10} × VIX < {20, 25, 30}), not
  PLAN2 F9's point.
- The test is `marginal` against G2_core at k = 0.5. Long SVXY carries equity beta, so alpha against zero would restate
  the equity premium; that alpha is reported only.
- The target is halved before 2018-02-28 (SVXY −1× until then, −0.5× after).
- The 15 % drawdown kill is a permanent latch, with a `no_kill` variant.

## What shipped (SPEC §27 has the detail)
- **`vix_term`** (features/state.py, groups.py): VIX, VX1 / VIX, VX2 / VX1 as of the bar, with no GARCH. vol_state's
  GARCH needs 500 returns, so a rule reading it would start in 2018.
- **`vix_carry`** (primaries/vix_carry.py): the region; flat in backwardation (VX1 / VIX < 1); the pre-break scale.
  - One decision per session at the 10:00 5Min mark, rolled to the next 10:00; the roll trades only the change.
  - Why not daily bars: the daily engine decides at the previous close, before the 16:20 settlements, so the open of
    day d would read d − 2's VIX.
  - SVXY lacks a 10:00 decision on 2018-02-06 and 2024-03-26 only.
- **Drawdown kill** (risk/profiles.py `kill_dd`, risk/portfolio.py): a latch, exit reason `kill`, `killed_at` in the
  runner row. Profiles `vrp` (3 % gate + 15 % kill) and `vrp_nokill`. A no-op when None (full suite, pins).
- **Engine regions** (families/spec.py `region_kind`, families/run.py `engine_cells` / `engine_region`).
  - Each cell runs as its own rule cell in the batch: no trial, no variant row.
  - Statistics are on the headline's days; a cell is flat before its live start.
  - Under `marginal`, cells coherence reads each cell's Δ Sharpe.
- **Reported additions**: `overlay_alone`, per-variant `kill`, and the `episodes` spec key with a day-by-day report
  section.
- **Scripts**: `scripts/u28_g6.py power | smoke` (status only).
- **Tests**: tests/test_u28.py has 11 tests.

## Result
- **G6 fails.** Sharpe(core + 0.5 × G6) 1.08 vs the core's 1.06: Δ +0.02, p 0.40, Holm 0.62. Floors pass and the
  cells are coherent (12 / 15 positive).
- The headline's kill fired on 2019-09-03, so it was flat for 2019-09 → 2025-09.
- The return is 2016–2017 at −1×: the post-break slice has Sharpe −0.28.
  - `unscaled` Δ +0.18 (p 0.066).
  - `no_kill` Δ −0.05, with a −27 % drawdown.
- February 2018: the 3 % gate flattened it on 01-29 and backwardation kept it flat through the 02-05 / 02-06 crash.
- Disregard the 2020–2025 split Sharpes and the quasi slice's 26.2: they are artifacts of a flat, near-zero-variance
  cash stream.
- Program over G1, G2, G3, G6: G1 PASS (Holm 0.0136); the others fail. Looks K = 209; trials 29 / 42.
- G2_core re-run under U28's code equals U25's stored stream exactly.

## Verification
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — 898 passed.
- `uv run python scripts/u28_g6.py power` and `... smoke`.
- Adversarial review: no BREAKING or SEVERE finding; three MINORs fixed before registration.

## Notes for the next session
- A heredoc turns `"\n"` into real newlines (it bit a test file again). Use the Write / Edit tools for code with
  escapes.
- On a 5Min schedule rule, the window's first ~2 sessions have no event (barrier-width warm-up). A window's last bet
  whose exit falls past the data is dropped, so the book is flat on the last session.
- families/registry.yaml still holds G1 as U25 wrote it (Holm 0.010 over three families). With G6 in the program,
  G1's Holm p is 0.0136. Re-run `python -m families registry G1` only if you want the entry to say so.

## Open decisions (the user's)
- Algo Trader Plus before U27. U27 then trades G1 alone: a diagnostic whose result is bound by the cost.
- Opening a PR for this branch; deleting the remote `origin/unit/*` branches.
- The G5 data purchase. Whether G2 is worth a forward record.

## Pick up here
- **U27** (forward test) once Algo Trader Plus is bought. Its input is `families/registry.yaml` (G1).
- **Deferred MINORs from U25:**
  - the loader's PLAN2 2 % default floor applies to paired specs;
  - a gate that delays a stream's first live day would drop reference days;
  - one failed core aborts a marginal test.
