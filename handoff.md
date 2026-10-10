# Session Handoff — U17 family-test tooling (branch, not merged)

## Where it started
The user asked me to review `handoff.md` (U16 merged as PR #18) and begin U17 per PLAN2.md and SPEC §17.

U17 was built on `unit/17-family-tests` and adversarially reviewed. The BREAKING and SEVERE findings are fixed, and it
was re-verified: full suite, U12 parity, the legacy run vs a clean `main` worktree, and the registered dry run. The
branch is committed locally; it is **not pushed and has no PR yet** (the user's call).

## Decisions locked + what shipped
- **Phase-1 cells are rule cells.** `META_MODEL: none` with stage F only, no meta-model and no WFO
  (`wfo/rule_pass.py`).
  - State (the `tod` profile, vol_state's GARCH) is fit causally in segments of TEST sessions after `RULE_WARMUP`
    (new RunConfig field, default 63); the last segment runs to the data end.
  - The stream starts at the rule's first sided event, so warm-up days are not test days.
  - Rule cells have n_trials 0. `experiments.runner.run` refuses F / G / H cells unless `family=True`.
- **Trial accounting.** One ledger row per variant (kind `family_variant`, n_trials 1). The trial key is
  `<family>/<configuration hash>`, so amendments reusing labels with new configurations cost new trials.
  TRIAL_BUDGET ≤ 12, and an amendment cannot raise its family's budget. Program caps live in
  `families/program.yaml` (8 families, 112 trials).
- **Registration.** `python -m families register <spec>` writes `registered: {sha, date}`; commit that line. Run
  refuses an untracked or dirty spec, a non-ancestor sha, or content that differs from the registered commit.
- **Floors.** The PLAN2 defaults apply (0.8 × benchmark for long-only, else 2 %/yr; edge-to-cost 3×) and cannot be
  switched off.
- **Test.** Ledoit–Wolf studentized circular block bootstrap (block 21) with a Newey–West alpha. Coherence reads each
  variant's Δ on the headline's days. Holm runs across families, and untested families count with p = 1. Passing
  requires Δ > 0.
- **Portfolio simulator.** New equity attrs `traded_notional` and `cost_paid`.
- **U16 bug fixed.** The VIX rules read `vol_state__vix`; before, they raised in any real run.
- **Records.**
  - SPEC §17.5 as built.
  - PLAN2 U17 status note (deferred list) and Status line.
  - `families/dryrun/F1_spy_dryrun.yaml`, registered.

## Key files for next session
- `PLAN2.md` U18 is next: write and commit `families/F1.yaml` … `F8.yaml`, register each, run in order.
  - F1 / F3 / F4 come first, then F2.
  - **F2 first needs a basket risk profile in code** (gross cap 1.0, borrow), because `standard` caps at 10
    positions.
- `families/` (spec, run, test, stats, report, `__main__`); `wfo/rule_pass.py`; `tests/test_u17.py` (36 tests).
- Memory files touched: none.

## Running state
- Background processes: none. The review agent finished.
- Branch `unit/17-family-tests`: 5 commits on `main`. Scratch `main` worktree removed.
- **Uncommitted:** `data/cache/sip/all/5Min/SPY.{json,npz}`. The dry run's quasi-holdout cell made the data layer
  fetch SPY 5Min 2026-09-26 … 09-30 (pre-holdout). Commit it or `git checkout` it, the user's call.
- Scratch (disposable) in the session scratchpad:
  - `dryrun17b/` (the dry-run ledger, root and report);
  - `parity17/`, `legacy17/`, `legacy_main17/`, `review17/` (the reviewer's probes);
  - `fam1/`, `rule17*`.

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q` — 748 passed (~4.5 min).
- `uvx ruff check . && uvx ruff format --check .` — clean.
- U12 parity — PASS (8/8), same three commands as before.
- Legacy run: outputs are identical to `main` in the same environment. Today `wfo_signals.csv` sha1 is `f8e0617e…`
  on both; it differs across sessions (environment), so compare against `main` and not against a recorded hash.
- Dry run:
  1. `LOKY_MAX_CPU_COUNT=1 uv run python -m families --ledger S/ledger.jsonl --root S/root --out S/out run families/dryrun/F1_spy_dryrun.yaml --jobs 4`
  2. `... summary` with the same --ledger / --out.

## Deferred + open questions
- Deferred (PLAN2 U17 list):
  - the F2 basket risk profile;
  - the F10 cross-sectional rank;
  - 1Day vol_state rules start ~500 sessions in (GARCH_MIN_OBS);
  - `path_monotone`;
  - `gamma_sign` (unavailable);
  - stage G (U19).
- Open (the user's call):
  - push the branch and open a PR / merge;
  - the data-cache change;
  - delete the remote `origin/unit/*` branches (carried over);
  - whether the legacy-hash instability (environment-dependent) deserves its own investigation.

## Pick up here
Merge U17 if the user approves, then start U18 on `unit/18-...` off `main`:
1. Add the basket risk profile.
2. Write F1 / F3 / F4 family specs from PLAN2's headline definitions.
3. Commit, register and commit each one.
4. Run them on the program ledger in order.
