# Session Handoff — U11 Stage A on the Windows PC: fastfracdiff, the E-core fix, Stage A run and survivor analysis; Stage B decided

## Where it started
The user moved U11 Stage A from the MacBook (estimated 3–4 days) to this Windows PC (i9-14900KF, 24 cores / 32 threads, 64 GB RAM) and asked me to review the previous handoff and run Stage A. Constraints that emerged:
- Every fitted cell must be recorded in the ledger as a trial.
- No code in the hashed packages may change while a run is in progress.
- Decision rules are pre-registered in PLAN before results are seen.
- Turnaround time matters: the user rejected a 22-hour run.

## Decisions locked + what shipped
- **Environment fix.** The locked environment could not import: `fracdiff` 0.9.0 pins statsmodels below 0.14, and 0.13.5 breaks under pandas 3. Fixed first with a uv override, then superseded: both `fracdiff` and `statsmodels` are now removed from `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\pyproject.toml`.
- **`fastfracdiff/`** at `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\fastfracdiff\`: numpy + scipy only, BSD-3 `NOTICE` for fracdiff and statsmodels.
  - The ADF lag search reads every nested candidate off one Cholesky of the centred, scaled Gram matrix.
  - A full d search on 190k 5Min bars went from 452 s to 2.7 s.
  - Equivalence against the old code: 158 training windows, 1,387 ADF evaluations, 0 d mismatches, 0 lag mismatches, t within 1.6e-11.
  - Tests: `tests\test_fastfracdiff.py` with the fixture `tests\fracdiff_reference.json`.
  - Added to `CODE_DIRS` in `experiments\runner.py`; `fracdiff` and `statsmodels` dropped from `LIBS` and from the feature-cache library list.
- **Windows file-replace race fixed.** `os.replace` fails with PermissionError when another process has the file open. Fixed via `publish()` and a retrying `load()` in `features\cache.py`; the runner's signals pickle write uses it too. Test: `test_cache_tolerates_concurrent_writers_and_readers_of_a_key`.
- **E-core confinement found and fixed.** Hidden detached runs get Windows EcoQoS throttling and run only on the 16 E-cores (each worker got about 0.53 of a core). The fix is a per-process `SetProcessInformation(ProcessPowerThrottling, StateMask=0)`, which made folds take 0.57× as long. It's built into the launcher `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\scripts\run_specs.ps1`, which detaches, runs specs in sequence with 30 jobs and `LOKY_MAX_CPU_COUNT=1`, and logs to `results\experiments\run_specs.log`. Its positional-binding bug is fixed.
- **5Min pilot rule.** Pre-registered, then evaluated: verdict **run** (21 of 30 upper bounds at or above 0.52). The tool is `python -m experiments.report pilot <spec>` in `experiments\report.py`, which is excluded from the code hash. Output: `results\experiments\report\pilot_u11_a1_5Min.md`.
- **Stage A amended to an xgb-only meta model, for cost.** `rf_ldp` cells were about 72% of the remaining run time.
  - `experiments\specs\u11_a_rf_cached.yaml` records the 83 already-fitted `rf_ldp` cells.
  - `experiments\specs\u11_a2.yaml` holds the 475 xgb cells.
  - Proxy check: rf − xgb AUC offset 0.0049 ± 0.0016, Pearson 0.98.
- **Stage A completed** 2026-10-03 at 13:27. The ledger has 739 rows: 83 `rf_ldp` and 475 xgb, all `ok`.
- **Pre-registered survivor rule:** AUC > 0.515, PSR > 0.5, top 20 by DSR, at most 2 per (symbol, timeframe), xgb rows only. Implemented as `stage_a_survivors` in `experiments\report.py`; CLI `python -m experiments.report survivors`. **54 cells passed both gates.**
- **The rule's DSR ranking turned out degenerate.**
  - The luck threshold is 1.81 annualized with N = 737, and every Sharpe is below it.
  - Below that threshold, DSR ranks the short 1Day out-of-sample windows (about 930 days from January 2022) above intraday windows of 2,200+ days, so 16 of the top 20 were 1Day.
  - Many of those are market exposure or a handful of lucky days.
  - Diagnostics are in `results\experiments\report\stage_a_survivors_diagnostics.csv` and `stage_a_survivors.{csv,md}`.
- **The user's decisions this turn (not yet written into PLAN):**
  1. The Stage A survivor set is **all 54 gate-passers**, which avoids the DSR-ranking artifact. This is a documented deviation from the pre-registered top 20.
  2. Stage B1's model axis uses **`rf_ldp_fast`** in place of full `rf_ldp`.

  Caveat for the PLAN write-up: the 99.7% equal-trade-decisions figure compares 200 against 500 trees at `max_features=1`. It does not compare against `rf_ldp`, which tunes `max_features` over {1, sqrt} (1 was picked in 81% of 21,948 fold fits).
- **Stage B models built** on branch `unit/11-stage-b-models`, in a worktree, **not merged**: `rf_ldp_fast` and `catboost`, in that worktree's `models\zoo.py`.
  - `rf_ldp_fast`: `max_features=1`, 200 trees.
  - `catboost`: Plain boosting, depth {4, 6}, 300 iterations, single-threaded.
  - Timings on a late SPY 5Min fold: logit_l2 4.5 s, rf_ldp_fast 8.9 s, xgb 15.6 s, catboost 33 s, rf_ldp 43.5 s. CatBoost's ordered boosting took 185 s and was rejected.
- **Stage B plan,** as told to the user:
  - Merge the branch first. That changes the code hash, so Stage A's xgb rows are the comparison and are not refit.
  - B1: the 54 cells × {logit_l2, rf_ldp_fast, catboost}, about 2–2.5 hours. The ~70 CPU-hour estimate I gave included full `rf_ldp`, which is now dropped.
  - B2: feature groups and cMDA selection on the best model per cell.
  - B3: the U7 sizers.
  - Selection between sub-stages is pre-registered and ranks by PSR, not DSR, adding alpha Sharpe > 0 and best-5-days share < 100% of P&L. DSR with the total ledger N stays the final test at the holdout.

## Key files for next session
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN.md` — the U11 Status section: Stage A amendment, pilot rule and outcome, survivor rule, Stage B meta-model axis, run notes. Read this first.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` — §9: the code hash now covers OS/architecture, CRLF normalization and `fastfracdiff`; also the fracdiff feature row.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\experiments\report.py` — pilot rule and survivor selection (excluded from the code hash).
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\results\experiments\report\stage_a_survivors.csv` — all 475 xgb cells with `passed` (54) and `survivor`.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\results\experiments\report\stage_a_survivors_diagnostics.csv` — beta, alpha Sharpe, exposure and moments for the DSR top 20.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b\models\zoo.py` — the Stage B models.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\scripts\run_specs.ps1` — the launcher for every long run.
- Plan file: none, besides PLAN.md.
- Memory files touched: `C:\Users\Nick\.claude\projects\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\memory\windows-pc-run-setup.md` and `...\memory\MEMORY.md`.

## Running state
- Background processes: none. Stage A's launcher (PID 21816) exited with "all specs done", and the earlier watcher (PID 14240) exited.
- Dev servers / ports: none.
- Open worktrees / branches:
  - `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model` on `unit/11-windows-setup` (pushed; no PR opened). Uncommitted:
    - `results\ledger.jsonl` (558 new Stage A rows);
    - untracked `results\experiments\signals\*`, `results\experiments\cells\*` and `results\experiments\report\*`;
    - new files under `data\cache\features\`;
    - run logs `results\experiments\u11_a1.*.log`, `u11_a_rf_cached.*.log`, `u11_a2.*.log` and `run_specs.log`;
    - `results\ledger.jsonl.lock`.
  - `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b` on `unit/11-stage-b-models`: pushed, commit 814acfb, with its own `.venv` that includes catboost 1.2.10.

## Verification — how to confirm things still work
- `uv run pytest -q` (main checkout, with `OMP_NUM_THREADS=1`) — 399 passed: 397 plus the 2 survivor and AUC tests added later.
- `uv run pytest -q` in the stage-b worktree — 401 passed. It crashed natively twice in 5 full runs; see Open below.
- `uvx ruff check . && uvx ruff format --check .` — clean in both checkouts.
- `wc -l results/ledger.jsonl` — 739.
- `uv run python -m experiments.report survivors` — "475 cells, 54 passed the gates".
- `ls data/cache/holdout_access.jsonl` — must not exist; the holdout has never been accessed.

## Deferred + open questions
- Deferred: the 392 `rf_ldp` cells in `u11_a_screen.yaml` — not run, by the Stage A amendment.
- Deferred: model-fit speedups that keep results identical (sharing ml_xgb fits, reusing xgboost's training matrices) — a code change, so best done when the branch merges.
- Deferred: committing the Stage A results and ledger, and opening PRs for `unit/11-windows-setup` and `unit/11-stage-b-models` — not requested yet.
- Open: intermittent native crash (faulthandler frames, exit 127) in the stage-b worktree's full test suite after catboost was added; not reproduced in 3 full runs plus 8 runs of `tests\test_pwfo.py`. Watch for `WorkerDied` rows in Stage B.
- Open: one load-dependent failure of `test_parallel_equals_serial_and_fits_each_symbol_once` early in the session; not reproduced.
- Open: exactly which feature-group sets B2 uses, and K / selection thresholds for each B sub-stage — to be pre-registered before each sub-stage runs.

## Pick up here
Write this turn's two decisions into PLAN as a dated amendment, with the deviation rationale and the `rf_ldp_fast` fidelity caveat. Then merge `unit/11-stage-b-models`, generate the B1 spec (54 cells × {logit_l2, rf_ldp_fast, catboost}) from `stage_a_survivors.csv`, and launch it with `scripts\run_specs.ps1`.