# Session Handoff — U11 Stages C and E run, verdicts applied, report published; U11 complete

## Where it started
The user asked me to review the previous `handoff.md` and begin U11 Stage C, which runs the power walk-forward (PWFO) on the 14 Stage B3 finalists. Carried-over constraints:
- Every fitted cell is a ledger trial.
- Rules are pre-registered in PLAN before results are seen.
- No hashed code changes while a run is in progress.
- Long runs are detached via `scripts\run_specs.ps1` on the i9 PC; the user launches or approves runs and reports back when they finish.

The session then continued through Stage E (the holdout, run once) and the final report.

## Decisions locked + what shipped
- **Stage C grid amendment** (user's choice). U9's default grid can't fit any 1Day window (~0.25 events per session) and IS 63/126 are unfittable intraday.
  - Grid: intraday IS {252, 378, 504, 756}, 1Day IS {1260, 1512}, each × OOS {5, 10, 21, 63}; 184 trials.
  - In `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN.md` (U11 Status) and `experiments\report.py` (`C_GRIDS`, `stage_c_cell`).
  - Spec: `experiments\specs\u11_c.yaml`.
- **C → D rule** (user's choice): the walk-forward live stream must pass the B gates, and PBO must be < 0.5.
  - Result: **0 of 14 passed**, so Stage D was not run.
  - Output: `results\experiments\report\stage_c_selection.{csv,md}`.
- **Stage E scope:** all 14 finalists (user's choice), each run through 2026-09-27, which is where the cached bars end (no Alpaca top-up).
  - New hashed-code flag `PWFO_PARTIAL_LAST` in `wfo\pwfo.py` and `utils\config.py`: lets the last OOS window run short so all 248 holdout days are scored.
  - Runner (`experiments\runner.py`): in a `--final` PWFO cell, ledger stats cover holdout days only; the full stream goes to `daily_returns_full.csv`.
  - Spec: `experiments\specs\u11_e.yaml`.
- **E verdict amended before any holdout access** (user's choice, prompted by their question about the 3.5 Sharpe bar):
  - edge = Holm-adjusted 1 − PSR(0) across the 14 < 0.05 and alpha Sharpe > 0.
  - negative = the 90% Sharpe interval lies entirely below 0.
  - not demonstrated = everything else.
  - Reported only: Benjamini–Hochberg q (user asked for it as a metric, not a rule), full-search DSR, buy-and-hold Sharpe.
  - Code: `stage_e_selection`, `holm`, `bh_fdr`, `sharpe_ci` in `experiments\report.py`.
- **Stage E ran once:** batch `33b8b08e02e9413c`, code hash `23c33327…`.
  - Three worker crashes, all memory corruption in scikit-learn forest code inside nested pools; one Windows access-violation event, no WHEA hardware error. Re-run within the same batch as registered: NVDA with `--retry-errors`, GOOGL serially with `--jobs 1 --inner-jobs 1`.
  - Every E cell's pre-holdout stream matches its Stage C stream exactly.
- **Stage E outcome:** 0 edge, 13 not demonstrated, 1 negative (XLE 1Hour).
  - QQQ 1Hour came closest: holdout Sharpe 2.31, Holm p 0.091.
  - Across the 14, holdout Sharpe mean is 0.02 (sd 1.03), and Spearman between development and holdout Sharpe is −0.15.
  - Output: `results\experiments\report\stage_e_verdicts.{csv,md}`.
- **Launcher (`scripts\run_specs.ps1`):** added `-InnerJobs` and `-Final`. New `scripts\pwfo_progress.ps1` reports read-only progress (`-Spec u11_e`).
- **Report published:** "Meta-Labeling Zoo Study", https://claude.ai/artifact/BeUrh91ELeDzHHE1nhuaMv (private). Copy at `results\experiments\report\u11_report.html`.
- **U11 marked complete in PLAN.** Cadence item: none recommended; descriptive best is IS 756/OOS 10 intraday and IS 1512/OOS 10 daily.
- **Commits:**
  - Branch `unit/11-stage-c`: 74cce55, ff90ad9, 539400d, ad0ed7c.
  - Branch `unit/11-stage-e` (branched from stage-c): 784eb80, 26267be, 5d4b3bf, 1dc68b8.
  - `data\cache\holdout_access.jsonl` is committed as the audit record.

## Key files for next session
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN.md` — read first. The U11 Status section holds every rule, amendment, run note and outcome for Stages C and E.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\experiments\report.py` — Stage C/E tooling; CLI `c-spec | c | e-spec | e`. Not hashed.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\results\experiments\report\stage_e_verdicts.csv` — the final verdicts.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\results\experiments\report\u11_report.html` — source of the published report. Republish edits with the Artifact tool using `url` https://claude.ai/artifact/BeUrh91ELeDzHHE1nhuaMv after an `action: "read"`.
- Plan file: none (PLAN.md drove the work).
- Memory files touched: `C:\Users\Nick\.claude\projects\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\memory\windows-pc-run-setup.md` — added a line on `-InnerJobs` and the cost of the slowest combo.

## Running state
- Background processes: none. All runs exited; the preview `http.server` on 127.0.0.1:8765 (task `b6p4sc4xy`) was stopped.
- Dev servers / ports: none.
- Open worktrees / branches:
  - Main checkout `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model` is on `unit/11-stage-e`, local only, not pushed or merged. It contains everything from `unit/11-stage-c`.
  - The worktree `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b` is carried over from before this session, untouched.

## Verification — how to confirm things still work
- `uv run pytest -q` with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` — 418 passed.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `uv run python -c "from experiments.runner import code_hash; print(code_hash()[:16])"` — `23c33327f7d89a05` while no hashed code has changed.
- `wc -l results/ledger.jsonl` — 1068 (1050 through Stage C, plus 1 holdout event row, 14 `ok` E rows and 3 `error` E rows).
- `cat data/cache/holdout_access.jsonl` — exactly 1 line, batch `33b8b08e02e9413c`.
- `uv run python -m experiments.report e` — "14 finalists, verdicts {'not_demonstrated': 13, 'negative': 1}" with no "NOT FINAL".

## Deferred + open questions
- Deferred: the cross_asset feature group in the runner, `kelly_capped`, `POSITION_MODE=average`, risk-profile ablation, and Stage D — none reached or run in U11.
- Deferred: removing the stage-b worktree; pushing and merging `unit/11-stage-c` and `unit/11-stage-e` (the user hasn't asked).
- Open: the source of the worker crashes in Stage E. Candidates are CPU instability (i9-14900KF Raptor Lake, BIOS 1836) or a native bug under nested pools. Suggested: a stability check, or `--inner-jobs 1` for future PWFO runs.
- Open: `handoff.md` has uncommitted edits from the earlier session and is now stale. Whether to replace or commit it is the user's call.
- Open: the report is private; the user shares it from the page's Share menu if needed.

## Pick up here
U11 is complete. Ask the user whether to push and merge `unit/11-stage-e` into `main`, and what comes after U11, for example the stability investigation or the deferred items in PLAN.