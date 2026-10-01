# Session Handoff — U11 started: Stage A spec, cost probe, Windows portability (uncommitted)

## Where it started
The user asked me to review the previous `handoff.md` (U10 merged; `main` = `upstream/main` = 98a921a) and begin U11. I branched `unit/11-zoo-study`, wrote the Stage A spec and timed it. The estimate came to about 860 CPU-hours. The user then said they will run Stage A on their more powerful Windows PC, so the rest of the session made the runner Windows-portable.

## Decisions locked + what shipped
- **Stage A spec** — `/Users/nick/Documents/Python Projects/wavelet/experiments/specs/u11_a_screen.yaml`.
  - 950 cells: 19 symbols × {5Min, 15Min, 30Min, 1Hour, 1Day} × {wavelet_trend, sma_cross, bollinger_mr, donchian_breakout, ml_xgb} × meta {xgb, rf_ldp}.
  - Default features, `oof`, `fixed` sizing, risk `none`, overrides `{TEST: 10}` (bi-weekly cadence on every timeframe).
  - 1Min is excluded (not cached). The spec loads and validates to 950 cells.
- **Scope** — full 950 cells, as PLAN says. The user will run them on the Windows PC rather than cutting the scope.
- **Cost probe** — SPY × 5 timeframes × {xgb, rf_ldp}, 10 jobs, scratch ledger, nothing counted in the real ledger.
  - Per fold: about 3 s for xgb and 9 s for rf_ldp early on, rising as the expanding train window grows.
  - Folds per cell: 92 for 1Day, about 207 for 1Hour, about 240 for 5Min.
  - CPU time per cell: 1Day xgb 197 s, 1Day rf 806 s, 1Hour xgb about 1045 s; 5Min rf extrapolated to about 2.5 h.
  - Total for Stage A: about 860 CPU-hours. The probe was killed once it had given the estimate.
- **Windows portability:**
  - `/Users/nick/Documents/Python Projects/wavelet/experiments/ledger.py`: `_lock`/`_unlock` use `msvcrt.locking` byte-range locks on `os.name == "nt"` and `fcntl.flock` elsewhere. The ledger is read as UTF-8. The Windows path is untested, because it cannot run on macOS.
  - Explicit `encoding="utf-8"` on all text I/O in:
    - experiments: `runner.py`, `legacy.py`, `report.py`, `spec.py`, `__main__.py`
    - data: `store.py`, `bars.py`
    - wfo: `pwfo.py`, `pwfo_run.py`
    - other: `models/compare.py`, `risk/run.py`, `sizing/compare.py`

    The WFO logs contain characters (═ σ ≥) that cp1252 cannot encode.
  - `experiments/__main__.py`: stdout and stderr are reconfigured to UTF-8 (`errors="replace"`).
  - `experiments/runner.py` `code_hash()`:
    - It normalizes CRLF to LF.
    - It now includes `platform.system()` and `platform.machine()`, so results are reused only on the platform that produced them.
    - This changes every code hash, which was fine because Stage A had not run.
  - New `/Users/nick/Documents/Python Projects/wavelet/.gitattributes`: `eol=lf` for py, yaml, md, toml and lock files.
- **Tests** — `/Users/nick/Documents/Python Projects/wavelet/tests/test_experiments.py` has 2 new tests:
  - `test_code_hash_ignores_line_endings_but_not_the_platform`
  - `test_ledger_lock_round_trip_and_utf8`
- **PLAN** — `/Users/nick/Documents/Python Projects/wavelet/PLAN.md`: the U11 Status changed from "Not started" to "In progress", with notes on the spec, the probe, portability and ledger continuity.
- **Ledger continuity rule:**
  - The PC must start from the Mac's `results/ledger.jsonl` (181 rows), `results/experiments/` (about 14 MB) and `data/cache/` (243 MB).
  - All three are copied, not re-fetched: `adjustment=all` history re-adjusts when new dividends are paid.
  - The PC ledger is canonical from Stage A onward.
  - No pipeline code may change during the run: the code hash would change, and cells would be refit and counted again.

## Key files for next session
- Plan file: `/Users/nick/Documents/Python Projects/wavelet/PLAN.md` — the U11 section and its Status note.
- `/Users/nick/Documents/Python Projects/wavelet/SPEC.md` — §9 (ledger, trial count, holdout, cell hash). It does not yet mention the platform in the code hash or CRLF normalization; it needs that update.
- `/Users/nick/Documents/Python Projects/wavelet/experiments/specs/u11_a_screen.yaml`
- `/Users/nick/Documents/Python Projects/wavelet/experiments/ledger.py` — portable locks.
- `/Users/nick/Documents/Python Projects/wavelet/experiments/runner.py` — `code_hash`.
- `/Users/nick/Documents/Python Projects/wavelet/experiments/report.py` — will need the U11 additions: heat-maps, ablations, verdicts.
- Memory files touched: none. `/Users/nick/.claude/projects/-Users-nick-Documents-Python-Projects-wavelet/memory/blas-single-thread.md` still applies.

## Running state
- Background processes: none. Shell `bsveoicid` (the probe) was killed and exited with 143. Monitor `bcg84toyj` failed on a zsh glob. Monitor `bfu3t2zbx` expired.
- Dev servers / ports: none.
- Open worktrees / branches:
  - `unit/11-zoo-study` is checked out in `/Users/nick/Documents/Python Projects/wavelet`. Everything is uncommitted.
  - Modified: `PLAN.md`, the ledger/runner/encoding files listed above, `tests/test_experiments.py`.
  - Untracked: `.gitattributes`, `experiments/specs/u11_a_screen.yaml`.
  - The user's own changes are untouched: `results/strategy_results.png`, `results/wfo_signals.csv`, and the untracked `handoff.md` (the U10 handoff).
- Scratch: `/private/tmp/claude-501/-Users-nick-Documents-Python-Projects-wavelet/eb6ac08c-a6a4-4323-a941-e675afd78bfa/scratchpad/probe/` holds the probe spec, ledger, logs and partial signals.

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run pytest -q` — 384 passed, about 4 min 16 s.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `uv run python -c "from experiments.spec import load_spec; print(len(load_spec('experiments/specs/u11_a_screen.yaml')[1]))"` — prints 950.
- `wc -l results/ledger.jsonl` — 181 (the real ledger, unchanged this session).
- `ls "/Users/nick/Documents/Python Projects/wavelet/data/cache/holdout_access.jsonl"` — must not exist.
- On the PC, after setup: `uv sync`, `uv run pytest -q`, then `uv run python -m experiments run experiments/specs/u11_a_screen.yaml --jobs <physical cores>`.

## Deferred + open questions
- Deferred:
  - 1Min timeframe — not cached; SPEC limits it to ≤ 3 symbols.
  - GPU xgb — recommended against: the fits are small and GPU results are not bit-reproducible.
  - SPEC §9 update for the code-hash changes.
  - Survivor selection — rule still to define: "meta AUC > 0.52 and positive PSR, top-K". Planned reading: PSR > 0.5, i.e. Sharpe > 0, with K to be chosen. The selection tool should land before Stage A starts if it lives in hashed code.
  - Report extensions for the U11 Artifact: heat-maps, ablations, holdout DSR, verdicts.
- Open:
  - Commit and push `unit/11-zoo-study` so the PC can pull it (no merge yet). The question was asked and not answered.
  - Memory per process for 5Min cells was not measured. `--jobs` on the PC should respect RAM.

## Pick up here
Get the user's answer on committing and pushing `unit/11-zoo-study`. Then update SPEC §9 for the code-hash change and help them set up the Stage A run on the Windows PC with the copied ledger, artifacts and data cache.