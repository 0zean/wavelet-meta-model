# Session Handoff — U14 event samplers, exit models, intraday volatility profile (merged to main)

## Where it started
The user asked me to review `handoff.md` (U13 merged as PR #15) and begin U14 per PLAN2.md and SPEC §13–§14. U14 was
built on `unit/14-samplers-exits-isom`, adversarially reviewed, fixed, re-verified, and then (at the user's request)
pushed, opened as a PR and merged into `main` together with this handoff.

## Decisions locked + what shipped
- New modules:
  - `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\features\vol_profile.py` — VolProfile, `bar_volatility` (moved here), `hold_scale`, `isom_counts`.
  - `...\features\events.py` — the cusum / dc / schedule samplers; `sample_events` moved here.
  - `...\features\exits.py` — `exit_frame` dispatcher for triple_barrier / time / hysteresis, plus `exit_phase`.
  - `...\features\session.py` — the session frame; the per-fold `session` group is registered in `features\groups.py`.
- RunConfig switches (`utils\config.py`), validated in `events.event_params` / `exits.exit_params`. Defaults reproduce U13 exactly:
  - `VOL_PROFILE` ("none" | "tod")
  - `EVENT_SAMPLER` ("cusum" | "dc" | "schedule") with `EVENT_PARAMS`
  - `EXIT_MODEL` ("triple_barrier" | "time" | "hysteresis") with `EXIT_PARAMS`
  - CLI flags in `wavelet_meta_model.py`.
- Engine wiring:
  - Every former `barrier_exits` caller now goes through `exit_frame`: meta_model, backtest, wfo_metrics, diagnostics, compare, portfolio.
  - `wfo\wfo_engine.py`: `window_events` re-samples events, widths and labels per window under "tod" (profile fit on [fit_start, train_end − embargo)).
  - `isom_folds` is written to signals.attrs and so to `wfo_run.json`.
  - `CalHistory` purges each pair on its own label span (`add(..., spans=labels)`).
- Deliberate deviations from SPEC, all documented in SPEC §14 "Implementation (U14, as built)":
  - **Barrier widths under "tod":** σ_base · √Σ s² over the held slots. Slot 0 carries the overnight gap: SPY s(0) = 8.7.
  - **Opening-print entries:** the first held slot counts the gap-free `s_open` (2.2 on SPY).
  - **DC threshold:** δ = dc_mult · σ_t.
  - **Hysteresis:** fills at the next bar's open.
  - **`first30` before 10:00:** the return so far.
  - **`activity`:** IAOM of DC events with plain σ.
- Schedule `days`:
  - A calendar event counts only if `available_at` ≤ the decision time.
  - amc earnings react in the next data session, within 4 calendar days; otherwise they match no session.
  - Early close = the session's last bar spans 13:00.
  - RunConfig refuses 09:30 entries with a non-time exit (unless HOLD_OVERNIGHT).
- Parity and hashing:
  - `scripts\u12_parity.py` `OLD_SWITCHES` += VOL_PROFILE none, EVENT_SAMPLER cusum, EXIT_MODEL triple_barrier. The pinned list in `tests\test_harness_speed.py` is updated to match.
  - `experiments\runner.py` `code_hash` now also hashes `data\calendar\*.csv`.
  - The five new fields are `_NON_FEATURE_FIELDS` in `features\cache.py`, but the feature code hash changed, so every cached static feature rebuilds once.
- Diagnostic: `scripts\u14_isom.py` → `results\u14\isom_spy_5min.{json,png}`.
  - The 30-minute open slot moves 2.2× the 12:30 slot.
  - First-hour event share: CUSUM 21.8% → 16.9% with tod; DC 23.4% → 17.5%.
- Records: the PLAN2.md U14 status note, and the SPEC.md §14 as-built notes.

## Key files for next session
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN2.md` — the U14 status note (deferred items). U15 (state features) and U16 (mechanism primaries) are next.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` — §13/§14 plus "Implementation (U14, as built)"; §15–§16 for the next units.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\features\events.py`, `features\exits.py`, `features\vol_profile.py`, `features\session.py` — the APIs U16 primaries will declare (SAMPLER / EXIT).
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\wfo\wfo_engine.py` — `window_events`, `CalHistory`, `isom_diagnostic`.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\tests\test_u14.py` — 55 tests: hand cases, causality and the mutation checks.
- Plan file: none (PLAN2.md drives the program).
- Memory files touched: none.

## Running state
- Background processes: none (the parity runs, the legacy runs and the reviewer all finished).
- Dev servers / ports: none.
- Open worktrees / branches:
  - `unit/14-samplers-exits-isom` is merged into `main`; it can be deleted.
  - Carried over and untouched: worktree `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b`, and `origin/unit/13-exo-data-costs` (merged).
- Scratch (disposable), in the session scratchpad: parity roots `parity\` and `parity2\`, legacy outputs, the reviewer probe scripts under `review\`.

## Verification — how to confirm things still work
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q` — 564 passed.
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `PYTHONIOENCODING=utf-8 MPLBACKEND=Agg uv run python wavelet_meta_model.py data/data.csv --out R` — `R/wfo_signals.csv` sha1 `551d8074…`.
- Parity (~17 min):
  1. `uv run python scripts/u12_parity.py spec P.yaml`
  2. `LOKY_MAX_CPU_COUNT=1 python -m experiments --root R --ledger R/ledger.jsonl run P.yaml --jobs 24`
  3. `uv run python scripts/u12_parity.py check R R/ledger.jsonl` → `parity: PASS` (8/8).
- `uv run python scripts/u14_isom.py` — reproduces `results\u14\isom_spy_5min.json` (cached SPY 5Min bars, no network).

## Deferred + open questions
- Deferred: under time/hysteresis exits, `width` is the sampler's barrier width, not the σ of the actual hold. Only the risk layer's vol target reads it — U16.
- Deferred: MOC entries (the `overnight` primary: buy at the close) are not modelled; every entry fills at a bar's open — U16.
- Deferred: event times treat the bar stamps as the calendar, so a session truncated mid-day can gain or lose a late scheduled event. The live loop must use the exchange calendar — U20.
- Deferred: the `session` group is per fold and uncached, about 0.7 s per window on SPY 5Min (about 3 min per 5Min WFO cell).
- Deferred: `mins_to_close` assumes 16:00, as the `intraday` group does.
- Carried over: stress-session costs (U17); the credit-spread source and VIX holiday rows (U15); exo `coverage_end` (U20); SVXY's 2018 break (F9/U21); cost-target tuning; the 1Day `MIN_VAL_EVENTS` calibration question (U17/U18).
- Open: delete the merged `unit/13-…` / `unit/14-…` branches and the stage-b worktree? (the user's call).

## Pick up here
Start U15 (state and context feature groups; PLAN2 U15, SPEC §15) on a new branch `unit/15-...` off `main`. It needs the U13 exo/event layers and extends the U14 `session` group's conventions (`needs=("exo",)`, `available_at` alignment).
