# Session Handoff — PLAN3: reassessment after the U18 null, engine/stats audits, new working plan (PR #22)

## Where it started
User asked for a rigorous reassessment after every PLAN2 family failed U18: verify (not assume) the engine's soundness, evaluate Meyers-style DSP strategies (RMedV, polynomial velocity, adaptive Goertzel), the user's own RMV repo and `fastgoertzel`, research current state-of-the-art intraday/swing edges for a retail Alpaca account (5Min–1Day), and write a new plan with units of work and compute budgets. Constraints carried over: nothing below 5Min, pre-registration, retail Alpaca.

## Decisions locked + what shipped
- Engine audit (adversarial subagent): no BREAKING/SEVERE; four invariants hold (always-long = B&H to 2e-16; random-sign gross ≈ 0 with exact booked cost; side-flip negates P&L; overnight×intraday = close-to-close); stored F3/F5/F7 cells rebuild to 1e-16; 772 tests pass. Nine MINOR items (auction prints not modelled, no generic same-bar fill-timing test, 11 ETF caches past HOLDOUT_START, stress spreads 1.5–3.5×, raw Sharpe/zero cash yield, full-window pooling weights, PDT unmodelled) → all queued in U22. Lives in `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN3.md` §0.3.
- Statistics audit (adversarial subagent): inference code correct (every p/DSR reproduced); design underpowered — pass probability at top of PLAN2 ranges 22 % (F1), 10 % (F3), ≤ 2 % others; Holm-level MDE Δ Sharpe 1.65–1.86 for overlays; 0.8×B&H floor decided coherence in 7/8 families. → PLAN3 §0.1, protocol v3 (§4).
- Diagnostics on cached SIP bars (SPY/QQQ 2016-01 → 2025-09-30, no later bars read, K₀ = 60 looks): Goertzel cycle forecast has no predictive power (corr −0.01), no daily-cycle power; band-breakout (Zarattini) and stop-and-reverse LS-velocity rules earn 3–6 bp/day gross, zero beta, long-vol; 4-leg portfolio Sharpe 1.29 / 0.95 / 0.32 at 0.3 / 1.0 / 2.3 bp round-trip cost. Alpaca 1Day close = official auction print, differs from the 15:55 bar by 0.4–1.2 bp. → PLAN3 §1.
- `PLAN3.md` written (649 lines): thesis = zero-beta long-vol intraday overlay (G1, pre-registered region portfolio over band/RMedV/SG-velocity cells, 30-min cadence) on a vol-managed core (G2), vol-state gate (G3), time–frequency state features (G4, splits only), gated stocks-in-play ORB (G5), optional SVXY VRP (G6); units U22–U31 with compute budgets; §7 open user decisions.
- Diagnostic + audit scripts preserved at `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\scripts\plan3_diagnostics\` (ruff-fixed/formatted; three machine-specific paths replaced by `Path(__file__)`-derived repo root; directory excluded from ruff in `pyproject.toml` line 30).
- Committed `6911d6e` on branch `plan/3-intraday-program`, pushed, PR opened: https://github.com/0zean/wavelet-meta-model/pull/22 (bound to this session; repo has no CI checks). Not merged.
- Both audit agents stopped (reports delivered). Four stray CSVs the stats agent wrote to the repo root were moved out before committing.

## Key files for next session
- Plan file: `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN3.md` — read first; §5 U22 is the next unit.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\handoff.md` — still the U18 handoff; not updated this session (PLAN3 supersedes its "pick up here").
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\scripts\plan3_diagnostics\` — evidence scripts (`positions.py`, `costs.py`, `splits.py`, `goertzel_fc.py`, `close_check.py`, `audit_engine/`, `audit_stats/`).
- Session scratchpad (may not survive): `C:\Users\Nick\AppData\Local\Temp\claude\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\96a7bfdb-8f7d-432b-8339-1f86d01d6658\scratchpad\` — Meyers PDFs, Zarattini PDF, RMV repo PLAN.md, audit logs/CSVs.
- Memory files touched: `C:\Users\Nick\.claude\projects\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\memory\plan3-intraday-program.md` (new), `...\memory\hypothesis-family-program.md` (marked closed), `...\memory\MEMORY.md` (index).

## Running state
- Background processes: none (both audit agents stopped).
- Dev servers / ports: none.
- Open worktrees / branches: `plan/3-intraday-program` checked out in the main working tree at `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model` (tracks origin); local `main` = `04ee709`; `unit/18-family-dev-tests` still exists locally and remotely.

## Verification — how to confirm things still work
- `uvx ruff check . && uvx ruff format --check .` — clean (verified after the exclude).
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q` — 772 passed (~5 min; run by the engine audit, not re-run after the commit; the commit touches no code under test).
- `uv run python scripts/plan3_diagnostics/positions.py SPY` — reproduces the §1.2 gross table (band half-hourly 2.95 bp/day, SAR N=6 4.12 bp/day).
- `gh pr view 22` — open, base `main`, head `plan/3-intraday-program`.

## Deferred + open questions
- Deferred: `handoff.md` not rewritten for PLAN3 — PLAN3 §9 is the status line; update handoff.md when U22 starts.
- Deferred: `uv run pytest -q` not re-run post-commit (no code changes).
- Open (user, PLAN3 §7): Algo Trader Plus (~$99/mo) before U27; buy a point-in-time stock universe for G5/U29; run G6 (SVXY) or not; `INIT_CASH` $30k for registered specs; truncate the eleven ETF caches that hold bars past 2026-10-01.
- Open: merge PR #22 (user's call); delete remote `origin/unit/*` branches (carried over from U18).

## Pick up here
Once PR #22 is merged (or on the branch), start U22 per PLAN3 §5 on `unit/22-engine-protocol-v3`: auction-print fills from 1Day bars, `SLIPPAGE_BP` cost curve, account profile (PDT), protocol v3 in `families/test.py`, `families/power.py`, `families/looks.jsonl` seeded with K₀ = 60.