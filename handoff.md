# Session Handoff — U23: intraday kernels and the region primary (branch `unit/23-intraday-kernels`)

## Where it started
PR #23 (U22) was merged to `main` (62aaad2). The user asked to begin U23 per PLAN3 §5. U23 was built in full on
`unit/23-intraday-kernels`; SPEC §23 describes the as-built code; PLAN3 §5 U23 lists every done-when with its
evidence; PLAN3 §9 is the status line.

## What shipped (SPEC §23 has the detail)
- **Kernels** `features/kernels.py` (numba, float64, session-bounded windows: NaN for the first N − 1 bars of each
  session): `rmedv_all` (Siegel repeated median; bit-equal to scipy siegelslopes), `sg_velocity_all` / `sg_weights`
  (Meyers' polynomial velocity at the next bar T+1; degree 1 = LS slope), `prior_sd` (trailing SD over the k previous
  sessions), `rmedv_normalized` (RMedV · √N · xmult, Meyers 2025 App. III) and `poly_normalized` (velocity / its SD
  per (degree, N), Meyers 2026 App. III), both refit every session over 21 sessions; `band_state` (Zarattini band,
  gap-adjusted distance in band units), `session_vwap`, `session_layout`.
- **Region primary** `primaries/region.py` `region_trend` (ALLOW_CONTINUOUS): 43 cells (band × VM 3, rmedv and sgv ×
  N 5 × θ 4), each velocity in its paper's normalization (θ in SDs), stop-and-reverse reset every session, VWAP stops, decisions = schedule entry times
  10:00 … 15:30 (read at the 09:55 … 15:25 closes), `next_event` exit to the closing auction; target = the mean cell.
  Variants supported: cadence 5/15/30/60, `first`, `exit "HH:MM"`, `vel_mode flat_inside`, `vel_stop vwap`,
  `band_stop`, `sg_degree`. Not yet: the MODWT-slope estimator and the vol-targeted size (G1 variants; U24).
  `needs_groups()` is empty (the plan said "session group"; a group would drop events on NaN columns).
- **Position backtest** `wfo/position_backtest.py`: `decision_grid`, `position_backtest` — a share-for-share numba
  replica of `simulate_portfolio` for one symbol on this schedule (rolls, per-bar costs or a scalar one-way cost,
  closing print, daily-loss gate, T-bill yield), many target series per call (the spec / cost curve).
- `scripts/u23_kernels.py parity | budgets`, `tests/test_u23.py` (15 tests), `primaries/__init__.py` registers the
  region, `tests/test_primaries.py` knows the new registry entry.

## Verification — how to confirm things still work
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — full suite
  819 passed on the final code (after the review fixes; 5 min).
- `uv run python scripts/u23_kernels.py parity` — Meyers examples 1.0; 4,400/4,400 bit-equal to scipy; sgv 7.2e-16.
- `uv run python scripts/u23_kernels.py budgets` — estimators 1.2 s, positions 1.3 s, spec curve 0.13 s, headline
  14 s (budgets 5 / 10 / 5 / 120 s); position_backtest vs simulator over SPY/QQQ 2016–2025 with the gate: max |Δ| 0.
  It prints timings and parity residuals only — **no look** was taken; `families/looks.jsonl` unchanged.
- Adversarial review (one pass): no BREAKING / SEVERE; four MINOR fixed (exit variant on early closes / missing exit
  bar; NaN targets raise; numpy-int lookbacks; the √N decision recorded in PLAN3 §7). Probe scripts:
  session scratchpad `review/` (may not survive).

## Open decisions (the user's)
- **Settled (user, 2026-10-10): the velocity normalization follows the Meyers papers** (PLAN3 §7 item 6; RMedV ·
  √N · xmult; polynomial velocity at T+1 / its SD per (degree, N)); refit every session over 21 sessions (the RMV
  repo's window). Still the user's at U24 registration: the refit window (21 built). `scripts/u23_kernels.py scale`:
  normalized SD 1.03–1.14 at every N on SPY/QQQ 2016–2025.
- Carried: Algo Trader Plus details before U27; the G5 data purchase; delete remote `origin/unit/*` branches.

## Key files for next session
- `PLAN3.md` §5 U24 (G1 registration and the dev-window test — the first look; append to `families/looks.jsonl`
  before reading results); SPEC §22 (protocol v3: `overlay_alpha`, power/MDE line, `cells_coherence`, cost curve),
  SPEC §23.
- `primaries/region.py` (add the MODWT-slope estimator and vol-targeted size for G1 variants 6–7),
  `wfo/position_backtest.py` (the spec curve: `position_backtest(df, cells_frame, cfg, costs=X/2 bp, grid=...)`),
  `families/run.py`, `families/report.py`.

## Running state
- No experiment runs; nothing written to the canonical ledger. Branch `unit/23-intraday-kernels`; PR opened at the
  end of the session (see PLAN3 §9 / the PR link).

## Pick up here
Merge the U23 PR (user's call); confirm the 21-session refit window with the user; then start U24 on `unit/24-g1-dev-test`:
write `families/G1.yaml` (headline R₀, 11 variants, splits, `overlay_alpha` one-sided, MDE line, floors at 1.0 bp,
INIT_CASH 30k, a gate-only risk profile with daily_loss 0.02), commit it, smoke 2016-01 → 2016-06 (status only),
register, run once, report with the cost and specification curves.
