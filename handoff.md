# Session Handoff — U22: engine corrections and protocol v3 (branch `unit/22-engine-protocol-v3`)

## Where it started
PR #22 (PLAN3) was merged; `main` checked out. The user asked to begin U22 per PLAN3 §5 and gave decisions: Algo Trader
Plus will be bought before U27 (details to come), G5's point-in-time universe purchase still under consideration, G6
will run, `INIT_CASH` $30k agreed. U22 was built in full on `unit/22-engine-protocol-v3`; SPEC §20–§22 describe the
as-built engine; PLAN3 §5 U22 lists every done-when with its evidence; PLAN3 §9 is the status line.

## What shipped (SPEC §20–§22 have the detail)
- **Auction prints.** Native `1DayPrint` timeframe (Alpaca daily open / close = the official prints; H/L/volume
  include extended hours and are unused), `data.bars.load_prints`, cached for the 30-symbol universe through
  2026-09-30. `RunConfig.FILL_AUCTION` (`print` default, `last_bar` regression): the simulator prices every auction
  fill at the print, counts fallbacks. Print vs 15:55 bar close: median 0.7–1.3 bp, no sign, up to 288 bp in March
  2020. F3 / F5 re-run under print: Sharpe −0.017 / −0.042.
- **Costs.** `SLIPPAGE_BP` per fill kind (or `measured`), `COST_TABLE=asof` (`data/costs/quotes_half_spread_asof.csv`,
  trailing four completed sample weeks per quarter; fills before 2016-04-01 use the first table, counted),
  `STRESS_MULT` on previous-VIX ≥ 30 sessions, `CASH_YIELD=tbill` (FRED DTB3 cached, ACT/360 on free cash), the
  per-session cost ledger by fill class (`daily_costs.csv`) and `families.test.reprice` → the report's cost curve
  (registered / 0.3 / 1.0 / 2.3 bp / measured). MOC orders are sized from the previous bar's close.
- **Account profile** `risk/account.py`: PDT rule, Reg-T buying power, cash settlement / no shorts, locate estimate;
  flags only, never changes a P&L. F5 on $10k: PDT violation from 2016-04-06 (98.7 % of its day trades blocked); on
  $30k tradable.
- **Protocol v3** (`families/`): `test.kind` overlay_alpha | marginal | sharpe_vs_benchmark, `sided`, `at_cost`;
  excess returns everywhere; `power` block with the MDE line (`families/power.py`; the spec loader checks it);
  `core` for marginal; `account`; benchmarks `constant_mix_ew/er` (PLAN2 names are aliases), drifting `buy_and_hold`;
  `families/looks.jsonl` (K₀ = 60 seeded; `python -m families look`); DSR out of the verdict; `program.yaml` scoped
  to G1, G2, G3, G5, G6 (5 families / 42 trials); ex-ante weights (first 252 sessions when the window starts with
  the data); N_eff; cells coherence hook for U23.
- **Engine-audit items.** Generic fill-timing test + the ENTRY / EXIT_HYST mutants fail it; per-event causality test
  + the VOL mutant fails it; `data/fetch.py` defaults to HOLDOUT_START and logs forward fetches; eleven ETF caches
  truncated with records in `data/cache/forward_access.jsonl`; exchange-calendar session clock (`session_clock`,
  threaded through sampler / exits / rule pass / runner); `slip_through` barrier option; `ALLOW_CONTINUOUS` primaries
  with the `next_event` time exit (the simulator's roll path trades only the change; verified by hand).
- **Fix found on the way.** `position_returns` grouped the last leg of a chain with the first of the next (the `rolled`
  flag marks the trade whose EXIT rolled); `families.test.chain_ids` fixes it, `risk.account` uses it (and ends a
  position at a side flip); the test_u18 synthetic case was corrected to the simulator convention.
- Statistics: overlay-alpha size 0.050 (iid) / 0.045 (GARCH-t) over 1,000 sims; power 0.905 at 3 bp/day, 0.795 at
  the analytic MDE; MDE at Holm over 4 families 3.17 bp/day (Sharpe 1.0).

## Verification — how to confirm things still work
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — 804 passed
  (6 min) on the final code, after the adversarial review fixes.
- Adversarial review (one pass) found two BREAKING, two SEVERE and six MINOR items, all fixed: the account module
  grouped rolled chains with the old convention (and ignored side flips), the edge floor counted T-bill interest as
  edge, F3 did not reproduce under the old switches (MOC sizing and the session clock now have switches
  `MOC_SIZE_FROM` / `SESSION_CLOCK`), the legacy backtest path dropped prints and the yield (`portfolio_path` now
  routes them), plus the minor items listed in PLAN3 §5 U22. F3 and F5 now reproduce bit for bit under the old
  switches; the print-only effect is F3 Sharpe 0.725 → 0.707, F5 −0.122 → −0.164.
- `uv run python scripts/u22_checks.py prints | f5-parity ROOT | print-rerun ROOT | power | tilt | account | mde`.
- U12 parity: `scripts/u12_parity.py spec` → `python -m experiments --root R --ledger R/ledger.jsonl run` →
  `scripts/u12_parity.py check` (OLD_SWITCHES now pin the five U22 switches).

## Key files for next session
- `PLAN3.md` §5 U23 is the next unit (kernels + region primary); SPEC §21 states what U22 built for it
  (`ALLOW_CONTINUOUS`, `next_event`, `cells_coherence`, the cost curve).
- `families/test.py`, `families/run.py` (rewritten), `risk/portfolio.py` (prints, ledger, yield), `risk/account.py`,
  `families/power.py`, `families/looks.py`, `scripts/u22_checks.py`, `tests/test_u22.py`.
- Scratch outputs of this session (may not survive): `…\scratchpad\f5_parity`, `print_rerun`, `u12`, `review`.

## Running state
- No experiment runs on the canonical ledger; the scratch runs wrote their own ledgers under the session scratchpad.
- Branch `unit/22-engine-protocol-v3` (from `main` at 682effe); PR to open at the end of the session.

## Deferred + open questions
- Deferred: the region primary, kernels and `position_backtest` (U23); the measured cost profile files
  (`data/costs/measured_slippage.csv`, `measured_cost.csv`) are written by U27; `tf_state` / gamma proxy (U26).
- Open (user): Algo Trader Plus details before U27; the G5 data purchase; the eleven truncated caches are
  re-fetchable (`python -m data.fetch --end 2026-10-10 …` logs a forward fetch) if ever needed.
- Open: delete remote `origin/unit/*` branches (carried over).

## Pick up here
Merge the U22 PR (user's call), then start U23 per PLAN3 §5 on `unit/23-intraday-kernels`: `features/kernels.py`
(rmedv_all, sg_velocity_all, band_state, session_vwap), `primaries/region.py` (`region_trend`, `ALLOW_CONTINUOUS`,
`next_event`), `wfo/position_backtest.py` with the parity test, `scripts/u23_kernels.py` budgets.
