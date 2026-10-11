# Session Handoff — U25: G2 portfolio test and G3 volatility gate (branch `unit/25-g2-g3`)

## Where it started
U24 (PR #25) was merged to `main` (51dc721). U25 was built, reviewed, registered and run on `unit/25-g2-g3`:
- SPEC §26 describes the as-built code, the review and the result;
- PLAN3 §5 U25 lists every done-when with its evidence;
- PLAN3 §9 is the status line.

## User decisions (2026-10-11)
- G3's headline gate is PLAN3's bottom vol quintile, written before G1's look. The bottom three quintiles that G1's
  split suggested were rejected.
- The band-width gate is dropped: it can never fire (the 10:00 band is ≥ 5.4 bp against a 2 bp threshold).
- The VIX < 13 gate is dropped until U26's state plumbing.

## What shipped (SPEC §26 has the detail)
- **Gate.** `region_trend` has `gate: {vol_quantile | vol_size | gamma}`, per instrument and known at the open, from
  features/kernels.py: `session_sigma`, `prior_rank`, `prior_lag1_autocorr`. `families.run.vol_rank` reads the same
  kernels. R₀ is unchanged.
- **Test kinds.**
  - New `paired` kind with `reference: {family, variant, sha}`.
  - `marginal` cores (`core: {family, variant, k, sha}`) are per variant. They and the paired reference are registered
    specs whose cells run in the family's batch: no trial, no Holm, pinned by sha.
- **Floors.** `max_dd_vs_core` and `min_net_ret_vs_reference`.
- **Power.** The Δ Sharpe MDE line, `families.power.mde_delta_sharpe`.
- **Report.** The matched-volatility comparison and the core + k × overlay account table.
- **Registry.** `families/registry.py` and `python -m families registry <id>`.
- **Specs.**
  - families/G2.yaml and G3.yaml, registered at e66c314;
  - G2_core.yaml and G2_core4.yaml, registered at 4208d7b.
- **Other.** `scripts/u25_g2_g3.py smoke` (status only) and tests/test_u25.py (17 tests).

## Result
- **G2 fails.**
  - Sharpe(core + G1) 1.33 vs the core's 1.06: Δ +0.26, p 0.040, Holm 0.080.
  - The drawdown floor fails (−20.5 % vs −19.4 %): the portfolio's worst drawdown is the 2020 crash.
  - Not coherent.
  - Cost-bound: +0.07 at the booked costs.
- **G3 fails.** Δ +0.03, p 0.31. The gate helps more at higher costs (+0.11 at 2.3 bp), reported only.
- **Program verdict.** G1 PASS (Holm 0.010). `families/registry.yaml` holds G1 only. Looks K = 171.
- **Disclosure.** A status grep of the run logs showed the floors, coherence and quasi lines before the looks were
  recorded. No p or headline statistic was seen first. Both looks' notes say so.

## Verification
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — the full suite passes.
- `uv run python scripts/u25_g2_g3.py smoke` — every variant, core and reference ok; parity 0.0; looks unchanged.
- `uv run python -m families summary` — G1 PASS, G2 fail, G3 fail.
- G1's headline under U25's code equals U24's stream exactly over 2,436 days.

## Open decisions (the user's)
- Algo Trader Plus before U27. U27 then trades G1 alone: a diagnostic whose result is bound by the cost.
- Whether G2's portfolio claim, which failed its drawdown floor, is worth keeping a forward record of. It cannot be
  re-selected: an amendment would be a new file with a reason.
- The G5 data purchase; deleting the remote `origin/unit/*` branches; opening a PR for this branch.

## Pick up here
- **U27** (forward test) once Algo Trader Plus is bought. Its input is `families/registry.yaml` (G1).
- **Meanwhile:** U26 (time–frequency state, VIX / SKEW / VVIX splits) or U28 (G6 SVXY), both independent of U27.
- **Deferred MINORs for a later unit:**
  - the loader's PLAN2 2 % default floor applies to paired specs;
  - a gate that delays a stream's first live day would drop reference days;
  - one failed core aborts a marginal test.
