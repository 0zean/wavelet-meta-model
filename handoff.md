# Session Handoff — U24: G1 registration and development-window test (branch `unit/24-g1-dev-test`)

## Where it started
U23 (PR #24) was merged to `main` (d7c83a5). U24 was built, registered, run and reviewed on `unit/24-g1-dev-test`:
- SPEC §25 describes the as-built code;
- PLAN3 §5 U24 lists every done-when with its evidence;
- PLAN3 §9 is the status line.

## What shipped (SPEC §25 has the detail)
- **`families/G1.yaml`.** R₀ exactly as PLAN3 §3 G1 states it:
  - 43 cells per instrument, SPY + QQQ pooled 50/50, every region parameter written out;
  - `loss_gate` (2 % daily) the only control, INIT_CASH 30k on margin_30k, U22 switches pinned;
  - `overlay_alpha` one-sided at a 1.0 bp round trip, `coherence: cells`;
  - floors 3 %/yr and edge ≥ 2× cost;
  - 10 variants (IWM/DIA is a sample split, since the loader refuses instruments as a variant);
  - registered as a **diagnostic**: expected 3.0 < MDE 3.17 bp/day (the user's choice).
- **Code:**
  - `modwt_slope_all` and `session_rv_sigma` (features/kernels.py);
  - the `modwt` estimator and `vol_target` size (primaries/region.py);
  - the `loss_gate` profile;
  - families: `weighting: equal`, `test.coherence: cells`, `region_cells`, trade stats, per-instrument alpha, realized
    MDE, the vol_quintile / opex_day splits, report sections. `region_cells` is the 86-cell specification curve
    re-simulated by position_backtest, with a parity check against the members (exactly 0).
- `scripts/u24_g1.py smoke` (status only), `tests/test_u24.py` (22 tests).

## Result (the first look at G1; recorded as n = 98, K now 158)
- **Headline:** alpha 2.09 bp/day (5.3 %/yr excess) at 1.0 bp; one-sided p 0.0034; floors pass; 83 / 86 cells positive.
  The decision rule passes, so **G1 proceeds to U25**.
- **Caveats:**
  - Cost-bound: p 0.108 at booked costs (≈ 2.6 bp round trip) and 0.152 at 2.3 bp.
  - IWM/DIA have the wrong sign.
  - The quasi-holdout slice is −1.34 bp/day.
  - The edge is in vol quintiles q4–q5 only.
  - 2018 and 2022 carry most of the result.

## Verification
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — 841 passed.
- `uv run python scripts/u24_g1.py smoke` — all trials ok, parity 0.0, looks unchanged.
- `results/families/G1/report.md` — the re-run under the review fixes (d0a31c0). The first run's result is in commit
  894ece8.

## Open decisions (the user's)
- Algo Trader Plus details before U27; the G5 data purchase; delete remote `origin/unit/*` branches.
- **New:** whether G2/G3 (U25) should register with the caveats above in view. In particular, G3's gate as PLAN3 words
  it ("skip the bottom quintile") reads a split already seen. Its registration should say so, or move the gate to the
  bottom three quintiles with that disclosed.

## Pick up here
Start U25 on `unit/25-g2-g3`:
- `families/G2.yaml`: core = the F1 headline on SPY/QQQ re-run under U22's fills and the cash yield; overlay = G1 at
  k = 1; `marginal` kind.
- `families/G3.yaml`: G1 × the vol gate, paired one-sided.
- Each needs its MDE line, registration before any run, and the report with the matched-vol comparison and the
  buying-power check.
- The `marginal` kind reads the core family's `streams.csv` from results/families/<core>/: the F1 core must be re-run
  (a new family id) first.
