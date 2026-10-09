# Session Handoff — U11 post-mortem review and the hypothesis-family plan (PLAN2)

## Where it started
The user asked for a critical review of the completed U11 study (14 finalists, holdout null): salvageability, defects in plan/data/design, speed, better features/mechanisms, plus two papers (Toulson WEAPON/STTS; Ahrabian phase synchronization) and seven Quant Beckman links. After accepting the review, the user asked to turn its findings into the plan and spec as hypothesis families with composable units, to explore intraday mechanisms at 5Min or coarser only, and to assess a third paper (Kablan 2009, ISOM). Constraints that emerged: retail Alpaca account, beat buy-and-hold at least risk-adjusted, nothing below 5Min, ML only as an overlay.

## Decisions locked + what shipped
- Verdict: none of the 14 finalists is usable; high-Sharpe cells earn ~0.2–1.8 %/yr (QQQ 1Hour: 1.0 %/yr at 0.4 % vol, flat 81 % of days), the high-return cells reversed; EW of the 14 holdout streams Sharpe −0.35. Written to `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\REVIEW.md` (§1–§11: defects, kept parts, speed anatomy, data/selection fixes, candidate edges, papers, blog).
- New plan file instead of appending: `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN2.md` (thesis, protocol additions, families F1–F11, units U12–U21 with scope/done-when/reviewer focus/cost/dependencies, status all "not started").
- Spec amended in place by appending §11–§19 to `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` (lines 517–776): windows/holdout/stages F,G,H and trial budget; engine defaults (`ZOO_FIXED_PARAMS`, `CALIBRATION=rolling`, `OOF_META=reuse`, `PWFO_COMBINE=average`, 2-combo grid, flat pools, `COST_MODEL=quotes`, `VOL_PROFILE`); exogenous data layer; event samplers (`cusum|dc|schedule`) and exit models (`triple_barrier|time|hysteresis`); ISOM-style time-of-day vol profile; state feature groups; mechanism primaries; family spec YAML and test protocol (headline + Ledoit–Wolf block-bootstrap Sharpe difference + floors + coherence + Holm across families); forward test; cost model.
- Banner prepended to `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN.md` pointing to PLAN2/REVIEW; U1–U11 text untouched.
- Holdout policy: dev window stays 2016-01-04 → 2025-09-30; 2025-10-01 → 2026-09-27 is a reported "quasi-holdout" (contaminated), never a gate; the real holdout is forward paper trading (U20); `HOLDOUT_START` to move to 2026-10-01 (planned in SPEC §11.1/§18, not yet changed in code).
- Trial budget: 8 families / 112 trials, enforced by the runner (planned); pre-registration = committed `families/<id>.yaml` with SHA check (planned).
- ISOM paper: trading result judged not credible; construct adopted as `VOL_PROFILE="tod"`, DC sampler, `session` features, ISOM/IAOM diagnostics (SPEC §14); F11 is the one-shot re-test of the U11 null under it.
- Data facts from web checks that shaped U13/U15: Alpaca options history from 2024-02 on the free indicative feed, Greeks yes, open interest unconfirmed; CBOE CSV pattern `cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv` confirmed for VIX3M only; intraday momentum literature R² 2–3.5 % with post-2013 decay, gamma-hedging mechanism (Baltussen et al. 2021).
- No code changed. The doc changes (PLAN.md banner, SPEC.md §11–§19, new PLAN2.md and REVIEW.md, this handoff) are committed to `main`.

## Key files for next session
- Plan file: `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN2.md` — read first; U12 is the next unit.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\SPEC.md` — §11–§19 are the interfaces U12–U21 implement; §1–§10 still apply unless §11 amends them.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\REVIEW.md` — rationale; §6 has the per-window fit-count anatomy U12 targets.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\PLAN.md` — history only.
- `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model\handoff.md` — this file.
- Memory files touched: `C:\Users\Nick\.claude\projects\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\memory\hypothesis-family-program.md` (new) and `...\memory\MEMORY.md` (index line added).
- Scratch (disposable): `C:\Users\Nick\AppData\Local\Temp\claude\C--Users-Nick-Desktop-Code-Python-Projects-wavelet-meta-model\b35dfad8-32f6-44c3-a1e4-e9333d6bdd54\scratchpad\spec_amendments.md`, `...\plan_banner.md`.

## Running state
- Background processes: none
- Dev servers / ports: none
- Open worktrees / branches: main checkout `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model` on `main` at 0554ead with the doc changes above committed; worktree `C:\Users\Nick\Desktop\Code\Python Projects\wavelet-meta-model-stage-b` on `unit/11-stage-b-models` (carried over from earlier sessions, untouched)

## Verification — how to confirm things still work
- `git log -1 main` — the docs commit for PLAN2/REVIEW/SPEC §11–§19; `git status --short` clean apart from U12 work
- `wc -l SPEC.md PLAN.md PLAN2.md REVIEW.md` — 776, 1125, 710, 452
- `grep -n '^## §1[1-9]' SPEC.md` — nine headings, §11 at line 517 through §19 at line 766
- `head -4 PLAN.md` — the dated banner, then the original title
- `uv run pytest -q` (with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) — expected 418 passed; not run this session (no `.py` touched)
- `uv run python -c "from experiments.runner import code_hash; print(code_hash()[:16])"` — expected `23c33327f7d89a05` (only Markdown changed); not run this session
- Note: bare `python` in Bash hits a pyenv shim with no version; use `uv run python`

## Deferred + open questions
- Deferred: `HOLDOUT_START` in `data/bars.py` is still 2025-10-01; PLAN2 moves it in U13/U20.
- Deferred: U13 must verify the VIX/VIX9D/VIX6M CSV URLs and whether Alpaca option snapshots include open interest (gamma proxy falls back to skew if not).
- Deferred: numba dependency decided by benchmark in U12 (not installed now).
- Deferred (carried over): removing the stage-b worktree; pushing/merging old `unit/11-*` branches.
- Open: whether to pursue F9 (VRP via SVXY/options) at all; the extended ETF universe for F2 (IEF, LQD, HYG, DBC, USO, UUP, EFA, EEM, VNQ, SLV, SVXY) is listed in U13 but not confirmed by the user.
- Open: the offer to publish REVIEW.md as a shareable page was not answered.

## Pick up here
Start U12 on branch `unit/12-harness-speed`: build `tests/fixtures/u12_parity.json` from four U11 Stage C cells and the parity test first, then the `ZOO_FIXED_PARAMS` / `CALIBRATION` / `OOF_META` / `PWFO_COMBINE` switches with their old values as defaults until parity passes.