# Session Handoff — U26: time–frequency state and gamma proxies (branch `unit/26-tf-state`)

## Where it started
U25 (PR #26) was merged to `main` (eb6133b). U26 was built, reviewed and its diagnostic recorded on `unit/26-tf-state`:
- SPEC §24 describes the as-built code, the review and the result;
- PLAN3 §5 U26 lists every done-when with its evidence;
- PLAN3 §9 is the status line.

## What shipped (SPEC §24 has the detail)
- **`tf_state`** (`features/tf_state.py`, an intraday group). Per session, known at the open, from the previous 5
  sessions' inside-session 5Min returns:
  - `pow_hf` / `pow_lf` (Goertzel band power at periods 2–6 / 12–78 bars vs white noise) and `tilt` (their log ratio);
  - `flatness`;
  - `modwt_e1..e5` (Haar MODWT energy vs white noise);
  - `ac1` (the revealed-gamma proxy);
  - `vov` (SD of log session σ over 21 sessions).
  - SPY 2016–2025 takes 0.69 s against a 10 s budget.
- **fastgoertzel pinned at 0.1.9**:
  - 1.x needs numpy ≥ 2.2, and `uv add` silently moved numpy 1.26 → 2.4, so it was reverted.
  - 0.1.9 truncates the bin, int(f · N): at f = k / N, 18 of 192 bins land on k − 1. `goertzel_power` calls it at
    (k + 0.5) / N; parity with a numpy DFT to 1e-10.
  - Worth fixing upstream in fastgoertzel (the user's package) if 1.x still truncates.
- **SKEW / VVIX** in `data/exo.py` (CBOE close-only files, caches committed) and `vol_state` (`skew`, `vvix`,
  `vvix_vix`).
- **Splits.**
  - `gamma_sign` and 14 tercile splits (`families/spec.py` `TERCILE_SPLITS`, `U26_SPLITS`).
  - `families.run.add_splits` and `python -m families splits <id> --look <label>` add splits to a stored result
    without a re-run. They require a recorded look, never overwrite, check stored splits first, and rewrite only the
    report's splits section.
- **Scripts and tests.** `scripts/u26_tf_state.py budget | null` (status only); tests/test_u26.py.

## Result
- The look `families/G1 U26 splits` (n = 15) was recorded before the splits were computed. K = 186.
- No tf_state column and not the gamma sign separates G1's P&L: no tercile pattern is monotone, and every interval
  overlaps.
- Low SKEW (4.42 vs 0.36 bp/day) and low VVIX / VIX (5.55 vs 0.50) separate most. These are the high-VIX states, so
  most likely the known vol effect restated.
- Everything is reported, never tested.
- The §1.1 session-shift null is reproduced exactly with the group's Goertzel.

## Verification
- `uvx ruff check . && uvx ruff format --check .` — clean.
- `OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 uv run pytest -q -p no:warnings` — the full suite passes.
- `uv run python scripts/u26_tf_state.py budget` and `... null`.
- Adversarial review: no BREAKING or SEVERE finding; the five MINORs were fixed before the look.

## Notes for the next session
- Write files with LF and UTF-8. On Windows, `Path.write_text` writes CRLF, so use `write_bytes` or the Write tool.
- A heredoc turns `"\\n"` into real newlines.
- The reviewer's real-data checks refetched SPY's cache metadata (`coverage_start` 2010). The sidecar was restored
  from git, and the bars were byte-identical. Watch for that after a review.
- The G1 report's state splits now follow the spec's order. The committed U25 version was alphabetical, from an
  earlier re-render.

## Open decisions (the user's)
- Algo Trader Plus before U27. U27 then trades G1 alone: a diagnostic whose result is bound by the cost.
- Whether G2's portfolio claim, which failed its drawdown floor, is worth keeping a forward record of.
- The G5 data purchase; deleting the remote `origin/unit/*` branches; opening a PR for this branch.

## Pick up here
- **U27** (forward test) once Algo Trader Plus is bought. Its input is `families/registry.yaml` (G1).
- **Meanwhile:** U28 (G6 SVXY).
- **Deferred MINORs from U25:**
  - the loader's PLAN2 2 % default floor applies to paired specs;
  - a gate that delays a stream's first live day would drop reference days;
  - one failed core aborts a marginal test.
