"""
U10: experiment specs, cell hashing, the ledger (trial counting, crash safety), the runner's cache / resume / error
capture / holdout guard, legacy import and the report.
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.runner as R
import wfo.pwfo as pw
import wfo.wfo_engine as eng
from data.bars import HoldoutError
from experiments import ledger as L
from experiments.legacy import import_legacy
from experiments.report import (
    alpha_stats,
    bh_fdr,
    holm,
    leaderboard,
    row_dsr,
    session_bootstrap_auc,
    stage_a_survivors,
    stage_b1_selection,
    stage_b2_selection,
    stage_b3_selection,
    stage_c_cell,
    stage_c_finalists,
    stage_c_selection,
    stage_e_cell,
    stage_e_selection,
    weighted_auc,
    write_report,
)
from experiments.spec import Cell, expand, normalize
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from validation.stats import dsr

SMALL = {"INITIAL_TRAIN": 700, "VAL": 350, "TEST": 100, "MIN_TRAIN_EVENTS": 50, "MIN_VAL_EVENTS": 30}
DEFAULTS = {"timeframe": "1Day", "start": "2012-01-01", "end": "2017-06-01", "overrides": SMALL}


def _die_elsewhere(pid: int):
    if os.getpid() != pid:
        os._exit(9)


class Killer:
    """Kills any other process that unpickles it (stands in for an OOM kill / segfault of a worker)."""

    def __reduce__(self):
        return (_die_elsewhere, (os.getpid(),))


class FakeSource:
    """Synthetic 1Day bars per symbol over business days [start, end); counts loads."""

    def __init__(self, fail: set[str] = frozenset()):
        self.fail, self.loads = fail, []

    def bars(self, symbol, timeframe, start, end, *, allow_holdout):
        self.loads.append((symbol, timeframe, start, end, allow_holdout))
        if symbol in self.fail:
            raise ValueError(f"no data for {symbol}")
        idx = pd.bdate_range(start, end, inclusive="left", tz="America/New_York")
        df = synthetic_daily(len(idx), seed=sum(map(ord, symbol)))
        df.index = idx
        if symbol == "DIE":
            df.attrs["killer"] = Killer()
        if timeframe == "5Min":  # four intraday bars per session (the Corwin–Schultz spread needs pairs)
            rng = np.random.default_rng(len(idx))
            stamps = [d + pd.Timedelta(minutes=m) for d in idx for m in (570, 575, 580, 585)]
            close = np.repeat(df["close"].to_numpy(), 4) * np.exp(rng.normal(0, 1e-3, 4 * len(idx)))
            spread = np.abs(rng.normal(0, 5e-4, close.size)) + 1e-4
            df = pd.DataFrame({"open": close, "high": close * (1 + spread), "low": close * (1 - spread),
                               "close": close, "volume": 1e5}, index=pd.DatetimeIndex(stamps))  # fmt: skip
        return df

    def sessions(self, end):
        return pd.bdate_range("2000-01-03", end, inclusive="left")

    def quotes_table(self, symbols, table="year"):
        from tests.test_costs import const_quotes_table

        return const_quotes_table(symbols)


def doc(stage="U10", grid=None, cells=None, **defaults):
    d = {"stage": stage, "defaults": {**DEFAULTS, **defaults}, "grid": grid or {"symbols": ["AAA"]}}
    if cells:
        d["cells"] = cells
    return d


@pytest.fixture
def env(tmp_path):
    return {
        "ledger": L.Ledger(tmp_path / "ledger.jsonl"),
        "root": tmp_path / "exp",
        "feature_cache_dir": None,
        "holdout_marker": tmp_path / "holdout_marker.jsonl",
    }


@pytest.fixture
def fits(monkeypatch):
    """Count fit_window calls (the WFO's and the PWFO's model fits)."""
    calls = []
    orig = eng.fit_window

    def spy(*a, **k):
        calls.append(a[3])
        return orig(*a, **k)

    monkeypatch.setattr(eng, "fit_window", spy)
    monkeypatch.setattr(pw, "fit_window", spy)
    return calls


# ── Spec ─────────────────────────────────────────────────────────────────────


def test_expand_is_a_cartesian_product_with_dotted_keys_and_dedup():
    cells = expand(doc(grid={"symbols": ["aaa", "BBB", ["bbb", "AAA"]], "sizer": ["fixed", "linear"],
                             "model.meta": ["logit_l2", "logit_l2"]}))  # fmt: skip
    assert len(cells) == 3 * 2  # the duplicated meta-model collapses
    assert {c.symbols for c in cells} == {("AAA",), ("BBB",), ("AAA", "BBB")}
    c = cells[0]
    assert c.spec["model"] == {"meta": "logit_l2", "primary": "legacy"} and c.stage == "U10"
    assert c.config().SIZER == "fixed" and c.config().INITIAL_TRAIN == 700


def test_explicit_cells_are_crossed_with_the_grid():
    cells = expand(doc(cells=[{"sizer": "linear"}, {"primary.name": "sma_cross", "risk_profile": "standard"}],
                       grid={"symbols": ["A", "B"]}))  # fmt: skip
    assert [(c.symbols[0], c.spec["sizer"], c.spec["primary"]["name"]) for c in cells] == [
        ("A", "linear", "wavelet_trend"), ("B", "linear", "wavelet_trend"),
        ("A", "fixed", "sma_cross"), ("B", "fixed", "sma_cross"),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "bad",
    [
        {"stage": "Z"},
        {"defaults": {**DEFAULTS, "colour": 1}},
        {"defaults": {**DEFAULTS, "model": {"meta": "nope"}}},
        {"defaults": {**DEFAULTS, "overrides": {"SIZER": "linear"}}},  # owned by the cell
        {"defaults": {**DEFAULTS, "overrides": {"NOT_A_FIELD": 1}}},
        {"defaults": {**DEFAULTS, "end": "2011-01-01"}},
        {"grid": {"symbols": []}},
        {"grid": {"symbols": [["A", "B"]]}, "defaults": {**DEFAULTS, "pwfo": {"is_grid": [400]}}},
    ],
)
def test_invalid_specs_fail_before_running(bad):
    with pytest.raises((ValueError, TypeError)):
        expand({**doc(), **bad})


def test_normalize_is_canonical():
    a = normalize({**DEFAULTS, "symbols": "spy", "primary": "wavelet_trend", "start": "2012-01-01T00:00"})
    b = normalize({**DEFAULTS, "symbols": ["SPY"], "primary": {"name": "wavelet_trend"}, "feature_groups": "default"})
    assert a == b and json.dumps(a, sort_keys=True) == Cell(b, "A").spec_json()


# ── Hashing ──────────────────────────────────────────────────────────────────


def _variant(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 0.9 if value else 0.5
    return None


def test_cell_hash_covers_every_runconfig_field():
    cell = expand(doc())[0]
    cfg = cell.config()
    h0 = R.cell_hash(cell, cfg, "d")
    tested = 0
    for f in dataclasses.fields(cfg):
        v = _variant(getattr(cfg, f.name))
        if v is None:
            continue
        try:
            alt = dataclasses.replace(cfg, **{f.name: v})
        except (ValueError, TypeError):
            continue
        assert R.cell_hash(cell, alt, "d") != h0, f.name
        tested += 1
    assert tested > 30
    assert R.cell_hash(cell, cfg, "other-data") != h0
    assert R.cell_hash(Cell(cell.spec, "B"), cfg, "d") == h0  # the stage is not part of the configuration


def test_code_hash_changes_with_source(monkeypatch, tmp_path):
    for d in R.CODE_DIRS:
        (tmp_path / d).mkdir()
    (tmp_path / "wfo" / "x.py").write_text("a = 1\n")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    R.code_hash.cache_clear()
    try:
        h1 = R.code_hash()
        (tmp_path / "wfo" / "x.py").write_text("a = 2\n")
        R.code_hash.cache_clear()
        assert R.code_hash() != h1
        (tmp_path / "experiments" / "report.py").write_text("# cannot change results\n")
        h2 = R.code_hash()
        R.code_hash.cache_clear()
        assert R.code_hash() == h2
    finally:
        R.code_hash.cache_clear()


def test_code_hash_ignores_line_endings_but_not_the_platform(monkeypatch, tmp_path):
    for d in R.CODE_DIRS:
        (tmp_path / d).mkdir()
    (tmp_path / "wfo" / "x.py").write_bytes(b"a = 1\nb = 2\n")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    R.code_hash.cache_clear()
    try:
        h1 = R.code_hash()
        (tmp_path / "wfo" / "x.py").write_bytes(b"a = 1\r\nb = 2\r\n")  # a Windows (autocrlf) checkout
        R.code_hash.cache_clear()
        assert R.code_hash() == h1
        monkeypatch.setattr(R.platform, "system", lambda: "Plan9")  # not the host, whichever it is
        monkeypatch.setattr(R.platform, "machine", lambda: "mips")
        R.code_hash.cache_clear()
        assert R.code_hash() != h1
    finally:
        R.code_hash.cache_clear()


def test_ledger_lock_round_trip_and_utf8(tmp_path):
    led = L.Ledger(tmp_path / "ledger.jsonl")
    with led.run_lock():
        led.append({"cell_hash": "h", "stage": "A", "status": "ok", "label": "σ ≥ 0 ═"})
        led.append({"cell_hash": "h2", "stage": "A", "status": "ok"})
    with led.run_lock():  # released
        pass
    assert [r["cell_hash"] for r in led.rows()] == ["h", "h2"] and led.rows()[0]["label"] == "σ ≥ 0 ═"


def test_backtest_only_fields_do_not_change_wfo_signals():
    df = synthetic_daily(1400)
    cfg = RunConfig.for_timeframe("1Day", PRIMARY="wavelet_trend", META_MODEL="logit_l2", META_TRAIN="oof", **SMALL)
    alt = cfg.replace(RISK_PROFILE="standard", SIZE_STEP=0.25, INIT_CASH=1e6, SIZE=0.5, PWFO_IS_GRID=(10,),
                      PWFO_OOS_GRID=(3,), PWFO_EXPANDING=True, PWFO_VAL_FRAC=0.2, PWFO_DEFAULT=(10, 3),
                      PWFO_MIN_WINDOWS=3, PWFO_WFE_MIN_T=1.0, SELECT_EVERY=3, SELECT_LOOKBACK=9)  # fmt: skip
    assert {k for k in R.cfg_fields(cfg) if R.cfg_fields(cfg)[k] != R.cfg_fields(alt)[k]} <= R.BACKTEST_ONLY
    assert R.signals_key("A", cfg, df) == R.signals_key("A", alt, df)
    pd.testing.assert_frame_equal(eng.run_wfo(df, cfg), eng.run_wfo(df, alt))
    alt2 = cfg.replace(POSITION_MODE="average", COST_MODEL="slippage")
    pd.testing.assert_frame_equal(eng.run_wfo(df, cfg), eng.run_wfo(df, alt2))


# ── Ledger ───────────────────────────────────────────────────────────────────


def _row(h, stage="A", status="ok", sharpe=0.5, n=1, **kw):
    return {"cell_hash": h, "stage": stage, "status": status, "sharpe": sharpe, "n_trials": n, **kw}


def test_trial_count_distinct_hashes_counted_statuses_and_stage_order():
    rows = [
        _row("a", "U9"), _row("b", "A"), _row("b", "B", cache_hit=True), _row("c", "A", status="error"),
        _row("d", "A", status="no_fit", sharpe=None), _row("e", "B", n=16), _row("f", "C"),
        {"event": "holdout_access", "batch_id": "x"},
    ]  # fmt: skip
    assert L.n_trials(rows, "U9") == 1
    assert L.n_trials(rows, "A") == 3  # a, b, d (no_fit counts, the error does not)
    assert L.n_trials(rows, "B") == 3 + 16  # b's cache hit is the same trial
    assert L.n_trials(rows, "E") == 20
    with pytest.raises(ValueError):
        L.n_trials(rows, "Q")


def test_var_trials_uses_pwfo_combo_sharpes_and_skips_selection_rows():
    rows = [_row("a", sharpe=1.0), _row("p", n=2, combos=[{"sharpe": 0.0}, {"sharpe": 2.0}, {"sharpe": None}]),
            _row("nested", n=0, sharpe=99.0)]  # fmt: skip
    srs = np.array([1.0, 0.0, 2.0]) / np.sqrt(252)
    assert L.var_trials(rows, "A") == pytest.approx(np.var(srs, ddof=1))


def test_truncated_last_line_is_skipped_and_isolated(env, capsys):
    led = env["ledger"]
    led.append([_row("a"), _row("b")])
    with open(led.path, "a") as f:
        f.write('{"cell_hash": "c", "sta')  # a killed write
    assert [r["cell_hash"] for r in led.rows()] == ["a", "b"]
    assert "malformed" in capsys.readouterr().err
    led.append(_row("d"))
    assert [r["cell_hash"] for r in led.rows()] == ["a", "b", "d"]
    assert led.path.read_text().count("\n") == 4


def test_ledger_rows_are_strict_json(env):
    env["ledger"].append(_row("a", sharpe=float("nan"), x=np.float64(np.inf), y=np.int64(3)))
    line = env["ledger"].path.read_text()
    assert "NaN" not in line and "Infinity" not in line
    r = env["ledger"].rows()[0]
    assert r["sharpe"] is None and r["x"] is None and r["y"] == 3


# ── Runner ───────────────────────────────────────────────────────────────────


def test_rerun_of_an_identical_spec_performs_zero_fits(env, fits):
    cells = expand(doc(grid={"symbols": ["AAA", "BBB", ["AAA", "BBB"]], "risk_profile": ["none"]}))
    src = FakeSource()
    rows = R.run(cells, source=src, **env)
    assert [r["status"] for r in rows] == ["ok"] * 3
    n_fits = len(fits)
    cell = cells[0]
    per_symbol = len(eng.wfo_folds(src.bars("AAA", "1Day", cell.spec["start"], cell.spec["end"], allow_holdout=False).index,
                                   cell.config()))  # fmt: skip
    assert per_symbol > 0 and n_fits == 2 * per_symbol  # the portfolio cell fitted nothing (signals cache)
    assert len({r["cell_hash"] for r in rows}) == 3
    # the portfolio cell reused both symbols' signals from the signals cache
    single = [len(json.loads(r["spec_json"])["symbols"]) == 1 for r in rows]
    assert sum(single) == 2
    again = R.run(expand(doc(grid={"symbols": ["AAA", "BBB", ["AAA", "BBB"]], "risk_profile": ["none"]})),
                  source=src, **env)  # fmt: skip
    assert again == [] and len(fits) == n_fits
    assert len(env["ledger"].rows()) == 3


def test_signals_cache_serves_risk_ablations_without_refitting(env, fits):
    R.run(expand(doc(risk_profile="none")), source=FakeSource(), **env)
    n = len(fits)
    rows = R.run(expand(doc(sizer="fixed", overrides={**SMALL, "POSITION_MODE": "average", "COST_MODEL": "slippage"})), source=FakeSource(),
                 **env)  # fmt: skip
    assert rows[0]["status"] == "ok" and len(fits) == n  # a new cell (different hash), same fits


def test_metrics_row_matches_the_backtest(env):
    rows = R.run(expand(doc()), source=FakeSource(), **env)
    r = rows[0]
    for k in ("cell_hash", "stage", "status", "git_sha", "spec_json", "started_at", "runtime_s", "n_oos_events",
              "n_trades", "meta_auc", "meta_logloss", "brier", "ret_ann", "vol_ann", "sharpe", "sortino", "calmar",
              "max_dd", "turnover", "sr_skew", "sr_kurt", "n_obs", "psr", "n_trials"):  # fmt: skip
        assert k in r, k
    daily = pd.read_csv(env["root"] / "cells" / r["cell_hash"] / "daily_returns.csv", index_col=0).iloc[:, 0]
    assert r["n_obs"] == len(daily)
    assert r["sharpe"] == pytest.approx(daily.mean() / daily.std(ddof=1) * np.sqrt(252))


def test_cross_stage_rerun_is_a_cache_hit_counted_once(env, fits):
    R.run(expand(doc(stage="A")), source=FakeSource(), **env)
    n = len(fits)
    rows = R.run(expand(doc(stage="B")), source=FakeSource(), **env)
    assert len(fits) == n and rows[0]["cache_hit"] and rows[0]["cache_from"] == "A"
    all_rows = env["ledger"].rows()
    assert L.n_trials(all_rows, "B") == 1 and rows[0]["sharpe"] == all_rows[0]["sharpe"]


def test_interrupted_run_resumes_without_duplicates(env, fits):
    spec = doc(grid={"symbols": ["AAA", "BBB", "CCC"]})

    def boom(row):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        R.run(expand(spec), source=FakeSource(), on_row=boom, **env)
    assert len(env["ledger"].rows()) == 1
    first = env["ledger"].rows()[0]["cell_hash"]
    before = len(fits)
    rows = R.run(expand(spec), source=FakeSource(), **env)
    assert first not in {r["cell_hash"] for r in rows} and len(rows) == 2
    hashes = [r["cell_hash"] for r in env["ledger"].rows()]
    assert len(hashes) == len(set(hashes)) == 3
    assert before > 0 and len(fits) == 3 * before  # only the two remaining cells were fitted


def test_failing_cell_is_an_error_row_with_a_traceback(env, monkeypatch):
    orig = R._run_wfo_cell

    def flaky(cell, *a, **k):
        if cell.symbols == ("BAD",):
            raise ZeroDivisionError("boom")
        return orig(cell, *a, **k)

    monkeypatch.setattr(R, "_run_wfo_cell", flaky)
    rows = R.run(expand(doc(grid={"symbols": ["AAA", "BAD", "MISSING"]})), source=FakeSource(fail={"MISSING"}), **env)
    st = {json.loads(r["spec_json"])["symbols"][0]: r for r in rows}
    assert st["AAA"]["status"] == "ok"
    for sym, msg in (("BAD", "ZeroDivisionError: boom"), ("MISSING", "no data for MISSING")):
        r = st[sym]
        assert r["status"] == "error" and msg in r["error"]
        tb = Path(r["error_path"]).read_text()
        assert msg.split(": ")[-1] in tb and tb.startswith("Traceback")
    assert L.n_trials(env["ledger"].rows(), "U10") == 1  # errors are not counted
    # errors are final unless retried
    monkeypatch.setattr(R, "_run_wfo_cell", orig)
    assert R.run(expand(doc(grid={"symbols": ["BAD"]})), source=FakeSource(), **env) == []
    rows = R.run(expand(doc(grid={"symbols": ["BAD"]})), source=FakeSource(), retry_errors=True, **env)
    assert rows[0]["status"] == "ok"


def test_no_fittable_window_is_a_counted_no_fit(env):
    rows = R.run(expand(doc(end="2014-06-01")), source=FakeSource(), **env)
    assert rows[0]["status"] == "no_fit" and rows[0]["n_trials"] == 1
    assert L.n_trials(env["ledger"].rows(), "U10") == 1


def test_parallel_equals_serial_and_fits_each_symbol_once(env, tmp_path):
    spec = doc(grid={"symbols": ["AAA", "BBB", ["AAA", "BBB"]], "risk_profile": ["none", "standard"]})
    serial = R.run(expand(spec), source=FakeSource(), **env)
    root2 = tmp_path / "e2"
    par = R.run(expand(spec), source=FakeSource(), ledger=L.Ledger(tmp_path / "l2.jsonl"), root=root2,
                feature_cache_dir=None, jobs=3)  # fmt: skip
    key = lambda rs: {r["cell_hash"]: (r["sharpe"], r["n_trades"], r["meta_auc"]) for r in rs}
    assert len(serial) == 6 and key(serial) == key(par)
    # phase 1 fitted each symbol's WFO once; every cell then read it from the signals cache
    assert len(list((root2 / "signals").glob("*.pkl"))) == 2
    for r in par:
        log = (root2 / "cells" / r["cell_hash"] / "log.txt").read_text()
        assert log.count("signals cache hit") == len(json.loads(r["spec_json"])["symbols"])


def test_pwfo_cell_counts_its_grid(env):
    pwfo = {"is_grid": [400, 600], "oos_grid": [50, 100]}
    over = {**SMALL, "PWFO_DEFAULT": [400, 50], "SELECT_LOOKBACK": 60, "SELECT_EVERY": 20}
    rows = R.run(expand(doc(pwfo=pwfo, overrides=over)), source=FakeSource(), **env)
    r = rows[0]
    assert r["status"] == "ok" and r["kind"] == "pwfo" and r["n_trials"] == 4 and len(r["combos"]) == 4
    assert r["pbo"] is not None and (env["root"] / "cells" / r["cell_hash"] / "summary_pwfo.csv").exists()
    assert L.n_trials(env["ledger"].rows(), "U10") == 4


# ── Holdout ──────────────────────────────────────────────────────────────────


def test_holdout_crossing_cells_are_refused_without_final(env):
    src = FakeSource()
    cells = expand(doc(grid={"symbols": ["AAA"], "end": ["2017-06-01", "2026-12-01"]}))
    with pytest.raises(HoldoutError, match="HOLDOUT_START"):
        R.run(cells, source=src, **env)
    assert src.loads == [] and not env["ledger"].path.exists()  # refused before any data or ledger access


def test_final_run_is_stage_e_once(env):
    src = FakeSource()
    hold = {"start": "2021-01-01", "end": "2026-12-01"}
    with pytest.raises(HoldoutError, match="stage E"):
        R.run(expand(doc(stage="D", **hold)), source=src, final=True, **env)
    rows = R.run(expand(doc(stage="E", **hold)), source=src, final=True, **env)
    assert rows[0]["status"] == "ok" and rows[0]["final"] is True
    assert all(load[-1] is True for load in src.loads)  # allow_holdout reaches the data layer
    events = L.holdout_events(env["ledger"].rows())
    assert len(events) == 1
    assert R.run(expand(doc(stage="E", **hold)), source=src, final=True, **env) == []  # same batch: resumes
    assert len(L.holdout_events(env["ledger"].rows())) == 1
    with pytest.raises(HoldoutError, match="already accessed"):
        R.run(expand(doc(stage="E", sizer="linear", **hold)), source=src, final=True, **env)


# ── Legacy import and report ─────────────────────────────────────────────────


def test_legacy_import_maps_u9_and_is_idempotent(env, tmp_path):
    p = tmp_path / "trials.jsonl"
    lines = [
        {"stage": "U9", "status": "ok", "combo": "IS252_OOS10", "sharpe": 0.1, "spec": {"symbols": ["SPY"]}},
        {"stage": "U9", "status": "no_fit", "combo": "IS63_OOS5", "spec": {"symbols": ["SPY"]}},
        {"stage": "U9", "status": "ok", "combo": "nested", "sharpe": 0.05, "dsr": 0.3, "spec": {"symbols": ["SPY"]}},
        {"stage": "U8", "status": "ok", "sharpe": 0.2, "meta_auc": {"SPY": 0.5, "QQQ": 0.6}, "spec": {"symbols": ["SPY"]}},
        {"stage": "U6", "status": "error", "error": "x", "spec": {"symbol": "SPY"}},
    ]  # fmt: skip
    p.write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert import_legacy([p], env["ledger"]) == 5
    assert import_legacy([p], env["ledger"]) == 0
    rows = env["ledger"].rows()
    assert [r["n_trials"] for r in rows] == [1, 1, 0, 1, 1]
    assert rows[3]["meta_auc"] == pytest.approx(0.55) and rows[2]["pwfo_dsr"] == 0.3
    assert L.n_trials(rows, "U9") == 3  # U8 row + two U9 combos (the nested selection adds none, the error none)
    assert np.isnan(row_dsr(rows[0], rows)[0])  # legacy rows get no DSR


def test_report_dsr_uses_the_ledger_trial_count(env):
    R.run(expand(doc(grid={"symbols": ["AAA", "BBB", "CCC"]})), source=FakeSource(), **env)
    rows = env["ledger"].rows()
    lb = leaderboard(rows)
    assert len(lb) == 3 and (lb["n_trials"] == 3).all()
    r = next(x for x in rows if x["cell_hash"] == lb.loc[0, "cell_hash"])
    v = L.var_trials(rows, "U10")
    expect = dsr(r["sharpe"] / np.sqrt(252), 3, v, r["n_obs"], r["sr_skew"], r["sr_kurt"])
    assert lb.loc[0, "dsr"] == pytest.approx(expect)
    paths = write_report(env["ledger"], env["root"], env["root"] / "report")
    md = paths["md"].read_text()
    assert "Stage funnel" in md and "PBO per stage" in md and "<table" in paths["html"].read_text()


def test_a_dead_worker_becomes_an_error_row_and_the_batch_completes(env):
    rows = R.run(expand(doc(grid={"symbols": ["AAA", "DIE", "BBB"]})), source=FakeSource(), jobs=2, **env)
    st = {json.loads(r["spec_json"])["symbols"][0]: r for r in rows}
    assert st["AAA"]["status"] == st["BBB"]["status"] == "ok"
    assert st["DIE"]["status"] == "error" and "WorkerDied" in st["DIE"]["error"]
    assert Path(st["DIE"]["error_path"]).exists()
    assert R.run(expand(doc(grid={"symbols": ["AAA", "DIE", "BBB"]})), source=FakeSource(), jobs=2, **env) == []


class LongCalendar(FakeSource):
    def sessions(self, end):
        return pd.bdate_range("1995-01-02", "2030-01-01")


def test_pwfo_hash_ignores_calendar_sessions_outside_the_data(env, fits):
    pwfo = {"is_grid": [400, 600], "oos_grid": [50, 100]}
    over = {**SMALL, "PWFO_DEFAULT": [400, 50], "SELECT_LOOKBACK": 60, "SELECT_EVERY": 20}
    spec = doc(pwfo=pwfo, overrides=over)
    first = R.run(expand(spec), source=FakeSource(), **env)
    n = len(fits)
    assert first[0]["status"] == "ok" and R.run(expand(spec), source=LongCalendar(), **env) == [] and len(fits) == n


def test_final_cells_must_reach_the_holdout_and_the_marker_spans_ledgers(env, tmp_path):
    src = FakeSource()
    with pytest.raises(HoldoutError, match="must run past"):
        R.run(expand(doc(stage="E")), source=src, final=True, **env)
    assert src.loads == [] and not env["holdout_marker"].exists()
    hold = {"start": "2021-01-01", "end": "2026-12-01"}
    R.run(expand(doc(stage="E", **hold)), source=src, final=True, **env)
    other = {**env, "ledger": L.Ledger(tmp_path / "elsewhere.jsonl")}
    with pytest.raises(HoldoutError, match="already accessed"):
        R.run(expand(doc(stage="E", sizer="linear", **hold)), source=src, final=True, **other)


def test_one_run_per_ledger(env):
    with env["ledger"].run_lock(), pytest.raises(RuntimeError, match="one run per ledger"):
        R.run(expand(doc()), source=FakeSource(), **env)


def test_ledger_rejects_rows_counting_cannot_read(env):
    for bad in ({"stage": "A", "status": "ok"}, {"cell_hash": "x", "stage": "U12", "status": "ok"},
                {"cell_hash": "x", "stage": "A", "status": "done"}):  # fmt: skip
        with pytest.raises(ValueError):
            env["ledger"].append(bad)
    assert env["ledger"].rows() == []


def test_legacy_import_skips_partial_lines_and_rejects_unknown_stages(env, tmp_path):
    p = tmp_path / "trials.jsonl"
    p.write_text(json.dumps({"stage": "U7", "status": "ok", "sharpe": 0.1}) + "\n" + '{"stage": "U7", "sta')
    assert import_legacy([p], env["ledger"]) == 1
    q = tmp_path / "bad.jsonl"
    q.write_text(json.dumps({"stage": "U12", "status": "ok"}) + "\n")
    with pytest.raises(ValueError, match="unknown stage"):
        import_legacy([q], env["ledger"])
    assert len(env["ledger"].rows()) == 1


def test_stage_pbo_counts_identical_streams_once(env):
    from experiments.report import stage_pbo

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=200)
    streams = {"a": rng.normal(0, 0.01, 200), "b": rng.normal(0, 0.01, 200)}
    streams["a2"] = streams["a"]
    rows = []
    for h, r in streams.items():
        d = env["root"] / "cells" / h
        d.mkdir(parents=True)
        pd.Series(r, index=days, name="ret").to_csv(d / "daily_returns.csv")
        rows.append(_row(h, "A"))
    res = stage_pbo(rows, env["root"]).iloc[0]
    assert res["n_cells"] == 3 and res["n_distinct"] == 2


# ── 5Min pilot rule (U11) ────────────────────────────────────────────────────


def test_weighted_auc_matches_sklearn_with_ties_and_integer_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    p = rng.integers(0, 20, 500) / 20  # many ties
    y = (rng.random(500) < 0.3 + 0.4 * p).astype(int)
    inv = np.unique(p, return_inverse=True)[1]
    assert weighted_auc(inv, y, np.ones(500)) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    w = rng.integers(0, 4, 500)
    rep = np.repeat(np.arange(500), w)  # weights = replicated events
    assert weighted_auc(inv, y, w.astype(float)) == pytest.approx(roc_auc_score(y[rep], p[rep]), abs=1e-12)


def test_session_bootstrap_bound_widens_with_within_session_dependence():
    rng = np.random.default_rng(4)
    n_s, k = 200, 30  # 200 sessions of 30 events; the outcome is a session-level coin flip
    sessions = np.repeat(np.arange(n_s), k)
    y = np.repeat(rng.integers(0, 2, n_s), k)
    p = rng.random(n_s * k)  # uninformative
    auc, ub = session_bootstrap_auc(y, p, sessions, n_boot=500)
    assert abs(auc - 0.5) < 0.03 and ub > auc
    assert session_bootstrap_auc(y, p, sessions, n_boot=500) == (auc, ub)  # seeded
    _, ub_iid = session_bootstrap_auc(y, p, np.arange(n_s * k), n_boot=500)  # events treated as independent
    assert ub - auc > ub_iid - auc  # clustering is not understated


def _a_row(h, sym, tf, primary, meta, auc, psr, sharpe):
    spec = {"symbols": [sym], "timeframe": tf, "primary": {"name": primary}, "model": {"meta": meta}}
    return {"cell_hash": h, "stage": "A", "status": "ok", "kind": "wfo", "label": h, "spec_json": json.dumps(spec),
            "meta_auc": auc, "psr": psr, "sharpe": sharpe, "n_obs": 2400, "sr_skew": 0.0, "sr_kurt": 3.0}  # fmt: skip


def test_stage_a_survivors_gates_ranks_caps_and_ignores_other_meta_models():
    rows = [
        _a_row("a1", "SPY", "5Min", "p1", "xgb", 0.53, 0.9, 1.5),  # best
        _a_row("a2", "SPY", "5Min", "p2", "xgb", 0.53, 0.9, 1.4),
        _a_row("a3", "SPY", "5Min", "p3", "xgb", 0.53, 0.9, 1.3),  # 3rd of its (symbol, timeframe): capped
        _a_row("b1", "QQQ", "1Day", "p1", "xgb", 0.515, 0.9, 1.2),  # AUC not > 0.515
        _a_row("b2", "QQQ", "1Day", "p2", "xgb", 0.52, 0.5, 1.2),  # PSR not > 0.5
        _a_row("c1", "TLT", "1Hour", "p1", "xgb", 0.52, 0.6, 0.2),
        _a_row("r1", "TLT", "1Hour", "p1", "rf_ldp", 0.60, 0.99, 3.0),  # recorded, never selects
        {**_a_row("e1", "IWM", "1Day", "p1", "xgb", 0.6, 0.9, 2.0), "status": "error"},
    ]
    t = stage_a_survivors(rows, k=3, per_pair=2).set_index("cell_hash")
    assert set(t.index) == {"a1", "a2", "a3", "b1", "b2", "c1"}
    assert t.loc[["a1", "a2", "a3", "c1"], "passed"].all() and not t.loc[["b1", "b2"], "passed"].any()
    assert list(t.index[t["survivor"]]) == ["a1", "a2", "c1"]  # DSR order, a3 capped, k = 3
    assert t["n_trials"].eq(7).all()  # the rf_ldp row still counts as a trial


def test_alpha_stats_hedges_beta_and_measures_top_day_share():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=500, freq="B", tz="America/New_York")
    bh = pd.Series(rng.normal(0.0005, 0.01, len(idx)), idx)
    alpha = pd.Series(rng.normal(0.0004, 0.005, len(idx)), idx)
    st = alpha_stats(0.5 * bh + alpha, bh)
    assert abs(st["beta"] - 0.5) < 0.05
    assert abs(st["alpha_sr"] - alpha.mean() / alpha.std() * np.sqrt(252)) < 0.15
    pure_beta = alpha_stats(0.5 * bh, bh)
    assert abs(pure_beta["alpha_sr"]) < 1e-6 or np.isnan(pure_beta["alpha_sr"])
    lucky = pd.Series(0.0, idx)
    lucky.iloc[:5] = 0.02
    lucky.iloc[5:] = -0.0001  # all of the P&L from 5 days
    assert alpha_stats(lucky, bh)["top_days_share"] > 1
    assert alpha_stats(-lucky.abs(), bh)["top_days_share"] == np.inf


def test_stage_b1_selection_picks_max_auc_model_and_gates_it(tmp_path):
    idx = pd.date_range("2022-01-03", periods=300, freq="B", tz="America/New_York")
    rng = np.random.default_rng(1)
    bh = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)
    good = pd.Series(rng.normal(0.001, 0.005, len(idx)), idx)

    def row(h, sym, meta, auc, psr, stage="B", **spec_extra):
        r = _a_row(h, sym, "1Day", "p1", meta, auc, psr, 1.0)
        spec = {**json.loads(r["spec_json"]), **spec_extra}
        (tmp_path / "cells" / h).mkdir(parents=True)
        (good if psr > 0.5 else -good).rename("ret").to_csv(tmp_path / "cells" / h / "daily_returns.csv")
        return {**r, "stage": stage, "spec_json": json.dumps(spec)}

    rows = [
        row("a_spy", "SPY", "xgb", 0.53, 0.9, stage="A"),
        row("b_spy_l", "SPY", "logit_l2", 0.56, 0.4),  # best AUC but PSR fails: SPY fails (no fallback)
        row("b_spy_c", "SPY", "catboost", 0.54, 0.95),
        row("a_qqq", "QQQ", "xgb", 0.53, 0.9, stage="A"),
        row("b_qqq_r", "QQQ", "rf_ldp_fast", 0.55, 0.8),  # QQQ: rf_ldp_fast
        row("b_qqq_x", "QQQ", "rf_ldp_fast", 0.60, 0.8, feature_groups="other"),  # a B2-style row: not a candidate
        row("b_qqq_e", "QQQ", "extra_trees", 0.70, 0.9),  # not a B1 model
        row("a_tlt", "TLT", "xgb", 0.51, 0.9, stage="A"),  # not a Stage A survivor
    ]
    t = stage_b1_selection(rows, tmp_path, bh_returns=lambda spec: bh).set_index("symbol")
    assert set(t.index) == {"SPY", "QQQ"}
    assert t.loc["SPY", "model"] == "logit_l2" and not t.loc["SPY", "passed"]
    assert t.loc["QQQ", "model"] == "rf_ldp_fast" and t.loc["QQQ", "cell_hash"] == "b_qqq_r" and t.loc["QQQ", "passed"]
    assert t.loc["SPY", "n_models"] == 3 and t.loc["QQQ", "n_models"] == 2
    assert t.loc["QQQ", "auc_xgb"] == 0.53 and np.isnan(t.loc["QQQ", "auc_catboost"])


def test_stage_b2_selection_picks_max_auc_arm_of_each_b1_passer(tmp_path):
    from experiments.report import B2_FULL
    from utils.config import DEFAULT_FEATURE_GROUPS

    idx = pd.date_range("2022-01-03", periods=300, freq="B", tz="America/New_York")
    rng = np.random.default_rng(2)
    bh = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)
    good = pd.Series(rng.normal(0.001, 0.005, len(idx)), idx)
    full, dflt = list(B2_FULL), list(DEFAULT_FEATURE_GROUPS)

    def row(h, sym, meta, auc, psr, stage="B", groups=dflt, sel=None, test=10):
        r = _a_row(h, sym, "1Day", "p1", meta, auc, psr, 1.0)
        ov = {"TEST": test} | ({"FEATURE_SELECTION": sel} if sel else {})
        spec = {**json.loads(r["spec_json"]), "feature_groups": groups, "overrides": ov}
        (tmp_path / "cells" / h).mkdir(parents=True)
        (good if psr > 0.5 else -good).rename("ret").to_csv(tmp_path / "cells" / h / "daily_returns.csv")
        return {**r, "stage": stage, "spec_json": json.dumps(spec)}

    rows = [
        row("a_qqq", "QQQ", "xgb", 0.53, 0.9, stage="A"),
        row("b_qqq", "QQQ", "rf_ldp_fast", 0.55, 0.8),  # B1 pick
        row("q_cmda", "QQQ", "rf_ldp_fast", 0.56, 0.9, sel="cmda"),
        row("q_full", "QQQ", "rf_ldp_fast", 0.54, 0.9, groups=full),
        row("q_fc", "QQQ", "rf_ldp_fast", 0.57, 0.4, groups=full, sel="cmda"),  # max AUC, fails PSR: no fallback
        row("q_xgb_full", "QQQ", "xgb", 0.70, 0.9, groups=full),  # another model: not a candidate
        row("q_t20", "QQQ", "rf_ldp_fast", 0.70, 0.9, sel="cmda", test=20),  # another override: not a candidate
        row("a_spy", "SPY", "xgb", 0.53, 0.9, stage="A"),  # B1 pick (only model)
        row("s_cmda", "SPY", "xgb", 0.52, 0.9, sel="cmda"),
        row("a_tlt", "TLT", "xgb", 0.51, 0.9, stage="A"),  # not a survivor
    ]
    t = stage_b2_selection(rows, tmp_path, bh_returns=lambda spec: bh).set_index("symbol")
    assert set(t.index) == {"QQQ", "SPY"}
    assert t.loc["QQQ", "arm"] == "full_cmda" and t.loc["QQQ", "cell_hash"] == "q_fc" and not t.loc["QQQ", "passed"]
    assert t.loc["QQQ", "n_arms"] == 4 and t.loc["QQQ", "auc_default"] == 0.55 and t.loc["QQQ", "auc_cmda"] == 0.56
    assert t.loc["SPY", "arm"] == "default" and t.loc["SPY", "cell_hash"] == "a_spy" and t.loc["SPY", "passed"]
    assert t.loc["SPY", "n_arms"] == 2 and np.isnan(t.loc["SPY", "auc_full"])


def test_stage_b3_selection_keeps_fixed_unless_a_passing_sizer_has_higher_psr(tmp_path):
    from utils.config import DEFAULT_FEATURE_GROUPS

    idx = pd.date_range("2022-01-03", periods=300, freq="B", tz="America/New_York")
    rng = np.random.default_rng(3)
    bh = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)
    good = pd.Series(rng.normal(0.001, 0.005, len(idx)), idx)
    lucky = pd.Series(-0.0001, idx)
    lucky.iloc[:5] = 0.05  # positive total, all of it from 5 days: fails the top-days gate

    def row(h, sym, meta, auc, psr, stage="B", sizer="fixed", ret=None):
        r = _a_row(h, sym, "1Day", "p1", meta, auc, psr, 1.0)
        spec = {**json.loads(r["spec_json"]), "feature_groups": list(DEFAULT_FEATURE_GROUPS),
                "overrides": {"TEST": 10}, "sizer": sizer}  # fmt: skip
        (tmp_path / "cells" / h).mkdir(parents=True)
        (good if ret is None else ret).rename("ret").to_csv(tmp_path / "cells" / h / "daily_returns.csv")
        return {**r, "stage": stage, "spec_json": json.dumps(spec)}

    rows = [
        row("a_qqq", "QQQ", "xgb", 0.53, 0.8, stage="A"),  # A, B1 and B2 pick (only row): fixed
        row("q_lin", "QQQ", "xgb", 0.53, 0.85, sizer="linear"),
        row("q_ldp", "QQQ", "xgb", 0.53, 0.95, sizer="ldp_sigmoid", ret=lucky),  # best PSR, fails a gate
        row("q_kelly", "QQQ", "xgb", 0.53, 0.99, sizer="kelly_capped"),  # not a B3 sizer
        row("a_spy", "SPY", "xgb", 0.53, 0.9, stage="A"),
        row("s_ecdf", "SPY", "xgb", 0.53, 0.7, sizer="ecdf"),  # passes but lower PSR: fixed stays
    ]
    t = stage_b3_selection(rows, tmp_path, bh_returns=lambda spec: bh).set_index("symbol")
    assert t.loc["QQQ", "sizer"] == "linear" and t.loc["QQQ", "cell_hash"] == "q_lin" and t.loc["QQQ", "passed"]
    assert t.loc["QQQ", "n_sizers"] == 3 and t.loc["QQQ", "psr_ldp_sigmoid"] == 0.95
    assert np.isnan(t.loc["QQQ", "psr_ecdf"])
    assert t.loc["SPY", "sizer"] == "fixed" and t.loc["SPY", "cell_hash"] == "a_spy" and t.loc["SPY", "passed"]
    assert t.loc["SPY", "n_sizers"] == 2 and t.loc["SPY", "psr_fixed"] == 0.9


def test_stage_c_finalists_takes_top_psr_passers_per_symbol():
    b3 = pd.DataFrame({
        "symbol": ["AMZN", "AMZN", "AMZN", "QQQ", "QQQ", "SPY"],
        "psr": [0.90, 0.95, 0.99, 0.80, 0.80, 0.70],
        "passed": [True, True, False, True, True, True],
        "label": ["a1", "a2", "a3", "q2", "q1", "s1"],
        "cell_hash": ["h1", "h2", "h3", "h4", "h5", "h6"],
    })  # fmt: skip
    t = stage_c_finalists(b3).set_index("cell_hash")
    assert set(t.index[t["finalist"]]) == {"h1", "h2", "h4", "h5", "h6"}  # h3 fails its gates despite the best PSR
    assert set(stage_c_finalists(b3, per_symbol=1).query("finalist")["label"]) == {"a2", "q1", "s1"}  # tie → label


def test_stage_c_cell_is_the_b3_spec_with_its_timeframe_grid():
    b3 = normalize({"symbols": "SPY", "timeframe": "1Hour", "start": "2016-01-01", "end": "2025-10-01",
                    "overrides": {"TEST": 10}, "sizer": "linear"})  # fmt: skip
    c = Cell(normalize(stage_c_cell(b3)), "C")
    cfg = c.config()
    assert (cfg.PWFO_IS_GRID, cfg.PWFO_OOS_GRID, cfg.PWFO_DEFAULT) == ((252, 378, 504, 756), (5, 10, 21, 63), (252, 10))
    assert not cfg.PWFO_EXPANDING and cfg.TEST == 10 and cfg.SIZER == "linear"
    assert {**c.spec, "pwfo": None, "overrides": {"TEST": 10}} == b3  # nothing else changes
    d = Cell(normalize(stage_c_cell({**b3, "timeframe": "1Day"})), "C").config()
    assert (d.PWFO_IS_GRID, d.PWFO_DEFAULT) == ((1260, 1512), (1512, 10))


def test_stage_c_selection_gates_the_nested_stream_and_pbo(tmp_path):
    idx = pd.date_range("2022-01-03", periods=300, freq="B", tz="America/New_York")
    rng = np.random.default_rng(4)
    bh = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)
    good = pd.Series(rng.normal(0.001, 0.005, len(idx)), idx)
    by_sym = {}

    def b3_row(h, sym):
        spec = normalize({"symbols": sym, "timeframe": "1Hour", "start": "2016-01-01", "end": "2025-10-01",
                          "overrides": {"TEST": 10}})  # fmt: skip
        by_sym[sym] = spec
        return {"cell_hash": h, "stage": "B", "status": "ok", "label": h, "spec_json": json.dumps(spec)}

    def c_row(h, sym, psr, pbo, status="ok", spec=None, ret=good):
        spec = spec or normalize(stage_c_cell(by_sym[sym]))
        (tmp_path / "cells" / h).mkdir(parents=True)
        ret.rename("ret").to_csv(tmp_path / "cells" / h / "daily_returns.csv")
        return {"cell_hash": h, "stage": "C", "status": status, "label": h, "spec_json": json.dumps(spec),
                "psr": psr, "sharpe": 1.0, "pbo": pbo, "n_trials": 16, "picks": {"IS504_OOS10": 3, "IS252_OOS5": 1}}  # fmt: skip

    rows = [b3_row(f"b_{s}", s) for s in ("SPY", "QQQ", "IWM", "TLT", "GLD")]
    rows += [
        c_row("c_spy", "SPY", 0.9, 0.3),  # passes
        c_row("c_qqq", "QQQ", 0.9, 0.6),  # PBO fails
        c_row("c_iwm", "IWM", 0.4, 0.1, ret=-good),  # PSR fails
        c_row("c_tlt_old", "TLT", 0.99, 0.1, spec=normalize({**by_sym["TLT"], "pwfo": {}})),  # U9 grid: not its row
        c_row("c_gld", "GLD", None, None, status="no_fit"),
    ]
    fin = pd.DataFrame({
        "symbol": ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE"], "timeframe": "1Hour", "primary": "p", "model": "m",
        "arm": "default", "sizer": "fixed", "psr": [0.9, 0.95, 0.8, 0.7, 0.6, 0.99],
        "cell_hash": ["b_SPY", "b_QQQ", "b_IWM", "b_TLT", "b_GLD", "b_XLE"],
        "finalist": [True, True, True, True, True, False],
    })  # fmt: skip
    t = stage_c_selection(rows, fin, tmp_path, bh_returns=lambda spec: bh).set_index("symbol")
    assert set(t.index) == {"SPY", "QQQ", "IWM", "TLT", "GLD"}  # XLE not a finalist
    assert list(t.index[t["passed"]]) == ["SPY"] and t.loc["SPY", "top_pick"] == "IS504_OOS10"
    assert t.loc["QQQ", "pbo"] == 0.6 and t.loc["IWM", "alpha_sr"] < 0
    assert not t.loc["TLT", "has_row"] and t.loc["GLD", "has_row"] and t.loc["GLD", "status"] == "no_fit"


class ExtendingSource(FakeSource):
    """One fixed synthetic history, sliced to [start, end): a later `end` only appends bars."""

    def bars(self, symbol, timeframe, start, end, *, allow_holdout):
        self.loads.append((symbol, timeframe, start, end, allow_holdout))
        idx = pd.bdate_range("2020-01-01", "2027-01-01", inclusive="left", tz="America/New_York")
        df = synthetic_daily(len(idx), seed=sum(map(ord, symbol)))
        df.index = idx
        lo, hi = pd.Timestamp(start, tz="America/New_York"), pd.Timestamp(end, tz="America/New_York")
        return df[(df.index >= lo) & (df.index < hi)]


def test_final_pwfo_cell_scores_only_the_holdout_and_extends_the_dev_stream(env, tmp_path):
    from data.bars import HOLDOUT_START

    pwfo = {"is_grid": [400, 600], "oos_grid": [50, 100]}
    over = {**SMALL, "PWFO_DEFAULT": [400, 50], "SELECT_LOOKBACK": 60, "SELECT_EVERY": 20}
    dev_spec = doc(stage="C", start="2020-01-01", end="2026-10-01", pwfo=pwfo, overrides=over)
    (dev,) = R.run(expand(dev_spec), source=ExtendingSource(), **env)
    e_spec = doc(stage="E", start="2020-01-01", end="2026-12-01", pwfo=pwfo,
                 overrides={**over, "PWFO_PARTIAL_LAST": True})  # fmt: skip
    (e,) = R.run(expand(e_spec), source=ExtendingSource(), final=True, **env)
    assert dev["status"] == e["status"] == "ok" and e["final"] is True

    def read(h, name):
        s = pd.read_csv(env["root"] / "cells" / h / name, index_col=0)["ret"]
        s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").normalize()
        return s

    ho, full, c = (
        read(e["cell_hash"], "daily_returns.csv"),
        read(e["cell_hash"], "daily_returns_full.csv"),
        read(dev["cell_hash"], "daily_returns.csv"),
    )
    start = HOLDOUT_START.tz_localize("America/New_York")
    assert ho.index.min() >= start and e["n_obs"] == len(ho) and e["holdout_start"] == "2026-10-01"
    assert ho.index.max() == pd.Timestamp("2026-11-30", tz="America/New_York")  # partial last window: to the data end
    assert e["n_pre_holdout_days"] == int((full.index < start).sum()) and len(full) == len(ho) + e["n_pre_holdout_days"]
    common = c.index[:-5]  # the dev cell's last days close its positions at its data end
    assert np.allclose(full.loc[common], c.loc[common], atol=1e-12, rtol=0)  # E continues C's stream
    assert "daily_returns_full.csv" not in {p.name for p in (env["root"] / "cells" / dev["cell_hash"]).iterdir()}


def test_holm_is_step_down_and_monotone():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
    assert np.allclose(holm([0.3, 0.6]), [0.6, 0.6])
    assert np.allclose(bh_fdr([0.01, 0.04, 0.03, 0.5]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])  # step-up
    assert np.allclose(bh_fdr([0.3, 0.6]), [0.6, 0.6])


def test_stage_e_selection_verdicts_and_integrity(tmp_path):
    idx = pd.date_range("2024-01-02", periods=700, freq="B", tz="America/New_York")
    ho = idx[idx >= pd.Timestamp("2025-10-01", tz="America/New_York")]
    rng = np.random.default_rng(5)
    bh = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)

    def stream(mu, n, seed):
        return pd.Series(np.random.default_rng(seed).normal(mu, 0.01, n))

    def write(h, name, s):
        (tmp_path / "cells" / h).mkdir(parents=True, exist_ok=True)
        s.rename("ret").to_csv(tmp_path / "cells" / h / name)

    rows, c_rows = [], []
    for j, sr in enumerate(np.linspace(-0.5, 0.5, 40)):  # earlier trials: they set the DSR's V[SR] and part of N
        rows.append({"cell_hash": f"b{j}", "stage": "B", "status": "ok", "label": f"b{j}", "spec_json": "{}",
                     "sharpe": sr, "n_obs": 1000, "sr_skew": 0.0, "sr_kurt": 3.0, "n_trials": 1})  # fmt: skip
    # (holdout stream drift, row Sharpe, row PSR(0), expected verdict)
    cases = {"SPY": (0.004, 8.0, 0.99999, "edge"), "QQQ": (0.0015, 2.0, 0.97, "not_demonstrated"),
             "XLK": (-0.002, 4.0, 0.9999, "not_demonstrated"),  # significant, but its alpha Sharpe is negative
             "IWM": (-0.001, -3.0, 0.001, "negative"), "TLT": (None, None, None, None)}  # fmt: skip
    for k, (sym, (mu, sr, p0, _)) in enumerate(cases.items()):
        spec = normalize({"symbols": sym, "timeframe": "1Hour", "start": "2016-01-01", "end": "2025-10-01",
                          "overrides": {"TEST": 10}, "pwfo": {}})  # fmt: skip
        dev = pd.Series(stream(0.0005, len(idx) - len(ho), k).to_numpy(), idx[: len(idx) - len(ho)])
        c = {"cell_hash": f"c_{sym}", "stage": "C", "status": "ok", "label": sym, "spec_json": json.dumps(spec),
             "sharpe": 0.5, "n_obs": len(dev), "sr_skew": 0.0, "sr_kurt": 3.0, "n_trials": 16}  # fmt: skip
        write(c["cell_hash"], "daily_returns.csv", dev)
        rows.append(c)
        c_rows.append(c)
        if mu is None:
            continue  # TLT: E not run
        h = pd.Series(stream(mu, len(ho), 100 + k).to_numpy(), ho)
        e = {"cell_hash": f"e_{sym}", "stage": "E", "status": "ok", "label": sym, "n_trials": 16,
             "spec_json": json.dumps(normalize(stage_e_cell(spec))), "sharpe": sr, "n_obs": len(h), "sr_skew": 0.0,
             "sr_kurt": 3.0, "psr": p0, "n_pre_holdout_days": len(dev)}  # fmt: skip
        write(e["cell_hash"], "daily_returns.csv", h)
        full = pd.concat([dev + (1e-3 if sym == "IWM" else 0.0), h])  # IWM's E does not continue its C stream
        write(e["cell_hash"], "daily_returns_full.csv", full)
        rows.append(e)
    t = stage_e_selection(rows, c_rows, tmp_path, bh_returns=lambda spec: bh).set_index("symbol")
    assert {s: t.loc[s, "verdict"] for s in ("SPY", "QQQ", "XLK", "IWM")} == {s: c[3] for s, c in cases.items() if c[3]}
    assert not t.loc["TLT", "has_row"] and t.loc["SPY", "n_trials_total"] == 40 + 5 * 16 + 4 * 16
    assert t.loc["SPY", "c_max_diff"] == 0 and t.loc["IWM", "c_max_diff"] == pytest.approx(1e-3)
    # Holm over 4 p-values: QQQ's 0.03 → 0.06 (no edge); Benjamini–Hochberg q 0.04 is reported only
    assert t.loc["QQQ", "p_holm"] == pytest.approx(0.06) and t.loc["QQQ", "p_bh"] == pytest.approx(0.04)
    assert t.loc["XLK", "p_holm"] < 0.05 and t.loc["XLK", "alpha_sr"] < 0
    assert t.loc["IWM", "ci_hi"] < 0 < t.loc["QQQ", "ci_hi"] and t.loc["SPY", "ci_lo"] > 0
    ho_bh = bh.reindex(ho)
    assert t.loc["SPY", "bh_sharpe"] == pytest.approx(ho_bh.mean() / ho_bh.std() * np.sqrt(252))
    assert t.loc["SPY", "dsr"] > 0.95  # reported, not deciding
