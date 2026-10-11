"""
U28 (PLAN3 §5 U28, SPEC §27): the vix_term group (point in time), the vix_carry region primary (cells, hard rule,
pre-break scale, one decision per session at the mark), the drawdown kill (a latch), the engine region of a family
(each cell its own engine run; coherence on the marginal Δ Sharpe), the overlay-alone and episode reports, and the
registered G6 spec.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from families import test as T
from families.spec import HEADLINE, load_family, parse, region_kind
from tests.test_risk import daily_bars, sig_at
from tests.test_u14 import intraday
from tests.test_u17 import OvernightSource
from tests.test_u24 import _region_doc
from tests.test_u25 import _sharpe_power
from utils.config import RunConfig

ROOT = Path(__file__).resolve().parent.parent
NY = "America/New_York"


def _series(days: pd.DatetimeIndex, values) -> pd.DataFrame:
    """A cboe-like exo frame: observation dates, `value`, available at 16:20 NY of the date."""
    days = pd.DatetimeIndex(days).normalize()
    avail = (days.tz_localize(NY) + pd.Timedelta(hours=16, minutes=20)).tz_convert("UTC")
    return pd.DataFrame({"value": np.asarray(values, dtype=float), "available_at": avail}, index=days)


# ── vix_term: point in time ──────────────────────────────────────────────────


def test_vix_term_reads_the_previous_sessions_settlement_at_every_bar():
    from features.exo_align import Exo
    from features.state import vix_term_frame

    df = intraday(6, seed=1)
    days = pd.DatetimeIndex(df.index.tz_localize(None).normalize().unique())
    vix, r1, r2 = 10.0 + np.arange(6), 1.0 + 0.01 * np.arange(6), 1.1 + 0.01 * np.arange(6)
    exo = Exo({"cboe/VIX": _series(days, vix), "cboe/VX1_VIX": _series(days, r1), "cboe/VX2_VX1": _series(days, r2)})
    cols = vix_term_frame(df, exo)
    k = np.searchsorted(days, df.index.tz_localize(None).normalize())  # each bar's session
    for name, v in (("vix", vix), ("vx1_vix", r1), ("vx2_vx1", r2)):
        want = np.where(k >= 1, v[np.maximum(k - 1, 0)], np.nan)  # every bar of day d (09:30 … 15:55) sees d − 1
        np.testing.assert_array_equal(cols[name], want)


def test_vix_term_is_a_registered_static_group():
    from features.registry import REGISTRY

    g = REGISTRY["vix_term"]
    assert not g.per_fold and g.exo == ("cboe/VIX", "cboe/VX1_VIX", "cboe/VX2_VX1")


# ── vix_carry ────────────────────────────────────────────────────────────────


def _prim(**params):
    from primaries.vix_carry import VixCarry

    return VixCarry(**params)


def _X(rows, start="2018-02-26"):
    idx = pd.date_range(pd.Timestamp(start) + pd.Timedelta("09:55:00"), periods=len(rows), freq="1D", tz=NY)
    vix, r1, r2 = np.array(rows, dtype=float).T
    return pd.DataFrame({"vix_term__vix": vix, "vix_term__vx1_vix": r1, "vix_term__vx2_vx1": r2}, index=idx)


def test_cells_target_the_hard_rule_and_the_pre_break_scale():
    p = _prim()
    assert len(p.cell_names()) == 15 and p.cell_names()[:3] == ["t1_vix20", "t1_vix25", "t1_vix30"]
    X = _X([(22.0, 1.06, 1.2), (22.0, 1.06, 1.2), (15.0, 1.2, 1.0), (15.0, 0.99, 1.3), (np.nan, 1.2, 1.1)])
    side, target = p.rule(None, X, None)
    # day 0 (2018-02-26, before the break): θ ∈ {1, 1.025, 1.05} × caps {25, 30} = 6 of 15 cells, × 0.5
    assert target[0] == pytest.approx(0.5 * 6 / 15) and side[0] == 1.0
    assert target[1] == pytest.approx(0.5 * 6 / 15)  # 2018-02-27: still before 2018-02-28
    assert target[2] == pytest.approx(1.0)  # 2018-02-28: every cell, unscaled
    assert target[3] == 0.0 and side[3] == 0.0  # backwardation: flat whatever the cells say
    assert target[4] == 0.0  # a NaN VIX compares False: flat
    # the VX2 / VX1 signal keeps the VX1 / VIX hard rule; a null cap is no cap
    q = _prim(signal="vx2_vx1", theta_list=[1.05], vix_list=[None], scale_before=None)
    assert list(q.rule(None, X, None)[1]) == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert q.cell_names() == ["t1.05_vixinf"]
    # each single-cell primary reproduces its column of the region
    cells = p.cells(None, X, None)
    for name, params in p.cell_params():
        one = _prim(**params)
        np.testing.assert_array_equal(one.cells(None, X, None)[name].to_numpy(), cells[name].to_numpy())
        assert one.cell_names() == [name]


def test_vix_carry_is_validated_and_configured():
    from primaries.mechanism import primary_config

    for bad in ({"signal": "vix"}, {"theta_list": []}, {"theta_list": [1.0, 1.0]}, {"vix_list": [0]},
                {"decision": "09:30"}, {"decision": "10:02"}, {"backwardation": -1},
                {"scale_before": {"date": "2018-02-28"}}, {"scale_before": {"date": "x", "scale": 0.5}},
                {"scale_before": {"date": "2018-02-28", "scale": 1.5}}):  # fmt: skip
        with pytest.raises(ValueError):
            _prim(**bad)
    cfg = primary_config("vix_carry", "5Min", {}, META_MODEL="none")
    assert cfg.EVENT_PARAMS == {"entry_times": ["10:00"], "every": "session"}
    assert cfg.EXIT_PARAMS == {"exit_time": "10:00", "exit_session": 1}
    assert cfg.SIZE_STEP == 0 and cfg.SIZER == "rule_size" and "vix_term" in cfg.FEATURE_GROUPS
    with pytest.raises(ValueError, match="runs on"):
        primary_config("vix_carry", "1Day", {})
    assert region_kind(type(_prim())) == "engine"
    from primaries.region import RegionTrend

    assert region_kind(RegionTrend) == "replica" and region_kind(object) is None


# ── The drawdown kill ────────────────────────────────────────────────────────


def test_the_drawdown_kill_flattens_and_never_rearms():
    from risk.portfolio import simulate_portfolio
    from risk.profiles import PROFILES, RiskProfile

    assert PROFILES["vrp"] == RiskProfile("vrp", daily_loss=0.03, kill_dd=0.15)
    assert PROFILES["vrp_nokill"].kill_dd is None and PROFILES["vrp_nokill"].daily_loss == 0.03
    with pytest.raises(ValueError, match="kill_dd"):
        RiskProfile("x", kill_dd=1.0)
    # long from open[1]; the close of bar 3 is 16 % below the high-water mark → killed; the price recovers, every later
    # event is refused, and the kill stays on to the end
    df = daily_bars([(100, 100), (100, 100), (100, 92), (92, 84), (84, 90), (90, 100), (100, 110), (110, 120)]
                    + [(120, 120)] * 4)  # fmt: skip
    cfg = RunConfig.for_timeframe("1Day", VERTICAL_BARS=2, SLIPPAGE_PCT=0.0)
    sig = sig_at(df, [0, 2, 4, 5, 6], [1, 1, 1, 1, 1], [1.0] * 5)
    eq, tr, log = simulate_portfolio({"X": df}, {"X": sig}, cfg, RiskProfile("k", kill_dd=0.15))
    assert eq.attrs["killed_at"] == df.index[3]
    assert list(log["kill"].to_numpy()) == [0, 0, 0] + [1] * (len(df) - 3)
    last = tr.iloc[-1]
    assert last["exit_reason"] == "kill" and last["exit_b"] == 4 and last["exit_px"] == 84
    assert tr["entry_b"].max() < 4  # nothing after the kill
    assert (eq.iloc[4:] == eq.iloc[4]).all()  # flat to the end
    # a 14 % drawdown does not fire it; the same bets then run
    df2 = daily_bars([(100, 100), (100, 100), (100, 92), (92, 86), (86, 90), (90, 100), (100, 110), (110, 120)]
                     + [(120, 120)] * 4)  # fmt: skip
    eq2, tr2, _ = simulate_portfolio({"X": df2}, {"X": sig}, cfg, RiskProfile("k", kill_dd=0.15))
    assert eq2.attrs["killed_at"] is None and tr2["entry_b"].max() > 4


def test_the_replica_refuses_a_profile_with_a_kill():
    from risk.profiles import PROFILES
    from wfo.position_backtest import _profile_loss

    with pytest.raises(ValueError, match="daily_loss only"):
        _profile_loss(PROFILES["vrp"])
    assert _profile_loss(PROFILES["vrp_nokill"]) == 0.03


# ── The family layer end to end (synthetic bars and VIX state) ───────────────


class VixSource(OvernightSource):
    """OvernightSource with synthetic cboe series: VIX and VX1 / VIX random walks, VX2 / VX1 around 1.05."""

    def __init__(self):
        super().__init__(0.0)
        rng = np.random.default_rng(28)
        days = pd.bdate_range(self.START, self.END, inclusive="left")
        self._exo = {"VIX": _series(days, np.clip(18 + np.cumsum(rng.normal(0, 1.0, len(days))), 11, 40)),
                     "VX1_VIX": _series(days, np.clip(1.04 + 0.03 * np.sin(np.arange(len(days)) / 6.0)
                                                      + rng.normal(0, 0.01, len(days)), 0.9, 1.2)),
                     "VX2_VX1": _series(days, 1.05 + rng.normal(0, 0.02, len(days)))}  # fmt: skip

    def exo(self, source, name, start, end, *, allow_holdout):
        if source != "cboe" or name not in self._exo:
            raise KeyError(f"no {source}/{name} in the synthetic source")
        f = self._exo[name]
        return f[(f.index >= pd.Timestamp(start)) & (f.index < pd.Timestamp(end))]


def _g6_doc(**extra):
    doc = {
        "id": "G6", "mechanism": "test", "instruments": ["SVXY"], "timeframe": "5Min",
        "window": {"start": "2022-01-03", "end": "2022-04-01"}, "weighting": "equal",
        "headline": {"primary": {"name": "vix_carry", "params": {"theta_list": [1.0, 1.05], "vix_list": [20, None],
                                                                 "scale_before": None}},
                     "cost_model": "quotes", "risk_profile": "vrp", "overrides": {"INIT_CASH": 30000}},
        "core": {"family": "C1", "variant": "headline", "k": 0.5},
        "variants": [{"label": "no_kill", "risk_profile": "vrp_nokill"}, {"label": "k1", "core.k": 1.0}],
        "state_splits": ["vix_tercile", "year"], "response": ["sharpe", "max_dd"],
        "episodes": [{"label": "feb", "start": "2022-02-01", "end": "2022-03-01"}],
        "benchmark": "constant_mix_ew", "floors": {"min_net_ret": 0.0, "min_edge_to_cost": 0.1, "max_dd_vs_core": 1.0},
        "test": {"kind": "marginal", "sided": "one", "at_cost": "registered", "alpha": 0.05, "block_days": 5,
                 "n_boot": 199, "coherence_share": 0.667, "coherence": "cells", "seed": 0},
        "power": _sharpe_power(), "TRIAL_BUDGET": 8,
    }  # fmt: skip
    doc.update(extra)
    return doc


def test_engine_region_specs_are_validated():
    fam = parse(_g6_doc())
    assert fam.episodes == [{"label": "feb", "start": "2022-02-01", "end": "2022-03-01"}]
    for bad in ([{"label": "x", "start": "2021-12-01", "end": "2022-02-01"}],  # outside the window
                [{"label": "x", "start": "2022-02-01", "end": "2022-02-01"}],  # empty
                [{"label": "x", "start": "2022-02-01"}],
                [{"label": "x", "start": "2022-02-01", "end": "2022-02-10"}] * 2):  # fmt: skip
        with pytest.raises(ValueError, match="episode"):
            parse(_g6_doc(episodes=bad))
    # cells coherence needs a region primary (replica or engine)
    doc = _region_doc()
    doc["headline"]["primary"] = {"name": "overnight", "params": {}}
    doc["test"] = {**doc["test"], "coherence": "cells"}
    with pytest.raises(ValueError, match="region primary"):
        parse(doc)


def _run_g6(tmp_path, doc=None):
    from experiments import ledger as L
    from families.run import run_family

    docs = {"C1": _region_doc(id="C1", variants=[]), "G6": doc or _g6_doc()}
    for k, d in docs.items():
        (tmp_path / f"{k}.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G6]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    ledger = L.Ledger(tmp_path / "ledger.jsonl")
    res = run_family(tmp_path / "G6.yaml", ledger=ledger, root=tmp_path / "root", out_dir=tmp_path / "out",
                     source=VixSource(), repo=tmp_path, check_registration=False, program=prog, quasi=False)  # fmt: skip
    return res, ledger


def test_an_engine_region_runs_every_cell_and_reads_coherence_on_the_marginal_statistic(tmp_path):
    from families.run import _by_symbol, _daily_costs, engine_cells

    res, ledger = _run_g6(tmp_path)
    fam = load_family(tmp_path / "G6.yaml")
    reg = res["region"]
    assert reg["method"] == "engine" and reg["statistic"] == "delta_sharpe" and reg["parity_ok"] is True
    assert reg["names"] == ["t1_vix20", "t1_vixinf", "t1.05_vix20", "t1.05_vixinf"]
    assert set(reg["status"].values()) == {"ok"} and reg["costs"] == ["registered", "0.3", "1.0", "2.3"]
    # the cells ran in the batch as rule cells, never as family-variant rows (no trials)
    rows = [r for r in ledger.rows() if r.get("kind") == "family_variant"]
    assert {r["variant"] for r in rows} == {HEADLINE, "no_kill", "k1"}
    root = tmp_path / "root"
    s = pd.read_csv(tmp_path / "out" / "G6" / "streams.csv", index_col=0, parse_dates=True)
    rf, core = s.get("rf"), s["core:C1/headline"].dropna()
    ecells = engine_cells(fam)
    hashes = {}
    for r in ledger.rows():
        if r.get("stage") == "F" and r.get("kind") == "rule":
            hashes[r["spec_json"]] = r["cell_hash"]
    vals = []
    for name, cells in ecells.items():
        h = hashes[cells[0].spec_json()]
        for ck in ("registered", "1.0"):
            cell = _by_symbol(cells, [h], root, None if ck == "registered" else float(ck))["SVXY"]
            # on the headline's days: a cell is flat (zero excess) before its own live start
            hd = s[HEADLINE].dropna().index
            a = T.excess(cell, rf).reindex(hd)
            a[hd < cell.index.min()] = 0.0
            assert a.notna().all()
            j = pd.concat([a.rename("a"), T.excess(core, rf).rename("c")], axis=1).dropna()
            want = T.sharpe_ann(j["c"] + 0.5 * j["a"]) - T.sharpe_ann(j["c"])
            st = reg["cells"][f"SVXY:{name}"][ck]
            assert st["delta_sharpe"] == pytest.approx(want, rel=1e-9, abs=1e-12)
            assert st["n_days"] == len(hd)
        assert _daily_costs(root, h) is not None
        vals.append(reg["cells"][f"SVXY:{name}"]["registered"]["delta_sharpe"])
    coh = res["evaluation"]["variants"][HEADLINE]["cells"]
    sign = np.sign(res["evaluation"]["verdict"]["delta_ann"])
    assert coh["n_cells"] == 4 and coh["share"] == pytest.approx(np.mean(np.sign(vals) == sign))
    assert res["evaluation"]["verdict"]["coherence_kind"] == "cells"
    # the overlay alone, the kill record and the report sections
    oa = res["overlay_alone"]
    assert oa["kind"] == "overlay_alpha" and "alpha_bp_per_day" in oa
    assert set(res["kill"]) == {HEADLINE, "no_kill", "k1"}
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    for s_ in ("## Region cells", "Δ Sharpe (core + k × cell vs core)", "## Episodes", "The overlay alone",
               "Drawdown kill"):  # fmt: skip
        assert s_ in md, s_


def test_one_decision_per_session_at_the_mark_reading_the_previous_settlement(tmp_path):
    """The headline member's fills: entries at the 10:00 bar's open only, and the shares traded on a session are
    nonzero exactly when the target (recomputed here from the previous session's synthetic VIX state) changes."""
    from families.run import _daily_costs, _read

    res, _ = _run_g6(tmp_path)
    h = res["members"][HEADLINE][0]
    root = tmp_path / "root"
    tr = _read(root, h, "trades.csv")
    entry = pd.to_datetime(tr["entry_time"], utc=True).dt.tz_convert(NY)
    assert set(entry.dt.strftime("%H:%M")) == {"10:00"}
    src = VixSource()
    days = pd.bdate_range("2022-01-03", "2022-04-01", inclusive="left")
    vix = src._exo["VIX"]["value"].reindex(days.union(src._exo["VIX"].index)).shift(1).reindex(days)
    r1 = src._exo["VX1_VIX"]["value"].reindex(days.union(src._exo["VX1_VIX"].index)).shift(1).reindex(days)
    on = lambda th, cap: (r1 >= 1.0) & (r1 > th) & (True if cap is None else vix < cap)
    target = sum(on(th, cap).astype(float) for th in (1.0, 1.05) for cap in (20, None)) / 4
    dc = _daily_costs(root, h)
    traded = sum(dc[f"notional_{k}"] for k in T.FILL_CLASSES).reindex(days).fillna(0.0)
    # the first sessions have no event (the barrier width's σ warms up; the member's live start): flat until then
    live = pd.Timestamp(_read(root, h, "daily_returns.csv").index[0][:10])
    assert days[0] < live <= days[3]
    target = target.where(target.index >= live, 0.0)
    changed = target.diff().abs() > 1e-12
    changed.iloc[0] = target.iloc[0] > 0
    gate = res["kill"][HEADLINE][h[:16]] is not None
    assert not gate  # the synthetic path never kills
    # sessions where the target held still trade nothing; sessions where it changed trade. The window's last session
    # closes the position (its bet would exit after the data: dropped), whatever the target
    last = days[-1]
    assert traded[last] > 0
    traded, changed = traded.drop(last), changed.drop(last)
    assert (traded[~changed] == 0).all()
    assert (traded[changed] > 0).all()
    # the episode table's position is the committed fraction held at the close
    ep = res["episodes"]["feb"]
    pos = pd.Series({pd.Timestamp(r["date"]): r["position"] for r in ep["rows"]})
    np.testing.assert_allclose(pos.to_numpy(), target.reindex(pos.index).to_numpy(), atol=1e-12)
    assert ep["summary"]["days"] == len(ep["rows"]) > 0 and "portfolio_total" in ep["summary"]


# ── The registered spec ──────────────────────────────────────────────────────


def test_g6_is_registered_as_the_user_chose():
    fam = load_family(ROOT / "families" / "G6.yaml")
    h = fam.headline.config
    p = h["primary"]["params"]
    assert h["primary"]["name"] == "vix_carry" and h["risk_profile"] == "vrp"
    assert p["theta_list"] == [1.0, 1.025, 1.05, 1.075, 1.10] and p["vix_list"] == [20, 25, 30]
    assert p["backwardation"] == 1.0 and p["decision"] == "10:00"
    assert p["scale_before"] == {"date": "2018-02-28", "scale": 0.5}
    assert fam.instruments == ("SVXY",) and fam.timeframe == "5Min"
    assert fam.test["kind"] == "marginal" and fam.test["coherence"] == "cells" and fam.test["at_cost"] == "registered"
    core = load_family(ROOT / "families" / "G2_core.yaml")
    assert fam.core == {"family": "G2_core", "variant": "headline", "k": 0.5, "sha": core.registered["sha"]}
    assert len(fam.variants) == 8 == fam.budget
    assert {v.label for v in fam.variants} >= {"no_kill", "k1", "core_bh"}
    assert fam.floors["max_dd_vs_core"] == 1.0 and fam.floors["min_net_ret"] == 0.03
    assert fam.power["diagnostic"] and fam.power["expected_delta_sharpe"] < fam.power["mde_delta_sharpe"]
    assert {e["label"] for e in fam.episodes} >= {"feb2018", "mar2020"}
    prog = yaml.safe_load((ROOT / "families" / "program.yaml").read_text(encoding="utf-8"))
    assert "G6" in prog["families"]


class CrashSource(VixSource):
    """VixSource with a state that is always fully long (VX1 / VIX 1.1, VIX 15) and SVXY falling 45 % inside the
    09:55 bar of CRASH_DAY: the kill fires at that close, one bar before the position's scheduled 10:00 exit."""

    CRASH_DAY = pd.Timestamp("2022-02-15")

    def __init__(self):
        super().__init__()
        days = pd.bdate_range(self.START, self.END, inclusive="left")
        self._exo = {"VIX": _series(days, np.full(len(days), 15.0)), "VX1_VIX": _series(days, np.full(len(days), 1.1)),
                     "VX2_VX1": _series(days, np.full(len(days), 1.05))}  # fmt: skip

    def _full(self, symbol):
        df = super()._full(symbol)
        if symbol != "SVXY" or getattr(self, "_crashed", False):
            return df
        df = df.copy()
        m = df.index >= (self.CRASH_DAY + pd.Timedelta("09:55:00")).tz_localize(NY)
        first = np.flatnonzero(m)[0]
        cols = ["open", "high", "low", "close"]
        df.loc[m, cols] *= 0.55
        df.iloc[first, df.columns.get_indexer(["open", "high"])] /= 0.55  # the 09:55 bar opens before the drop
        self._cache[symbol], self._crashed = df, True
        return df


def test_a_kill_one_bar_before_the_scheduled_exit_is_labelled_kill_and_latches(tmp_path, monkeypatch):
    from experiments import ledger as L
    from families.run import _read, run_family

    docs = {"C1": _region_doc(id="C1", variants=[]), "G6": _g6_doc()}
    for k, d in docs.items():
        (tmp_path / f"{k}.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G6]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    res = run_family(tmp_path / "G6.yaml", ledger=L.Ledger(tmp_path / "ledger.jsonl"), root=tmp_path / "root",
                     out_dir=tmp_path / "out", source=CrashSource(), repo=tmp_path, check_registration=False,
                     program=prog, quasi=False)  # fmt: skip
    h = res["members"][HEADLINE][0]
    k = pd.Timestamp(res["kill"][HEADLINE][h[:16]]).tz_convert(NY)
    assert k == (CrashSource.CRASH_DAY + pd.Timedelta("09:55:00")).tz_localize(NY)
    tr = _read(tmp_path / "root", h, "trades.csv")
    last = tr.iloc[-1]
    assert last["exit_reason"] == "kill"
    assert pd.Timestamp(last["exit_time"]).tz_convert(NY) == k + pd.Timedelta(minutes=5)
    assert (pd.to_datetime(tr["entry_time"], utc=True) < pd.Timestamp(last["exit_time"])).all()  # no later entry
    # no_kill keeps trading after the crash
    h2 = res["members"]["no_kill"][0]
    tr2 = _read(tmp_path / "root", h2, "trades.csv")
    assert (pd.to_datetime(tr2["entry_time"], utc=True) > pd.Timestamp(last["exit_time"])).any()
