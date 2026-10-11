"""
U25 (PLAN3 §5 U25, SPEC §26): G3's gate kernels and the region primary's `gate`, the Δ Sharpe MDE line, the `paired`
kind and its `reference`, the `marginal` kind's per-variant core run in the family's batch (core.k / core.variant /
core.family, the benchmark core), the new floors, coherence on the right statistic, the combined account check, the
registry, and the registered G2 / G3 / core specs.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from families import test as T
from families.spec import HEADLINE, load_family, parse
from features.kernels import prior_lag1_autocorr, prior_rank, session_layout, session_sigma
from tests.test_u14 import intraday
from tests.test_u24 import _cfg, _region_doc

ROOT = Path(__file__).resolve().parent.parent


# ── Kernels ──────────────────────────────────────────────────────────────────


def _old_vol_rank(df, sessions):
    """families.run.vol_rank as U24 built it (pandas), the reference for the one kernel definition."""
    day = df.index.tz_convert("America/New_York").normalize().tz_localize(None)
    x = np.log(df["close"].astype(float)).groupby(day).diff().groupby(day).std(ddof=1).to_numpy()
    out = np.full(len(x), np.nan)
    for d in range(sessions, len(x)):
        w = x[d - sessions : d]
        w = w[np.isfinite(w)]
        v = x[d - 1]
        if np.isfinite(v) and len(w) >= sessions // 2:
            out[d] = ((w < v).sum() + 0.5 * (w == v).sum()) / len(w)
    return out


def test_session_sigma_and_prior_rank_are_the_u24_vol_rank():
    from families.run import vol_rank

    df = intraday(60, seed=11)
    sess = session_layout(df.index, 5)[0]
    sig = session_sigma(df["close"].to_numpy(), sess)
    day = df.index.tz_convert("America/New_York").normalize()
    ref = np.log(df["close"]).groupby(day).diff().groupby(day).std(ddof=1).to_numpy()
    np.testing.assert_allclose(sig, ref, rtol=1e-12)
    old = _old_vol_rank(df, 20)
    np.testing.assert_array_equal(prior_rank(sig, 20), old)
    np.testing.assert_array_equal(vol_rank(df, sessions=20).to_numpy(), old)
    with pytest.raises(ValueError, match="k must"):
        prior_rank(sig, 1)


def test_prior_rank_and_autocorr_are_known_at_the_open():
    df = intraday(40, seed=12)
    sess = session_layout(df.index, 5)[0]
    c = df["close"].to_numpy()
    r0, a0 = prior_rank(session_sigma(c, sess), 10), prior_lag1_autocorr(c, sess, 5)
    c2 = c.copy()
    last = sess == sess[-1]
    c2[last] *= np.exp(np.random.default_rng(1).normal(0, 0.02, last.sum()))  # today's bars move
    np.testing.assert_array_equal(prior_rank(session_sigma(c2, sess), 10), r0)
    np.testing.assert_array_equal(prior_lag1_autocorr(c2, sess, 5), a0)
    assert np.isnan(a0[:5]).all() and np.isfinite(a0[5:]).all()


def test_prior_lag1_autocorr_by_hand():
    df = intraday(12, seed=13)
    sess = session_layout(df.index, 5)[0]
    c = df["close"].to_numpy()
    ac = prior_lag1_autocorr(c, sess, 3)
    xs, ys = [], []
    for s in (6, 7, 8):  # session 9 reads sessions 6, 7, 8: pairs inside one session only
        r = np.diff(np.log(c[sess == s]))
        xs.append(r[:-1])
        ys.append(r[1:])
    assert ac[9] == pytest.approx(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1], rel=1e-10)


# ── The region primary's gate ────────────────────────────────────────────────


GATE_BASE = {"estimators": ["band", "sgv"], "n_list": [6], "theta_list": [0.5], "norm_sessions": 5,
             "band_lookback": 5}  # fmt: skip


def test_the_gate_flattens_the_sessions_it_closes_and_r0_is_unchanged():
    from primaries import make_primary

    df = intraday(50, seed=14)
    sess = session_layout(df.index, 5)[0]
    cfg0 = _cfg(GATE_BASE)
    p0 = make_primary(cfg0)
    dec, m = p0.cell_matrix(df, cfg0)
    X = pd.DataFrame(index=df.index[dec])
    s0, m0 = p0.rule(df, X, cfg0)
    assert (p0.target_scale(df) == 1.0).all()
    np.testing.assert_array_equal(s0 * m0, m.mean(axis=1))  # R₀: no gate, the cells' mean exactly
    rank = prior_rank(session_sigma(df["close"].to_numpy(), sess), 10)
    for gate, want in (
        ({"kind": "vol_quantile", "below": 0.3, "sessions": 10}, np.where(np.isnan(rank), 1.0, rank >= 0.3)),
        ({"kind": "vol_size", "sessions": 10}, np.where(np.isnan(rank), 1.0, np.ceil(np.nan_to_num(rank) * 5) / 5)),
    ):
        cfg = _cfg({**GATE_BASE, "gate": gate})
        p = make_primary(cfg)
        sc = p.gate_scale(df)
        np.testing.assert_array_equal(sc, want.astype(float)[sess])
        s, mm = p.rule(df, X, cfg)
        np.testing.assert_allclose(s * mm, m.mean(axis=1) * sc[dec], atol=0)
    assert set(np.unique(make_primary(_cfg({**GATE_BASE, "gate": {"kind": "vol_size", "sessions": 10}}))
                         .gate_scale(df))) <= {0.2, 0.4, 0.6, 0.8, 1.0}  # fmt: skip
    ac = prior_lag1_autocorr(df["close"].to_numpy(), sess, 5)
    g = make_primary(_cfg({**GATE_BASE, "gate": {"kind": "gamma", "min": 0.0}})).gate_scale(df)
    np.testing.assert_array_equal(g, np.where(np.isnan(ac), 1.0, ac >= 0.0).astype(float)[sess])
    closed = make_primary(_cfg({**GATE_BASE, "gate": {"kind": "vol_quantile", "below": 0.3, "sessions": 10}}))
    assert (closed.gate_scale(df)[sess < 10] == 1.0).all()  # undefined state: the gate is open


def test_the_gate_is_validated():
    from primaries import make_primary

    for bad, match in (({"kind": "moon"}, "gate must"), ({"kind": "vol_quantile"}, "below"),
                       ({"kind": "vol_quantile", "below": 1.2}, "below"),
                       ({"kind": "vol_size", "below": 0.2}, "takes"), ({"kind": "gamma", "min": 2}, "min"),
                       ({"kind": "gamma", "sessions": 1}, "sessions"), ("vol", "gate must")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            make_primary(_cfg({**GATE_BASE, "gate": bad}))


# ── Power ────────────────────────────────────────────────────────────────────


def test_mde_delta_sharpe_matches_the_simulated_sd_of_a_sharpe_difference():
    from families.power import mde_delta_sharpe

    rng = np.random.default_rng(0)
    n, rho, sr = 2400, 0.9, 0.8
    s_d = sr / np.sqrt(252)
    cov = np.array([[1, rho], [rho, 1]]) * 1e-4
    d = []
    for _ in range(1500):
        x = rng.multivariate_normal([s_d * 1e-2, s_d * 1e-2], cov, n)
        sh = x.mean(0) / x.std(0, ddof=1) * np.sqrt(252)
        d.append(sh[0] - sh[1])
    m = mde_delta_sharpe(n, rho, sr, families=4)
    assert np.std(d) == pytest.approx(m["se_delta_sharpe"], rel=0.08)
    assert m["mde_delta_sharpe"] == pytest.approx((m["z_alpha"] + m["z_power"]) * m["se_delta_sharpe"])
    assert mde_delta_sharpe(n, 0.99, sr)["mde_delta_sharpe"] < mde_delta_sharpe(n, 0.8, sr)["mde_delta_sharpe"]
    with pytest.raises(ValueError):
        mde_delta_sharpe(n, 1.0, sr)


# ── Spec ─────────────────────────────────────────────────────────────────────


def _sharpe_power(n=60, rho=0.9, sr=0.5, expected=0.1, families=1):
    from families.power import mde_delta_sharpe

    m = mde_delta_sharpe(n, rho, sr, families=families)["mde_delta_sharpe"]
    return {"n_days": n, "rho": rho, "sharpe_ref": sr, "families": families, "mde_delta_sharpe": round(m, 3),
            "expected_delta_sharpe": expected, "diagnostic": expected < m}  # fmt: skip


def _paired_doc(**extra):
    doc = _region_doc(id="G3", reference={"family": "G1"}, power=_sharpe_power(),
                      floors={"min_net_ret_vs_reference": 0.9, "min_edge_to_cost": 2.0})  # fmt: skip
    doc["headline"]["primary"]["params"]["gate"] = {"kind": "vol_quantile", "below": 0.3, "sessions": 10}
    doc["test"] = {**doc["test"], "kind": "paired", "coherence": "variants"}
    doc["variants"] = [{"label": "gamma", "primary.params.gate": {"kind": "gamma", "min": 0.0}}]
    doc.update(extra)
    return doc


def _marginal_doc(**extra):
    doc = _region_doc(id="G2", core={"family": "C1", "variant": "headline", "k": 1.0}, power=_sharpe_power(),
                      floors={"min_net_ret": 0.0, "min_edge_to_cost": 0.1, "max_dd_vs_core": 1.0})  # fmt: skip
    doc["test"] = {**doc["test"], "kind": "marginal", "coherence": "variants"}
    doc["variants"] = [{"label": "k2", "core.k": 2.0}, {"label": "core_bh", "core.variant": "benchmark"}]
    doc.update(extra)
    return doc


def test_paired_and_marginal_specs_are_validated():
    fam = parse(_paired_doc())
    assert fam.reference == {"family": "G1"} and fam.power["mde_delta_sharpe"] > 0
    with pytest.raises(ValueError, match="needs reference"):
        parse({k: v for k, v in _paired_doc().items() if k != "reference"})
    with pytest.raises(ValueError, match="another family"):
        parse(_paired_doc(reference={"family": "G3"}))
    with pytest.raises(ValueError, match="one reference"):
        parse(_paired_doc(variants=[{"label": "x", "reference.family": "G2"}]))
    with pytest.raises(ValueError, match="paired' only"):
        parse(_region_doc(reference={"family": "G1"}))
    with pytest.raises(ValueError, match="min_net_ret_vs_reference"):
        parse(_region_doc(floors={"min_net_ret": 0.03, "min_edge_to_cost": 2.0, "min_net_ret_vs_reference": 0.9}))
    with pytest.raises(ValueError, match="max_dd_vs_core"):
        parse(_paired_doc(floors={"min_net_ret_vs_reference": 0.9, "min_edge_to_cost": 2.0, "max_dd_vs_core": 1.0}))
    with pytest.raises(ValueError, match="disagrees"):
        parse(_paired_doc(power={**_sharpe_power(), "mde_delta_sharpe": 9.0}))
    with pytest.raises(ValueError, match="diagnostic"):
        parse(_paired_doc(power={**_sharpe_power(), "diagnostic": False}))
    m = parse(_marginal_doc(variants=[{"label": "k2", "core.k": 2.0}, {"label": "c2", "core.family": "C2"},
                                      {"label": "bh", "core.variant": "benchmark"}]))  # fmt: skip
    assert [v.core for v in m.variants] == [{"family": "C1", "variant": "headline", "k": 1.0},
                                            {"family": "C1", "variant": "headline", "k": 2.0},
                                            {"family": "C2", "variant": "headline", "k": 1.0},
                                            {"family": "C1", "variant": "benchmark", "k": 1.0}]  # fmt: skip
    # a core makes a distinct configuration: the k / core variants have their own trial keys (U25)
    keys = [m.trial_key(v.label) for v in m.variants]
    assert len(set(keys)) == 4
    with pytest.raises(ValueError, match="family id"):
        parse(_marginal_doc(variants=[{"label": "bad", "core.family": "../x"}]))


def test_trial_keys_without_a_core_are_unchanged():
    """G1's registered trial keys (in the ledger since U24) must not move with the core-aware key."""
    g1 = load_family(ROOT / "families" / "G1.yaml")
    assert g1.trial_key(HEADLINE) == "G1/e4edaf46faea"


# ── Statistics ───────────────────────────────────────────────────────────────


def _days(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_paired_compare_floors_and_coherence_statistic():
    from families.stats import sharpe_test

    d = _days(800)
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(3e-4, 0.006, 800), index=d)
    keep = rng.random(800) > 0.2
    a = ref.where(keep, 0.0)  # a gated stream: the reference on the kept days, flat otherwise
    rf = pd.Series(1e-4, index=d)
    test = {"kind": "paired", "sided": "one", "block_days": 21, "n_boot": 299, "alpha": 0.05, "seed": 0}
    c = T.compare(a, ref, "constant_mix_ew", test, rf=rf, ref=ref)
    lw = sharpe_test((a - rf).to_numpy(), (ref - rf).to_numpy(), block=21, n_boot=299, seed=0, sided="one")
    assert c["delta_ann"] == pytest.approx(lw["delta_ann"]) and c["p"] == lw["p"]
    assert c["days_differ"] == int((~keep & (ref != 0)).sum()) and c["statistic"] == "delta_sharpe_paired"
    fl = T.floors(a, ref, {"pnl": 1.0, "cost_paid": 0.1, "traded_notional": 10.0},
                  {"min_net_ret_vs_reference": 0.9, "min_edge_to_cost": 2.0}, rf, ref)  # fmt: skip
    x = fl["min_net_ret_vs_reference"]
    assert x["value"] == pytest.approx(T.stream_stats(a - rf)["ret_ann"])
    assert x["threshold"] == pytest.approx(0.9 * T.stream_stats(ref - rf)["ret_ann"])
    assert (
        T.floors(
            a, ref, {"pnl": 1.0, "cost_paid": 0.1, "traded_notional": 10.0}, {"min_net_ret_vs_reference": 0.9}, rf
        )["min_net_ret_vs_reference"]["ok"]
        is False
    )  # no ref
    # the coherence statistic of a paired / marginal variant is its own Δ Sharpe, not the benchmark's
    bench = pd.Series(rng.normal(5e-4, 0.01, 800), index=d)
    dc = T.delta_common(a, a, bench, "constant_mix_ew", rf, test, ref)
    assert dc == pytest.approx(T.sharpe_ann(a - rf) - T.sharpe_ann(ref - rf))
    mt = {**test, "kind": "marginal"}
    dm = T.delta_common(a, a, bench, "constant_mix_ew", rf, mt, ref, 2.0)
    assert dm == pytest.approx(T.sharpe_ann((ref - rf) + 2.0 * (a - rf)) - T.sharpe_ann(ref - rf))
    assert np.isnan(T.delta_common(a, a, bench, "constant_mix_ew", rf, mt, None))


def test_marginal_dd_floor_and_matched_volatility():
    d = _days(600)
    rng = np.random.default_rng(4)
    core = pd.Series(rng.normal(4e-4, 0.01, 600), index=d)
    ov = pd.Series(rng.normal(2e-4, 0.005, 600), index=d)
    test = {"kind": "marginal", "sided": "one", "block_days": 21, "n_boot": 199, "alpha": 0.05, "seed": 0}
    c = T.compare(ov, core, "cash", test, ref=core, k=2.0)
    port = core + 2.0 * ov
    mv = c["matched_vol"]
    assert mv["scale"] == pytest.approx(port.std() / core.std())
    assert mv["matched_core_ret_ann"] == pytest.approx(T.stream_stats(core * mv["scale"])["ret_ann"])
    fl = T.floors(ov, core, {"pnl": 1.0, "cost_paid": 0.1, "traded_notional": 10.0}, {"max_dd_vs_core": 1.0},
                  None, core, 2.0)  # fmt: skip
    x = fl["max_dd_vs_core"]
    assert x["value"] == pytest.approx(T.drawdown(port)) and x["core"] == pytest.approx(T.drawdown(core))
    assert x["ok"] == bool(T.drawdown(port) >= T.drawdown(core))


# ── End to end on synthetic bars ─────────────────────────────────────────────


def _run(tmp_path, docs: dict, fid: str, **kw):
    from experiments import ledger as L
    from families.run import run_family
    from tests.test_u17 import OvernightSource

    for k, doc in docs.items():
        (tmp_path / f"{k}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    prog = tmp_path / "program.yaml"
    prog.write_text(f"program: P\nfamilies: [{fid}]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    ledger = L.Ledger(tmp_path / "ledger.jsonl")
    res = run_family(tmp_path / f"{fid}.yaml", ledger=ledger, root=tmp_path / "root", out_dir=tmp_path / "out",
                     source=OvernightSource(0.0), repo=tmp_path, check_registration=False, program=prog, quasi=False,
                     **kw)  # fmt: skip
    return res, ledger


def test_a_paired_family_tests_the_gate_against_its_reference_at_every_cost(tmp_path):
    from families.run import _by_symbol

    ref_doc = _region_doc(variants=[])
    res, ledger = _run(tmp_path, {"G1": ref_doc, "G3": _paired_doc()}, "G3")
    h = res["evaluation"]["variants"][HEADLINE]["compare"]
    assert h["kind"] == "paired" and np.isfinite(h["delta_ann"]) and h["days_differ"] > 0
    # the reference ran in G3's batch: no G1 family-variant row, and its stream at each cost is G1's members re-priced
    rows = [r for r in ledger.rows() if r.get("kind") == "family_variant"]
    assert {r["family"] for r in rows} == {"G3"}
    g1 = load_family(tmp_path / "G1.yaml")
    members = res["refs"]["G1"]["variants"][HEADLINE]["members"]
    root = tmp_path / "root"
    s = pd.read_csv(tmp_path / "out" / "G3" / "streams.csv", index_col=0, parse_dates=True)
    rf, days = s.get("rf"), s[HEADLINE].dropna().index
    sharpes = set()
    for ck in ("1.0", "2.3", "registered"):
        cost = None if ck == "registered" else float(ck)
        refs = T.day_index(T.pool(_by_symbol(g1.cells[HEADLINE], members, root, cost),
                                  pd.Series({"QQQ": 0.5, "SPY": 0.5})))  # fmt: skip
        if ck == "1.0":  # the verdict's column: the saved reference stream
            np.testing.assert_allclose(s["reference:G1/headline"].dropna().to_numpy(), refs.to_numpy())
        cc = res["cost_curve"][ck][HEADLINE]["compare"]
        want = T.sharpe_ann(T.excess(refs, rf).reindex(days).dropna())
        assert cc["ref_sharpe"] == pytest.approx(want, rel=1e-9)
        sharpes.add(round(cc["ref_sharpe"], 9))
    assert len(sharpes) == 3  # the reference is priced at each column's cost, not once
    # the gate closes sessions after its 10-session warm-up only, and those sessions are flat in G3's members
    assert res["region"]["parity"]["SPY"]["max_abs_diff"] == 0.0
    assert res["evaluation"]["verdict"]["coherence_kind"] == "variants"
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    for s_ in ("## Reference families", "the reference's", "MDE line: Δ Sharpe"):
        assert s_ in md, s_


def test_a_marginal_family_reads_each_variants_core_from_its_batch(tmp_path):
    core_doc = _region_doc(id="C1", variants=[])
    core_doc["headline"]["primary"]["params"]["vm_list"] = [1.5]  # a different rule: its own cells
    res, ledger = _run(tmp_path, {"C1": core_doc, "G2": _marginal_doc()}, "G2")
    ev = res["evaluation"]["variants"]
    assert ev[HEADLINE]["compare"]["k"] == 1.0 and ev["k2"]["compare"]["k"] == 2.0
    s = pd.read_csv(tmp_path / "out" / "G2" / "streams.csv", index_col=0, parse_dates=True)
    rf = s.get("rf")
    core, bh = s["core:C1/headline"].dropna(), s["core:C1/benchmark"].dropna()
    assert not np.allclose(core.to_numpy()[:20], bh.loc[core.index].to_numpy()[:20])
    # each variant's statistic reads its own core and k
    for label, c_, k in ((HEADLINE, core, 1.0), ("k2", core, 2.0), ("core_bh", bh, 1.0)):
        j = pd.concat([T.excess(s[label], rf).rename("a"), T.excess(c_, rf).rename("c")], axis=1).dropna()
        assert ev[label]["compare"]["portfolio_sharpe"] == pytest.approx(T.sharpe_ann(j["c"] + k * j["a"]), rel=1e-9)
        assert ev[label]["compare"]["core_sharpe"] == pytest.approx(T.sharpe_ann(j["c"]), rel=1e-9)
    # trial keys: three variants, three keys (k and the core are part of the configuration)
    keys = {r["trial_key"] for r in ledger.rows() if r.get("kind") == "family_variant"}
    assert len(keys) == 3 and not any(
        r.get("family") == "C1" for r in ledger.rows() if r.get("kind") == "family_variant"
    )
    pa = res["portfolio_account"]
    assert set(pa) == {HEADLINE, "k2", "core_bh"}
    assert pa["k2"]["buying_power"]["max_gross_intraday"] > pa[HEADLINE]["buying_power"]["max_gross_intraday"]
    assert pa["core_bh"]["buying_power"]["max_gross_overnight"] == pytest.approx(1.0)  # the held mix
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    for s_ in ("## Core families", "Matched volatility", "≤ 2× (PLAN3)"):
        assert s_ in md, s_


def test_a_failed_or_missing_core_is_reported_not_tested(tmp_path):
    with pytest.raises(FileNotFoundError, match="no spec"):
        _run(tmp_path, {"G2": _marginal_doc()}, "G2")
    core_doc = _region_doc(id="C1", variants=[], window={"start": "2022-01-03", "end": "2022-03-01"})
    with pytest.raises(ValueError, match="runs over"):
        _run(tmp_path, {"C1": core_doc, "G2": _marginal_doc()}, "G2")


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_records_a_passing_family_and_refuses_a_failing_one(tmp_path):
    from families.registry import RegistryError, add_entry

    out = tmp_path / "out"
    (out / "G1").mkdir(parents=True)
    result = {"id": "G1", "registered": {"sha": "abc", "date": "2026-10-11"}, "run_at": "t", "git_sha": "g",
              "code_hash": "c", "members": {HEADLINE: ["h1", "h2"]}, "variant_hashes": {HEADLINE: "vh"},
              "power": {"diagnostic": True}, "test": {"kind": "overlay_alpha"},
              "evaluation": {"verdict": {"p": 0.003, "delta_ann": 0.05, "floors_ok": True, "positive": True,
                                         "coherence": {"coherent": True}, "kind": "overlay_alpha", "at_cost": "1.0"}}}  # fmt: skip
    (out / "G1" / "result.json").write_text(json.dumps(result), encoding="utf-8")
    bad = copy.deepcopy(result)
    bad["id"] = "G3"
    bad["evaluation"]["verdict"]["p"] = 0.4
    (out / "G3").mkdir()
    (out / "G3" / "result.json").write_text(json.dumps(bad), encoding="utf-8")
    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G1, G3]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    reg = tmp_path / "registry.yaml"
    e = add_entry("G1", out, reg, program=prog, freeze_date="2026-10-12")
    assert e["members"] == ["h1", "h2"] and e["freeze_date"] == "2026-10-12" and e["diagnostic"]
    assert e["verdict"]["p_holm"] == pytest.approx(0.006)  # Holm over G1 and G3
    with pytest.raises(RegistryError, match="does not pass"):
        add_entry("G3", out, reg, program=prog)
    doc = yaml.safe_load(reg.read_text(encoding="utf-8"))
    assert list(doc["claims"]) == ["G1"]


# ── The registered specs ─────────────────────────────────────────────────────


def test_g2_g3_and_the_cores_are_registered_as_plan3_states_them():
    fam = {f: load_family(ROOT / "families" / f"{f}.yaml") for f in ("G1", "G2", "G3", "G2_core", "G2_core4")}
    g1 = sorted(c.spec_json() for c in fam["G1"].cells[HEADLINE])
    assert sorted(c.spec_json() for c in fam["G2"].cells[HEADLINE]) == g1  # the overlay is G1's headline exactly
    g3 = fam["G3"].headline.config["primary"]["params"]
    assert g3["gate"] == {"kind": "vol_quantile", "below": 0.2, "sessions": 252}
    assert {k: v for k, v in g3.items() if k != "gate"} == fam["G1"].headline.config["primary"]["params"]
    assert fam["G3"].reference == {"family": "G1", "variant": "headline"} and fam["G3"].test["kind"] == "paired"
    assert [v.label for v in fam["G3"].variants] == [HEADLINE, "gate_q2", "size_quintile", "gamma_gate"]
    assert fam["G3"].floors["min_net_ret_vs_reference"] == 0.9 and fam["G3"].power["diagnostic"]
    g2 = fam["G2"]
    assert g2.test["kind"] == "marginal" and g2.floors["max_dd_vs_core"] == 1.0 and g2.floors["min_net_ret"] == 0.03
    assert [(v.label, v.core["family"], v.core["variant"], v.core["k"]) for v in g2.variants] == [
        (HEADLINE, "G2_core", "headline", 1.0), ("k0.5", "G2_core", "headline", 0.5),
        ("k2", "G2_core", "headline", 2.0), ("core_bh", "G2_core", "benchmark", 1.0),
        ("core_f1_4", "G2_core4", "headline", 1.0), ("overlay_vol_target", "G2_core", "headline", 1.0)]  # fmt: skip
    assert g2.power["diagnostic"] and g2.budget == 6 and fam["G3"].budget == 6
    f1 = yaml.safe_load((ROOT / "families" / "F1.yaml").read_text(encoding="utf-8"))["headline"]["primary"]
    for core, inst, w in (
        ("G2_core", ("QQQ", "SPY"), "equal"),
        ("G2_core4", ("DIA", "IWM", "QQQ", "SPY"), "equal_risk"),
    ):
        c = fam[core]
        assert c.headline.config["primary"] == f1 and c.instruments == inst and c.weighting == w
        assert c.headline.config["overrides"] == fam["G1"].headline.config["overrides"]
        assert c.window == fam["G2"].window
    prog = yaml.safe_load((ROOT / "families" / "program.yaml").read_text(encoding="utf-8"))
    assert "G2_core" not in prog["families"] and {"G2", "G3"} <= set(prog["families"])


# ── U25 review: pinned cores, identical streams, a losing reference ──────────


def _git(repo, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_a_core_changed_after_the_result_is_refused(tmp_path):
    """The reviewer's SEVERE scenario: a core edited and re-registered after G2's run must not be read silently."""
    from experiments import ledger as L
    from families.run import run_family
    from families.spec import register
    from tests.test_u17 import OvernightSource

    repo = tmp_path / "repo"
    fams = repo / "families"
    fams.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "x@y")
    _git(repo, "config", "user.name", "x")
    core = _region_doc(id="C1", variants=[])
    core["headline"]["primary"]["params"]["vm_list"] = [1.5]
    (fams / "C1.yaml").write_text(yaml.safe_dump(core, sort_keys=False), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "core")
    sha = register(fams / "C1.yaml", repo)["sha"]
    _git(repo, "commit", "-qam", "register core")
    unpinned = _marginal_doc(variants=[])
    pinned = _marginal_doc(variants=[], core={"family": "C1", "variant": "headline", "k": 1.0, "sha": sha})
    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G2]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    kw = {"ledger": L.Ledger(tmp_path / "ledger.jsonl"), "root": tmp_path / "root", "source": OvernightSource(0.0),
              "repo": repo, "program": prog, "quasi": False}  # fmt: skip
    for doc, ok in ((unpinned, False), (pinned, True)):
        (fams / "G2.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "g2")
        register(fams / "G2.yaml", repo)
        _git(repo, "commit", "-qam", "register g2")
        if ok:
            assert (
                run_family(fams / "G2.yaml", out_dir=tmp_path / "out", **kw)["refs"]["C1"]["registered"]["sha"] == sha
            )
        else:
            with pytest.raises(ValueError, match="pins"):
                run_family(fams / "G2.yaml", out_dir=tmp_path / "out", **kw)
            (fams / "G2.yaml").unlink()
            _git(repo, "commit", "-qam", "drop unpinned")
    core["headline"]["primary"]["params"]["vm_list"] = [1.25]  # the core changed after the result
    (fams / "C1.yaml").write_text(yaml.safe_dump(core, sort_keys=False), encoding="utf-8")
    _git(repo, "commit", "-qam", "new core")
    register(fams / "C1.yaml", repo)
    _git(repo, "commit", "-qam", "re-register core")
    with pytest.raises(ValueError, match="pins"):
        run_family(fams / "G2.yaml", out_dir=tmp_path / "out", **kw)


def test_a_paired_stream_identical_to_its_reference_has_no_statistic_and_a_losing_reference_floor():
    d = _days(300)
    ref = pd.Series(np.random.default_rng(5).normal(1e-4, 0.006, 300), index=d)
    test = {"kind": "paired", "sided": "one", "block_days": 21, "n_boot": 99, "alpha": 0.05, "seed": 0}
    c = T.compare(ref, ref, "constant_mix_ew", test, ref=ref)  # a gate that never closed: no crash
    assert c["p"] == 1.0 and np.isnan(c["delta_ann"]) and "error" in c
    costs = {"pnl": 1.0, "cost_paid": 0.1, "traded_notional": 10.0}
    lose = pd.Series(-4e-4, index=d) + ref * 0
    for a, ok in ((lose * 1.05, True), (lose * 1.2, False)):  # not lower than a losing reference by more than 10 %
        f = T.floors(a, ref, costs, {"min_net_ret_vs_reference": 0.9}, None, lose)["min_net_ret_vs_reference"]
        rc = T.stream_stats(lose)["ret_ann"]
        assert f["threshold"] == pytest.approx(1.1 * rc) and f["ok"] is ok
