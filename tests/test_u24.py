"""
U24 (PLAN3 §5 U24, SPEC §25): the G1 variants' new mechanisms (the MODWT-slope estimator, the vol-targeted size), the
family layer's region support (equal weighting, the region cells' specification curve with its parity check, cells
coherence in the verdict, the vol_quintile / opex_day splits, trade statistics) and the registered G1 spec's shape.
Synthetic data only: nothing here reads the development window.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from families import test as T
from families.spec import HEADLINE, load_family, parse
from features.kernels import (
    modwt_slope_all,
    modwt_slope_weights,
    session_layout,
    session_rv_sigma,
)
from tests.test_u14 import intraday

ROOT = Path(__file__).resolve().parent.parent


# ── MODWT slope ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("filt,j", [("db1", 3), ("db1", 4), ("la8", 2), ("db2", 3)])
def test_modwt_slope_equals_the_causal_pyramid_per_session(filt, j):
    """The FIR equals features.groups.causal_modwt's smooth, differenced, on each session's own bars; NaN for exactly
    the first (window − 1) bars of every session."""
    from features.groups import causal_modwt

    df = intraday(6, seed=3)
    sess, start, pos, _ = session_layout(df.index, 5)
    x = np.log(df["close"].to_numpy())
    out = modwt_slope_all(x, start, [j], filt)[0]
    width = len(modwt_slope_weights(j, filt))
    worst = 0.0
    for s in np.unique(sess):
        m = sess == s
        _, sm = causal_modwt(pd.Series(x[m]), filt, j)
        ref = sm.diff().to_numpy()
        assert np.array_equal(np.isnan(ref), np.isnan(out[m]))
        worst = max(worst, float(np.nanmax(np.abs(ref - out[m]))))
    assert worst < 1e-14
    assert np.array_equal(np.isnan(out), pos < width - 1)


def test_modwt_slope_haar_is_the_scaled_2j_bar_return_and_ignores_the_gap():
    df = intraday(3, seed=4)
    sess, start, pos, _ = session_layout(df.index, 5)
    x = np.log(df["close"].to_numpy())
    out = modwt_slope_all(x, start, [3])[0]
    t = np.flatnonzero((sess == 1) & (pos == 20))[0]
    assert out[t] == pytest.approx((x[t] - x[t - 8]) / 8, abs=1e-15)
    w = modwt_slope_weights(4)
    assert len(w) == 17 and abs(w.sum()) < 1e-15
    gapped = x.copy()
    gapped[sess >= 1] += 0.05  # a +5 % overnight gap before session 1
    out2 = modwt_slope_all(gapped, start, [3])[0]
    np.testing.assert_allclose(out2[sess >= 1], out[sess >= 1], atol=1e-13, equal_nan=True)


def test_modwt_slope_is_causal():
    df = intraday(4, seed=5)
    _, start, _, _ = session_layout(df.index, 5)
    x = np.log(df["close"].to_numpy())
    base = modwt_slope_all(x, start, [3, 4])
    cut = 150
    y = x.copy()
    y[cut + 1 :] += np.random.default_rng(0).normal(0, 0.01, len(y) - cut - 1)
    np.testing.assert_array_equal(modwt_slope_all(y, start, [3, 4])[:, : cut + 1], base[:, : cut + 1])


def test_session_rv_sigma_by_hand_and_blind_to_its_own_session():
    df = intraday(6, seed=6)
    sess, _, _, _ = session_layout(df.index, 5)
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    sig = session_rv_sigma(o, c, sess, 2)
    first = np.r_[True, sess[1:] != sess[:-1]]
    r = np.log(c / np.where(first, o, np.r_[np.nan, c[:-1]]))
    rv = pd.Series(r * r).groupby(sess).sum().to_numpy()
    assert np.isnan(sig[sess < 2]).all()
    for d in range(2, 6):
        assert np.allclose(sig[sess == d], np.sqrt(rv[d - 2 : d].mean()))
    c2 = c.copy()
    c2[sess == 4] *= 1.03  # the current session moves: its own σ̂ does not
    assert np.array_equal(session_rv_sigma(o, c2, sess, 2)[sess == 4], sig[sess == 4])


# ── Region primary: the modwt estimator and the vol target ───────────────────


def _cfg(params=None, **over):
    from primaries.mechanism import primary_config

    return primary_config("region_trend", "5Min", params or {}, META_MODEL="none", **over)


def test_modwt_cells_and_validation():
    from primaries import make_primary

    p = make_primary(_cfg({"estimators": ["band", "rmedv", "sgv", "modwt"]}))
    names = p.cell_names()
    assert len(names) == 51 and names[-8:] == [f"modwt_j{j}_t{t:g}" for j in (3, 4) for t in (0.75, 1.0, 1.5, 2.0)]
    df = intraday(30, seed=7)
    cfg = _cfg({"estimators": ["modwt"], "theta_list": [1.0], "norm_sessions": 5})
    dec, m = make_primary(cfg).cell_matrix(df, cfg)
    assert m.shape == (len(dec), 2) and set(np.unique(m)) <= {-1.0, 0.0, 1.0} and (m != 0).any()
    for bad, match in (({"modwt_filter": "nope"}, "modwt_filter"), ({"modwt_j_list": [3, 3]}, "modwt_j_list"),
                       ({"modwt_filter": "la8", "modwt_j_list": [4]}, "fit in a session"),
                       ({"vol_target": -0.1}, "vol_target"), ({"vol_cap": 0}, "vol_cap")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            make_primary(_cfg({"estimators": ["band", "modwt"], **bad}))


def test_vol_target_scales_the_mean_target_and_sets_size():
    from primaries import make_primary

    base = {"estimators": ["band", "sgv"], "n_list": [6], "theta_list": [0.5], "norm_sessions": 5}
    df = intraday(30, seed=8)
    cfg0 = _cfg(base)
    cfg1 = _cfg({**base, "vol_target": 0.10, "vol_cap": 2.0})
    assert cfg0.SIZE == 1.0 and cfg1.SIZE == 2.0
    p0, p1 = make_primary(cfg0), make_primary(cfg1)
    dec, m = p0.cell_matrix(df, cfg0)
    X = pd.DataFrame(index=df.index[dec])
    s0, m0 = p0.rule(df, X, cfg0)
    s1, m1 = p1.rule(df, X, cfg1)
    sess = session_layout(df.index, 5)[0]
    sig = session_rv_sigma(df["open"], df["close"], sess, 5)[dec]
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.minimum(2.0, 0.10 / np.sqrt(252) / sig) / 2.0
    k = np.where(np.isfinite(k), k, 0.0)
    target = m.mean(axis=1)
    np.testing.assert_allclose(m1, np.abs(target * k), atol=1e-15)
    np.testing.assert_allclose(s0 * m0, target, atol=1e-15)
    assert (m1 <= 1.0).all() and (m1[sess[dec] < 5] == 0).all()
    assert np.array_equal(np.sign(s1 * m1), np.sign(target * k))


# ── Family layer ─────────────────────────────────────────────────────────────


def _region_doc(**extra):
    from families.power import mde_alpha

    mde = mde_alpha(60, 0.08, families=1)["mde_bp_per_day"]
    doc = {
        "id": "G1",
        "mechanism": "test",
        "instruments": ["SPY", "QQQ"],
        "timeframe": "5Min",
        "window": {"start": "2022-01-03", "end": "2022-04-01"},
        "weighting": "equal",
        "headline": {
            "primary": {
                "name": "region_trend",
                "params": {
                    "estimators": ["band", "rmedv", "sgv"],
                    "n_list": [6, 12],
                    "theta_list": [0.75, 1.5],
                    "vm_list": [1.0],
                    "norm_sessions": 10,
                    "band_lookback": 10,
                },
            },
            "cost_model": "quotes",
            "risk_profile": "loss_gate",
            "overrides": {"INIT_CASH": 30000},
        },
        "variants": [{"label": "cadence15", "primary.params.cadence": 15}],
        "state_splits": ["vol_quintile", "opex_day", "year"],
        "response": ["sharpe", "worst_day", "longest_flat_run", "hit_rate"],
        "benchmark": "constant_mix_ew",
        "floors": {"min_net_ret": 0.03, "min_edge_to_cost": 2.0},
        "test": {"kind": "overlay_alpha", "sided": "one", "at_cost": 1.0, "n_boot": 199, "coherence": "cells"},
        "power": {
            "n_days": 60,
            "vol_ann": 0.08,
            "families": 1,
            "mde_alpha_bp_per_day": round(mde, 2),
            "expected_alpha_bp_per_day": 1.0,
            "diagnostic": True,
        },
    }
    doc.update(extra)
    return doc


def test_spec_weighting_and_cells_coherence_are_validated():
    fam = parse(_region_doc())
    assert fam.weighting == "equal" and fam.test["coherence"] == "cells"
    assert parse(_region_doc(weighting="equal_risk")).weighting == "equal_risk"
    assert parse({k: v for k, v in _region_doc().items() if k != "weighting"}).weighting == "equal_risk"
    with pytest.raises(ValueError, match="weighting"):
        parse(_region_doc(weighting="cap"))
    with pytest.raises(ValueError, match="test.coherence"):
        parse(_region_doc(test={**_region_doc()["test"], "coherence": "both"}))
    from tests.test_u22 import _doc

    with pytest.raises(ValueError, match="region primary"):  # an overnight rule has no cells
        parse(_doc(test={"coherence": "cells"}))


def test_cells_coherence_replaces_the_variant_coherence_in_the_verdict():
    head = {"compare": {"p": 0.01, "delta_ann": 0.05, "ci_ann": [0.01, 0.09]}, "floors": {"ok": True},
            "cells": T.cells_coherence({"a": 1.0, "b": -2.0, "c": -1.0}, 2.0, 0.667)}  # fmt: skip
    var = {"compare": {"delta_ann": -0.02}, "delta_common": -0.02, "floors": {"ok": False}}
    v = T.headline_verdict({HEADLINE: head, "x": var}, {"coherence_share": 0.667, "coherence": "cells"})
    assert v["coherence_kind"] == "cells" and v["coherence"]["n_cells"] == 3 and v["coherence"]["coherent"] is False
    assert v["variant_coherence"]["coherent"] is False and v["coherence"]["share"] == pytest.approx(1 / 3, abs=1e-12)
    head["cells"] = T.cells_coherence({"a": 1.0, "b": 2.0, "c": -1.0}, 2.0, 0.667)  # 2 / 3 = 0.667 to 3 decimals
    assert T.headline_verdict({HEADLINE: head}, {"coherence_share": 0.667, "coherence": "cells"})["coherence"][
        "coherent"
    ]
    head.pop("cells")
    v = T.headline_verdict({HEADLINE: head}, {"coherence_share": 0.667, "coherence": "cells"})
    assert v["coherence"]["coherent"] is False  # a region whose cells were not evaluated is not coherent
    v = T.headline_verdict({HEADLINE: head, "x": var}, {"coherence_share": 0.667})
    assert v["coherence_kind"] == "variants" and "variant_coherence" not in v


def test_vol_rank_is_a_causal_mid_rank():
    from families.run import vol_rank

    df = intraday(40, seed=9)
    r = vol_rank(df, sessions=10)
    day = df.index.tz_convert("America/New_York").normalize().tz_localize(None)
    sig = np.log(df["close"]).groupby(day).diff().groupby(day).std(ddof=1).to_numpy()
    assert r.iloc[:10].isna().all() and r.iloc[10:].between(0, 1).all()
    for d in (10, 25, 39):
        w, v = sig[d - 10 : d], sig[d - 1]
        assert r.iloc[d] == pytest.approx(((w < v).sum() + 0.5 * (w == v).sum()) / 10)
    df2 = df.copy()
    last = day == day[-1]
    df2.loc[last, "close"] = df2.loc[last, "close"] * np.exp(np.random.default_rng(1).normal(0, 0.02, last.sum()))
    assert vol_rank(df2, sessions=10).iloc[-1] == r.iloc[-1]  # today's σ is not today's state


def test_vol_quintile_and_opex_groups():
    days = pd.bdate_range("2022-01-03", periods=6)
    state = pd.DataFrame({"vol_rank": [np.nan, 0.05, 0.25, 0.5, 0.95, 1.0],
                          "opex": [False, True, False, False, False, True]}, index=days)  # fmt: skip
    g = T._groups("vol_quintile", days, state, pd.Series(0.0, index=days))
    assert g.tolist() == ["n/a", "q1 low", "q2", "q3", "q5 high", "q5 high"]
    assert T._groups("opex_day", days, state, None).tolist() == ["other", "opex", "other", "other", "other", "opex"]
    assert T._groups("vol_quintile", days, None, None) is None


def test_equal_weighting_reads_no_data():
    from families.run import family_weights

    w, basis = family_weights(None, ["SPY", "QQQ"], "5Min", ("2016-01-04", "2025-10-01"), "equal")
    assert w.to_dict() == {"QQQ": 0.5, "SPY": 0.5} and basis["window"] == "equal"


def test_a_region_family_runs_with_its_cells_curve_parity_and_cells_coherence(tmp_path):
    """End to end on synthetic bars: the region's re-simulated stream equals the headline members' bit for bit, every
    (instrument, cell) has a statistic at every cost, the verdict's coherence is the cells' share at 1.0 bp, the
    report renders the new sections."""
    from experiments import ledger as L
    from families.run import run_family
    from tests.test_u17 import OvernightSource

    path = tmp_path / "G1.yaml"
    path.write_text(yaml.safe_dump(_region_doc(), sort_keys=False), encoding="utf-8")
    prog = tmp_path / "program.yaml"
    prog.write_text("program: P\nfamilies: [G1]\nmax_families: 5\nmax_trials: 42\n", encoding="utf-8")
    ledger = L.Ledger(tmp_path / "ledger.jsonl")
    res = run_family(path, ledger=ledger, root=tmp_path / "root", out_dir=tmp_path / "out", source=OvernightSource(0.0),
                     repo=tmp_path, check_registration=False, program=prog, quasi=False)  # fmt: skip
    reg = res["region"]
    names = reg["names"]
    assert len(names) == 1 + 2 * 2 * 2 and reg["costs"] == ["registered", "0.3", "1.0", "2.3"]
    for sym in ("SPY", "QQQ"):
        assert reg["parity"][sym]["max_abs_diff"] == 0.0 and reg["parity"][sym]["common_days"] > 40
    assert len(reg["cells"]) == 2 * len(names) and all(set(v) == set(reg["costs"]) for v in reg["cells"].values())
    v = res["evaluation"]["verdict"]
    assert v["coherence_kind"] == "cells" and v["at_cost"] == "1.0" and "variant_coherence" in v
    alphas = np.array([x["1.0"]["alpha_bp"] for x in reg["cells"].values()])
    head = res["evaluation"]["variants"][HEADLINE]["compare"]["alpha_bp_per_day"]
    assert v["coherence"]["n_cells"] == len(alphas)
    assert v["coherence"]["share"] == pytest.approx(np.mean(np.sign(alphas) == np.sign(head)))
    assert res["weights"] == {"QQQ": 0.5, "SPY": 0.5}
    ts = res["trade_stats"]
    assert set(ts["per_instrument"]) == {"SPY", "QQQ"} and ts["pooled"]["round_trips_per_day"] > 0
    assert set(res["per_instrument_alpha"]) == {"SPY", "QQQ"} and res["power_realized"]["n_days"] > 40
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    for s in ("## Region cells", "Coherence (cells, the verdict's)", "## Trade statistics", "max |Δ daily return| 0",
              "**opex_day**", "## Per-instrument headline alpha"):  # fmt: skip
        assert s in md, s
    assert (Path(res["paths"]["md"]).parent / "cells_curve.png").exists()
    json.loads((tmp_path / "out" / "G1" / "result.json").read_text(encoding="utf-8"))


# ── The registered spec ──────────────────────────────────────────────────────


def test_g1_spec_is_r0_as_plan3_states_it():
    fam = load_family(ROOT / "families" / "G1.yaml")
    assert fam.instruments == ("QQQ", "SPY") and fam.weighting == "equal" and fam.timeframe == "5Min"
    assert len(fam.variants) == 11 and fam.budget == 12
    t = fam.test
    assert (t["kind"], t["sided"], t["at_cost"], t["coherence"], t["n_boot"]) == ("overlay_alpha", "one", 1.0, "cells",
                                                                                 5000)  # fmt: skip
    assert fam.floors["min_net_ret"] == 0.03 and fam.floors["min_edge_to_cost"] == 2.0
    assert fam.power["families"] == 4 and fam.account["name"] == "margin_30k"
    from primaries import make_primary

    cfg = fam.cells[HEADLINE][0].config()
    p = make_primary(cfg)
    assert len(p.cell_names()) == 43 and p.marks() == [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]
    assert (cfg.RISK_PROFILE, cfg.INIT_CASH, cfg.SIZE, cfg.SIZE_STEP, cfg.SIZER) == (
        "loss_gate",
        30000,
        1.0,
        0,
        "rule_size",
    )
    assert (cfg.FILL_AUCTION, cfg.COST_MODEL, cfg.COST_TABLE, cfg.CASH_YIELD, cfg.SESSION_CLOCK, cfg.STRESS_MULT) == (
        "print", "quotes", "asof", "tbill", "calendar", 1.0)  # fmt: skip
    from primaries.region import RegionTrend

    assert {k: v for k, v in p.params.items() if k in fam.headline.config["primary"]["params"]} == {
        k: RegionTrend.DEFAULTS[k] for k in fam.headline.config["primary"]["params"]}  # fmt: skip
    from risk.profiles import RiskProfile, get_profile

    assert get_profile("loss_gate") == RiskProfile("loss_gate", daily_loss=0.02)
    labels = [v.label for v in fam.variants[1:]]
    assert labels == ["cadence15", "cadence5", "first0935", "exit1530", "vel_vwap_stop", "vel_flat_inside", "modwt",
                      "vol_target", "fill_1555", "stress_2x"]  # fmt: skip
    assert {s["label"] for s in fam.sample_splits} == {"iwm_dia", "2016_2019", "2020_2025"}
