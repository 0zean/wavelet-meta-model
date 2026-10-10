"""
U17 (SPEC §17): the rule pass (META_MODEL "none", stage-F rule cells), the family-test statistics (Ledoit–Wolf
Sharpe-difference test: size, power; the block bootstrap; Newey–West alpha; Holm; James–Stein), pooled streams,
floors, coherence, the family spec (budget, variants, registration in git), program budgets, and end to end: a planted
overnight drift passes F3's family test while a no-drift control fails, and the report renders.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import ledger as L
from experiments.spec import Cell, expand, normalize
from families import test as T
from families.run import BudgetError, check_budget, run_family
from families.spec import HEADLINE, RegistrationError, check_registered, load_family, parse, register
from families.stats import circular_block_indices, holm, james_stein, newey_west_alpha, sharpe_test
from primaries.mechanism import primary_config
from risk.portfolio import simulate_portfolio
from risk.profiles import get_profile
from tests.test_costs import const_quotes_table
from tests.test_u14 import intraday
from tests.test_u16 import signals_for
from utils.config import RunConfig
from wfo.rule_pass import SIGNAL_COLUMNS, rule_signals

NY = "America/New_York"


# ── Rule pass ────────────────────────────────────────────────────────────────


def test_a_stateless_rule_pass_is_the_fold_free_signal_of_every_event():
    df = intraday(40, seed=3)
    cfg = primary_config("overnight", "5Min", META_MODEL="none")
    sig = rule_signals(df, cfg, symbol="SPY")
    ref = signals_for(df, cfg, symbol="SPY")
    assert list(sig.columns) == SIGNAL_COLUMNS
    assert sig.index.equals(ref.index) and len(sig) > 30
    np.testing.assert_array_equal(sig["trade_signal"], ref["signed_dir"])
    np.testing.assert_array_equal(sig["bet_size"], ref["magnitude"])
    assert sig["meta_prob"].isna().all() and sig.attrs["live_start"] == sig.index[sig["signed_dir"] != 0][0]


def test_a_stateful_rule_pass_fits_each_segment_on_earlier_bars_only():
    """VOL_PROFILE 'tod': each segment's profile is fit on bars before it, so bars after a cut leave every event
    before the cut unchanged. Segments of TEST sessions start after RULE_WARMUP sessions and the last one runs to the
    data's end (58 sessions: 15 + 8 × 5 + a last one of 3); the stream starts at the first sided event."""
    kw = {"META_MODEL": "none", "VOL_PROFILE": "tod", "RULE_WARMUP": 15, "TEST": 5, "EMBARGO": 1}
    cfg = primary_config("intraday_momentum", "5Min", {"threshold_sigma": 0.0}, **kw)
    df = intraday(58, seed=4, u_shape=True)
    sig = rule_signals(df, cfg)
    days = df.index.normalize().unique()
    assert sig.attrs["n_segments"] == 9 and sig["fold"].max() == 9
    assert sig.index.normalize().unique()[0] == days[15] and sig.index.normalize().unique()[-1] == days[-1]
    assert sig.attrs["live_start"] == sig.index[sig["signed_dir"] != 0][0]
    cut = df.index[df.index.normalize() == days[40]][0]
    bumped = df.copy()
    after = bumped.index >= cut
    bumped.loc[after, ["open", "high", "low", "close"]] *= np.exp(
        np.random.default_rng(9).normal(0, 0.01, after.sum())
    )[:, None]
    sig2 = rule_signals(bumped, cfg)
    pd.testing.assert_frame_equal(sig.loc[: cut - pd.Timedelta("1min")], sig2.loc[: cut - pd.Timedelta("1min")])


def test_rule_pass_configuration_is_validated():
    with pytest.raises(ValueError, match="META_MODEL='none'"):
        rule_signals(intraday(5), primary_config("overnight", "5Min"))
    with pytest.raises(ValueError, match="fixed rule primary"):
        RunConfig.for_timeframe("5Min", META_MODEL="none", PRIMARY="ml_xgb")
    with pytest.raises(ValueError, match="SIZER must be 'fixed' or 'rule_size'"):
        primary_config("overnight", "5Min", META_MODEL="none", SIZER="linear")


def test_rule_cells_are_stage_f_only_and_family_cells_count_no_trials():
    from experiments.runner import cell_trials

    raw = {"symbols": "SPY", "timeframe": "5Min", "start": "2024-01-02", "end": "2024-06-01", "primary": "overnight",
           "model": {"meta": "none"}}  # fmt: skip
    assert cell_trials(Cell(normalize(raw), "F")) == 0
    with pytest.raises(ValueError, match="rule cells"):
        Cell(normalize(raw), "A")
    # U18 (F11): a model cell may be a family member (stage F) and then counts no trials; elsewhere it counts its grid
    model = {**raw, "model": {"meta": "logit_l2"}, "pwfo": {"is_grid": [504, 756], "oos_grid": [21]}}
    assert cell_trials(Cell(normalize(model), "F")) == 0 and cell_trials(Cell(normalize(model), "C")) == 2
    with pytest.raises(ValueError, match="family stage"):
        expand({"stage": "F", "defaults": raw, "grid": {"symbols": ["SPY"]}})


def test_vix_rules_read_the_feature_sets_prefixed_column():
    """U16 bug fixed here: vol_state emits `vol_state__vix`; the VIX rules read it (and a bare `vix`, hand-built X)."""
    df = intraday(3)
    cfg = primary_config("overnight", "5Min", {"vix_max": 25.0})
    from features.events import sample_events

    ev = sample_events(df, cfg).dropna(subset=["width"]).index
    for col in ("vol_state__vix", "vix"):
        X = pd.DataFrame({col: np.where(np.arange(len(ev)) == 0, 30.0, 20.0)}, index=ev)
        assert list(signals_for(df, cfg, X=X)["signed_dir"]) == [0.0] + [1.0] * (len(ev) - 1)


def test_portfolio_simulator_totals_traded_notional_and_costs_by_hand():
    """MOC entry at the 15:55 close (cost c_close) → MOO exit at the next 09:30 open (c_open): notional = q·(P_in +
    P_out), costs = q·(P_in·c_close + P_out·c_open)."""
    df = intraday(5, seed=1)
    cfg = primary_config("overnight", "5Min", META_MODEL="none", COST_MODEL="quotes")
    sig = signals_for(df, cfg)
    from risk.costs import fill_costs

    costs = fill_costs(df.index, cfg, const_quotes_table(["X"], auction={"open_auction": 0.3, "close_auction": 0.3}))
    eq, tr, _ = simulate_portfolio({"X": df}, {"X": sig}, cfg, get_profile("none"), side_col="trade_signal",
                                   size_col="bet_size", costs={"X": costs})  # fmt: skip
    assert len(tr) >= 2 and not tr["rolled"].any()
    notional = (tr["qty"] * (tr["entry_px"] + tr["exit_px"])).sum()
    paid = (tr["qty"] * (tr["entry_px"] * tr["entry_cost_bp"] + tr["exit_px"] * tr["exit_cost_bp"])).sum() * 1e-4
    assert (tr["entry_cost_bp"] == 0.3).all() and (tr["exit_cost_bp"] == 0.3).all()
    assert eq.attrs["traded_notional"] == pytest.approx(notional)
    assert eq.attrs["cost_paid"] == pytest.approx(paid)
    assert eq.iloc[-1] - cfg.INIT_CASH == pytest.approx(tr["pnl"].sum())


# ── Statistics ───────────────────────────────────────────────────────────────


def test_sharpe_difference_test_has_the_right_size_under_the_null():
    """Two independent noise streams with equal Sharpe ratios: 5 % ± 1.5 % rejections over 1,000 simulations."""
    rej = 0
    for i in range(1000):
        rng = np.random.default_rng(10**6 + i)
        a, b = rng.normal(3e-4, 0.01, 1000), rng.normal(3e-4, 0.01, 1000)
        rej += sharpe_test(a, b, n_boot=499, seed=i)["p"] < 0.05
    assert 0.035 <= rej / 1000 <= 0.065


def test_sharpe_difference_test_has_power_on_a_planted_gap():
    """A 0.3 annualized Sharpe gap between two streams correlated 0.95 over ten years is found in most samples."""
    hits = 0
    for i in range(100):
        rng = np.random.default_rng(5000 + i)
        common = rng.normal(0, 0.01, 2520)
        x = np.sqrt(0.95) * common + np.sqrt(0.05) * rng.normal(0, 0.01, 2520) + 0.3 / np.sqrt(252) * 0.01
        y = np.sqrt(0.95) * common + np.sqrt(0.05) * rng.normal(0, 0.01, 2520)
        r = sharpe_test(x, y, n_boot=499, seed=i)
        hits += r["p"] < 0.05 and r["delta_ann"] > 0
    assert hits / 100 >= 0.7


def test_one_sample_sharpe_test_rejects_a_planted_sharpe_and_not_noise():
    rng = np.random.default_rng(1)
    assert sharpe_test(rng.normal(0.01 * 1.5 / np.sqrt(252), 0.01, 2520), n_boot=499)["p"] < 0.01
    rej = sum(sharpe_test(np.random.default_rng(i).normal(0, 0.01, 750), n_boot=299, seed=i)["p"] < 0.05
              for i in range(200))  # fmt: skip
    assert rej / 200 <= 0.09


def test_the_circular_block_bootstrap_keeps_an_ar1s_autocorrelation():
    """AR(1) φ = 0.6: block-21 resamples keep the lag-1 autocorrelation (≈ φ·(1 − 1/b)) and the variance of the mean
    (≈ the long-run variance / n); single-observation blocks destroy both."""
    phi, n = 0.6, 4000
    rng = np.random.default_rng(0)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    lrv = 1 / (1 - phi) ** 2  # σ_e = 1
    for block, keeps in ((21, True), (1, False)):
        idx = circular_block_indices(n, block, 400, np.random.default_rng(1))
        xs = x[idx]
        ac = np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in xs])
        v = xs.mean(axis=1).var() * xs.shape[1]
        if keeps:
            assert ac == pytest.approx(phi * (1 - 1 / block), abs=0.05)
            assert v == pytest.approx(lrv, rel=0.3)
        else:
            assert abs(ac) < 0.05 and v < lrv / 3


def test_newey_west_alpha_recovers_a_planted_alpha():
    rng = np.random.default_rng(2)
    x = rng.normal(4e-4, 0.01, 2500)
    y = 2e-4 + 0.5 * x + rng.normal(0, 0.004, 2500)
    a = newey_west_alpha(y, x)
    se = a["alpha"] / a["alpha_t"]
    assert a["beta"] == pytest.approx(0.5, abs=0.02) and abs(a["alpha"] - 2e-4) < 3 * se
    assert a["alpha_p"] < 0.05 and a["alpha_ann"] == pytest.approx(a["alpha"] * 252)


def test_holm_on_a_planted_set():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_james_stein_shrinks_toward_the_mean_and_needs_four_instruments():
    sr = np.array([0.02, 0.05, 0.08, 0.11, -0.01, 0.06])
    js = james_stein(sr, 2000)
    assert js.mean() == pytest.approx(sr.mean())
    assert np.all(np.abs(js - sr.mean()) <= np.abs(sr - sr.mean()) + 1e-15)
    assert list(np.argsort(js)) == list(np.argsort(sr))
    np.testing.assert_array_equal(james_stein(sr[:3], 2000), sr[:3])


# ── Pooled streams, floors, coherence, verdict ───────────────────────────────


def _days(n, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def test_pooled_stream_is_the_equal_risk_mix_on_the_union_of_days():
    d = _days(6)
    bh = {"A": pd.Series([0.01, -0.01] * 3, index=d), "B": pd.Series([0.02, -0.02] * 3, index=d)}
    w = T.risk_weights(bh)
    assert w["A"] == pytest.approx(2 / 3) and w["B"] == pytest.approx(1 / 3)
    s = {"A": pd.Series(0.01, index=d[:4]), "B": pd.Series(0.03, index=d[2:])}
    p = T.pool(s, w)
    np.testing.assert_allclose(p.to_numpy(), [2 / 3 * 0.01] * 2 + [2 / 3 * 0.01 + 0.01] * 2 + [0.01] * 2)
    with pytest.raises(ValueError, match="differ"):
        T.pool({"A": s["A"]}, w)
    ew = T.benchmark("buy_and_hold_ew", bh, w)
    np.testing.assert_allclose(ew.to_numpy(), [0.015, -0.015] * 3)
    assert (T.benchmark("cash", bh, w) == 0).all()


def test_floors_by_hand():
    d = _days(504)
    r = pd.Series(np.r_[0.001, -0.0005].repeat(252), index=d)
    b = pd.Series(0.0004, index=d)
    costs = {"pnl": 100.0, "cost_paid": 20.0, "traded_notional": 1e5}
    f = T.floors(r, b, costs, {"min_net_ret": 0.05, "min_net_ret_vs_benchmark": 0.8, "min_edge_to_cost": 3.0,
                               "max_dd": 0.05})  # fmt: skip
    assert f["min_edge_to_cost"]["gross_edge_bp"] == pytest.approx(12.0)  # (100 + 20) / 1e5
    assert f["min_edge_to_cost"]["cost_bp"] == pytest.approx(2.0) and f["min_edge_to_cost"]["ok"]  # 12 ≥ 3 × 2
    assert not f["max_dd"]["ok"]  # 252 days of −5 bp: a 12 % drawdown
    assert f["min_net_ret"]["ok"] is (T.stream_stats(r)["ret_ann"] >= 0.05)
    assert not f["ok"]
    flat = T.floors(r, b, {"pnl": 0.0, "cost_paid": 0.0, "traded_notional": 0.0}, {"min_edge_to_cost": 3.0})
    assert not flat["min_edge_to_cost"]["ok"]  # no trades, no edge


def test_coherence_needs_the_share_and_a_median_variant_that_clears_the_floors():
    v = lambda d, ok=True: {"delta_ann": d, "floors_ok": ok}
    good = {HEADLINE: v(0.4), "a": v(0.3), "b": v(0.1), "c": v(-0.1)}
    c = T.coherence(good, 0.667)
    assert c["share"] == pytest.approx(2 / 3) and c["median_variant"] == "b" and c["coherent"]
    assert not T.coherence({**good, "b": v(0.1, ok=False)}, 0.667)["coherent"]  # the (lower) median fails a floor
    assert not T.coherence({**good, "a": v(-0.2)}, 0.667)["coherent"]  # 1 / 3 share the sign
    assert T.coherence({HEADLINE: v(0.4)}, 0.667)["coherent"]  # no variants: vacuous


def test_program_verdict_applies_holm_floors_coherence_and_sign():
    fam = lambda p, d=0.3, fl=True, c=True: {"p": p, "delta_ann": d, "floors_ok": fl, "coherence": {"coherent": c},
                                             "positive": d > 0}  # fmt: skip
    t = T.program_verdict({"F1": fam(0.001), "F2": fam(0.02), "F3": fam(0.03), "F4": fam(0.0001, d=-0.3),
                           "F5": fam(0.0002, fl=False)})  # fmt: skip
    t = t.set_index("family")
    np.testing.assert_allclose(t["p_holm"], holm([0.001, 0.02, 0.03, 0.0001, 0.0002]))
    # Holm: 0.0001·5, 0.0002·4, 0.001·3, 0.02·2, max(0.04, 0.03·1) → every p_holm < 0.05; F4 is negative, F5 fails a
    # floor
    assert t["passes"].to_dict() == {"F1": True, "F2": True, "F3": True, "F4": False, "F5": False}
    assert (
        not T.program_verdict({"F1": fam(0.001), "F2": fam(0.04), "F3": fam(0.03)})
        .set_index("family")
        .loc["F2", "passes"]
    )  # 0.04 · 2 = 0.08


# ── Family spec, registration, budgets ───────────────────────────────────────

SPEC = {
    "id": "F3_test",
    "mechanism": "Index returns accrue overnight.",
    "instruments": ["SPY", "QQQ"],
    "timeframe": "5Min",
    "window": {"start": "2022-01-03", "end": "2025-01-01"},
    "headline": {"primary": {"name": "overnight"}, "cost_model": "quotes", "risk_profile": "none"},
    "variants": [{"label": "slippage", "cost_model": "slippage"},
                 {"label": "exit_0935", "exit.params": {"exit_time": "09:35", "exit_session": 1}}],
    "state_splits": ["macro_day", "abs_move_tercile", "prior_day_sign", "vix_tercile"],
    "response": ["sharpe", "max_dd", "mean_per_trade_bp", "hit_rate", "exposure"],
    "sample_splits": [{"label": "first_year", "start": "2022-01-03", "end": "2023-01-01"},
                      {"label": "iwm", "instruments": ["IWM"]}],
    "benchmark": "buy_and_hold_ew",
    "floors": {"min_net_ret_vs_benchmark": 0.7, "min_edge_to_cost": 3.0},  # overnight ≈ 0.8 × B&H here
    "test": {"n_boot": 499},
    "TRIAL_BUDGET": 8,
}  # fmt: skip


def test_variants_are_dotted_changes_to_the_headline_and_build_valid_cells():
    fam = parse(SPEC)
    assert [v.label for v in fam.variants] == [HEADLINE, "slippage", "exit_0935"]
    assert [len(fam.cells[v.label]) for v in fam.variants] == [2, 2, 2]
    c = fam.cells["exit_0935"][0].config()
    assert c.EXIT_PARAMS["exit_time"] == "09:35" and c.EXIT_PARAMS["exit_session"] == 1 and c.META_MODEL == "none"
    assert fam.cells["slippage"][0].config().COST_MODEL == "slippage"
    assert all(cell.stage == "F" and cell.is_rule for cs in fam.cells.values() for cell in cs)
    basket = parse({**{k: v for k, v in SPEC.items() if k != "instruments"}, "basket": {"symbols": ["SPY", "QQQ"]}})
    assert [cell.symbols for cell in basket.cells[HEADLINE]] == [("QQQ", "SPY")]


@pytest.mark.parametrize(
    "change, match",
    [
        ({"TRIAL_BUDGET": 2}, "exceed TRIAL_BUDGET"),
        ({"instruments": ["SPY", "spy"]}, "duplicate instruments"),
        ({"variants": [{"label": 2016, "cost_model": "slippage"}]}, "string label"),
        ({"variants": [{"label": "x", "instruments": ["IWM"]}]}, "sample split"),
        ({"variants": [{"label": "x", "overrides.COST_MODEL": "cs"}]}, "cost_model"),
        ({"headline": {"primary": "sma_cross"}}, "mechanism primary"),
        ({"benchmark": "spy"}, "benchmark"),
        ({"window": {"end": "2026-01-01"}}, "development window"),
        ({"state_splits": ["moon_phase"]}, "state split"),
        ({"floors": {"min_sharpe": 1}}, "floor"),
    ],
)
def test_invalid_family_specs_are_refused(change, match):
    with pytest.raises((ValueError, TypeError), match=match):
        parse({**SPEC, **change})


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "families").mkdir(parents=True)
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "commit", "-q", "--allow-empty", "-m", "root")
    return r


def _write(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_registration_is_refused_until_the_spec_is_committed_unchanged(repo):
    p = _write(repo / "families" / "F3_test.yaml", SPEC)
    with pytest.raises(RegistrationError, match="not committed"):
        register(p, repo)
    with pytest.raises(RegistrationError, match="not registered"):
        check_registered(p, repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "spec")
    reg = register(p, repo, today="2026-10-10")
    assert load_family(p).doc["registered"]["sha"] == reg["sha"]
    with pytest.raises(RegistrationError, match="uncommitted"):
        check_registered(p, repo)  # the registered line is not committed yet
    _git(repo, "commit", "-qam", "register")
    assert check_registered(p, repo)["sha"] == reg["sha"]
    with pytest.raises(RegistrationError, match="already registered"):
        register(p, repo)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    doc["floors"]["min_edge_to_cost"] = 1.0  # a post-registration change, even committed, is refused
    _write(p, doc)
    with pytest.raises(RegistrationError, match="uncommitted"):
        check_registered(p, repo)
    _git(repo, "commit", "-qam", "loosen a floor")
    with pytest.raises(RegistrationError, match="differs from its registered version"):
        check_registered(p, repo)
    doc["registered"]["sha"] = "0" * 40
    _write(p, doc)
    _git(repo, "commit", "-qam", "forge")
    with pytest.raises(RegistrationError, match="not an ancestor"):
        check_registered(p, repo)


def _vrow(fam, key, sha="abc", status="ok", sharpe=0.5, budget=None):
    base = fam.split(".v")[0]
    return {"stage": "F", "status": status, "kind": "family_variant", "cell_hash": f"{fam}-{key}-{sha}",
            "family": fam, "base_family": base, "trial_key": key if "/" in key else f"{base}/{key}",
            "registered_sha": sha, "n_trials": 1, "sharpe": sharpe, "budget": budget}  # fmt: skip


def test_budgets_count_configurations_per_family_and_program():
    fam = parse(SPEC)
    fam.registered = {"sha": "abc"}
    caps = {"max_families": 8, "max_trials": 112}
    acc = check_budget(fam, [_vrow("F3_test", fam.trial_key(HEADLINE))], caps)
    assert acc["family_trials"] == 3 and acc["program_trials"] == 3  # a re-run of the same configuration is not new
    amended = parse({**SPEC, "id": "F3_test.v2", "variants": SPEC["variants"] + [{"label": "v2", "cost_model": "cs"}]})
    amended.registered = {"sha": "def"}
    assert amended.trial_key(HEADLINE) == fam.trial_key(HEADLINE)  # same configuration, same trial
    rows = [_vrow("F3_test", k) for k in ("a", "b", "c", "d", "e")]
    with pytest.raises(BudgetError, match="TRIAL_BUDGET 8"):  # 5 earlier + 4 new keys of the same family
        check_budget(amended, rows, caps)
    others = [_vrow(f"G{i}", f"v{j}") for i in range(7) for j in range(16)]  # 112 trials in 7 families
    with pytest.raises(BudgetError, match="program trials"):
        check_budget(fam, others, caps)
    with pytest.raises(BudgetError, match="families exceed"):
        check_budget(fam, [_vrow(f"G{i}", "h") for i in range(8)], caps)
    with pytest.raises(BudgetError, match="another registered sha|registered sha"):
        check_budget(fam, [_vrow("F3_test", HEADLINE, sha="zzz")], caps)
    errors = [_vrow("F3_test", k, status="error") for k in "abcdefgh"]  # errors are not counted trials
    assert check_budget(fam, errors, caps)["family_trials"] == 3


def test_an_amendment_that_reuses_labels_with_new_configurations_costs_new_trials():
    """Review B1: trial keys identify the configuration, not the label, so a stream of amendments that keep the labels
    but change the headline and the variants consumes the family budget; and no amendment can raise it."""
    caps = {"max_families": 8, "max_trials": 112}
    rows = []
    for k, vix in enumerate((12.0, 13.0, 14.0)):
        head = {**SPEC["headline"], "primary": {"name": "overnight", "params": {"vix_max": vix}}}
        fam = parse({**SPEC, "id": f"F3_test.v{k + 2}", "headline": head})
        fam.registered = {"sha": f"s{k}"}
        if k == 2:
            with pytest.raises(BudgetError, match="exceed TRIAL_BUDGET 8"):  # 3 configurations x 3 versions = 9 > 8
                check_budget(fam, rows, caps)
            break
        check_budget(fam, rows, caps)
        rows += [_vrow(fam.id, fam.trial_key(v.label), sha=f"s{k}", budget=fam.budget) for v in fam.variants]
    roomy = parse({**SPEC, "id": "F3_test.v9", "TRIAL_BUDGET": 12})
    roomy.registered = {"sha": "s9"}
    assert check_budget(roomy, rows[:3], caps)["family_budget"] == 8  # the earlier versions' budget still binds
    with pytest.raises(ValueError, match="TRIAL_BUDGET must be"):
        parse({**SPEC, "TRIAL_BUDGET": 40})


def test_floors_default_to_the_plans_and_cannot_be_switched_off():
    """Review S1: a spec without floors gets PLAN2's (long-only: 0.8 x benchmark; else 2 %/yr; edge-to-cost 3x)."""
    long_only = parse({k: v for k, v in SPEC.items() if k != "floors"})
    assert long_only.floors["min_net_ret_vs_benchmark"] == 0.8 and long_only.floors["min_net_ret"] is None
    assert long_only.floors["min_edge_to_cost"] == 3.0
    rest = {k: v for k, v in SPEC.items() if k not in ("floors", "variants")}
    sided = parse({**rest, "headline": {"primary": "intraday_momentum"}})
    assert sided.floors["min_net_ret"] == 0.02 and sided.floors["min_net_ret_vs_benchmark"] is None
    with pytest.raises(ValueError, match="net-return floor"):
        parse({**SPEC, "floors": {"min_net_ret_vs_benchmark": None}})
    with pytest.raises(ValueError, match="min_edge_to_cost"):
        parse({**SPEC, "floors": {"min_edge_to_cost": None}})
    d = _days(300)
    flat = pd.Series(0.0, index=d)
    assert not T.floors(flat, flat, {"pnl": 0, "cost_paid": 0, "traded_notional": 0}, {})["ok"]


def test_no_op_and_duplicate_variants_are_refused():
    """Review M1: a variant that runs the headline's configuration (or another variant's) would pad coherence."""
    for v in (
        {"label": "same", "cost_model": "quotes"},
        {"label": "same", "risk_profile": "none"},
        {"label": "dup", "cost_model": "slippage"},
    ):
        with pytest.raises(ValueError, match="same configuration"):
            parse({**SPEC, "variants": SPEC["variants"] + [v]})


def test_family_stage_cells_run_only_through_the_family_runner(tmp_path):
    from experiments.runner import run

    raw = {"symbols": "SPY", "timeframe": "5Min", "start": "2024-01-02", "end": "2024-06-01", "primary": "overnight",
           "model": {"meta": "none"}}  # fmt: skip
    with pytest.raises(ValueError, match="family stages"):
        run([Cell(normalize(raw), "F")], ledger=L.Ledger(tmp_path / "l.jsonl"), root=tmp_path)
    assert not (tmp_path / "l.jsonl").exists()


def test_coherence_compares_variants_on_the_headlines_days():
    """Review S2: a variant that starts later is compared with the benchmark over the days it shares with the
    headline."""
    d = _days(400)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(5e-4, 0.01, 400), index=d)
    head = bench * 0.5 + rng.normal(2e-4, 0.003, 400)
    late = head.iloc[200:] * 1.0
    dc = T.delta_common(late, head.iloc[100:], bench, "buy_and_hold_ew")
    j = slice(d[200], d[-1])
    assert dc == pytest.approx(T.sharpe_ann(late.loc[j]) - T.sharpe_ann(bench.loc[j]))


# ── End to end on synthetic bars ─────────────────────────────────────────────


class OvernightSource:
    """Synthetic 5Min sessions 2021-06 → 2026-10 per symbol (deterministic, sliced by [start, end)): overnight gaps
    N(drift, 0.4 %), zero-drift intraday random walk with 1 % daily σ; flat-ish small auction costs."""

    START, END = "2021-06-01", "2026-10-01"

    def __init__(self, drift: float):
        self.drift, self._cache = drift, {}

    def _full(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            rng = np.random.default_rng(sum(map(ord, symbol)) + int(self.drift * 1e6))
            days = pd.bdate_range(self.START, self.END, inclusive="left")
            idx = pd.DatetimeIndex([t for d in days for t in pd.date_range(d + pd.Timedelta("09:30:00"), periods=78,
                                                                              freq="5min")]).tz_localize(NY)  # fmt: skip
            r = rng.normal(0, 0.01 / np.sqrt(78), len(idx))
            first = np.r_[True, idx.normalize()[1:] != idx.normalize()[:-1]]
            gap = np.where(first, rng.normal(self.drift, 0.004, len(idx)), 0.0)
            gap[0] = 0.0
            c = 100 * np.exp(np.cumsum(r + gap))
            o = c * np.exp(-r)  # the bar opens after its gap, before its intraday move
            self._cache[symbol] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0002,
                                                "low": np.minimum(o, c) * 0.9998, "close": c, "volume": 1e5},
                                               index=idx)  # fmt: skip
        return self._cache[symbol]

    def bars(self, symbol, timeframe, start, end, *, allow_holdout):
        assert timeframe == "5Min"
        df = self._full(symbol)
        day = df.index.tz_localize(None).normalize()
        return df[(day >= pd.Timestamp(start)) & (day < pd.Timestamp(end))]

    def sessions(self, end):
        return pd.bdate_range(self.START, end, inclusive="left")

    def quotes_table(self, symbols):
        return const_quotes_table(symbols, regular=lambda b: 0.5, auction={"open_auction": 0.3, "close_auction": 0.3})

    def exo(self, *a, **k):
        raise KeyError("no exo series in the synthetic source")


def _run(tmp_path, drift, repo=None, path=None):
    src = OvernightSource(drift)
    prog = tmp_path / "program.yaml"
    prog.write_text("max_families: 8\nmax_trials: 112\n", encoding="utf-8")
    ledger = L.Ledger(tmp_path / f"ledger_{drift}.jsonl")
    if path is None:
        path = _write(tmp_path / "F3_test.yaml", SPEC)
    res = run_family(path, ledger=ledger, root=tmp_path / "root", out_dir=tmp_path / f"out_{drift}", source=src,
                     repo=repo or tmp_path, check_registration=repo is not None, program=prog, quasi=False)  # fmt: skip
    return res, ledger


def test_a_planted_overnight_drift_passes_f3s_family_test_and_a_no_drift_control_fails(tmp_path, repo):
    p = _write(repo / "families" / "F3_test.yaml", SPEC)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "spec")
    register(p, repo, today="2026-10-10")
    _git(repo, "commit", "-qam", "register")
    res, ledger = _run(tmp_path, 1.5e-3, repo=repo, path=p)
    v = res["evaluation"]["verdict"]
    h = res["evaluation"]["variants"][HEADLINE]
    assert h["compare"]["delta_ann"] > 1.0 and v["p"] < 0.01
    assert v["floors_ok"] and v["coherence"]["coherent"] and v["positive"]
    assert h["floors"]["min_edge_to_cost"]["value"] > 3.0
    rows = ledger.rows()
    variant_rows = [r for r in rows if r.get("kind") == "family_variant"]
    assert len(variant_rows) == 3 and all(r["n_trials"] == 1 and r["registered_sha"] for r in variant_rows)
    assert L.n_trials(rows, "F") == 3  # 6 + 2 member cells count 0: a variant is one trial whatever its instruments
    assert {r["kind"] for r in rows if r.get("kind") != "family_variant"} == {"rule"}
    # the report renders; the registered-vs-run diff is the registered line only
    md = Path(res["paths"]["md"]).read_text(encoding="utf-8")
    assert "## Headline test" in md and "Specification curve" in md and "macro_day" in md
    assert (Path(res["paths"]["md"]).parent / "spec_curve.png").exists()
    assert "-floors" not in md and "+registered" in md
    assert res["described"]["sample_splits"]["iwm"]["n_days"] > 700
    assert res["described"]["state_splits"]["vix_tercile"] == {"unavailable": True}
    # a second run is a no-op for the ledger (same hashes)
    n = len(rows)
    _run(tmp_path, 1.5e-3, repo=repo, path=p)
    assert len(ledger.rows()) == n

    ctrl, _ = _run(tmp_path, 0.0)
    cv = ctrl["evaluation"]["verdict"]
    table = T.program_verdict({"F3_drift": v, "F3_control": cv})
    assert table.set_index("family")["passes"].to_dict() == {"F3_drift": True, "F3_control": False}
    from families.report import program_summary

    summary = program_summary(tmp_path / "out_0.0")
    assert "F3_test" in summary["md"].read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out_0.0" / "F3_test" / "result.json").read_text())["id"] == "F3_test"
