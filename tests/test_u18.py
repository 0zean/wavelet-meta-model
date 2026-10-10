"""
U18 (PLAN2 Phase 1) tooling: the union of calendar windows (`calendar_drift` with a window list, the schedule's
`windows`), the `basket` risk profile, per-instrument headline patches and per-variant timeframes in family specs, and
a basket family's instrument sample split.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import ledger as L
from families.run import bh_returns, run_family
from families.spec import HEADLINE, parse
from features.exits import exit_frame
from primaries.mechanism import primary_config
from risk.portfolio import simulate_portfolio
from risk.profiles import RiskProfile, get_profile
from tests.test_costs import const_quotes_table
from tests.test_u16 import daily, signals_for
from tests.test_u17 import _write
from utils.config import RunConfig

NY = "America/New_York"


# ── calendar_drift: the union of windows ─────────────────────────────────────


def _held_sessions(df: pd.DataFrame, cfg: RunConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Close-marked equity of a 1Day calendar cell (profile none, slippage costs) and its trades."""
    sig = signals_for(df, cfg)
    eq, trades, _ = simulate_portfolio({"_": df}, {"_": sig}, cfg, get_profile("none"))
    return eq, trades


def _held_days(df: pd.DataFrame, trades: pd.DataFrame) -> set:
    """Sessions whose close-to-close return the position earns: (entry session, exit session]."""
    out = set()
    for _, t in trades.iterrows():
        out |= {d.strftime("%Y-%m-%d") for d in df.index[int(t["entry_b"]) + 1 : int(t["exit_b"]) + 1]}
    return out


def test_a_union_that_adds_a_contained_window_reproduces_the_single_window_exactly():
    d = daily(700, start="2023-01-03")
    single = primary_config("calendar_drift", "1Day", {"window": "tom"})
    union = primary_config("calendar_drift", "1Day", {"window": ["tom", "month_end"]})  # month_end's −1 ⊂ tom's
    assert union.EVENT_PARAMS["windows"] == [{"days": "month_end", "day_offset": 1, "hold": 4},
                                             {"days": "month_end", "day_offset": 1, "hold": 1}]  # fmt: skip
    assert union.EXIT_PARAMS == {"exit_time": "close", "exit_session": 1}
    eq1, tr1 = _held_sessions(d, single)
    eq2, tr2 = _held_sessions(d, union)
    pd.testing.assert_series_equal(eq1, eq2)  # one entry, rolled through the window untraded, one exit
    assert _held_days(d, tr1) == _held_days(d, tr2)
    assert tr2["rolled"].any() and len(tr1) < len(tr2)


def test_a_union_holds_each_session_any_window_holds_once():
    d = daily(330, start="2023-09-01")
    union = primary_config("calendar_drift", "1Day", {"window": ["fomc_pre", "tom"]})
    _, tr = _held_sessions(d, union)
    held = _held_days(d, tr)
    # January 2024: FOMC 01-31 (fomc_pre holds 01-31) and the turn of the month (−1 = 01-31, +1 … +3 = 02-01 … 02-05)
    jan = {x for x in held if "2024-01-15" <= x <= "2024-02-10"}
    assert jan == {"2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"}
    # June 2024: FOMC 06-12 alone, then the turn of the month (−1 = 06-28, +1 … +3 = 07-01 … 07-03)
    jun = {x for x in held if "2024-06-06" <= x <= "2024-07-10"}  # after May's +3 (06-05)
    assert jun == {"2024-06-12", "2024-06-28", "2024-07-01", "2024-07-02", "2024-07-03"}
    for w in ("fomc_pre", "tom"):  # the union's held set is the union of the single windows' held sets
        _, t = _held_sessions(d, primary_config("calendar_drift", "1Day", {"window": w}))
        assert _held_days(d, t) <= held
    assert (tr["frac"] <= 1.0 + 1e-12).all()  # overlapping windows never stack


def test_union_windows_are_validated():
    with pytest.raises(ValueError, match="two or more distinct"):
        primary_config("calendar_drift", "1Day", {"window": ["tom"]})
    with pytest.raises(ValueError, match="two or more distinct"):
        primary_config("calendar_drift", "1Day", {"window": ["tom", "tom"]})
    with pytest.raises(ValueError, match="ends at 14:00"):
        primary_config("calendar_drift", "5Min", {"window": ["fomc_pre", "tom"]})
    with pytest.raises(ValueError, match="same sessions"):  # on 1Day both end at the FOMC day's close
        primary_config("calendar_drift", "1Day", {"window": ["fomc_pre", "fomc_day"]})
    with pytest.raises(ValueError, match="window must be one of"):
        primary_config("calendar_drift", "1Day", {"window": ["tom", "xmas"]})
    ok = primary_config("calendar_drift", "5Min", {"window": ["fomc_day", "tom"]})
    assert ok.EVENT_PARAMS["entry_times"] == ["close"]
    base = {"entry_times": ["close"], "windows": [{"days": "fomc", "day_offset": 1, "hold": 1},
                                                  {"days": "month_end", "day_offset": 1, "hold": 4}]}  # fmt: skip
    ex = {"EVENT_SAMPLER": "schedule", "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "close", "exit_session": 1}}
    RunConfig.for_timeframe("1Day", EVENT_PARAMS=base, **ex)
    for change, match in [
        ({"every": "week"}, "every 'session'"),
        ({"days": "fomc"}, "days 'all'"),
        ({"windows": base["windows"][:1]}, "at least two"),
        ({"windows": [{"days": "earnings", "day_offset": 0, "hold": 1}, base["windows"][1]]}, "window days"),
        ({"windows": [{"days": "fomc", "day_offset": 1, "hold": 0}, base["windows"][1]]}, "hold must be"),
        ({"windows": [{"days": "fomc", "day_offset": 1}, base["windows"][1]]}, "days, day_offset, hold"),
        ({"windows": [base["windows"][0], base["windows"][0]]}, "duplicate window"),
    ]:
        with pytest.raises(ValueError, match=match):
            RunConfig.for_timeframe("1Day", EVENT_PARAMS={**base, **change}, **ex)


def test_a_union_entry_needs_its_window_known_at_that_windows_decision():
    # the unscheduled 2020-03-03 statement (announced that morning) never selects its eve, alone or in a union
    d = daily(400, start="2019-06-03")
    for w in ("fomc_pre", ["fomc_pre", "opex_week"]):
        cfg = primary_config("calendar_drift", "1Day", {"window": w})
        sig = signals_for(d, cfg)
        lab = exit_frame(d, sig.index, sig["width"], cfg)
        exits = {d.index[p].strftime("%Y-%m-%d") for p in lab["exit_pos"]}
        assert "2020-03-03" not in exits and "2020-01-29" in exits


# ── The basket risk profile ──────────────────────────────────────────────────


def test_basket_profile_scales_entries_to_gross_one_and_charges_borrow_on_shorts():
    p = get_profile("basket")
    assert (p.max_gross, p.max_net, p.borrow_bps, p.vol_target, p.max_concurrent, p.dd_tiers, p.daily_loss) == (
        1.0, 1.0, 50.0, None, None, (), None)  # fmt: skip
    bars = {s: daily(60, seed=i) for i, s in enumerate("ABC")}
    cfg = RunConfig.for_timeframe("1Day", POSITION_MODE="single", VERTICAL_BARS=20)
    pos, sides = 5, {"A": 1, "B": 1, "C": -1}
    sig = {s: pd.DataFrame({"trade_signal": [sides[s]], "bet_size": [1.0], "width": [0.05]},
                           index=bars[s].index[[pos]]) for s in bars}  # fmt: skip
    _, tr, log = simulate_portfolio(bars, sig, cfg, p)
    assert np.allclose(tr["frac"], 1 / 3)  # three full-size bets share the gross cap pro-rata
    assert log["gross"].iloc[tr["entry_b"].iloc[0]] == pytest.approx(1.0)  # at the entries' open (marks drift later)
    no_borrow = RiskProfile("no_borrow", max_gross=1.0, max_net=1.0)
    _, tr0, _ = simulate_portfolio(bars, sig, cfg, no_borrow)
    short, short0 = tr.set_index("sym").loc["C"], tr0.set_index("sym").loc["C"]
    held = int(short["exit_b"] - short["entry_b"]) + 1
    borrow = short["qty"] * short["entry_fill"] * 50e-4 * held / cfg.bars_per_year  # 50 bp/yr on the entry notional
    assert short0["pnl"] - short["pnl"] == pytest.approx(borrow, rel=0.02)
    assert tr.set_index("sym").loc["A", "pnl"] == tr0.set_index("sym").loc["A", "pnl"]


# ── Family specs: per-instrument patches, per-variant timeframes ─────────────

F4_LIKE = {
    "id": "F4_like",
    "mechanism": "Scheduled flows and information.",
    "instruments": ["SPY", "TLT"],
    "timeframe": "1Day",
    "headline": {"primary": {"name": "calendar_drift", "params": {"window": ["fomc_pre", "tom"]}},
                 "per_instrument": {"TLT": {"primary": {"params": {"window": "month_end"}}}}},
    "variants": [{"label": "tom_2_2", "primary.params.window": ["fomc_pre", "tom_2_2"]},
                 {"label": "fomc_1400", "timeframe": "5Min", "primary.params.window": "fomc_pre"}],
    "sample_splits": [{"label": "qqq_iwm", "instruments": ["QQQ", "IWM"]}],
    "floors": {"min_net_ret": 0.02},
}  # fmt: skip


def test_per_instrument_patches_and_variant_timeframes_build_the_right_cells():
    fam = parse(F4_LIKE)
    by = {c.symbols[0]: c.config() for c in fam.cells[HEADLINE]}
    assert by["SPY"].PRIMARY_PARAMS["window"] == ["fomc_pre", "tom"] and "windows" in by["SPY"].EVENT_PARAMS
    assert by["TLT"].PRIMARY_PARAMS["window"] == "month_end" and by["TLT"].EVENT_PARAMS["days"] == "month_end"
    tom22 = {c.symbols[0]: c.config() for c in fam.cells["tom_2_2"]}
    assert tom22["SPY"].PRIMARY_PARAMS["window"] == ["fomc_pre", "tom_2_2"]
    assert tom22["TLT"].PRIMARY_PARAMS["window"] == "month_end"  # the patch still applies to TLT
    f14 = {c.symbols[0]: c.config() for c in fam.cells["fomc_1400"]}
    assert f14["SPY"].TIMEFRAME == "5Min" and f14["SPY"].EXIT_PARAMS["exit_time"] == "14:00"
    assert f14["TLT"].TIMEFRAME == "5Min" and f14["TLT"].EXIT_PARAMS["exit_time"] == "close"
    assert all(c.config().TIMEFRAME == "1Day" for c in fam.cells[HEADLINE])
    assert fam.trial_key("tom_2_2") != fam.trial_key(HEADLINE)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"headline": {**F4_LIKE["headline"], "per_instrument": {"GLD": {"sizer": "fixed"}}}}, "not an instrument"),
        ({"headline": {**F4_LIKE["headline"], "per_instrument": {"TLT": {"timeframe": "5Min"}}}}, "partial headline"),
        ({"headline": {**F4_LIKE["headline"], "per_instrument": {"TLT": {"colour": 1}}}}, "partial headline"),
        ({"variants": [{"label": "x", "per_instrument.TLT.primary.params.window": "tom"},
                       {"label": "y", "per_instrument.TLT.primary.params.window": "tom"}]}, "same configuration"),
    ],
)  # fmt: skip
def test_bad_per_instrument_patches_are_refused(change, match):
    with pytest.raises((ValueError, TypeError), match=match):
        parse({**F4_LIKE, **change})


def test_a_basket_takes_no_per_instrument_patch():
    doc = {**{k: v for k, v in F4_LIKE.items() if k not in ("instruments", "variants", "sample_splits")},
           "basket": {"symbols": ["SPY", "TLT"]}}  # fmt: skip
    with pytest.raises(ValueError, match="basket"):
        parse(doc)


# ── A basket family's instrument sample split (end to end, synthetic daily bars) ──


class TrendSource:
    """Synthetic daily sessions 2018-06 → 2026-10 per symbol (deterministic): AR(1)-drifting random walks, so tsmom
    takes both sides; constant auction costs."""

    START, END = "2018-06-01", "2026-10-01"

    def __init__(self):
        self._cache = {}

    def _full(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            rng = np.random.default_rng(sum(map(ord, symbol)))
            days = pd.bdate_range(self.START, self.END, inclusive="left").tz_localize(NY)
            mu = np.zeros(len(days))
            for i in range(1, len(days)):
                mu[i] = 0.995 * mu[i - 1] + rng.normal(0, 4e-5)
            c = 100 * np.exp(np.cumsum(mu + rng.normal(0, 0.01, len(days))))
            o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.002, len(days)))
            self._cache[symbol] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002,
                                                "low": np.minimum(o, c) * 0.998, "close": c, "volume": 1e6},
                                               index=days)  # fmt: skip
        return self._cache[symbol]

    def bars(self, symbol, timeframe, start, end, *, allow_holdout):
        assert timeframe == "1Day"
        df = self._full(symbol)
        day = df.index.tz_localize(None)
        return df[(day >= pd.Timestamp(start)) & (day < pd.Timestamp(end))]

    def sessions(self, end):
        return pd.bdate_range(self.START, end, inclusive="left")

    def quotes_table(self, symbols):
        return const_quotes_table(symbols, regular=lambda b: 0.5, auction={"open_auction": 0.3, "close_auction": 0.3})

    def exo(self, *a, **k):
        raise KeyError("no exo series in the synthetic source")


def test_a_basket_familys_instrument_split_is_its_own_portfolio_cell_and_benchmark(tmp_path):
    doc = {
        "id": "F2_like",
        "mechanism": "Trend continuation across assets.",
        "basket": {"symbols": ["AAA", "BBB", "CCC"]},
        "timeframe": "1Day",
        "window": {"start": "2020-01-02"},
        "headline": {"primary": {"name": "tsmom", "params": {"lookback": 126}}, "risk_profile": "basket"},
        "variants": [{"label": "long_only", "primary.params.long_only": True}],
        "sample_splits": [{"label": "two", "instruments": ["AAA", "BBB"]}],
        "test": {"n_boot": 199},
    }
    path = _write(tmp_path / "F2_like.yaml", doc)
    prog = tmp_path / "program.yaml"
    prog.write_text("max_families: 8\nmax_trials: 112\n", encoding="utf-8")
    ledger = L.Ledger(tmp_path / "ledger.jsonl")
    res = run_family(path, ledger=ledger, root=tmp_path / "root", out_dir=tmp_path / "out", source=TrendSource(),
                     repo=tmp_path, check_registration=False, program=prog, quasi=False)  # fmt: skip
    split = res["described"]["sample_splits"]["two"]
    assert split["n_days"] > 1000
    head = res["evaluation"]["variants"][HEADLINE]
    assert split["sharpe"] != pytest.approx(head["stats"]["sharpe"])  # its own cell, not the headline's
    # its benchmark is buy-and-hold of both split symbols (before U18: of the first symbol only)
    from families import test as T

    bh = bh_returns(TrendSource(), ["AAA", "BBB"], "1Day", "2020-01-02", "2025-10-01")
    ew = T.benchmark("buy_and_hold_ew", bh, T.risk_weights(bh)).iloc[-split["n_days"] :]
    assert split["bench_sharpe"] == pytest.approx(T.sharpe_ann(ew))
    assert split["bench_sharpe"] != pytest.approx(T.sharpe_ann(T.day_index(bh["AAA"]).iloc[-split["n_days"] :]))
