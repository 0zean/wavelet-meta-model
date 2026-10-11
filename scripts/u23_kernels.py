"""
U23 done-when measurements (PLAN3 §5 U23, §6): the kernel oracles and the compute budgets on the cached SPY / QQQ
development window (2016-01-04 → 2025-09-30). Nothing here reads a performance statistic: the budgets time the
computations and print shapes, counts and parity residuals only, so no look is taken (PLAN3 §5: looks are counted
before a result is read; U24 takes the first one).

    uv run python scripts/u23_kernels.py parity       # Meyers worked examples, 4,400 random (N, t) pairs vs scipy
    uv run python scripts/u23_kernels.py budgets      # estimators, region positions, specification curve, headline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NS = [6, 9, 12, 18, 24]
START, END = "2016-01-01", "2025-10-01"


def parity() -> None:
    from scipy.stats import siegelslopes

    from features.kernels import rmedv_all, session_layout, sg_velocity_all
    from tests.test_u14 import intraday

    for label, y in (("Meyers 2005 p.2", [1, 2, 3, 4, 5, 15, 12, 8, 9, 10]),
                     ("Meyers 2025 p.2", [1, 2, 10, 4, 5, 6, 7, 8, 9, 18, 11, 12, 13, 18, 15, 20])):  # fmt: skip
        y = np.asarray(y, dtype=float)
        print(f"{label}: repeated median slope = {rmedv_all(y, np.zeros(len(y), np.int64), [len(y)])[0, -1]!r}")
    df = intraday(60, seed=1)
    _, start, _, _ = session_layout(df.index, 5)
    x = np.log(df["close"].to_numpy())
    rm, sg = rmedv_all(x, start, NS), sg_velocity_all(x, start, NS, 1)
    rng = np.random.default_rng(2)
    worst_rm, worst_sg, exact, n = 0.0, 0.0, 0, 0
    for a, k in enumerate(NS):
        ok = np.flatnonzero(np.arange(len(x)) - start >= k - 1)
        for t in rng.choice(ok, 880, replace=False):
            w = x[t - k + 1 : t + 1]
            d = abs(rm[a, t] - siegelslopes(w, np.arange(k, dtype=float)).slope)
            worst_rm, exact = max(worst_rm, d), exact + (d == 0.0)
            worst_sg = max(worst_sg, abs(sg[a, t] - np.polyfit(np.arange(k), w, 1)[0]))
            n += 1
    print(
        f"{n} random (N, t) pairs, N in {NS}: max |rmedv - scipy siegelslopes| = {worst_rm:.3g} ({exact} bit-equal);"
        f" max |sgv(degree 1) - polyfit slope| = {worst_sg:.3g}"
    )


def _clock(days: pd.DatetimeIndex):
    try:
        from data.bars import get_calendar

        cal = pd.DatetimeIndex(get_calendar(END).index).normalize()
        return cal[(cal >= days[0]) & (cal <= days[-1])]
    except Exception as e:  # noqa: BLE001 — the data clock then (no network)
        print(f"  (calendar unavailable: {type(e).__name__}; the data's sessions are the clock)")
        return None


def budgets() -> None:
    from data.bars import load_bars, load_prints
    from data.exo import load_series
    from data.quotes import read_asof_table
    from experiments.runner import rate_asof
    from families.stats import mean_test
    from features.kernels import band_state, prior_sigma, rmedv_all, session_layout, session_vwap, sg_velocity_all
    from primaries import make_primary
    from primaries.mechanism import primary_config
    from risk.costs import fill_costs
    from risk.portfolio import simulate_portfolio
    from risk.profiles import RiskProfile
    from wfo.position_backtest import decision_grid, position_backtest
    from wfo.rule_pass import rule_signals

    spy = load_bars("SPY", "5Min", START, END)
    print(f"SPY 5Min {spy.index[0].date()} → {spy.index[-1].date()}: {len(spy):,} bars")
    sess, start, _, slot = session_layout(spy.index, 5)
    x = np.log(spy["close"].to_numpy())
    o, c, v = (spy[k].to_numpy(dtype=float) for k in ("open", "close", "volume"))

    def estimators():
        rmedv_all(x, start, NS)
        sg_velocity_all(x, start, NS, 1)
        band_state(o, c, sess, slot, 14)
        session_vwap(c, v, sess)
        prior_sigma(x, sess, 5)

    t = time.perf_counter()
    estimators()  # includes numba's compile on a cold cache
    first = time.perf_counter() - t
    t = time.perf_counter()
    estimators()
    warm = time.perf_counter() - t
    print(f"[estimators] all G1 estimators, N in {NS}: {warm:.2f} s warm ({first:.2f} s first call)  budget <= 5 s")

    cfg = primary_config("region_trend", "5Min", META_MODEL="none", COST_MODEL="quotes", INIT_CASH=30_000)
    p = make_primary(cfg)
    t = time.perf_counter()
    dec, cells = p.cell_matrix(spy, cfg)
    t_cells = time.perf_counter() - t
    print(
        f"[positions]  {cells.shape[1]}-cell region positions at {len(dec):,} decisions: {t_cells:.2f} s  "
        "budget <= 10 s"
    )

    quotes = read_asof_table()
    qt = {s: quotes[quotes["symbol"] == s].reset_index(drop=True) for s in ("SPY", "QQQ")}
    prints = {s: load_prints(s, START, END) for s in ("SPY", "QQQ")}
    days = pd.DatetimeIndex(sorted(set(spy.index.tz_convert("America/New_York").normalize().tz_localize(None))))
    clock = _clock(days)
    cy = rate_asof(load_series("fred", "DTB3", "2015-12-15", END), clock if clock is not None else days) / 100.0
    grid = decision_grid(spy, cfg, clock)
    frame = pd.DataFrame(cells, index=spy.index[dec], columns=p.cell_names())
    reg_costs = fill_costs(spy.index, cfg, qt["SPY"])
    position_backtest(spy.iloc[:2000], frame.loc[: spy.index[1990]], cfg, costs=1e-5)  # compile
    t = time.perf_counter()
    for cost in (0.3e-4 / 2, 1.0e-4 / 2, 2.3e-4 / 2, reg_costs):  # round trip 0.3 / 1.0 / 2.3 bp, the registered model
        res = position_backtest(spy, frame, cfg, costs=cost, prints=prints["SPY"], cash_yield=cy, grid=grid)
    t_curve = time.perf_counter() - t
    print(
        f"[spec curve] {frame.shape[1]} cells x 4 costs, {res.ret.shape[0]:,} sessions: {t_curve:.2f} s  budget <= 5 s"
    )

    t0 = time.perf_counter()
    gate = RiskProfile("gate", daily_loss=0.02)
    streams = {}
    for sym in ("SPY", "QQQ"):
        df = spy if sym == "SPY" else load_bars(sym, "5Min", START, END)
        t = time.perf_counter()
        sig = rule_signals(df, cfg, symbol=sym, sessions=clock)
        bars = df.loc[sig.index[0] :]
        costs = {sym: fill_costs(bars.index, cfg, qt[sym])}
        eq, tr, _ = simulate_portfolio({sym: bars}, {sym: sig}, cfg, gate, costs=costs, prints={sym: prints[sym]},
                                       cash_yield=cy, sessions=None if clock is None else {sym: clock})  # fmt: skip
        d = eq.groupby(eq.index.tz_convert("America/New_York").normalize()).last()
        streams[sym] = d / np.r_[cfg.INIT_CASH, d.to_numpy()[:-1]] - 1
        print(
            f"  {sym}: rule pass + portfolio simulator, {len(sig):,} decisions, {len(tr):,} trade rows: "
            f"{time.perf_counter() - t:.1f} s"
        )
        pb = position_backtest(bars, sig["trade_signal"] * sig["bet_size"], cfg, costs=costs[sym], prints=prints[sym],
                               cash_yield=cy, profile=gate, sessions=clock)  # fmt: skip
        print(
            f"  {sym}: position_backtest vs the simulator over the window (loss gate on): max |Δ daily return| = "
            f"{np.abs(pb.ret.iloc[:, 0].to_numpy() - streams[sym].to_numpy()).max():.3g}"
        )
    both = pd.concat(streams, axis=1).dropna()
    t = time.perf_counter()
    mean_test(both.mean(axis=1).to_numpy(), n_boot=5000)  # the overlay-alpha bootstrap; its result is not read
    print(f"  overlay-alpha bootstrap (5,000 resamples, {len(both):,} days): {time.perf_counter() - t:.1f} s")
    print(
        f"[headline]   G1 headline on SPY + QQQ incl. the bootstrap: {time.perf_counter() - t0:.1f} s  budget <= 120 s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["parity", "budgets"])
    a = ap.parse_args()
    {"parity": parity, "budgets": budgets}[a.cmd]()


if __name__ == "__main__":
    main()
