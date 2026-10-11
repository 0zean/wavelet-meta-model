"""
The family test (SPEC §17.2–§17.3; protocol v3 in SPEC §22, U22): pooled streams, the headline test, floors,
coherence, reported splits and the program verdict. Pure functions of daily return streams (pd.Series indexed by NY
session date) and cost totals; the runner (families/run.py) builds the inputs.

Separation of decision and report: `headline_verdict` reads only the headline's stream, its benchmark, its cost totals
and the variants' statistics and floors (coherence); the splits, the quasi-holdout slice, shrinkage and responses are
computed by `describe` and never enter a verdict. No variant can replace the headline: the headline is the spec's
first variant by construction (families/spec.py) and every function here takes it by label.

Protocol v3 (U22). The spec's `test.kind` chooses the headline statistic, every one computed on EXCESS returns (the
stream and the benchmark less the T-bill accrual `rf` per session, SPEC §22.7):
- `sharpe_vs_benchmark` (PLAN2): the Ledoit–Wolf Sharpe difference vs the benchmark (`cash`: Sharpe vs 0);
- `overlay_alpha`: the mean of the overlay's daily excess return against zero (families.stats.mean_test: a
  studentized circular block bootstrap, Newey–West t alongside, PSR(0)); `delta_ann` is then the annualized alpha;
- `marginal`: the paired Ledoit–Wolf Sharpe difference of core + k × overlay vs the core (`core` = the core family's
  stream on the same days), with the drawdown difference and its bootstrap interval.
`test.sided` is "one" (H1: the statistic is positive) or "two". The cost curve (`reprice`) re-prices a member's
stream at another one-way cost per fill class from its per-session cost ledger (risk.portfolio `daily_costs`), so
the report shows every statistic at 0.3 / 1.0 / 2.3 bp round trip and at the measured profile; the verdict reads the
stream at the spec's `test.at_cost`. Coherence over region cells (`cells_coherence`) reads each cell's alpha sign
and the median cell's alpha (the region primary of U23 supplies the cells); with `test.coherence: cells` (U24) it is
the verdict's coherence and the §17 variant coherence is reported beside it.
"""

import re

import numpy as np
import pandas as pd

from families.spec import HEADLINE, FamilySpec
from families.stats import (
    TRADING_DAYS,
    circular_block_indices,
    drawdown,
    james_stein,
    mean_ci_nw,
    mean_test,
    newey_west_alpha,
    sharpe_ann,
    sharpe_test,
)

CRISES = (("2020-02-19", "2020-03-24"), ("2022-01-03", "2022-10-13"))  # SPY peak → trough (2020Q1, 2022)
COST_CURVE_BP = (0.3, 1.0, 2.3)  # round-trip costs of the report's cost curve (PLAN3 §1.2, §4.4)
FILL_CLASSES = ("open_auction", "open", "intra", "close", "close_auction")


# ── Streams ──────────────────────────────────────────────────────────────────


def day_index(s: pd.Series) -> pd.Series:
    """The series re-indexed by tz-naive NY session dates (the unit every stream is aligned on)."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return pd.Series(s.to_numpy(dtype=float), index=idx.normalize(), name=s.name)


def risk_weights(bh: dict[str, pd.Series]) -> pd.Series:
    """Equal-risk weights ∝ 1 / σ of each instrument's buy-and-hold daily returns over the weighting window (U22:
    the trailing 252 sessions before the family window when the data reach back, else the window's first 252
    sessions; families/run.py), summing to 1 (fixed ex ante: they depend on the instruments' own volatility, not on
    any strategy's result)."""
    inv = pd.Series({s: 1.0 / r.std(ddof=1) for s, r in bh.items()})
    if not np.isfinite(inv).all() or (inv <= 0).any():
        raise ValueError(f"buy-and-hold volatility undefined for {inv[~np.isfinite(inv)].index.tolist()}")
    return inv / inv.sum()


def pool(streams: dict[str, pd.Series], weights: pd.Series) -> pd.Series:
    """Σ w_i r_i on the union of the instruments' days (an instrument without a day there is flat: 0), from the first
    day any instrument's stream covers. Each instrument enters once (weights indexed by symbol)."""
    if set(streams) != set(weights.index):
        raise ValueError(f"pooled streams {sorted(streams)} and weights {sorted(weights.index)} differ")
    frame = pd.concat({s: day_index(r) for s, r in streams.items()}, axis=1, sort=True).fillna(0.0)
    return (frame * weights[frame.columns]).sum(axis=1).rename("ret")


def benchmark(kind: str, bh: dict[str, pd.Series], weights: pd.Series) -> pd.Series:
    """`constant_mix_ew` = mean of the instruments' daily returns (daily rebalanced), `constant_mix_er` = the
    daily-rebalanced equal-risk mix, `buy_and_hold` (U22) = the equal-risk mix bought once and left to drift,
    `cash` = 0 (the test is then of Sharpe > 0). The PLAN2 names buy_and_hold_ew / buy_and_hold_er are aliases of
    the constant mixes (families.spec)."""
    frame = pd.concat({s: day_index(r) for s, r in bh.items()}, axis=1, sort=True).dropna(how="all")
    if kind == "cash":
        return pd.Series(0.0, index=frame.index, name="bench")
    if kind in ("constant_mix_ew", "buy_and_hold_ew"):
        w = pd.Series(1.0 / len(bh), index=frame.columns)
    else:
        w = weights[frame.columns]
    if kind == "buy_and_hold":
        growth = (1.0 + frame.fillna(0.0)).cumprod().shift(1).fillna(1.0)  # each holding's value at the day's start
        wt = growth.mul(w, axis=1).where(frame.notna(), 0.0)
        return (wt * frame.fillna(0.0)).sum(axis=1).div(wt.sum(axis=1)).rename("bench")
    # an instrument without a bar that day (not listed yet, a dropped session) is left out and the rest re-weighted
    return (frame.fillna(0.0) * w).sum(axis=1).div(frame.notna().mul(w).sum(axis=1)).rename("bench")


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """The two streams on the days both cover (the paired test's sample)."""
    return pd.concat([day_index(a).rename("a"), day_index(b).rename("b")], axis=1, sort=True).dropna()


def excess(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    """r less the T-bill accrual per session (`rf`, a daily fraction by NY date; 0 where absent or None)."""
    r = day_index(r)
    if rf is None:
        return r
    return (r - day_index(rf).reindex(r.index).fillna(0.0)).rename(r.name)


def rf_accrual(rate: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The daily cash accrual the simulator credits (risk.portfolio): for session d, the annual rate as of the
    previous session × calendar days since it / 360; 0 on the first session or where the rate is unknown."""
    sess = pd.DatetimeIndex(sessions)
    r = day_index(rate).reindex(sess)
    days = np.r_[0, np.diff(sess.to_numpy().astype("M8[D]").astype(np.int64))]
    prev = r.shift(1).to_numpy()
    return pd.Series(np.nan_to_num(prev * days / 360.0), index=sess, name="rf")


def stream_stats(r: pd.Series) -> dict:
    from experiments.runner import daily_stats

    return daily_stats(day_index(r))


# ── Cost curve ───────────────────────────────────────────────────────────────


def reprice(r: pd.Series, daily_costs: pd.DataFrame, cost) -> tuple[pd.Series, dict]:
    """
    A member's daily stream re-priced at another one-way cost: `cost` is a round-trip cost in bp (one-way = half,
    on every fill class) or a dict of one-way bp per fill class (the measured profile). Per session the booked cost
    is added back and the alternative subtracted, both as fractions of the session's starting equity (first order: the
    positions are not re-simulated). Returns the stream and its totals {cost_paid, pnl_delta} in cash.
    """
    r = day_index(r)
    dc = daily_costs.copy()
    dc.index = pd.DatetimeIndex(dc.index).normalize()
    booked = sum(dc[f"cost_{k}"] for k in FILL_CLASSES)
    if isinstance(cost, dict):
        alt = sum(dc[f"notional_{k}"] * float(cost[k]) * 1e-4 for k in FILL_CLASSES)
    else:
        alt = sum(dc[f"notional_{k}"] for k in FILL_CLASSES) * (float(cost) / 2.0) * 1e-4
    adj = ((booked - alt) / dc["equity_start"]).reindex(r.index).fillna(0.0)
    return (r + adj).rename(r.name), {"cost_paid": float(alt.sum()), "pnl_delta": float((booked - alt).sum())}


# ── Headline test, floors, coherence ─────────────────────────────────────────


def compare(
    r: pd.Series,
    bench: pd.Series,
    kind: str,
    test: dict,
    rf: pd.Series | None = None,
    core: pd.Series | None = None,
    k: float = 1.0,
) -> dict:
    """The headline statistic of a stream (module docstring): `test["kind"]` on excess returns, `test["sided"]`."""
    tkind, sided = test.get("kind", "sharpe_vs_benchmark"), test.get("sided", "two")
    j = align(r, bench)
    a = excess(j["a"], rf)
    b = excess(j["b"], rf) if kind != "cash" else j["b"]
    out = {
        "n_days": len(j),
        "sharpe": sharpe_ann(a),
        "bench_sharpe": sharpe_ann(b) if kind != "cash" else 0.0,
        "kind": tkind,
        "sided": sided,
    }
    kw = {"block": test["block_days"], "n_boot": test["n_boot"], "alpha": test["alpha"], "seed": test["seed"],
          "sided": sided}  # fmt: skip
    try:
        if tkind == "overlay_alpha":
            mt = mean_test(a.to_numpy(), **kw)
            out |= {"delta_ann": mt["mean_ann"], "p": mt["p"], "ci_ann": list(mt["ci_ann"]), "statistic": "alpha_ann",
                    "alpha_bp_per_day": mt["mean_bp_per_day"], "t": mt["t"], "t_nw": mt["t_nw"], "mean_test": mt}  # fmt: skip
        elif tkind == "marginal":
            if core is None:
                raise ValueError("test kind 'marginal' needs the core stream")
            jc = pd.concat([a.rename("a"), excess(core, rf).rename("c")], axis=1, sort=True).dropna()
            if len(jc) < 2 * test["block_days"]:
                raise ValueError(f"core and overlay share only {len(jc)} days")
            port = jc["c"] + float(k) * jc["a"]
            lw = sharpe_test(port, jc["c"], **kw)
            dd = _dd_difference(port.to_numpy(), jc["c"].to_numpy(), test)
            out |= {"delta_ann": lw["delta_ann"], "p": lw["p"], "ci_ann": list(lw["ci_ann"]), "lw": lw,
                    "statistic": "delta_sharpe_marginal", "k": float(k), "n_days": len(jc),
                    "core_sharpe": sharpe_ann(jc["c"]), "portfolio_sharpe": sharpe_ann(port),
                    "dd_core": drawdown(jc["c"]), "dd_portfolio": drawdown(port), "dd_difference": dd}  # fmt: skip
        else:
            lw = sharpe_test(a, None if kind == "cash" else b, **kw)
            out |= {"delta_ann": lw["delta_ann"], "p": lw["p"], "ci_ann": list(lw["ci_ann"]), "lw": lw,
                    "statistic": "delta_sharpe"}  # fmt: skip
    except ValueError as e:  # a flat stream (no trades) has no statistic: nothing to test
        out |= {"delta_ann": np.nan, "p": 1.0, "ci_ann": [np.nan, np.nan], "error": str(e),
                "statistic": {"overlay_alpha": "alpha_ann", "marginal": "delta_sharpe_marginal"}.get(tkind,
                                                                                                  "delta_sharpe")}  # fmt: skip
    if kind != "cash" and len(j) > 10 and a.std() > 0:
        out["alpha"] = newey_west_alpha(a, b, lags=5)
    return out


def _dd_difference(port: np.ndarray, core: np.ndarray, test: dict) -> dict:
    """Max drawdown of core + overlay minus the core's, with a circular block-bootstrap percentile interval."""
    rng = np.random.default_rng(test["seed"])
    T = len(port)
    n = min(int(test["n_boot"]), 1000)
    idx = circular_block_indices(T, test["block_days"], n, rng)
    P, C = port[idx], core[idx]

    def dd(X):
        curve = np.cumprod(1 + X, axis=1)
        return (curve / np.maximum.accumulate(curve, axis=1) - 1).min(axis=1)

    diff = dd(P) - dd(C)
    lo, hi = np.quantile(diff, [test["alpha"] / 2, 1 - test["alpha"] / 2])
    return {"value": float(drawdown(port) - drawdown(core)), "ci": [float(lo), float(hi)], "n_boot": n}


def floors(r: pd.Series, bench: pd.Series, costs: dict, spec_floors: dict, rf: pd.Series | None = None) -> dict:
    """
    Each floor (SPEC §17.2, §22.4) → {value, threshold, ok}; `ok` = every set floor holds. Net return = the
    annualized compounded EXCESS return of the stream (less the T-bill accrual `rf`: the cash yield the simulator
    credits is not return, U22 review); vs benchmark: ≥ k × the benchmark's excess return on the same days; edge to
    cost: gross trading P&L per unit traded notional (net P&L without the cash interest + costs paid) ≥ k × costs
    paid per unit traded notional; max_dd: the daily stream's drawdown ≥ −x. A stream that never trades fails the
    edge floor.
    """
    st = stream_stats(excess(r, rf))
    out: dict = {}
    f = spec_floors
    if f.get("min_net_ret") is not None:
        out["min_net_ret"] = {"value": st["ret_ann"], "threshold": f["min_net_ret"],
                              "ok": bool(st["ret_ann"] >= f["min_net_ret"])}  # fmt: skip
    if f.get("min_net_ret_vs_benchmark") is not None:
        j = align(r, bench)
        b = stream_stats(excess(j["b"], rf))["ret_ann"] if len(j) else np.nan
        a = stream_stats(excess(j["a"], rf))["ret_ann"] if len(j) else np.nan
        thr = f["min_net_ret_vs_benchmark"] * b
        out["min_net_ret_vs_benchmark"] = {"value": a, "benchmark": b, "threshold": thr, "ok": bool(a >= thr)}
    if f.get("min_edge_to_cost") is not None:
        n = costs.get("traded_notional", 0.0)
        edge = (costs["pnl"] + costs["cost_paid"]) / n if n > 0 else np.nan
        cost = costs["cost_paid"] / n if n > 0 else np.nan
        ok = bool(n > 0 and edge >= f["min_edge_to_cost"] * cost)
        out["min_edge_to_cost"] = {"value": edge / cost if n > 0 and cost > 0 else np.nan,
                                   "gross_edge_bp": edge * 1e4, "cost_bp": cost * 1e4,
                                   "threshold": f["min_edge_to_cost"], "ok": ok}  # fmt: skip
    if f.get("max_dd") is not None:
        out["max_dd"] = {"value": st["max_dd"], "threshold": -f["max_dd"], "ok": bool(st["max_dd"] >= -f["max_dd"])}
    out["ok"] = bool(out) and all(v["ok"] for v in out.values())  # no floor set is not a pass
    return out


def coherence(variants: dict[str, dict], share: float) -> dict:
    """
    Beckman's coherent signature over the non-headline variants: the share with the headline's sign on the headline
    statistic (≥ `share`), and the lower-median variant (by that statistic) clearing every floor. `variants` =
    label → {"delta_ann", "floors_ok"} with the headline under HEADLINE. With no variant the check is vacuous (True).
    """
    h = variants[HEADLINE]
    rest = {k: v for k, v in variants.items() if k != HEADLINE}
    if not rest:
        return {"n_variants": 0, "share": np.nan, "median_variant": None, "coherent": True, "note": "no variants"}
    sign = np.sign(h["delta_ann"]) if np.isfinite(h["delta_ann"]) else 0.0
    deltas = {k: (v["delta_ann"] if np.isfinite(v["delta_ann"]) else -np.inf * sign) for k, v in rest.items()}
    same = [k for k, d in deltas.items() if sign != 0 and np.sign(d) == sign]
    sh = len(same) / len(rest)
    order = sorted(deltas, key=lambda k: (deltas[k], k))
    med = order[(len(order) - 1) // 2]
    ok = bool(sign != 0 and sh >= share - 5e-4 and rest[med]["floors_ok"])  # to 3 decimals: 0.667 = two thirds
    return {"n_variants": len(rest), "share": sh, "same_sign": same, "median_variant": med,
            "median_floors_ok": bool(rest[med]["floors_ok"]), "coherent": ok}  # fmt: skip


def cells_coherence(cell_alphas: dict[str, float], headline_alpha: float, share: float) -> dict:
    """
    Coherence over a region's cells (protocol v3, §4.1): the share of cells whose alpha has the headline's sign
    (≥ `share`) and the median cell's alpha (reported; the headline is the region, never a cell). `cell_alphas` =
    cell label → mean daily excess return (bp per day or any one unit).
    """
    if not cell_alphas:
        return {"n_cells": 0, "share": np.nan, "median_alpha": np.nan, "coherent": True, "note": "no cells"}
    sign = np.sign(headline_alpha) if np.isfinite(headline_alpha) else 0.0
    vals = np.array([v for v in cell_alphas.values()], dtype=float)
    same = int(np.sum(np.sign(vals) == sign)) if sign != 0 else 0
    sh = same / len(vals)
    return {"n_cells": len(vals), "share": sh, "median_alpha": float(np.median(vals)),
            "coherent": bool(sign != 0 and sh >= share - 5e-4)}  # fmt: skip


def delta_common(
    r: pd.Series,
    headline: pd.Series,
    bench: pd.Series,
    kind: str,
    rf: pd.Series | None = None,
    test: dict | None = None,
) -> float:
    """A variant's headline statistic on the days it shares with the headline (and the benchmark): the coherence
    comparison, so a variant that starts later (a longer warm-up, a state fit) or covers fewer days is compared over
    the same sessions as the headline rather than over its own sample. The headline's own test is unaffected (it never
    reads a variant). Sharpe kinds: the Sharpe difference on excess returns; `overlay_alpha`: the annualized mean
    excess return."""
    j = pd.concat([day_index(r).rename("a"), day_index(headline).rename("h"), day_index(bench).rename("b")], axis=1,
                  sort=True).dropna()  # fmt: skip
    if len(j) < 2:
        return np.nan
    a = excess(j["a"], rf)
    if test is not None and test.get("kind") == "overlay_alpha":
        return float(a.mean() * TRADING_DAYS)
    b = excess(j["b"], rf) if kind != "cash" else j["b"]
    return sharpe_ann(a) - (0.0 if kind == "cash" else sharpe_ann(b))


def headline_verdict(results: dict[str, dict], test: dict) -> dict:
    """The family's decision inputs (before Holm across families): headline p, its floors, coherence. With
    `test.coherence` "cells" the coherence is the region cells' (PLAN3 §4.1) and the variants' is `variant_coherence`;
    a region whose cells were not evaluated is not coherent."""
    h = results[HEADLINE]
    coh = coherence({k: {"delta_ann": v["compare"]["delta_ann"] if k == HEADLINE else v["delta_common"],
                         "floors_ok": v["floors"]["ok"]} for k, v in results.items()}, test["coherence_share"])  # fmt: skip
    out = {"p": h["compare"]["p"], "delta_ann": h["compare"]["delta_ann"], "ci_ann": h["compare"]["ci_ann"],
           "floors_ok": h["floors"]["ok"], "coherence": coh, "positive": bool(h["compare"]["delta_ann"] > 0),
           "kind": test.get("kind", "sharpe_vs_benchmark"), "sided": test.get("sided", "two"),
           "cells": h.get("cells"), "coherence_kind": test.get("coherence", "variants")}  # fmt: skip
    if out["coherence_kind"] == "cells":
        out["variant_coherence"] = coh
        out["coherence"] = h.get("cells") or {"coherent": False, "note": "the region's cells were not evaluated"}
    return out


def evaluate(
    fam: FamilySpec,
    streams: dict[str, pd.Series],
    bench: pd.Series,
    costs: dict[str, dict],
    rf: pd.Series | None = None,
    core: pd.Series | None = None,
    k_by_variant: dict[str, float] | None = None,
    cell_alphas: dict[str, float] | None = None,
) -> dict:
    """Per variant: stream statistics, the headline statistic and the floors; plus the headline verdict. `core` and
    `k_by_variant` serve the `marginal` kind; `cell_alphas` (a region's cells) the cells coherence."""
    out = {}
    for v in fam.variants:
        r = streams[v.label]
        k = (k_by_variant or {}).get(v.label, 1.0)
        out[v.label] = {"stats": stream_stats(r), "compare": compare(r, bench, fam.benchmark, fam.test, rf, core, k),
                        "floors": floors(r, bench, costs[v.label], fam.floors, rf), "costs": costs[v.label],
                        "start": str(day_index(r).index.min().date()) if len(r) else None,
                        "delta_common": delta_common(r, streams[HEADLINE], bench, fam.benchmark, rf, fam.test)}  # fmt: skip
    from validation.stats import psr, return_moments

    h = excess(streams[HEADLINE], rf)
    try:
        m = return_moments(h.to_numpy())
        out[HEADLINE]["psr0"] = psr(m.sr, m.n_obs, m.skew, m.kurt)
    except ValueError:
        out[HEADLINE]["psr0"] = np.nan
    if cell_alphas is not None:
        hc = out[HEADLINE]["compare"]
        head_alpha = hc.get("alpha_bp_per_day", hc["delta_ann"])
        out[HEADLINE]["cells"] = cells_coherence(cell_alphas, head_alpha, fam.test["coherence_share"])
    return {"variants": out, "verdict": headline_verdict(out, fam.test)}


def n_eff(per_instrument: dict[str, pd.Series], weights: pd.Series) -> dict:
    """Effective number of instruments of the pooled stream (the statistics audit's formula): (Σ w_i σ_i)² /
    Var(Σ w_i r_i) on the instruments' common span — the Sharpe gain of pooling if every instrument had the same
    Sharpe ratio. 1 = one instrument's worth of diversification."""
    if not per_instrument:
        return {"n_eff": np.nan, "n": 0}
    F = pd.concat({s: day_index(r) for s, r in per_instrument.items()}, axis=1, sort=True)
    first = max(F[c].first_valid_index() for c in F.columns)
    last = min(F[c].last_valid_index() for c in F.columns)
    F = F.loc[first:last].fillna(0.0)
    w = weights[F.columns].to_numpy(dtype=float)
    sd = F.std(ddof=1).to_numpy()
    var = float(w @ F.cov().to_numpy() @ w)
    return {"n_eff": float((w * sd).sum() ** 2 / var) if var > 0 else np.nan, "n": int(F.shape[1]),
            "corr": F.corr().round(3).to_dict()}  # fmt: skip


# ── Reported (never tested) ──────────────────────────────────────────────────


def _lpm2(r: np.ndarray) -> float:
    return float(np.sqrt((np.minimum(r, 0) ** 2).mean())) if r.size else np.nan


def responses(r: pd.Series, trades: pd.DataFrame | None, names: list[str]) -> dict:
    """The spec's response measures of one stream (and its trades)."""
    r = day_index(r)
    st = stream_stats(r)
    x = r.to_numpy()
    out = {}
    for n in names:
        if n in ("sharpe", "sortino", "calmar", "ret_ann", "vol_ann", "max_dd"):
            out[n] = st[n]
        elif n == "skew":
            out[n] = st["sr_skew"]
        elif n == "lpm2":
            out[n] = _lpm2(x) * np.sqrt(TRADING_DAYS)
        elif n == "exposure":
            out[n] = float((x != 0).mean()) if x.size else np.nan
        elif n == "crisis_return":
            out[n] = {f"{a}..{b}": float((1 + r.loc[a:b]).prod() - 1) for a, b in CRISES if len(r.loc[a:b])}
        elif n in ("mean_per_trade_bp", "hit_rate"):
            if trades is None or not len(trades):
                out[n] = np.nan
            else:
                p = position_returns(trades)
                out[n] = float(p.mean() * 1e4) if n == "mean_per_trade_bp" else float((p > 0).mean())
        elif n == "worst_day":
            out[n] = float(x.min()) if x.size else np.nan
        elif n == "longest_flat_run":
            out[n] = _longest_zero_run(x)
    return out


def _longest_zero_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return int(best)


def chain_ids(rolled: np.ndarray) -> np.ndarray:
    """Position ids of a symbol's trades in entry order: a trade flagged `rolled` had its exit rolled into the next
    trade's entry (risk.portfolio), so a chain ends at the first trade that is not rolled (U22 fix: the id advances
    AFTER an unrolled trade, not at it)."""
    rolled = np.asarray(rolled, dtype=bool)
    return np.r_[0, np.cumsum(~rolled)[:-1]] if len(rolled) else np.zeros(0, int)


def position_returns(trades: pd.DataFrame) -> np.ndarray:
    """Per-unit return of each position: a chain of rolled trades (a position held through rebalances: each trade
    whose `rolled` flag is set rolled its exit into the next trade's entry) is one position, its trades' returns
    compounded. Without the column, every trade is a position."""
    t = trades.reset_index(drop=True)
    if "rolled" not in t or "sym" not in t:
        return t["pnl_pct"].to_numpy(dtype=float)
    order = t.sort_values(["sym", "entry_b"], kind="stable") if "entry_b" in t else t
    rolled = order["rolled"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    sym = order["sym"].to_numpy()
    chain = np.zeros(len(order), int)
    for s in pd.unique(sym):
        m = sym == s
        chain[m] = chain_ids(rolled[m])
    g = (1.0 + order["pnl_pct"].astype(float)).groupby([sym, chain]).prod() - 1.0
    return g.to_numpy(dtype=float)


def _groups(kind: str, days: pd.DatetimeIndex, state: pd.DataFrame, bench: pd.Series) -> pd.Series | None:
    """Day → group label of a state split; None when its input is unavailable."""
    if kind in ("vix_tercile", "vix_median"):
        if state is None or "vix_prev" not in state:
            return None
        v = state["vix_prev"].reindex(days)
        q = (1 / 3, 2 / 3) if kind == "vix_tercile" else (0.5,)
        cuts = v.quantile(list(q)).to_numpy()
        lab = np.searchsorted(cuts, v.to_numpy(), side="right")
        names = ["low", "mid", "high"] if kind == "vix_tercile" else ["below", "above"]
        return pd.Series(np.where(v.isna(), "n/a", np.array(names, dtype=object)[np.minimum(lab, len(names) - 1)]),
                         index=days)  # fmt: skip
    if kind == "macro_day":
        if state is None or "macro" not in state:
            return None
        return pd.Series(
            np.where(state["macro"].reindex(days).fillna(False).astype(bool), "macro", "other"), index=days
        )
    if kind == "abs_move_tercile":
        m = day_index(bench).abs().reindex(days)
        cuts = m.quantile([1 / 3, 2 / 3]).to_numpy()
        lab = np.searchsorted(cuts, m.to_numpy(), side="right")
        return pd.Series(np.array(["small", "mid", "large"], dtype=object)[np.minimum(lab, 2)], index=days)
    if kind == "prior_day_sign":
        prev = day_index(bench).shift(1).reindex(days)
        return pd.Series(np.where(prev > 0, "up", np.where(prev < 0, "down", "flat")), index=days)
    if kind == "day_of_week":
        return pd.Series(days.day_name(), index=days)
    if kind == "year":
        return pd.Series(days.year.astype(str), index=days)
    if kind == "vol_quintile":  # U24: the instruments' mean causal percentile of the prior session's 5Min σ
        if state is None or "vol_rank" not in state:
            return None
        v = state["vol_rank"].reindex(days).to_numpy()
        lab = np.minimum((np.nan_to_num(v, nan=0.0) * 5).astype(int), 4)
        return pd.Series(np.where(np.isnan(v), "n/a", np.array(["q1 low", "q2", "q3", "q4", "q5 high"],
                                                                dtype=object)[lab]), index=days)  # fmt: skip
    if kind == "opex_day":
        if state is None or "opex" not in state:
            return None
        return pd.Series(np.where(state["opex"].reindex(days).fillna(False).astype(bool), "opex", "other"), index=days)
    return None  # gamma_sign: no gamma proxy (PLAN2 U15 note; U26 adds the revealed-gamma proxy)


def split_table(r: pd.Series, groups: pd.Series) -> list[dict]:
    """Per group: days, mean bp with its Newey–West 95 % interval, annualized Sharpe over the group's days."""
    r = day_index(r)
    rows = []
    for g in sorted(groups.dropna().unique()):
        x = r[groups.reindex(r.index) == g].to_numpy()
        m, lo, hi = mean_ci_nw(x)
        rows.append({"group": str(g), "n_days": int(x.size), "mean_bp": m * 1e4, "ci_bp": [lo * 1e4, hi * 1e4],
                     "sharpe": sharpe_ann(x)})  # fmt: skip
    return rows


def describe(
    fam: FamilySpec,
    streams: dict[str, pd.Series],
    bench: pd.Series,
    trades: dict[str, pd.DataFrame],
    per_instrument: dict[str, pd.Series],
    state: pd.DataFrame | None,
    sample_streams: dict[str, tuple[pd.Series, pd.Series]],
    quasi: tuple[pd.Series, pd.Series] | None,
) -> dict:
    """Everything reported and never tested: responses per variant, state splits, sample splits, the quasi-holdout
    slice, per-instrument Sharpes with James–Stein shrinkage, and the specification curve."""
    h = day_index(streams[HEADLINE])
    out: dict = {"responses": {v.label: responses(streams[v.label], trades.get(v.label), fam.response)
                               for v in fam.variants}}  # fmt: skip
    out["state_splits"] = {}
    for k in fam.state_splits:
        g = _groups(k, h.index, state, bench)
        out["state_splits"][k] = {"unavailable": True} if g is None else {"groups": split_table(h, g)}
    out["sample_splits"] = {}
    for s in fam.sample_splits:
        if "start" in s:
            r, b = h.loc[s["start"] : pd.Timestamp(s["end"]) - pd.Timedelta(days=1)], bench
        elif s["label"] in sample_streams:
            r, b = sample_streams[s["label"]]
        else:
            out["sample_splits"][s["label"]] = {"unavailable": True}
            continue
        out["sample_splits"][s["label"]] = _slice(r, b, fam)
    if quasi is not None:
        out["quasi_holdout"] = {**_slice(*quasi, fam), "note": QUASI_NOTE}
    if per_instrument:
        sr = {s: day_index(r) for s, r in per_instrument.items()}
        raw = {s: sharpe_ann(r) for s, r in sr.items()}
        ok = [s for s in raw if np.isfinite(raw[s])]
        js = james_stein([raw[s] / np.sqrt(TRADING_DAYS) for s in ok], [len(sr[s]) for s in ok])
        out["per_instrument"] = {s: {"sharpe": raw[s], "sharpe_js": (float(js[ok.index(s)] * np.sqrt(TRADING_DAYS))
                                                                      if s in ok else np.nan)}
                                 for s in raw}  # fmt: skip
    return out


QUASI_NOTE = ("quasi-holdout 2025-10-01 → 2026-09-30: contaminated (the U11 holdout exposed these days' buy-and-hold "
              "outcomes before the families were chosen); a robustness slice, never a gate")  # fmt: skip


def _slice(r: pd.Series, b: pd.Series, fam: FamilySpec) -> dict:
    j = align(r, b)
    out = {"n_days": len(j), "sharpe": sharpe_ann(j["a"]) if len(j) else np.nan,
           "bench_sharpe": sharpe_ann(j["b"]) if len(j) and fam.benchmark != "cash" else 0.0,
           "ret": float((1 + j["a"]).prod() - 1) if len(j) else np.nan,
           "max_dd": drawdown(j["a"]) if len(j) else np.nan}  # fmt: skip
    m, lo, hi = mean_ci_nw(j["a"].to_numpy()) if len(j) > 2 else (np.nan, np.nan, np.nan)
    out |= {"mean_bp": m * 1e4, "ci_bp": [lo * 1e4, hi * 1e4]}
    return out


# ── Program verdict (SPEC §17.3, §22) ────────────────────────────────────────


def family_order(fid: str) -> list:
    """Natural sort key: F2 before F11, an amendment after its family (F1, F1.v2, F2, ..., F11)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", fid)]


def program_verdict(families: dict[str, dict], alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm over the families' headline p-values; a family passes iff its Holm-adjusted p < alpha, its headline clears
    every floor, it is coherent and its statistic is positive (a two-sided test's significant shortfall is not a
    pass; a one-sided p already encodes the direction). `families` = id → its evaluate()["verdict"]. The DSR is not
    read here (U22: it left the verdict and stays a ledger diagnostic in the report).
    """
    from families.stats import holm

    if not families:
        return pd.DataFrame(columns=["family", "p", "p_holm", "floors_ok", "coherent", "positive", "passes"])
    ids = sorted(families, key=family_order)
    p = np.array([families[f]["p"] for f in ids], dtype=float)
    adj = holm(np.nan_to_num(p, nan=1.0))
    rows = []
    for f, pa in zip(ids, adj, strict=True):
        v = families[f]
        coh = bool(v["coherence"]["coherent"])
        rows.append({"family": f, "kind": v.get("kind", "sharpe_vs_benchmark"), "delta_ann": v["delta_ann"],
                     "p": v["p"], "p_holm": float(pa), "floors_ok": bool(v["floors_ok"]), "coherent": coh,
                     "positive": bool(v["positive"]),
                     "passes": bool(pa < alpha and v["floors_ok"] and coh and v["positive"])})  # fmt: skip
    return pd.DataFrame(rows)
