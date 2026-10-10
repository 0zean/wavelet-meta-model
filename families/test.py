"""
The family test (SPEC §17.2–§17.3): pooled streams, the headline test, floors, coherence, reported splits and the
program verdict. Pure functions of daily return streams (pd.Series indexed by NY session date) and cost totals; the
runner (families/run.py) builds the inputs.

Separation of decision and report: `headline_verdict` reads only the headline's stream, its benchmark, its cost totals
and the variants' Sharpe differences and floors (coherence); the splits, the quasi-holdout slice, shrinkage and
responses are computed by `describe` and never enter a verdict. No variant can replace the headline: the headline is
the spec's first variant by construction (families/spec.py) and every function here takes it by label.
"""

import numpy as np
import pandas as pd

from families.spec import HEADLINE, FamilySpec
from families.stats import TRADING_DAYS, drawdown, james_stein, mean_ci_nw, newey_west_alpha, sharpe_ann, sharpe_test

CRISES = (("2020-02-19", "2020-03-24"), ("2022-01-03", "2022-10-13"))  # SPY peak → trough (2020Q1, 2022)


# ── Streams ──────────────────────────────────────────────────────────────────


def day_index(s: pd.Series) -> pd.Series:
    """The series re-indexed by tz-naive NY session dates (the unit every stream is aligned on)."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return pd.Series(s.to_numpy(dtype=float), index=idx.normalize(), name=s.name)


def risk_weights(bh: dict[str, pd.Series]) -> pd.Series:
    """Equal-risk weights ∝ 1 / σ of each instrument's buy-and-hold daily returns over the window, summing to 1
    (fixed ex ante: they depend on the instruments' own volatility, not on any strategy's result)."""
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
    """`buy_and_hold_ew` = mean of the instruments' daily returns (daily rebalanced), `buy_and_hold_er` = the equal-risk
    mix, `cash` = 0 (the test is then of Sharpe > 0)."""
    frame = pd.concat({s: day_index(r) for s, r in bh.items()}, axis=1, sort=True).dropna(how="all")
    if kind == "cash":
        return pd.Series(0.0, index=frame.index, name="bench")
    w = pd.Series(1.0 / len(bh), index=frame.columns) if kind == "buy_and_hold_ew" else weights[frame.columns]
    # an instrument without a bar that day (not listed yet, a dropped session) is left out and the rest re-weighted
    return (frame.fillna(0.0) * w).sum(axis=1).div(frame.notna().mul(w).sum(axis=1)).rename("bench")


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """The two streams on the days both cover (the paired test's sample)."""
    return pd.concat([day_index(a).rename("a"), day_index(b).rename("b")], axis=1, sort=True).dropna()


def stream_stats(r: pd.Series) -> dict:
    from experiments.runner import daily_stats

    return daily_stats(day_index(r))


# ── Headline test, floors, coherence ─────────────────────────────────────────


def compare(r: pd.Series, bench: pd.Series, kind: str, test: dict) -> dict:
    """The Sharpe-difference test of a stream vs its benchmark (one-sample vs 0 for `cash`) and the NW alpha."""
    j = align(r, bench)
    out = {
        "n_days": len(j),
        "sharpe": sharpe_ann(j["a"]),
        "bench_sharpe": sharpe_ann(j["b"]) if kind != "cash" else 0.0,
    }
    try:
        lw = sharpe_test(j["a"], None if kind == "cash" else j["b"], block=test["block_days"], n_boot=test["n_boot"],
                         alpha=test["alpha"], seed=test["seed"])  # fmt: skip
    except ValueError as e:  # a flat stream (no trades) has no Sharpe ratio: nothing to test
        lw = {"delta_ann": np.nan, "p": 1.0, "ci_ann": (np.nan, np.nan), "error": str(e)}
    out |= {"delta_ann": lw["delta_ann"], "p": lw["p"], "ci_ann": list(lw["ci_ann"]), "lw": lw}
    if kind != "cash" and len(j) > 10 and j["a"].std() > 0:
        out["alpha"] = newey_west_alpha(j["a"], j["b"], lags=5)
    return out


def floors(r: pd.Series, bench: pd.Series, costs: dict, spec_floors: dict) -> dict:
    """
    Each floor (SPEC §17.2) → {value, threshold, ok}; `ok` = every set floor holds. Net return = the annualized
    compounded return of the stream; vs benchmark: ≥ k × the benchmark's on the same days; edge to cost: gross P&L per
    unit traded notional (net P&L + costs paid) ≥ k × costs paid per unit traded notional; max_dd: the daily stream's
    drawdown ≥ −x. A stream that never trades fails the edge floor (no edge).
    """
    st = stream_stats(r)
    out: dict = {}
    f = spec_floors
    if f.get("min_net_ret") is not None:
        out["min_net_ret"] = {"value": st["ret_ann"], "threshold": f["min_net_ret"],
                              "ok": bool(st["ret_ann"] >= f["min_net_ret"])}  # fmt: skip
    if f.get("min_net_ret_vs_benchmark") is not None:
        j = align(r, bench)
        b = stream_stats(j["b"])["ret_ann"] if len(j) else np.nan
        a = stream_stats(j["a"])["ret_ann"] if len(j) else np.nan
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
    Beckman's coherent signature over the non-headline variants: the share with the headline's sign on the Sharpe
    difference (≥ `share`), and the lower-median variant (by Sharpe difference) clearing every floor. `variants` =
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


def delta_common(r: pd.Series, headline: pd.Series, bench: pd.Series, kind: str) -> float:
    """A variant's Sharpe difference vs the benchmark on the days it shares with the headline (and the benchmark): the
    coherence comparison, so a variant that starts later (a longer warm-up, a state fit) or covers fewer days is
    compared over the same sessions as the headline rather than over its own sample. The headline's own test is
    unaffected (it never reads a variant)."""
    j = pd.concat([day_index(r).rename("a"), day_index(headline).rename("h"), day_index(bench).rename("b")], axis=1,
                  sort=True).dropna()  # fmt: skip
    if len(j) < 2:
        return np.nan
    return sharpe_ann(j["a"]) - (0.0 if kind == "cash" else sharpe_ann(j["b"]))


def headline_verdict(results: dict[str, dict], test: dict) -> dict:
    """The family's decision inputs (before Holm across families): headline p, its floors, coherence."""
    h = results[HEADLINE]
    coh = coherence({k: {"delta_ann": v["compare"]["delta_ann"] if k == HEADLINE else v["delta_common"],
                         "floors_ok": v["floors"]["ok"]} for k, v in results.items()}, test["coherence_share"])  # fmt: skip
    return {"p": h["compare"]["p"], "delta_ann": h["compare"]["delta_ann"], "ci_ann": h["compare"]["ci_ann"],
            "floors_ok": h["floors"]["ok"], "coherence": coh, "positive": bool(h["compare"]["delta_ann"] > 0)}  # fmt: skip


def evaluate(fam: FamilySpec, streams: dict[str, pd.Series], bench: pd.Series, costs: dict[str, dict]) -> dict:
    """Per variant: stream statistics, the comparison vs the benchmark and the floors; plus the headline verdict."""
    out = {}
    for v in fam.variants:
        r = streams[v.label]
        out[v.label] = {"stats": stream_stats(r), "compare": compare(r, bench, fam.benchmark, fam.test),
                        "floors": floors(r, bench, costs[v.label], fam.floors), "costs": costs[v.label],
                        "start": str(day_index(r).index.min().date()) if len(r) else None,
                        "delta_common": delta_common(r, streams[HEADLINE], bench, fam.benchmark)}  # fmt: skip
    from validation.stats import psr, return_moments

    h = day_index(streams[HEADLINE])
    try:
        m = return_moments(h.to_numpy())
        out[HEADLINE]["psr0"] = psr(m.sr, m.n_obs, m.skew, m.kurt)
    except ValueError:
        out[HEADLINE]["psr0"] = np.nan
    return {"variants": out, "verdict": headline_verdict(out, fam.test)}


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
    return out


def position_returns(trades: pd.DataFrame) -> np.ndarray:
    """Per-unit return of each position: a chain of rolled trades (a trade with `rolled` continues its symbol's
    previous one, held through a rebalance) is one position, its trades' returns compounded. Without the column, every
    trade is a position."""
    t = trades.reset_index(drop=True)
    if "rolled" not in t or "sym" not in t:
        return t["pnl_pct"].to_numpy(dtype=float)
    order = t.sort_values(["sym", "entry_b"], kind="stable") if "entry_b" in t else t
    rolled = order["rolled"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    chain = np.cumsum(~rolled)
    g = (1.0 + order["pnl_pct"].astype(float)).groupby([order["sym"].to_numpy(), chain]).prod() - 1.0
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
    return None  # gamma_sign: no gamma proxy (PLAN2 U15 note)


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


# ── Program verdict (SPEC §17.3) ─────────────────────────────────────────────


def program_verdict(families: dict[str, dict], alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm over the families' headline p-values; a family passes iff its Holm-adjusted p < alpha, its headline clears
    every floor and it is coherent. `families` = id → its evaluate()["verdict"] (+ "dsr" when known). The test is
    two-sided; a significant difference with the wrong sign (the headline below its benchmark) does not pass.
    """
    from families.stats import holm

    if not families:
        return pd.DataFrame(columns=["family", "p", "p_holm", "floors_ok", "coherent", "positive", "passes"])
    ids = sorted(families)
    p = np.array([families[f]["p"] for f in ids], dtype=float)
    adj = holm(np.nan_to_num(p, nan=1.0))
    rows = []
    for f, pa in zip(ids, adj, strict=True):
        v = families[f]
        coh = bool(v["coherence"]["coherent"])
        rows.append({"family": f, "delta_ann": v["delta_ann"], "p": v["p"], "p_holm": float(pa),
                     "floors_ok": bool(v["floors_ok"]), "coherent": coh, "positive": bool(v["positive"]),
                     "dsr": v.get("dsr", np.nan),
                     "passes": bool(pa < alpha and v["floors_ok"] and coh and v["positive"])})  # fmt: skip
    return pd.DataFrame(rows)
