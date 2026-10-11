"""
Family reports (SPEC §17, PLAN2 U17): per family `report.md` / `report.html` and `spec_curve.png` under
results/families/<id>/, and the program summary (the funnel of families, not of cells) `program_summary.md` / `.html`
over every family result in a results directory. Reads only the result.json files families/run.py writes.
"""

import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from families.spec import HEADLINE, ROOT


def _f(x, nd=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{100 * x:.{max(nd - 2, 1)}f} %" if pct else f"{x:.{nd}f}"


def _g(x) -> str:
    """A residual in %g notation ("—" when absent)."""
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3g}"


def _basis(wb: dict) -> str:
    if wb.get("window") == "equal":
        return "equal weights"
    return f"from the {wb.get('window', '?')} {wb.get('sessions', '')} sessions"


def _table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def spec_curve(result: dict, path: Path) -> Path | None:
    """Every variant's Sharpe difference vs the benchmark with its interval, sorted; the headline highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vs = result.get("evaluation", {}).get("variants", {})
    pts = [(k, v["compare"]["delta_ann"], v["compare"]["ci_ann"]) for k, v in vs.items() if "compare" in v]
    pts = [p for p in pts if p[1] is not None and np.isfinite(p[1])]
    if not pts:
        return None
    pts.sort(key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(max(6.5, 0.6 * len(pts) + 2), 4.0))
    for i, (k, d, ci) in enumerate(pts):
        c = "#c2410c" if k == HEADLINE else "#1d4ed8"
        lo, hi = (np.nan, np.nan) if ci is None else ci
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([i, i], [lo, hi], color=c, lw=1.5)
        ax.plot(i, d, "o", color=c)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xticks(range(len(pts)), [p[0] for p in pts], rotation=45, ha="right", fontsize=8)
    kind = result.get("test", {}).get("kind", "sharpe_vs_benchmark")
    ax.set_ylabel(
        {"overlay_alpha": "alpha (ann., excess)", "marginal": "Δ Sharpe core + k·overlay vs core"}.get(
            kind, "Sharpe (ann.)" if result.get("benchmark") == "cash" else "Δ Sharpe vs benchmark (ann.)"
        )
    )
    ax.set_title(f"{result['id']}: specification curve, 95 % intervals\n(headline in orange; variants reported, "
                 f"never selected; benchmark {result.get('benchmark')})", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


ESTIMATOR_COLORS = {"band": "#15803d", "rmedv": "#1d4ed8", "sgv": "#7c3aed", "modwt": "#0891b2"}


def _verdict_cost(result: dict) -> str:
    return (result.get("evaluation") or {}).get("verdict", {}).get("at_cost") or "registered"


def cells_curve(result: dict, path: Path) -> Path | None:
    """U24: every (instrument, cell) of a region headline — mean daily excess return at the verdict's cost with its
    Newey–West 95 % interval, sorted, coloured by estimator; the region per instrument in orange."""
    reg = result.get("region")
    if not reg or not reg.get("cells"):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ck = _verdict_cost(result)
    pts = [(k, v[ck]["alpha_bp"], v[ck]["ci_bp"]) for k, v in reg["cells"].items() if ck in v]
    pts += [(f"{k}:REGION", v[ck]["alpha_bp"], v[ck]["ci_bp"]) for k, v in reg.get("region", {}).items()
            if k != "pooled" and ck in v]  # fmt: skip
    pts = [p for p in pts if p[1] is not None and np.isfinite(p[1])]
    if not pts:
        return None
    pts.sort(key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.14 * len(pts) + 2), 4.2))
    for i, (k, a, ci) in enumerate(pts):
        est = k.split(":")[1].split("_")[0]
        c = "#c2410c" if k.endswith(":REGION") else ESTIMATOR_COLORS.get(est, "#6b7280")
        if ci and all(x is not None and np.isfinite(x) for x in ci):
            ax.plot([i, i], ci, color=c, lw=1.0, alpha=0.6)
        ax.plot(i, a, "o", color=c, ms=3.5)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(pts)} (instrument, cell) points, sorted; colours: "
                  + ", ".join(f"{k} {v}" for k, v in ESTIMATOR_COLORS.items()) + "; region orange", fontsize=7)  # fmt: skip
    ax.set_ylabel(f"alpha, bp/day (excess; cost {ck})")
    ax.set_title(f"{result['id']}: region cells, 95 % Newey–West intervals (reported; no cell is selected)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _region_md(result: dict) -> list[str]:
    reg = result.get("region")
    if not reg:
        return []
    ck = _verdict_cost(result)
    md = ["## Region cells (the specification curve over every cell; reported, the region is the headline)", ""]
    par = reg.get("parity") or {}
    if reg.get("parity_ok") is False:
        md.append("- **The re-simulation does not reproduce the members: the cells are not used for coherence.**")
    md.append("- Parity of the re-simulation (wfo.position_backtest) with the headline members at the booked costs: "
              + "; ".join(f"{k} max |Δ daily return| {_g(v.get('max_abs_diff'))} over {v.get('common_days')} days"
                          for k, v in par.items()))  # fmt: skip
    rg = reg.get("region") or {}
    rows = []
    for k in [x for x in rg if x != "pooled"] + (["pooled"] if "pooled" in rg else []):
        row = {"region": k}
        for c in reg.get("costs", []):
            e = rg[k].get(c)
            row[c] = "—" if e is None else f"{_f(e['alpha_bp'], 2)} bp ({_f(e['sharpe'], 2)})"
        rows.append(row)
    md += ["", ("Region re-simulated at each cost: alpha bp/day (excess Sharpe). The headline test re-prices the "
                "members' cost ledgers instead (first order); the two agree to the re-pricing error."), "",
           _table(rows, ["region", *reg.get("costs", [])]), ""]  # fmt: skip
    cells = reg.get("cells") or {}
    syms = sorted({k.split(":")[0] for k in cells})
    vals = [v[ck]["alpha_bp"] for v in cells.values() if ck in v]
    coh = ((result.get("evaluation") or {}).get("variants", {}).get(HEADLINE, {}) or {}).get("cells") or {}
    md.append(f"- {len(cells)} (instrument, cell) points at cost {ck}: share with the headline's sign "
              f"{_f(coh.get('share'), 3)}, median cell {_f(coh.get('median_alpha'), 2)} bp/day; positive "
              f"{int(np.sum(np.array(vals) > 0))} / {len(vals)}")  # fmt: skip
    by_est: dict[str, list[float]] = {}
    for k, v in cells.items():
        if ck in v:
            by_est.setdefault(k.split(":")[1].split("_")[0], []).append(v[ck]["alpha_bp"])
    md.append("- By estimator: " + "; ".join(f"{e} {int(np.sum(np.array(a) > 0))}/{len(a)} positive, median "
                                             f"{_f(float(np.median(a)), 2)} bp" for e, a in by_est.items()))  # fmt: skip
    md += ["", "![region cells](cells_curve.png)", ""]
    pc = reg.get("pooled_cells") or {}
    rows = []
    for name in reg.get("names", []):
        row = {"cell": name}
        for sym in syms:
            e = cells.get(f"{sym}:{name}", {}).get(ck)
            row[sym] = "—" if e is None else _f(e["alpha_bp"], 2)
        e = pc.get(name, {}).get(ck)
        if e is not None:
            row["pooled"] = _f(e["alpha_bp"], 2)
            row["CI (pooled)"] = f"{_f(e['ci_bp'][0], 2)} … {_f(e['ci_bp'][1], 2)}"
            row["Sharpe (pooled)"] = _f(e["sharpe"], 2)
            row["_k"] = e["alpha_bp"]
        rows.append(row)
    rows.sort(key=lambda r: -r.get("_k", -np.inf))
    md += [f"Every cell at cost {ck}, alpha bp/day (excess), sorted by the pooled cell:", "",
           _table(rows, ["cell", *syms, "pooled", "CI (pooled)", "Sharpe (pooled)"]), "", f"*{reg.get('note', '')}*",
           ""]  # fmt: skip
    return md


def _trade_md(result: dict) -> list[str]:
    md = []
    ts = result.get("trade_stats") or {}
    if ts.get("per_instrument"):
        md += ["## Trade statistics (headline; booked costs)", ""]
        rows = [{"instrument": k, **{c: _f(v.get(c), 3) for c in ("round_trips_per_day", "sessions_traded",
                                                                  "long_pnl_bp_per_day", "short_pnl_bp_per_day")},
                 "long / short trades": f"{v.get('long_trades', '—')} / {v.get('short_trades', '—')}"}
                for k, v in ts["per_instrument"].items()]  # fmt: skip
        rows.append({"instrument": "pooled", **{c: _f(ts["pooled"].get(c), 3) for c in (
            "round_trips_per_day", "sessions_traded", "long_pnl_bp_per_day", "short_pnl_bp_per_day")}})  # fmt: skip
        md += [_table(rows, ["instrument", "round_trips_per_day", "sessions_traded", "long_pnl_bp_per_day",
                             "short_pnl_bp_per_day", "long / short trades"]), "",
               ("Round trips per day: traded notional / the session's starting equity / 2, averaged over sessions. "
                "Long / short: trading P&L by side (booked costs) per day of the member's starting cash."), ""]  # fmt: skip
    pa = result.get("per_instrument_alpha") or {}
    if pa:
        md += [f"## Per-instrument headline alpha (cost {_verdict_cost(result)}; excess of the T-bill)", "",
               _table([{"instrument": k, "days": v["n_days"], "alpha_bp": _f(v["alpha_bp"], 2),
                        "CI (NW 95 %)": f"{_f(v['ci_bp'][0], 2)} … {_f(v['ci_bp'][1], 2)}",
                        "sharpe": _f(v["sharpe"], 2)} for k, v in pa.items()],
                      ["instrument", "days", "alpha_bp", "CI (NW 95 %)", "sharpe"]), ""]  # fmt: skip
    return md


def spec_diff(result: dict, repo: Path = ROOT) -> str:
    """The registered spec vs the spec that ran: `git diff <registered sha> -- <file>` limited to the spec (empty
    when they agree; the runner refuses a differing spec, so a non-empty diff is the `registered` line only)."""
    reg = result.get("registered") or {}
    sha = reg.get("sha")
    if not sha:
        return "(not registered: dry run without the registration check)"
    repo = Path(result.get("repo") or repo)
    try:
        rel = Path(result["spec_path"]).relative_to(repo).as_posix()
    except (KeyError, ValueError):
        rel = f"families/{result['id']}.yaml"
    p = subprocess.run(["git", "-C", str(repo), "diff", str(sha), "HEAD", "--", rel], capture_output=True, text=True,
                       check=False)  # fmt: skip
    if p.returncode:
        return f"(git diff failed: {p.stderr.strip()})"
    lines = [ln for ln in p.stdout.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    return "\n".join(lines) if lines else "(none)"


def family_markdown(result: dict) -> str:
    rid = result["id"]
    md = [f"# Family {rid}", ""]
    spec = result.get("spec", {})
    md += [f"**Mechanism.** {spec.get('mechanism', '').strip()}", ""]
    reg = result.get("registered") or {}
    md.append(f"- Registered: {reg.get('sha', '—')} ({reg.get('date', '—')}); run at {result.get('run_at')}, code "
              f"{str(result.get('code_hash'))[:12]}, git {str(result.get('git_sha'))[:12]}")  # fmt: skip
    weights = ", ".join(f"{k} {v:.2f}" for k, v in result.get("weights", {}).items())
    wb = result.get("weights_basis") or {}
    md.append(f"- Instruments: {', '.join(result['instruments'])} ({'basket' if result.get('basket') else 'pooled'}, "
              f"weights {weights}, {_basis(wb)}); window "
              f"{result['window'][0]} → {result['window'][1]}; benchmark {result.get('benchmark')}")  # fmt: skip
    ne = result.get("n_eff") or {}
    if ne.get("n_eff") is not None:
        md.append(f"- N_eff {_f(ne['n_eff'], 2)} of {ne.get('n')} instruments (the pooled stream's effective count)")
    t = result.get("test") or {}
    md.append(f"- Test: {t.get('kind', 'sharpe_vs_benchmark')}, {t.get('sided', 'two')}-sided, at cost "
              f"{t.get('at_cost', 'registered')}; excess returns: {result.get('rf', 'raw returns')}")  # fmt: skip
    pw = result.get("power")
    if pw:
        md.append(f"- MDE line: {_f(pw['mde_alpha_bp_per_day'], 2)} bp/day at {pw.get('power', 0.8):.0%} power over "
                  f"{pw['n_days']} days, vol {pw['vol_ann']:.1%}, {pw['families']} families; expected "
                  f"{_f(pw['expected_alpha_bp_per_day'], 2)} bp/day → "
                  f"{'a DIAGNOSTIC (below its MDE)' if pw.get('diagnostic') else 'a test'}")  # fmt: skip
        pr = result.get("power_realized")
        if pr:
            md.append(f"- MDE at the realized excess volatility {pr['vol_ann']:.1%} over {pr['n_days']} days: "
                      f"{_f(pr['mde_bp_per_day'], 2)} bp/day (Sharpe {_f(pr['mde_sharpe_ann'], 2)}; reported, the "
                      "registered line is the one above)")  # fmt: skip
    acc = result.get("accounting", {})
    md.append(f"- Trials: family {acc.get('family_trials')} / budget {acc.get('family_budget')}; program "
              f"{acc.get('program_trials')} / {acc.get('max_trials')} trials, {acc.get('program_families')} / "
              f"{acc.get('max_families')} families")  # fmt: skip
    md.append("")
    if "error" in result:
        return "\n".join([*md, f"**No test:** {result['error']}", ""])
    ev = result["evaluation"]
    v = ev["verdict"]
    coh = v["coherence"]
    md += ["## Headline test", ""]
    h = ev["variants"][HEADLINE]
    c = h["compare"]
    block = result["spec"].get("test", {}).get("block_days", 21)
    kind = c.get("kind", "sharpe_vs_benchmark")
    if kind == "overlay_alpha":
        md.append(f"- Alpha {_f(c['delta_ann'], 3, pct=True)}/yr ({_f(c.get('alpha_bp_per_day'), 2)} bp/day) on "
                  f"{c['n_days']} days, excess of the T-bill; bootstrap t {_f(c.get('t'), 2)}, Newey–West t "
                  f"{_f(c.get('t_nw'), 2)}; {c.get('sided')}-sided p = {_f(c['p'], 4)} (block {block} sessions); "
                  f"interval {_f(c['ci_ann'][0], 3, pct=True)} … {_f(c['ci_ann'][1], 3, pct=True)}; Sharpe "
                  f"{_f(c['sharpe'], 2)} (benchmark {_f(c['bench_sharpe'], 2)})")  # fmt: skip
    elif kind == "marginal":
        dd = c.get("dd_difference") or {}
        md.append(f"- Sharpe(core + {_f(c.get('k'), 2)} × overlay) {_f(c.get('portfolio_sharpe'), 2)} vs core "
                  f"{_f(c.get('core_sharpe'), 2)} on {c['n_days']} days: difference {_f(c['delta_ann'], 2)} "
                  f"(interval {_f(c['ci_ann'][0], 2)} … {_f(c['ci_ann'][1], 2)}), Ledoit–Wolf {c.get('sided')}-sided "
                  f"p = {_f(c['p'], 4)}; max drawdown {_f(c.get('dd_portfolio'), 3, pct=True)} vs "
                  f"{_f(c.get('dd_core'), 3, pct=True)} (difference {_f(dd.get('value'), 3, pct=True)}, interval "
                  f"{_f((dd.get('ci') or [None, None])[0], 3, pct=True)} … "
                  f"{_f((dd.get('ci') or [None, None])[1], 3, pct=True)})")  # fmt: skip
    else:
        md.append(f"- Sharpe {_f(c['sharpe'], 2)} vs benchmark {_f(c['bench_sharpe'], 2)} on {c['n_days']} days "
                  f"(excess returns); difference {_f(c['delta_ann'], 2)} (interval {_f(c['ci_ann'][0], 2)} … "
                  f"{_f(c['ci_ann'][1], 2)}), Ledoit–Wolf {c.get('sided', 'two')}-sided p = {_f(c['p'], 4)} "
                  f"(block {block} sessions)")  # fmt: skip
    lk = result.get("looks") or {}
    md.append(f"- Looks: K = {lk.get('K', '—')} configurations examined on the development window "
              f"(families/looks.jsonl); Bonferroni bound p × K = {_f(lk.get('bonferroni_p'), 4)}")  # fmt: skip
    md.append(f"- PSR(0) {_f(h.get('psr0'), 3)}; DSR {_f(v.get('dsr'), 3)} (N = {v.get('dsr_n')} program trials, "
              f"V = {_f(v.get('dsr_v'), 6)}) — a ledger diagnostic, not part of the verdict")  # fmt: skip
    if "alpha" in c:
        a = c["alpha"]
        md.append(f"- Alpha {_f(a['alpha_ann'], 3, pct=True)}/yr (NW t {_f(a['alpha_t'], 2)}, p {_f(a['alpha_p'], 4)}),"
                  f" beta {_f(a['beta'], 2)}")  # fmt: skip
    fl = h["floors"]
    md.append("- Floors: " + ("; ".join(f"{k} {_f(x.get('value'), 3)} vs {_f(x.get('threshold'), 3)} → "
                                       f"{'ok' if x['ok'] else 'FAIL'}" for k, x in fl.items() if k != "ok")
                              or "none set"))  # fmt: skip
    vcoh = v.get("variant_coherence") if v.get("coherence_kind") == "cells" else coh
    if v.get("coherence_kind") == "cells":
        md.append(f"- Coherence (cells, the verdict's): {coh.get('n_cells')} (instrument, cell) points, share with the "
                  f"headline's sign {_f(coh.get('share'), 3)} (needs {_f(t.get('coherence_share'), 3)}), median cell "
                  f"{_f(coh.get('median_alpha'), 2)} bp/day → {'coherent' if coh.get('coherent') else 'NOT coherent'}")  # fmt: skip
    md.append(f"- Coherence ({'variants, reported' if v.get('coherence_kind') == 'cells' else 'variants'}): "
              f"{vcoh.get('n_variants')} variants, share with the headline's sign {_f(vcoh.get('share'), 2)}"
              f", median variant {vcoh.get('median_variant')} (floors {_f(vcoh.get('median_floors_ok'))}) → "
              f"{'coherent' if vcoh['coherent'] else 'NOT coherent'}")  # fmt: skip
    md.append(f"- Before Holm across families: p {_f(v['p'], 4)}, floors {_f(v['floors_ok'])}, coherent "
              f"{_f(coh['coherent'])}, positive {_f(v['positive'])} (at cost {v.get('at_cost', 'registered')})")  # fmt: skip
    md.append("")
    cc = result.get("cost_curve") or {}
    if cc:
        md += ["## Cost curve (every variant re-priced from its cost ledger; the verdict reads one column)", ""]
        cols = list(cc)
        rows = []
        for label in ev["variants"]:
            if label not in cc.get(cols[0], {}):
                continue
            row = {"variant": label}
            for ck in cols:
                e = cc[ck].get(label)
                if e is None or "unavailable" in e:
                    row[ck] = "—"
                    continue
                st = e["stats"]
                cell = f"SR {_f(st['sharpe'], 2)}, ret {_f(st['ret_ann'], 3, pct=True)}, floors {_f(e['floors']['ok'])}"
                if "compare" in e:
                    cell = f"{_f(e['compare']['delta_ann'], 3)} (p {_f(e['compare']['p'], 3)}); " + cell
                row[ck] = cell
            rows.append(row)
        md += [_table(rows, ["variant", *cols]), ""]
        note = ("Columns: `registered` = the simulated costs (quotes half-spread + slippage per side); a number = "
                "that round-trip cost in bp on every fill; `measured` = the forward test's fill reconciliation. The "
                "headline column shows its statistic and p first.")  # fmt: skip
        md += [note, ""]
    acc = result.get("account") or {}
    if acc:
        md += ["## Account check (risk/account.py; nothing here changes a P&L)", ""]
        trad = acc.get("tradable")
        md.append(f"- {acc.get('profile')} ({acc.get('kind')} account, equity {acc.get('equity', 0):,.0f}): "
                  + ("tradable" if trad else ("not checked" if trad is None else "NOT tradable")))  # fmt: skip
        for r in acc.get("reasons", []):
            md.append(f"  - {r}")
        dt, bp = acc.get("day_trades") or {}, acc.get("buying_power") or {}
        if dt:
            md.append(f"- Day trades: {dt.get('total')} in total, at most {_f(dt.get('max_in_window'), 0)} in 5 "
                      f"sessions; PDT rule applies: {_f(dt.get('pdt_applies'))}; first flag {dt.get('first_pdt_flag')}")  # fmt: skip
        if bp:
            md.append(f"- Gross exposure: {_f(bp.get('max_gross_intraday'), 2)}× intraday (limit "
                      f"{_f(bp.get('limit_intraday'), 0)}×), {_f(bp.get('max_gross_overnight'), 2)}× overnight (limit "
                      f"{_f(bp.get('limit_overnight'), 0)}×); first breach {bp.get('first_breach')}")  # fmt: skip
        co = acc.get("cost_otherwise") or {}
        if co:
            md.append(f"- What the PDT rule would cost on an account under the floor: {_f(co.get('share_of_day_trades'), 3, pct=True)} "
                      f"of the day trades blocked ({_f(co.get('blocked_day_trades'), 0)})")  # fmt: skip
        lo = acc.get("locate") or {}
        if lo.get("bps"):
            md.append(f"- Locate fees at {lo['bps']} bp/yr on short notional: ≈ {_f(lo.get('cost_frac_per_year'), 4, pct=True)}/yr of equity (estimate)")  # fmt: skip
        md.append("")
    fb = result.get("auction_fallbacks") or {}
    notes = result.get("data_notes") or {}
    counters = (result.get("counters") or {}).get(HEADLINE) or {}
    if any(fb.values()) or notes or counters:
        md += ["## Data notes", ""]
        if counters:
            md.append(f"- Headline members: {counters.get('asof_before_first', 0)} session(s) priced with the first "
                      f"as-of cost table (2016Q1), {counters.get('stress_fills', 0)} bar(s) in stress sessions, "
                      f"{counters.get('cash_yield_missing_sessions', 0)} session(s) without a cash-yield rate")  # fmt: skip
        if any(fb.values()):
            md.append(
                "- Auction fills priced at the bar (no print cached): "
                + ", ".join(f"{k} {n}" for k, n in fb.items() if n)
            )
        for k, d in notes.items():
            for kk, vv in d.items():
                md.append(f"- {k}: {kk}: {vv}")
        md.append("")
    md += _region_md(result)
    md += _trade_md(result)
    md += ["## Specification curve (reported; no variant is selected)", "", "![spec curve](spec_curve.png)", ""]
    rows = []
    for k, x in ev["variants"].items():
        if "compare" not in x:
            rows.append({"variant": k, "status": x.get("status")})
            continue
        st = x["stats"]
        rows.append({"variant": k, "status": "ok", "start": x.get("start"), "days": x["compare"]["n_days"],
                     "sharpe": _f(x["compare"]["sharpe"], 2), "Δ vs bench": _f(x["compare"]["delta_ann"], 2),
                     "Δ on headline days": "—" if k == HEADLINE else _f(x.get("delta_common"), 2),
                     "CI": f"{_f(x['compare']['ci_ann'][0], 2)} … {_f(x['compare']['ci_ann'][1], 2)}",
                     "p": _f(x["compare"]["p"], 4), "net ret": _f(st["ret_ann"], 3, pct=True),
                     "max dd": _f(st["max_dd"], 3, pct=True), "floors": _f(x["floors"]["ok"])})  # fmt: skip
    cols = ["variant", "status", "start", "days", "sharpe", "Δ vs bench", "CI", "p", "Δ on headline days", "net ret",
            "max dd", "floors"]  # fmt: skip
    md += [_table(rows, cols), ""]
    note = ("Coherence reads each variant's Δ on the days it shares with the headline (column 'Δ on headline days'); "
            "the curve and the p-values are each variant's own sample.")  # fmt: skip
    md += [note, ""]
    if result.get("skipped_segments"):
        md += ["**Skipped segments** (state could not be fit: no events there, flat days): "
               + "; ".join(f"{k}: {v}" for k, v in result["skipped_segments"].items()), ""]  # fmt: skip
    d = result.get("described", {})
    md += ["## Responses", ""]
    resp = d.get("responses", {})
    names = sorted({n for r in resp.values() for n in r})
    md += [
        _table([{"variant": k, **{n: _resp(r.get(n)) for n in names}} for k, r in resp.items()], ["variant", *names]),
        "",
    ]
    ac = d.get("at_cost")
    md += [f"Responses, splits and slices below: excess of the T-bill{f', at cost {ac}' if ac else ''}.", ""]
    md += ["## State splits (headline; reported with 95 % Newey–West intervals, not tested)", ""]
    for k, s in d.get("state_splits", {}).items():
        if s.get("unavailable"):
            md += [f"- {k}: unavailable", ""]
            continue
        md += [f"**{k}**", "", _table([{**g, "mean_bp": _f(g["mean_bp"], 2), "ci_bp": f"{_f(g['ci_bp'][0], 2)} … "
                                        f"{_f(g['ci_bp'][1], 2)}", "sharpe": _f(g["sharpe"], 2)}
                                       for g in s["groups"]], ["group", "n_days", "mean_bp", "ci_bp", "sharpe"]), ""]  # fmt: skip
    if d.get("sample_splits"):
        md += ["## Sample splits (headline; reported)", ""]
        md += [_table([{"split": k, **_slice_row(s)} for k, s in d["sample_splits"].items()],
                      ["split", "n_days", "sharpe", "bench_sharpe", "ret", "max_dd", "mean_bp"]), ""]  # fmt: skip
    if d.get("quasi_holdout"):
        q = d["quasi_holdout"]
        md += ["## Quasi-holdout slice (reported; never a gate)", "", f"*{q['note']}*", "",
               _table([{"slice": "2025-10 → 2026-09", **_slice_row(q)}],
                      ["slice", "n_days", "sharpe", "bench_sharpe", "ret", "max_dd", "mean_bp"]), ""]  # fmt: skip
    if d.get("per_instrument"):
        md += ["## Per-instrument headline Sharpe (raw and James–Stein shrunk toward the family mean)", "",
               _table([{"instrument": k, "sharpe": _f(x["sharpe"], 2), "sharpe_js": _f(x["sharpe_js"], 2)}
                       for k, x in d["per_instrument"].items()], ["instrument", "sharpe", "sharpe_js"]), ""]  # fmt: skip
    md += ["## Registered vs run spec", "", "```", spec_diff(result), "```", ""]
    return "\n".join(md)


def _resp(x) -> str:
    if isinstance(x, dict):  # crisis_return: window → return
        return "; ".join(f"{k}: {_f(v, 3, pct=True)}" for k, v in x.items())
    return _f(x, 3)


def _slice_row(s: dict) -> dict:
    if s.get("unavailable"):
        return {"n_days": "unavailable"}
    return {"n_days": s["n_days"], "sharpe": _f(s["sharpe"], 2), "bench_sharpe": _f(s["bench_sharpe"], 2),
            "ret": _f(s["ret"], 3, pct=True), "max_dd": _f(s["max_dd"], 3, pct=True), "mean_bp": _f(s["mean_bp"], 2)}  # fmt: skip


def _html(title: str, md: str) -> str:
    """Markdown → a plain self-contained HTML page (tables, headings, lists, code, images; nothing else needed)."""
    out, table, code = [], [], False
    for ln in md.splitlines():
        if ln.startswith("```"):
            out.append("</pre>" if code else "<pre>")
            code = not code
            continue
        if code:
            out.append(html.escape(ln))
            continue
        if ln.startswith("|"):
            if set(ln.replace("|", "").strip()) <= {"-"}:
                continue
            cells = [html.escape(c.strip()) for c in ln.strip("|").split("|")]
            tag = "th" if not table else "td"
            table.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if table:
            out.append("<table>" + "".join(table) + "</table>")
            table = []
        if ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{html.escape(ln[n:].strip())}</h{n}>")
        elif ln.startswith("!["):
            src = ln[ln.index("(") + 1 : ln.rindex(")")]
            out.append(f'<img src="{html.escape(src)}" alt="">')
        elif ln.startswith("- "):
            out.append(f"<li>{html.escape(ln[2:])}</li>")
        elif ln.strip():
            out.append(f"<p>{html.escape(ln)}</p>")
    if table:
        out.append("<table>" + "".join(table) + "</table>")
    css = ("body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111;"
           "background:#fff}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ddd;padding:3px 8px;"
           "text-align:right}th{background:#f3f4f6}img{max-width:100%}pre{background:#f3f4f6;padding:8px}")  # fmt: skip
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}"
            f"</style></head><body>{''.join(out)}</body></html>")  # fmt: skip


def write_family_report(result: dict, out: Path) -> dict:
    out = Path(out)
    spec_curve(result, out / "spec_curve.png")
    cells_curve(result, out / "cells_curve.png")
    md = family_markdown(result)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(_html(f"Family {result['id']}", md), encoding="utf-8")
    return {"md": out / "report.md", "html": out / "report.html"}


UNTESTED = {"p": 1.0, "delta_ann": float("nan"), "floors_ok": False, "coherence": {"coherent": False},
            "positive": False}  # fmt: skip


def program_summary(out_dir, alpha: float = 0.05, ledger=None, program=None) -> dict:
    """
    The program verdict (SPEC §17.3, §22): Holm across the families' headline p-values, floors, coherence →
    program_summary.md / .html / .csv. The families are every <out_dir>/<id>/result.json and, with `ledger`, every
    family with a stage-F variant row there, restricted to the program's listed families when `program` (a
    program.yaml path with a `families` list; the CLI passes families/program.yaml) is given; one without a completed
    test (an errored headline, a missing result file) enters Holm with p = 1, so a family cannot leave the
    multiplicity count by failing or being deleted.
    """
    from families.run import KIND
    from families.test import program_verdict

    out_dir = Path(out_dir)
    results = {}
    from families.test import family_order

    for p in sorted(out_dir.glob("*/result.json"), key=lambda p: family_order(p.parent.name)):
        r = json.loads(p.read_text(encoding="utf-8"))
        results[r["id"]] = r
    in_ledger = set()
    if ledger is not None:
        in_ledger = {r["family"] for r in ledger.rows() if r.get("kind") == KIND and r.get("stage") == "F"}
    # U22: the program's families only (families/program.yaml `families`); the PLAN2 families are a closed program
    from families.run import program_caps

    caps = program_caps(program) if program is not None else {}
    listed = caps.get("families")
    if listed is not None:
        results = {k: r for k, r in results.items() if k.split(".v")[0] in listed}
        in_ledger = {f for f in in_ledger if f.split(".v")[0] in listed}
    verdicts = {k: dict(r["evaluation"]["verdict"]) for k, r in results.items() if "evaluation" in r}
    dsr_n = None
    if ledger is not None:  # DSR with the program's trials now, not those counted when each family ran
        from families.run import program_dsr

        rows_ = ledger.rows()
        for k, v in verdicts.items():
            f = out_dir / k / "streams.csv"
            if f.exists():
                h = pd.read_csv(f, index_col=0, parse_dates=True)[HEADLINE].dropna()
                v["dsr"], dsr_n, _ = program_dsr(rows_, h)
    untested = sorted((set(results) | in_ledger) - set(verdicts))
    verdicts |= {k: dict(UNTESTED) for k in untested}
    table = program_verdict(verdicts, alpha)
    intro = (f"{len(verdicts)} families, {len(verdicts) - len(untested)} tested"
             + (f" (program {caps.get('program')}: {', '.join(listed)})" if listed else "")
             + f". A family passes iff its Holm-adjusted headline p < {alpha}, its headline clears every floor, it is "
             "coherent and its statistic is positive (SPEC §17.3, §22). The DSR column is a ledger diagnostic.")  # fmt: skip
    md = ["# Hypothesis-family program: summary", "", intro, ""]
    acc = max((r.get("accounting", {}) for r in results.values()), key=lambda a: a.get("program_trials", 0),
              default={})  # fmt: skip
    md.append(f"Trial accounting: {acc.get('program_trials', 0)} / {acc.get('max_trials', '—')} program trials, "
              f"{acc.get('program_families', 0)} / {acc.get('max_families', '—')} families.")  # fmt: skip
    md.append("")
    rows = []
    for _, t in table.iterrows():
        r = results.get(t["family"], {})
        if "evaluation" not in r:
            why = r.get("error", "no result file") if r else "no result file"
            rows.append({"family": t["family"], "p": "1 (untested)", "p_holm": _f(t["p_holm"], 4),
                         "verdict": f"no test ({why})"})  # fmt: skip
            continue
        h = r["evaluation"]["variants"][HEADLINE]["compare"]
        rows.append({"family": t["family"], "kind": t.get("kind", "sharpe_vs_benchmark"), "sharpe": _f(h["sharpe"], 2),
                     "bench": _f(h["bench_sharpe"], 2), "statistic": _f(t["delta_ann"], 3),
                     "CI": f"{_f(h['ci_ann'][0], 2)} … {_f(h['ci_ann'][1], 2)}",
                     "p": _f(t["p"], 4), "p_holm": _f(t["p_holm"], 4), "floors": _f(bool(t["floors_ok"])),
                     "coherence": (f"{_f(r['evaluation']['verdict']['coherence'].get('share'), 2)} "
                                   f"({'yes' if r['evaluation']['verdict']['coherence'].get('coherent') else 'no'})"),
                     "DSR": _f(verdicts[t["family"]].get("dsr"), 3), "verdict": "PASS" if t["passes"] else "fail"})  # fmt: skip
    md += [_table(rows, ["family", "kind", "sharpe", "bench", "statistic", "CI", "p", "p_holm", "floors", "coherence",
                         "DSR", "verdict"]), ""]  # fmt: skip
    md += [("Coherence: share of variants with the headline's sign (coherent or not: the median variant must also "
            "clear the floors). DSR: " + (f"N = {dsr_n} program trials now (SPEC §17.3)." if dsr_n else
                                          "as recorded when each family ran.")), ""]  # fmt: skip
    md += ["Per-family reports: " + ", ".join(f"[{k}]({k}/report.md)" for k in results), ""]
    text = "\n".join(md)
    (out_dir / "program_summary.md").write_text(text, encoding="utf-8")
    (out_dir / "program_summary.html").write_text(_html("Family program summary", text), encoding="utf-8")
    table.to_csv(out_dir / "program_summary.csv", index=False)
    return {"md": out_dir / "program_summary.md", "table": table}


def load_result(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stream_frame(out: Path) -> pd.DataFrame:
    return pd.read_csv(Path(out) / "streams.csv", index_col=0, parse_dates=True)
