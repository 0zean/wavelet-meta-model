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
    ax.set_ylabel("Sharpe (ann.)" if result.get("benchmark") == "cash" else "Δ Sharpe vs benchmark (ann.)")
    ax.set_title(f"{result['id']}: specification curve, 95 % intervals\n(headline in orange; variants reported, "
                 f"never selected; benchmark {result.get('benchmark')})", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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
    md.append(f"- Instruments: {', '.join(result['instruments'])} ({'basket' if result.get('basket') else 'pooled'}, "
              f"weights {weights}); window {result['window'][0]} → {result['window'][1]}; benchmark "
              f"{result.get('benchmark')}")  # fmt: skip
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
    md.append(f"- Sharpe {_f(c['sharpe'], 2)} vs benchmark {_f(c['bench_sharpe'], 2)} on {c['n_days']} days; difference "
              f"{_f(c['delta_ann'], 2)} (95 % CI {_f(c['ci_ann'][0], 2)} … {_f(c['ci_ann'][1], 2)}), Ledoit–Wolf p = "
              f"{_f(c['p'], 4)} (block {block} sessions)")  # fmt: skip
    md.append(f"- PSR(0) {_f(h.get('psr0'), 3)}; DSR {_f(v.get('dsr'), 3)} (N = {v.get('dsr_n')} program trials, "
              f"V = {_f(v.get('dsr_v'), 6)})")  # fmt: skip
    if "alpha" in c:
        a = c["alpha"]
        md.append(f"- Alpha {_f(a['alpha_ann'], 3, pct=True)}/yr (NW t {_f(a['alpha_t'], 2)}, p {_f(a['alpha_p'], 4)}),"
                  f" beta {_f(a['beta'], 2)}")  # fmt: skip
    fl = h["floors"]
    md.append("- Floors: " + ("; ".join(f"{k} {_f(x.get('value'), 3)} vs {_f(x.get('threshold'), 3)} → "
                                       f"{'ok' if x['ok'] else 'FAIL'}" for k, x in fl.items() if k != "ok")
                              or "none set"))  # fmt: skip
    md.append(f"- Coherence: {coh.get('n_variants')} variants, share with the headline's sign {_f(coh.get('share'), 2)}"
              f", median variant {coh.get('median_variant')} (floors {_f(coh.get('median_floors_ok'))}) → "
              f"{'coherent' if coh['coherent'] else 'NOT coherent'}")  # fmt: skip
    md.append(f"- Before Holm across families: p {_f(v['p'], 4)}, floors {_f(v['floors_ok'])}, coherent "
              f"{_f(coh['coherent'])}, positive {_f(v['positive'])}")  # fmt: skip
    md.append("")
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
    md = family_markdown(result)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(_html(f"Family {result['id']}", md), encoding="utf-8")
    return {"md": out / "report.md", "html": out / "report.html"}


UNTESTED = {"p": 1.0, "delta_ann": float("nan"), "floors_ok": False, "coherence": {"coherent": False},
            "positive": False}  # fmt: skip


def program_summary(out_dir, alpha: float = 0.05, ledger=None) -> dict:
    """
    The program verdict (SPEC §17.3): Holm across the families' headline p-values, floors, coherence →
    program_summary.md / .html / .csv. The families are every <out_dir>/<id>/result.json and, with `ledger`, every
    family with a stage-F variant row there; one without a completed test (an errored headline, a missing result
    file) enters Holm with p = 1, so a family cannot leave the multiplicity count by failing or being deleted.
    """
    from families.run import KIND
    from families.test import program_verdict

    out_dir = Path(out_dir)
    results = {}
    for p in sorted(out_dir.glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        results[r["id"]] = r
    in_ledger = set()
    if ledger is not None:
        in_ledger = {r["family"] for r in ledger.rows() if r.get("kind") == KIND and r.get("stage") == "F"}
    verdicts = {k: r["evaluation"]["verdict"] for k, r in results.items() if "evaluation" in r}
    untested = sorted((set(results) | in_ledger) - set(verdicts))
    verdicts |= {k: dict(UNTESTED) for k in untested}
    table = program_verdict(verdicts, alpha)
    intro = (f"{len(verdicts)} families, {len(verdicts) - len(untested)} tested. A family passes iff its Holm-adjusted "
             f"headline p < {alpha}, its headline clears every floor, it is coherent and its difference is positive "
             "(SPEC §17.3).")  # fmt: skip
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
        rows.append({"family": t["family"], "sharpe": _f(h["sharpe"], 2), "bench": _f(h["bench_sharpe"], 2),
                     "Δ": _f(t["delta_ann"], 2), "CI": f"{_f(h['ci_ann'][0], 2)} … {_f(h['ci_ann'][1], 2)}",
                     "p": _f(t["p"], 4), "p_holm": _f(t["p_holm"], 4), "floors": _f(bool(t["floors_ok"])),
                     "coherence": _f(r["evaluation"]["verdict"]["coherence"].get("share"), 2),
                     "DSR": _f(t["dsr"], 3), "verdict": "PASS" if t["passes"] else "fail"})  # fmt: skip
    md += [_table(rows, ["family", "sharpe", "bench", "Δ", "CI", "p", "p_holm", "floors", "coherence", "DSR",
                         "verdict"]), ""]  # fmt: skip
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
