"""
U22 done-when checks that read the real caches, the ledger or the stored family results (PLAN3 §5 U22). Each
subcommand prints what the status note quotes; nothing here writes to the canonical ledger.

    uv run python scripts/u22_checks.py prints          # print vs 15:55-bar close on the cached SPY/QQQ/IWM/DIA history
    uv run python scripts/u22_checks.py f5-parity ROOT  # F5's headline cells under the old switches vs the U18 files
    uv run python scripts/u22_checks.py print-rerun ROOT [F3 F5]   # the headline cells under FILL_AUCTION print
    uv run python scripts/u22_checks.py power [--sims 1000]        # size and power of overlay_alpha (iid, GARCH-t)
    uv run python scripts/u22_checks.py tilt            # benchmark Sharpe raw vs excess on the U18 family streams
    uv run python scripts/u22_checks.py account         # F5's registered headline on $10k / $30k / a cash account
    uv run python scripts/u22_checks.py mde             # the G1 MDE line of PLAN3 (2,400 days, 8 % vol, 4 families)

`f5-parity` and `print-rerun` run the member cells through experiments.runner into a scratch ROOT (its own ledger),
so they take minutes per cell (the rule pass with a time-of-day profile over ten years of 5Min bars).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OLD = {"FILL_AUCTION": "last_bar", "COST_TABLE": "year", "CASH_YIELD": "none", "STRESS_MULT": 1.0,
       "SLIPPAGE_BP": {"open": 1.0, "intra": 1.0, "close": 1.0}, "MOC_SIZE_FROM": "fill",
       "SESSION_CLOCK": "data"}  # fmt: skip


def _rows():
    text = (ROOT / "results/ledger.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _headline_members(fam: str) -> list[tuple[str, dict]]:
    rows = _rows()
    h = [r for r in rows if r.get("label") == f"{fam}:headline"][-1]
    out = []
    for m in h["members"]:
        spec = _json(ROOT / f"results/experiments/cells/{m}/spec.json")["spec"]
        out.append((m, spec))
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _stream(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0)["ret"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    return s


def cmd_prints(a):
    from data.bars import load_bars, load_prints

    for sym in a.symbols:
        b = load_bars(sym, "5Min", "2016-01-01", "2026-10-01")
        p = load_prints(sym, "2016-01-01", "2026-10-01")
        day = b.index.normalize()
        j = pd.concat([b["close"].groupby(day).last().rename("bar_close"), p["close"].rename("print_close"),
                       b["open"].groupby(day).first().rename("bar_open"), p["open"].rename("print_open")], axis=1,
                      sort=True).dropna()  # fmt: skip
        dc = (j.print_close / j.bar_close - 1) * 1e4
        do = (j.print_open / j.bar_open - 1) * 1e4
        print(f"{sym}: {len(j)} sessions | close print vs 15:55 bar: median |Δ| {dc.abs().median():.2f} bp, mean "
              f"{dc.mean():+.3f} bp, share positive {(dc > 0).mean():.2f}, p99 |Δ| {dc.abs().quantile(0.99):.1f} bp, "
              f"max |Δ| {dc.abs().max():.0f} bp on {dc.abs().idxmax().date()} | open print vs 09:30 bar: median |Δ| "
              f"{do.abs().median():.2f} bp, p99 {do.abs().quantile(0.99):.1f} bp, max {do.abs().max():.0f} bp on "
              f"{do.abs().idxmax().date()}")  # fmt: skip
    for d in ("2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"):
        b = load_bars("SPY", "5Min", d, str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()))
        p = load_prints("SPY", d, str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()))
        print(f"SPY {d}: 15:55 close {b['close'].iloc[-1]:.2f} vs print {p['close'].iloc[0]:.2f} "
              f"({(p['close'].iloc[0] / b['close'].iloc[-1] - 1) * 1e4:+.2f} bp); 09:30 open {b['open'].iloc[0]:.2f} vs print "
              f"{p['open'].iloc[0]:.2f} (adjusted prices; the audit's raw 2024-03-04 values were 512.25 / 512.30)")  # fmt: skip


def _run_cells(fam: str, root: Path, overrides: dict, jobs: int) -> dict[str, str]:
    """Run the family's headline member cells with `overrides` into a scratch root; returns member → new hash."""
    from experiments import ledger as L
    from experiments.runner import run
    from experiments.spec import Cell, normalize

    cells, mine = [], {}
    for m, spec in _headline_members(fam):
        raw = {k: v for k, v in spec.items()}
        raw["overrides"] = {**raw.get("overrides", {}), **overrides}
        cell = Cell(normalize(raw), "F")
        cells.append(cell)
        mine[cell.spec_json()] = m
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    run(cells, ledger=L.Ledger(root / "ledger.jsonl"), root=root, jobs=jobs, spec_name=f"u22_{fam}", family=True,
        on_hash=lambda c, h: hashes.__setitem__(mine[c.spec_json()], h))  # fmt: skip
    return hashes


def cmd_f5_parity(a):
    root = Path(a.root)
    new = _run_cells(a.family, root, OLD, a.jobs)
    bad = 0
    for m, h in new.items():
        old_f, new_f = (
            ROOT / f"results/experiments/cells/{m}/daily_returns.csv",
            root / "cells" / h / "daily_returns.csv",
        )
        if not new_f.exists():
            print(f"MISSING  {m} → {h}")
            bad += 1
            continue
        same = _sha(old_f) == _sha(new_f)
        if not same:
            d = (_stream(old_f) - _stream(new_f)).abs()
            print(f"MISMATCH {m} → {h}: max |Δ daily return| {d.max():.3e} on {d.idxmax().date()}")
        else:
            print(f"OK       {m} → {h}: daily_returns.csv identical (sha256 {_sha(new_f)[:16]})")
        bad += not same
    print(f"{a.family} parity under the old switches:", "PASS" if not bad else f"FAIL ({bad})")


def cmd_print_rerun(a):
    from families import test as T

    root = Path(a.root)
    for fam in a.families:
        res = _json(ROOT / f"results/families/{fam}/result.json")
        w = res["weights"]
        new = _run_cells(fam, root, {**OLD, "FILL_AUCTION": "print"}, a.jobs)  # the print alone changes
        old_s, new_s = {}, {}
        text = (root / "ledger.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        for m, spec in _headline_members(fam):
            sym = spec["symbols"][0]
            old_s[sym] = _stream(ROOT / f"results/experiments/cells/{m}/daily_returns.csv")
            new_s[sym] = _stream(root / "cells" / new[m] / "daily_returns.csv")
            r = [x for x in rows if x.get("cell_hash") == new[m]][-1]
            print(f"{fam} {sym}: Sharpe {T.sharpe_ann(old_s[sym]):.3f} (U18, 15:55 close) → {T.sharpe_ann(new_s[sym]):.3f} "
                  f"(print); auction fallbacks {r.get('auction_fallbacks')}; trades {r.get('n_trades')}")  # fmt: skip
        ws = pd.Series(w)
        po, pn = T.pool(old_s, ws), T.pool(new_s, ws)
        j = pd.concat([po.rename("old"), pn.rename("new")], axis=1).dropna()
        print(f"{fam} pooled headline: Sharpe {T.sharpe_ann(j.old):.3f} → {T.sharpe_ann(j.new):.3f} "
              f"(Δ {T.sharpe_ann(j.new) - T.sharpe_ann(j.old):+.3f}); mean {j.old.mean() * 1e4:+.3f} → "
              f"{j.new.mean() * 1e4:+.3f} bp/day; max |Δ daily| {(j.new - j.old).abs().max() * 1e4:.2f} bp")  # fmt: skip


def cmd_power(a):
    from families.power import mde_alpha, simulate_power

    n, vol = 2400, 0.08
    m1 = mde_alpha(n, vol)
    print(f"MDE (1 family, one-sided 5 %, 80 % power): {m1['mde_bp_per_day']:.3f} bp/day")
    for stream in ("iid", "garch_t"):
        s = simulate_power(n, vol, 0.0, n_sims=a.sims, n_boot=a.n_boot, stream=stream, seed=1)
        print(f"size ({stream}, {a.sims} sims, n_boot {a.n_boot}): {s['rejection_rate']:.3f}")
    p = simulate_power(n, vol, 3.0, n_sims=min(a.sims, 400), n_boot=a.n_boot, stream="iid", seed=2)
    print(
        f"power at a planted 3 bp/day alpha, 8 % vol, 2,400 days (iid, {p['n_sims']} sims): {p['rejection_rate']:.3f}"
    )
    p2 = simulate_power(n, vol, m1["mde_bp_per_day"], n_sims=min(a.sims, 400), n_boot=a.n_boot, stream="iid", seed=3)
    print(f"power at the analytic MDE {m1['mde_bp_per_day']:.2f} bp/day: {p2['rejection_rate']:.3f} (target 0.80)")


def cmd_tilt(a):
    from experiments.runner import rate_asof
    from families import test as T

    try:
        from data.exo import load_series

        dtb3 = load_series("fred", "DTB3", "2015-12-01", "2026-10-01")
    except Exception as e:  # noqa: BLE001
        print(f"no DTB3 ({e})")
        return
    for fam in ("F1", "F2", "F3", "F4", "F5", "F7", "F8", "F11"):
        p = ROOT / f"results/families/{fam}/streams.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        b = d["benchmark"].dropna()
        h = d["headline"].dropna()
        sess = pd.DatetimeIndex(b.index)
        rf = T.rf_accrual(rate_asof(dtb3, sess) / 100.0, sess)
        print(f"{fam}: benchmark Sharpe raw {T.sharpe_ann(b):.3f} → excess {T.sharpe_ann(T.excess(b, rf)):.3f} "
              f"(tilt {T.sharpe_ann(b) - T.sharpe_ann(T.excess(b, rf)):+.3f}); headline raw {T.sharpe_ann(h):.3f} → "
              f"excess {T.sharpe_ann(T.excess(h, rf)):.3f}; mean rf {rf.mean() * 252:.2%}/yr")  # fmt: skip


def cmd_account(a):
    from risk.account import check_account, get_account

    res = _json(ROOT / "results/families/F5/result.json")
    w = res["weights"]
    stream = pd.read_csv(ROOT / "results/families/F5/streams.csv", index_col=0, parse_dates=True)["headline"].dropna()
    members = []
    for m, spec in _headline_members("F5"):
        tr = pd.read_csv(ROOT / f"results/experiments/cells/{m}/trades.csv")
        members.append((spec["symbols"][0], w[spec["symbols"][0]], tr, 10_000.0))
    for name in ("margin_10k", "margin_30k", "cash_30k"):
        c = check_account(members, stream, get_account(name))
        print(f"{name}: tradable {c['tradable']}; {c['reasons'] or 'no rule broken'}; day trades {c['day_trades']}; "
              f"gross {c['buying_power']['max_gross_intraday']:.2f}x; PDT would block "
              f"{c['cost_otherwise']['share_of_day_trades']:.1%} of the day trades")  # fmt: skip


def cmd_mde(a):
    from families.power import mde_alpha

    for fams in (1, 4, 5):
        m = mde_alpha(2400, 0.08, families=fams)
        print(
            f"{fams} families: MDE {m['mde_bp_per_day']:.2f} bp/day = {m['mde_ann']:.2%}/yr, Sharpe {m['mde_sharpe_ann']:.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prints")
    s.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "DIA"])
    s = sub.add_parser("f5-parity")
    s.add_argument("root")
    s.add_argument("--family", default="F5")
    s.add_argument("--jobs", type=int, default=4)
    s = sub.add_parser("print-rerun")
    s.add_argument("root")
    s.add_argument("families", nargs="*", default=["F3", "F5"])
    s.add_argument("--jobs", type=int, default=4)
    s = sub.add_parser("power")
    s.add_argument("--sims", type=int, default=1000)
    s.add_argument("--n-boot", type=int, default=999)
    sub.add_parser("tilt")
    sub.add_parser("account")
    sub.add_parser("mde")
    a = ap.parse_args()
    {"prints": cmd_prints, "f5-parity": cmd_f5_parity, "print-rerun": cmd_print_rerun, "power": cmd_power,
     "tilt": cmd_tilt, "account": cmd_account, "mde": cmd_mde}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    main()
