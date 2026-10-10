"""
U16 diagnostics (PLAN2 U16 done-criteria, SPEC §16): the mechanism primaries on cached SPY bars over the development
window.

    uv run python scripts/u16_primaries.py [--smoke <root>]

- Per-slot diagnostics for F5 (`intraday_momentum`) and F7 (`gap_fade`) on SPY 5Min, 2016-01-04 → 2025-09-30
  (primaries.diagnostics.slot_diagnostics on fold-free signals: the rules fit nothing, so every sampled event is
  sided as in any WFO fold). F5: the headline gate (|predictor| > 0.5 σ_day), both predictors, entry slots 15:00 /
  15:30 / 15:45 (its registered entry variants) → MOC. F7: the headline fade (|gap| > 1 σ_overnight, non-macro days,
  exit 10:30) split by entry slot 09:35 (the headline) / 09:45 / 10:00, and the news-day follow leg at 09:35. Each
  table gives per slot: sided events, the ISOM count (events per session), long share, precision (P(side return >
  META_MIN_RET)), opportunity, recall, turnover and net bp (side return − 2 · SLIPPAGE_PCT). A diagnostic, not a
  test and not a selection step: the families' headline entries are fixed in PLAN2 before this table exists.
- With --smoke: the U16 smoke cells' signal files (experiments/specs/u16_smoke.yaml run on a scratch root): the
  schema (the WFO signal columns) and the status of every cell.

Outputs: results/families/u16_slot_diagnostics.csv and .json (and results/u16/smoke_cells.json with --smoke).
Network: none (cached bars and the event table).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.bars import load_bars
from features.events import sample_events
from primaries import check_signal, make_primary
from primaries.diagnostics import slot_diagnostics
from primaries.mechanism import primary_config

START, END = "2016-01-04", "2025-10-01"
OUT = ROOT / "results" / "families"
WFO_COLUMNS = ["clf_prob", "direction", "signed_dir", "magnitude", "signal", "confidence", "meta_prob",
               "trade_signal", "width", "fold", "primary", "bet_size"]  # fmt: skip

VARIANTS = [
    ("F5", "intraday_momentum open_to_now", "intraday_momentum", {"predictor": "open_to_now"},
     {"entry_times": ["15:00", "15:30", "15:45"]}),
    ("F5", "intraday_momentum first30", "intraday_momentum", {"predictor": "first30"},
     {"entry_times": ["15:00", "15:30", "15:45"]}),
    ("F7", "gap_fade fade non_macro", "gap_fade", {}, {"entry_times": ["09:35", "09:45", "10:00"]}),
    ("F7", "gap_fade follow macro", "gap_fade", {"follow_on_news": True}, {}),
]  # fmt: skip


def fold_free_signals(df: pd.DataFrame, cfg, symbol: str) -> pd.DataFrame:
    ev = sample_events(df, cfg, symbol=symbol).dropna(subset=["width"])
    X = pd.DataFrame(index=ev.index)
    p = make_primary(cfg)
    fr = check_signal(p.signal(df, X, cfg), X, p)
    return fr.assign(width=ev["width"], fold=0, primary=cfg.PRIMARY)


def slot_tables(symbol: str = "SPY") -> pd.DataFrame:
    df = load_bars(symbol, "5Min", START, END)
    frames = []
    for family, label, name, params, ev_over in VARIANTS:
        base = primary_config(name, "5Min", params)
        cfg = base.replace(EVENT_PARAMS={**base.EVENT_PARAMS, **ev_over}) if ev_over else base
        sig = fold_free_signals(df, cfg, symbol)
        tab = slot_diagnostics(df, sig, cfg, symbol=symbol)
        tab.insert(0, "variant", label)
        tab.insert(0, "family", family)
        frames.append(tab)
        print(f"[U16]  {label}: {len(sig)} events")
    return pd.concat(frames, ignore_index=True)


def smoke_cells(root: Path) -> list[dict]:
    rows = []
    for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "cell_hash" not in r:
            continue
        cell = root / "cells" / r["cell_hash"]
        spec = json.loads((cell / "spec.json").read_text(encoding="utf-8"))["spec"]
        cols = {}
        for f in sorted(cell.glob("signals_*.csv")):
            sig = pd.read_csv(f, index_col=0, nrows=5)
            cols[f.stem.removeprefix("signals_")] = list(sig.columns) == WFO_COLUMNS
        rows.append({"cell_hash": r["cell_hash"], "status": r["status"], "label": r.get("label"),
                     "primary_params": spec["primary"]["params"], "event_params": spec["overrides"].get("EVENT_PARAMS"),
                     "exit_params": spec["overrides"].get("EXIT_PARAMS"), "n_oos_events": r.get("n_oos_events"),
                     "n_trades": r.get("n_trades"), "schema_unchanged": cols})  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", default=None, help="root of a u16_smoke.yaml run (scratch ledger)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tab = slot_tables()
    tab.to_csv(OUT / "u16_slot_diagnostics.csv", index=False)
    rec = json.loads(tab.replace({np.nan: None}).to_json(orient="records"))
    meta = {"symbol": "SPY", "timeframe": "5Min", "start": START, "end": END, "costs": "net_bp = side return − 2 bp",
            "note": "fold-free signals of fixed rules; diagnostic only, not a family test"}  # fmt: skip
    (OUT / "u16_slot_diagnostics.json").write_text(json.dumps({**meta, "rows": rec}, indent=2), encoding="utf-8")
    cols = ["family", "variant", "slot", "n_events", "per_session", "long_share", "precision", "net_bp"]
    print(tab[cols].round(4).to_string(index=False))
    print(f"[OUT]  {OUT / 'u16_slot_diagnostics.csv'}")
    if a.smoke:
        rows = smoke_cells(Path(a.smoke))
        out = ROOT / "results" / "u16"
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_cells.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        for r in rows:
            print(f"[SMOKE]  {r['status']:3s} {r['label']:60s} schema {r['schema_unchanged']} "
                  f"events {r['n_oos_events']}")  # fmt: skip
        print(f"[OUT]  {out / 'smoke_cells.json'}")


if __name__ == "__main__":
    main()
