import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]  # the repo root
sys.path.insert(0, str(REPO))
from families import test as T

FAMS = ["F1", "F2", "F3", "F4", "F5", "F7", "F8", "F11"]
ROWS = [json.loads(l) for l in open(REPO / "results/ledger.jsonl", encoding="utf-8")]
BYH = {r.get("cell_hash"): r for r in ROWS}


def streams(f):
    d = pd.read_csv(REPO / f"results/families/{f}/streams.csv", index_col=0, parse_dates=True)
    return d


def pair(f):
    d = streams(f)
    return T.align(d["headline"].dropna(), d["benchmark"].dropna())


def result(f):
    return json.load(open(REPO / f"results/families/{f}/result.json", encoding="utf-8"))


def per_instrument(f, label="headline"):
    h = [r for r in ROWS if r.get("label") == f"{f}:{label}"][-1]
    by = {}
    for m in h["members"]:
        sym = json.loads(BYH[m]["spec_json"])["symbols"]
        sym = sym[0] if len(sym) == 1 else "_basket"
        s = pd.read_csv(REPO / f"results/experiments/cells/{m}/daily_returns.csv", index_col=0)["ret"]
        s.index = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        by.setdefault(sym, []).append(T.day_index(s))
    return {k: pd.concat(v, axis=1, sort=True).fillna(0.0).sum(axis=1) for k, v in by.items()}, h
