import contextlib
import io
import json

from common import *

from experiments.spec import Cell

q = io.StringIO()
for fam, idx in [("F7", 0), ("F3", 0), ("F5", 3)]:
    r = json.load(open(f"results/families/{fam}/result.json"))
    h = r["members"]["headline"][idx]
    js = json.load(open(f"results/experiments/cells/{h}/spec.json"))
    cell = Cell(stage="F", spec=js["spec"]) if "stage" in Cell.__dataclass_fields__ else Cell(js["spec"])
    cfg = cell.config()
    sym = js["spec"]["symbols"][0]
    with contextlib.redirect_stdout(q):
        df = load_bars(sym, js["spec"]["timeframe"], js["spec"]["start"], js["spec"]["end"])
        sig, eq, tr = run_rule(df, cfg, sym=sym)
    mine = daily_returns(eq)
    mine.index = mine.index.tz_localize(None)
    stored = pd.read_csv(f"results/experiments/cells/{h}/daily_returns.csv", index_col=0)["ret"]
    stored.index = pd.to_datetime(stored.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    j = pd.concat([mine.rename("m"), stored.rename("s")], axis=1, sort=True)
    print(
        fam, sym, h, "days mine/stored", mine.size, stored.size, "max|diff|", (j.m - j.s).abs().max(), "trades", len(tr)
    )
