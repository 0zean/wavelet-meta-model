import json

from common import *

from families import test as T

for fam in ["F1", "F3", "F5", "F7"]:
    r = json.load(open(f"results/families/{fam}/result.json"))
    s = pd.read_csv(f"results/families/{fam}/streams.csv", index_col=0, parse_dates=True)
    spec_test = r["spec"].get("test", {}) if isinstance(r["spec"], dict) else {}
    test = {"alpha": 0.05, "block_days": 21, "n_boot": 2000, "seed": spec_test.get("seed", 0), **spec_test}
    out = T.compare(s["headline"].dropna(), s["benchmark"].dropna(), "buy_and_hold_er", test)
    ev = r["evaluation"]["variants"]["headline"]["compare"]
    print(
        fam,
        "repro sharpe %.4f delta %.4f p %.4f | stored sharpe %.4f delta %.4f p %.4f"
        % (out["sharpe"], out["delta_ann"], out["p"], ev["sharpe"], ev["delta_ann"], ev["p"]),
    )
