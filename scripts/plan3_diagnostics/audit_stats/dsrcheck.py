from common import *

from families.run import program_dsr
from validation.stats import dsr, return_moments

for f in FAMS:
    h = T.day_index(streams(f)["headline"].dropna())
    d, n, v = program_dsr(ROWS, h)
    m = return_moments(h.to_numpy())
    srs = [r["sharpe"] for r in ROWS if r.get("kind") == "family_variant" and r.get("label", "").startswith(f + ":")]
    # headline-only accounting: N=8, V = sampling variance of a per-day SR at this T
    d8 = dsr(m.sr, 8, (1 + 0.5 * m.sr**2) / m.n_obs, m.n_obs, m.skew, m.kurt)
    print(f, "DSR(N=%d)=%.3f" % (n, d), " DSR(N=8,V=noise)=%.3f" % d8)
