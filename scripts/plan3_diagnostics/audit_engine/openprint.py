import contextlib
import io

from common import *

for sym in ["SPY", "IWM", "AAPL"]:
    with contextlib.redirect_stdout(io.StringIO()):
        df = bars(sym, "5Min")
    day = df.index.normalize()
    same = np.r_[False, day[1:] == day[:-1]]
    j = np.abs(np.log(df.open.to_numpy()[1:] / df.close.to_numpy()[:-1]))[same[1:]] * 1e4
    print(
        sym,
        "|open[t+1]/close[t]| bp: median %.2f mean %.2f p90 %.2f p99 %.2f"
        % (np.median(j), j.mean(), np.percentile(j, 90), np.percentile(j, 99)),
    )
