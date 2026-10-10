import contextlib
import io

from common import *

from data.bars import get_calendar

cal = get_calendar("2025-10-01")
cal = cal.loc["2016-01-04":"2025-09-30"]
for sym in ["SPY", "QQQ", "IWM", "DIA", "TLT", "XLK", "UUP"]:
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        df = bars(sym, "5Min")
    day = df.index.tz_localize(None).normalize()
    cnt = pd.Series(1, index=day).groupby(level=0).size()
    exp = ((cal.session_close - cal.session_open).dt.total_seconds() / 300).astype(int)
    exp = exp.reindex(cnt.index)
    missing_sessions = cal.index.difference(cnt.index)
    short = cnt[cnt < exp]
    first = df.index.to_series().groupby(day).first().dt.strftime("%H:%M")
    last = df.index.to_series().groupby(day).last().dt.strftime("%H:%M")
    early = cal[(cal.session_close.dt.hour == 13)].index
    print(
        f"{sym}: sessions {len(cnt)}/{len(cal)} missing {list(missing_sessions.date)}; sessions w/ missing bars {len(short)} "
        f"(bars missing total {int((exp[short.index] - short).sum())}); first!=09:30: {int((first != '09:30').sum())}; "
        f"last!=15:55 on full days: {int(((last != '15:55') & ~last.index.isin(early)).sum())}; early closes {len(early)} last bar {sorted(set(last[last.index.isin(early)]))}; "
        f"log: {f.getvalue().strip()[:200]}"
    )
# DST check: tz-aware stamps of first bar around DST changes
df = bars("SPY", "5Min")
for d in ["2016-03-11", "2016-03-14", "2016-11-04", "2016-11-07", "2024-03-08", "2024-03-11"]:
    x = df.loc[d]
    print(d, x.index[0], x.index[-1], len(x), x.index[0].tz_convert("UTC").strftime("%H:%M UTC"))
