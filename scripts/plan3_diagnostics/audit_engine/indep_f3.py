import contextlib
import io
import json

from common import *

from risk.costs import quotes_half_spread

r = json.load(open("results/families/F3/result.json"))
h = r["members"]["headline"][0]
js = json.load(open(f"results/experiments/cells/{h}/spec.json"))["spec"]
print("F3 headline spec primary", js["primary"], js["overrides"].get("EXIT_PARAMS"), js["symbols"])
sym = js["symbols"][0]
with contextlib.redirect_stdout(io.StringIO()):
    df = load_bars(sym, "5Min", js["start"], js["end"])
tbl = read_table().query("symbol == @sym")
day = df.index.tz_localize(None).normalize()
g = df.groupby(day)
first_t, last_t = g.apply(lambda x: x.index[0]), g.apply(lambda x: x.index[-1])
o = g["open"].first()
c = g["close"].last()
ok_close = last_t.dt.strftime("%H:%M").isin(["15:55", "12:55"])
ok_open = first_t.dt.strftime("%H:%M") == "09:30"
yrs = o.index.year.to_numpy()
ca = quotes_half_spread(yrs, np.full(len(yrs), "close_auction", object), tbl)
oa = quotes_half_spread(yrs, np.full(len(yrs), "open_auction", object), tbl)
E = 10000.0
eq = {}
days = o.index
for i in range(1, len(days)):
    eq[days[i - 1]] = eq.get(days[i - 1], E)
for i in range(len(days) - 1):
    pass
# compound: hold overnight d -> d+1 when close bar of d and open bar of d+1 are auction bars
E = 10000.0
out = {}
start = None
for i in range(len(days) - 1):
    d, n = days[i], days[i + 1]
    if ok_close.iloc[i] and ok_open.iloc[i + 1]:
        if start is None:
            start = d
        fill_in = c.iloc[i] * (1 + ca[i])
        q = E / fill_in
        E_close_d = q * c.iloc[i]  # marked at d's close
        out[d] = E_close_d
        fill_out = o.iloc[i + 1] * (1 - oa[i + 1])
        E = E + q * (fill_out - fill_in)
        out[n] = E  # flat at n's close
    else:
        out.setdefault(d, E)
        out[n] = E
s = pd.Series(out).sort_index()
s = s[s.index >= start]
mine = s / s.shift(1) - 1
stored = pd.read_csv(f"results/experiments/cells/{h}/daily_returns.csv", index_col=0)["ret"]
stored.index = pd.to_datetime(stored.index, utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
j = pd.concat([mine.rename("m"), stored.rename("s")], axis=1, sort=True).dropna()
print(
    sym, "independent vs engine: days", len(j), "max|diff|", (j.m - j.s).abs().max(), " mean diff", (j.m - j.s).mean()
)
print(j[(j.m - j.s).abs() > 1e-9].head())
