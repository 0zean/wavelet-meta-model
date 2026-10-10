import glob

from common import *

from data.alpaca_source import NY_TZ
from data.quotes import AUCTION_BINS, EVENT_SAMPLES_DIR, SAMPLES_DIR, half_spread_bp, load_samples, read_table, tod_bin

days = sorted(pd.Timestamp(p.split("\\")[-1].split("/")[-1][:10]) for p in glob.glob(str(SAMPLES_DIR / "*.npz")))
S = pd.concat([load_samples(SAMPLES_DIR, d) for d in days], ignore_index=True)
S["hs"] = half_spread_bp(S)
S["year"] = S["mark"].dt.tz_convert(NY_TZ).dt.year
S["bin"] = [l if l in AUCTION_BINS else tod_bin(l) for l in S["label"]]
print(
    "sessions",
    len(days),
    "samples",
    len(S),
    "per-year sessions",
    S.groupby("year")["mark"].apply(lambda m: m.dt.normalize().nunique()).to_dict(),
)
tbl = read_table()
syms = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "AAPL", "NVDA"]


def eff(x):  # floored, as the cost model applies it
    return np.maximum(x, 0.25)


rows = []
for s in syms:
    d = S[(S.symbol == s) & S.hs.notna() & (S.year <= 2025)]
    g = d.groupby(["year", "bin"])["hs"]
    med, mean = g.median(), g.mean()
    rows.append(
        {
            "sym": s,
            "median_of_cells(eff)": eff(med).mean(),
            "mean_of_cells(eff)": eff(mean).mean(),
            "mean/median (eff)": (eff(mean) / eff(med)).median(),
            "close_auction med": eff(med.xs("close_auction", level="bin")).mean(),
            "close_auction mean": eff(mean.xs("close_auction", level="bin")).mean(),
            "15:30 med": eff(med.xs("15:30", level="bin")).mean(),
            "15:30 mean": eff(mean.xs("15:30", level="bin")).mean(),
        }
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))
# within-year look-ahead: same-year table vs previous-year table (causal alternative), regular bins, floored
t = tbl[tbl.symbol.isin(syms) & ~tbl.bin.isin(["day"])].copy()
t["eff"] = eff(t.half_spread_bp)
p = t.pivot_table(index=["symbol", "bin"], columns="year", values="eff")
ratio = p[list(range(2017, 2026))].to_numpy() / p[list(range(2016, 2025))].to_numpy()
print(
    "same-year / previous-year table value, regular+auction bins, 2017-2025: median %.3f, 10%% %.3f, 90%% %.3f, max %.2f"
    % tuple(np.nanpercentile(ratio, [50, 10, 90, 100]))
)
# stress sessions from event samples
edays = sorted(pd.Timestamp(p.split("\\")[-1].split("/")[-1][:10]) for p in glob.glob(str(EVENT_SAMPLES_DIR / "*.npz")))
E = pd.concat([load_samples(EVENT_SAMPLES_DIR, d) for d in edays], ignore_index=True)
E["hs"] = half_spread_bp(E)
E["year"] = E["mark"].dt.tz_convert(NY_TZ).dt.year
E["bin"] = [l if l in AUCTION_BINS else tod_bin(l) for l in E["label"]]
E["day"] = E["mark"].dt.tz_convert(NY_TZ).dt.strftime("%Y-%m-%d")
from data.quotes import STRESS_DAYS

st = E[E.day.isin(STRESS_DAYS)]
m = (
    st.groupby(["symbol", "year", "bin"])["hs"]
    .median()
    .rename("stress")
    .reset_index()
    .merge(tbl, on=["symbol", "year", "bin"])
)
m["ratio_eff"] = eff(m.stress) / eff(m.half_spread_bp)
out = m[m.symbol.isin(syms)].groupby(["symbol"])["ratio_eff"].median()
print("stress/table (floored) median by symbol:", out.round(2).to_dict())
print(
    "stress/table by bin type for SPY,QQQ,IWM,DIA:",
    m[m.symbol.isin(["SPY", "QQQ", "IWM", "DIA"])]
    .assign(k=lambda x: np.where(x.bin.isin(AUCTION_BINS), x.bin, "regular"))
    .groupby("k")["ratio_eff"]
    .median()
    .round(2)
    .to_dict(),
)
