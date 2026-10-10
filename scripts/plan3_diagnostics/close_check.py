# Does Alpaca's 1Day bar close equal the last 5Min RTH bar close (15:55 bar), or the closing-auction print?
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # the repo .env, found by walking up from this file
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

c = StockHistoricalDataClient(os.environ["API_KEY"], os.environ["SECRET_KEY"])
start, end = pd.Timestamp("2024-03-04", tz="America/New_York"), pd.Timestamp("2024-03-09", tz="America/New_York")


def bars(tf):
    r = StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=tf, start=start, end=end, feed=DataFeed.SIP, adjustment=Adjustment.RAW
    )
    df = c.get_stock_bars(r).df.reset_index()
    df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    return df


d = bars(TimeFrame.Day)
m5 = bars(TimeFrame(5, TimeFrameUnit.Minute))
m1 = bars(TimeFrame.Minute)
for day, g in m5.groupby(m5["t"].dt.date):
    rth = g[(g["t"].dt.time >= pd.Timestamp("09:30").time()) & (g["t"].dt.time < pd.Timestamp("16:00").time())]
    last5 = rth.iloc[-1]
    b16 = g[g["t"].dt.time == pd.Timestamp("16:00").time()]
    g1 = m1[m1["t"].dt.date == day]
    b1_1559 = g1[g1["t"].dt.time == pd.Timestamp("15:59").time()]
    b1_1600 = g1[g1["t"].dt.time == pd.Timestamp("16:00").time()]
    dd = d[d["t"].dt.date == day]
    print(
        day,
        "daily close",
        float(dd["close"].iloc[0]),
        "| 15:55 5Min close",
        float(last5["close"]),
        "| 1Min 15:59 close",
        float(b1_1559["close"].iloc[0]) if len(b1_1559) else None,
        "| 16:00 1Min bar o/c/vol",
        (float(b1_1600["open"].iloc[0]), float(b1_1600["close"].iloc[0]), int(b1_1600["volume"].iloc[0]))
        if len(b1_1600)
        else None,
        "| 16:00 5Min bar vol",
        int(b16["volume"].iloc[0]) if len(b16) else None,
        "| daily open",
        float(dd["open"].iloc[0]),
        "| 09:30 5Min open",
        float(rth.iloc[0]["open"]),
    )
