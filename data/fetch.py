"""
Fetch and cache Alpaca bars, optionally writing a data-quality report.

    uv run python -m data.fetch --symbols SPY QQQ --timeframes 5Min 1Day --start 2016-01-01 --report
"""

import argparse
from pathlib import Path

import pandas as pd

from data.bars import DEFAULT_CACHE_DIR, HOLDOUT_START, get_calendar, load_bars
from data.quality import quality_report
from data.timeframes import TIMEFRAMES

DEFAULT_UNIVERSE = [
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "TLT", "GLD",
    # Stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH",
    # U13 (PLAN2 F2 cross-asset ETFs, SVXY for F9): all listed before 2016. SVXY went from −1× to −0.5× VIX
    # short-term futures on 2018-02-28 (after 2018-02-05); USO reverse-split 1:8 on 2020-04-29 (bars are adjusted)
    "IEF", "LQD", "HYG", "DBC", "USO", "UUP", "EFA", "EEM", "VNQ", "SLV", "SVXY",
]  # fmt: skip


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--timeframes", nargs="+", default=["5Min", "1Day"], choices=list(TIMEFRAMES))
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    # The report measures raw coverage (incomplete sessions kept) on the development window only
    p.add_argument("--report", type=Path, nargs="?", const=Path("results/data_quality.csv"), default=None)
    args = p.parse_args(argv)

    kw = {"feed": args.feed, "adjustment": args.adjustment, "cache_dir": args.cache_dir, "min_session_coverage": 0.0}
    # Caching may run past HOLDOUT_START; the quality report only looks at the development window
    report_end = min(pd.Timestamp(args.end), HOLDOUT_START)
    rows = []
    for sym in args.symbols:
        for tf in args.timeframes:
            df = load_bars(sym, tf, args.start, args.end, allow_holdout=True, **kw)
            span = f"{df.index[0]} → {df.index[-1]}" if len(df) else "(no bars)"
            print(f"[DATA]  {sym:6s} {tf:6s} {len(df):>9,} bars  {span}")
            if args.report:
                cal = get_calendar(report_end, args.cache_dir)
                dev = load_bars(sym, tf, args.start, report_end, **kw)
                base = load_bars(sym, "5Min", args.start, report_end, **kw) if tf == "1Day" else None
                rows.append({"symbol": sym, "timeframe": tf, **quality_report(dev, cal, tf, base=base)})

    if args.report:
        report = pd.DataFrame(rows)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report, index=False)
        print(f"[DATA]  Quality report → {args.report}")
        return report
    return None


if __name__ == "__main__":
    main()
