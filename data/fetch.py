"""
Fetch and cache Alpaca bars, optionally writing a data-quality report.

    uv run python -m data.fetch --symbols SPY QQQ --timeframes 5Min 1DayPrint --start 2016-01-01 --report
    uv run python -m data.fetch --truncate-forward        # drop cached bars on / after HOLDOUT_START (logged)

`--end` defaults to HOLDOUT_START (exclusive: the development window and the quasi-holdout end the day before it,
SPEC §11.1). An explicit later end is a forward-window fetch: it is allowed for caching but recorded in
data/cache/forward_access.jsonl (SPEC §18), as `--truncate-forward` is.
"""

import argparse
from pathlib import Path

import pandas as pd

from data.bars import DEFAULT_CACHE_DIR, HOLDOUT_START, get_calendar, load_bars, log_forward_access, truncate_forward
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
# Product changes that split a symbol's history into regimes (SPEC §12): a family using the symbol must start after
# the date or model the break. SVXY is also the survivor of the 2018 short-vol ETPs (XIV was terminated).
REGIME_BREAKS = {"SVXY": ("2018-02-28", "target changed from -1x to -0.5x the S&P 500 VIX Short-Term Futures Index")}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--timeframes", nargs="+", default=["5Min", "1DayPrint"], choices=list(TIMEFRAMES))
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=HOLDOUT_START.strftime("%Y-%m-%d"), help="exclusive; default HOLDOUT_START")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    # The report measures raw coverage (incomplete sessions kept) on the development window only
    p.add_argument("--report", type=Path, nargs="?", const=Path("results/data_quality.csv"), default=None)
    p.add_argument("--truncate-forward", action="store_true",
                   help="drop cached bars on / after HOLDOUT_START from every cache (logged); no fetch")  # fmt: skip
    return p


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    args = parser().parse_args(argv)
    if args.truncate_forward:
        events = truncate_forward(args.cache_dir)
        print(f"[DATA]  truncated {len(events)} cache(s)")
        return None

    kw = {"feed": args.feed, "adjustment": args.adjustment, "cache_dir": args.cache_dir, "min_session_coverage": 0.0}
    forward = pd.Timestamp(args.end) > HOLDOUT_START
    if forward:  # caching may run past HOLDOUT_START, with an audit record (SPEC §18); research code never does
        log_forward_access({"event": "fetch", "symbols": list(args.symbols), "timeframes": list(args.timeframes),
                            "start": args.start, "end": args.end, "cache_dir": str(args.cache_dir)})  # fmt: skip
    # the quality report only looks at the development window
    report_end = min(pd.Timestamp(args.end), HOLDOUT_START)
    rows = []
    for sym in args.symbols:
        for tf in args.timeframes:
            df = load_bars(sym, tf, args.start, args.end, allow_holdout=forward, **kw)
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
