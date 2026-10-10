"""
`load_bars` — the single entry point the pipeline uses for market data.

Ranges are whole NY trading days: [start, end) with start/end as dates. Native
timeframes (1Min, 5Min) are fetched from Alpaca, RTH-filtered and cached;
15Min/30Min/1Hour/1Day are resampled on the fly from cached RTH 5Min bars (cheap,
and never stale). `1DayPrint` (U22, SPEC §20) is Alpaca's native daily bar, cached
for its open and close: the official opening and closing auction prints
(`load_prints`); its high / low / volume include extended hours and are not used.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.alpaca_source import NY_TZ, AlpacaSource, calendar_from_raw
from data.sessions import (
    drop_incomplete_sessions,
    drop_sessions_missing_last_bar,
    resample_daily,
    resample_session,
    rth_filter,
    to_ny,
)
from data.store import atomic_write, cache_paths, load_bars_cache, merge_bars, save_bars
from data.timeframes import get_timeframe

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
CALENDAR_FIRST_DAY = pd.Timestamp("2010-01-01")
OVERLAP = pd.Timedelta(days=7)  # re-fetched on top-ups to detect retroactive (corporate-action) adjustments
ADJ_RTOL = 1e-6
FETCH_CHUNK = pd.Timedelta(days=366)
CALENDAR_MAX_AGE = pd.Timedelta(days=30)
# Holdout (SPEC §9, moved by §11.1 in U13): research code may not load bars on or after this day. The U1–U11 holdout
# 2025-10-01 → 2026-09-27 is now the reported quasi-holdout; the real holdout is the forward test (PLAN2 U20)
HOLDOUT_START = pd.Timestamp("2026-10-01")


class HoldoutError(ValueError):
    pass


def _day(x) -> pd.Timestamp:
    """Date-like → tz-naive midnight (a NY calendar day)."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    if ts != ts.normalize():
        raise ValueError(f"load_bars ranges are whole days; got {x!r}")
    return ts


def _today_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ).tz_localize(None).normalize()


# ── Calendar ────────────────────────────────────────────────────────────────


def get_calendar(end, cache_dir: Path = DEFAULT_CACHE_DIR, source=None) -> pd.DataFrame:
    """
    Exchange sessions from CALENDAR_FIRST_DAY through at least `end`, cached as
    `calendar.json` ({"fetched_at", "rows"} of raw Alpaca rows, written atomically).
    Re-fetched when `end` is past the cached range or the cache is older than
    CALENDAR_MAX_AGE (unscheduled closures are announced at short notice).
    """
    path = Path(cache_dir) / "calendar.json"
    end = _day(end)
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            cal = calendar_from_raw(blob["rows"])
            age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(blob["fetched_at"])
            if cal.index[-1] >= end and age <= CALENDAR_MAX_AGE:
                return cal
    source = source or AlpacaSource()
    fetch_end = max(end, _today_ny()) + timedelta(days=366)
    cal = source.calendar(CALENDAR_FIRST_DAY.date(), fetch_end.date())
    rows = [
        {"date": d.strftime("%Y-%m-%d"), "open": o.strftime("%H:%M"), "close": c.strftime("%H:%M")}
        for d, o, c in zip(cal.index, cal["session_open"], cal["session_close"])
    ]
    blob = {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": rows}
    atomic_write(path, lambda fh: fh.write(json.dumps(blob).encode()))
    if cal.index[-1] < end:
        raise ValueError(f"Alpaca calendar ends {cal.index[-1].date()}, before requested {end.date()}")
    return cal


# ── Bars ────────────────────────────────────────────────────────────────────


def _fetch(source, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, adjustment: str, calendar):
    """Native minute bars for NY days [start, end), fetched in yearly chunks and RTH-filtered."""
    tf = get_timeframe(timeframe)
    parts = []
    a = start
    while a < end:
        b = min(a + FETCH_CHUNK, end)
        parts.append(source.bars(symbol, timeframe, a.tz_localize(NY_TZ), b.tz_localize(NY_TZ), adjustment))
        a = b
    df = to_ny(pd.concat(parts)) if parts else None
    if df is None or df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if tf.is_daily:  # one bar per exchange session, stamped at the session's NY midnight
        return session_filter(df, calendar)
    return rth_filter(df, calendar, tf.minutes)


def session_filter(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Daily bars: keep the rows whose NY date is an exchange session, stamped at that session's midnight (NY)."""
    df = to_ny(df)
    day = df.index.normalize()
    keep = day.tz_localize(None).isin(calendar.index)
    out = df[keep].copy()
    out.index = day[keep]
    return out


def _adjustment_changed(cached: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    common = cached.index.intersection(fresh.index)
    if common.empty:
        return False
    a, b = cached.loc[common, "close"], fresh.loc[common, "close"]
    return bool(((a - b).abs() / b.abs()).max() > ADJ_RTOL)


def load_bars(
    symbol: str,
    timeframe: str,
    start,
    end,
    *,
    feed: str = "sip",
    adjustment: str = "all",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    source=None,
    refresh: bool = False,
    min_session_coverage: float = 0.5,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """
    RTH bars for `symbol` over NY trading days [start, end).

    Index = bar OPEN time, tz-aware America/New_York. Columns: open, high, low,
    close, volume, vwap, trade_count (float64). `end` is clamped to today (only
    completed sessions are cached). Missing bars are not filled; sessions with too
    few bars are dropped (see min_session_coverage).

    Top-ups re-fetch a 7-day overlap with the cache; if the overlapping closes
    differ (a split/dividend re-adjusted history) the whole range is re-fetched
    and the cache replaced, with a log line.

    Args:
        symbol (str): Ticker.
        timeframe (str): One of data.timeframes.TIMEFRAMES.
        start, end: Dates (NY) — [start, end).
        feed (str, optional): "sip" or "iex". Never substituted on failure. Defaults to "sip".
        adjustment (str, optional): Alpaca adjustment. Defaults to "all".
        cache_dir (Path, optional): Cache root. Defaults to data/cache.
        source (optional): Object with .bars/.calendar (AlpacaSource by default; created only if a fetch is needed).
        refresh (bool, optional): Re-fetch the union of the request and the cached coverage. Defaults to False.
        min_session_coverage (float, optional): Drop (and log) sessions with fewer than this fraction of
            their expected base bars; applied before resampling. For 1Day, sessions missing their last
            bar are dropped too (the daily close would be wrong). 0 keeps everything. Defaults to 0.5.
        allow_holdout (bool, optional): Permit `end` past HOLDOUT_START. Defaults to False (raises).

    Returns:
        pd.DataFrame: Bars.
    """
    symbol = symbol.upper()
    tf = get_timeframe(timeframe)
    start, end = _day(start), _day(end)
    if end > HOLDOUT_START and not allow_holdout:
        raise HoldoutError(
            f"{symbol}: requested data through {end.date()}, past HOLDOUT_START {HOLDOUT_START.date()} "
            "(pass allow_holdout=True only for caching or the final evaluation)"
        )
    end = min(end, _today_ny())
    if start >= end:
        raise ValueError(f"Empty range [{start.date()}, {end.date()})")
    label = f"{symbol} {timeframe}: "

    def src():
        nonlocal source
        if source is None:
            source = AlpacaSource(feed=feed)
        return source

    calendar = get_calendar(end, cache_dir, source=source)

    if not tf.native:
        base = load_bars(
            symbol,
            tf.base,
            start,
            end,
            feed=feed,
            adjustment=adjustment,
            cache_dir=cache_dir,
            source=source,
            refresh=refresh,
            min_session_coverage=min_session_coverage,
            allow_holdout=allow_holdout,
        )
        if not tf.is_daily:
            return resample_session(base, calendar, timeframe)
        if min_session_coverage > 0:  # a daily close needs the session's last bar
            base = drop_sessions_missing_last_bar(base, calendar, tf.base, label=label)
        return resample_daily(base)

    npz, js = cache_paths(cache_dir, feed, adjustment, timeframe, symbol)
    ident = {"symbol": symbol, "timeframe": timeframe, "feed": feed, "adjustment": adjustment}
    cached = meta = None
    if npz.exists():
        if refresh:  # re-fetch, but never shrink what the cache covered
            old = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
            if "coverage_start" in old:
                start_all = min(start, _day(old["coverage_start"]))
                end_all = max(end, _day(old["coverage_end"]))
            else:
                start_all, end_all = start, end
        else:
            cached, meta = load_bars_cache(npz, js, expect=ident)

    if cached is None:
        lo, hi = (start_all, end_all) if refresh and npz.exists() else (start, end)
        cs = ce = None
        df = _fetch(src(), symbol, timeframe, lo, hi, adjustment, calendar)
    else:
        cs, ce = _day(meta["coverage_start"]), _day(meta["coverage_end"])
        lo, hi = min(start, cs), max(end, ce)
        df = cached
        if lo < cs or hi > ce:
            fresh = []
            if lo < cs:
                fresh.append(_fetch(src(), symbol, timeframe, lo, min(cs + OVERLAP, ce), adjustment, calendar))
            if hi > ce:
                fresh.append(_fetch(src(), symbol, timeframe, max(ce - OVERLAP, cs), hi, adjustment, calendar))
            fresh = [f for f in fresh if f is not None and not f.empty]
            if any(_adjustment_changed(cached, f) for f in fresh):
                print(f"[DATA]  {label}cached prices no longer match Alpaca's adjustment — re-fetching full range")
                df = _fetch(src(), symbol, timeframe, lo, hi, adjustment, calendar)
            else:
                for f in fresh:
                    df = merge_bars(df, f)

    if cached is None or lo < cs or hi > ce:
        if df is None or df.empty:
            raise ValueError(f"Alpaca returned no {timeframe} bars for {symbol} in [{lo.date()}, {hi.date()})")
        hi = _verified_end(df, calendar, lo, hi, ce, label)
        save_bars(
            npz,
            js,
            df,
            {
                **ident,
                "tz": NY_TZ,
                "session": "rth",
                "source": "alpaca",
                "coverage_start": lo.date().isoformat(),
                "coverage_end": hi.date().isoformat(),
                "fetched_at": datetime.now().astimezone().isoformat(),
            },
        )

    lo_ts, hi_ts = start.tz_localize(NY_TZ), end.tz_localize(NY_TZ)
    out = df[(df.index >= lo_ts) & (df.index < hi_ts)]
    return drop_incomplete_sessions(out, calendar, timeframe, min_session_coverage, label=label)


def _verified_end(
    df: pd.DataFrame, calendar: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, prev_end, label: str
) -> pd.Timestamp:
    """
    Coverage end to record: `hi`, unless trailing calendar sessions came back with no
    bars (outage, delayed data) — then stop after the last session that has bars so a
    later load re-fetches them. Empty sessions elsewhere in the range are logged.
    """
    days = df.index.tz_localize(None).normalize()
    sessions = calendar.index[(calendar.index >= lo) & (calendar.index < hi)]
    empty = sessions[~sessions.isin(days)]
    if empty.empty:
        return hi
    last_day = days[-1]
    tail = empty[empty > last_day]
    head = empty[empty < days[0]]
    middle = empty[(empty > days[0]) & (empty < last_day)]
    if len(head):
        print(
            f"[DATA]  {label}no bars in {len(head)} session(s) {head[0].date()} → {head[-1].date()} (before listing?)"
        )
    if len(middle):
        print(
            f"[DATA]  {label}no bars in {len(middle)} session(s) inside the range: {', '.join(str(d.date()) for d in middle[:10])}"
        )
    if len(tail):
        new_hi = last_day + timedelta(days=1)
        if prev_end is not None:
            new_hi = max(new_hi, prev_end)
        print(
            f"[DATA]  {label}no bars for {len(tail)} trailing session(s) {tail[0].date()} → {tail[-1].date()}; "
            f"not marked as cached (coverage ends {new_hi.date()})"
        )
        return new_hi
    return hi


# ── Auction prints (U22, SPEC §20) ───────────────────────────────────────────

PRINT_COLUMNS = ["open", "close", "volume"]


def load_prints(symbol: str, start, end, **kw) -> pd.DataFrame:
    """
    Official auction prints per session: Alpaca's native daily bar (`1DayPrint`) reduced to `open` (the opening
    auction print), `close` (the closing auction print) and `volume`, indexed by the session's NY midnight. Keyword
    arguments as load_bars (feed, adjustment, cache_dir, source, refresh, allow_holdout).
    """
    from data.timeframes import PRINT_TIMEFRAME

    kw.setdefault("min_session_coverage", 0.0)
    df = load_bars(symbol, PRINT_TIMEFRAME, start, end, **kw)
    return df[PRINT_COLUMNS]


FORWARD_ACCESS = DEFAULT_CACHE_DIR / "forward_access.jsonl"


def log_forward_access(event: dict, path: Path = FORWARD_ACCESS) -> None:
    """Append an audit record of a read or fetch past HOLDOUT_START (SPEC §18) to data/cache/forward_access.jsonl."""
    row = {"at": pd.Timestamp.now(tz="UTC").isoformat(), **event}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def truncate_forward(cache_dir: Path = DEFAULT_CACHE_DIR, log_path: Path = FORWARD_ACCESS) -> list[dict]:
    """
    Drop every cached bar stamped on or after HOLDOUT_START from every bar cache under `cache_dir` (U22: the caches
    that data/fetch.py topped up to "today" before its default end was HOLDOUT_START), set their coverage_end to
    HOLDOUT_START and log one `truncate` event per cache touched. Returns the events.
    """
    events = []
    cut = HOLDOUT_START.tz_localize(NY_TZ)
    for js in sorted(Path(cache_dir).glob("*/*/*/*.json")):
        npz = js.with_suffix(".npz")
        if not npz.exists():
            continue
        meta = json.loads(js.read_text(encoding="utf-8"))
        if "coverage_end" not in meta or _day(meta["coverage_end"]) <= HOLDOUT_START:
            continue
        df, meta = load_bars_cache(npz, js)
        keep = df.index < cut
        dropped = int((~keep).sum())
        new_meta = {k: v for k, v in meta.items() if k not in ("schema_version", "columns", "n_bars", "first_ts",
                                                                  "last_ts", "sha256")}  # fmt: skip
        new_meta["coverage_end"] = HOLDOUT_START.date().isoformat()
        save_bars(npz, js, df[keep], new_meta)
        ev = {"event": "truncate", "symbol": meta["symbol"], "timeframe": meta["timeframe"], "feed": meta["feed"],
              "adjustment": meta["adjustment"], "old_coverage_end": meta["coverage_end"], "dropped_bars": dropped,
              "new_coverage_end": new_meta["coverage_end"]}  # fmt: skip
        log_forward_access(ev, log_path)
        print(f"[DATA]  {meta['symbol']} {meta['timeframe']}: dropped {dropped} bar(s) on / after "
              f"{HOLDOUT_START.date()} (coverage ended {meta['coverage_end']})")  # fmt: skip
        events.append(ev)
    return events
