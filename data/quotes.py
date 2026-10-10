"""
Quotes sampler and the half-spread table of the `quotes` cost model (SPEC §12 `quotes`, §19; U13).

Sample. One trading week per calendar quarter: sessions 6–10 of the quarter's middle month (Feb, May, Aug, Nov). By
construction it holds no turn-of-month, month-end or third-Friday (OPEX) session and almost no FOMC / NFP session
(the U13 review counted 0, 0, 1, 1 of 195 days over 2016Q1–2025Q3, against 3–5 % base rates; CPI days are 16 % of
it, 3.5× their rate), and its days are slightly calmer than average (VIX 17.98 vs 18.52; March 2020 and 2024-08-05
are outside it). The table is therefore the cost of an ordinary session. The `events` plan (`event_days`) samples one
FOMC, NFP, OPEX and month-end session per year plus a few stress sessions, cached separately and never mixed into the
table, to measure how much wider those sessions are (`scripts/u13_smoke.py quotes`). On each sampled session, the
prevailing SIP NBBO of every symbol at:
- the regular marks open + 5 min, open + 10 min, …, close − 5 min (09:35 … 15:55 on a full session);
- the auction proxies open + 1 min (`open_auction`) and close − 1 min (`close_auction`): the quoted market around
  the auctions, a conservative stand-in for the cost of a market-on-open / market-on-close fill.
The prevailing quote at mark m is the last quote in [m − 5 s, m); symbols without one are re-requested over
[m − 60 s, m) and then [m − 15 min, m), each window starting no earlier than the session open (no pre-open quote);
still none → NaN (counted). One request serves all symbols at a mark;
requests are paced under Alpaca's 200 / min. Each session is cached as `{cache}/samples/{date}.npz` + `.json`
(atomic, hash-verified), so a run resumes where it stopped.

Table. Half-spread in bp = (ask − bid) / (ask + bid) · 1e4, from quotes with bid > 0 and ask ≥ bid (a crossed or
one-sided quote is dropped and counted) stamped at or after 09:30 of their session (a pre-open quote, which the
first U13 fetch could reach through the 15-minute window, is dropped and counted). Per (symbol, year, bin): the median over the samples, with `n`. Bins: the
15-minute buckets of the regular marks labelled by their start (09:30 = marks 09:35 and 09:40, …, 15:45 = 15:45 …
15:55), `open_auction`, `close_auction`, and `day` (all regular marks of the year — the cost of an intrabar fill at
an unknown time, used for daily bars). Stored as the checked-in `data/costs/quotes_half_spread.csv` + `.json`
(provenance, sample plan, counts).

As-of table (U22, SPEC §20; `COST_TABLE="asof"`). The per-year medians price a fill early in a year with that
year's later sample weeks (a within-year look-ahead of ± 25 % on tenths of a bp). `quotes_half_spread_asof.csv`
holds, per (symbol, asof, bin), the median over the samples of the ASOF_WEEKS most recent sample weeks completed
before `asof` (the first day of a quarter; a quarter's own week, in its middle month, completes before the next
quarter starts), so a fill in quarter Q is priced with what was observable at Q's start. Tables exist from the
quarter after the first sample week (2016Q2, one week, n ≈ 10 per bin) and pool four weeks (n ≈ 40, the yearly
table's sample size) from 2017Q1. Fills before the first table's quarter use the first table (the only remaining
look-ahead: 2016Q1, counted by risk.costs.fill_costs).

    uv run python -m data.quotes fetch [--plan base|events] [--symbols ...] [--start-quarter 2016Q1] [--end-quarter 2026Q3]
    uv run python -m data.quotes table
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.alpaca_source import NY_TZ
from data.exo import EXO_CACHE_DIR
from data.store import CacheError, _sha256, atomic_write

SAMPLES_DIR = EXO_CACHE_DIR / "quotes" / "samples"
EVENT_SAMPLES_DIR = EXO_CACHE_DIR / "quotes" / "event_samples"
EVENT_KINDS = ("FOMC", "NFP", "OPEX", "MONTH_END")
# Stress sessions: the 2018-02-06 short-vol unwind, the 2020-03 crash week, the 2024-08-05 carry unwind
STRESS_DAYS = ("2018-02-06", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2024-08-05")
COSTS_DIR = Path(__file__).resolve().parent / "costs"
TABLE_CSV = COSTS_DIR / "quotes_half_spread.csv"
TABLE_COLUMNS = ["symbol", "year", "bin", "half_spread_bp", "n"]
ASOF_TABLE_CSV = COSTS_DIR / "quotes_half_spread_asof.csv"
ASOF_COLUMNS = ["symbol", "asof", "bin", "half_spread_bp", "n"]
ASOF_WEEKS = 4  # sample weeks (one per quarter) pooled by a table as of a quarter start
SCHEMA_VERSION = 1
MARK_STEP = pd.Timedelta(minutes=5)
BIN_MINUTES = 15
WINDOWS_S = (5, 60, 900)
MIN_REQUEST_INTERVAL = 0.32  # seconds between requests: ≤ 188 / min under the 200 / min limit
MIDDLE_MONTHS = (2, 5, 8, 11)
WEEK_SESSIONS = slice(5, 10)  # sessions 6–10 of the month (0-based)
SAMPLE_COLUMNS = ["mark", "bid", "ask", "bid_size", "ask_size", "quote_ts", "window_s"]
AUCTION_BINS = ("open_auction", "close_auction")


# ── Sample plan ──────────────────────────────────────────────────────────────


def sample_days(sessions: pd.DatetimeIndex, start_quarter: str, end_quarter: str) -> list[pd.Timestamp]:
    """Sessions 6–10 of each quarter's middle month, for quarters [start_quarter, end_quarter] (e.g. "2016Q1")."""
    s = pd.DatetimeIndex(sessions).sort_values()
    out = []
    for q in pd.period_range(pd.Period(start_quarter, "Q"), pd.Period(end_quarter, "Q"), freq="Q"):
        month = MIDDLE_MONTHS[q.quarter - 1]
        days = s[(s.year == q.year) & (s.month == month)]
        if len(days) < WEEK_SESSIONS.stop:
            raise ValueError(f"{q}: the calendar has only {len(days)} sessions in {q.year}-{month:02d}")
        out.extend(days[WEEK_SESSIONS])
    return out


def event_days(sessions: pd.DatetimeIndex, events: pd.DataFrame, first_year: int, last_day) -> dict[str, str]:
    """
    {date: kind} for the `events` plan: per year and EVENT_KINDS kind, the (year mod n)-th of that year's n scheduled
    sessions of the kind (so the month rotates across years), plus STRESS_DAYS; dates before `last_day` only.
    """
    s = set(pd.DatetimeIndex(sessions))
    last = pd.Timestamp(last_day)
    out = {d: "STRESS" for d in STRESS_DAYS if pd.Timestamp(d) < last}
    ev = events[events["kind"].isin(EVENT_KINDS) & (events["scheduled"].astype(int) == 1)]
    ev = ev[ev["date"].map(lambda d: pd.Timestamp(d) in s and pd.Timestamp(d) < last)]
    for (kind, year), g in ev.groupby([ev["kind"], ev["date"].str[:4].astype(int)]):
        if year < first_year:
            continue
        dates = sorted(g["date"])
        out.setdefault(dates[year % len(dates)], kind)
    return dict(sorted(out.items()))


def sample_marks(session_open: pd.Timestamp, session_close: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """(label, instant) for one session: the regular 5-minute marks ("HH:MM") and the two auction proxies."""
    marks = [("open_auction", session_open + pd.Timedelta(minutes=1))]
    t = session_open + MARK_STEP
    while t < session_close:
        marks.append((t.strftime("%H:%M"), t))
        t += MARK_STEP
    marks.append(("close_auction", session_close - pd.Timedelta(minutes=1)))
    return marks


# ── Fetch ────────────────────────────────────────────────────────────────────


class _Pacer:
    def __init__(self, interval: float):
        self.interval, self.last = interval, 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self.last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self.last = time.monotonic()


def prevailing(raw: list[dict]) -> dict | None:
    """The last quote of a window (rows time-ascending), or None."""
    return raw[-1] if raw else None


def fetch_session(
    source, symbols: list[str], marks: list[tuple[str, pd.Timestamp]], pacer=None, session_open=None
) -> pd.DataFrame:
    """
    The prevailing NBBO of every symbol at every mark (see the module docstring), one row per (symbol, mark).

    Args:
        source: Object with `.quotes(symbols, start, end) -> {symbol: [raw quote dicts]}` (AlpacaSource).
        symbols (list[str]): Tickers.
        marks: (label, tz-aware instant) pairs from sample_marks.
        pacer (optional): Request pacer (none in tests).
        session_open (optional): No window starts before this instant (no pre-open quote is taken).

    Returns:
        pd.DataFrame: symbol, label, mark (UTC), bid, ask, bid_size, ask_size, quote_ts (UTC), window_s.
    """
    rows = []
    for label, m in marks:
        got: dict[str, tuple[dict, int]] = {}
        todo = list(symbols)
        for w in WINDOWS_S:
            if not todo:
                break
            if pacer is not None:
                pacer.wait()
            lo = m - pd.Timedelta(seconds=w)
            if session_open is not None:
                lo = max(lo, session_open)
            raw = source.quotes(todo, lo, m)
            for s in todo:
                q = prevailing(raw.get(s, []))
                if q is not None:
                    got[s] = (q, w)
            todo = [s for s in todo if s not in got]
        for s in symbols:
            q, w = got.get(s, (None, 0))
            rows.append({
                "symbol": s, "label": label, "mark": m.tz_convert("UTC").as_unit("ns"),
                "bid": float(q["bp"]) if q else np.nan, "ask": float(q["ap"]) if q else np.nan,
                "bid_size": float(q["bs"]) if q else np.nan, "ask_size": float(q["as"]) if q else np.nan,
                "quote_ts": pd.Timestamp(q["t"]).tz_convert("UTC").as_unit("ns") if q else pd.NaT,
                "window_s": float(w),
            })  # fmt: skip
    return pd.DataFrame(rows)


def _paths(cache_dir: Path, day: pd.Timestamp) -> tuple[Path, Path]:
    base = Path(cache_dir) / day.strftime("%Y-%m-%d")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_samples(cache_dir: Path, day: pd.Timestamp, df: pd.DataFrame) -> None:
    symbols = sorted(df["symbol"].unique())
    labels = list(dict.fromkeys(df["label"]))
    sym_idx = df["symbol"].map({s: i for i, s in enumerate(symbols)}).to_numpy(np.float64)
    lab_idx = df["label"].map({s: i for i, s in enumerate(labels)}).to_numpy(np.float64)
    arrays = {
        "ts": pd.DatetimeIndex(df["mark"]).as_unit("ns").asi8.astype(np.int64),
        "symbol_idx": sym_idx,
        "label_idx": lab_idx,
        **{c: df[c].to_numpy(np.float64) for c in ("bid", "ask", "bid_size", "ask_size", "window_s")},
        "quote_ts": pd.DatetimeIndex(df["quote_ts"]).as_unit("ns").asi8.astype(np.int64),
    }
    cols = [k for k in arrays if k != "ts"]
    meta = {"schema_version": SCHEMA_VERSION, "date": day.strftime("%Y-%m-%d"), "symbols": symbols, "labels": labels,
            "columns": cols, "n_rows": len(df), "feed": "sip", "sha256": _sha256(arrays, cols),
            "fetched_at": datetime.now().astimezone().isoformat()}  # fmt: skip
    npz, js = _paths(cache_dir, day)
    atomic_write(npz, lambda fh: np.savez(fh, **arrays))
    atomic_write(js, lambda fh: fh.write(json.dumps(meta, indent=2).encode()))


def load_samples(cache_dir: Path, day: pd.Timestamp) -> pd.DataFrame | None:
    npz, js = _paths(cache_dir, day)
    if not npz.exists():
        return None
    if not js.exists():
        raise CacheError(f"Missing metadata sidecar {js}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(f"{js}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in ["ts", *meta["columns"]]}
    if _sha256(arrays, meta["columns"]) != meta["sha256"]:
        raise CacheError(f"{npz}: array hash does not match {js}")
    qts = arrays["quote_ts"]
    return pd.DataFrame({
        "symbol": np.asarray(meta["symbols"])[arrays["symbol_idx"].astype(int)],
        "label": np.asarray(meta["labels"])[arrays["label_idx"].astype(int)],
        "mark": pd.DatetimeIndex(arrays["ts"].astype("datetime64[ns]")).tz_localize("UTC"),
        **{c: arrays[c] for c in ("bid", "ask", "bid_size", "ask_size", "window_s")},
        "quote_ts": pd.DatetimeIndex(np.where(qts == np.iinfo(np.int64).min, np.datetime64("NaT"),
                                              qts.astype("datetime64[ns]"))).tz_localize("UTC"),
    })  # fmt: skip


def fetch(symbols: list[str], days: list[pd.Timestamp], calendar: pd.DataFrame, source, cache_dir: Path) -> None:
    """Sample every day not cached yet (or cached without some of `symbols`); logs progress."""
    pacer = _Pacer(MIN_REQUEST_INTERVAL)
    t0 = time.monotonic()
    for i, day in enumerate(days, 1):
        old = load_samples(cache_dir, day)
        need = sorted(set(symbols) - set(old["symbol"])) if old is not None else list(symbols)
        if not need:
            continue
        cal = calendar.loc[day]
        marks = sample_marks(cal["session_open"], cal["session_close"])
        df = fetch_session(source, need, marks, pacer, session_open=cal["session_open"])
        if old is not None:
            df = pd.concat([old, df], ignore_index=True)
        save_samples(cache_dir, day, df)
        miss = int(df["bid"].isna().sum())
        rate = (time.monotonic() - t0) / i
        print(f"[QUOTES] {day.date()} {i}/{len(days)}  {len(need)} symbols  missing {miss}  "
              f"~{rate * (len(days) - i) / 60:.0f} min left", flush=True)  # fmt: skip


# ── Table ────────────────────────────────────────────────────────────────────


def tod_bin(label: str) -> str:
    """BIN_MINUTES bucket (from 09:30) holding a time "HH:MM": "09:35" → "09:30", "09:45" → "09:45", "15:55" → "15:45"."""
    h, m = (int(x) for x in label.split(":"))
    start = 9 * 60 + 30 + ((h * 60 + m - (9 * 60 + 30)) // BIN_MINUTES) * BIN_MINUTES
    return f"{start // 60:02d}:{start % 60:02d}"


def half_spread_bp(samples: pd.DataFrame) -> pd.Series:
    """(ask − bid) / (ask + bid) · 1e4 for valid quotes (bid > 0, ask ≥ bid); NaN otherwise."""
    bid, ask = samples["bid"], samples["ask"]
    ok = (bid > 0) & (ask >= bid) & ~preopen(samples)
    return ((ask - bid) / (ask + bid) * 1e4).where(ok)


def preopen(samples: pd.DataFrame) -> pd.Series:
    """Quotes stamped before 09:30 NY on their mark's date (reached by a long fallback window before the open)."""
    q = samples["quote_ts"].dt.tz_convert(NY_TZ)
    m = samples["mark"].dt.tz_convert(NY_TZ)
    return (q < m.dt.normalize() + pd.Timedelta(hours=9, minutes=30)).fillna(False)


def _long(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per valid sample: symbol, year, day (NY date), bin, hs; the `day` bin duplicates the regular marks."""
    local = samples["mark"].dt.tz_convert(NY_TZ)
    df = samples.assign(hs=half_spread_bp(samples), year=local.dt.year, day=local.dt.normalize().dt.tz_localize(None))
    regular = ~df["label"].isin(AUCTION_BINS)
    df["bin"] = [lab if lab in AUCTION_BINS else tod_bin(lab) for lab in df["label"]]
    parts = [df[["symbol", "year", "day", "bin", "hs"]],
             df.loc[regular, ["symbol", "year", "day", "hs"]].assign(bin="day")]  # fmt: skip
    return pd.concat(parts, ignore_index=True).dropna(subset=["hs"])


def build_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Median half-spread per (symbol, year, bin) — see the module docstring."""
    long = _long(samples)
    out = long.groupby(["symbol", "year", "bin"])["hs"].agg(half_spread_bp="median", n="size").reset_index()
    return out[TABLE_COLUMNS].sort_values(["symbol", "year", "bin"]).reset_index(drop=True)


def build_asof_table(samples: pd.DataFrame, weeks: int = ASOF_WEEKS) -> pd.DataFrame:
    """Median half-spread per (symbol, asof, bin) over the `weeks` most recent sample weeks completed before `asof`
    (a quarter start, "YYYY-MM-DD") — see the module docstring. Tables run from the quarter after the first sample
    week to the quarter after the last."""
    long = _long(samples)
    long["q"] = long["day"].dt.to_period("Q")
    have = sorted(long["q"].unique())
    rows = []
    for q in pd.period_range(have[0] + 1, have[-1] + 1, freq="Q"):
        used = [p for p in have if p < q][-weeks:]
        g = long[long["q"].isin(used)].groupby(["symbol", "bin"])["hs"].agg(half_spread_bp="median", n="size")
        rows.append(g.reset_index().assign(asof=str(q.start_time.date())))
    out = pd.concat(rows, ignore_index=True)
    return out[ASOF_COLUMNS].sort_values(["symbol", "asof", "bin"]).reset_index(drop=True)


def read_table(path: Path = TABLE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str, "bin": str})
    if list(df.columns) != TABLE_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {TABLE_COLUMNS}")
    if df.duplicated(["symbol", "year", "bin"]).any():
        raise ValueError(f"{path}: duplicate (symbol, year, bin) rows")
    return df


def read_asof_table(path: Path = ASOF_TABLE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str, "bin": str, "asof": str})
    if list(df.columns) != ASOF_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {ASOF_COLUMNS}")
    if df.duplicated(["symbol", "asof", "bin"]).any():
        raise ValueError(f"{path}: duplicate (symbol, asof, bin) rows")
    return df


def table(cache_dir: Path, days: list[pd.Timestamp], out: Path = TABLE_CSV) -> pd.DataFrame:
    frames = [s for d in days if (s := load_samples(cache_dir, d)) is not None]
    if not frames:
        raise ValueError(f"no cached quote samples in {cache_dir}")
    samples = pd.concat(frames, ignore_index=True)
    hs = half_spread_bp(samples)
    tbl = build_table(samples)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out, index=False, lineterminator="\n", float_format="%.6g")
    used = sorted({str(d.date()) for d in samples["mark"].dt.tz_convert(NY_TZ).dt.normalize()})
    meta = {
        "built_at": datetime.now().astimezone().isoformat(),
        "source": "Alpaca historical stock quotes, feed=sip (NBBO)",
        "plan": "sessions 6-10 of Feb/May/Aug/Nov; marks open+5min..close-5min every 5 min, auction proxies "
        "open+1min / close-1min; prevailing quote = last in [m-5s, m), then 60 s, then 15 min",
        "bins": f"{BIN_MINUTES}-minute buckets of the regular marks, open_auction, close_auction, day",
        "days": used,
        "n_samples": len(samples),
        "n_missing_quote": int(samples["bid"].isna().sum()),
        "n_invalid_quote": int((samples["bid"].notna() & hs.isna()).sum()),
        "n_preopen_quote": int(preopen(samples).sum()),
        "window_counts": {str(int(k)): int(v) for k, v in samples["window_s"].value_counts().sort_index().items()},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[QUOTES] table {len(tbl):,} rows from {len(samples):,} samples on {len(used)} sessions → {out}")
    asof = build_asof_table(samples)
    asof_out = out.with_name(ASOF_TABLE_CSV.name) if out == TABLE_CSV else out.with_name(out.stem + "_asof.csv")
    asof.to_csv(asof_out, index=False, lineterminator="\n", float_format="%.6g")
    asofs = sorted(asof["asof"].unique())
    asof_out.with_suffix(".json").write_text(json.dumps({
        "built_at": meta["built_at"], "source": meta["source"], "plan": meta["plan"], "bins": meta["bins"],
        "weeks_pooled": ASOF_WEEKS, "asof_quarters": asofs, "days": used,
        "note": f"a table as of a quarter start pools the <= {ASOF_WEEKS} sample weeks completed before it; fills "
                f"before {asofs[0]} use the {asofs[0]} table (U22, SPEC §20)",
    }, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    print(f"[QUOTES] as-of table {len(asof):,} rows, {len(asofs)} quarters {asofs[0]} → {asofs[-1]} → {asof_out}")
    return tbl


def main(argv: list[str] | None = None) -> None:
    from data.bars import DEFAULT_CACHE_DIR, get_calendar
    from data.fetch import DEFAULT_UNIVERSE

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["fetch", "table"])
    p.add_argument("--plan", choices=["base", "events"], default="base",
                   help="fetch: the week-per-quarter plan, or the event / stress sessions (cached apart, not in the table)")  # fmt: skip
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--start-quarter", default="2016Q1")
    p.add_argument("--end-quarter", default="2026Q3")
    p.add_argument("--cache-dir", type=Path, default=SAMPLES_DIR)
    args = p.parse_args(argv)
    cal = get_calendar(pd.Timestamp.today().normalize(), DEFAULT_CACHE_DIR)
    days = sample_days(cal.index, args.start_quarter, args.end_quarter)
    if args.action == "fetch" and args.plan == "events":
        from data.events import read_events

        last = pd.Period(args.end_quarter, "Q").end_time.normalize() + pd.Timedelta(days=1)
        base = set(days)
        plan = event_days(cal.index, read_events(), int(args.start_quarter[:4]), last)
        days = [pd.Timestamp(d) for d in plan if pd.Timestamp(d) not in base]
        args.cache_dir = EVENT_SAMPLES_DIR if args.cache_dir == SAMPLES_DIR else args.cache_dir
    if args.action == "fetch":
        from data.alpaca_source import AlpacaSource

        fetch([s.upper() for s in args.symbols], days, cal, AlpacaSource(feed="sip"), args.cache_dir)
    else:
        table(args.cache_dir, days)


if __name__ == "__main__":
    main()
