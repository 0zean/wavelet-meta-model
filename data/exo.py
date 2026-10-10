"""
Point-in-time exogenous series (SPEC §12, U13).

    load_series(source, name, start, end) -> DataFrame indexed by observation date (NY, tz-naive midnight), columns
    `value` (float), optional extra columns, and `available_at` (tz-aware UTC instant from which the row may be used)

Sources (the `available_at` rules are in AVAILABLE_AT and the functions below):
- `cboe`: VIX, VIX9D, VIX3M, VIX6M daily OHLC (`value` = close) from CBOE's index history files, available at the
  session date 16:20 ET. VX1, VX2 (continuous front / second monthly VIX future settlements, rolled on the expiry
  date, whose row carries the final settlement) and the ratios VX1_VIX, VX2_VX1, from CFE's per-expiry files,
  available at the settlement date 16:20 ET.
- `fred`: VIXCLS, DGS2, DGS10, T10Y2Y, BAMLH0A0HYM2, BAA10Y, DTWEXBGS, DTB3 from fredgraph.csv. VIXCLS: next
  federal business day 09:00 ET; the H.15 series (DTB3, the 3-month T-bill secondary-market rate, is one: the cash
  yield of SPEC §22), HY OAS and BAA10Y: next federal business day 16:30 ET (H.15's daily update is 16:15 ET);
  DTWEXBGS (H.10, weekly): the Monday after the observation's week 16:30 ET (next business day if a holiday).
  BAMLH0A0HYM2 only has the last three years on FRED (ICE licence), so it cannot cover the development window;
  BAA10Y (Moody's seasoned Baa yield − DGS10, daily, full history; U15's credit spread) takes the H.15 rule.
- `calendar`, `earnings`: the checked-in tables of data/events.py (no network, not cached here).

Cache: `{cache_dir}/{source}/{name}.npz` + `.json` (schema_version, source, name, columns, n_rows, coverage_start,
coverage_end, fetched_at, url, sha256), written atomically, hash-verified on load. A fetch always downloads the
whole published history (these files are small) and replaces the cache; values that changed on dates the cache
already held are logged as revisions (no vintages are kept). `coverage_end` is the NY date of the fetch: a request
ending after it re-fetches. `end > HOLDOUT_START` raises unless `allow_holdout`.

The point-in-time rule is enforced where a series meets bars (the U15 feature builder): a row is used on a bar only
when the bar's information time is at or after `available_at`.
"""

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.alpaca_source import NY_TZ
from data.bars import DEFAULT_CACHE_DIR, HOLDOUT_START, HoldoutError
from data.store import CacheError, _sha256, atomic_write

EXO_CACHE_DIR = DEFAULT_CACHE_DIR / "exo"
SCHEMA_VERSION = 1
HTTP_TIMEOUT = 60
USER_AGENT = "Mozilla/5.0 (wavelet-meta-model research)"

CBOE_INDEX_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CFE_LIST_URL = "https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"
CFE_FILE_URL = "https://cdn.cboe.com/{path}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}"

CBOE_INDEXES = ("VIX", "VIX9D", "VIX3M", "VIX6M")
VX_SERIES = ("VX1", "VX2", "VX1_VIX", "VX2_VX1")
FRED_SERIES = ("VIXCLS", "DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "BAA10Y", "DTWEXBGS", "DTB3")
SETTLE_TIME = "16:20"  # CBOE index closes and CFE settlements (16:15 ET) plus 5 minutes


def _day(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    if ts != ts.normalize():
        raise ValueError(f"exogenous ranges are whole days; got {x!r}")
    return ts


def _today_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ).tz_localize(None).normalize()


def http_get(url: str, headers: dict | None = None) -> str:
    """GET `url` as text; raises on any non-200 response (tests pass their own `http` callable instead)."""
    import requests

    r = requests.get(url, headers={"User-Agent": USER_AGENT, **(headers or {})}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


# ── available_at rules ───────────────────────────────────────────────────────


def at_time(dates: pd.DatetimeIndex, hhmm: str) -> pd.DatetimeIndex:
    """NY wall-clock `hhmm` on each (tz-naive) date, as UTC instants."""
    h, m = (int(x) for x in hhmm.split(":"))
    local = pd.DatetimeIndex(dates) + pd.Timedelta(hours=h, minutes=m)
    return local.tz_localize(NY_TZ).tz_convert("UTC")


def _fed_bday():
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    return CustomBusinessDay(calendar=USFederalHolidayCalendar())


def next_business_day(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The next US federal business day after each date (FRED / Federal Reserve release days)."""
    return pd.DatetimeIndex(dates) + _fed_bday()


def following_monday(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Monday after each date's Mon–Sun week, rolled forward to a federal business day (H.10 weekly release)."""
    d = pd.DatetimeIndex(dates)
    monday = d + pd.to_timedelta(7 - d.dayofweek, unit="D")
    bday = _fed_bday()
    return pd.DatetimeIndex([bday.rollforward(x) for x in monday])


def available_at(source: str, name: str, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC instant from which each observation of (source, name) may be used (SPEC §12 table)."""
    if source == "cboe":
        return at_time(dates, SETTLE_TIME)
    if source == "fred":
        if name == "VIXCLS":
            return at_time(next_business_day(dates), "09:00")
        if name == "DTWEXBGS":
            return at_time(following_monday(dates), "16:30")
        return at_time(next_business_day(dates), "16:30")
    raise ValueError(f"no available_at rule for {source}/{name}")


# ── Parsers ──────────────────────────────────────────────────────────────────


def parse_cboe_index(text: str) -> pd.DataFrame:
    """CBOE index history CSV (DATE MM/DD/YYYY, OPEN, HIGH, LOW, CLOSE) → value = close, plus open / high / low."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().upper() for c in raw.columns]
    if not {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"} <= set(raw.columns):
        raise ValueError(f"unexpected CBOE columns {list(raw.columns)}")
    idx = pd.DatetimeIndex(pd.to_datetime(raw["DATE"], format="%m/%d/%Y"), name="date")
    out = pd.DataFrame(
        {k: pd.to_numeric(raw[k.upper()], errors="raise").to_numpy(float) for k in ("open", "high", "low")}, index=idx
    )
    out.insert(0, "value", pd.to_numeric(raw["CLOSE"], errors="raise").to_numpy(float))
    return _clean(out, "cboe index")


def parse_fred(text: str, name: str) -> pd.DataFrame:
    """fredgraph.csv (observation_date, NAME) → value; empty / '.' observations (holidays) are dropped."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    if list(raw.columns) != ["observation_date", name]:
        raise ValueError(f"unexpected FRED columns {list(raw.columns)} for {name}")
    v = pd.to_numeric(raw[name].str.strip().replace({".": None, "": None}), errors="raise")
    out = pd.DataFrame({"value": v.to_numpy(float)}, index=pd.DatetimeIndex(pd.to_datetime(raw["observation_date"]),
                                                                            name="date"))  # fmt: skip
    return _clean(out.dropna(), f"FRED {name}")


def parse_cfe_contract(text: str) -> pd.Series:
    """One CFE VX contract file → settlement by trade date (rows with a non-positive settle are dropped)."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip() for c in raw.columns]
    if not {"Trade Date", "Settle"} <= set(raw.columns):
        raise ValueError(f"unexpected CFE columns {list(raw.columns)}")
    s = pd.Series(pd.to_numeric(raw["Settle"], errors="raise").to_numpy(float),
                  index=pd.DatetimeIndex(pd.to_datetime(raw["Trade Date"]), name="date"))  # fmt: skip
    s = s[s > 0]
    if s.index.has_duplicates:
        raise ValueError("CFE contract file has duplicate trade dates")
    return s.sort_index()


def _clean(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if df.index.has_duplicates:
        raise ValueError(f"{what}: duplicate observation dates")
    return df.sort_index().astype("float64")


def vx_continuous(contracts: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """
    Continuous VX1 / VX2 from monthly contracts {expiry: settle by trade date}.

    On trade date d, VX1 is the contract with the earliest expiry ≥ d (on its expiry date the contract's row is its
    final settlement; the roll happens the session after) and VX2 the next one. Only dates on or after the earliest
    loaded expiry are kept: before it the true front month is a contract that was not loaded, and a later contract
    (months out) would be mislabelled VX1. A date where either contract has no settlement row is NaN for that leg (and
    logged); dates where VX1 is missing are dropped.
    """
    exps = sorted(contracts)
    dates = sorted(d for d in set().union(*(s.index for s in contracts.values())) if d >= exps[0])
    vx1, vx2, missing = [], [], []
    for d in dates:
        k = next((i for i, e in enumerate(exps) if e >= d), None)
        if k is None:
            vx1.append(np.nan)
            vx2.append(np.nan)
            continue
        a = contracts[exps[k]].get(d, np.nan)
        b = contracts[exps[k + 1]].get(d, np.nan) if k + 1 < len(exps) else np.nan
        if np.isnan(a) or np.isnan(b):
            missing.append(d)
        vx1.append(a)
        vx2.append(b)
    out = pd.DataFrame({"VX1": vx1, "VX2": vx2}, index=pd.DatetimeIndex(dates, name="date"))
    if missing:
        print(f"[EXO]   VX: {len(missing)} date(s) without a settlement for VX1 or VX2 (e.g. {missing[0].date()})")
    return out[out["VX1"].notna()]


# ── Fetchers (whole published history) ───────────────────────────────────────


def _fetch_cboe_index(name: str, http) -> tuple[pd.DataFrame, str]:
    url = CBOE_INDEX_URL.format(name=name)
    return parse_cboe_index(http(url)), url


def _fetch_vx(cache_dir: Path, http) -> pd.DataFrame:
    """VX1, VX2 from every monthly contract since 2015; expired contract files are cached as raw text."""
    listing = json.loads(http(CFE_LIST_URL))
    raw_dir = Path(cache_dir) / "cboe" / "vx_contracts"
    today = _today_ny()
    contracts = {}
    for year, rows in listing.items():
        if int(year) < 2015:
            continue
        for row in rows:
            if row.get("duration_type") != "M":  # weeklies are not part of the monthly term structure
                continue
            exp = pd.Timestamp(row["expire_date"])
            path = raw_dir / Path(row["path"]).name
            if path.exists() and exp < today:
                text = path.read_text(encoding="utf-8")
            else:
                text = http(CFE_FILE_URL.format(path=row["path"]))
                if exp < today:
                    atomic_write(path, lambda fh, t=text: fh.write(t.encode()))
            s = parse_cfe_contract(text)
            if len(s):
                contracts[exp] = s
    return vx_continuous(contracts)


def _fetch(source: str, name: str, cache_dir: Path, http) -> tuple[pd.DataFrame, str]:
    if source == "cboe" and name in CBOE_INDEXES:
        return _fetch_cboe_index(name, http)
    if source == "cboe" and name in VX_SERIES:
        vx = _fetch_vx(cache_dir, http)
        if name in ("VX1", "VX2"):
            return pd.DataFrame({"value": vx[name]}).dropna(), CFE_LIST_URL
        if name == "VX2_VX1":
            return pd.DataFrame({"value": vx["VX2"] / vx["VX1"]}).dropna(), CFE_LIST_URL
        vix, _ = _fetch_cboe_index("VIX", http)
        ratio = vx["VX1"] / vix["value"].reindex(vx.index)
        return pd.DataFrame({"value": ratio}).dropna(), CFE_LIST_URL
    if source == "fred" and name in FRED_SERIES:
        url = FRED_URL.format(name=name)
        return parse_fred(http(url), name), url
    raise ValueError(f"unknown exogenous series {source}/{name}")


# ── Cache ────────────────────────────────────────────────────────────────────


def cache_paths(cache_dir: Path, source: str, name: str) -> tuple[Path, Path]:
    base = Path(cache_dir) / source / name
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _arrays(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str]]:
    cols = [c for c in df.columns if c != "available_at"]
    arrays = {"ts": df.index.as_unit("ns").asi8.astype(np.int64)}
    arrays |= {c: df[c].to_numpy(dtype=np.float64) for c in cols}
    arrays["available_at"] = pd.DatetimeIndex(df["available_at"]).tz_convert("UTC").as_unit("ns").asi8
    return arrays, [*cols, "available_at"]


def save_series(npz: Path, js: Path, df: pd.DataFrame, meta: dict) -> dict:
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise CacheError("exogenous series index must be sorted and unique")
    arrays, columns = _arrays(df)
    full = {
        **meta,
        "schema_version": SCHEMA_VERSION,
        "columns": columns,
        "n_rows": len(df),
        "sha256": _sha256(arrays, columns),
    }
    atomic_write(npz, lambda fh: np.savez(fh, **arrays))
    atomic_write(js, lambda fh: fh.write(json.dumps(full, indent=2, sort_keys=True).encode()))
    return full


def load_series_cache(npz: Path, js: Path, expect: dict) -> tuple[pd.DataFrame, dict]:
    if not js.exists():
        raise CacheError(f"Missing metadata sidecar {js}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(f"{js}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    for k, v in expect.items():
        if meta.get(k) != v:
            raise CacheError(f"{js}: {k}={meta.get(k)!r}, expected {v!r}")
    columns = meta["columns"]
    with np.load(npz, allow_pickle=False) as z:
        arrays = {c: z[c] for c in ["ts", *columns]}
    if _sha256(arrays, columns) != meta["sha256"]:
        raise CacheError(f"{npz}: array hash does not match {js}")
    idx = pd.DatetimeIndex(arrays["ts"].astype("datetime64[ns]"), name="date")
    df = pd.DataFrame({c: arrays[c] for c in columns if c != "available_at"}, index=idx)
    df["available_at"] = pd.DatetimeIndex(arrays["available_at"].astype("datetime64[ns]")).tz_localize("UTC")
    return df, meta


def _revisions(old: pd.DataFrame, new: pd.DataFrame) -> int:
    common = old.index.intersection(new.index)
    a, b = old.loc[common, "value"].to_numpy(), new.loc[common, "value"].to_numpy()
    lost = len(old.index.difference(new.index))
    return int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum()) + lost


def load_series(
    source: str,
    name: str,
    start,
    end,
    *,
    allow_holdout: bool = False,
    cache_dir: Path = EXO_CACHE_DIR,
    refresh: bool = False,
    http=None,
) -> pd.DataFrame:
    """
    Observations of (source, name) dated in NY days [start, end), with `available_at` (see the module docstring).

    Args:
        source (str): "cboe", "fred", "calendar" or "earnings".
        name (str): Series name (calendar: an event kind; earnings: a stock symbol).
        start, end: Dates (NY) — [start, end).
        allow_holdout (bool, optional): Permit `end` past HOLDOUT_START. Defaults to False (raises).
        cache_dir (Path, optional): Cache root. Defaults to data/cache/exo.
        refresh (bool, optional): Re-download even if the cache covers the request. Defaults to False.
        http (optional): url → text callable (requests by default; tests pass a fake).

    Returns:
        pd.DataFrame: `value`, extra columns, `available_at` (UTC); indexed by observation date.
    """
    start, end = _day(start), _day(end)
    if end > HOLDOUT_START and not allow_holdout:
        raise HoldoutError(
            f"{source}/{name}: requested data through {end.date()}, past HOLDOUT_START {HOLDOUT_START.date()}"
        )
    if start >= end:
        raise ValueError(f"Empty range [{start.date()}, {end.date()})")
    if source in ("calendar", "earnings"):
        from data.events import event_series

        df = event_series(source, name)
    else:
        df = _cached(source, name, min(end, _today_ny()), Path(cache_dir), refresh, http or http_get)
    return df[(df.index >= start) & (df.index < end)]


def _cached(source: str, name: str, end: pd.Timestamp, cache_dir: Path, refresh: bool, http) -> pd.DataFrame:
    npz, js = cache_paths(cache_dir, source, name)
    ident = {"source": source, "name": name}
    old = None
    if npz.exists():
        old, meta = load_series_cache(npz, js, ident)
        if not refresh and pd.Timestamp(meta["coverage_end"]) >= end:
            return old
    df, url = _fetch(source, name, cache_dir, http)
    if df.empty:
        raise ValueError(f"{source}/{name}: the source returned no observations ({url})")
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index, name="date").as_unit("ns")  # the unit a cache load returns
    df["available_at"] = available_at(source, name, df.index).as_unit("ns")
    if old is not None:
        n = _revisions(old, df)
        if n:
            print(f"[EXO]   {source}/{name}: {n} cached observation(s) revised or removed by the source — replaced")
    today = _today_ny()
    save_series(
        npz,
        js,
        df,
        {
            **ident,
            "coverage_start": str(df.index[0].date()),
            "coverage_end": str(today.date()),
            "fetched_at": datetime.now().astimezone().isoformat(),
            "url": url,
        },
    )
    print(f"[EXO]   {source}/{name}: {len(df):,} observations {df.index[0].date()} → {df.index[-1].date()}")
    return df
