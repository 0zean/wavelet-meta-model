"""
Scheduled-event and earnings tables (SPEC §12 `calendar` and `earnings`, U13): checked-in CSVs, read without network.

    data/calendar/events.csv    kind, date, event_time, value, known_from, scheduled, source
    data/calendar/earnings.csv  symbol, date, timing, accepted_at, known_from, accession

Kinds: FOMC (statement days, 14:00 ET; the two unscheduled 2020 statements at their own times), CPI and NFP
(release days, 08:30 ET, from BLS's release archives and schedules; BLS also releases on Good Friday, a federal
workday when the exchange is closed — 2017-04-14, 2020-04-10, 2021-04-02, 2023-04-07, 2026-04-03 — so a release
day is not always a session), OPEX (third Friday, the session before when it
is a holiday), MONTH_END (last session of the month), TOM (value = −1 for the month's last session, +1 … +3 for the
next month's first three), PRE_HOLIDAY (a session followed by a weekday closure). Early closes are in the exchange
calendar already.

`known_from` is the NY date from whose midnight a row is public (its `available_at`):
- scheduled releases and calendar-derived kinds: 1 January of the event's year (the Fed, BLS and NYSE publish the
  year's schedule before it starts);
- releases moved by the 2025 shutdown and the early-February 2026 funding lapse (CPI / NFP dated 2025-10-01 …
  2026-02-28): the release date itself (conservative; the new dates were announced days to weeks ahead);
- PRE_HOLIDAY before an unscheduled closure (2018-12-05, 2025-01-09): the Monday of the closure week
  (conservative; both were announced earlier);
- unscheduled FOMC statements: the statement instant (`available_at` = `event_at`).

Earnings (SEC EDGAR, Form 8-K Item 2.02 "Results of Operations", 2016 →): the acceptance time is the "Accepted"
field (Eastern time) of each filing's index page. The submissions JSON's `acceptanceDateTime` is not used: for some
filers (AAPL, AMZN, META, JPM, UNH) it is the true instant plus one more New York UTC offset (e.g. AAPL 2024-08-01
16:30:26 ET appears as 2024-08-02T00:30:26Z; GOOGL's values are right). Date = the acceptance's NY date; `timing`:
before 09:30 ET `bmo`, 16:00 ET or later `amc`, otherwise `dmh`. The announcement dates of the releases are not in
EDGAR: a release matching the company's usual timing (`amc` for AAPL MSFT NVDA AMZN GOOGL META, `bmo` for JPM UNH
XOM) is public from midnight of its day; any other filing (pre-announcements, XOM's mid-quarter "earnings
considerations") only from its acceptance time.

    uv run python -m data.events    # rebuild both CSVs (network: federalreserve.gov, bls.gov, sec.gov)
"""

import json
import re
import time
from pathlib import Path

import pandas as pd

from data.alpaca_source import NY_TZ

CALENDAR_DIR = Path(__file__).resolve().parent / "calendar"
EVENTS_CSV = CALENDAR_DIR / "events.csv"
EARNINGS_CSV = CALENDAR_DIR / "earnings.csv"
EVENT_KINDS = ("FOMC", "CPI", "NFP", "OPEX", "MONTH_END", "TOM", "PRE_HOLIDAY")
EVENT_COLUMNS = ["kind", "date", "event_time", "value", "known_from", "scheduled", "source"]
EARNINGS_COLUMNS = ["symbol", "date", "timing", "accepted_at", "known_from", "accession"]
TIMING = {"bmo": -1.0, "dmh": 0.0, "amc": 1.0}
FIRST_YEAR, LAST_YEAR = 2016, 2027

FOMC_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_ARCHIVE_URL = {"CPI": "https://www.bls.gov/bls/news-release/cpi.htm",
                   "NFP": "https://www.bls.gov/bls/news-release/empsit.htm"}  # fmt: skip
BLS_SCHEDULE_URL = {"CPI": "https://www.bls.gov/schedule/news_release/cpi.htm",
                    "NFP": "https://www.bls.gov/schedule/news_release/empsit.htm"}  # fmt: skip
BLS_PREFIX = {"CPI": "cpi", "NFP": "empsit"}
# bls.gov rejects requests without browser-like headers
BLS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 "
    "Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Dest": "document",
    "Upgrade-Insecure-Requests": "1",
}
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
SEC_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{acc}-index.htm"
SEC_HEADERS = {"User-Agent": "wavelet-meta-model research admin@example.com"}  # SEC asks for a contact-style UA
EARNINGS_CIK = {"AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724, "GOOGL": 1652044,
                "META": 1326801, "JPM": 19617, "XOM": 34088, "UNH": 731766}  # fmt: skip
EARNINGS_CONVENTION = {**dict.fromkeys(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"], "amc"),
                       **dict.fromkeys(["JPM", "UNH", "XOM"], "bmo")}  # fmt: skip

# Unscheduled FOMC policy statements (the Fed's pages list the meetings, not the statement times): the 2020-03-02
# conference call's rate cut was announced 2020-03-03 10:00 ET; the 2020-03-15 (Sunday) statement at 17:00 ET.
# The 2019-10-04 unscheduled meeting issued no statement that day (reserve-management purchases followed 2019-10-11).
FOMC_UNSCHEDULED = {"2020-03-02": ("2020-03-03", "10:00"), "2020-03-15": ("2020-03-15", "17:00"), "2019-10-04": None}
UNSCHEDULED_CLOSURES = {"2018-12-05": "2018-12-03", "2025-01-09": "2025-01-06"}  # closure → conservative known_from
SHUTDOWN_RESCHEDULED = ("2025-10-01", "2026-02-28")
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}  # fmt: skip


# ── Readers ──────────────────────────────────────────────────────────────────


def read_events(path: Path = EVENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"event_time": str, "source": str}, keep_default_na=False)
    if list(df.columns) != EVENT_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {EVENT_COLUMNS}")
    unknown = set(df["kind"]) - set(EVENT_KINDS)
    if unknown:
        raise ValueError(f"{path}: unknown kinds {sorted(unknown)}")
    if df.duplicated(["kind", "date"]).any():
        raise ValueError(f"{path}: duplicate (kind, date) rows")
    return df


def read_earnings(path: Path = EARNINGS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != EARNINGS_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {EARNINGS_COLUMNS}")
    if not set(df["timing"]) <= set(TIMING):
        raise ValueError(f"{path}: timing must be one of {sorted(TIMING)}")
    return df


def _midnight_utc(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(NY_TZ).tz_convert("UTC")


def event_series(source: str, name: str) -> pd.DataFrame:
    """
    One kind (calendar) or one symbol (earnings) as an exo series: index = date, `value`, `event_at` (UTC, NaT when
    the event has no time), for earnings `timing` (−1 bmo, 0 dmh, +1 amc), and `available_at` (UTC).
    """
    if source == "calendar":
        if name not in EVENT_KINDS:
            raise ValueError(f"unknown calendar kind {name!r}; expected one of {EVENT_KINDS}")
        ev = read_events()
        ev = ev[ev["kind"] == name]
        idx = pd.DatetimeIndex(pd.to_datetime(ev["date"]), name="date")
        times = [f"{d} {t}" if t else None for d, t in zip(ev["date"], ev["event_time"])]
        event_at = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize(NY_TZ).tz_convert("UTC")
        avail = _midnight_utc(ev["known_from"])
        unsched = (ev["scheduled"].astype(int) == 0).to_numpy()
        avail = avail.where(~unsched, event_at)
        out = pd.DataFrame({"value": ev["value"].astype(float).to_numpy()}, index=idx)
    elif source == "earnings":
        ea = read_earnings()
        ea = ea[ea["symbol"] == name.upper()]
        if ea.empty:
            raise ValueError(f"no earnings rows for {name!r}; symbols: {sorted(set(read_earnings()['symbol']))}")
        idx = pd.DatetimeIndex(pd.to_datetime(ea["date"]), name="date")
        event_at = pd.DatetimeIndex(pd.to_datetime(ea["accepted_at"], utc=True))
        known = _midnight_utc(ea["known_from"])
        avail = known.where(ea["known_from"].str.len().to_numpy() > 0, event_at)
        out = pd.DataFrame({"value": 1.0, "timing": ea["timing"].map(TIMING).to_numpy(float)}, index=idx)
    else:
        raise ValueError(f"event_series serves 'calendar' and 'earnings', not {source!r}")
    out["event_at"] = event_at
    out["available_at"] = avail
    return out.sort_index()


# ── Parsers ──────────────────────────────────────────────────────────────────


def _month(token: str) -> int:
    return MONTHS[token.strip().lower()[:3]]


def _meeting_end(year: int, months: str, days: str) -> pd.Timestamp:
    """Last day of a meeting written as months "June" / "Jan/Feb" and days "14-15" / "31-1" / "15-16*"."""
    ms = [_month(m) for m in months.split("/")]
    ds = [int(d) for d in re.findall(r"\d+", days)]
    month = ms[-1] if len(ds) > 1 and ds[-1] < ds[0] else ms[0]  # "31-1": the meeting ends in the second month
    return pd.Timestamp(year=year, month=month, day=ds[-1])


def parse_fomc_historical(html: str, year: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """
    A fomchistoricalYYYY page → (scheduled statement days, unscheduled meeting days). Headings read
    "June 14-15 Meeting - 2016", "Jan/Feb 31-1 Meeting - 2017", "March 15 (unscheduled) Meeting - 2020",
    "March 17-18 (cancelled) Meeting - 2020", "March 19 (notation vote) - 2020"; notation votes and cancelled
    meetings issue no statement.
    """
    scheduled, unscheduled = [], []
    for text in re.findall(r"<h5[^>]*>([^<]*)</h5>", html):
        m = re.match(rf"\s*([A-Za-z/]+)\s+([\d\-]+)\s*(\((\w[\w ]*)\))?\s*(Meeting)?\s*-\s*{year}\s*$", text)
        if not m:
            continue
        months, days, note = m.group(1), m.group(2), (m.group(4) or "").lower()
        if note in ("notation vote", "cancelled"):
            continue
        if note not in ("", "unscheduled"):
            raise ValueError(f"unrecognised FOMC heading {text!r}")
        (unscheduled if note == "unscheduled" else scheduled).append(_meeting_end(year, months, days))
    return scheduled, unscheduled


def parse_fomc_calendar(html: str) -> dict[int, list[pd.Timestamp]]:
    """fomccalendars.htm → {year: scheduled statement days}; notation votes are skipped, "*" (SEP) kept."""
    out: dict[int, list[pd.Timestamp]] = {}
    panels = re.split(r"<h4><a id=\"\d+\">(\d{4}) FOMC Meetings</a></h4>", html)
    for i in range(1, len(panels), 2):
        year, body = int(panels[i]), panels[i + 1]
        months = re.findall(r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>", body)
        dates = re.findall(r"fomc-meeting__date[^>]*>([^<]+)<", body)
        if len(months) != len(dates):
            raise ValueError(f"FOMC {year}: {len(months)} months vs {len(dates)} dates")
        days = []
        for mo, d in zip(months, dates):
            low = d.lower()
            if "notation" in low or "cancel" in low:
                continue
            if "unscheduled" in low:
                raise ValueError(f"FOMC {year}: unscheduled meeting {mo} {d} needs a statement time in "
                                 "FOMC_UNSCHEDULED")  # fmt: skip
            days.append(_meeting_end(year, mo, d))
        out[year] = days
    return out


def parse_bls_archive(html: str, prefix: str) -> list[pd.Timestamp]:
    """BLS release archive page → release days from the links `{prefix}_MMDDYYYY.htm` (placeholders are skipped)."""
    stamps = sorted(set(re.findall(rf"/{prefix}_(\d{{8}})\.htm", html)))
    return [pd.Timestamp(f"{s[4:]}-{s[:2]}-{s[2:4]}") for s in stamps]


def parse_bls_schedule(html: str) -> list[pd.Timestamp]:
    """BLS release schedule table → release days (cells like "Feb. 13, 2026", "May 12, 2026", "Sept. 11, 2026")."""
    out = []
    for mon, day, year in re.findall(r"<td>\s*([A-Z][a-z]{2,4})\.?\s+(\d{1,2}),\s+(\d{4})\s*</td>", html):
        out.append(pd.Timestamp(year=int(year), month=_month(mon), day=int(day)))
    return sorted(set(out))


# ── Derived calendar kinds ───────────────────────────────────────────────────


def derived_events(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """OPEX, MONTH_END, TOM and PRE_HOLIDAY rows from exchange sessions (whole months inside the calendar only)."""
    s = pd.DatetimeIndex(sessions).sort_values()
    first_m, last_m = s[0].to_period("M") + 1, s[-1].to_period("M") - 1  # complete months only
    rows = []
    by_month = pd.Series(s, index=s.to_period("M")).groupby(level=0)
    months = {m: pd.DatetimeIndex(g.to_numpy()) for m, g in by_month}
    for m in pd.period_range(first_m, last_m, freq="M"):
        days = months.get(m)
        if days is None:
            continue
        third_fri = pd.Timestamp(year=m.year, month=m.month, day=1) + pd.offsets.WeekOfMonth(week=2, weekday=4)
        opex = days[days <= third_fri][-1]
        rows.append(("OPEX", opex, "16:00", 1.0))
        rows.append(("MONTH_END", days[-1], "16:00", 1.0))
        rows.append(("TOM", days[-1], "", -1.0))
        nxt = months.get(m + 1)
        if nxt is not None and m + 1 <= last_m:
            for k in range(3):
                rows.append(("TOM", nxt[k], "", float(k + 1)))
    in_s = set(s)
    for d in s[:-1]:
        nxt_weekday = d + pd.offsets.BDay(1)
        if nxt_weekday <= s[-1] and nxt_weekday not in in_s:
            rows.append(("PRE_HOLIDAY", d, "", 1.0))
    df = pd.DataFrame(rows, columns=["kind", "date", "event_time", "value"])
    df = df[(df["date"].dt.year >= FIRST_YEAR)]
    known = df["date"].map(lambda d: pd.Timestamp(year=d.year, month=1, day=1))
    for closure, known_from in UNSCHEDULED_CLOSURES.items():
        pre = s[s < pd.Timestamp(closure)][-1]
        known = known.where(~((df["kind"] == "PRE_HOLIDAY") & (df["date"] == pre)), pd.Timestamp(known_from))
    df["known_from"] = known
    df["scheduled"] = 1
    df["source"] = "exchange calendar (Alpaca), data/cache/calendar.json"
    return df


# ── Builders (network) ───────────────────────────────────────────────────────


def build_events(http, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    cal = parse_fomc_calendar(http(FOMC_CALENDAR_URL, None))
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        if year in cal:
            sched, unsched, url = cal[year], [], FOMC_CALENDAR_URL
        else:
            url = FOMC_HISTORICAL_URL.format(year=year)
            sched, unsched = parse_fomc_historical(http(url, None), year)
        for d in sched:
            rows.append(("FOMC", d, "14:00", 1.0, pd.Timestamp(year=year, month=1, day=1), 1, url))
        for d in unsched:
            key = str(d.date())
            if key not in FOMC_UNSCHEDULED:
                raise ValueError(f"unscheduled FOMC meeting {key} needs a statement time in FOMC_UNSCHEDULED")
            if FOMC_UNSCHEDULED[key] is None:
                continue
            day, hhmm = FOMC_UNSCHEDULED[key]
            rows.append(("FOMC", pd.Timestamp(day), hhmm, 1.0, pd.Timestamp(day), 0, url))
    lo, hi = (pd.Timestamp(x) for x in SHUTDOWN_RESCHEDULED)
    for kind in ("CPI", "NFP"):
        days = set(parse_bls_archive(http(BLS_ARCHIVE_URL[kind], BLS_HEADERS), BLS_PREFIX[kind]))
        days |= set(parse_bls_schedule(http(BLS_SCHEDULE_URL[kind], BLS_HEADERS)))
        for d in sorted(days):
            if not FIRST_YEAR <= d.year <= LAST_YEAR:
                continue
            known = d if lo <= d <= hi else pd.Timestamp(year=d.year, month=1, day=1)
            rows.append((kind, d, "08:30", 1.0, known, 1, f"{BLS_ARCHIVE_URL[kind]} + {BLS_SCHEDULE_URL[kind]}"))
    ev = pd.DataFrame(rows, columns=["kind", "date", "event_time", "value", "known_from", "scheduled", "source"])
    ev = pd.concat([ev, derived_events(sessions)], ignore_index=True)
    ev = ev.sort_values(["kind", "date"], kind="stable").reset_index(drop=True)
    for c in ("date", "known_from"):
        ev[c] = pd.to_datetime(ev[c]).dt.strftime("%Y-%m-%d")
    return ev[EVENT_COLUMNS]


def _sec_filings(cik: int, http) -> list[dict]:
    d = json.loads(http(SEC_SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"), SEC_HEADERS))
    blocks = [d["filings"]["recent"]]
    for f in d["filings"].get("files", []):
        if f["filingTo"] >= f"{FIRST_YEAR}-01-01":
            time.sleep(0.2)  # SEC fair-access limit: ≤ 10 requests / s
            blocks.append(json.loads(http(SEC_SUBMISSIONS_URL.format(name=f["name"]), SEC_HEADERS)))
    rows = []
    for b in blocks:
        for i in range(len(b["form"])):
            rows.append({k: b[k][i] for k in ("form", "items", "filingDate", "accessionNumber")})
    return rows


def parse_accepted(html: str) -> pd.Timestamp:
    """The "Accepted" field of an EDGAR filing index page (Eastern time) → tz-aware UTC instant."""
    m = re.search(r'infoHead">Accepted</div>\s*<div class="info">(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)</div>', html)
    if not m:
        raise ValueError("no Accepted field on the EDGAR index page")
    return pd.Timestamp(m.group(1)).tz_localize(NY_TZ).tz_convert("UTC")


def classify_earnings(symbol: str, accepted_utc: pd.Timestamp) -> tuple[str, str, str]:
    """(date, timing, known_from) of one 8-K Item 2.02 filing (rules in the module docstring)."""
    t = accepted_utc.tz_convert(NY_TZ)
    hm = t.hour * 60 + t.minute
    timing = "bmo" if hm < 9 * 60 + 30 else "amc" if hm >= 16 * 60 else "dmh"
    date = str(t.date())
    known = date if timing == EARNINGS_CONVENTION[symbol] else ""
    return date, timing, known


def build_earnings(http) -> pd.DataFrame:
    rows = []
    for sym, cik in EARNINGS_CIK.items():
        for f in _sec_filings(cik, http):
            if f["form"] != "8-K" or "2.02" not in f["items"].split(",") or f["filingDate"] < f"{FIRST_YEAR}-01-01":
                continue
            a = f["accessionNumber"]
            time.sleep(0.15)  # SEC fair-access limit: ≤ 10 requests / s
            acc = parse_accepted(http(SEC_INDEX_URL.format(cik=cik, nodash=a.replace("-", ""), acc=a), SEC_HEADERS))
            if str(acc.tz_convert(NY_TZ).date()) > f["filingDate"]:
                raise ValueError(f"{sym} {a}: accepted {acc} after its filing date {f['filingDate']}")
            date, timing, known = classify_earnings(sym, acc)
            rows.append((sym, date, timing, acc.isoformat(), known, a))
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=EARNINGS_COLUMNS).sort_values(["symbol", "date", "accepted_at"])
    dup = df.duplicated(["symbol", "date"], keep=False)
    if dup.any():
        print(f"[EVENTS] {int(dup.sum())} same-day 8-K 2.02 filings kept (e.g. {df[dup].iloc[0].tolist()[:3]})")
    return df.reset_index(drop=True)


def curl_get(url: str, headers: dict) -> str:
    import subprocess

    args = ["curl", "-sS", "--fail", "--compressed", "--max-time", "60"]
    for k, v in headers.items():
        args += ["-A", v] if k == "User-Agent" else ["-H", f"{k}: {v}"]
    return subprocess.run([*args, url], check=True, capture_output=True).stdout.decode("utf-8", errors="replace")


def main() -> None:
    from data.bars import get_calendar
    from data.exo import http_get

    def http(url, headers):
        if url.startswith("https://www.bls.gov/"):  # bls.gov answers 403 to python-requests whatever the headers
            return curl_get(url, headers)
        return http_get(url, headers)

    sessions = get_calendar(pd.Timestamp.today().normalize()).index
    ev = build_events(http, sessions)
    CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVENTS_CSV, index=False, lineterminator="\n")
    print(f"[EVENTS] {len(ev):,} rows → {EVENTS_CSV}")
    print(ev.groupby("kind")["date"].agg(["count", "min", "max"]).to_string())
    ea = build_earnings(http)
    ea.to_csv(EARNINGS_CSV, index=False, lineterminator="\n")
    print(f"[EVENTS] {len(ea):,} rows → {EARNINGS_CSV}")
    print(ea.groupby(["symbol", "timing"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
