"""
Point-in-time access to exogenous series for feature groups (SPEC §12 invariants, §15; U15).

A group that declares `needs=("exo",)` never sees the raw frames of `context["exo"]`: the feature builder hands it an
`Exo` view restricted to the series the group declares (`FeatureGroup.exo`), and every accessor applies the
point-in-time rule

    a row is used on a bar only when available_at ≤ the bar's stamp (its open; NY time for a naive index),

so a value observed later the same day (the VIX close, available 16:20 ET) cannot reach an earlier bar.

    asof(name, index, column)      value of the visible row with the greatest available_at (ties: the latest date)
    ratio(num, den, index)         num / den built on the two series' common observation dates first, the row
                                   available when both are (VIX holiday rows from 2022 have no VIX3M partner)
    change(name, index, periods)   change over `periods` observations, computed on the observation series and
                                   available when its later row is (log=True: log change)
    next_date / last_date          the first visible row dated on or after / last dated on or before the bar's
                                   NY date (calendar kinds: scheduled rows are visible long before their date)
    on_day(name, index, column)    the visible row dated on the bar's NY date
    known_dates(name, index)       (k, dates): rows ordered by available_at; row i is visible at bar j iff i < k[j]
"""

import hashlib

import numpy as np
import pandas as pd

from data.alpaca_source import NY_TZ


def bar_stamps_utc(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar stamps as UTC int64 ns (a naive index is NY time)."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize(NY_TZ) if idx.tz is None else idx
    return idx.tz_convert("UTC").as_unit("ns").asi8


def bar_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Each bar's NY date as datetime64[D]."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    return idx.normalize().to_numpy().astype("datetime64[D]")


def _avail_ns(frame: pd.DataFrame) -> np.ndarray:
    return pd.DatetimeIndex(frame["available_at"]).tz_convert("UTC").as_unit("ns").asi8


def _asof(values: np.ndarray, avail: np.ndarray, dates: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """values[i] of the row with the greatest avail[i] ≤ stamp (ties: the latest date); NaN where none."""
    order = np.lexsort((dates, avail))
    pos = np.searchsorted(avail[order], stamps, side="right") - 1
    out = np.full(len(stamps), np.nan)
    ok = pos >= 0
    out[ok] = np.asarray(values, dtype=float)[order][pos[ok]]
    return out


class Exo:
    """Point-in-time view of exogenous series (see the module docstring). `frames`: name → load_series frame."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        for name, f in frames.items():
            if "available_at" not in f or pd.DatetimeIndex(f["available_at"]).tz is None:
                raise ValueError(f"exo series {name!r} needs a tz-aware `available_at` column")
            if f["available_at"].isna().any():
                raise ValueError(f"exo series {name!r} has rows without `available_at`")
        self._frames = {n: f.sort_index() for n, f in frames.items()}

    @property
    def names(self) -> list[str]:
        return sorted(self._frames)

    def restrict(self, names) -> "Exo":
        """The view on `names` only (a group sees the series it declares, no others)."""
        missing = [n for n in names if n not in self._frames]
        if missing:
            raise ValueError(f"exo series {missing} are not in the context (have {self.names})")
        return Exo({n: self._frames[n] for n in names})

    def digest(self) -> str:
        """Hash of every series' index, columns and available_at (the feature-cache key's context part)."""
        h = hashlib.sha256()
        for name in self.names:
            f = self._frames[name]
            h.update(name.encode())
            h.update(np.ascontiguousarray(f.index.as_unit("ns").asi8).tobytes())
            for col in sorted(f.columns):
                h.update(col.encode())
                x = f[col]
                arr = pd.DatetimeIndex(x).tz_convert("UTC").as_unit("ns").asi8 if col in ("available_at", "event_at") \
                    else x.to_numpy(dtype=float)  # fmt: skip
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def _get(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            raise KeyError(f"exo series {name!r} is not declared by this group (have {self.names})")
        return self._frames[name]

    @staticmethod
    def _dates(f: pd.DataFrame) -> np.ndarray:
        return f.index.as_unit("ns").asi8

    # ── value series ──

    def asof(self, name: str, index: pd.DatetimeIndex, column: str = "value") -> np.ndarray:
        f = self._get(name)
        return _asof(f[column].to_numpy(float), _avail_ns(f), self._dates(f), bar_stamps_utc(index))

    def ratio(self, num: str, den: str, index: pd.DatetimeIndex) -> np.ndarray:
        a, b = self._get(num), self._get(den)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a["value"].to_numpy(float) / b["value"].to_numpy(float)
        r = np.where(np.isfinite(r), r, np.nan)
        avail = np.maximum(_avail_ns(a), _avail_ns(b))
        return _asof(r, avail, self._dates(a), bar_stamps_utc(index))

    def change(self, name: str, index: pd.DatetimeIndex, periods: int = 1, log: bool = False) -> np.ndarray:
        f = self._get(name)
        v = f["value"].to_numpy(float)
        x = np.log(v) if log else v
        d = np.full(len(x), np.nan)
        d[periods:] = x[periods:] - x[:-periods]
        avail = np.maximum.accumulate(_avail_ns(f))  # a change is known once its later row (and every earlier) is
        return _asof(d, avail, self._dates(f), bar_stamps_utc(index))

    # ── dated rows (calendar kinds) ──

    def _rows(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        f = self._get(name)
        return f.index.to_numpy().astype("datetime64[D]"), _avail_ns(f)

    def next_date(self, name: str, index: pd.DatetimeIndex) -> np.ndarray:
        """The first visible row dated on or after each bar's NY date (NaT where none is visible)."""
        dates, avail = self._rows(name)
        day, stamp = bar_days(index), bar_stamps_utc(index)
        pos = np.searchsorted(dates, day, side="left")
        out = np.full(len(day), np.datetime64("NaT"), dtype="datetime64[D]")
        todo = np.arange(len(day))
        while len(todo):
            p = pos[todo]
            inside = p < len(dates)
            todo, p = todo[inside], p[inside]
            seen = avail[p] <= stamp[todo]
            out[todo[seen]] = dates[p[seen]]
            todo = todo[~seen]
            pos[todo] += 1
        return out

    def last_date(self, name: str, index: pd.DatetimeIndex) -> np.ndarray:
        """The last visible row dated on or before each bar's NY date (NaT where none is visible)."""
        dates, avail = self._rows(name)
        day, stamp = bar_days(index), bar_stamps_utc(index)
        pos = np.searchsorted(dates, day, side="right") - 1
        out = np.full(len(day), np.datetime64("NaT"), dtype="datetime64[D]")
        todo = np.arange(len(day))
        while len(todo):
            p = pos[todo]
            inside = p >= 0
            todo, p = todo[inside], p[inside]
            seen = avail[p] <= stamp[todo]
            out[todo[seen]] = dates[p[seen]]
            todo = todo[~seen]
            pos[todo] -= 1
        return out

    def on_day(self, name: str, index: pd.DatetimeIndex, column: str = "value") -> np.ndarray:
        """`column` of the visible row dated on each bar's NY date: float (NaN) or, for event_at, UTC int64 ns
        (np.iinfo(int64).min where there is none)."""
        f = self._get(name)
        dates, avail = self._rows(name)
        day, stamp = bar_days(index), bar_stamps_utc(index)
        pos = np.searchsorted(dates, day, side="left")
        hit = pos < len(dates)
        hit[hit] = dates[pos[hit]] == day[hit]
        hit[hit] = avail[pos[hit]] <= stamp[hit]
        if column == "event_at":
            vals = pd.DatetimeIndex(f["event_at"]).tz_convert("UTC").as_unit("ns").asi8
            out = np.full(len(day), np.iinfo(np.int64).min, dtype=np.int64)
        else:
            vals = f[column].to_numpy(float)
            out = np.full(len(day), np.nan)
        out[hit] = vals[pos[hit]]
        return out

    def known_dates(self, name: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """(k, dates): the rows' dates ordered by available_at (ties: date); row i is visible at bar j iff i < k[j]."""
        dates, avail = self._rows(name)
        order = np.lexsort((dates, avail))
        k = np.searchsorted(avail[order], bar_stamps_utc(index), side="right")
        return k, dates[order]
