"""
Bar cache as native numpy arrays.

Each (feed, adjustment, timeframe, symbol) is two files:
    {SYMBOL}.npz  — `ts` (int64 ns since epoch, UTC) + one float64 array per column
    {SYMBOL}.json — metadata incl. a sha256 over the arrays, verified on every load

Writes are atomic (temp file + os.replace; the .npz lands before the .json, so a
crash in between leaves a hash mismatch that load() rejects rather than stale data).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


class CacheError(RuntimeError):
    pass


def cache_paths(cache_dir: Path, feed: str, adjustment: str, timeframe: str, symbol: str) -> tuple[Path, Path]:
    base = Path(cache_dir) / feed / adjustment / timeframe / symbol.upper()
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if df.index.tz is None:
        raise CacheError("Refusing to cache a tz-naive index")
    ts = df.index.tz_convert("UTC").as_unit("ns").asi8.astype(np.int64)
    return {"ts": ts, **{c: df[c].to_numpy(dtype=np.float64) for c in df.columns}}


def _sha256(arrays: dict[str, np.ndarray], columns: list[str]) -> str:
    h = hashlib.sha256()
    for name in ["ts", *columns]:
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_bars(npz_path: Path, json_path: Path, df: pd.DataFrame, meta: dict) -> dict:
    """
    Write df (tz-aware index, float columns) and its metadata.

    Args:
        npz_path (Path): Target .npz path.
        json_path (Path): Target .json sidecar path.
        df (pd.DataFrame): Bars to store; index must be unique and sorted.
        meta (dict): Caller metadata (symbol, timeframe, feed, adjustment, tz, coverage, ...).

    Returns:
        dict: The full metadata written.
    """
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise CacheError("Index must be sorted and unique")
    columns = list(df.columns)
    arrays = _arrays(df)
    full_meta = {
        **meta,
        "schema_version": SCHEMA_VERSION,
        "columns": columns,
        "n_bars": len(df),
        "first_ts": df.index[0].isoformat() if len(df) else None,
        "last_ts": df.index[-1].isoformat() if len(df) else None,
        "sha256": _sha256(arrays, columns),
    }
    atomic_write(npz_path, lambda fh: np.savez(fh, **arrays))
    atomic_write(json_path, lambda fh: fh.write(json.dumps(full_meta, indent=2, sort_keys=True).encode()))
    return full_meta


def load_bars_cache(npz_path: Path, json_path: Path, expect: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Load and verify a cached bar set.

    Raises CacheError if the sidecar is missing, the schema version differs, the
    array hash does not match, or any `expect` key disagrees with the metadata
    (e.g. feed/adjustment implied by the cache path).

    Returns:
        tuple[pd.DataFrame, dict]: Bars indexed in meta["tz"], and the metadata.
    """
    if not json_path.exists():
        raise CacheError(f"Missing metadata sidecar {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(f"{json_path}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    for key, val in (expect or {}).items():
        if meta.get(key) != val:
            raise CacheError(f"{json_path}: {key}={meta.get(key)!r}, expected {val!r}")
    columns = meta["columns"]
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in ["ts", *columns]}
    if _sha256(arrays, columns) != meta["sha256"]:
        raise CacheError(f"{npz_path}: array hash does not match {json_path}")
    index = pd.DatetimeIndex(arrays["ts"].astype("datetime64[ns]")).tz_localize("UTC").tz_convert(meta["tz"])
    df = pd.DataFrame({c: arrays[c] for c in columns}, index=index)
    return df, meta


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union on timestamp; for duplicate stamps the newer fetch wins. Result sorted and unique."""
    both = pd.concat([old, new])
    return both[~both.index.duplicated(keep="last")].sort_index()
