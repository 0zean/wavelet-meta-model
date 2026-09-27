"""
On-disk cache for static feature groups (SPEC §3).

Key = sha256 over (group, symbol, timeframe, code hash, feature-relevant cfg fields, data hash).
The data hash covers the bars' timestamps and OHLCV(+vwap) values, so a different range,
adjustment or top-up is a different key. The code hash covers every `features/*.py`
source file (any feature-code change invalidates every group — safe over precise) and
the numpy/pandas/scipy/pywddff/fracdiff versions.
Files: `{root}/{symbol}/{timeframe}/{group}/{key}.npz`, plain numpy arrays, no pickle.
"""

import dataclasses
import hashlib
import json
import os
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "cache" / "features"

# RunConfig fields that cannot change a feature value. Everything else is hashed, so a new
# field is feature-relevant (cache-busting) unless it is added here.
_NON_FEATURE_FIELDS = {
    "CLF_PARAMS",
    "REG_PARAMS",
    "META_PARAMS",
    "SEED",
    "CLF_THRESH",
    "META_THRESH",
    "META_MIN_RET",
    "WINDOW_UNIT",
    "INITIAL_TRAIN",
    "VAL",
    "TEST",
    "EMBARGO",
    "MIN_TRAIN_EVENTS",
    "MIN_VAL_EVENTS",
    "SLIPPAGE_PCT",
    "INIT_CASH",
    "SIZE",
    "ALLOW_HOLDOUT",
    "LOW_MOVE_PCTILE",
    "FEATURE_GROUPS",
    "FEATURE_SELECTION",
    "CMDA_TREES",
    "CMDA_SPLITS",
    "CV_EMBARGO_PCT",
}


@cache
def code_hash() -> str:
    """Source of every features/*.py file plus the versions of the numeric libraries they call."""
    from importlib.metadata import version

    h = hashlib.sha256()
    for lib in ("numpy", "pandas", "scipy", "pywddff", "fracdiff"):
        h.update(f"{lib}={version(lib)}".encode())
    for path in sorted((Path(__file__).resolve().parent).glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.index.tz).encode())
    h.update(np.ascontiguousarray(df.index.asi8).tobytes())
    for col in ("open", "high", "low", "close", "volume", "vwap"):
        if col in df:
            h.update(col.encode())
            h.update(np.ascontiguousarray(df[col].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def cfg_hash(cfg) -> str:
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.name not in _NON_FEATURE_FIELDS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def cache_key(group: str, symbol: str, cfg, df: pd.DataFrame, context: dict | None) -> str:
    parts = [group, symbol, cfg.TIMEFRAME, code_hash(), cfg_hash(cfg), data_hash(df)]
    for name in sorted(context or {}):
        parts += [name, data_hash(context[name])]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _path(root: Path, symbol: str, timeframe: str, group: str, key: str) -> Path:
    return Path(root) / symbol / timeframe / group / f"{key}.npz"


def load(root, symbol: str, timeframe: str, group: str, key: str, index: pd.DatetimeIndex) -> pd.DataFrame | None:
    path = _path(root, symbol, timeframe, group, key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["__ts__"], index.asi8):
            raise ValueError(f"feature cache {path}: index does not match the bars (hash collision?)")
        cols = [str(c) for c in z["__columns__"]]
        return pd.DataFrame({c: z[f"c{i}"] for i, c in enumerate(cols)}, index=index)


def save(root, symbol: str, timeframe: str, group: str, key: str, feats: pd.DataFrame) -> None:
    path = _path(root, symbol, timeframe, group, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"__ts__": feats.index.asi8, "__columns__": np.array(feats.columns, dtype=str)}
    arrays |= {f"c{i}": feats[c].to_numpy(dtype=float) for i, c in enumerate(feats.columns)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
