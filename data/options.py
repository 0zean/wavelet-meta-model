"""
Forward collector of the SPY / QQQ option chain (SPEC §12 `alpaca_options`, U13).

What Alpaca serves (checked 2026-10-09, free plan): `/v1beta1/options/snapshots/{underlying}` is the *current* chain
only (latest quote and trade, and `impliedVolatility` / `greeks` for contracts with a usable quote; indicative feed);
`/v2/options/contracts` carries `open_interest` as of `open_interest_date` (the previous session), also current only;
historical option *bars* exist from 2024-02. There is no historical snapshot or open-interest series, so the chain
can only be accumulated forward, one snapshot per day — the U15 gamma proxy falls back to a VIX-term-structure / skew
proxy for the development window.

    uv run python -m data.options SPY QQQ [--days 60]    # today's snapshot → {cache}/{UNDERLYING}/{date}.npz + .json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.alpaca_source import NY_TZ
from data.exo import EXO_CACHE_DIR
from data.store import CacheError, _sha256, atomic_write

OPTIONS_DIR = EXO_CACHE_DIR / "alpaca_options"
SCHEMA_VERSION = 1
NUMERIC = ["strike", "bid", "ask", "iv", "delta", "gamma", "theta", "vega", "open_interest"]
CHAIN_COLUMNS = ["contract", "type", "expiry", *NUMERIC, "quote_ts"]


def chain_frame(snapshots: dict[str, dict], contracts: list[dict]) -> pd.DataFrame:
    """Snapshots + contract metadata → one row per contract (NaN where the snapshot has no IV / Greeks / quote)."""
    meta = {c["symbol"]: c for c in contracts}
    rows = []
    for sym, s in snapshots.items():
        c = meta.get(sym)
        if c is None:  # a snapshot without contract metadata (strike, type, expiry) cannot be placed
            continue
        q, g = s.get("latestQuote") or {}, s.get("greeks") or {}
        oi = c.get("open_interest")
        rows.append({
            "contract": sym, "type": c["type"], "expiry": c["expiration_date"], "strike": float(c["strike_price"]),
            "bid": float(q.get("bp", np.nan)), "ask": float(q.get("ap", np.nan)),
            "iv": float(s.get("impliedVolatility", np.nan)),
            **{k: float(g.get(k, np.nan)) for k in ("delta", "gamma", "theta", "vega")},
            "open_interest": float(oi) if oi not in (None, "") else np.nan,
            "quote_ts": pd.Timestamp(q["t"]).tz_convert("UTC").as_unit("ns") if q.get("t") else pd.NaT,
        })  # fmt: skip
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    return df.sort_values(["expiry", "type", "strike"]).reset_index(drop=True)


def _paths(cache_dir: Path, underlying: str, day: pd.Timestamp) -> tuple[Path, Path]:
    base = Path(cache_dir) / underlying.upper() / day.strftime("%Y-%m-%d")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_chain(cache_dir: Path, underlying: str, day: pd.Timestamp, df: pd.DataFrame, oi_date: str | None) -> None:
    arrays = {"ts": pd.DatetimeIndex(df["quote_ts"]).as_unit("ns").asi8.astype(np.int64)}
    arrays |= {c: df[c].to_numpy(np.float64) for c in NUMERIC}
    arrays["is_call"] = (df["type"] == "call").to_numpy(np.float64)
    arrays["expiry"] = pd.to_datetime(df["expiry"]).to_numpy("datetime64[D]").astype(np.int64).astype(np.float64)
    cols = [k for k in arrays if k != "ts"]
    meta = {"schema_version": SCHEMA_VERSION, "underlying": underlying.upper(), "date": day.strftime("%Y-%m-%d"),
            "contracts": df["contract"].tolist(), "columns": cols, "n_rows": len(df), "open_interest_date": oi_date,
            "feed": "indicative", "sha256": _sha256(arrays, cols),
            "fetched_at": datetime.now().astimezone().isoformat()}  # fmt: skip
    npz, js = _paths(cache_dir, underlying, day)
    atomic_write(npz, lambda fh: np.savez(fh, **arrays))
    atomic_write(js, lambda fh: fh.write(json.dumps(meta).encode()))


def load_chain(cache_dir: Path, underlying: str, day) -> pd.DataFrame:
    npz, js = _paths(cache_dir, underlying, pd.Timestamp(day))
    if not js.exists():
        raise CacheError(f"Missing metadata sidecar {js}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(f"{js}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in ["ts", *meta["columns"]]}
    if _sha256(arrays, meta["columns"]) != meta["sha256"]:
        raise CacheError(f"{npz}: array hash does not match {js}")
    ts = arrays["ts"]
    return pd.DataFrame({
        "contract": meta["contracts"], "type": np.where(arrays["is_call"] > 0, "call", "put"),
        "expiry": pd.to_datetime(arrays["expiry"].astype(np.int64), unit="D").strftime("%Y-%m-%d"),
        **{c: arrays[c] for c in NUMERIC},
        "quote_ts": pd.DatetimeIndex(np.where(ts == np.iinfo(np.int64).min, np.datetime64("NaT"),
                                              ts.astype("datetime64[ns]"))).tz_localize("UTC"),
    })[CHAIN_COLUMNS]  # fmt: skip


def collect(underlyings: list[str], days: int, source, cache_dir: Path = OPTIONS_DIR) -> None:
    today = pd.Timestamp.now(tz=NY_TZ).tz_localize(None).normalize()
    horizon = (today + pd.Timedelta(days=days)).date()
    for u in underlyings:
        contracts = source.option_contracts(u, horizon)
        df = chain_frame(source.option_chain(u, horizon), contracts)
        oi_dates = sorted({c.get("open_interest_date") for c in contracts if c.get("open_interest_date")})
        save_chain(cache_dir, u, today, df, oi_dates[-1] if oi_dates else None)
        print(f"[OPTIONS] {u} {today.date()}: {len(df):,} contracts, IV on {int(df['iv'].notna().sum()):,}, "
              f"open interest on {int(df['open_interest'].notna().sum()):,}")  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    from data.alpaca_source import AlpacaSource

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("underlyings", nargs="*", default=["SPY", "QQQ"])
    p.add_argument("--days", type=int, default=60, help="expiries up to this many days ahead")
    a = p.parse_args(argv)
    collect([u.upper() for u in a.underlyings], a.days, AlpacaSource())


if __name__ == "__main__":
    main()
