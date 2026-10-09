"""
Primary diagnostics (PLAN U4), computed post hoc from the WFO signal frame — evaluation only:
outcomes are read after every side is fixed.

Per fold and overall, on the OOS events:
  n_events      events sided
  long_share    share of long sides (flags a primary stuck on one side)
  precision     P(side earns > META_MIN_RET)            — side-correct rate net of round-trip costs
  opportunity   P(long or short would earn > META_MIN_RET)
  recall        P(side earns > META_MIN_RET | a profitable side existed)
  turnover      share of consecutive events whose side flips
  net_bp        mean side-return net of round-trip slippage, in bp

Both sides are evaluated with the backtest's barrier rules (adverse barrier on same-bar ties). Flat events (side 0,
the SPEC §16 mechanism primaries) take no position: the metrics cover the sided events, and `n_flat` counts the flat
ones (a column present only when there are any).

`slot_diagnostics` (U16) splits the same metrics by the time-of-day slot of the entry bar (intraday), with the ISOM
count of each slot (events per session over the signals' sessions, SPEC §14).

    uv run python -m primaries.diagnostics runs/*/ --out results/primary_diagnostics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features.exits import exit_frame
from features.vol_profile import ny_dates
from utils.config import RunConfig


def _side_returns(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig, s: int) -> pd.Series:
    out = exit_frame(df, signals.index, signals["width"], cfg, side=pd.Series(s, index=signals.index))
    return s * out["ret"]


def _summary(side: pd.Series, r_long: pd.Series, r_short: pd.Series, cfg: RunConfig) -> dict:
    flat = side == 0
    n_flat = int(flat.sum())
    side, r_long, r_short = side[~flat], r_long[~flat], r_short[~flat]
    r_side = r_long.where(side > 0, r_short)
    ok = r_side > cfg.META_MIN_RET
    opp = (r_long > cfg.META_MIN_RET) | (r_short > cfg.META_MIN_RET)
    return {
        "n_events": len(side),
        "long_share": (side > 0).mean(),
        "precision": ok.mean(),
        "opportunity": opp.mean(),
        "recall": ok[opp].mean() if opp.any() else np.nan,
        "turnover": (side.diff().fillna(0) != 0).iloc[1:].mean() if len(side) > 1 else np.nan,
        "net_bp": (r_side.mean() - 2 * cfg.SLIPPAGE_PCT) * 1e4,
        **({"n_flat": n_flat} if n_flat else {}),
    }


def primary_diagnostics(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig, symbol: str = "") -> pd.DataFrame:
    """
    Diagnostics table: one row per fold plus fold="all".

    Args:
        df (pd.DataFrame): OHLC data covering the signals and their exits.
        signals (pd.DataFrame): WFO output (signed_dir, width, fold, primary).
        cfg (RunConfig): Run configuration.
        symbol (str, optional): Label for the symbol column. Defaults to "".

    Returns:
        pd.DataFrame: Columns symbol, timeframe, primary, fold, start, end + the metrics above.
    """
    r_long = _side_returns(df, signals, cfg, +1)
    r_short = _side_returns(df, signals, cfg, -1)
    sig = signals.loc[r_long.index.intersection(r_short.index)]
    rows = []
    for fold, g in [*sig.groupby("fold"), ("all", sig)]:
        rows.append(
            {
                "symbol": symbol,
                "timeframe": cfg.TIMEFRAME,
                "primary": g["primary"].iloc[0] if "primary" in g else cfg.PRIMARY,
                "fold": fold,
                "start": g.index[0],
                "end": g.index[-1],
                **_summary(g["signed_dir"], r_long.loc[g.index], r_short.loc[g.index], cfg),
            }
        )
    return pd.DataFrame(rows)


def slot_diagnostics(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig, symbol: str = "") -> pd.DataFrame:
    """
    The primary_diagnostics metrics per entry slot (the HH:MM of each event's entry bar, the bar after the event) plus
    slot "all"; `per_session` = the slot's sided events per session of the signals' span (its ISOM count).

    Args:
        df (pd.DataFrame): Intraday OHLC data covering the signals and their exits.
        signals (pd.DataFrame): Signal frame (signed_dir, width; fold-free signals are fine).
        cfg (RunConfig): Run configuration (its exit model evaluates both sides).
        symbol (str, optional): Label for the symbol column. Defaults to "".
    """
    r_long = _side_returns(df, signals, cfg, +1)
    r_short = _side_returns(df, signals, cfg, -1)
    sig = signals.loc[r_long.index.intersection(r_short.index)]
    pos = df.index.get_indexer(sig.index) + 1
    local = df.index[pos].tz_convert("America/New_York") if df.index.tz is not None else df.index[pos]
    slot = pd.Series(local.strftime("%H:%M"), index=sig.index)
    span = df.index[(df.index >= sig.index[0]) & (df.index <= sig.index[-1])] if len(sig) else df.index[:0]
    n_sess = max(len(np.unique(ny_dates(span))), 1)
    rows = []
    for name, g in [*sig.groupby(slot), ("all", sig)]:
        row = _summary(g["signed_dir"], r_long.loc[g.index], r_short.loc[g.index], cfg)
        rows.append({"symbol": symbol, "timeframe": cfg.TIMEFRAME, "primary": cfg.PRIMARY, "slot": name,
                     "per_session": row["n_events"] / n_sess, **row})  # fmt: skip
    return pd.DataFrame(rows)


def combine(run_dirs: list[str], out: str) -> pd.DataFrame:
    """Concatenate the primary_diagnostics.csv of several runs into one table."""
    frames = [pd.read_csv(Path(d) / "primary_diagnostics.csv") for d in run_dirs]
    table = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Combine per-run primary diagnostics")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="results/primary_diagnostics.csv")
    a = ap.parse_args()
    t = combine(a.run_dirs, a.out)
    cols = ["symbol", "timeframe", "primary", "n_events", "long_share", "precision", "recall", "turnover", "net_bp"]
    print(t[t["fold"].astype(str) == "all"][cols].round(4).to_string(index=False))
    print(f"[OUT]   {a.out}")
