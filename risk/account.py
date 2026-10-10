"""
Account profile (U22, SPEC §20): what a retail Alpaca account can execute. Nothing here changes a P&L — the checks
read simulated trades and equity and refuse or flag (PLAN3 U22 "the account profile silently changing a P&L" is the
failure mode this module must not have).

Rules modelled:
- Reg-T buying power. A margin account holds up to `margin_bp` (2×) of its equity overnight; intraday it may use
  `day_trade_bp` (4×) only while its equity is at least `pdt_min_equity` ($25k, the pattern-day-trader floor),
  else 2×. A cash account holds 1× and cannot short.
- The pattern-day-trader rule. A margin account under `pdt_min_equity` may make at most `max_day_trades` (3) day
  trades in any `PDT_WINDOW` (5) consecutive sessions; the fourth flags the account (the broker then restricts it).
  A day trade = a position opened and closed in the same session; a chain of rolled trades (a position held through
  rebalances) is one position.
- Settlement in a cash account. Sale proceeds settle `settlement_days` sessions later (T+1 since 2024-05-28); buying
  with unsettled proceeds is a good-faith violation, so a purchase needs settled cash.
- Locate fees. Shorts pay `locate_bps` per year on their notional (0 for easy-to-borrow ETFs); reported as an
  estimate, never deducted (the risk profile's `borrow_bps` is the deduction, SPEC §7).

`check_account(members, stream, profile)` evaluates a family (or one cell: a single member with weight 1) on the
registered account: `members` = [(symbol, weight, trades, init_cash)] from the member cells (trades.csv rows with
entry_time / exit_time bar stamps; `frac` = the committed fraction of the member's equity), `stream` = the family's
daily net returns (the account's equity path). Positions are measured as fractions of the account's equity: a
member's `frac` × its weight.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PDT_WINDOW = 5
NY = "America/New_York"


@dataclass(frozen=True)
class AccountProfile:
    name: str
    kind: Literal["margin", "cash"] = "margin"
    equity: float = 30_000.0
    pdt_min_equity: float = 25_000.0
    day_trade_bp: float = 4.0  # intraday buying power (× equity) of a margin account at or above pdt_min_equity
    margin_bp: float = 2.0  # Reg-T overnight buying power of a margin account (its intraday one under the floor)
    max_day_trades: int = 3  # per PDT_WINDOW sessions under pdt_min_equity
    settlement_days: int = 1  # cash accounts (T+1 since 2024-05-28)
    locate_bps: float = 0.0  # annual, on short notional

    def __post_init__(self):
        if self.kind not in ("margin", "cash"):
            raise ValueError(f"kind must be 'margin' or 'cash', got {self.kind!r}")
        if not self.equity > 0 or not self.pdt_min_equity > 0:
            raise ValueError("equity and pdt_min_equity must be > 0")
        if not (self.day_trade_bp >= self.margin_bp >= 1.0):
            raise ValueError("need day_trade_bp >= margin_bp >= 1")
        if self.max_day_trades < 0 or self.settlement_days < 0 or self.locate_bps < 0:
            raise ValueError("max_day_trades, settlement_days and locate_bps must be >= 0")

    def with_equity(self, equity: float) -> "AccountProfile":
        from dataclasses import replace

        return replace(self, equity=float(equity))


PROFILES: dict[str, AccountProfile] = {
    "margin_30k": AccountProfile("margin_30k"),  # PLAN3 §7: INIT_CASH $30k, above the PDT floor
    "margin_10k": AccountProfile("margin_10k", equity=10_000.0),  # the PLAN2 registration (U18)
    "cash_30k": AccountProfile("cash_30k", kind="cash"),
}


def get_account(name: str) -> AccountProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown account profile {name!r}; expected one of {sorted(PROFILES)}") from None


# ── Positions from trades ────────────────────────────────────────────────────


def _ny_day(ts) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    return idx.tz_convert(NY).tz_localize(None).normalize()


def positions(trades: pd.DataFrame) -> pd.DataFrame:
    """
    One row per position from a trades frame (risk.portfolio; rule cells add entry_time / exit_time): a chain of
    rolled trades is one position from the chain's first entry to its last exit, with the chain's maximum committed
    fraction; a side flip inside a chain ends the position and starts the next (U22 review). Columns: side, frac,
    entry (UTC), exit (UTC), entry_day, exit_day (NY dates), day_trade.
    """
    cols = {"side", "frac", "entry_time", "exit_time"}
    if trades is None or not len(trades):
        return pd.DataFrame(columns=["side", "frac", "entry", "exit", "entry_day", "exit_day", "day_trade"])
    if not cols <= set(trades.columns):
        raise ValueError(f"trades need columns {sorted(cols)} (rule cells write entry_time / exit_time)")
    t = trades.reset_index(drop=True)
    order = (
        t.sort_values("entry_time", kind="stable") if "entry_b" not in t else t.sort_values("entry_b", kind="stable")
    )
    rolled = order["rolled"].astype(str).str.lower().isin(("true", "1")).to_numpy() if "rolled" in order else \
        np.zeros(len(order), bool)  # fmt: skip
    # a trade flagged `rolled` continues INTO the next trade (risk.portfolio close_lot), so a chain ends after the
    # first unrolled trade; a side flip inside a chain closes one position and opens another
    from families.test import chain_ids

    chain = chain_ids(rolled)
    side = order["side"].to_numpy(dtype=int)
    flip = np.r_[False, (side[1:] != side[:-1]) & (chain[1:] == chain[:-1])] if len(chain) else np.zeros(0, bool)
    chain = chain + np.cumsum(flip)
    g = order.assign(chain=chain).groupby("chain")
    out = pd.DataFrame({
        "side": g["side"].first().astype(int),
        "frac": g["frac"].max().astype(float),
        "entry": pd.to_datetime(g["entry_time"].first(), utc=True),
        "exit": pd.to_datetime(g["exit_time"].last(), utc=True),
    })  # fmt: skip
    out["entry_day"] = _ny_day(out["entry"])
    out["exit_day"] = _ny_day(out["exit"])
    out["day_trade"] = out["entry_day"] == out["exit_day"]
    return out.reset_index(drop=True)


def day_trades(pos: pd.DataFrame) -> pd.Series:
    """Day trades per NY session date (positions opened and closed in the same session)."""
    if not len(pos):
        return pd.Series(dtype=int)
    return pos.loc[pos["day_trade"], "entry_day"].value_counts().sort_index().astype(int)


# ── The check ────────────────────────────────────────────────────────────────


def _equity_path(stream: pd.Series | None, equity: float, sessions: pd.DatetimeIndex) -> pd.Series:
    """Equity at the START of each session (before its returns): equity × Π(1 + r) over the earlier sessions."""
    if stream is None or not len(stream):
        return pd.Series(equity, index=sessions)
    r = pd.Series(stream, dtype=float)
    r.index = pd.DatetimeIndex(r.index)
    if r.index.tz is not None:
        r.index = r.index.tz_convert(NY).tz_localize(None)
    r.index = r.index.normalize()
    growth = (1.0 + r.sort_index()).cumprod().shift(1).fillna(1.0)
    return (equity * growth).reindex(sessions).ffill().fillna(equity)


def check_account(members: list[tuple], stream: pd.Series | None, profile: AccountProfile) -> dict:
    """
    Evaluate the account rules (module docstring) on a family's member positions. `members` = [(symbol, weight,
    trades, init_cash)]; a member's positions are `frac` × weight of the account's equity. Returns a dict with
    `tradable` (no rule broken on this account), `reasons`, the day-trade, buying-power, settlement and locate
    statistics, and `cost_otherwise`: the share of day trades the PDT rule would have blocked.
    """
    rows = []
    for sym, w, trades, _init in members:
        p = positions(trades)
        if len(p):
            p = p.assign(sym=sym, frac=p["frac"] * float(w))
            rows.append(p)
    pos = pd.concat(rows, ignore_index=True) if rows else positions(None).assign(sym=pd.Series(dtype=object))
    reasons: list[str] = []
    sessions = pd.DatetimeIndex(sorted(set(pos["entry_day"]) | set(pos["exit_day"]))) if len(pos) else \
        pd.DatetimeIndex([])  # fmt: skip
    if stream is not None and len(stream):
        s_idx = pd.DatetimeIndex(stream.index)
        s_idx = (s_idx.tz_convert(NY).tz_localize(None) if s_idx.tz is not None else s_idx).normalize()
        sessions = sessions.union(s_idx)
    equity = _equity_path(stream, profile.equity, sessions)

    # ── day trades and the PDT rule ──
    dt = day_trades(pos).reindex(sessions, fill_value=0) if len(sessions) else pd.Series(dtype=int)
    window = dt.rolling(PDT_WINDOW, min_periods=1).sum() if len(dt) else pd.Series(dtype=float)
    under = equity < profile.pdt_min_equity if profile.kind == "margin" else pd.Series(False, index=sessions)
    flagged = window[(window > profile.max_day_trades) & under.reindex(window.index).fillna(True)]
    first_flag = None if not len(flagged) else str(flagged.index[0].date())
    if first_flag is not None:
        reasons.append(f"pattern-day-trader rule: {int(flagged.iloc[0])} day trades in {PDT_WINDOW} sessions on "
                       f"{first_flag} with equity under {profile.pdt_min_equity:,.0f}")  # fmt: skip
    # what the rule would cost: the share of day trades beyond the limit on sessions under the floor (any account)
    blocked = 0.0
    if len(dt):
        under_all = equity < profile.pdt_min_equity
        excess = (window - profile.max_day_trades).clip(lower=0)
        blocked = float(np.minimum(excess, dt)[under_all.reindex(dt.index).fillna(True)].sum())
    n_dt = int(dt.sum()) if len(dt) else 0

    # ── buying power on decision-time fractions ──
    max_intra = max_over = 0.0
    first_breach = None
    limit_over = 1.0 if profile.kind == "cash" else profile.margin_bp
    if len(pos):
        ev = pd.concat(
            [
                pd.DataFrame({"t": pos["entry"], "d": pos["frac"].abs(), "day": pos["entry_day"], "k": 1}),
                pd.DataFrame({"t": pos["exit"], "d": -pos["frac"].abs(), "day": pos["exit_day"], "k": 0}),
            ]
        ).sort_values(["t", "k"], kind="stable")  # fmt: skip (at one instant exits precede entries: a roll)
        ev["gross"] = ev["d"].cumsum().round(12)
        for day, g in ev.groupby("day", sort=True):
            eq_d = float(equity.get(day, profile.equity))
            lim_intra = (profile.day_trade_bp if (profile.kind == "margin" and eq_d >= profile.pdt_min_equity)
                         else limit_over)  # fmt: skip
            gi = float(g["gross"].max())
            go = float(g["gross"].iloc[-1])  # held into the night after the day's last fill
            max_intra, max_over = max(max_intra, gi), max(max_over, go)
            if first_breach is None and (gi > lim_intra + 1e-9 or go > limit_over + 1e-9):
                first_breach = str(day.date())
                reasons.append(f"buying power: gross {gi:.2f}× intraday / {go:.2f}× overnight on {first_breach} "
                               f"(limits {lim_intra:.0f}× / {limit_over:.0f}×)")  # fmt: skip
    bp = {"max_gross_intraday": max_intra, "max_gross_overnight": max_over, "limit_overnight": limit_over,
          "limit_intraday": (profile.day_trade_bp if profile.kind == "margin" else 1.0), "first_breach": first_breach}  # fmt: skip

    # ── cash account: no shorts, settled funds only ──
    settle: dict = {}
    if profile.kind == "cash":
        short = bool((pos["side"] < 0).any()) if len(pos) else False
        settle["short_in_cash"] = short
        if short:
            reasons.append("a cash account cannot short")
        settle["violation_at"] = _settlement_violation(pos, sessions, profile.settlement_days)
        if settle["violation_at"] is not None:
            reasons.append(f"good-faith violation: a purchase with unsettled proceeds at {settle['violation_at']}")

    # ── locate fees on shorts (an estimate, never deducted) ──
    locate = {"bps": profile.locate_bps, "cost_frac_per_year": 0.0}
    if len(pos) and profile.locate_bps > 0 and (pos["side"] < 0).any():
        sh = pos[pos["side"] < 0]
        days_held = ((sh["exit"] - sh["entry"]).dt.total_seconds() / 86400.0).clip(lower=1.0)
        total = float((sh["frac"] * days_held / 365.0 * profile.locate_bps * 1e-4).sum())
        years = max(len(sessions) / 252.0, 1e-9)
        locate["cost_frac_per_year"] = total / years

    return {
        "profile": profile.name, "kind": profile.kind, "equity": profile.equity, "tradable": not reasons,
        "reasons": reasons,
        "day_trades": {"total": n_dt, "max_in_window": float(window.max()) if len(window) else 0.0,
                       "pdt_applies": bool(under.any()) if len(under) else profile.equity < profile.pdt_min_equity,
                       "first_pdt_flag": first_flag},
        "buying_power": bp, "settlement": settle, "locate": locate,
        "cost_otherwise": {"blocked_day_trades": blocked, "share_of_day_trades": (blocked / n_dt) if n_dt else 0.0},
        "n_positions": len(pos),
    }  # fmt: skip


def _settlement_violation(pos: pd.DataFrame, sessions: pd.DatetimeIndex, settlement_days: int) -> str | None:
    """First purchase that needs unsettled proceeds, as fractions of the account's equity (P&L ignored)."""
    if not len(pos):
        return None
    sess = pd.DatetimeIndex(sessions)
    pending: list[tuple[int, float]] = []  # (session position at which the proceeds settle, fraction)
    settled = 1.0
    ev = pd.concat([
        pd.DataFrame({"t": pos["entry"], "f": pos["frac"].abs(), "day": pos["entry_day"], "kind": "buy", "k": 1}),
        pd.DataFrame({"t": pos["exit"], "f": pos["frac"].abs(), "day": pos["exit_day"], "kind": "sell", "k": 0}),
    ]).sort_values(["t", "k"], kind="stable")  # fmt: skip
    for _, e in ev.iterrows():
        d = int(sess.get_indexer([e["day"]])[0])
        settled += sum(f for p, f in pending if p <= d)
        pending = [(p, f) for p, f in pending if p > d]
        if e["kind"] == "sell":
            pending.append((d + settlement_days, float(e["f"])))
        elif e["f"] > settled + 1e-9:
            return str(pd.Timestamp(e["t"]).tz_convert(NY))
        else:
            settled -= float(e["f"])
    return None


def enforce(check: dict) -> None:
    """Raise AccountError when the check found a broken rule (the forward loop refuses to trade; a report flags)."""
    if not check["tradable"]:
        raise AccountError("; ".join(check["reasons"]))


class AccountError(RuntimeError):
    """The registered account cannot execute the strategy as simulated."""
