"""
Named risk profiles (SPEC §7 risk layer, U8). `RunConfig.RISK_PROFILE` selects one by name; "none" is the pre-U8
behaviour (no vol target, no caps, no loss controls, costs = SLIPPAGE_PCT only).
"""

from dataclasses import dataclass
from typing import Literal

INF = float("inf")


@dataclass(frozen=True)
class RiskProfile:
    name: str
    # Vol target: per-trade notional × min(1, vol_target / σ_hold), σ_hold = σ_bar·√VERTICAL_BARS (None = off)
    vol_target: float | None = None
    # Caps as fractions of equity (|notional| / equity); max_concurrent = open positions across symbols
    max_position: float = INF
    max_symbol: float = INF
    max_gross: float = INF
    max_net: float = INF
    max_concurrent: int | None = None
    # Held positions are trimmed back to a cap only when price drift takes them past cap · (1 + drift_tol)
    drift_tol: float = 0.10
    # Drawdown throttle: ((dd, multiplier), ...) ascending; the multiplier of the highest dd ≤ current DD applies
    dd_tiers: tuple[tuple[float, float], ...] = ()
    # Daily loss gate: session P&L ≤ −daily_loss · session-start equity → flatten at the next open, block entries
    daily_loss: float | None = None
    # Costs: "cs" adds the trailing Corwin–Schultz half-spread (floored) to SLIPPAGE_PCT on every notional change
    spread: Literal["none", "cs"] = "none"
    spread_floor: float = 0.5e-4
    spread_window_days: int = 21
    borrow_bps: float = 0.0  # annual short-borrow rate, charged on short notional at entry for the bars held

    def __post_init__(self):
        if self.vol_target is not None and not self.vol_target > 0:
            raise ValueError(f"vol_target must be > 0 or None, got {self.vol_target}")
        for f in ("max_position", "max_symbol", "max_gross", "max_net"):
            if not getattr(self, f) > 0:
                raise ValueError(f"{f} must be > 0, got {getattr(self, f)}")
        if self.max_concurrent is not None and self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1 or None, got {self.max_concurrent}")
        if self.drift_tol < 0:
            raise ValueError("drift_tol must be >= 0")
        dds = [d for d, _ in self.dd_tiers]
        if dds != sorted(dds) or any(not 0 < d < 1 for d in dds) or any(not 0 <= m <= 1 for _, m in self.dd_tiers):
            raise ValueError(f"dd_tiers must be ascending (dd in (0, 1), multiplier in [0, 1]); got {self.dd_tiers}")
        if self.daily_loss is not None and not 0 < self.daily_loss < 1:
            raise ValueError(f"daily_loss must be in (0, 1) or None, got {self.daily_loss}")
        if self.spread not in ("none", "cs"):
            raise ValueError(f"spread must be 'none' or 'cs', got {self.spread!r}")
        if self.spread_floor < 0 or self.spread_window_days < 1 or self.borrow_bps < 0:
            raise ValueError("spread_floor and borrow_bps must be >= 0, spread_window_days >= 1")

    @property
    def active(self) -> bool:
        """False only for a profile that changes nothing relative to the pre-U8 backtest."""
        return self != RiskProfile(self.name)

    @property
    def position_cap(self) -> float:
        """Cap on one position: one position per symbol, so the per-symbol cap binds too."""
        return min(self.max_position, self.max_symbol)

    def dd_multiplier(self, dd: float) -> float:
        m = 1.0
        for level, mult in self.dd_tiers:
            if dd >= level:
                m = mult
        return m


PROFILES: dict[str, RiskProfile] = {
    "none": RiskProfile("none"),
    # SPEC §7 defaults
    "standard": RiskProfile(
        "standard",
        vol_target=0.005,
        max_position=0.20,
        max_symbol=0.25,
        max_gross=1.0,
        max_net=1.0,
        max_concurrent=10,
        dd_tiers=((0.10, 0.5), (0.20, 0.0)),
        daily_loss=0.02,
        spread="cs",
    ),
}


def get_profile(name: str) -> RiskProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown risk profile {name!r}; expected one of {sorted(PROFILES)}") from None
