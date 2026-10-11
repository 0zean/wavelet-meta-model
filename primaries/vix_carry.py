"""
VIX carry region primary (SPEC §27, U28; PLAN3 §3 G6): long SVXY (short VIX short-term futures) while the futures
curve is in contango and the VIX is below a cap, flat otherwise; an equal-weight region of (threshold, cap) cells.

Cells (G6's headline region with the defaults: 5 thresholds × 3 caps = 15, centred on PLAN2 F9's point 1.05 / 25):
    signal × θ × cap       +1 while `signal` > θ and VIX < cap (cap null: no cap), else 0
`signal` is the front future over the VIX (vx1_vix: VX1 / VIX, contango > 1) or the second future over the front
(vx2_vx1). Hard rule (PLAN2 U21, every cell): flat while VX1 / VIX < `backwardation` (null: off). The target is the
cells' mean, a continuous long position in [0, 1] (ALLOW_CONTINUOUS, SPEC §21), × `scale_before.scale` on sessions
before `scale_before.date`: SVXY tracked −1× the VIX short-term futures index until 2018-02-27 and −0.5× from
2018-02-28 (data/fetch.py REGIME_BREAKS), so a 0.5 scale holds the forward −0.5× futures exposure throughout.

Decision (one per session): the close of the bar before `decision` (default 10:00, the 09:55 bar), filled at the
`decision` bar's open and held to the same mark of the next session (time exit "HH:MM", exit_session 1), where the
next decision's target replaces it — the portfolio simulator's roll trades only the change (an unchanged target holds
its shares, untraded). The state is the vix_term group (features/state.py) as of the decision bar: the CBOE closes and
CFE settlements of the previous session (available 16:20 ET). A session without its decision bar (SVXY 2016-01 →
2025-09: 2018-02-06, no bar before 10:00; 2024-03-26, no 10:00 bar) has no decision: the held position exits at the
first bar at or after the mark and the session stays flat. A NaN state drops the event (wfo.rule_pass), with the same effect.

The specification curve runs every cell as its own engine cell (`cell_params`; families/run.py `engine_cells`, U28),
not by a replica: the positions are held overnight, which wfo.position_backtest does not model.

    rule(df, X, cfg)     (sign, |target|): the cells' mean × the pre-break scale
    cell_params()        [(name, params)] of the single-cell primaries (θ_list = [θ], vix_list = [cap])
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from features.events import parse_time
from features.vol_profile import OPEN_MINUTE
from primaries.base import primary
from primaries.mechanism import CLOSE_MINUTE, MechanismPrimary, _check, _num, _x_col

SIGNALS = ("vx1_vix", "vx2_vx1")
GROUP = "vix_term"


def _cap_name(cap) -> str:
    return "vixinf" if cap is None else f"vix{cap:g}"


@primary("vix_carry")
class VixCarry(MechanismPrimary):
    """G6's region of VIX-carry cells (module docstring)."""

    TIMEFRAMES = ("5Min",)
    LONG_ONLY = True
    ALLOW_CONTINUOUS = True
    FEATURE_GROUPS = (GROUP,)
    DEFAULTS: ClassVar[dict] = {
        "signal": "vx1_vix",
        "theta_list": [1.0, 1.025, 1.05, 1.075, 1.10],
        "vix_list": [20, 25, 30],
        "backwardation": 1.0,
        "decision": "10:00",
        "scale_before": {"date": "2018-02-28", "scale": 0.5},
    }

    def validate(self):
        p = self.params
        _check(self.name, p["signal"] in SIGNALS, f"signal must be one of {SIGNALS}")
        th = p["theta_list"]
        _check(self.name, isinstance(th, list) and len(th) >= 1 and all(_num(v) and v > 0 for v in th)
               and len(set(th)) == len(th), "theta_list must be a list of distinct numbers > 0")  # fmt: skip
        caps = p["vix_list"]
        _check(self.name, isinstance(caps, list) and len(caps) >= 1 and all(v is None or (_num(v) and v > 0)
                                                                          for v in caps)
               and len(set(caps)) == len(caps), "vix_list must be a list of distinct numbers > 0 or null (no cap)")  # fmt: skip
        b = p["backwardation"]
        _check(self.name, b is None or (_num(b) and b > 0), "backwardation must be null or a number > 0")
        d = p["decision"]
        _check(self.name, isinstance(d, str) and d.count(":") == 1, "decision must be 'HH:MM'")
        m = parse_time(d)
        _check(self.name, OPEN_MINUTE < m < CLOSE_MINUTE and (m - OPEN_MINUTE) % 5 == 0,
               "decision must be a 5Min bar open after 09:30 (a 09:30 decision is read at the previous close, before "
               "that session's settlements)")  # fmt: skip
        s = p["scale_before"]
        if s is not None:
            _check(self.name, isinstance(s, dict) and set(s) == {"date", "scale"},
                   "scale_before must be null or {date, scale}")  # fmt: skip
            try:
                pd.Timestamp(s["date"])
            except (TypeError, ValueError):
                _check(self.name, False, f"scale_before.date {s['date']!r} is not a date")
            _check(self.name, _num(s["scale"]) and 0 < s["scale"] <= 1, "scale_before.scale must be in (0, 1]")

    def config_overrides(self, timeframe):
        d = self.params["decision"]
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": [d], "every": "session"},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": d, "exit_session": 1}, "SIZE_STEP": 0}  # fmt: skip

    # ── cells ───────────────────────────────────────────────────────────────

    def cell_names(self) -> list[str]:
        p = self.params
        return [f"t{th:g}_{_cap_name(cap)}" for th in p["theta_list"] for cap in p["vix_list"]]

    def cell_params(self) -> list[tuple[str, dict]]:
        """The single-cell primaries of the region, in cell_names order (the specification curve's engine cells)."""
        p = self.params
        return [(f"t{th:g}_{_cap_name(cap)}", {**p, "theta_list": [th], "vix_list": [cap]})
                for th in p["theta_list"] for cap in p["vix_list"]]  # fmt: skip

    def cells(self, df: pd.DataFrame, X: pd.DataFrame, cfg) -> pd.DataFrame:
        """Every cell's position (0 / 1) at the events X.index; flat wherever a state it reads is NaN (a cell without
        a cap included)."""
        p = self.params
        sig = _x_col(X, p["signal"], self.name, GROUP)
        vix = _x_col(X, "vix", self.name, GROUP)
        vx1 = _x_col(X, "vx1_vix", self.name, GROUP)
        ok = np.isfinite(sig) & np.isfinite(vix) & np.isfinite(vx1)
        if p["backwardation"] is not None:
            ok &= vx1 >= p["backwardation"]
        cols = {}
        for th in p["theta_list"]:
            for cap in p["vix_list"]:
                on = ok & (sig > th) & (True if cap is None else vix < cap)
                cols[f"t{th:g}_{_cap_name(cap)}"] = np.where(on, 1.0, 0.0)
        return pd.DataFrame(cols, index=X.index)

    def scale(self, X: pd.DataFrame) -> np.ndarray:
        """Per event: `scale_before.scale` on sessions dated before `scale_before.date`, else 1."""
        s = self.params["scale_before"]
        if s is None:
            return np.ones(len(X))
        idx = pd.DatetimeIndex(X.index)
        day = (idx.tz_convert("America/New_York") if idx.tz is not None else idx).normalize().tz_localize(None)
        return np.where(day < pd.Timestamp(s["date"]), float(s["scale"]), 1.0)

    def rule(self, df, X, cfg):
        target = self.cells(df, X, cfg).to_numpy().mean(axis=1) * self.scale(X)
        return np.where(target > 0, 1.0, 0.0), target
