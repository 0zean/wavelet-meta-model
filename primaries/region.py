"""
Region primary (SPEC §23, U23; PLAN3 §3 G1): an equal-weight region of intraday trend cells, decided on a fixed
schedule, flat overnight. The rule's target at each decision is the mean of its cells' positions (each in {−1, 0, +1}),
a continuous target in [−1, +1] (ALLOW_CONTINUOUS, SPEC §21) held to the next decision by the `next_event` time exit;
the portfolio simulator trades only the change.

Cells (G1's region R₀ with the defaults: 3 + 2 × 5 × 4 = 43):
    band   × VM ∈ vm_list        +1 while the gap-adjusted distance from the noise band > VM, −1 while < −VM, else 0
                                 (features.kernels.band_state); with band_stop "vwap" a long also needs close > the
                                 session VWAP and a short close < VWAP (Zarattini et al.'s current-band + VWAP stop)
    rmedv  × N × θ               v = slope · √N / σ₅, slope = the repeated-median slope of log close over N bars
    sgv    × N × θ               the same with the Savitzky–Golay degree-`sg_degree` endpoint derivative (1 = LS)
σ₅ = the σ of the within-session 5Min log returns of the `sigma_sessions` previous sessions (features.kernels.
prior_sigma). Velocity cells, vel_mode "sar" (stop-and-reverse): +1 after v > θ, −1 after v < −θ, held until the
opposite signal; "flat_inside": sign(v) while |v| > θ, else 0. vel_stop "vwap" flattens a long below the VWAP (a short
above) and the state stays flat until the next signal. Every cell starts each session flat (no state crosses a
session) and is flat while its estimator is undefined (warm-up: the first N − 1 bars of the session, the first
`band_lookback` / `sigma_sessions` sessions).

Decisions: the closes of the bars ending at the marks — `first` (default 10:00), then every `cadence` minutes on the
grid anchored at 09:30 up to 15:30 / 15:45 / 15:55 — i.e. the schedule sampler's entry times (the decision at the
09:55 bar's close fills at the 10:00 bar's open). `exit` "close": the last decision's target is held to the closing
auction (the `next_event` exit's MOC); "HH:MM": the mark at that time is a decision whose target is 0 for every cell
(flat from that bar's open) and no later mark exists. The state of a SAR cell runs over every scheduled decision of
the session (schedule_events), whichever events the rule pass later keeps.

    cells(df, X, cfg)   every cell's position at the events X.index (the specification curve's input)
    rule(df, X, cfg)    (sign, |mean|) of the cells' mean position
"""

from typing import ClassVar

import numba
import numpy as np
import pandas as pd

from features.events import parse_time
from features.vol_profile import OPEN_MINUTE
from primaries.base import primary
from primaries.mechanism import CLOSE_MINUTE, MechanismPrimary, _check, _hhmm, _num

ESTIMATORS = ("band", "rmedv", "sgv")
VEL_MODES = ("sar", "flat_inside")
CADENCES = (5, 15, 30, 60)


@numba.njit(cache=True)
def _velocity_cells(v: np.ndarray, close: np.ndarray, vwap: np.ndarray, dsess: np.ndarray, flat: np.ndarray,
                    thetas: np.ndarray, sar: bool, stop: bool, out: np.ndarray) -> None:  # fmt: skip
    """out[k, a·len(thetas) + j] = the position of the (N_a, θ_j) cell at decision k; v[a, k] the normalized velocity
    at decision k (NaN undefined); decisions are in time order, dsess their session; flat[k] forces a flat decision."""
    n_dec = v.shape[1]
    for a in range(v.shape[0]):
        for j in range(thetas.shape[0]):
            col = a * thetas.shape[0] + j
            th = thetas[j]
            state = 0.0
            for k in range(n_dec):
                if k == 0 or dsess[k] != dsess[k - 1]:
                    state = 0.0  # every session starts flat
                x = v[a, k]
                if flat[k]:
                    state = 0.0
                else:
                    sig = 0.0
                    if x > th:
                        sig = 1.0
                    elif x < -th:
                        sig = -1.0
                    if sar:
                        if sig != 0.0:
                            state = sig
                    else:
                        state = sig  # NaN compares False: an undefined velocity is flat
                    # a long needs close > VWAP, a short close < VWAP (no VWAP yet: flat), as the band's stop
                    if stop and ((state > 0.0 and not close[k] > vwap[k]) or (state < 0.0 and not close[k] < vwap[k])):
                        state = 0.0
                out[k, col] = state


@primary("region_trend")
class RegionTrend(MechanismPrimary):
    """G1's region of band and velocity cells (module docstring)."""

    TIMEFRAMES = ("5Min",)
    ALLOW_CONTINUOUS = True
    DEFAULTS: ClassVar[dict] = {
        "estimators": ["band", "rmedv", "sgv"],
        "n_list": [6, 9, 12, 18, 24],
        "theta_list": [0.75, 1.0, 1.5, 2.0],
        "vm_list": [1.0, 1.25, 1.5],
        "band_lookback": 14,
        "sigma_sessions": 5,
        "sg_degree": 1,
        "cadence": 30,
        "first": "10:00",
        "exit": "close",
        "band_stop": "vwap",
        "vel_stop": None,
        "vel_mode": "sar",
    }

    def validate(self):
        p = self.params
        est = p["estimators"]
        _check(self.name, isinstance(est, list) and len(est) >= 1 and len(set(est)) == len(est)
               and all(e in ESTIMATORS for e in est), f"estimators must be a list of distinct names in {ESTIMATORS}")  # fmt: skip
        for key, lo in (("n_list", 3), ("band_lookback", 1), ("sigma_sessions", 1)):
            vals = p[key] if key == "n_list" else [p[key]]
            ok = isinstance(vals, list) and len(vals) >= 1 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                   and v >= lo for v in vals)  # fmt: skip
            _check(
                self.name, ok and len(set(vals)) == len(vals), f"{key} must be (a list of distinct) integers >= {lo}"
            )
        for key in ("theta_list", "vm_list"):
            vals = p[key]
            ok = isinstance(vals, list) and len(vals) >= 1 and all(_num(v) and v > 0 for v in vals)
            _check(self.name, ok and len(set(vals)) == len(vals), f"{key} must be a list of distinct numbers > 0")
        _check(self.name, isinstance(p["sg_degree"], int) and 1 <= p["sg_degree"] < min(p["n_list"]),
               "sg_degree must be an integer in [1, min(n_list))")  # fmt: skip
        _check(self.name, p["cadence"] in CADENCES, f"cadence must be one of {CADENCES} minutes")
        _check(self.name, isinstance(p["first"], str) and p["first"].count(":") == 1, "first must be 'HH:MM'")
        f = parse_time(p["first"])
        _check(self.name, OPEN_MINUTE < f < CLOSE_MINUTE and (f - OPEN_MINUTE) % 5 == 0,
               "first must be a 5Min bar open after 09:30 (a 09:30 decision would be read at the previous close)")  # fmt: skip
        if p["exit"] != "close":
            _check(
                self.name, isinstance(p["exit"], str) and p["exit"].count(":") == 1, "exit must be 'close' or 'HH:MM'"
            )
            x = parse_time(p["exit"])
            _check(self.name, f < x < CLOSE_MINUTE and (x - OPEN_MINUTE) % 5 == 0,
                   "exit must be a 5Min bar open after the first decision and before 16:00")  # fmt: skip
        _check(self.name, p["band_stop"] in (None, "vwap"), "band_stop must be null or 'vwap'")
        _check(self.name, p["vel_stop"] in (None, "vwap"), "vel_stop must be null or 'vwap'")
        _check(self.name, p["vel_mode"] in VEL_MODES, f"vel_mode must be one of {VEL_MODES}")

    # ── the schedule ────────────────────────────────────────────────────────

    def marks(self) -> list[str]:
        """The decision marks (entry times): `first`, then the cadence grid anchored at 09:30, up to the exit."""
        p = self.params
        f = parse_time(p["first"])
        end = CLOSE_MINUTE if p["exit"] == "close" else parse_time(p["exit"])
        grid = [m for m in range(OPEN_MINUTE + p["cadence"], CLOSE_MINUTE, p["cadence"]) if f < m < end]
        out = [f, *grid]
        if p["exit"] != "close":
            out.append(end)
        return [_hhmm(m) for m in out]

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": self.marks()},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "next_event"}, "SIZE_STEP": 0}  # fmt: skip

    def needs_groups(self):
        return ()  # the cells read bars only (the kernels); no feature group, so no event is dropped for a NaN

    # ── cells ───────────────────────────────────────────────────────────────

    def cell_names(self) -> list[str]:
        p = self.params
        out = []
        for e in p["estimators"]:
            if e == "band":
                out += [f"band_vm{vm:g}" for vm in p["vm_list"]]
            else:
                out += [f"{e}_n{n}_t{th:g}" for n in p["n_list"] for th in p["theta_list"]]
        return out

    def _decisions(self, df: pd.DataFrame, cfg) -> np.ndarray:
        """Bar positions of every scheduled decision (the schedule sampler's event bars, time order)."""
        from features.events import schedule_events

        return df.index.get_indexer(schedule_events(df, cfg))

    def cell_matrix(self, df: pd.DataFrame, cfg) -> tuple[np.ndarray, np.ndarray]:
        """(decision bar positions, positions float64[n_decisions, n_cells]) over every scheduled decision of df."""
        from data.timeframes import get_timeframe
        from features.kernels import band_state, prior_sigma, rmedv_all, session_layout, session_vwap, sg_velocity_all
        from features.vol_profile import ny_minutes

        p = self.params
        minutes = get_timeframe(cfg.TIMEFRAME).minutes
        dec = self._decisions(df, cfg)
        sess, start, _, slot = session_layout(df.index, minutes)
        o = df["open"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        x = np.log(c)
        dsess = sess[dec]
        # exit "HH:MM": a decision whose fill opens at / after the exit mark is flat, and so is every session's last
        # decision, so a session without its exit bar (a 13:00 early close, a missing bar) is flat before its auction
        flat = np.zeros(len(dec), bool)
        if p["exit"] != "close":
            last = np.r_[dsess[1:] != dsess[:-1], True] if len(dec) else flat
            flat = (ny_minutes(df.index[np.minimum(dec + 1, len(df) - 1)]) >= parse_time(p["exit"])) | last
        vwap = session_vwap(c, df["volume"].to_numpy(dtype=float), sess)[dec]
        cols = []
        for e in p["estimators"]:
            if e == "band":
                dist, _ = band_state(o, c, sess, slot, p["band_lookback"])
                d = dist[dec]
                for vm in p["vm_list"]:
                    pos = np.where(d > vm, 1.0, np.where(d < -vm, -1.0, 0.0))  # NaN compares False: flat
                    if p["band_stop"] == "vwap":
                        wrong = ((pos > 0) & ~(c[dec] > vwap)) | ((pos < 0) & ~(c[dec] < vwap))
                        pos = np.where(wrong, 0.0, pos)
                    cols.append(np.where(flat, 0.0, pos)[:, None])
            else:
                ns = np.asarray(p["n_list"], dtype=np.int64)
                raw = rmedv_all(x, start, ns) if e == "rmedv" else sg_velocity_all(x, start, ns, p["sg_degree"])
                sig = prior_sigma(x, sess, p["sigma_sessions"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.ascontiguousarray(raw[:, dec] * np.sqrt(ns)[:, None] / sig[dec])
                out = np.empty((len(dec), len(ns) * len(p["theta_list"])))
                _velocity_cells(v, c[dec], vwap, dsess, flat, np.asarray(p["theta_list"], dtype=float),
                                p["vel_mode"] == "sar", p["vel_stop"] == "vwap", out)  # fmt: skip
                cols.append(out)
        return dec, np.hstack(cols) if cols else np.zeros((len(dec), 0))

    def cells(self, df: pd.DataFrame, X: pd.DataFrame, cfg) -> pd.DataFrame:
        """Every cell's position at the events X.index (each a scheduled decision of df)."""
        dec, m = self.cell_matrix(df, cfg)
        row = pd.Series(np.arange(len(dec)), index=df.index[dec]).reindex(X.index)
        if row.isna().any():
            raise ValueError(f"primary {self.name!r}: {int(row.isna().sum())} events are not scheduled decisions")
        return pd.DataFrame(m[row.to_numpy(dtype=int)], index=X.index, columns=self.cell_names())

    def rule(self, df, X, cfg):
        target = self.cells(df, X, cfg).to_numpy().mean(axis=1)
        return np.sign(target), np.abs(target)
