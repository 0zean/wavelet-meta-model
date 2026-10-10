"""
Mechanism primaries (SPEC §16, U16): one fixed rule per hypothesis family (PLAN2 F1–F10), each a thin rule over the
schedule sampler and the time / hysteresis exits (features/events.py, exits.py).

Every primary is a RulePrimary (no fit; parameters fixed per family variant) that declares its timeframes and its
default event sampler and exit model (`config_overrides`, applied by `primary_config`). Its frame is the usual
PRIMARY_COLUMNS; `magnitude` ∈ [0, 1] is the rule's size hint, which the `rule_size` sizer and the primary-only
backtest stream use as the bet size. Sides may be 0 (ALLOW_FLAT): no position for that holding period — a
long-only rule's short signal, a VIX gate, or an event where the rule is not yet defined (its warm-up: no trailing
year of returns, no σ of earlier gaps). The side of event t uses bars <= t only (and features of row t).

    vol_target          F1  1Day   daily at the open → next open (held, rolled)   +1, m = min(1, σ*/σ̂) with a band
    tsmom               F2  1Day   weekly at the open → next week's open          sign(trailing return | MODWT slope)
    overnight           F3  5Min   MOC → next MOO                                 +1 (flat when VIX > vix_max)
    calendar_drift      F4  1Day   calendar window(s) (MOC → MOC / 14:00)         +1
    intraday_momentum   F5  5Min   15:30 → MOC, gate |predictor| > k·σ_day        sign(open→now | first 30 min)
    gap_fade            F7  5Min   09:35 → 10:30 (or hysteresis), |gap| > k·σ_on  −sign(gap) (+ on news days)
    event_reaction      F8  5Min   release + 15 min → + 90 min, FOMC | CPI/NFP    sign(post-release return)
    weekly_reversal     F10 1Day   weekly at the open → next week's open          −sign(last week's return)

    primary_config("overnight", "5Min")   # RunConfig with the primary's sampler, exit and the rule_size sizer
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from features.events import parse_time
from features.vol_profile import bar_volatility, n_slots, ny_dates, ny_minutes
from primaries.base import PRIMARY_COLUMNS, REGISTRY, primary
from primaries.rules import RulePrimary
from utils.config import RunConfig

TRADING_DAYS = 252
FIRST30_END = 10 * 60
OVERNIGHT_MIN_SESSIONS = 20  # as features.session: σ_overnight needs 20 earlier gaps
CLOSE_MINUTE = 16 * 60


def _flat_frame(side: np.ndarray, magnitude: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Primary frame from sides in {-1, 0, +1} (NaN → 0, flat) and magnitudes (NaN or flat → 0), clipped to [0, 1]."""
    sd = np.nan_to_num(np.asarray(side, dtype=float), nan=0.0)
    m = np.clip(np.nan_to_num(np.asarray(magnitude, dtype=float), nan=0.0), 0.0, 1.0)
    m = np.where(sd == 0, 0.0, m)
    sd = sd.astype(int)
    return pd.DataFrame(
        {
            "clf_prob": np.nan,
            "direction": (sd > 0).astype(int),
            "signed_dir": sd,
            "magnitude": m,
            "signal": sd * m,
            "confidence": m,
        },
        index=index,
    )[list(PRIMARY_COLUMNS)]


def _sign(x) -> np.ndarray:
    """sign with NaN kept (undefined → flat later); an exact 0 goes long, as rule_frame."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), np.nan, np.where(x >= 0, 1.0, -1.0))


def session_state(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """
    The session columns the intraday rules read, per bar (bars <= t only; the formulas of features.session with no
    time-of-day profile): open_to_now, first30, gap_sigma, sigma_day, sigma_overnight, plus `prev_close` (the previous
    session's last close) and `open_d` (the session's first open).
    """
    from data.timeframes import get_timeframe

    m = get_timeframe(cfg.TIMEFRAME).minutes
    if m is None:
        raise ValueError("session_state is intraday only")
    idx = df.index
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    day = ny_dates(idx)
    mins = ny_minutes(idx)
    first = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, bool)
    last = np.r_[day[1:] != day[:-1], True] if n else np.zeros(0, bool)
    sess = np.cumsum(first) - 1
    first_pos, last_pos = np.flatnonzero(first), np.flatnonzero(last)
    o_s, c_s = o[first_pos], c[last_pos]
    open_d = o_s[sess]
    prev_c_s = np.r_[np.nan, c_s[:-1]]
    prev_c = prev_c_s[sess]
    gap_s = np.log(o_s / prev_c_s)
    sig_on_s = pd.Series(gap_s).ewm(span=cfg.VOL_SPAN, min_periods=OVERNIGHT_MIN_SESSIONS).std().shift(1).to_numpy()
    pos = np.arange(n)
    early = np.where(mins + m <= FIRST30_END, pos, -1)
    kstar = np.maximum(pd.Series(early).groupby(sess).transform("max").to_numpy(), first_pos[sess])
    ref = np.minimum(pos, kstar)
    sigma = bar_volatility(df["close"].astype(float), cfg.VOL_SPAN).to_numpy()
    sigma_day = np.r_[np.nan, sigma[last_pos][:-1]][sess] * np.sqrt(n_slots(m))
    return pd.DataFrame(
        {
            "open_to_now": np.log(c / open_d),
            "first30": np.log(c[ref] / open_d),
            "gap_sigma": gap_s[sess] / sig_on_s[sess],
            "sigma_day": sigma_day,
            "sigma_overnight": sig_on_s[sess],
            "prev_close": prev_c,
            "open_d": open_d,
        },
        index=idx,
    )


def _ann_sigma(close: pd.Series, window: int, kind: str = "rv") -> pd.Series:
    """Annualized σ of daily log returns known at each close: rolling std over `window` (rv) or EWM span (ewm)."""
    r = np.log(close.astype(float)).diff()
    s = r.rolling(window, min_periods=window).std() if kind == "rv" else r.ewm(span=window, min_periods=window).std()
    return s * np.sqrt(TRADING_DAYS)


def _banded(target: np.ndarray, band: float) -> np.ndarray:
    """Sticky target: keeps the last adopted value until the target moves by >= band from it (NaN: undefined)."""
    out = np.full(len(target), np.nan)
    cur = np.nan
    for i, x in enumerate(target):
        if np.isnan(x):
            continue
        if np.isnan(cur) or abs(x - cur) >= band - 1e-12:
            cur = x
        out[i] = cur
    return out


def _x_col(X: pd.DataFrame, col: str, name: str, group: str) -> np.ndarray:
    """The group's column `col` of X: as the feature set names it (`<group>__<col>`), or bare (a hand-built X)."""
    for c in (f"{group}__{col}", col):
        if c in X:
            return X[c].to_numpy(dtype=float)
    raise ValueError(f"primary {name!r} reads feature {col!r}: add the {group!r} group to FEATURE_GROUPS")


def _check(name: str, ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"primary {name!r}: {msg}")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _win(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 2


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


class MechanismPrimary(RulePrimary):
    """
    Base of the SPEC §16 primaries. Subclasses set TIMEFRAMES, LONG_ONLY, DEFAULTS and implement `rule(df, X, cfg)` →
    (side, magnitude) arrays for the events X.index, and `config_overrides(timeframe)` → their sampler / exit fields.
    """

    TIMEFRAMES: ClassVar[tuple[str, ...]] = ()
    LONG_ONLY: ClassVar[bool] = False
    ALLOW_FLAT: ClassVar[bool] = True
    # U22 (SPEC §21): the rule's (side, magnitude) is a target position in [−1, +1] re-evaluated at every event (a
    # region primary's target is the mean of its cells' positions); the portfolio simulator's roll path trades only
    # the change between consecutive targets. RunConfig then requires SIZER "rule_size" and SIZE_STEP 0, and the
    # time exit "next_event" holds each target to the next decision (features/exits.py).
    ALLOW_CONTINUOUS: ClassVar[bool] = False
    FEATURE_GROUPS: ClassVar[tuple[str, ...]] = ()  # groups the rule reads from X (primary_config adds them)

    def validate(self) -> None:  # each subclass checks its own parameters (no bar-window default rule here)
        pass

    @property
    def long_only(self) -> bool:
        return self.LONG_ONLY or bool(self.params.get("long_only", False))

    def config_overrides(self, timeframe: str) -> dict:
        raise NotImplementedError

    def needs_groups(self) -> tuple[str, ...]:
        return self.FEATURE_GROUPS

    def rule(self, df: pd.DataFrame, X: pd.DataFrame, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def score(self, df: pd.DataFrame, cfg: RunConfig) -> pd.Series:
        raise ValueError(f"primary {self.name!r} has no bar-level score: it cannot drive a hysteresis exit")

    def signal(self, df: pd.DataFrame, X: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
        side, mag = self.rule(df, X, cfg)
        side = np.asarray(side, dtype=float)
        if self.long_only:
            side = np.where(side < 0, 0.0, side)
        return _flat_frame(side, mag, X.index)

    def _at(self, s: pd.Series | np.ndarray, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        """Bar-level values (aligned to df) at the events X.index."""
        pos = df.index.get_indexer(X.index)
        if (pos < 0).any():
            raise ValueError(f"primary {self.name!r}: events outside the bars")
        return np.asarray(s, dtype=float)[pos]


def primary_config(name: str, timeframe: str, params: dict | None = None, **overrides) -> RunConfig:
    """
    RunConfig for a mechanism primary: RunConfig.for_timeframe(timeframe) with PRIMARY / PRIMARY_PARAMS, the
    primary's sampler and exit (config_overrides), SIZER "rule_size", and the feature groups its rule reads added;
    `overrides` (RunConfig fields) are applied last (a family variant's exit, an EVENT_PARAMS period, ...).
    """
    cls = REGISTRY.get(name)
    if cls is None or not issubclass(cls, MechanismPrimary):
        raise ValueError(f"{name!r} is not a mechanism primary; expected one of {sorted(MECHANISM)}")
    if timeframe not in cls.TIMEFRAMES:
        raise ValueError(f"primary {name!r} runs on {list(cls.TIMEFRAMES)}, not {timeframe!r}")
    p = cls(**(params or {}))
    base = RunConfig.for_timeframe(timeframe)
    groups = tuple(base.FEATURE_GROUPS) + tuple(g for g in p.needs_groups() if g not in base.FEATURE_GROUPS)
    kw = {"PRIMARY": name, "PRIMARY_PARAMS": dict(p.params), "SIZER": "rule_size", "FEATURE_GROUPS": groups}
    return base.replace(**(kw | p.config_overrides(timeframe) | overrides))


# ── F1: volatility-managed exposure ──────────────────────────────────────────


@primary("vol_target")
class VolTarget(MechanismPrimary):
    """
    Long, exposure m = min(1, σ*/σ̂) (Moreira & Muir 2017), rebalanced at the next open when the target has moved by
    >= `band` from the held exposure (otherwise the held m is kept: the position rolls through the daily schedule at
    no cost). σ̂: `rv` = std of the last `window` daily log returns, `ewm` = EWM std (span `window`), annualized; `vix`
    = VIX / 100 (the vol_state group; its band path runs over the events passed, so it restarts at each window).
    """

    TIMEFRAMES = ("1Day",)
    LONG_ONLY = True
    DEFAULTS: ClassVar[dict] = {"sigma_target": 0.15, "window": 21, "band": 0.10, "vol_source": "rv"}

    def validate(self):
        p = self.params
        _check(self.name, _num(p["sigma_target"]) and p["sigma_target"] > 0, "sigma_target must be > 0")
        _check(self.name, _win(p["window"]), "window must be an integer >= 2")
        _check(self.name, _num(p["band"]) and 0 <= p["band"] < 1, "band must be in [0, 1)")
        _check(self.name, p["vol_source"] in ("rv", "ewm", "vix"), "vol_source must be 'rv', 'ewm' or 'vix'")

    def needs_groups(self):
        return ("vol_state",) if self.params["vol_source"] == "vix" else ()

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["open"], "every": "session"},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "next"}}  # fmt: skip

    def rule(self, df, X, cfg):
        p = self.params
        if p["vol_source"] == "vix":
            sig = _x_col(X, "vix", self.name, "vol_state") / 100.0
            target = np.minimum(1.0, p["sigma_target"] / sig)
            m = _banded(target, p["band"])
        else:
            sig = _ann_sigma(df["close"], p["window"], p["vol_source"]).to_numpy()
            with np.errstate(divide="ignore"):
                target = np.minimum(1.0, p["sigma_target"] / sig)
            m = self._at(_banded(target, p["band"]), df, X)
        return np.where(np.isnan(m), np.nan, 1.0), m


# ── F2: time-series momentum ─────────────────────────────────────────────────


@primary("tsmom")
class Tsmom(MechanismPrimary):
    """
    Moskowitz, Ooi & Pedersen (2012): side = sign of the trailing `lookback`-session log return (`ret`) or of the
    slope of the causal MODWT smooth S_J of log close (`modwt`, J = modwt_j); m = min(1, gross_cap, σ*/σ̂) with σ̂ the
    annualized std of the last `vol_window` daily returns. Rebalanced weekly (the schedule's `every`). long_only:
    a short signal is flat. Plain returns (no risk-free rate).
    """

    TIMEFRAMES = ("1Day",)
    DEFAULTS: ClassVar[dict] = {"lookback": 252, "vol_window": 63, "estimator": "ret", "modwt_j": 6,
                                "sigma_target": 0.10, "long_only": False, "gross_cap": 1.0}  # fmt: skip

    def validate(self):
        p = self.params
        _check(self.name, _win(p["lookback"]) and _win(p["vol_window"]), "lookback / vol_window must be ints >= 2")
        _check(self.name, p["estimator"] in ("ret", "modwt"), "estimator must be 'ret' or 'modwt'")
        _check(self.name, isinstance(p["modwt_j"], int) and 1 <= p["modwt_j"] <= 10, "modwt_j must be in 1..10")
        _check(self.name, _num(p["sigma_target"]) and p["sigma_target"] > 0, "sigma_target must be > 0")
        _check(self.name, isinstance(p["long_only"], bool), "long_only must be a bool")
        _check(self.name, _num(p["gross_cap"]) and 0 < p["gross_cap"] <= 1, "gross_cap must be in (0, 1]")

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["open"], "every": "week"},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "next"}}  # fmt: skip

    def rule(self, df, X, cfg):
        p = self.params
        logp = np.log(df["close"].astype(float))
        if p["estimator"] == "ret":
            x = logp - logp.shift(p["lookback"])
        else:
            from features.groups import causal_modwt

            _, smooth = causal_modwt(logp, cfg.WAVELET_FILTER, p["modwt_j"])
            x = smooth.diff()
        sig = _ann_sigma(df["close"], p["vol_window"]).to_numpy()
        with np.errstate(divide="ignore"):
            m = np.minimum(min(1.0, p["gross_cap"]), p["sigma_target"] / sig)
        return self._at(_sign(x), df, X), self._at(m, df, X)


# ── F3: overnight ────────────────────────────────────────────────────────────


@primary("overnight")
class Overnight(MechanismPrimary):
    """Long from the closing auction to the next opening auction (Lou, Polk & Skouras 2019); flat when the VIX known
    at the decision exceeds `vix_max` (the vol_state group; null = always long)."""

    TIMEFRAMES = ("5Min", "1Day")
    LONG_ONLY = True
    DEFAULTS: ClassVar[dict] = {"vix_max": None}

    def validate(self):
        v = self.params["vix_max"]
        _check(self.name, v is None or (_num(v) and v > 0), "vix_max must be null or > 0")

    def needs_groups(self):
        return () if self.params["vix_max"] is None else ("vol_state",)

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["close"]},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "open"}}  # fmt: skip

    def rule(self, df, X, cfg):
        side = np.ones(len(X))
        if self.params["vix_max"] is not None:
            vix = _x_col(X, "vix", self.name, "vol_state")
            side = np.where(np.isnan(vix), np.nan, np.where(vix > self.params["vix_max"], 0.0, 1.0))
        return side, np.ones(len(X))


# ── F4: calendar and event drift ─────────────────────────────────────────────

# window → (EVENT_PARAMS, EXIT_PARAMS on 1Day, EXIT_PARAMS intraday). Every window enters at the closing auction of
# the session `day_offset` sessions before its calendar day and exits `exit_session` sessions later.
CALENDAR_WINDOWS = {
    # Lucca & Moench (2015): t−1 close → t 14:00 (intraday) / t close (1Day, the variant)
    "fomc_pre": ({"days": "fomc", "day_offset": 1}, {"exit_time": "close", "exit_session": 1},
                 {"exit_time": "14:00", "exit_session": 1}),
    "fomc_day": ({"days": "fomc", "day_offset": 1}, {"exit_time": "close", "exit_session": 1},
                 {"exit_time": "close", "exit_session": 1}),
    # turn of month (Etula et al. 2020): the month's last session and the next three (−1 / +3), or −2 / +2
    "tom": ({"days": "month_end", "day_offset": 1}, {"exit_time": "close", "exit_session": 4},
            {"exit_time": "close", "exit_session": 4}),
    "tom_2_2": ({"days": "month_end", "day_offset": 2}, {"exit_time": "close", "exit_session": 4},
                {"exit_time": "close", "exit_session": 4}),
    # month-end duration extension (TLT): the month's last session
    "month_end": ({"days": "month_end", "day_offset": 1}, {"exit_time": "close", "exit_session": 1},
                  {"exit_time": "close", "exit_session": 1}),
    # options-expiration week: the five sessions to the monthly expiry (four in a holiday week, which then starts
    # on the previous week's last session)
    "opex_week": ({"days": "opex", "day_offset": 5}, {"exit_time": "close", "exit_session": 5},
                  {"exit_time": "close", "exit_session": 5}),
    # CPI / NFP release day: previous close → release-day close
    "cpi_nfp": ({"days": "cpi_nfp", "day_offset": 1}, {"exit_time": "close", "exit_session": 1},
                {"exit_time": "close", "exit_session": 1}),
}  # fmt: skip


@primary("calendar_drift")
class CalendarDrift(MechanismPrimary):
    """
    Long over one calendar window (CALENDAR_WINDOWS), or over the union of several (`window` a list, U18): one
    position long on every session some window holds, entered at the closing auction before the first held session
    and rolled daily (the schedule's `windows`, a time exit at the next close), so overlapping windows never stack.
    A union needs every window to end at a close on the cell's timeframe (fomc_pre ends 14:00 on 5Min).
    """

    TIMEFRAMES = ("1Day", "5Min")
    LONG_ONLY = True
    DEFAULTS: ClassVar[dict] = {"window": "fomc_pre"}

    def validate(self):
        w = self.params["window"]
        ws = w if isinstance(w, list) else [w]
        ok = all(isinstance(x, str) and x in CALENDAR_WINDOWS for x in ws)
        _check(self.name, ok, f"window must be one of {list(CALENDAR_WINDOWS)} or a list of them")
        if isinstance(w, list):
            _check(self.name, len(w) >= 2 and len(set(w)) == len(w), "a window list holds two or more distinct windows")

    def config_overrides(self, timeframe):
        w = self.params["window"]
        if not isinstance(w, list):
            ev, daily, intra = CALENDAR_WINDOWS[w]
            return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["close"], **ev},
                    "EXIT_MODEL": "time", "EXIT_PARAMS": dict(daily if timeframe == "1Day" else intra)}  # fmt: skip
        windows = []
        for name in w:
            ev, daily, intra = CALENDAR_WINDOWS[name]
            ex = daily if timeframe == "1Day" else intra
            _check(self.name, ex["exit_time"] == "close", f"window {name!r} ends at {ex['exit_time']} on {timeframe}: "
                   "a union holds whole sessions (close to close)")  # fmt: skip
            win = {"days": ev["days"], "day_offset": ev["day_offset"], "hold": ex["exit_session"]}
            _check(self.name, win not in windows, f"window {name!r} holds the same sessions as another in the list")
            windows.append(win)
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["close"], "windows": windows},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "close", "exit_session": 1}}  # fmt: skip

    def rule(self, df, X, cfg):
        return np.ones(len(X)), np.ones(len(X))


# ── F5: market intraday momentum ─────────────────────────────────────────────


@primary("intraday_momentum")
class IntradayMomentum(MechanismPrimary):
    """
    Gao, Han, Li & Zhou (2018); Baltussen et al. (2021): enter at `entry` (decided at the previous bar's close) in the
    direction of the `predictor` (open_to_now = the session open → decision-bar close return; first30 = the first 30
    minutes' return), exit at the closing auction; trade only when |predictor| > threshold_sigma · σ_day (the
    schedule gate, with the run's time-of-day profile). m = min(1, |predictor| / σ_day) (plain σ_day).
    """

    TIMEFRAMES = ("5Min",)
    DEFAULTS: ClassVar[dict] = {"entry": "15:30", "predictor": "open_to_now", "threshold_sigma": 0.5}

    def validate(self):
        p = self.params
        _check(self.name, p["predictor"] in ("open_to_now", "first30"), "predictor must be 'open_to_now' or 'first30'")
        _check(self.name, _num(p["threshold_sigma"]) and p["threshold_sigma"] >= 0, "threshold_sigma must be >= 0")
        _check(self.name, isinstance(p["entry"], str), "entry must be 'HH:MM'")
        m = parse_time(p["entry"])
        _check(self.name, 10 * 60 < m < CLOSE_MINUTE, "entry must be after 10:00 and before 16:00")

    def config_overrides(self, timeframe):
        p = self.params
        gate = None if p["threshold_sigma"] == 0 else f"abs({p['predictor']}) > {p['threshold_sigma']} * sigma_day"
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": [p["entry"]], "gate": gate},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "close"}}  # fmt: skip

    def rule(self, df, X, cfg):
        st = session_state(df, cfg)
        pred = self._at(st[self.params["predictor"]], df, X)
        sd = self._at(st["sigma_day"], df, X)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.minimum(1.0, np.abs(pred) / sd)
        return _sign(pred), m


# ── F7: overnight gap fade / follow ──────────────────────────────────────────


@primary("gap_fade")
class GapFade(MechanismPrimary):
    """
    Fade the overnight gap in the first hour on days without scheduled macro news (follow_on_news: follow it on
    FOMC / CPI / NFP days): enter at the 09:35 bar's open (decided at the 09:30 bar's close), when |gap| >
    min_gap_sigma · σ_overnight (σ of earlier gaps), side = −sign(gap) (+sign when following), m = min(1, |gap_σ| / 2);
    exit at `exit` ("HH:MM", or "hysteresis": the bar-level score below crosses 0, i.e. the gap is filled, else 10:30).
    Score (hysteresis): −log(close / previous close) / σ_overnight (sign flipped when following).
    """

    TIMEFRAMES = ("5Min",)
    DEFAULTS: ClassVar[dict] = {"min_gap_sigma": 1.0, "exit": "10:30", "follow_on_news": False}
    ENTRY = "09:35"

    def validate(self):
        p = self.params
        _check(self.name, _num(p["min_gap_sigma"]) and p["min_gap_sigma"] >= 0, "min_gap_sigma must be >= 0")
        _check(self.name, isinstance(p["follow_on_news"], bool), "follow_on_news must be a bool")
        if p["exit"] != "hysteresis":
            _check(self.name, isinstance(p["exit"], str), "exit must be 'HH:MM' or 'hysteresis'")
            _check(self.name, parse_time(self.ENTRY) < parse_time(p["exit"]) < CLOSE_MINUTE, "exit must be after 09:35")

    def _mult(self) -> float:
        return 1.0 if self.params["follow_on_news"] else -1.0

    def config_overrides(self, timeframe):
        p = self.params
        ev = {"entry_times": [self.ENTRY], "days": "macro" if p["follow_on_news"] else "non_macro",
              "gate": f"abs(gap_sigma) > {p['min_gap_sigma']}"}  # fmt: skip
        if p["exit"] == "hysteresis":
            from data.timeframes import get_timeframe

            # the 10:25 bar is the last held: the max-hold exit is its close (10:30), as the time exit's 10:30 open
            bars = (parse_time("10:30") - parse_time(self.ENTRY)) // get_timeframe(timeframe).minutes
            exit_ = {"EXIT_MODEL": "hysteresis", "EXIT_PARAMS": {"beta": 0.0, "max_bars": bars}}
        else:
            exit_ = {"EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": p["exit"]}}
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": ev, **exit_}

    def rule(self, df, X, cfg):
        g = self._at(session_state(df, cfg)["gap_sigma"], df, X)
        return self._mult() * _sign(g), np.minimum(1.0, np.abs(g) / 2.0)

    def score(self, df, cfg):
        st = session_state(df, cfg)
        x = self._mult() * np.log(df["close"] / st["prev_close"]) / st["sigma_overnight"]
        return pd.Series(x.to_numpy(dtype=float), index=df.index)


# ── F8: macro-announcement reaction ──────────────────────────────────────────

RELEASE_TIME = {"fomc": "14:00", "cpi_nfp": "09:30"}  # CPI / NFP (08:30) are pre-market: the RTH open


@primary("event_reaction")
class EventReaction(MechanismPrimary):
    """
    Continuation after a scheduled release: on FOMC days observe 14:00 → 14:00 + observe_min (CPI / NFP: the RTH open
    09:30 → 09:30 + observe_min), enter then in the direction of that return (from the release bar's open to the
    decision bar's close), exit hold_min later. m = min(1, |reaction| / σ_day) (plain σ_day). A session without its
    release bar is flat.
    """

    TIMEFRAMES = ("5Min",)
    DEFAULTS: ClassVar[dict] = {"release": "fomc", "observe_min": 15, "hold_min": 90}

    def validate(self):
        p = self.params
        _check(self.name, p["release"] in RELEASE_TIME, f"release must be one of {list(RELEASE_TIME)}")
        for k in ("observe_min", "hold_min"):
            _check(self.name, isinstance(p[k], int) and not isinstance(p[k], bool) and p[k] > 0, f"{k} must be > 0")
        _check(self.name, self._exit_minute() < CLOSE_MINUTE, "release + observe_min + hold_min must be before 16:00")

    def _entry_minute(self) -> int:
        return parse_time(RELEASE_TIME[self.params["release"]]) + self.params["observe_min"]

    def _exit_minute(self) -> int:
        return self._entry_minute() + self.params["hold_min"]

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule",
                "EVENT_PARAMS": {"entry_times": [_hhmm(self._entry_minute())], "days": self.params["release"]},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": _hhmm(self._exit_minute())}}  # fmt: skip

    def rule(self, df, X, cfg):
        T = parse_time(RELEASE_TIME[self.params["release"]])
        day = ny_dates(df.index)
        mins = ny_minutes(df.index)
        rel = pd.Series(np.where(mins == T, df["open"].to_numpy(dtype=float), np.nan)).groupby(day).transform("max")
        rel = rel.to_numpy()  # the release bar's open in each session (NaN without one)
        pos = df.index.get_indexer(X.index)
        ok = mins[pos] >= T  # the decision bar is at or after the release bar
        react = np.where(ok, np.log(df["close"].to_numpy(dtype=float)[pos] / rel[pos]), np.nan)
        sd = self._at(session_state(df, cfg)["sigma_day"], df, X)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.minimum(1.0, np.abs(react) / sd)
        return _sign(react), m


# ── F10: weekly reversal ─────────────────────────────────────────────────────


@primary("weekly_reversal")
class WeeklyReversal(MechanismPrimary):
    """Short-term reversal: side = −sign(the last `lookback` sessions' log return), held one week (weekly schedule).
    The cross-sectional rank within the sector-ETF set (SPEC §16) needs a basket cell (U17); m = 1 here."""

    TIMEFRAMES = ("1Day",)
    DEFAULTS: ClassVar[dict] = {"lookback": 5}

    def validate(self):
        _check(self.name, isinstance(self.params["lookback"], int) and self.params["lookback"] >= 1, "lookback >= 1")

    def config_overrides(self, timeframe):
        return {"EVENT_SAMPLER": "schedule", "EVENT_PARAMS": {"entry_times": ["open"], "every": "week"},
                "EXIT_MODEL": "time", "EXIT_PARAMS": {"exit_time": "next"}}  # fmt: skip

    def rule(self, df, X, cfg):
        logp = np.log(df["close"].astype(float))
        x = (logp - logp.shift(self.params["lookback"])).to_numpy()
        return -self._at(_sign(x), df, X), np.ones(len(X))


MECHANISM = tuple(n for n, c in REGISTRY.items() if isinstance(c, type) and issubclass(c, MechanismPrimary))
