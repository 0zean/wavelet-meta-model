"""
Power Walk-Forward Optimisation (SPEC §6, U9): choose the in-sample length and retraining cadence by running a
full walk-forward for every (IS, OOS) combo, then walk forward the *choice itself* (nested selection).

    windows    = pwfo_windows(df.index, Combo(252, 10, 84), embargo=1, sessions=calendar_dates)
    run        = run_combo(df, cfg, prep, combo, sessions=...)          # one combo's WFO
    stats      = combo_stats(df, run, cfg)                               # WFE, OOS Sharpe, per-window table
    choice     = nested_select(daily_returns_by_combo, is_sharpe, cfg)   # no selection look-ahead
    result     = run_pwfo(df, cfg, sessions=...)                         # all of the above + PBO + DSR

Window w of combo (IS, OOS), in units (trading sessions, or bars for the legacy regression):
    IS  = [a_w, b_w)  with b_w = IS + w·OOS, a_w = w·OOS (rolling) or 0 (expanding)
          train = [a_w, b_w − val), val = [b_w − val, b_w)
    OOS = [b_w, b_w + OOS)   — retraining cadence = OOS; the OOS windows tile [IS, …) contiguously
Only full OOS windows are run, unless `partial_last` (PWFO_PARTIAL_LAST): then a last, shorter OOS window runs
to the end of the data, as the last retrain would trade live (U11 Stage E). The embargo is the WFO's (SPEC §5): fitting samples whose exit falls in the last
`embargo` units before the end of their split (train → val, val → OOS) are dropped; no unit is skipped.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.config import RunConfig
from validation.pbo import pbo
from validation.stats import dsr, psr, return_moments
from wfo.backtest import run_backtest
from wfo.wfo_engine import NoFitError, Prepared, WindowFit, fit_window, prepare

TRADING_DAYS = 252


class Combo(NamedTuple):
    is_len: int  # IS units (train + val)
    oos_len: int  # OOS units = retraining cadence
    val_len: int  # final val units inside IS

    @property
    def label(self) -> str:
        return f"IS{self.is_len}_OOS{self.oos_len}"


class Window(NamedTuple):
    w: int
    is_start: int  # bar positions: IS = [is_start, val_end), OOS = [val_end, oos_end)
    train_end: int
    val_end: int
    oos_end: int
    train_embargo: int  # bars before train_end in which no train exit may fall
    val_embargo: int  # bars before val_end in which no val exit may fall


def unit_bounds(index: pd.DatetimeIndex, unit: str = "days", sessions=None) -> np.ndarray:
    """
    Bar position of the first bar of each unit, then len(index). unit="bars": every bar is a unit. unit="days":
    `sessions` = exchange-calendar session dates (None → the sessions present in the data); calendar sessions
    between the first and last data session count even when the data layer dropped them (they hold no bars, so
    their boundary equals the next session's). Half-days are ordinary sessions with fewer bars.
    """
    n = len(index)
    if unit == "bars":
        return np.arange(n + 1)
    if unit != "days":
        raise ValueError(f"unit must be 'days' or 'bars', got {unit!r}")
    day = index.normalize()
    if day.tz is not None:
        day = day.tz_localize(None)
    data_days = day[np.r_[True, day[1:] != day[:-1]]]
    if sessions is None:
        cal = data_days
    else:
        cal = pd.DatetimeIndex(sessions).normalize()
        cal = cal.tz_localize(None) if cal.tz is not None else cal
        cal = cal[(cal >= data_days[0]) & (cal <= data_days[-1])].unique().sort_values()
        missing = data_days.difference(cal)
        if len(missing):
            raise ValueError(f"{len(missing)} data sessions are not exchange sessions, e.g. {missing[0].date()}")
    return np.r_[np.searchsorted(day.to_numpy(), cal.to_numpy(), side="left"), n]


def pwfo_windows(
    index: pd.DatetimeIndex,
    combo: Combo,
    *,
    embargo: int,
    expanding: bool = False,
    unit: str = "days",
    sessions=None,
    partial_last: bool = False,
) -> list[Window]:
    """Walk-forward windows of one combo as bar positions (module docstring). Raises on an infeasible split."""
    is_len, oos_len, val_len = combo
    if not 0 < val_len < is_len or oos_len < 1:
        raise ValueError(f"need 0 < val < IS and OOS >= 1; got {combo}")
    if not 0 <= embargo < min(is_len - val_len, val_len):
        raise ValueError(
            f"embargo {embargo} must be >= 0 and shorter than train ({is_len - val_len}) and val ({val_len})"
        )
    bounds = unit_bounds(index, unit, sessions)
    n_units = len(bounds) - 1
    out = []
    w = 0
    while (b := is_len + w * oos_len) < n_units:
        if b + oos_len > n_units and not partial_last:
            break
        a = 0 if expanding else w * oos_len
        tr = b - val_len
        out.append(
            Window(
                w,
                int(bounds[a]),
                int(bounds[tr]),
                int(bounds[b]),
                int(bounds[min(b + oos_len, n_units)]),
                int(bounds[tr] - bounds[tr - embargo]),
                int(bounds[b] - bounds[b - embargo]),
            )
        )
        w += 1
    return out


def make_grid(cfg: RunConfig) -> list[Combo]:
    """IS × OOS grid with val = round(PWFO_VAL_FRAC · IS) (≥ 1, < IS)."""
    frac = cfg.PWFO_VAL_FRAC if cfg.PWFO_VAL_FRAC is not None else cfg.VAL / (cfg.INITIAL_TRAIN + cfg.VAL)
    return [Combo(i, o, min(max(1, round(frac * i)), i - 1)) for i in cfg.PWFO_IS_GRID for o in cfg.PWFO_OOS_GRID]


class ComboRun(NamedTuple):
    combo: Combo
    signals: pd.DataFrame  # OOS predictions of every run window, in time order (run_wfo format, fold = w + 1)
    in_sample: dict[int, pd.DataFrame]  # w → the window's models' predictions on their own fitting events
    windows: pd.DataFrame  # one row per window: bounds (timestamps), status, event counts


def run_combo(
    df: pd.DataFrame,
    cfg: RunConfig,
    prep: Prepared,
    combo: Combo,
    *,
    unit: str = "days",
    sessions=None,
) -> ComboRun:
    """Every window of one combo through wfo_engine.fit_window (rolling unless cfg.PWFO_EXPANDING)."""
    wins = pwfo_windows(
        df.index, combo, embargo=cfg.EMBARGO, expanding=cfg.PWFO_EXPANDING, unit=unit, sessions=sessions,
        partial_last=cfg.PWFO_PARTIAL_LAST,
    )  # fmt: skip
    if not wins:
        raise ValueError(f"{combo.label}: the data holds no full window")
    frames, ins, rows = [], {}, []
    ix = df.index
    for win in wins:
        print(
            f"\n[PWFO]  {combo.label} window {win.w}: IS[{win.is_start}:{win.val_end}]  OOS[{win.val_end}:{win.oos_end}]"
        )
        empty = win.val_end == win.oos_end  # the OOS sessions hold no bars (all dropped by the data layer)
        res = (
            WindowFit("empty_oos", None, None, False, (), 0)
            if empty
            else fit_window(
                df, cfg, prep, win.w + 1, win.is_start, win.train_end, win.val_end, win.oos_end,
                win.train_embargo, win.val_embargo, in_sample=True,
            )
        )  # fmt: skip
        rows.append(
            {
                "w": win.w,
                "is_start": ix[win.is_start],
                "train_end": ix[win.train_end],
                "oos_start": pd.NaT if empty else ix[win.val_end],
                "oos_last": pd.NaT if empty else ix[win.oos_end - 1],
                "status": res.status,
                "n_fit_events": res.n_fit_events,
                "n_oos_events": 0 if res.oos is None else len(res.oos),
                "n_oos_trades": 0 if res.oos is None else int((res.oos["trade_signal"] != 0).sum()),
                "meta_skipped": res.meta_skipped,
            }
        )
        if res.status == "ok":
            frames.append(res.oos)
            ins[win.w] = res.in_sample
    signals = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    table = pd.DataFrame(rows)
    print(f"[PWFO]  {combo.label}: {len(wins)} windows, status " + table["status"].value_counts().to_dict().__repr__())
    return ComboRun(combo, signals, ins, table)


def daily_returns(equity: pd.Series) -> pd.Series:
    """Close-to-close session returns of a bar-level equity curve (the first relative to its starting equity)."""
    last = equity.groupby(equity.index.normalize()).last()
    prev = last.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return last / prev - 1.0


def _sharpe(r) -> float:
    """Annualized Sharpe of daily returns; 0 for a flat series (no trades is neither skill nor loss)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def _ann(total: float, n_days: int) -> float:
    return float((1.0 + total) ** (TRADING_DAYS / n_days) - 1.0) if n_days else np.nan


def wfe(oos: float, is_values, min_t: float) -> tuple[float, float]:
    """
    Walk-forward efficiency OOS / mean(IS) and the t-statistic of that mean over the windows. NaN unless the mean IS
    figure is > 0 *and* its t-statistic ≥ `min_t`: a ratio to an IS loss is meaningless, and a ratio to an IS figure
    indistinguishable from 0 explodes (e.g. OOS −0.02 / IS 0.003 = −6.5).
    """
    v = np.asarray(is_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan
    m = v.mean()
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    t = m / sd * np.sqrt(v.size) if sd > 0 else (np.inf if m > 0 else (-np.inf if m < 0 else 0.0))
    return (float(oos / m) if m > 0 and t >= min_t else np.nan), float(t)


def _equity(df: pd.DataFrame, signals: pd.DataFrame, cfg: RunConfig, spread_bars) -> pd.Series:
    """Meta-filtered, sized equity of `signals` on `df` (run_backtest: cfg.POSITION_MODE, cfg.RISK_PROFILE)."""
    if signals.empty:
        return pd.Series(float(cfg.INIT_CASH), index=df.index, name="equity")
    return run_backtest(df, signals, cfg, spread_bars=spread_bars)["Meta-filtered"][0]


class ComboStats(NamedTuple):
    summary: pd.Series  # per-combo statistics (SPEC §6)
    windows: pd.DataFrame  # run.windows + per-window IS / OOS performance
    oos_daily: pd.Series  # stitched OOS daily returns over [first OOS session, last OOS session]


def combo_stats(df: pd.DataFrame, run: ComboRun, cfg: RunConfig, spread_bars=None) -> ComboStats:
    """
    OOS: the combo's stitched OOS signals are backtested once over [first OOS bar, end of data) (trades carry across
    window boundaries as they would live) and cut to daily returns on the OOS sessions. Window w's OOS return is the
    compounded daily return over its sessions (0 for a window that could not be fit: no trades).
    IS: each run window's in-sample predictions are backtested on its own IS bars; return and Sharpe annualized.

    WFE = annualized stitched OOS return / mean annualized IS return over all windows (a window that could not be
    fit is flat on both sides), NaN unless that mean is > 0 with t-statistic ≥ PWFO_WFE_MIN_T (`wfe`; `n_is_nonpos`
    counts the fitted windows with IS return ≤ 0); WFE_sharpe likewise with Sharpe ratios. `weak` = fewer than
    PWFO_MIN_WINDOWS fitted windows. `pct_profitable_oos` counts fitted windows without an OOS trade as not profitable;
    `pct_profitable_traded` is the share among the windows that traded.
    """
    combo, signals, ins, win = run
    win = win.copy()
    first, last = win["oos_start"].dropna().iloc[0], win["oos_last"].dropna().iloc[-1]
    start = df.index.get_loc(first)
    eq = _equity(df.iloc[start:], signals, cfg, spread_bars)
    oos_daily = daily_returns(eq)
    oos_daily = oos_daily[oos_daily.index <= last.normalize()]
    days = oos_daily.index
    w_ret, w_days = [], []
    for s, e in zip(win["oos_start"], win["oos_last"], strict=True):
        if pd.isna(s):  # empty OOS window
            w_ret.append(0.0)
            w_days.append(0)
            continue
        r = oos_daily[(days >= s.normalize()) & (days <= e.normalize())]
        w_ret.append(float((1 + r).prod() - 1))
        w_days.append(len(r))
    win["oos_ret"], win["oos_days"] = w_ret, w_days
    win["oos_ret_ann"] = [_ann(r, n) for r, n in zip(w_ret, w_days, strict=True)]

    is_ret, is_sr = np.full(len(win), np.nan), np.full(len(win), np.nan)
    for i, (w, s, e) in enumerate(zip(win["w"], win["is_start"], win["oos_start"], strict=True)):
        if w not in ins:
            continue
        a, b = df.index.get_loc(s), df.index.get_loc(e)
        r = daily_returns(_equity(df.iloc[a:b], ins[w], cfg, spread_bars))
        is_ret[i], is_sr[i] = _ann(float((1 + r).prod() - 1), len(r)), _sharpe(r)
    win["is_ret_ann"], win["is_sharpe"] = is_ret, is_sr

    ok = win["status"] == "ok"
    n_ok = int(ok.sum())
    oos_ann = _ann(float((1 + oos_daily).prod() - 1), len(oos_daily))
    oos_sr = _sharpe(oos_daily)
    # IS over every window, a window that could not be fit counting flat (0) as its OOS does: both sides of the WFE
    # cover the same windows
    is_all, is_sr_all = np.where(ok, is_ret, 0.0), np.where(ok, is_sr, 0.0)
    wfe_ret, is_t = wfe(oos_ann, is_all, cfg.PWFO_WFE_MIN_T)
    wfe_sr, is_sr_t = wfe(oos_sr, is_sr_all, cfg.PWFO_WFE_MIN_T)
    traded = ok & (win["n_oos_trades"] > 0)
    curve = (1 + oos_daily).cumprod()
    downside = np.sqrt((np.minimum(oos_daily, 0) ** 2).mean())
    rho = (
        spearmanr(win.loc[ok, "is_ret_ann"], win.loc[ok, "oos_ret"]).statistic
        if n_ok >= 3 and win.loc[ok, "oos_ret"].nunique() > 1 and win.loc[ok, "is_ret_ann"].nunique() > 1
        else np.nan
    )
    try:
        mom = return_moments(oos_daily.to_numpy())
        psr0 = psr(mom.sr, mom.n_obs, mom.skew, mom.kurt)
    except ValueError:
        psr0 = np.nan
    summary = pd.Series(
        {
            "is_len": combo.is_len,
            "oos_len": combo.oos_len,
            "val_len": combo.val_len,
            "n_oos_windows": len(win),
            "n_ok_windows": n_ok,
            "weak": n_ok < cfg.PWFO_MIN_WINDOWS,
            "n_oos_days": len(oos_daily),
            "oos_ret_ann": oos_ann,
            "oos_sharpe": oos_sr,
            "oos_sortino": float(oos_daily.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
            "max_dd": float((curve / curve.cummax() - 1).min()),
            "psr0": psr0,
            "is_ret_ann_mean": float(is_all.mean()),
            "is_ret_t": is_t,
            "is_sharpe_mean": float(is_sr_all.mean()),
            "is_sharpe_t": is_sr_t,
            "n_is_nonpos": int((is_ret[ok.to_numpy()] <= 0).sum()),
            "wfe": wfe_ret,
            "wfe_sharpe": wfe_sr,
            "pct_profitable_oos": float((win.loc[ok, "oos_ret"] > 0).mean()) if n_ok else np.nan,
            "pct_windows_traded": float(traded.sum() / n_ok) if n_ok else np.nan,
            "pct_profitable_traded": float((win.loc[traded, "oos_ret"] > 0).mean()) if traded.any() else np.nan,
            "is_oos_spearman": rho,
            "n_trades": int(win["n_oos_trades"].sum()),
            "turnover": float(eq.attrs.get("turnover", 0.0)) * TRADING_DAYS / max(len(oos_daily), 1),
        },
        name=combo.label,
    )
    return ComboStats(summary, win, oos_daily)


# ── Nested selection ─────────────────────────────────────────────────────────


def nested_select(
    returns: pd.DataFrame,
    is_sharpe: pd.DataFrame,
    is_len: dict[str, int],
    default: str,
    every: int,
    lookback: int,
) -> pd.DataFrame:
    """
    Walk-forward of the walk-forward (SPEC §6). `returns`: daily OOS returns, one column per combo (NaN outside the
    combo's OOS span), rows = sessions. `is_sharpe`: rows (combo, oos_start, is_sharpe) — a window's IS Sharpe is
    known when its OOS starts. Every `every` sessions from the default combo's first OOS session, at decision
    session d the combo with the highest Sharpe over its last `lookback` returns strictly before d is chosen
    (ties → higher WFE_sharpe over that lookback = that Sharpe / mean IS Sharpe of the windows whose OOS started in
    it → shorter IS → column order). Until every combo has `lookback` returns before d the default combo is used
    (burn-in). The last decision period ends where the first combo's returns end.

    Returns:
        pd.DataFrame: one row per decision: `start`, `end` (sessions, inclusive), `chosen`, `burn_in`,
            `score` (the chosen combo's trailing Sharpe; NaN in burn-in).
    """
    if default not in returns:
        raise ValueError(f"default combo {default!r} has no OOS returns")
    days = returns.index
    first = returns[default].first_valid_index()
    end = min(returns[c].last_valid_index() for c in returns)
    i, stop = days.get_loc(first), days.get_loc(end)
    vals = returns.to_numpy()
    valid = ~np.isnan(vals)
    is_start = is_sharpe.assign(pos=days.get_indexer(pd.DatetimeIndex(is_sharpe["oos_start"]).normalize()))
    rows = []
    while i <= stop:
        j = min(i + every, stop + 1)
        past = slice(max(0, i - lookback), i)
        full = valid[past].sum(axis=0) >= lookback if i >= lookback else np.zeros(len(returns.columns), bool)
        if full.all():
            keys = []
            for k, c in enumerate(returns.columns):
                r = vals[past, k]
                sr = round(_sharpe(r), 12)
                m = is_start[(is_start["combo"] == c) & (is_start["pos"] >= i - lookback) & (is_start["pos"] < i)]
                mean_is = m["is_sharpe"].mean() if len(m) else np.nan
                wfe = round(sr / mean_is, 12) if mean_is > 0 else -np.inf
                keys.append((sr, wfe, -is_len[c], -k))
            k = max(range(len(keys)), key=keys.__getitem__)
            rows.append((days[i], days[j - 1], returns.columns[k], False, keys[k][0]))
        else:
            rows.append((days[i], days[j - 1], default, True, np.nan))
        i = j
    return pd.DataFrame(rows, columns=["start", "end", "chosen", "burn_in", "score"])


def stitch(returns: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    """Daily PWFO returns: each decision period takes its chosen combo's returns. Columns `ret`, `combo`, `burn_in`."""
    parts = []
    for start, end, chosen, burn, _ in choice.itertuples(index=False):
        r = returns.loc[start:end, chosen]
        parts.append(pd.DataFrame({"ret": r, "combo": chosen, "burn_in": burn}, index=r.index))
    return pd.concat(parts)


# ── Full PWFO ────────────────────────────────────────────────────────────────


class PWFOResult(NamedTuple):
    summary: pd.DataFrame  # one row per combo
    windows: pd.DataFrame  # every combo's per-window table
    returns: pd.DataFrame  # daily OOS returns, one column per run combo
    choice: pd.DataFrame  # nested-selection decisions
    pwfo: pd.DataFrame  # stitched PWFO daily returns
    stats: dict  # headline PWFO statistics, PBO and DSR
    signals: dict[str, pd.DataFrame]  # combo label → OOS signals


def _combo_job(df, cfg, combo, unit, sessions, prep_kw, spread_bars):
    prep = prepare(df, cfg, **prep_kw)
    run = run_combo(df, cfg, prep, combo, unit=unit, sessions=sessions)
    if run.signals.empty:
        return run, None
    return run, combo_stats(df, run, cfg, spread_bars)


def run_pwfo(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    grid: list[Combo] | None = None,
    unit: str = "days",
    sessions=None,
    spread_bars=None,
    jobs: int = 1,
    log_dir=None,
    **prep_kw,
) -> PWFOResult:
    """
    Run every combo of `grid` (default make_grid(cfg)), then nested selection, PBO across combos and DSR.

    A combo with no window that could be fit (e.g. IS too short for MIN_TRAIN_EVENTS) is reported with
    n_ok_windows = 0 and left out of selection and PBO; it still counts as a trial for the DSR. jobs > 1 runs combos
    in processes (pin BLAS to one thread, see risk/run.py); `log_dir` receives each combo's stdout.
    """
    from concurrent.futures import ProcessPoolExecutor

    grid = make_grid(cfg) if grid is None else grid
    default = Combo(*cfg.PWFO_DEFAULT, 0).label
    if default not in {c.label for c in grid}:
        raise ValueError(f"PWFO_DEFAULT {cfg.PWFO_DEFAULT} is not in the grid")

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(grid))) as ex:
            futs = [ex.submit(_logged_job, log_dir, df, cfg, c, unit, sessions, prep_kw, spread_bars) for c in grid]
            outs = [f.result() for f in futs]
    else:
        outs = [_logged_job(log_dir, df, cfg, c, unit, sessions, prep_kw, spread_bars) for c in grid]

    summ, wins, rets, sigs, is_rows = [], [], {}, {}, []
    for combo, (run, st) in zip(grid, outs, strict=True):
        if st is None:
            s = pd.Series(
                {"is_len": combo.is_len, "oos_len": combo.oos_len, "val_len": combo.val_len,
                 "n_oos_windows": len(run.windows), "n_ok_windows": 0, "weak": True},
                name=combo.label,
            )  # fmt: skip
            summ.append(s)
            wins.append(run.windows.assign(combo=combo.label))
            continue
        summ.append(st.summary)
        wins.append(st.windows.assign(combo=combo.label))
        rets[combo.label] = st.oos_daily
        sigs[combo.label] = run.signals
        ok = st.windows["status"] == "ok"
        is_rows.append(
            pd.DataFrame(
                {"combo": combo.label, "oos_start": st.windows.loc[ok, "oos_start"],
                 "is_sharpe": st.windows.loc[ok, "is_sharpe"]}
            )
        )  # fmt: skip
    summary = pd.DataFrame(summ)
    windows = pd.concat(wins, ignore_index=True)
    if default not in rets:
        raise NoFitError(f"the default combo {default} produced no OOS windows; nested selection needs it")
    returns = pd.DataFrame(rets).sort_index()
    is_sharpe = pd.concat(is_rows, ignore_index=True)
    is_len = {c.label: c.is_len for c in grid}
    choice = nested_select(returns, is_sharpe, is_len, default, cfg.SELECT_EVERY, cfg.SELECT_LOOKBACK)
    pw = stitch(returns, choice)
    live = pw.loc[~pw["burn_in"], "ret"]

    stats: dict = {
        "n_combos": len(grid),
        "n_run_combos": len(rets),
        "n_decisions": len(choice),
        "n_burn_in_days": int(pw["burn_in"].sum()),
        "n_live_days": len(live),
        "live_start": str(live.index[0].date()) if len(live) else None,
        "live_end": str(live.index[-1].date()) if len(live) else None,
        "picks": choice.loc[~choice["burn_in"], "chosen"].value_counts().to_dict(),
    }
    if len(live) > 1:
        curve = (1 + live).cumprod()
        stats |= {
            "pwfo_ret_ann": _ann(float(curve.iloc[-1] - 1), len(live)),
            "pwfo_sharpe": _sharpe(live),
            "pwfo_max_dd": float((curve / curve.cummax() - 1).min()),
        }
        # DSR: the PWFO stream against the best of n_combos trials (per-period Sharpes, variance across the run
        # combos over the same live days)
        span = returns.loc[live.index].dropna(axis=1, how="any")
        srs = [s / np.sqrt(TRADING_DAYS) for s in (_sharpe(span[c]) for c in span)]
        try:
            mom = return_moments(live.to_numpy())
            stats |= {
                "pwfo_psr0": psr(mom.sr, mom.n_obs, mom.skew, mom.kurt),
                "pwfo_dsr": dsr(mom.sr, len(grid), float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0,
                                mom.n_obs, mom.skew, mom.kurt),
                "dsr_n_trials": len(grid),
                "dsr_var_trials": float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0,
            }  # fmt: skip
        except ValueError as e:  # flat stream
            stats["dsr_error"] = str(e)
    common = returns.dropna(axis=0, how="any")
    if returns.shape[1] >= 2 and len(common) >= 2 * cfg.PBO_BLOCKS:
        pr = pbo(common, cfg.PBO_BLOCKS)
        stats |= {
            "pbo": pr.pbo,
            "pbo_prob_oos_loss": pr.prob_oos_loss,
            "pbo_degradation_slope": pr.degradation_slope,
            "pbo_n_days": len(common),
            "pbo_n_combos": returns.shape[1],
        }
    else:
        stats["pbo"] = np.nan
    return PWFOResult(summary, windows, returns, choice, pw, stats, sigs)


def _logged_job(log_dir, df, cfg, combo, unit, sessions, prep_kw, spread_bars):
    import contextlib
    from pathlib import Path

    if log_dir is None:
        return _combo_job(df, cfg, combo, unit, sessions, prep_kw, spread_bars)
    with (
        open(Path(log_dir) / f"{combo.label}.log", "w", buffering=1, encoding="utf-8") as f,
        contextlib.redirect_stdout(f),
    ):
        return _combo_job(df, cfg, combo, unit, sessions, prep_kw, spread_bars)
