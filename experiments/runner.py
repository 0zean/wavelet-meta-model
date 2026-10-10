"""
Experiment runner (SPEC §9, U10): cells → cached, parallel trials → ledger rows.

Cell hash = sha256(canonical cell spec, every resolved RunConfig field, code hash, data hash)[:16]. The code hash
covers every project source file that can change a result (all *.py under the pipeline packages and experiments/,
except the report) plus the numeric library versions; the data hash covers each symbol's bars, its cost inputs
(COST_MODEL "cs": its 5Min bars; "quotes": its quotes-table rows) and the exchange sessions of a PWFO cell. Artifacts live in
`<root>/cells/<hash>/`. A cell whose (hash, stage) is already in the ledger is skipped; a hash already run in another
stage is re-recorded for this stage without fitting (`cache_hit`). Per-symbol WFO signals are also cached under
`<root>/signals/<key>.pkl`, keyed by the same hash minus the backtest-only fields (BACKTEST_ONLY), so risk-profile
and position-mode ablations reuse the fits.

Execution: data is loaded (and hashed) in the parent; the distinct per-symbol WFOs the cells need are fitted first
(once each), then the cells run in one flat pool of joblib processes (BLAS pinned to one thread) whose tasks are the
WFO cells and every PWFO cell's combos (SPEC §11.2; no nested pools); the parent assembles a PWFO cell when its last
combo is back and appends each cell's row as it finishes. A failing cell is an `error` row with its traceback under
`<root>/cells/<hash>/traceback.txt`; a cell with no fittable window is `no_fit`.

Rule cells (model.meta "none", stage F; SPEC §17): each symbol's rule pass (wfo/rule_pass.py, through the signals
cache) simulated through the portfolio simulator from the first live bar, no meta-model; kind "rule", n_trials 0 (the
family runner's variant row is the counted trial), plus the cost totals the family test's floors read.

Holdout: a cell whose range ends after HOLDOUT_START is refused unless `final=True`; a final run needs every cell in
stage E and past HOLDOUT_START, records a `holdout_access` event first (in the ledger and in HOLDOUT_MARKER, outside
any ledger), and is refused if either already holds a holdout access of a different batch (the holdout is evaluated
once; resuming the same batch is allowed).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib
import dataclasses
import hashlib
import json
import platform
import tempfile
import time
import traceback
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from data.bars import HOLDOUT_START, HoldoutError
from experiments import ledger as L
from experiments.spec import FINAL_STAGE, Cell
from features import cache as feature_cache
from utils.config import RunConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "results" / "experiments"
CODE_DIRS = ("data", "fastfracdiff", "features", "primaries", "models", "sizing", "risk", "validation", "wfo", "utils",
             "experiments")  # fmt: skip
CODE_EXCLUDE = {"experiments/report.py"}  # cannot change a result
LIBS = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost", "pywddff", "numba")
# RunConfig fields run_wfo never reads (only the backtest / PWFO do): left out of the per-symbol signals-cache key
BACKTEST_ONLY = frozenset(
    {"RISK_PROFILE", "COST_MODEL", "POSITION_MODE", "SIZE_STEP", "INIT_CASH", "SIZE", "PWFO_IS_GRID", "PWFO_OOS_GRID",
     "PWFO_EXPANDING", "PWFO_PARTIAL_LAST", "PWFO_VAL_FRAC", "PWFO_DEFAULT", "PWFO_MIN_WINDOWS", "PWFO_WFE_MIN_T", "SELECT_EVERY",
     "SELECT_LOOKBACK"}
)  # fmt: skip
TRADING_DAYS = 252


# ── Hashing ──────────────────────────────────────────────────────────────────


@cache
def code_hash() -> str:
    from importlib.metadata import version

    # the platform too: equal library versions on another OS / CPU (arm64 macOS vs x86-64 Windows) need not give
    # bit-identical floats, so a result is only reused where it was computed
    h = hashlib.sha256(f"python={platform.python_version()} {platform.system()} {platform.machine()}".encode())
    for lib in LIBS:
        h.update(f"{lib}={version(lib)}".encode())
    for d in CODE_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in CODE_EXCLUDE:
                continue
            h.update(rel.encode())
            h.update(path.read_bytes().replace(b"\r\n", b"\n"))  # a CRLF checkout is the same code
    for path in sorted((ROOT / "data" / "calendar").glob("*.csv")):  # the schedule sampler's `days` tables (U14)
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def cfg_fields(cfg: RunConfig, exclude=frozenset()) -> dict:
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.name not in exclude}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def data_hash(data: dict) -> str:
    parts = {f"bars:{s}": feature_cache.data_hash(df) for s, df in data["bars"].items()}
    for s, ctx in (data.get("context") or {}).items():  # feature context (U15): market / sector bars, exo series
        parts |= {f"context:{s}:{k}": feature_cache.context_hash(v) for k, v in ctx.items()}
    for s, d in (data.get("cost") or {}).items():  # cs: the 5Min bars; quotes: the symbol's quotes-table rows
        parts[f"cost:{s}"] = _sha(d.to_dict("list")) if "half_spread_bp" in d else feature_cache.data_hash(d)
    if data.get("sessions") is not None:
        parts["sessions"] = _sha([str(d.date()) for d in data["sessions"]])
    return _sha(parts)


def cell_hash(cell: Cell, cfg: RunConfig, dhash: str) -> str:
    return _sha({"spec": cell.spec, "cfg": cfg_fields(cfg), "code": code_hash(), "data": dhash})[:16]


def signals_key(symbol: str, cfg: RunConfig, df: pd.DataFrame, context: dict | None = None) -> str:
    key = {"symbol": symbol, "cfg": cfg_fields(cfg, BACKTEST_ONLY), "code": code_hash(),
           "data": feature_cache.data_hash(df)}  # fmt: skip
    if context:
        key["context"] = {k: feature_cache.context_hash(v) for k, v in context.items()}
    return _sha(key)[:24]


# ── Data ─────────────────────────────────────────────────────────────────────


class CachedBars:
    """Bars and exchange sessions from the data layer (data/bars.py; Alpaca + on-disk cache)."""

    def bars(self, symbol: str, timeframe: str, start: str, end: str, *, allow_holdout: bool) -> pd.DataFrame:
        from data.bars import load_bars

        return load_bars(symbol, timeframe, start, end, allow_holdout=allow_holdout)

    def sessions(self, end: str) -> pd.DatetimeIndex:
        from data.bars import get_calendar

        return get_calendar(end).index

    def exo(self, source: str, name: str, start, end, *, allow_holdout: bool) -> pd.DataFrame:
        from data.exo import load_series

        return load_series(source, name, start, end, allow_holdout=allow_holdout)

    def quotes_table(self, symbols: list[str]) -> pd.DataFrame:
        """Rows of the quotes half-spread table (data/costs/quotes_half_spread.csv) for `symbols`."""
        from data.quotes import read_table

        table = read_table()
        return table[table["symbol"].isin(symbols)]


def load_cell_data(cell: Cell, cfg: RunConfig, source, final: bool) -> dict:
    from features.context import load_context

    s = cell.spec
    bars = {sym: source.bars(sym, s["timeframe"], s["start"], s["end"], allow_holdout=final) for sym in cell.symbols}
    cost = None
    if cfg.COST_MODEL == "cs":  # Corwin–Schultz half-spreads are estimated from 5Min bars (SPEC §7)
        cost = {
            sym: bars[sym] if s["timeframe"] == "5Min" else source.bars(sym, "5Min", s["start"], s["end"],
                                                                         allow_holdout=final)
            for sym in cell.symbols
        }  # fmt: skip
    elif cfg.COST_MODEL == "quotes":  # SPEC §19: the symbol's rows of the quotes half-spread table
        table = source.quotes_table(list(cell.symbols))
        cost = {sym: table[table["symbol"] == sym].reset_index(drop=True) for sym in cell.symbols}
        missing = [sym for sym, rows in cost.items() if rows.empty]
        if missing:
            raise ValueError(f"COST_MODEL='quotes': no quotes-table rows for {missing} (run python -m data.quotes)")
    sessions = None
    if cell.is_pwfo:
        # Only the sessions between the first and last data session reach the PWFO (wfo.pwfo.unit_bounds); the
        # calendar itself extends a year past today and is refreshed monthly, which must not change the cell hash
        days = bars[cell.symbols[0]].index.normalize().tz_localize(None)
        cal = pd.DatetimeIndex(source.sessions(s["end"])).normalize()
        cal = cal.tz_localize(None) if cal.tz is not None else cal
        sessions = cal[(cal >= days[0]) & (cal <= days[-1])]
    # Feature context (SPEC §15): market / sector bars and exo series, only for cells whose groups read them
    context = {}
    for sym in cell.symbols:
        ctx = load_context(sym, s["timeframe"], s["start"], s["end"], cfg.FEATURE_GROUPS, bars=source.bars,
                           exo=lambda *a, **k: source.exo(*a, **k), allow_holdout=final)  # fmt: skip
        if ctx:
            context[sym] = ctx
    return {"bars": bars, "cost": cost, "sessions": sessions, "context": context}


# ── Metrics ──────────────────────────────────────────────────────────────────


def daily_stats(daily: pd.Series) -> dict:
    """Ledger return statistics from daily returns (comparable across timeframes; the DSR's per-period unit)."""
    from validation.stats import psr, return_moments

    r = daily.to_numpy(dtype=float)
    n = len(r)
    curve = np.cumprod(1 + r) if n else np.array([1.0])
    ret_ann = float(curve[-1] ** (TRADING_DAYS / n) - 1) if n else np.nan
    sd = r.std(ddof=1) if n > 1 else 0.0
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) if n else 0.0
    max_dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    out = {
        "n_obs": n,
        "ret_ann": ret_ann,
        "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
        "max_dd": max_dd,
        "calmar": ret_ann / abs(max_dd) if max_dd < 0 else np.nan,
        "sr_skew": np.nan,
        "sr_kurt": np.nan,
        "psr": np.nan,
    }
    try:
        m = return_moments(r)
        out |= {"sr_skew": float(m.skew), "sr_kurt": float(m.kurt), "psr": float(psr(m.sr, m.n_obs, m.skew, m.kurt))}
    except ValueError:  # flat (no trades) or too short: moments undefined
        pass
    return out


def _diag(df: pd.DataFrame, sig: pd.DataFrame, cfg: RunConfig) -> dict:
    from wfo.wfo_metrics import signal_diagnostics

    d = signal_diagnostics(df, sig, cfg)
    return {
        "n_oos_events": int(d["OOS events"]),
        "meta_auc": float(d["Meta AUC"]),
        "meta_logloss": float(d["Meta log-loss"]),
        "brier": float(d["Meta Brier"]),
    }


# ── One cell ─────────────────────────────────────────────────────────────────


def _signals(sym: str, df: pd.DataFrame, cfg: RunConfig, root: Path, feature_cache_dir, context=None) -> pd.DataFrame:
    """run_wfo for one symbol (the rule pass for META_MODEL "none"), through the signals cache."""
    from wfo.rule_pass import rule_signals
    from wfo.wfo_engine import run_wfo

    path = root / "signals" / f"{signals_key(sym, cfg, df, context)}.pkl"
    if path.exists():
        print(f"[EXP]  {sym}: signals cache hit {path.name}")
        return pd.read_pickle(path)
    fn = rule_signals if cfg.META_MODEL == "none" else run_wfo
    sig = fn(df, cfg, context=context, symbol=sym, feature_cache_dir=feature_cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pkl")
    os.close(fd)
    try:
        sig.to_pickle(tmp)
        feature_cache.publish(tmp, path)  # content-addressed too: a concurrent writer of the key wrote the same
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return sig


def _run_wfo_cell(cell: Cell, cfg: RunConfig, data: dict, root: Path, out: Path, feature_cache_dir) -> dict:
    from risk.costs import fill_costs
    from risk.portfolio import simulate_portfolio
    from risk.profiles import get_profile
    from wfo.backtest import run_backtest
    from wfo.pwfo import daily_returns
    from wfo.wfo_metrics import strategy_metrics

    sigs, diags = {}, {}
    context = data.get("context") or {}
    for sym, df in data["bars"].items():
        sigs[sym] = _signals(sym, df, cfg, root, feature_cache_dir, context.get(sym))
        sigs[sym].to_csv(out / f"signals_{sym}.csv")
        diags[sym] = _diag(df, sigs[sym], cfg)
    cost = data.get("cost") or {}
    row: dict = {}
    if len(sigs) == 1:
        (sym, sig), df = next(iter(sigs.items())), next(iter(data["bars"].values()))
        res = run_backtest(df.loc[sig.index[0] :], sig, cfg, cost_data=cost.get(sym))
        eq, trades = res["Meta-filtered"]
        row["sharpe_primary"] = daily_stats(daily_returns(res["Primary only"][0]))["sharpe"]
        row |= diags[sym]
        row["kind"] = "wfo"
        row["calibration_folds"] = sig.attrs.get("calibration_folds")
    else:
        profile = get_profile(cfg.RISK_PROFILE)
        bars = {s: data["bars"][s].loc[sig.index[0] :] for s, sig in sigs.items()}
        costs = None
        if cfg.COST_MODEL != "slippage":
            costs = {s: fill_costs(bars[s].index, cfg, cost.get(s)) for s in sigs}
        eq, trades, _ = simulate_portfolio(bars, sigs, cfg, profile, costs=costs)
        aucs = [d["meta_auc"] for d in diags.values()]
        row |= {
            "kind": "portfolio",
            "n_oos_events": int(sum(d["n_oos_events"] for d in diags.values())),
            "meta_auc": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan,
            "per_symbol": diags,
        }
    m = strategy_metrics(eq, trades, cfg.bars_per_year)
    daily = daily_returns(eq)
    daily.rename("ret").to_csv(out / "daily_returns.csv")
    row |= daily_stats(daily)
    row |= {
        "n_trades": int(m["Num Trades"]),
        "turnover": float(m["Turnover (x/yr)"]),
        "max_dd": float(m["Max Drawdown (%)"]) / 100,  # bar-level drawdown (the daily one misses intraday troughs)
        "n_trials": 1,
    }
    row["calmar"] = row["ret_ann"] / abs(row["max_dd"]) if row["max_dd"] < 0 else np.nan
    return row


def _run_rule_cell(cell: Cell, cfg: RunConfig, data: dict, root: Path, out: Path, feature_cache_dir) -> dict:
    """
    A rule cell (META_MODEL "none", stage F; SPEC §17): the rule pass of each symbol (wfo.rule_pass), simulated from
    the first live bar through the portfolio simulator (one or many symbols, cfg.RISK_PROFILE and cfg.COST_MODEL),
    with no meta-model. Writes daily_returns.csv and trades.csv; the row adds the totals the family test's floors read
    (traded_notional, cost_paid, pnl in cash on INIT_CASH). n_trials = 0: a rule cell is one instrument (or the basket)
    of a family variant, and the variant is the counted trial (its own ledger row, families/run.py).
    """
    from risk.costs import fill_costs
    from risk.portfolio import simulate_portfolio
    from risk.profiles import get_profile
    from wfo.pwfo import daily_returns
    from wfo.wfo_engine import NoFitError
    from wfo.wfo_metrics import strategy_metrics

    context, cost = data.get("context") or {}, data.get("cost") or {}
    sigs = {}
    for sym, df in data["bars"].items():
        sigs[sym] = _signals(sym, df, cfg, root, feature_cache_dir, context.get(sym))
        sigs[sym].to_csv(out / f"signals_{sym}.csv")
    starts = [s.attrs.get("live_start") for s in sigs.values() if s.attrs.get("live_start") is not None]
    if not starts:
        raise NoFitError("the rule pass has no segment: the data is shorter than the warm-up (INITIAL_TRAIN + VAL)")
    live = min(starts)
    bars = {s: data["bars"][s].loc[live:] for s in sigs}
    costs = None
    if cfg.COST_MODEL != "slippage":
        costs = {s: fill_costs(bars[s].index, cfg, cost.get(s)) for s in sigs}
    eq, trades, _ = simulate_portfolio(bars, sigs, cfg, get_profile(cfg.RISK_PROFILE), side_col="trade_signal",
                                       size_col="bet_size", costs=costs)  # fmt: skip
    trades.to_csv(out / "trades.csv")
    m = strategy_metrics(eq, trades, cfg.bars_per_year)
    daily = daily_returns(eq)
    daily.rename("ret").to_csv(out / "daily_returns.csv")
    row = {"kind": "rule", **daily_stats(daily)}
    row |= {
        "n_trades": int(m["Num Trades"]),
        "turnover": float(m["Turnover (x/yr)"]),
        "max_dd": float(m["Max Drawdown (%)"]) / 100,
        "n_trials": 0,
        "live_start": str(live),
        "init_cash": float(cfg.INIT_CASH),
        "pnl": float(eq.iloc[-1] - cfg.INIT_CASH),
        "traded_notional": float(eq.attrs["traded_notional"]),
        "cost_paid": float(eq.attrs["cost_paid"]),
        "n_oos_events": int(sum(len(s) for s in sigs.values())),
        "n_sided": int(sum(int((s["signed_dir"] != 0).sum()) for s in sigs.values())),
        "skipped_segments": {s: sig.attrs.get("skipped_segments", []) for s, sig in sigs.items()},
    }
    row["calmar"] = row["ret_ann"] / abs(row["max_dd"]) if row["max_dd"] < 0 else np.nan
    return row


def _run_pwfo_cell(cell: Cell, cfg: RunConfig, data: dict, out: Path, feature_cache_dir) -> dict:
    """A PWFO cell in this process, its combos one after another (jobs = 1, or a crash re-run)."""
    from wfo.pwfo import run_pwfo

    (sym, df), cost = next(iter(data["bars"].items())), (data.get("cost") or {})
    (out / "logs").mkdir(exist_ok=True)
    res = run_pwfo(
        df, cfg, sessions=data["sessions"], cost_data=cost.get(sym), log_dir=out / "logs", symbol=sym,
        feature_cache_dir=feature_cache_dir, context=(data.get("context") or {}).get(sym),
    )  # fmt: skip
    return _pwfo_row(cell, cfg, res, out)


def _pwfo_row(cell: Cell, cfg: RunConfig, res, out: Path) -> dict:
    """Write a PWFO result's files and build the cell's ledger row."""
    from wfo.pwfo import Combo
    from wfo.pwfo_run import write_outputs

    write_outputs(res, cfg, out, "pwfo", f"PWFO {cell.label()}")
    live = res.pwfo.loc[~res.pwfo["burn_in"], "ret"]
    holdout = {}
    if cfg.ALLOW_HOLDOUT:  # stage E: the row's return statistics are the holdout days'; the full stream is kept
        live.to_csv(out / "daily_returns_full.csv")
        ho = HOLDOUT_START if live.index.tz is None else HOLDOUT_START.tz_localize(live.index.tz)
        holdout = {"holdout_start": str(HOLDOUT_START.date()), "n_pre_holdout_days": int((live.index < ho).sum())}
        live = live[live.index >= ho]
    live.to_csv(out / "daily_returns.csv")
    st, summ = res.stats, res.summary
    nested = cfg.PWFO_COMBINE == "nested"
    default = Combo(*cfg.PWFO_DEFAULT, 0).label
    combos = [
        {"combo": c, "sharpe": s.get("oos_sharpe"), "wfe": s.get("wfe"), "n_oos_windows": int(s["n_oos_windows"]),
         "n_ok_windows": int(s["n_ok_windows"])}
        for c, s in summ.iterrows()
    ]  # fmt: skip
    return {
        "kind": "pwfo",
        **daily_stats(live),
        "n_trials": int(st["n_combos"]),  # every grid combo is a trial (SPEC §6), fitted or not
        "n_trades": int(summ["n_trades"].fillna(0).sum()),
        # nested: the default combo's windows (U9–U11); average: every combo's (all of them are traded)
        "n_oos_windows": int(summ.loc[default, "n_oos_windows"] if nested else summ["n_oos_windows"].sum()),
        "combine": cfg.PWFO_COMBINE,
        "n_run_combos": int(st["n_run_combos"]),  # combos with a fitted window (the average / selection is over these)
        "calibration_windows": st.get("calibration_windows"),
        "pbo": st.get("pbo"),
        "pwfo_dsr": st.get("pwfo_dsr"),
        "n_decisions": int(st["n_decisions"]),
        "picks": st.get("picks"),
        "combos": combos,
        **holdout,
    }


def _cell_dir(cell: Cell, chash: str, root) -> Path:
    out = Path(root) / "cells" / chash
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(
        json.dumps({"stage": cell.stage, "spec": cell.spec}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def _guarded(cell: Cell, out: Path, body) -> dict:
    """row = {"status": "ok"} | body() under the cell's log; never raises: no_fit / error rows with a traceback."""
    from wfo.wfo_engine import NoFitError

    row: dict = {"status": "ok"}
    with open(out / "log.txt", "a", buffering=1, encoding="utf-8") as log, contextlib.redirect_stdout(log):
        try:
            row |= body()
        except NoFitError as e:  # no window could be fit: a counted trial without a result
            row = {"status": "no_fit", "error": str(e), "n_trials": cell_trials(cell)}
        except Exception as e:  # noqa: BLE001 — a failing cell is recorded (status=error), never dropped
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            row = {"status": "error", "error": f"{type(e).__name__}: {e}", "error_path": str(out / "traceback.txt")}
    return row


def _finish(out: Path, row: dict, runtime_s: float) -> dict:
    row["runtime_s"] = round(runtime_s, 2)
    (out / "result.json").write_text(json.dumps(L.clean(row), indent=2, sort_keys=True), encoding="utf-8")
    return row


def run_cell(cell: Cell, chash: str, data: dict, final: bool, root, feature_cache_dir) -> dict:
    """Run one cell in this process (a PWFO cell's combos serially); never raises (see _guarded)."""
    root = Path(root)
    out = _cell_dir(cell, chash, root)
    (out / "log.txt").write_text("", encoding="utf-8")
    t0 = time.time()

    def body():
        cfg = cell.config(final)
        if cell.is_pwfo:
            return _run_pwfo_cell(cell, cfg, data, out, feature_cache_dir)
        if cfg.META_MODEL == "none":
            return _run_rule_cell(cell, cfg, data, root, out, feature_cache_dir)
        return _run_wfo_cell(cell, cfg, data, root, out, feature_cache_dir)

    return _finish(out, _guarded(cell, out, body), time.time() - t0)


def pwfo_grid(cell: Cell, final: bool) -> list:
    from wfo.pwfo import check_grid, make_grid

    cfg = cell.config(final)
    grid = make_grid(cfg)
    check_grid(cfg, grid)
    return grid


def run_combo_task(cell: Cell, chash: str, data: dict, final: bool, root, feature_cache_dir, combo) -> tuple:
    """
    One combo of a PWFO cell (a flat-pool task): wfo.pwfo.combo_job with its stdout in the cell's logs/<combo>.log.
    Returns (outcome, traceback, seconds): outcome = (ComboRun, ComboStats | None), or None with the traceback.
    """
    from wfo.pwfo import combo_job

    out = Path(root) / "cells" / chash
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (sym, df), cost = next(iter(data["bars"].items())), (data.get("cost") or {})
    t0 = time.time()
    with (
        open(out / "logs" / f"{combo.label}.log", "w", buffering=1, encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        try:
            res = combo_job(df, cell.config(final), combo, "days", data["sessions"], cost.get(sym), symbol=sym,
                            feature_cache_dir=feature_cache_dir,
                            context=(data.get("context") or {}).get(sym))  # fmt: skip
            return res, None, time.time() - t0
        except Exception:  # noqa: BLE001 — recorded as the cell's error row by finish_pwfo_cell
            return None, traceback.format_exc(), time.time() - t0


def finish_pwfo_cell(cell: Cell, chash: str, final: bool, root, grid: list, results: list) -> dict:
    """
    The parent's half of a flat-pool PWFO cell: `results` = run_combo_task outputs in grid order → wfo.pwfo.assemble →
    files and ledger row. A failed combo makes the cell an `error` row (every combo's traceback in traceback.txt).
    runtime_s = the combos' summed task seconds + assembly (what the cell costs serially).
    """
    from wfo.pwfo import assemble

    out = Path(root) / "cells" / chash
    t0 = time.time()
    seconds = sum(s for _, _, s in results)
    failed = [(c.label, tb) for c, (_, tb, _) in zip(grid, results, strict=True) if tb is not None]
    if failed:
        (out / "traceback.txt").write_text("\n".join(f"── {c} ──\n{tb}" for c, tb in failed), encoding="utf-8")
        msg = "; ".join(f"{c}: {tb.strip().splitlines()[-1]}" for c, tb in failed)
        row = {"status": "error", "error": f"{len(failed)} of {len(grid)} combo(s) failed: {msg}",
               "error_path": str(out / "traceback.txt")}  # fmt: skip
        return _finish(out, row, seconds)

    def body():
        cfg = cell.config(final)
        return _pwfo_row(cell, cfg, assemble(cfg, grid, [res for res, _, _ in results]), out)

    return _finish(out, _guarded(cell, out, body), seconds + time.time() - t0)


def cell_trials(cell: Cell) -> int:
    if cell.is_rule:  # a family member: the family variant row is the counted trial (families/run.py)
        return 0
    return len(cell.spec["pwfo"]["is_grid"]) * len(cell.spec["pwfo"]["oos_grid"]) if cell.is_pwfo else 1


# ── A batch ──────────────────────────────────────────────────────────────────


def git_sha() -> str:
    from models.compare import _git_sha

    return _git_sha()


# Second record of every holdout access, outside any ledger file: a final run against another --ledger still sees it
HOLDOUT_MARKER = ROOT / "data" / "cache" / "holdout_access.jsonl"


def _batch_id(cells: list[Cell]) -> str:
    """Identity of a final (holdout) batch: its cells and code, without touching holdout data."""
    return _sha(sorted(_sha({"spec": c.spec, "stage": c.stage}) for c in cells) + [code_hash()])[:16]


def check_holdout(cells: list[Cell], final: bool, events: list[dict]) -> str | None:
    """
    Refuse holdout-crossing cells without `final`; validate a final batch against the recorded holdout accesses
    (`events`). Returns the batch id when final.
    """
    hs = str(HOLDOUT_START.date())
    crossing = [c.label() for c in cells if c.spec["end"] > hs]
    if not final:
        if crossing:
            raise HoldoutError(
                f"{len(crossing)} cell(s) end after HOLDOUT_START {hs} (e.g. {crossing[0]}); "
                "only `run --final` (stage E) may read the holdout"
            )
        return None
    wrong = sorted({c.stage for c in cells} - {FINAL_STAGE})
    if wrong:
        raise HoldoutError(f"--final runs only stage {FINAL_STAGE} cells; the spec has stage(s) {wrong}")
    dev = [c.label() for c in cells if c.spec["end"] <= hs]
    if dev:  # a final batch that reads no holdout data would still use up the one holdout access
        raise HoldoutError(f"--final cells must run past HOLDOUT_START {hs}; {len(dev)} do not (e.g. {dev[0]})")
    batch = _batch_id(cells)
    other = [e for e in events if e.get("batch_id") != batch]
    if other:
        raise HoldoutError(
            f"the holdout was already accessed by batch {other[0].get('batch_id')} at {other[0].get('started_at')}; "
            "it is evaluated once (SPEC §9)"
        )
    return batch


def _crash_row(out: Path, err: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "traceback.txt").write_text(err, encoding="utf-8")
    return {"status": "error", "error": err, "error_path": str(out / "traceback.txt"), "runtime_s": 0}


def _isolated(fn, *args):
    """fn(*args) in a fresh single-use process; None if that process dies (OOM kill, segfault, ...)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        try:
            return ex.submit(fn, *args).result()
        except BrokenProcessPool:
            return None


def run(
    cells: list[Cell],
    *,
    ledger: L.Ledger,
    root=DEFAULT_ROOT,
    source=None,
    jobs: int = 1,
    final: bool = False,
    retry_errors: bool = False,
    spec_name: str | None = None,
    feature_cache_dir=feature_cache.DEFAULT_ROOT,
    holdout_marker=HOLDOUT_MARKER,
    on_row=None,
    on_hash=None,
) -> list[dict]:
    """
    Run `cells` (module docstring) and append their ledger rows; returns the rows written by this call.
    One run per ledger at a time (an exclusive run lock; a second run raises). jobs >= 2 runs cells in worker
    processes: a worker that dies is detected, and the cells not yet recorded are re-run one per fresh process, a
    cell whose process dies again becoming an `error` row. jobs = 1 runs in-process (no crash isolation).
    `on_row(row)` is called after each append (progress / tests); `on_hash(cell, chash)` for every cell once its hash
    is known, whether it then runs or is skipped (the family runner reads the cells' artifacts by hash).
    """
    with ledger.run_lock():
        return _run(cells, ledger, Path(root), source or CachedBars(), jobs, final, retry_errors,
                    spec_name, feature_cache_dir, L.Ledger(holdout_marker), on_row, on_hash)  # fmt: skip


def _run(cells, ledger, root, source, jobs, final, retry_errors, spec_name, feature_cache_dir, marker,
         on_row, on_hash=None) -> list[dict]:  # fmt: skip
    from joblib import Parallel, delayed
    from joblib.externals.loky.process_executor import BrokenProcessPool

    rows = ledger.rows()
    events = L.holdout_events(rows) + L.holdout_events(marker.rows())
    batch = check_holdout(cells, final, events)
    run_id = f"{datetime.now(UTC).isoformat()}-{os.getpid()}"
    sha = git_sha()
    common = {"git_sha": sha, "code_hash": code_hash(), "run_id": run_id, "final": final, "spec_name": spec_name}
    if final:
        event = {"event": "holdout_access", "batch_id": batch, "started_at": datetime.now(UTC).isoformat(),
                 "ledger": str(ledger.path), **common, "n_cells": len(cells)}  # fmt: skip
        if not L.holdout_events(marker.rows()):
            marker.append(event)
        if not L.holdout_events(rows):
            ledger.append(event)

    done = L.done(rows)
    by_hash: dict[str, dict] = {}  # any finished trial row per hash (for cross-stage cache hits)
    for r in L.trials(rows):
        if r.get("status") in L.COUNTED and r.get("kind") != "legacy":
            by_hash.setdefault(r["cell_hash"], r)
    written: list[dict] = []

    def emit(row: dict) -> None:
        ledger.append(row)
        written.append(row)
        print(f"[EXP]  {row['stage']} {row['status']:<7} {row['cell_hash']}  {row.get('label', '')}")
        if on_row is not None:
            on_row(row)

    todo, seen = [], set()
    for cell in cells:
        started = datetime.now(UTC).isoformat()
        base = {"stage": cell.stage, "label": cell.label(), "spec_json": cell.spec_json(), "started_at": started}
        cfg = cell.config(final)
        try:
            data = load_cell_data(cell, cfg, source, final)
            dhash = data_hash(data)
        except Exception:  # noqa: BLE001 — data that cannot be loaded is a recorded failure
            data, dhash = None, "load-error"
            err = traceback.format_exc()
        chash = cell_hash(cell, cfg, dhash)
        if on_hash is not None:
            on_hash(cell, chash)
        if (chash, cell.stage) in seen:
            continue
        seen.add((chash, cell.stage))
        prev = done.get((chash, cell.stage))
        if prev is not None and (prev["status"] in L.COUNTED or not retry_errors):
            print(f"[EXP]  skip {chash} ({cell.stage}, {prev['status']} in the ledger)")
            continue
        base |= {"cell_hash": chash, "data_hash": dhash, **common}
        if data is None:
            row = _crash_row(root / "cells" / chash, err)
            emit({**base, **row, "error": err.strip().splitlines()[-1]})
            continue
        if chash in by_hash:  # same configuration, code and data, recorded in another stage: no refit
            hit = {k: v for k, v in by_hash[chash].items() if k not in base and k not in ("cache_hit", "cache_from")}
            emit({**base, **hit, "cache_hit": True, "cache_from": by_hash[chash]["stage"], "runtime_s": 0})
            continue
        todo.append((cell, chash, data, base))

    # Phase 1: every distinct per-symbol WFO the WFO / portfolio cells need, once each (cells sharing a symbol's fits,
    # e.g. risk-profile ablations or a portfolio and its members, would otherwise fit it concurrently). A failure
    # here, even a dead worker, is only logged: the cell re-runs it and records the error.
    sig_jobs = {}
    for cell, _, data, _ in todo:
        if cell.is_pwfo:
            continue
        cfg = cell.config(final)
        context = data.get("context") or {}
        for sym, df in data["bars"].items():
            key = signals_key(sym, cfg, df, context.get(sym))
            if key not in sig_jobs and not (root / "signals" / f"{key}.pkl").exists():
                sig_jobs[key] = (sym, df, cfg, context.get(sym))
    if len(sig_jobs) > 1 and jobs > 1:
        print(f"[EXP]  fitting {len(sig_jobs)} per-symbol WFO(s) with {jobs} job(s)")
        par = Parallel(n_jobs=min(jobs, len(sig_jobs)), return_as="generator_unordered")
        try:
            for key, err in par(
                delayed(_signals_job)(k, *v, str(root), feature_cache_dir) for k, v in sig_jobs.items()
            ):
                if err:
                    print(f"[EXP]  signals {key} failed ({err}); its cells will record the error")
        except BrokenProcessPool as e:
            print(f"[EXP]  a signals worker died ({type(e).__name__}); the cells will re-run their WFOs")

    # Phase 2: the cells. jobs = 1: each in this process. jobs >= 2: one flat pool (SPEC §11.2) whose tasks are the
    # WFO cells and every PWFO cell's combos; the parent assembles a PWFO cell once its last combo is back.
    pending = dict(enumerate(todo))
    if not pending:
        return written
    print(f"[EXP]  running {len(pending)} cell(s) with {jobs} job(s)")
    if jobs == 1:
        for i in list(pending):
            cell, chash, data, base = pending.pop(i)
            emit({**base, **run_cell(cell, chash, data, final, root, feature_cache_dir)})
        return written

    tasks, grids, combo_out = [], {}, {}
    for i, (cell, chash, data, base) in list(pending.items()):
        if not cell.is_pwfo:
            tasks.append(delayed(_job)(i, cell, chash, data, final, str(root), feature_cache_dir))
            continue
        out = _cell_dir(cell, chash, root)
        (out / "log.txt").write_text("", encoding="utf-8")
        try:
            grids[i] = pwfo_grid(cell, final)
        except Exception:  # noqa: BLE001 — an invalid grid is the cell's error row
            pending.pop(i)
            emit({**base, **_finish(out, _guarded(cell, out, lambda c=cell: pwfo_grid(c, final)), 0)})
            continue
        combo_out[i] = {}
        tasks += [delayed(_combo_job)(i, k, cell, chash, data, final, str(root), feature_cache_dir, combo)
                  for k, combo in enumerate(grids[i])]  # fmt: skip
    if not tasks:
        return written
    par = Parallel(n_jobs=min(jobs, len(tasks)), return_as="generator_unordered")
    try:
        for i, k, res in par(tasks):
            if k is None:
                emit({**pending.pop(i)[3], **res})
                continue
            combo_out[i][k] = res
            if len(combo_out[i]) == len(grids[i]):
                cell, chash, _, base = pending.pop(i)
                done_k = combo_out.pop(i)
                results = [done_k[j] for j in range(len(grids[i]))]
                emit({**base, **finish_pwfo_cell(cell, chash, final, root, grids[i], results)})
    except BrokenProcessPool as e:
        print(f"[EXP]  a worker died ({type(e).__name__}); re-running {len(pending)} cell(s) one per process")
        for i in list(pending):
            cell, chash, data, base = pending.pop(i)
            res = _isolated(_job, i, cell, chash, data, final, str(root), feature_cache_dir)
            if res is None:
                msg = "WorkerDied: the cell's process terminated abruptly (out of memory, segfault or a kill)"
                emit({**base, **_crash_row(root / "cells" / chash, msg)})
            else:
                emit({**base, **res[2]})
    return written


def _signals_job(key, sym, df, cfg, context, root, feature_cache_dir):
    root = Path(root)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    with (
        open(root / "signals" / f"{key}.log", "w", buffering=1, encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        try:
            _signals(sym, df, cfg, root, feature_cache_dir, context)
            return key, None
        except Exception as e:  # noqa: BLE001 — re-raised (and recorded) by the cell
            return key, f"{type(e).__name__}: {e}"


def _job(i, cell, chash, data, final, root, feature_cache_dir):
    return i, None, run_cell(cell, chash, data, final, root, feature_cache_dir)


def _combo_job(i, k, cell, chash, data, final, root, feature_cache_dir, combo):
    return i, k, run_combo_task(cell, chash, data, final, root, feature_cache_dir, combo)
