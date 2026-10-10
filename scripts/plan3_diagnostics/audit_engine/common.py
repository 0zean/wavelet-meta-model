import os
import sys

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])  # the repo root
sys.path.insert(0, REPO)
os.chdir(REPO)
from data.bars import load_bars
from data.quotes import read_table
from risk.costs import fill_costs
from risk.portfolio import simulate_portfolio
from risk.profiles import get_profile
from wfo.rule_pass import rule_signals


def bars(sym, tf, start="2016-01-01", end="2025-10-01"):
    return load_bars(sym, tf, start, end)


def run_rule(df, cfg, sym="SPY", costs=True, signals=None, side_col="trade_signal", size_col="bet_size"):
    sig = rule_signals(df, cfg, symbol=sym) if signals is None else signals
    live = sig.attrs.get("live_start")
    b = df.loc[live:] if live is not None else df
    c = None
    if costs and cfg.COST_MODEL != "slippage":
        tbl = read_table()
        tbl = tbl[tbl.symbol == sym]
        c = {sym: fill_costs(b.index, cfg, tbl)}
    eq, tr, log = simulate_portfolio(
        {sym: b}, {sym: sig}, cfg, get_profile(cfg.RISK_PROFILE), side_col=side_col, size_col=size_col, costs=c
    )
    return sig, eq, tr
