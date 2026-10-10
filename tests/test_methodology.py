"""
Leakage, labeling and execution checks.

The causality tests perturb every bar from a boundary onward and require that
nothing computed for earlier bars changes.
"""

import numpy as np
import pandas as pd
import pytest
import pywddff.pywddff as pyw
from scipy.stats import siegelslopes

from features.causal_modwt import wavelet_ar_features
from features.events import sample_events
from features.feature_builder import build_features
from features.fractional_diff import fit_fracdiff_d, fracdiff_transform
from features.indicators import compute_siegel_slope
from features.triple_barrier_labels import average_uniqueness, barrier_exits, triple_barrier_labels
from utils.config import RunConfig
from utils.data_loader import load_ohlcv
from wfo.backtest import equity_curve, simulate_trades
from wfo.wfo_engine import purged, run_wfo

BOUNDARY = 4250
CFG = RunConfig.legacy_5min()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_ohlcv("data/data.csv")


def perturb_after(df: pd.DataFrame, start: int, seed: int = 0) -> pd.DataFrame:
    """Multiply prices from bar `start` onward by a random walk and scramble volume."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = len(df) - start
    shock = np.exp(np.cumsum(rng.normal(0, 0.003, k)))
    for col in ("open", "high", "low", "close"):
        out.iloc[start:, out.columns.get_loc(col)] *= shock
    out.iloc[start:, out.columns.get_loc("volume")] = rng.integers(1e5, 1e7, k).astype(float)
    return out


# ── Causality ────────────────────────────────────────────────────────────────


def test_features_are_causal(df):
    a = build_features(df, CFG).iloc[:BOUNDARY]
    b = build_features(perturb_after(df, BOUNDARY), CFG).iloc[:BOUNDARY]
    pd.testing.assert_frame_equal(a, b)


def test_fracdiff_is_causal(df):
    fs = fit_fracdiff_d(df["close"].iloc[:2000])
    a = fracdiff_transform(df["close"], fs).loc[: df.index[BOUNDARY - 1]]
    b = fracdiff_transform(perturb_after(df, BOUNDARY)["close"], fs).loc[: df.index[BOUNDARY - 1]]
    pd.testing.assert_series_equal(a, b)


def _events_causal(df, cfg, n_cuts: int = 40):
    """Per-event cuts (U22, engine audit): for each of the first `n_cuts` events after bar 2000, perturb every bar
    after the event bar and require the events and widths up to and including it unchanged. A single cut far from any
    event does not bite on a one-bar look-ahead in the width (the VOL mutant); a cut right after each event does."""
    a = sample_events(df, cfg)
    pos = df.index.get_indexer(a.index)
    for t in pos[pos >= 2000][:n_cuts]:
        b = sample_events(perturb_after(df, int(t) + 1, seed=int(t)), cfg)
        cut = df.index[t]
        pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


def test_events_and_widths_are_causal(df):
    _events_causal(df, CFG)


def test_the_per_event_causality_check_catches_a_one_bar_look_ahead_in_the_width(df, monkeypatch):
    """The engine audit's VOL mutant (bar_volatility reads the next bar's return) escaped the single-cut test."""
    import features.events as EV

    orig = EV.bar_volatility

    def leaky(close, span, profile=None):
        lr = np.log(close).diff().shift(-1)
        return lr.ewm(span=span, min_periods=span).std().rename("bar_vol")

    monkeypatch.setattr(EV, "bar_volatility", leaky)
    with pytest.raises(AssertionError):
        _events_causal(df, CFG, n_cuts=10)
    monkeypatch.setattr(EV, "bar_volatility", orig)


def test_labels_only_depend_on_their_window(df):
    ev = sample_events(df, CFG)
    a = triple_barrier_labels(df, ev, CFG)
    b = triple_barrier_labels(perturb_after(df, BOUNDARY), ev, CFG)
    done = a.index[a["exit_pos"] < BOUNDARY]
    pd.testing.assert_frame_equal(a.loc[done], b.loc[done])


def test_purged_samples_never_reach_next_split(df):
    labels = triple_barrier_labels(df, sample_events(df, CFG), CFG)
    for start, end in [(0, 2000), (2000, 3000), (3000, 4000)]:
        rows = purged(labels, start, end)
        t = rows["entry_pos"] - 1
        assert ((t >= start) & (t < end)).all()
        assert (rows["exit_pos"] < end).all()
        # labels that were dropped really did straddle the boundary
        t_all = labels["entry_pos"] - 1
        straddle = labels[(t_all >= start) & (t_all < end) & (labels["exit_pos"] >= end)]
        assert rows.index.intersection(straddle.index).empty


def test_wfo_predictions_are_causal(df):
    base = df.iloc[:4500]
    a = run_wfo(base, CFG)
    b = run_wfo(perturb_after(base, BOUNDARY), CFG)
    cut = base.index[BOUNDARY - 1]
    cols = ["clf_prob", "magnitude", "meta_prob", "trade_signal", "width"]
    pd.testing.assert_frame_equal(a.loc[:cut, cols], b.loc[:cut, cols])


# ── Feature implementations ──────────────────────────────────────────────────


def test_siegel_matches_scipy(df):
    close = df["close"].iloc[:300]
    w = CFG.SIEGEL_WINDOW
    ref = [np.nan] * (w - 1) + [siegelslopes(close.to_numpy()[i - w + 1 : i + 1]).slope for i in range(w - 1, 300)]
    np.testing.assert_allclose(compute_siegel_slope(close, w).to_numpy(), ref, equal_nan=True)


def test_modwt_matches_pywddff(df):
    close = df["close"].iloc[:500]
    coefs = pyw.modwt(x=close.to_numpy(), filter=CFG.WAVELET_FILTER, J=CFG.WAVELET_J, remove_bc=True)
    ref = pd.Series(coefs[:, -1], index=close.index[-coefs.shape[0] :]).reindex(close.index).shift(1)
    np.testing.assert_allclose(
        wavelet_ar_features(close, CFG.WAVELET_FILTER, CFG.WAVELET_J, CFG.AR_LAGS)["w_lag_1"].to_numpy(),
        ref.to_numpy(),
        equal_nan=True,
    )


# ── Triple barrier mechanics ─────────────────────────────────────────────────


def make_bars(rows: list[tuple[float, float, float, float]], start: str = "2024-01-02 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    out = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    out["volume"] = 1.0
    return out


def one_event(
    df: pd.DataFrame, width: float, side: float | None = None, vertical_bars: int = 3, hold_overnight: bool = False
) -> pd.Series:
    ev = df.index[[0]]
    w = pd.Series(width, index=df.index)
    s = None if side is None else pd.Series(side, index=ev)
    return barrier_exits(df, ev, w, side=s, vertical_bars=vertical_bars, hold_overnight=hold_overnight).iloc[0]


def test_entry_is_next_open_and_upper_hit():
    bars = make_bars([(100, 100, 100, 100), (101, 101.5, 100.8, 101), (101, 102.2, 100.9, 102), (102, 102, 102, 102)])
    out = one_event(bars, width=0.01)
    assert out["entry_px"] == 101
    assert out["barrier"] == "upper"
    assert out["exit_px"] == pytest.approx(102.01)
    assert out["exit_pos"] == 2


def test_gap_through_barrier_fills_at_open():
    bars = make_bars([(100, 100, 100, 100), (100, 100.2, 99.9, 100), (97, 97.5, 96.5, 97), (97, 97, 97, 97)])
    out = one_event(bars, width=0.01)
    assert out["barrier"] == "lower"
    assert out["exit_px"] == 97  # opened below the 99.0 barrier


def test_same_bar_tie_is_adverse_for_known_side():
    bars = make_bars([(100, 100, 100, 100), (100, 101.5, 98.5, 100.2), (100, 100, 100, 100), (100, 100, 100, 100)])
    assert one_event(bars, 0.01, side=1)["barrier"] == "lower"
    assert one_event(bars, 0.01, side=-1)["barrier"] == "upper"
    assert one_event(bars, 0.01)["barrier"] == "tie"


def test_vertical_barrier_truncated_at_session_close():
    bars = make_bars([(100, 100, 100, 100)] * 3, start="2024-01-02 15:45").pipe(
        lambda d: pd.concat([d, make_bars([(100, 100, 100, 100)] * 3, start="2024-01-03 09:30")])
    )
    out = one_event(bars, width=0.05, vertical_bars=5)
    assert out["barrier"] == "vertical"
    assert out["t1"] == pd.Timestamp("2024-01-02 15:55")


def test_average_uniqueness():
    labels = pd.DataFrame({"entry_pos": [0, 0, 10], "exit_pos": [3, 3, 12]})
    np.testing.assert_allclose(average_uniqueness(labels, 20).to_numpy(), [0.5, 0.5, 1.0])


# ── Backtest ─────────────────────────────────────────────────────────────────


def test_backtest_slippage_is_adverse_and_equity_compounds(df):
    events = sample_events(df, CFG).iloc[:200]
    signals = events.assign(trade_signal=np.where(np.arange(len(events)) % 2, 1, -1))
    trades = simulate_trades(df, signals, CFG)
    assert (trades["side"] * (trades["entry_fill"] - trades["entry_px"]) > 0).all()  # pay up to enter
    assert (trades["side"] * (trades["exit_fill"] - trades["exit_px"]) < 0).all()  # give up to exit
    assert (trades["entry_pos"].to_numpy()[1:] > trades["exit_pos"].to_numpy()[:-1]).all()  # no overlap
    eq = equity_curve(df, trades, CFG.INIT_CASH, CFG.SIZE)
    assert eq.iloc[-1] == pytest.approx(CFG.INIT_CASH * np.prod(1 + trades["pnl_pct"]))
