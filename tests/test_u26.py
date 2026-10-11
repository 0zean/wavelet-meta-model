"""
U26: time–frequency state (SPEC §24): fastgoertzel's parity with a numpy DFT, the spectral and MODWT state on processes
with known spectra, the session windows (previous sessions only, the overnight gap excluded), SKEW / VVIX through the
point-in-time rule, the U26 state splits and their re-render on a stored result. Causality of `tf_state` under bar
perturbation and truncation, and of `vol_state`'s new columns under exo perturbation, runs in tests/test_features.py
and tests/test_u15.py (every registered group).
"""

import json

import numpy as np
import pandas as pd
import pytest

import features.tf_state as TF
from data.exo import parse_cboe_close
from families import run as R
from families import test as T
from families.spec import STATE_SPLITS, TERCILE_SPLITS, U26_SPLITS
from features.groups import causal_modwt
from features.kernels import prior_lag1_autocorr, session_layout
from tests.test_features import group_frame, synthetic_exo
from tests.test_u15 import CFG5, bars_on, frame, utc


def numpy_power(x: np.ndarray, k: int) -> float:
    m = len(x)
    return abs(np.sum(x * np.exp(-2j * np.pi * k * np.arange(m) / m))) ** 2 / m


def ar1(n: int, phi: float, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0, 1e-3, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def sessions_from_returns(r: np.ndarray, per: int = 78, start: str = "2024-01-02", gaps=None) -> pd.DataFrame:
    """5Min bars of full sessions whose inside-session close → close log returns are r (per − 1 per session); each
    session opens at its own level (`gaps`: the overnight log gap per session, random by default)."""
    n_sess = len(r) // (per - 1)
    days = pd.bdate_range(start, periods=n_sess)
    gaps = np.random.default_rng(99).normal(0, 0.01, n_sess) if gaps is None else gaps
    closes, level = [], 0.0
    for d in range(n_sess):
        level += gaps[d]
        closes.append(level + np.r_[0.0, np.cumsum(r[d * (per - 1) : (d + 1) * (per - 1)])])
    close = 100 * np.exp(np.concatenate(closes))
    idx = bars_on(*[f"{d.date()} {9 + (30 + 5 * i) // 60:02d}:{(30 + 5 * i) % 60:02d}" for d in days
                    for i in range(per)])  # fmt: skip
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e5}, index=idx)


# ── fastgoertzel ─────────────────────────────────────────────────────────────


def test_goertzel_power_matches_a_numpy_dft_on_1000_random_windows():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(1000):
        m = int(rng.integers(20, 600))
        x = rng.standard_normal(m) * 10 ** rng.uniform(-4, 1)
        k = int(rng.integers(1, (m - 1) // 2 + 1))
        a, b = TF.goertzel_power(x, k), numpy_power(x, k)
        worst = max(worst, abs(a - b) / max(b, 1e-300 + 1e-12 * np.sum(x * x)))
    assert worst < 1e-8


@pytest.mark.parametrize("k", [0, 50, 2.5, -1])
def test_goertzel_power_refuses_anything_but_a_fourier_bin(k):
    with pytest.raises(ValueError, match="bin k"):
        TF.goertzel_power(np.random.default_rng(1).standard_normal(100), k)


def test_periodogram_obeys_parseval():
    x = np.random.default_rng(2).standard_normal(385)
    x -= x.mean()
    assert np.isclose(2 * TF.periodogram(x).sum(), np.sum(x * x), rtol=1e-10)  # odd M: no Nyquist bin


# ── the state on processes with known spectra ────────────────────────────────


def test_spectral_state_of_white_noise():
    out = np.array([TF.spectral_state(np.random.default_rng(s).standard_normal(385)) for s in range(200)])
    hf, lf, tilt, flat = out.mean(axis=0)
    assert abs(hf - 1) < 0.02 and abs(lf - 1) < 0.05 and abs(tilt) < 0.05
    assert abs(flat - np.exp(-np.euler_gamma)) < 0.02  # E[log I] − log E[I] = −γ for exponential ordinates


def test_tilt_reads_reversal_and_continuation():
    rev = np.mean([TF.spectral_state(ar1(385, -0.3, s))[2] for s in range(50)])
    cont = np.mean([TF.spectral_state(ar1(385, 0.3, s))[2] for s in range(50)])
    assert rev > 0.3 and cont < -0.3


def test_bands_cover_the_stated_periods():
    m = 385
    k = np.arange(1, (m - 1) // 2 + 1)
    hf, lf = k[TF._band(m, TF.HF_PERIODS)], k[TF._band(m, TF.LF_PERIODS)]
    assert np.all((m / hf >= 2) & (m / hf <= 6)) and np.all((m / lf >= 12) & (m / lf <= 78))
    assert hf[0] == np.ceil(m / 6) and hf[-1] == k[-1] and lf[0] == np.ceil(m / 78) and lf[-1] == m // 12


def test_haar_modwt_white_noise_variance_is_sigma2_over_2j():
    x = pd.Series(np.random.default_rng(3).standard_normal(200_000))
    details, _ = causal_modwt(x, TF.MODWT_FILTER, TF.MODWT_LEVELS)
    for j, w in enumerate(details, start=1):
        assert abs(np.nanmean(w.to_numpy() ** 2) * 2**j - 1) < 0.03


def test_session_state_is_scale_free_gap_blind_and_reads_previous_sessions_only():
    r = np.random.default_rng(4).normal(0, 1e-3, 40 * 77)
    df = sessions_from_returns(r)
    sess = session_layout(df.index, 5)[0]
    base = TF.tf_state_sessions(df["close"].to_numpy(), sess)
    assert base.iloc[: TF.TF_SESSIONS, :4].isna().all().all() and base.iloc[TF.TF_SESSIONS :, :-1].notna().all().all()
    # the overnight gaps are excluded: other gaps, same state
    other = sessions_from_returns(r, gaps=np.random.default_rng(5).normal(0, 0.05, 40))
    pd.testing.assert_frame_equal(TF.tf_state_sessions(other["close"].to_numpy(), sess), base, rtol=1e-9)
    # scale-free: returns × 10 leave every column but vov (a log, shifted) unchanged
    big = TF.tf_state_sessions(sessions_from_returns(10 * r)["close"].to_numpy(), sess)
    pd.testing.assert_frame_equal(big, base, rtol=1e-7, atol=1e-9)
    # session 20's bars move rows 21 … 25 (the spectral window) and later, never rows <= 20
    r2 = r.copy()
    r2[20 * 77 : 21 * 77] = np.random.default_rng(6).normal(0, 3e-3, 77)
    moved = TF.tf_state_sessions(sessions_from_returns(r2)["close"].to_numpy(), sess)
    pd.testing.assert_frame_equal(moved.iloc[:21], base.iloc[:21])
    assert not np.isclose(moved.loc[21, "tilt"], base.loc[21, "tilt"])
    assert np.isclose(moved.loc[26, "tilt"], base.loc[26, "tilt"])  # out of the 5-session window again
    np.testing.assert_array_equal(base["ac1"], prior_lag1_autocorr(df["close"].to_numpy(), sess, TF.TF_SESSIONS))


def test_modwt_energy_reads_continuation_at_long_scales():
    sess_r = ar1(60 * 77, 0.3, 7)
    df = sessions_from_returns(sess_r)
    st = TF.tf_state_sessions(df["close"].to_numpy(), session_layout(df.index, 5)[0]).dropna()
    assert (st["modwt_e5"] > st["modwt_e1"]).mean() > 0.9 and st["modwt_e5"].mean() > 0.3


def test_group_columns_are_broadcast_per_session():
    df = sessions_from_returns(np.random.default_rng(8).normal(0, 1e-3, 30 * 77))
    f = group_frame("tf_state", df, CFG5, None)
    assert list(f.columns) == [f"tf_state__{c}" for c in TF.COLUMNS]
    assert (f.groupby(df.index.date).nunique(dropna=False) <= 1).all().all()


# ── SKEW / VVIX ──────────────────────────────────────────────────────────────


def test_parse_cboe_close():
    out = parse_cboe_close("DATE,SKEW\n01/02/2024,140.5\n01/03/2024,141.0\n", "SKEW")
    assert list(out.columns) == ["value"] and out["value"].tolist() == [140.5, 141.0]
    with pytest.raises(ValueError, match="unexpected CBOE columns"):
        parse_cboe_close("DATE,OPEN,CLOSE\n01/02/2024,1,2\n", "SKEW")


def test_skew_and_vvix_are_not_visible_before_their_available_at():
    df = sessions_from_returns(np.random.default_rng(9).normal(0, 1e-3, 30 * 77), start="2024-02-01")
    exo = synthetic_exo(df.index)
    day = pd.Timestamp("2024-02-13")
    for name in ("cboe/SKEW", "cboe/VVIX"):
        f = exo[name].copy()
        f.loc[day, "value"] = 999.0
        exo[name] = f
    out = group_frame("vol_state", df, CFG5, None, exo=exo)
    for c in ("vol_state__skew", "vol_state__vvix"):
        assert not (out.loc["2024-02-13", c] == 999.0).any()
        assert out.loc["2024-02-14 09:30", c] == 999.0
    vix = exo["cboe/VIX"]["value"]
    assert np.isclose(out.loc["2024-02-14 09:30", "vol_state__vvix_vix"], 999.0 / vix.loc[day])


def test_vvix_vix_pairs_common_dates_only():
    days = ["2024-05-24", "2024-05-27", "2024-05-28"]  # 05-27: a VIX-only holiday row
    vix = frame(days, [12.0, 30.0, 13.0], [utc(f"{d} 16:20") for d in days])
    vvix = frame([days[0], days[2]], [90.0, 91.0], [utc(f"{days[0]} 16:20"), utc(f"{days[2]} 16:20")])
    from features.exo_align import Exo

    e = Exo({"cboe/VIX": vix, "cboe/VVIX": vvix})
    np.testing.assert_allclose(e.ratio("cboe/VVIX", "cboe/VIX", bars_on("2024-05-28 09:30")), [90 / 12])


# ── state splits ─────────────────────────────────────────────────────────────


def test_u26_splits_are_known_state_splits():
    assert set(U26_SPLITS) <= set(STATE_SPLITS) and len(U26_SPLITS) == len(TF.COLUMNS) + 3 + 1
    assert set(TERCILE_SPLITS.values()) == set(TF.COLUMNS) | {"skew", "vvix", "vvix_vix"}


def test_gamma_sign_and_tercile_groups():
    days = pd.bdate_range("2024-01-01", periods=6)
    state = pd.DataFrame({"ac1": [0.1, -0.2, 0.0, np.nan, -0.01, 0.3],
                          "tilt_rank": [0.1, 0.34, 0.5, 0.67, 1.0, np.nan]}, index=days)  # fmt: skip
    g = T._groups("gamma_sign", days, state, None)
    assert g.tolist() == ["ac1 >= 0 (short gamma)", "ac1 < 0 (long gamma)", "ac1 >= 0 (short gamma)", "n/a",
                          "ac1 < 0 (long gamma)", "ac1 >= 0 (short gamma)"]  # fmt: skip
    assert T._groups("tilt_tercile", days, state, None).tolist() == ["t1 low", "t2", "t2", "t3 high", "t3 high", "n/a"]
    assert T._groups("vov_tercile", days, state, None) is None  # its input is missing: reported unavailable


class TFSource:
    """Full-session 5Min random-walk bars per symbol and CBOE closes (SKEW, VVIX, VIX) on business days."""

    def bars(self, symbol, timeframe, start, end, *, allow_holdout):
        n = len(pd.bdate_range(start, end, inclusive="left"))
        r = np.random.default_rng(sum(map(ord, symbol))).normal(0, 1e-3, n * 77)
        return sessions_from_returns(r, start=start)

    def exo(self, source, name, start, end, *, allow_holdout):
        idx = pd.bdate_range(start, end, inclusive="left", name="date")
        v = {"VIX": 15.0, "VVIX": 90.0}.get(name, 140.0) + np.arange(len(idx)) % 7
        return pd.DataFrame({"value": v}, index=idx)


def test_load_state_ranks_are_known_at_the_open():
    start, end = "2023-01-02", "2024-06-01"
    st = R.load_state(TFSource(), start, end, ["SPY", "QQQ"], "5Min", list(U26_SPLITS))
    for c in [*(f"{c}_rank" for c in TF.COLUMNS), "skew_rank", "vvix_rank", "vvix_vix_rank", "ac1"]:
        assert c in st and st[c].notna().sum() > 100, c
    # the exo rank is the previous observation's: shifting the series by one day shifts the rank by one day
    close = R._cboe_close(TFSource(), "skew", start, end)
    days = st.index
    expect = R.trailing_rank(close).reindex(days.union(close.index)).ffill().shift(1).reindex(days)
    pd.testing.assert_series_equal(st["skew_rank"], expect, check_names=False)
    # tf_state: the instruments' mean, then its own trailing rank
    per = [TF.tf_state_days(TFSource().bars(s, "5Min", start, end, allow_holdout=False)) for s in ("SPY", "QQQ")]
    mean = (per[0]["tilt"] + per[1]["tilt"]) / 2
    pd.testing.assert_series_equal(st["tilt_rank"], R.trailing_rank(mean).reindex(days), check_names=False)


def _stored(tmp_path, h: pd.Series, groups_year: list[dict]) -> None:
    d = tmp_path / "GX"
    d.mkdir()
    result = {"id": "GX", "window": ["2023-01-02", "2024-06-01"], "instruments": ["SPY", "QQQ"],
              "spec": {"timeframe": "5Min"}, "evaluation": {"verdict": {"p": 0.5}},
              "described": {"state_splits": {"year": {"groups": groups_year}, "gamma_sign": {"unavailable": True}}}}  # fmt: skip
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    pd.DataFrame({"headline": h, "benchmark": h * 0.5, "rf": 1e-5}).to_csv(d / "streams.csv")


def test_add_splits_adds_rerenders_and_checks_the_stored_stream(tmp_path, monkeypatch):
    import families.report as REP

    monkeypatch.setattr(REP, "write_family_report", lambda result, out: {"md": out / "report.md"})
    days = pd.bdate_range("2023-01-02", "2024-06-01", inclusive="left")
    h = pd.Series(np.random.default_rng(10).normal(0, 1e-3, len(days)), index=days)
    year = T.split_table(T.excess(h, pd.Series(1e-5, index=days)), T._groups("year", days, None, None))
    _stored(tmp_path, h, year)
    res = R.add_splits("GX", list(U26_SPLITS), tmp_path, source=TFSource(), check=("year",))
    saved = json.loads((tmp_path / "GX" / "result.json").read_text(encoding="utf-8"))
    ss = saved["described"]["state_splits"]
    assert set(ss) == {"year", *U26_SPLITS} and ss["year"]["groups"] == json.loads(json.dumps(year))
    assert all("groups" in ss[k] for k in U26_SPLITS)
    assert saved["evaluation"] == {"verdict": {"p": 0.5}} and saved["splits_added"][0]["checked"] == ["year"]
    assert res["splits_added"][0]["splits"] == list(U26_SPLITS)


def test_add_splits_refuses_a_stream_that_does_not_reproduce_the_stored_split(tmp_path):
    days = pd.bdate_range("2023-01-02", "2024-06-01", inclusive="left")
    h = pd.Series(np.random.default_rng(11).normal(0, 1e-3, len(days)), index=days)
    year = T.split_table(T.excess(h, pd.Series(1e-5, index=days)), T._groups("year", days, None, None))
    year[0]["mean_bp"] += 0.01
    _stored(tmp_path, h, year)
    before = (tmp_path / "GX" / "result.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="not reproduced"):
        R.add_splits("GX", ["gamma_sign"], tmp_path, source=TFSource(), check=("year",))
    assert (tmp_path / "GX" / "result.json").read_text(encoding="utf-8") == before
    with pytest.raises(ValueError, match="unknown"):
        R.add_splits("GX", ["nope"], tmp_path, source=TFSource())
