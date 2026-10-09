"""
U12 adversarial review: regression tests for the review's findings (S1 extra train-fit sizers crashing mid-run, S2
shipped specs refused, M1 an unchecked threshold length), written by the reviewer and kept after the fixes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.spec import load_spec
from features.groups import cusum_state
from tests.test_runconfig import synthetic_daily
from utils.config import RunConfig
from wfo.wfo_engine import run_wfo

ROOT = Path(__file__).resolve().parent.parent
SMALL_XGB = {k: {**getattr(RunConfig(), k), "n_estimators": 50} for k in ("CLF_PARAMS", "REG_PARAMS", "META_PARAMS")}
SPECS = sorted((ROOT / "experiments" / "specs").glob("*.yaml"))


def test_extra_train_fit_sizer_with_the_default_zoo_switches_does_not_crash_mid_run():
    """`run_wfo(sizers=...)` (what `python -m sizing.compare` does with its default sizer list) fits `ecdf` beside
    cfg.SIZER, but RunConfig only validates cfg.SIZER. With the new defaults (rolling, reuse, fixed logit_l2 C) fold 3,
    the first rolling fold, holds no OOF predictions and `oof_meta_prob` raises RuntimeError (not the ZooFitError that
    `fold_sizers` catches), killing the whole run. Passing = the run completes (or is refused before the first fit)."""
    cfg = RunConfig.for_timeframe(
        "1Day", PRIMARY="wavelet_trend", META_MODEL="logit_l2", META_TRAIN="oof", INITIAL_TRAIN=700, VAL=350, TEST=100,
        MIN_TRAIN_EVENTS=50, MIN_VAL_EVENTS=30, **SMALL_XGB,
    )  # fmt: skip
    assert (cfg.CALIBRATION, cfg.OOF_META, cfg.SIZER) == ("rolling", "reuse", "fixed")  # the defaults, nothing pinned
    sig = run_wfo(synthetic_daily(1400), cfg, sizers=("fixed", "ecdf"))
    assert "bet_size:ecdf" in sig


def test_compiled_cusum_state_still_rejects_a_threshold_shorter_than_the_prices():
    """The pre-U12 Python loop raised IndexError at h[len(h)]; the numba kernel has no bounds check and reads past the
    end of the threshold array, returning garbage (inf / -1.8e268 states) with no error. Passing = an exception."""
    close = pd.Series([100, 101, 99, 103, 98, 105, 97, 106.0], index=pd.date_range("2020", periods=8, freq="D"))
    short = pd.Series([0.02] * 5, index=close.index[:5])
    with pytest.raises((IndexError, ValueError)):
        pos, _ = cusum_state(close, short)
        pytest.fail(f"no error; garbage state returned: max |pos| = {np.nanmax(np.abs(pos)):.3g}")


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_shipped_spec_files_still_load(path):
    """The U11 stage B3 / C / E specs (sizer ecdf + a zoo meta-model, no CALIBRATION / OOF_META override) are refused by
    the new RunConfig validation, so they cannot be loaded or re-run any more. Passing = every shipped spec loads."""
    load_spec(path)
