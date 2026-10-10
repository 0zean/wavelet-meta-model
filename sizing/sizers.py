"""
Bet sizers (SPEC §7, U7): calibrated meta-probability p → bet size m ∈ [0, 1] (the side comes from the primary).

Every sizer returns m = 0 for p < τ (`META_THRESH`) and is non-decreasing in p, except `rule_size` (SPEC §16), which
passes the primary's own size hint (its `magnitude` column, in [0, 1]) through on every approved bet. Sizers that learn from data
(`ecdf`, `kelly_capped`) are fit per WFO fold on the **train-window** out-of-fold meta-probabilities of the
meta-model's fitting events (`models.meta_model.oof_meta_prob`) and those events' side returns — never on test
events. A sizer that cannot be fit (no approved OOF events, no win or no loss) raises `SizerFitError`; the WFO then
takes no trades in that fold and records it (no silent fallback).

Post-processing (active-bet averaging, then `discretize`) happens in the backtest (wfo/backtest.py).
"""

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from scipy.stats import norm

from utils.config import RunConfig

REGISTRY: dict[str, type["Sizer"]] = {}


class SizerFitError(ValueError):
    """The sizer's train-window inputs cannot define it (the fold then takes no trades)."""


def sizer(name: str) -> Callable:
    def register(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return register


def make_sizer(name: str, cfg: RunConfig) -> "Sizer":
    if name not in REGISTRY:
        raise ValueError(f"unknown sizer {name!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)


def discretize(m, step: float) -> np.ndarray:
    """m ← round(|m|/step)·step with m's sign, clipped to [−1, 1]; step 0 leaves m unchanged."""
    m = np.asarray(m, dtype=float)
    if step == 0:
        return m.copy()
    return np.sign(m) * np.minimum(np.round(np.abs(m) / step) * step, 1.0)


class Sizer:
    name: str
    needs_train: ClassVar[bool] = False  # fit() uses the train-window OOF probabilities / returns

    def __init__(self, cfg: RunConfig):
        self.tau = cfg.META_THRESH
        self.cfg = cfg

    def fit(self, p_oof, ret) -> "Sizer":
        """p_oof: OOF meta-probabilities of the fitting events; ret: their side returns (gross, barrier exit)."""
        return self

    def _m(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def size(self, p, hint=None) -> np.ndarray:
        """Bet size per event from its meta-probability p; `hint` = the primary's magnitude (read by rule_size)."""
        p_in = np.asarray(p)
        # threshold in p's own dtype, exactly as meta_predict's trade_signal (float32 legacy probabilities)
        approved = p_in >= self.tau
        p = p_in.astype(float)
        if np.isnan(p).any():
            raise ValueError(f"{self.name}: NaN meta-probability")
        self._hint = hint
        return np.where(approved, np.clip(self._m(p), 0.0, 1.0), 0.0)


@sizer("fixed")
class Fixed(Sizer):
    """All-in on every approved bet (pre-U7 behaviour)."""

    def _m(self, p):
        return np.ones_like(p)


@sizer("rule_size")
class RuleSize(Sizer):
    """m = the primary's magnitude (its rule's size hint in [0, 1]) on every approved bet; equals `fixed` when the
    magnitude is 1."""

    def _m(self, p):
        if self._hint is None:
            raise ValueError("rule_size needs the primary's magnitude (size(p, hint=magnitude))")
        m = np.asarray(self._hint, dtype=float)
        if m.shape != p.shape or np.isnan(m).any() or (m < 0).any() or (m > 1).any():
            raise ValueError("rule_size: the primary's magnitude must be in [0, 1] for every event")
        return m


@sizer("linear")
class Linear(Sizer):
    """m = (p − τ)/(1 − τ)."""

    def _m(self, p):
        return (p - self.tau) / (1.0 - self.tau)


@sizer("ldp_sigmoid")
class LdpSigmoid(Sizer):
    """López de Prado (2018) §10.3 for two outcomes: z = (p − ½)/√(p(1−p)), m = 2Φ(z) − 1 (≤ 0 below ½)."""

    def _m(self, p):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        return 2.0 * norm.cdf(z) - 1.0  # p ∈ {0, 1} → z = ∓inf → m = ∓1, clipped by size()


@sizer("ecdf")
class Ecdf(Sizer):
    """m = F̂(p), F̂ = right-continuous ECDF of the train-window OOF probabilities of approved events (p ≥ τ)."""

    needs_train = True

    def fit(self, p_oof, ret):
        p_oof = np.asarray(p_oof, dtype=float)
        self.ref_ = np.sort(p_oof[p_oof >= self.tau])
        if self.ref_.size == 0:
            raise SizerFitError("ecdf: no train-window OOF probability reaches META_THRESH")
        return self

    def _m(self, p):
        return np.searchsorted(self.ref_, p, side="right") / self.ref_.size


@sizer("kelly_capped")
class KellyCapped(Sizer):
    """
    Fractional Kelly: f* = p − (1 − p)/b, m = λ·f* (λ = KELLY_FRACTION); b = mean win / mean loss of the
    train-window OOF-approved events' net returns (side return − round-trip SLIPPAGE_PCT; win ⇔ net > 0, which is
    the meta-label when META_MIN_RET is the default round-trip cost).
    """

    needs_train = True

    def fit(self, p_oof, ret):
        p_oof, ret = np.asarray(p_oof, dtype=float), np.asarray(ret, dtype=float)
        if p_oof.shape != ret.shape:
            raise ValueError("kelly_capped: p_oof and ret must align")
        net = ret[p_oof >= self.tau] - 2 * self.cfg.SLIPPAGE_PCT
        win, loss = net[net > 0], -net[net <= 0]
        if win.size == 0 or loss.size == 0 or loss.mean() == 0:
            raise SizerFitError(f"kelly_capped: need approved OOF wins and losses (got {win.size} / {loss.size})")
        self.b_ = float(win.mean() / loss.mean())
        return self

    def _m(self, p):
        return self.cfg.KELLY_FRACTION * (p - (1.0 - p) / self.b_)
