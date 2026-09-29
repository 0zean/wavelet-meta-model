"""U5 validation toolkit: purged splitters, CPCV paths, PBO, PSR/DSR/MinTRL, probabilistic scoring (SPEC §5–6)."""

import math
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

from utils.config import RunConfig
from validation.pbo import pbo
from validation.purged_cv import CombinatorialPurgedCV, PurgedKFold, embargo_bars
from validation.scoring import (
    brier,
    neg_log_loss,
    purged_cv_predict,
    purged_cv_score,
    score_report,
    selection_score,
)
from validation.stats import (
    dsr,
    expected_max_sharpe,
    min_track_record_length,
    psr,
    return_moments,
    sharpe_ratio,
)

# ── Splitters ────────────────────────────────────────────────────────────────


def _random_spans(rng, n):
    t0 = np.sort(rng.integers(0, 3 * n, n))  # duplicate event bars on purpose
    t1 = t0 + rng.integers(0, 40, n)
    return t0, t1


def _brute_force_train(t0, t1, test, emb):
    """Definition, sample by sample: drop i if its span overlaps any test span, or if t0_i falls within
    `emb` bars after the end of a contiguous run of test samples."""
    test_set = set(test.tolist())
    runs = np.split(test, np.flatnonzero(np.diff(test) > 1) + 1)
    run_ends = [t1[r].max() for r in runs if r.size]
    keep = []
    for i in range(len(t0)):
        if i in test_set:
            continue
        if any(t0[i] <= t1[j] and t0[j] <= t1[i] for j in test):
            continue
        if any(b < t0[i] <= b + emb for b in run_ends):
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def _random_splitters(rng):
    emb = int(rng.integers(0, 30))
    n_groups = int(rng.integers(2, 8))
    yield PurgedKFold(int(rng.integers(2, 6)), emb)
    yield CombinatorialPurgedCV(n_groups, int(rng.integers(1, n_groups)), emb)


@pytest.mark.parametrize("seed", range(40))
def test_splitters_match_the_purge_embargo_definition(seed):
    rng = np.random.default_rng(seed)
    t0, t1 = _random_spans(rng, int(rng.integers(20, 120)))
    for cv in _random_splitters(rng):
        splits = list(cv.split(t0, t1))
        assert len(splits) == cv.n_splits
        for train, test in splits:
            assert not np.intersect1d(train, test).size
            # no train span overlaps any test span
            ov = (t0[train][:, None] <= t1[test][None, :]) & (t0[test][None, :] <= t1[train][:, None])
            assert not ov.any()
            # the purge + embargo is exact: nothing is dropped that the definition keeps
            assert np.array_equal(train, _brute_force_train(t0, t1, test, cv.embargo))


def test_embargo_follows_each_test_run_only():
    t0 = np.arange(0, 100, 10)  # 10 samples, spans [10i, 10i + 4]
    t1 = t0 + 4
    cv = CombinatorialPurgedCV(n_groups=5, k=2, embargo=10)  # groups of 2 samples
    train, test = next(iter(cv.split(t0, t1)))  # test = groups 0,1 → samples 0..3, run end t1 = 34
    assert test.tolist() == [0, 1, 2, 3]
    assert train.tolist() == [5, 6, 7, 8, 9]  # sample 4 (t0 = 40) is embargoed: 34 < 40 <= 44
    train, test = list(cv.split(t0, t1))[1]  # groups 0,2 → samples 0,1,4,5: two runs
    assert test.tolist() == [0, 1, 4, 5]
    # sample 2 (t0 20) embargoed after run 1 (end 14); 3 (t0 30) kept; 6 (t0 60) embargoed after run 2 (end 54)
    assert train.tolist() == [3, 7, 8, 9]


def test_splitter_input_errors():
    cv = PurgedKFold(3)
    with pytest.raises(ValueError, match="sorted"):
        list(cv.split(np.array([2, 1, 3]), np.array([2, 1, 3])))
    with pytest.raises(ValueError, match="t1 >= t0"):
        list(cv.split(np.array([1, 2, 3]), np.array([1, 1, 3])))
    with pytest.raises(TypeError, match="integer"):
        list(cv.split(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])))
    with pytest.raises(ValueError, match="required"):
        list(CombinatorialPurgedCV(5, 2).split(np.arange(4), np.arange(4)))
    with pytest.raises(ValueError, match="k must be"):
        CombinatorialPurgedCV(5, 5)
    with pytest.raises(ValueError, match="embargo"):
        CombinatorialPurgedCV(5, 2, embargo=-1)
    with pytest.raises(ValueError, match="embargo_pct"):
        embargo_bars(1.5, 100)
    assert embargo_bars(0.01, 1001) == 11 and embargo_bars(0.0, 500) == 0
    assert embargo_bars(0.07, 100) == 7  # 0.07 · 100 = 7.000000000000001 in floating point


# ── CPCV paths ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("n_groups", "k"), [(10, 2), (6, 3), (5, 1), (7, 4), (4, 3)])
def test_cpcv_split_count_group_frequency_and_paths(n_groups, k):
    cv = CombinatorialPurgedCV(n_groups, k)
    n = 53  # unequal group sizes
    t0 = np.arange(n)
    splits = list(cv.split(t0, t0))
    phi = math.comb(n_groups - 1, k - 1)
    assert len(splits) == math.comb(n_groups, k) == cv.n_splits
    assert cv.n_paths == phi == math.comb(n_groups, k) * k // n_groups
    groups = cv.groups(n)
    tested = [sum(np.isin(g, test).all() for _, test in splits) for g in groups]
    assert tested == [phi] * n_groups  # each group is tested in exactly φ splits
    for g in groups:  # and never partially
        assert all(np.isin(g, test).all() or not np.isin(g, test).any() for _, test in splits)

    ps = cv.path_splits()
    assert ps.shape == (phi, n_groups)
    used = {(int(s), g) for p in range(phi) for g, s in enumerate(ps[p])}
    assert len(used) == phi * n_groups  # every (split, tested group) pair used exactly once
    assert used == {(s, g) for s, c in enumerate(cv.combos) for g in c}

    # each split "predicts" (sample id, split id): every path must cover every sample once, from a split testing it
    vals = [np.c_[test, np.full(test.size, s)] for s, (_, test) in enumerate(splits)]
    paths = cv.assemble_paths(vals, n)
    assert paths.shape == (phi, n, 2)
    assert (paths[:, :, 0] == np.arange(n)).all()
    for p in range(phi):
        for g, s in enumerate(ps[p]):
            assert (paths[p, groups[g], 1] == s).all()
    # the φ paths draw each sample from φ distinct splits
    assert all(np.unique(paths[:, i, 1]).size == phi for i in range(n))


def test_cpcv_default_is_45_splits_9_paths_and_from_cfg():
    cfg = RunConfig.for_timeframe("1Day")
    cv = CombinatorialPurgedCV.from_cfg(cfg, n_bars=1000)
    assert (cv.n_groups, cv.k, cv.embargo) == (10, 2, 10)
    assert (cv.n_splits, cv.n_paths) == (45, 9)


def test_cpcv_assemble_paths_rejects_misaligned_outputs():
    cv = CombinatorialPurgedCV(4, 2)
    n = 20
    vals = [np.zeros(test.size) for _, test in cv.split(np.arange(n), np.arange(n))]
    with pytest.raises(ValueError, match="expected 6"):
        cv.assemble_paths(vals[:-1], n)
    vals[2] = np.zeros(3)
    with pytest.raises(ValueError, match="split 2"):
        cv.assemble_paths(vals, n)


# ── scikit-learn adapter ─────────────────────────────────────────────────────


def _classification_data(n=400, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (signal * X[:, 0] + rng.normal(size=n) > 0).astype(int)
    t0 = np.arange(n) * 3
    t1 = t0 + rng.integers(0, 12, n)
    w = rng.uniform(0.5, 1.5, n)
    return X, y, t0, t1, w


def test_bound_splitter_works_with_sklearn():
    X, y, t0, t1, _ = _classification_data()
    for cv in (PurgedKFold(4, embargo=5), CombinatorialPurgedCV(5, 2, embargo=5)):
        bound = cv.bind(t0, t1)
        assert bound.get_n_splits() == cv.n_splits
        assert [(a.tolist(), b.tolist()) for a, b in bound.split(X)] == [
            (a.tolist(), b.tolist()) for a, b in cv.split(t0, t1)
        ]
        scores = cross_val_score(LogisticRegression(), X, y, cv=bound, scoring="neg_log_loss")
        assert scores.shape == (cv.n_splits,) and (scores > -0.6).all()
    gs = GridSearchCV(LogisticRegression(), {"C": [0.01, 1.0]}, cv=PurgedKFold(3).bind(t0, t1), scoring="neg_log_loss")
    gs.fit(X, y)
    assert gs.best_params_["C"] == 1.0
    with pytest.raises(ValueError, match="bound to"):
        list(PurgedKFold(3).bind(t0, t1).split(X[:-1]))


# ── Scoring ──────────────────────────────────────────────────────────────────


def test_scores_match_hand_formulas():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4, 0.5])
    w = np.array([1.0, 2.0, 1.0, 0.5, 1.0])
    ll = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()
    assert neg_log_loss(y, p, w) == pytest.approx(-ll)
    assert brier(y, p, w) == pytest.approx((w * (p - y) ** 2).sum() / w.sum())
    assert selection_score("neg_log_loss", y, p, w) == pytest.approx(-ll)
    assert selection_score("brier", y, p, w) == pytest.approx(-brier(y, p, w))
    # higher is better for both selection scores
    assert selection_score("brier", y, y.astype(float)) > selection_score("brier", y, p)
    r = score_report(y, p, w)
    assert set(r) == {"neg_log_loss", "brier", "auc", "f1", "accuracy"}
    assert r["accuracy"] == pytest.approx((w * ((p >= 0.5) == y)).sum() / w.sum())
    assert np.isnan(score_report(np.ones(3, int), np.full(3, 0.7))["auc"])
    with pytest.raises(ValueError, match="unknown"):
        selection_score("auc", y, p)
    with pytest.raises(ValueError, match="binary"):
        brier(np.array([0, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="probabilities"):
        brier(np.array([0, 1]), np.array([0.1, 1.2]))


def test_purged_cv_predict_uses_only_purged_train_rows():
    X, y, t0, t1, w = _classification_data()
    cv = CombinatorialPurgedCV(6, 2, embargo=4)
    base = list(purged_cv_predict(LogisticRegression(), X, y, cv, t0, t1, w))
    assert [(f.train.tolist(), f.test.tolist()) for f in base] == [
        (a.tolist(), b.tolist()) for a, b in cv.split(t0, t1)
    ]
    rng = np.random.default_rng(1)
    for s, f in enumerate(base):
        # scramble every row the split may not learn from (test + purged + embargoed): predictions unchanged
        out = np.setdiff1d(np.arange(len(y)), f.train)
        X2, y2, w2 = X.copy(), y.copy(), w.copy()
        X2[out] = rng.normal(size=(out.size, X.shape[1])) * 10
        y2[out] = 1 - y2[out]
        w2[out] = rng.uniform(0, 5, out.size)
        again = list(purged_cv_predict(LogisticRegression(), X2, y2, cv, t0, t1, w2))[s]
        assert np.allclose(
            LogisticRegression().fit(X[f.train], y[f.train], w[f.train]).predict_proba(X[f.test])[:, 1], f.proba
        )
        assert np.allclose(
            LogisticRegression().fit(X2[f.train], y2[f.train], w2[f.train]).predict_proba(X[f.test])[:, 1], f.proba
        )
        assert again.train.tolist() == f.train.tolist()


def test_purged_cv_score_prefers_signal_and_uses_weights():
    X, y, t0, t1, w = _classification_data(signal=2.0)
    cv = PurgedKFold(5, embargo=3)
    good = purged_cv_score(LogisticRegression(), X, y, cv, t0, t1, w)
    noise = purged_cv_score(LogisticRegression(), X[:, 1:], y, cv, t0, t1, w)
    assert good.shape == (5,) and good.mean() > noise.mean() + 0.1
    assert not np.allclose(good, purged_cv_score(LogisticRegression(), X, y, cv, t0, t1))
    b = purged_cv_score(LogisticRegression(), X, y, cv, t0, t1, w, metric="brier")
    assert (b < 0).all() and (b > -0.25).all()
    with pytest.raises(ValueError, match="single class"):
        purged_cv_score(LogisticRegression(), X, np.r_[np.zeros(320, int), y[320:]], cv, t0, t1)
    with pytest.raises(ValueError, match="unknown"):
        purged_cv_score(LogisticRegression(), X, y, cv, t0, t1, metric="auc")


def test_selection_metric_config_is_validated():
    assert RunConfig().SELECTION_METRIC == "neg_log_loss"
    with pytest.raises(ValueError, match="SELECTION_METRIC"):
        RunConfig(SELECTION_METRIC="auc")
    with pytest.raises(ValueError, match="CPCV_GROUPS"):
        RunConfig(CPCV_GROUPS=3, CPCV_TEST_GROUPS=5)
    with pytest.raises(ValueError, match="PBO_BLOCKS"):
        RunConfig(PBO_BLOCKS=15)


# ── PSR / DSR / MinTRL ───────────────────────────────────────────────────────


def test_dsr_reproduces_bailey_lopez_de_prado_2014_example():
    # "A numerical example": annualized SR 2.5 over T = 1250 daily returns, N = 100 trials, annualized
    # V[SR_n] = 1/2, skew −3, kurtosis 10; 250 observations per year. Paper: SR₀ = 0.1132, DSR = 0.9004;
    # with N = 46 trials DSR = 0.9505.
    sr, var = 2.5 / np.sqrt(250), 0.5 / 250
    assert round(expected_max_sharpe(100, var), 4) == 0.1132
    assert round(dsr(sr, 100, var, 1250, skew=-3, kurt=10), 4) == 0.9004
    assert round(dsr(sr, 46, var, 1250, skew=-3, kurt=10), 4) == 0.9505
    # DSR is PSR at the deflated threshold, and falls as trials grow
    assert dsr(sr, 100, var, 1250, -3, 10) == psr(sr, 1250, -3, 10, sr_star=expected_max_sharpe(100, var))
    assert dsr(sr, 1000, var, 1250, -3, 10) < dsr(sr, 100, var, 1250, -3, 10)
    assert dsr(sr, 1, var, 1250, -3, 10) == psr(sr, 1250, -3, 10)


def test_psr_uses_raw_kurtosis():
    sr, T = 0.1, 500
    normal = norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 + sr**2 / 2))
    assert psr(sr, T, skew=0, kurt=3) == pytest.approx(normal)
    assert psr(sr, T) == pytest.approx(normal)
    assert psr(sr, T, kurt=0) != pytest.approx(normal)  # excess-kurtosis 0 would be a different answer
    # fat tails and negative skew lower the PSR of a positive SR
    assert psr(sr, T, skew=-1, kurt=8) < normal


def test_return_moments_and_sharpe():
    rng = np.random.default_rng(3)
    r = rng.normal(0.05, 1.0, 200_000)
    m = return_moments(r)
    assert m.sr == pytest.approx(r.mean() / r.std(ddof=1))
    assert m.skew == pytest.approx(0, abs=0.02) and m.kurt == pytest.approx(3, abs=0.05)
    t = rng.standard_t(6, 400_000)
    assert return_moments(t).kurt == pytest.approx(3 + 6 / (6 - 4), rel=0.15)
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_ratio(np.ones(10))
    with pytest.raises(ValueError, match="NaN"):
        sharpe_ratio(np.array([0.1, np.nan, 0.2]))


def test_psr_is_calibrated_under_the_null():
    # under a true SR of 0, P(PSR(0) > 0.95) ≈ 5% (Normal returns)
    rng = np.random.default_rng(7)
    T, reps = 250, 4000
    r = rng.normal(0, 0.01, (reps, T))
    moments = [return_moments(x) for x in r]
    hits = [psr(m.sr, m.n_obs, m.skew, m.kurt) > 0.95 for m in moments]
    assert np.mean(hits) == pytest.approx(0.05, abs=0.012)


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(11)
    for n, var in [(10, 0.5), (100, 0.25), (1000, 1.0)]:
        mc = (rng.normal(0, np.sqrt(var), (5000, n))).max(1).mean()
        assert expected_max_sharpe(n, var) == pytest.approx(mc, rel=0.03)
    assert expected_max_sharpe(1, 0.5) == 0.0
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0, 0.5)


def test_min_track_record_length_inverts_psr():
    for sr, sr_star, sk, ku, alpha in [
        (0.1, 0.0, 0.0, 3.0, 0.05),
        (0.08, 0.02, -1.0, 6.0, 0.01),
        (0.2, 0.1, 0.5, 4, 0.1),
    ]:
        n = min_track_record_length(sr, sr_star, sk, ku, alpha)
        # PSR is evaluated with n_obs − 1 in the root; MinTRL solves that continuously
        z = (sr - sr_star) * np.sqrt(n - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2)
        assert norm.cdf(z) == pytest.approx(1 - alpha)
        assert psr(sr, math.ceil(n), sk, ku, sr_star) >= 1 - alpha - 1e-12
        assert psr(sr, math.floor(n) - 1, sk, ku, sr_star) < 1 - alpha
    assert min_track_record_length(0.05, 0.05) == float("inf")
    assert min_track_record_length(-0.1) == float("inf")


# ── PBO ──────────────────────────────────────────────────────────────────────


def test_pbo_hand_example_and_rank_direction():
    # 2 blocks of 2 rows, metric = mean. Columns A, B, C have block means (3, 1), (2, 2), (1, 3):
    # the IS winner is always the OOS loser → rank 1, ω = 1/4, λ = log(1/3) in both combinations.
    M = np.array([[3, 2, 1], [3, 2, 1], [1, 2, 3], [1, 2, 3]], float)
    res = pbo(M, n_blocks=2, metric=lambda A: A.mean(0))
    assert res.n_combinations == 2 and res.pbo == 1.0
    assert np.allclose(res.logits, np.log(1 / 3)) and res.best.tolist() == [0, 2]
    # persistent performance: the IS winner is the OOS winner → rank N, λ = log(3) > 0
    P = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1]], float)
    res = pbo(P, n_blocks=2, metric=lambda A: A.mean(0))
    assert res.pbo == 0.0 and np.allclose(res.logits, np.log(3))
    # IS = block 0: A wins, then ties B out of sample → average rank 2.5 of 3 → ω = 2.5/4.
    # IS = block 1: A and B tie in sample → the lower index (A) is n*; it ranks 3 OOS.
    Q = np.array([[3, 2, 1], [3, 2, 1], [2, 2, 1], [2, 2, 1]], float)
    res = pbo(Q, n_blocks=2, metric=lambda A: A.mean(0))
    assert res.best.tolist() == [0, 0] and np.allclose(res.logits, [np.log(2.5 / 1.5), np.log(3)])


def test_pbo_median_rank_counts_half():
    # N = 3: the IS winner always lands on the OOS median rank 2 → ω = ½, λ = 0 → PBO = ½ (not 1, not 0)
    M = np.array([[3, 2, 1], [3, 2, 1], [2, 3, 1], [2, 3, 1]], float)
    res = pbo(M, n_blocks=2, metric=lambda A: A.mean(0))
    assert np.allclose(res.logits, 0) and res.pbo == 0.5


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_pbo_noise_baseline_is_one_half_for_any_n(n):
    vals = [pbo(np.random.default_rng(s).normal(0, 0.01, (320, n)), n_blocks=8).pbo for s in range(40)]
    assert abs(np.mean(vals) - 0.5) < 0.08


def test_pbo_default_sharpe_matches_explicit_metric():
    rng = np.random.default_rng(2)
    M = rng.normal(0.001, 0.01, (320, 12))
    fast = pbo(M, n_blocks=8)
    slow = pbo(M, n_blocks=8, metric=lambda A: A.mean(0) / A.std(0, ddof=1))
    assert fast.n_combinations == math.comb(8, 4)
    assert np.allclose(fast.logits, slow.logits) and np.array_equal(fast.best, slow.best)
    assert np.allclose(fast.is_perf, slow.is_perf) and np.allclose(fast.oos_perf, slow.oos_perf)


def test_pbo_noise_is_about_one_half():
    vals = [pbo(np.random.default_rng(s).normal(0, 0.01, (800, 20))).pbo for s in range(12)]
    assert pbo(np.random.default_rng(0).normal(0, 0.01, (800, 20))).n_combinations == math.comb(16, 8)
    assert 0.4 <= np.mean(vals) <= 0.6


def test_pbo_is_low_with_one_dominant_strategy():
    rng = np.random.default_rng(4)
    M = rng.normal(0, 0.01, (1000, 30))
    M[:, 17] += 0.004  # per-period SR ≈ 0.4
    res = pbo(M)
    assert res.pbo < 0.05 and (res.best == 17).mean() > 0.95
    assert res.prob_oos_loss < 0.05


def test_pbo_is_high_when_is_winners_revert():
    # every strategy earns +μ in half the blocks and −μ in the other half: the IS winner has, by construction,
    # most of its good blocks in-sample and so most of its bad ones out-of-sample.
    rng = np.random.default_rng(5)
    S, per, N = 16, 50, 24
    M = rng.normal(0, 0.01, (S * per, N))
    for j in range(N):
        good = rng.permutation(S)[: S // 2]
        sign = -np.ones(S)
        sign[good] = 1
        M[:, j] += np.repeat(sign, per) * 0.004
    res = pbo(M, n_blocks=S)
    assert res.pbo > 0.9 and res.degradation_slope < 0


def test_pbo_input_errors():
    with pytest.raises(ValueError, match="even"):
        pbo(np.zeros((100, 3)), n_blocks=5)
    with pytest.raises(ValueError, match="2 trials"):
        pbo(np.zeros((100, 1)))
    with pytest.raises(ValueError, match="rows per block"):
        pbo(np.zeros((20, 3)), n_blocks=16)
    with pytest.raises(ValueError, match="NaN"):
        pbo(np.full((64, 3), np.nan))


def test_pbo_handles_flat_strategies():
    rng = np.random.default_rng(6)
    M = rng.normal(0, 0.01, (400, 5))
    M[:, 2] = 0.0  # never trades: Sharpe 0, not NaN
    res = pbo(M, n_blocks=8)
    assert np.isfinite(res.logits).all()
    M[:, 2] = 0.1  # flat but positive: +inf (no float cancellation leaves a huge finite Sharpe)
    M[:, 3] = -0.03
    res = pbo(M, n_blocks=8)
    assert (res.best == 2).all() and np.isinf(res.is_perf).all() and np.isinf(res.oos_perf).all()
    assert res.pbo == 0.0 and np.isnan(res.degradation_slope)
    # a large common offset does not destroy the Sharpe of a varying column
    X = rng.normal(0, 1e-3, (400, 3)) + 100.0
    ref = pbo(X - 100.0 + 100.0, n_blocks=8, metric=lambda A: A.mean(0) / A.std(0, ddof=1))
    assert np.allclose(pbo(X, n_blocks=8).is_perf, ref.is_perf, rtol=1e-6)


def test_combinations_order_is_documented():
    assert CombinatorialPurgedCV(5, 2).combos == list(combinations(range(5), 2))
