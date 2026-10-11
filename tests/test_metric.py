"""The DTI implementation must agree with a literal transcription of the metric.

Also pins the two facts the whole emission rule rests on:
  DTI(binary dots) == TP_w / (alpha*n + beta*|G|)
  adding one dot raises DTI iff its realised kernel credit k > alpha*DTI
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems59.metric import (ALPHA, BETA, dti_binary, dti_bruteforce, dti_exact,
                           dti_from_components, kernel, marginal_accept,
                           marginal_inclusion_threshold, offsets)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_kernel_values_on_the_100m_lattice():
    assert kernel(0.0) == 1.0
    assert abs(kernel(1.0) - 2 / 3) < 1e-12
    assert abs(kernel(np.sqrt(2)) - (1 - np.sqrt(2) / 3)) < 1e-12
    assert kernel(3.0) == 0.0
    assert kernel(10.0) == 0.0


def test_offsets_are_exactly_the_subradius_lattice():
    dy, dx, kw = offsets(3.0)
    # 1 at d=0, 4 at d=1, 4 at d=sqrt2, 4 at d=2, 8 at d=sqrt5, 4 at d=2sqrt2
    assert len(dy) == 25
    assert abs(kw.sum() - 9.380298) < 1e-5
    for j, i, k in zip(dy, dx, kw):
        assert abs(k - max(1 - np.hypot(j, i) / 3, 0)) < 1e-12
        assert np.hypot(j, i) < 3.0


def test_binary_matches_bruteforce():
    r = _rng(1)
    truth = np.zeros((40, 40), bool)
    truth[r.integers(0, 40, 25), r.integers(0, 40, 25)] = True
    pred = np.zeros((40, 40), bool)
    pred[r.integers(0, 40, 30), r.integers(0, 40, 30)] = True
    a = dti_binary(pred, truth)
    b = dti_bruteforce(pred.astype(float), truth)
    assert abs(a["dti"] - b["dti"]) < 1e-9
    assert abs(a["tp"] - b["tp"]) < 1e-9
    assert abs(a["fp"] - b["fp"]) < 1e-9


def test_exact_matches_bruteforce_on_soft_predictions():
    r = _rng(2)
    truth = np.zeros((30, 30), bool)
    truth[r.integers(0, 30, 15), r.integers(0, 30, 15)] = True
    pred = np.zeros((30, 30))
    idx = (r.integers(0, 30, 20), r.integers(0, 30, 20))
    pred[idx] = r.random(20)
    a = dti_exact(pred, truth)
    b = dti_bruteforce(pred, truth)
    assert abs(a["dti"] - b["dti"]) < 1e-9


def test_denominator_uses_M_not_T_and_the_collapse_only_holds_when_T_equals_M():
    """FP_w = n - M with M summed over dots, NOT n - TP_w.

    A cluster of dots on one truth cell has M > T, so the naive
    ``T / (alpha*n + beta*K)`` form is wrong in general.  This test pins the
    distinction that the shared template's prose gets wrong.
    """
    truth = np.zeros((30, 30), bool)
    truth[15, 15] = True                       # a single truth cell
    pred = np.zeros((30, 30), bool)
    pred[15, 15] = True
    pred[15, 16] = True                        # two dots, one truth cell
    res = dti_binary(pred, truth)
    assert abs(res["tp"] - 1.0) < 1e-12        # T saturates at 1 for the truth cell
    # the second dot sits 1 px away, so its self-credit is 2/3 and it contributes
    # 1 - 2/3 = 1/3 of false positive; FP_w = n - M exactly
    assert abs(res["fp"] - 1 / 3) < 1e-12
    n, K = 2, 1
    naive = res["tp"] / (ALPHA * n + BETA * K)
    assert abs(res["dti"] - naive) > 1e-6      # the naive collapse is measurably wrong
    # exact form with M = 1 + 2/3
    M = 1.0 + 2 / 3
    exact = res["tp"] / (ALPHA * (res["tp"] + n - M) + BETA * K)
    assert abs(res["dti"] - exact) < 1e-12


def test_denominator_collapse_holds_for_spread_out_dots():
    """When dots and truth cells pair up one-to-one, T == M and the simple form holds."""
    truth = np.zeros((60, 60), bool)
    pred = np.zeros((60, 60), bool)
    for k in range(12):                        # widely separated pairs
        y, x = 4 + 5 * (k % 6), 4 + 9 * (k // 6)
        truth[y, x] = True
        pred[y, x + 1] = True                  # each dot 1 px from its own truth cell
    res = dti_binary(pred, truth)
    n, K = int(pred.sum()), int(truth.sum())
    assert abs(res["dti"] - res["tp"] / (ALPHA * n + BETA * K)) < 1e-9


def test_marginal_bar_is_alpha_times_dti():
    """For a non-redundant dot, dDTI > 0 exactly when k > alpha*DTI."""
    truth = np.zeros((50, 50), bool)
    truth[10:20, 10:20] = True
    base = np.zeros((50, 50), bool)
    base[10, 10] = True                      # start from a state with DTI > 0
    r0 = dti_binary(base, truth)
    assert r0["dti"] > 0
    bar = marginal_inclusion_threshold(r0["dti"])
    # a dot 2 px from an uncovered truth pixel has k = 1/3 > bar -> must help
    better = base.copy(); better[18, 20] = True
    assert 1 / 3 > bar
    assert dti_binary(better, truth)["dti"] > r0["dti"]
    # a dot 3 px from every truth pixel has k = 0 -> must hurt
    worse = base.copy(); worse[40, 40] = True
    assert dti_binary(worse, truth)["dti"] < r0["dti"]
    # and the boundary is exactly alpha*DTI: scan a synthetic k against the rule
    assert marginal_accept(0.05, 0.05, r0["tp"], 1.0, 1, int(truth.sum())) == (0.05 > bar)
    assert marginal_accept(0.02, 0.02, r0["tp"], 1.0, 1, int(truth.sum())) == (0.02 > bar)


def test_masks_are_respected():
    truth = np.zeros((20, 20), bool); truth[5, 5] = True
    pred = np.zeros((20, 20), bool); pred[5, 6] = True
    full = dti_binary(pred, truth)
    # masking the truth pixel out of the scored domain leaves no truth at all
    known = np.zeros((20, 20), bool); known[5, 5] = True
    masked = dti_binary(pred, truth, known=known)
    assert masked["n_truth"] == 0
    assert full["n_truth"] == 1


def test_marginal_accept_agrees_with_direct_dti_recomputation():
    """``marginal_accept`` must reproduce DTI(T+dT)/(D+dD) > DTI exactly.

    This is the test that catches the off-by-a-factor-of-D error recorded as
    ``IR-57-BAR-01``, which made the greedy allocation run away to its cap.
    """
    T, M, n, K = 500.0, 520.0, 900, 4000
    for dT, k in [(0.0, 0.0), (0.05, 0.05), (0.5, 0.5), (1.0, 1.0),
                  (0.02, 0.6), (0.30, 0.34), (0.001, 0.001)]:
        D0 = ALPHA * (T + n - M) + BETA * K
        D1 = ALPHA * (T + dT + (n + 1) - (M + k)) + BETA * K
        want = (T + dT) / D1 > T / D0
        assert marginal_accept(dT, k, T, M, n, K) == want, (dT, k)
    # a dot with zero marginal truth credit can never help, whatever its self-credit
    assert marginal_accept(0.0, 0.9, T, M, n, K) is False
    # and the non-redundant bar is exactly alpha * DTI
    dti = T / (ALPHA * (T + n - M) + BETA * K)
    just_above = ALPHA * dti * 1.0000001
    just_below = ALPHA * dti * 0.9999999
    assert marginal_accept(just_above, just_above, T, M, n, K) is True
    assert marginal_accept(just_below, just_below, T, M, n, K) is False


def test_out_of_range_predictions_are_rejected():
    truth = np.zeros((10, 10), bool)
    bad = np.full((10, 10), 1.5)
    try:
        dti_exact(bad, truth)
    except ValueError:
        return
    raise AssertionError("dti_exact accepted a prediction above 1")


def test_dti_from_components_matches_closed_form():
    assert abs(dti_from_components(10, 20, 30) - 10 / (10 + 0.2 * 20 + 0.8 * 30)) < 1e-12
