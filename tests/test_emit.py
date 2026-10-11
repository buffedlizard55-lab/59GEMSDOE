"""The allocator must be self-consistent and must beat trivial allocations.

These are the regressions found in this session:
  IR-57-BAR-01    the marginal test was off by a factor of D and ran to its cap
  IR-57-KCLIP-01  E[k] is a sum and can exceed 1, which flipped the bar negative
                  and let zero-credit dots through (duplicate acceptances)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems59.emit import expected_credit, greedy_allocate
from gems59.metric import ALPHA, dti_binary


def _field(seed=0, n_truth=400, side=400):
    rng = np.random.default_rng(seed)
    truth = np.zeros((side, side), bool)
    ty = rng.integers(20, side - 20, n_truth)
    tx = rng.integers(20, side - 20, n_truth)
    truth[ty, tx] = True
    p = np.zeros((side, side), np.float32)
    p[ty, tx] = 0.6
    p[ty, tx + 1] = 0.25
    p[rng.integers(0, side, 3000), rng.integers(0, side, 3000)] = 0.02
    return truth, p, np.ones((side, side), bool)


def test_dot_count_equals_emitted_sum():
    """IR-57-KCLIP-01: duplicate acceptances made n_dots overcount by 2.4x."""
    truth, p, allowed = _field()
    a = greedy_allocate(p, allowed, k_truth=float(truth.sum()))
    assert a.n_dots == int(a.emitted.sum())
    assert a.n_dots < 5000                    # the buggy version emitted 13,574


def test_expected_credit_is_the_kernel_convolution():
    p = np.zeros((21, 21), np.float32)
    p[10, 10] = 1.0
    ek = expected_credit(p)
    assert abs(ek[10, 10] - 1.0) < 1e-6
    assert abs(ek[10, 11] - 2 / 3) < 1e-6
    assert abs(ek[10, 13] - 0.0) < 1e-12      # 3 px away is outside the kernel
    assert abs(ek.sum() - 9.380298) < 1e-4    # total kernel mass


def test_allocator_terminates_and_beats_trivial_baselines():
    truth, p, allowed = _field()
    a = greedy_allocate(p, allowed, k_truth=float(truth.sum()))
    d_greedy = dti_binary(a.emitted, truth)["dti"]
    d_all = dti_binary(allowed, truth)["dti"]
    rng = np.random.default_rng(1)
    rnd = np.zeros_like(allowed)
    rnd[rng.integers(0, 400, 400), rng.integers(0, 400, 400)] = True
    d_rnd = dti_binary(rnd, truth)["dti"]
    assert d_greedy > 5 * d_all
    assert d_greedy > 5 * d_rnd
    assert a.rounds < 400, "hit the round cap -- not converged"


def test_surrogate_dti_is_monotone_over_the_trace():
    """Freezing the bar inside a round used to let DTI-lowering dots through."""
    truth, p, allowed = _field()
    a = greedy_allocate(p, allowed, k_truth=float(truth.sum()), trace_every=5)
    d = [t["dti_est"] for t in a.trace]
    assert len(d) > 5
    assert all(d[i + 1] >= d[i] - 1e-12 for i in range(len(d) - 1))


def test_empty_allowed_gives_empty_allocation():
    p = np.full((30, 30), 0.5, np.float32)
    a = greedy_allocate(p, np.zeros((30, 30), bool), k_truth=100.0)
    assert a.n_dots == 0 and not a.emitted.any()


def test_no_dot_is_placed_outside_allowed():
    truth, p, allowed = _field(side=120, n_truth=40)
    allowed[:, :60] = False
    a = greedy_allocate(p, allowed, k_truth=float(truth.sum()))
    assert not a.emitted[:, :60].any()
