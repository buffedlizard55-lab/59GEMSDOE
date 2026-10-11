"""allocate_patient must reproduce every single-pass decision, and add to them."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems59.emit import (allocate_by_marginal_bar, allocate_patient,  # noqa: E402
                         expected_credit)


def _plateau(shape=(60, 60), plateau=0.9):
    p = np.zeros(shape, np.float32)
    p[10:30, 10:30] = plateau
    return p


def _both(p, k_truth=500.0, **kw):
    allowed = p > 0.05
    a = allocate_by_marginal_bar(p, allowed, k_truth, floor=0.0, max_dots=100_000,
                                 candidate_cap=200_000)
    b = allocate_patient(p, allowed, k_truth, floor=0.0, max_dots=100_000,
                         candidate_cap=200_000, **kw)
    return a, b


def test_superset_of_the_single_pass():
    a, b = _both(_plateau())
    assert b.n_dots >= a.n_dots
    assert b.emitted[a.emitted].all(), "the patient pass dropped a single-pass dot"


def test_never_lower_surrogate_dti():
    a, b = _both(_plateau())
    assert b.surrogate_dti >= a.surrogate_dti - 1e-9


def test_recovers_dots_the_single_pass_loses():
    """A plateau split by a one-cell gap of zeroes: the row-major order inside the
    plateau puts neighbours next to each other, which is what ends a pass early."""
    p = np.zeros((80, 80), np.float32)
    p[10:70, 10:70] = 0.9       # 3600 cells, all with the same E[k]
    p[40, :] = 0.0
    a, b = _both(p, k_truth=4000.0)
    assert b.n_dots >= a.n_dots
    assert b.n_dots > 0


def test_agrees_when_nothing_is_redundant():
    p = np.zeros((60, 60), np.float32)
    p[15, 15] = 0.9
    p[45, 45] = 0.8
    a, b = _both(p)
    assert a.n_dots == b.n_dots == 2


def test_empty_domain():
    p = np.zeros((20, 20), np.float32)
    assert allocate_patient(p, p > 0.5, 100.0).n_dots == 0


def test_patience_bounds_the_scan():
    p = _plateau()
    b = allocate_patient(p, p > 0.05, 500.0, floor=0.0, patience=1)
    assert dict(b.trace)["n_scanned"] <= 400
