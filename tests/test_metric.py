"""Verification of the DW-Tversky evaluator against official and brute-force references."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems59 import metric  # noqa: E402


def test_organizer_worked_example():
    """The organizer's own numbers: TP=3.00, FP=1.89, FN=2.00, a=0.2, b=0.8 -> 0.60.

    Source: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
    ("Scoring example": TI_w = 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.60)
    """
    got = 3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00)
    assert abs(got - 0.60) < 0.005, got


def test_fast_matches_bruteforce():
    rng = np.random.default_rng(11)
    for trial in range(6):
        H, W = 18, 18
        truth = rng.random((H, W)) < 0.06
        pred = np.zeros((H, W))
        idx = rng.choice(H * W, size=int(0.5 * H * W), replace=False)
        pred.flat[idx] = rng.random(idx.size)
        a = metric.dti(pred, truth)
        b = metric.brute_force_dti(pred, truth)
        assert abs(a - b) < 1e-5, (trial, a, b)   # float32 dilation, not float64


def test_tp_plus_fn_equals_n_truth():
    rng = np.random.default_rng(12)
    H, W = 24, 24
    truth = rng.random((H, W)) < 0.08
    pred = (rng.random((H, W)) < 0.2).astype(float)
    t = metric.dti_terms(pred, truth)
    assert abs(t["tp"] + t["fn"] - t["n_truth"]) < 1e-3


def test_closed_form_identity():
    """DTI = TP / (a*TP + a*FP + b*|G|) must equal the definition."""
    rng = np.random.default_rng(13)
    H, W = 20, 20
    truth = rng.random((H, W)) < 0.05
    pred = (rng.random((H, W)) < 0.3).astype(float)
    t = metric.dti_terms(pred, truth)
    d1 = t["tp"] / (t["tp"] + 0.2 * t["fp"] + 0.8 * t["fn"] + 1e-12)
    d2 = t["tp"] / (0.2 * t["tp"] + 0.2 * t["fp"] + 0.8 * t["n_truth"] + 1e-12)
    assert abs(d1 - d2) < 1e-5


def test_perfect_prediction_scores_one():
    truth = np.zeros((16, 16), bool)
    truth[8, :] = True
    assert metric.dti(truth.astype(float), truth) > 0.9999


def test_kernel_support_is_300m():
    dr, dc, kw = metric.kernel_offsets()
    assert int(np.hypot(dr, dc).max()) <= 2          # offsets at d=3 have k=0
    assert abs(kw[np.argmax((dr == 0) & (dc == 0))] - 1.0) < 1e-12
    # perpendicular cross-section of a straight trace sums to 3.0
    tot = sum(k for r, c, k in zip(dr, dc, kw) if c == 0)
    assert abs(tot - 3.0) < 1e-12, tot


def test_breakeven_ordering():
    """break-even delta is ~alpha*DTI, and grows with DTI."""
    for d in (0.20, 0.28, 0.3774):
        b = metric.breakeven_delta(d)
        assert 0.9 * 0.2 * d < b < 1.15 * 0.2 * d, (d, b)
    assert metric.breakeven_delta(0.3774) > metric.breakeven_delta(0.28)
