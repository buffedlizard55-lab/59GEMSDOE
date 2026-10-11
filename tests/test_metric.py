"""Hand-computed checks for the distance-weighted Tversky index (src/gems_dti/metric.py).

Every expected value below is derived by hand from the official formula, not from the code.
"""

import numpy as np
import pytest

from gems_dti.metric import (
    ALPHA,
    BETA,
    break_even_distance_px,
    break_even_kernel_credit,
    dti,
)


def _grid(h=20, w=20):
    return np.zeros((h, w)), np.zeros((h, w), dtype=bool)


def test_perfect_prediction_scores_one():
    pred, truth = _grid()
    truth[10, 10] = True
    pred[10, 10] = 1.0
    r = dti(pred, truth)
    assert r.dti == pytest.approx(1.0, abs=1e-6)
    assert r.fp_w == pytest.approx(0.0)
    assert r.fn_w == pytest.approx(0.0)


def test_empty_prediction_scores_zero():
    pred, truth = _grid()
    truth[10, 10] = True
    r = dti(pred, truth)
    assert r.dti == 0.0
    assert r.fn_w == pytest.approx(1.0)  # one truth pixel, nothing predicted


def test_half_probability_on_truth():
    # TP_w = 0.5, FN_w = 0.5, FP_w = 0  ->  0.5 / (0.5 + 0.8*0.5) = 0.5556
    pred, truth = _grid()
    truth[10, 10] = True
    pred[10, 10] = 0.5
    r = dti(pred, truth)
    assert r.dti == pytest.approx(0.5 / (0.5 + BETA * 0.5), rel=1e-6)


def test_neighbour_one_pixel_away_uses_kernel():
    # d = 100 m = 1 px, k = 1 - 1/3 = 2/3
    # TP_w = 2/3, FN_w = 1/3, FP_w = 1 * (1 - 2/3) = 1/3
    # DTI = (2/3) / (2/3 + 0.2/3 + 0.8/3) = 2/3
    pred, truth = _grid()
    truth[10, 10] = True
    pred[10, 11] = 1.0
    r = dti(pred, truth)
    assert r.tp_w == pytest.approx(2 / 3, rel=1e-9)
    assert r.fn_w == pytest.approx(1 / 3, rel=1e-9)
    assert r.fp_w == pytest.approx(1 / 3, rel=1e-9)
    assert r.dti == pytest.approx(2 / 3, rel=1e-6)


def test_prediction_at_kernel_edge_gets_no_credit():
    # d = 300 m = R, so k = 0: no TP credit, full FP and FN
    pred, truth = _grid()
    truth[10, 10] = True
    pred[10, 13] = 1.0
    r = dti(pred, truth)
    assert r.tp_w == pytest.approx(0.0)
    assert r.fp_w == pytest.approx(1.0)
    assert r.fn_w == pytest.approx(1.0)
    assert r.dti == pytest.approx(0.0, abs=1e-8)


def test_official_worked_example_arithmetic():
    # Official page: TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 -> TI_w = 0.60
    value = 3.00 / (3.00 + ALPHA * 1.89 + BETA * 2.00)
    assert round(value, 2) == 0.60


def test_identity_tp_plus_fn_equals_truth_count():
    rng = np.random.default_rng(0)
    pred = rng.random((40, 40)) * (rng.random((40, 40)) < 0.2)
    truth = rng.random((40, 40)) < 0.03
    r = dti(pred, truth)
    assert r.tp_w + r.fn_w == pytest.approx(r.n_truth, rel=1e-9)
    # exact re-expression used for the break-even rule
    alt = r.tp_w / (ALPHA * (r.tp_w + r.fp_w) + BETA * r.n_truth + 1e-9)
    assert r.dti == pytest.approx(alt, rel=1e-9)


def test_ignore_mask_removes_pixels_from_every_term():
    pred, truth = _grid()
    truth[10, 10] = True
    truth[5, 5] = True
    pred[10, 10] = 1.0
    pred[15, 15] = 1.0  # false positive, far from truth
    ignore = np.zeros_like(truth)
    ignore[5, 5] = True  # the other truth pixel is masked out
    ignore[15, 15] = True  # and the false positive sits in the masked area
    r = dti(pred, truth, ignore=ignore)
    assert r.n_truth == 1
    assert r.fp_w == pytest.approx(0.0)
    assert r.dti == pytest.approx(1.0, abs=1e-6)


def test_rejects_nan_and_out_of_range():
    pred, truth = _grid()
    truth[3, 3] = True
    pred[3, 3] = np.nan
    with pytest.raises(ValueError):
        dti(pred, truth)
    pred[3, 3] = 1.2
    with pytest.raises(ValueError):
        dti(pred, truth)


def test_break_even_numbers():
    # at the public 0.2778 score the break-even kernel credit is 0.2 * 0.2778
    assert break_even_kernel_credit(0.2778) == pytest.approx(0.05556, abs=1e-5)
    # which corresponds to a distance of about 2.83 px = 283 m (inside the 300 m kernel)
    assert break_even_distance_px(0.2778) == pytest.approx(2.8333, abs=1e-3)
    # at the public 0.3774 leader the break-even distance is 2.77 px
    assert break_even_distance_px(0.3774) == pytest.approx(2.7735, abs=1e-3)
