"""Shared evaluator repair: continuous and binary exact-kernel comparisons."""
import numpy as np
from gems59 import evaluate_holdout as eh
from gems59 import metric


def _fold(truth, visible):
    region = np.ones(truth.shape, bool)
    return dict(region=region, truth=truth, visible=visible)


def test_binary_evaluator_matches_independent_edt_with_visible_mask():
    truth = np.zeros((23,23), bool)
    truth[4,4] = truth[12,14] = truth[18,10] = True
    visible = np.zeros_like(truth); visible[5,5] = True
    pred = np.zeros(truth.shape, np.float32)
    pred[5,5] = 1  # visible prediction must be removed pixel-exactly
    pred[4,5] = pred[13,14] = pred[1,1] = 1
    result, terms = eh.evaluate(pred, _fold(truth, visible), np.ones_like(truth), block_side=6)
    direct = metric.dti_binary(pred > 0, truth, known=visible)
    assert abs(result['dti'] - direct['dti']) < 1e-7
    assert abs(result['tpw'] - direct['tp']) < 1e-7
    assert abs(result['fpw'] - direct['fp']) < 1e-7
    assert abs(result['fnw'] - direct['fn']) < 1e-7
    np.testing.assert_allclose(terms.sum(axis=0),
                               [direct['tp'],direct['fp'],direct['fn'],3], atol=1e-6)


def test_continuous_max_not_sum_and_zero_outside_region():
    truth = np.zeros((17,17), bool); truth[8,8] = True
    visible = np.zeros_like(truth)
    region = np.ones_like(truth); region[:3,:] = False
    pred = np.zeros(truth.shape, np.float32)
    pred[8,8] = 0.3
    pred[8,9] = 0.9  # kernel 2/3, so max coverage is 0.6, not 0.9
    pred[1,1] = 1  # outside region, zero penalty
    r, _ = eh.evaluate(pred, dict(region=region, truth=truth, visible=visible),
                       np.ones_like(truth), block_side=5)
    assert abs(r['tpw'] - 0.6) < 1e-6
    assert r['emitted'] == 2
    assert abs(r['fpw'] - (0.3*(1-1) + 0.9*(1-2/3))) < 1e-6
