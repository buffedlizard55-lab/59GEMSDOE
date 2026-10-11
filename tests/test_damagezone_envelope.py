"""Unit tests for the H58 damage-zone envelope module.

Deliberately synthetic and small: these check that the estimators do what their
docstrings claim (recovery of a known exponent, clipping, strict flank
inequality, index alignment, input guards), not that the real grid has any
particular structure.  The real measurements live in
``evidence/h58_structure.json`` and are reproduced by
``scripts/run_h58_experiments.py --stage structure``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gems59.damagezone_envelope import (EXTRA_FEATURES, FLANK_PX_DEFAULT, LEN_EDGES,
                                        OBLIQUITY_EDGES, W_MAX_PX, W_MIN_PX,
                                        fit_width_law, flank_suppress, obliquity_at_anchor,
                                        signed_obliquity)


def _synthetic_law(rng, n=4000, gamma=0.55, w0=12.0, l0=40.0):
    lengths = np.exp(rng.uniform(np.log(2.0), np.log(200.0), n))
    half = w0 * (lengths / l0) ** gamma
    dist = np.abs(rng.normal(0.0, 1.0, n)) * half          # crude heavy-tailed halo
    return lengths, dist


def test_fit_width_law_recovers_a_known_sublinear_exponent():
    rng = np.random.default_rng(7)
    lengths, dist = _synthetic_law(rng)
    law = fit_width_law(lengths, dist, np.ones(len(lengths), bool), quantile=0.9)
    assert abs(law.gamma - 0.55) < 0.15, law.gamma
    assert law.sublinear is True and law.gamma_ci95[1] < 1.0
    assert law.n_bins >= 3 and law.quantile == 0.9
    assert set(law.bins[0]) >= {"len_lo", "len_hi", "n_positives", "length_median_px", "width_px"}
    assert all(b["width_px"] > 0 for b in law.bins)


def test_fit_width_law_widens_monotonically_and_is_clipped():
    rng = np.random.default_rng(11)
    lengths, dist = _synthetic_law(rng, gamma=0.8, w0=25.0)
    law = fit_width_law(lengths, dist, np.ones(len(lengths), bool))
    grid = np.geomspace(1.0, 5000.0, 40)
    width = law.half_width(grid)
    assert np.all(np.diff(width) >= 0.0), "a positive exponent must not narrow with length"
    assert width.min() >= W_MIN_PX and width.max() <= W_MAX_PX
    assert np.isfinite(width).all()


def test_fit_width_law_rejects_malformed_inputs():
    rng = np.random.default_rng(3)
    lengths, dist = _synthetic_law(rng, n=600)
    ones = np.ones(len(lengths), bool)
    with pytest.raises(ValueError, match="aligned"):
        fit_width_law(lengths, dist[:-1], ones)
    with pytest.raises(ValueError, match="finite"):
        bad = dist.copy(); bad[0] = np.nan
        fit_width_law(lengths, bad, ones)
    with pytest.raises(ValueError, match="too few"):
        fit_width_law(lengths, dist, np.zeros(len(lengths), bool))
    with pytest.raises(ValueError, match="three populated"):
        tight = np.where(lengths < 6.0, lengths, 5.0)      # collapses into one bin
        fit_width_law(tight, dist, ones)


def test_sparse_bins_cannot_drive_the_exponent():
    rng = np.random.default_rng(5)
    lengths, dist = _synthetic_law(rng, n=2000)
    keep = np.ones(len(lengths), bool)
    law_clean = fit_width_law(lengths, dist, keep)
    # one lonely 50 km mapped component claiming a 40 km halo must be dropped,
    # because a single-pixel bin would otherwise set the exponent
    law_diluted = fit_width_law(np.r_[lengths, [500.0]], np.r_[dist, [400.0]],
                                np.r_[keep, [True]])
    assert all(b["n_positives"] >= 200 for b in law_diluted.bins)
    assert all(b["len_hi"] <= 320.0 for b in law_diluted.bins)
    assert len(law_clean.bins) == law_clean.n_bins


def test_signed_obliquity_is_bounded_axially_and_ray_constant_on_a_straight_host():
    grid = np.zeros((48, 60), bool)
    grid[6:42, 30] = True                                   # straight N-S host trace
    rows = np.array([10, 10, 25, 25, 40, 40], np.intp)
    cols = np.array([24, 36, 24, 36, 24, 36], np.intp)
    rel, censored = signed_obliquity(grid, rows, cols)
    assert rel.shape == (6,) and rel.dtype == np.float32
    assert np.all(np.abs(rel) <= 90.0 + 1e-6)
    assert censored == 0
    # the axial convention folds +/- along the trace, and a fixed ray angle from a
    # straight host must give the same value regardless of position along it
    assert np.isclose(rel[0], rel[2], atol=1e-5) and np.isclose(rel[2], rel[4], atol=1e-5)
    assert np.isclose(abs(rel[1]), abs(rel[0]), atol=1e-5)  # mirror side folds to the same magnitude
    np.testing.assert_allclose(obliquity_at_anchor(grid, rows, cols), rel, atol=0)


def test_signed_obliquity_guards_and_censoring():
    with pytest.raises(ValueError, match="empty"):
        signed_obliquity(np.zeros((8, 8), bool), np.array([1]), np.array([1]))
    with pytest.raises(ValueError, match="matching"):
        signed_obliquity(np.ones((8, 8), bool), np.array([1, 2]), np.array([1]))
    single = np.zeros((24, 24), bool)
    single[12, 12] = True
    rel, censored = signed_obliquity(single, np.array([12, 13, 11]), np.array([13, 12, 12]))
    assert rel.shape == (3,) and np.all(np.isfinite(rel))
    assert 0 <= censored <= 3                               # censoring is reported, never hidden
    assert np.all(np.abs(rel) <= 90.0 + 1e-6)


def test_flank_suppress_is_strict_and_matches_the_shipped_default():
    d = np.array([0.0, 1.0, 2.0, 2.0001, 3.0, 20.0])
    keep = flank_suppress(d)
    assert keep.dtype == bool
    assert keep.tolist() == [False, False, False, True, True, True]
    assert bool(flank_suppress(d, 0.0)[1])                      # d=1 px is kept at flank 0
    assert np.array_equal(flank_suppress(d), flank_suppress(d, FLANK_PX_DEFAULT))
    assert not flank_suppress(np.array([np.nan]))[0]            # unknown distance is never a free pass


def test_module_constants_line_up_with_the_documented_pipeline():
    assert EXTRA_FEATURES == ("obliq_signed", "obliq_signed_sense", "d_over_w")
    assert np.all(np.diff(OBLIQUITY_EDGES) > 0) and OBLIQUITY_EDGES[0] == -90 and OBLIQUITY_EDGES[-1] == 90
    assert np.all(np.diff(LEN_EDGES) > 0) and LEN_EDGES[0] == 0
    assert W_MIN_PX < W_MAX_PX and FLANK_PX_DEFAULT == 2.0
    # the obliquity bin set is symmetric about the host strike, so near-parallel
    # strands on either side land in the same central bin by construction
    assert np.array_equal(OBLIQUITY_EDGES, -OBLIQUITY_EDGES[::-1])
    assert np.digitize(-6.0, OBLIQUITY_EDGES) == np.digitize(6.0, OBLIQUITY_EDGES)
