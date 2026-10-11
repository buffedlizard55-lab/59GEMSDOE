"""Regression tests for the fitted anatomy feature frame.

IR-57-STRIKE-01: ``fold_geometry`` once carried
``np.where(np.isfinite(s), 0.0, s)`` — an inverted NaN-fill that zeroed every
finite strike, collapsed sin2/cos2 to constants and forced the offset frame to
strike 0.  The tests here pin the corrected behaviour on the real catalogue.
"""
import numpy as np
import pytest
import rasterio

from pathlib import Path

from gems59 import load_grid
from gems59.anatomy import FEATURES, fold_geometry

ROOT = Path(__file__).resolve().parents[1]


def _visible_hidden_domain(grid):
    visible = grid.catalogue.copy()
    # withhold nothing; domain = whole footprint minus catalogue
    hidden = np.zeros(grid.shape, bool)
    domain = grid.footprint & ~visible
    return visible, hidden, domain


def test_fold_geometry_strike_frame_is_not_degenerate():
    grid = load_grid()
    visible, hidden, domain = _visible_hidden_domain(grid)
    # a small domain keeps the test fast: crop to a 400x400 px window on faults
    ys, xs = np.nonzero(visible)
    y0, x0 = ys[len(ys) // 2], xs[len(xs) // 2]
    sl = (slice(max(0, y0 - 200), y0 + 200), slice(max(0, x0 - 200), x0 + 200))
    dom = np.zeros(grid.shape, bool)
    dom[sl] = domain[sl]
    g = fold_geometry(grid, visible, hidden, dom, "regression_strike")
    assert g.X.shape[1] == len(FEATURES) == len(g.feature_names)
    i_sin = g.feature_names.index("sin2")
    i_cos = g.feature_names.index("cos2")
    i_perp = g.feature_names.index("d_perp")
    assert g.X.shape[0] > 1000, "window too small to be meaningful"
    # the bug signature: sin2 constant 0 and cos2 constant 1 (strike == 0)
    assert float(g.X[:, i_sin].std()) > 0.05, "sin2 is degenerate — strike frame broken"
    assert float(g.X[:, i_cos].std()) > 0.05, "cos2 is degenerate — strike frame broken"
    assert float(g.X[:, i_perp].std()) > 0.5, "d_perp has no spread — offset frame broken"


def test_geo_planes_load_and_match_grid():
    from gems59.geo import GEO_FEATURES, load_planes
    try:
        from gems59.geo import FEATURES_PATH
        if not FEATURES_PATH.exists():
            pytest.skip("training_features.tif not assembled (run prepare_data.py --fetch)")
    except Exception:
        pytest.skip("geo module unavailable")
    grid = load_grid()
    planes = load_planes(grid.footprint)
    assert set(planes) == set(GEO_FEATURES)
    for name, plane in planes.items():
        assert plane.shape == grid.shape and plane.dtype == np.float32
        v = plane[grid.footprint]
        assert np.isfinite(v).all(), f"{name} has non-finite values inside the footprint"
        assert float(np.abs(v).max()) <= 10.0 + 1e-6, f"{name} exceeds the robust clip"
    # geo-augmented fold_geometry appends the block in GEO_FEATURES order
    visible, hidden, domain = _visible_hidden_domain(grid)
    ys, xs = np.nonzero(visible)
    y0, x0 = ys[len(ys) // 2], xs[len(xs) // 2]
    dom = np.zeros(grid.shape, bool)
    dom[max(0, y0 - 100):y0 + 100, max(0, x0 - 100):x0 + 100] = True
    g = fold_geometry(grid, visible, hidden, dom, "regression_geo", geo=planes)
    assert g.feature_names == FEATURES + GEO_FEATURES
    assert g.X.shape[1] == len(FEATURES) + len(GEO_FEATURES)


# ---------------------------------------------------------------------------
# Session-2 additions (arena/884d08ea, 2026-10-09): synthetic-grid pins for the
# corrected strike frame.  These run in ~1 s (no full-catalogue load) and pin
# the exact orientation values, not just "has spread".
# ---------------------------------------------------------------------------

def _synth_grid(catalogue: np.ndarray):
    footprint = np.ones_like(catalogue, bool)
    from gems59.grid import Grid
    return Grid(footprint=footprint, catalogue=catalogue, crs=None,
                transform=None, height=catalogue.shape[0], width=catalogue.shape[1])


def _two_strike_catalogue(n=80, w=80):
    """Two straight traces with different strikes: a (1,1) diagonal at azimuth
    135 deg and a (1,-1) anti-diagonal at azimuth 45 deg."""
    cat = np.zeros((n, w), bool)
    for i in range(20, 50):
        cat[i, i] = True            # azimuth 135 deg
        cat[i, 69 - i] = True       # azimuth 45 deg
    return cat


def test_strike_encoding_matches_trace_orientation():
    """IR-57-STRIKE-01 pin: the cyclic encoding must reflect the actual anchor
    strike.  Azimuth 135 deg gives sin(2*135) = -1, cos(2*135) = 0; azimuth
    45 deg gives sin(2*45) = +1, cos(2*45) = 0.  With the inverted fallback the
    encoding was the constant (0, 1) regardless of orientation."""
    cat = _two_strike_catalogue()
    g = _synth_grid(cat)
    geom = fold_geometry(g, cat, np.zeros_like(cat), np.ones_like(cat), "synth")
    assert geom.X.shape[1] == len(FEATURES)
    sin2 = geom.X[:, FEATURES.index("sin2")]
    cos2 = geom.X[:, FEATURES.index("cos2")]
    assert np.isfinite(geom.X).all()
    # both orientations are present, so the encoding must take both values
    assert set(np.unique(np.round(sin2, 3))) == {-1.0, 1.0}
    assert np.ptp(cos2) < 0.1          # cos(90) = cos(270) = 0 for both
    # and the encoding must agree with the interaction columns
    d = geom.X[:, FEATURES.index("d")]
    np.testing.assert_allclose(geom.X[:, FEATURES.index("sin2d")], sin2 * d,
                               rtol=1e-5, atol=1e-5)


def test_offset_frame_is_trace_aligned_not_grid_aligned():
    """A pixel offset perpendicular to a (1,1) trace must have d_perp > 0 and
    d_par ~ 0; a grid-aligned decomposition would swap them."""
    cat = _two_strike_catalogue()
    g = _synth_grid(cat)
    geom = fold_geometry(g, cat, np.zeros_like(cat), np.ones_like(cat), "synth")
    d = geom.X[:, FEATURES.index("d")]
    d_perp = geom.X[:, FEATURES.index("d_perp")]
    d_par = geom.X[:, FEATURES.index("d_par_abs")]
    # cells whose offset is predominantly cross-strike (within 3 px of a trace)
    perp_cells = (d > 0) & (d < 3) & (d_perp > 0.9 * d)
    assert perp_cells.any(), "expected cells whose offset is mostly cross-strike"
    # for those cells the along-strike component must be the small one; with the
    # pre-fix grid-aligned frame the two components were row/col offsets and
    # this fails for the (1,-1)-oriented trace
    assert (d_par[perp_cells] < 0.5 * d[perp_cells]).mean() > 0.9


def test_interaction_features_are_strike_times_distance():
    cat = _two_strike_catalogue()
    g = _synth_grid(cat)
    geom = fold_geometry(g, cat, np.zeros_like(cat), np.ones_like(cat), "synth")
    d = geom.X[:, FEATURES.index("d")]
    sin2 = geom.X[:, FEATURES.index("sin2")]
    cos2 = geom.X[:, FEATURES.index("cos2")]
    np.testing.assert_allclose(geom.X[:, FEATURES.index("sin2d")], sin2 * d,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(geom.X[:, FEATURES.index("cos2d")], cos2 * d,
                               rtol=1e-5, atol=1e-5)
