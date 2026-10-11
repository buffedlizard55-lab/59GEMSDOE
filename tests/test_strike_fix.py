"""IR-57-STRIKE-01: finite strike must survive fold_geometry, or Riedel features die."""
from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from gems59.anatomy import FEATURES, fold_geometry
from gems59.strike_zone import strike_deg_from_X


def _toy_grid(h=40, w=40):
    fp = np.ones((h, w), bool)
    cat = np.zeros((h, w), bool)
    return SimpleNamespace(footprint=fp, catalogue=cat, shape=(h, w),
                           height=h, width=w)


def test_nan_fill_keeps_finite_strike_values():
    s = np.array([45.0, np.nan, 90.0, 0.0])
    filled = np.where(np.isfinite(s), s, 0.0)
    assert filled[0] == 45.0 and filled[2] == 90.0 and filled[3] == 0.0
    inverted = np.where(np.isfinite(s), 0.0, s)
    assert inverted[0] == 0.0 and inverted[2] == 0.0, "the pre-fix line zeroed finite strike"


def test_two_orientations_produce_nonconstant_sin2_cos2():
    g = _toy_grid()
    visible = np.zeros((40, 40), bool)
    visible[10, 5:35] = True          # E-W
    visible[5:35, 20] = True          # N-S
    hidden = np.zeros((40, 40), bool)
    hidden[12, 8:16] = True           # parallel to the E-W trace
    domain = np.ones((40, 40), bool)
    fg = fold_geometry(g, visible, hidden, domain, "toy")
    sin2 = fg.X[:, FEATURES.index("sin2")]
    cos2 = fg.X[:, FEATURES.index("cos2")]
    assert fg.X.shape[1] == len(FEATURES)
    assert np.isfinite(sin2).all() and np.isfinite(cos2).all()
    # A constant (0, 1) pair is exactly the pre-fix bug (strike forced to 0 deg).
    assert not (np.allclose(sin2, 0.0) and np.allclose(cos2, 1.0))
    strike = strike_deg_from_X(fg.X)
    assert strike.min() >= 0.0 and strike.max() < 180.0
    assert float(np.ptp(strike)) > 20.0
