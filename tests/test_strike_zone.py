"""Histogram intensity: train labels never leak into a test apply."""
import numpy as np
from types import SimpleNamespace

from gems59.anatomy import FEATURES
from gems59.strike_zone import fit_zone, strike_deg_from_X


def _geom(n=1000, seed=0):
    # n and the draw order are chosen so the test does not depend on the
    # exact RNG stream: d_perp/strike are drawn BEFORE the feature matrix
    # (whose width grows when FEATURES grows) and n is large enough that
    # every d bin clears MIN_BIN_TOTAL=40 with positives (IR-57-STRIKE-01
    # session-2 union: the 11-feature FEATURES shifted the old stream and
    # the old n=200 left every bin under the keep threshold).
    rng = np.random.default_rng(seed)
    d_perp = rng.integers(0, 12, size=n).astype(np.float32)
    strike = rng.uniform(0, 180, size=n)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    # encode a real strike in sin2/cos2
    X[:, FEATURES.index("sin2")] = np.sin(np.radians(2 * strike))
    X[:, FEATURES.index("cos2")] = np.cos(np.radians(2 * strike))
    X[:, FEATURES.index("d_perp")] = d_perp
    y = (d_perp <= 3).astype(np.int8)
    return SimpleNamespace(X=X, y=y, n_hidden=int(y.sum()))


def test_strike_roundtrip():
    g = _geom()
    s = strike_deg_from_X(g.X)
    assert s.min() >= 0 and s.max() < 180


def test_fit_drops_unenriched_bins_and_does_not_use_heldout_y():
    train = [_geom(seed=1), _geom(seed=2)]
    test = _geom(seed=99)
    m = fit_zone(train, d_col=FEATURES.index("d_perp"), outer=False, isotropic=True)
    assert m.n_kept_bins > 0
    p_test = m.intensity(test.X)
    assert p_test.shape == (test.X.shape[0],)
    assert p_test.min() >= 0 and p_test.max() <= 1
    # held-out labels must not be required to apply
    m2 = fit_zone(train, d_col=FEATURES.index("d_perp"), outer=True, isotropic=False)
    # outer zeros d_perp < 2
    p = m2.intensity(test.X)
    close = test.X[:, FEATURES.index("d_perp")] < 2
    assert (p[close] == 0).all()
