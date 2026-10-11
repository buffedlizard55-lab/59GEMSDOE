"""Unit tests for the pieces the lane's correctness rests on.

Run with ``python3 -m unittest tests.test_lane`` from the repository root (or with pytest:
this file is collected by the repository's pytest run).  It imports the replication's own driver,
``scripts/replication_run_lane.py``, which is namespaced so it cannot be confused with the
fault-zone-anatomy driver at ``scripts/run_lane.py`` that other sessions own.
The metric tests are parity checks against the literal brute-force transcription;
the emission tests are invariants that must hold for any release.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gems59 import metric as mt  # noqa: E402
from gems59 import zone as zn  # noqa: E402


class TestMetricParity(unittest.TestCase):
    def test_binary_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        for _ in range(4):
            truth = rng.random((24, 24)) < 0.05
            pred = rng.random((24, 24)) < 0.05
            if truth.sum() == 0 or pred.sum() == 0:
                continue
            fast = mt.dti_binary(pred, truth)
            slow = mt.dti_bruteforce(pred.astype(float), truth)
            self.assertAlmostEqual(fast["tp"], slow["tp"], places=6)
            self.assertAlmostEqual(fast["fp"], slow["fp"], places=6)
            self.assertAlmostEqual(fast["dti"], slow["dti"], places=6)

    def test_soft_matches_bruteforce(self):
        rng = np.random.default_rng(11)
        truth = rng.random((20, 20)) < 0.04
        pred = rng.random((20, 20)) * (rng.random((20, 20)) < 0.05)
        fast = mt.dti_exact(pred, truth)
        slow = mt.dti_bruteforce(pred, truth)
        self.assertAlmostEqual(fast["tp"], slow["tp"], places=6)
        self.assertAlmostEqual(fast["dti"], slow["dti"], places=6)

    def test_known_masks_are_excluded(self):
        truth = np.zeros((10, 10), bool)
        truth[5, 5] = True
        known = np.zeros((10, 10), bool)
        known[5, 5] = True          # truth pixel inside the masked catalogue
        pred = np.zeros((10, 10), bool)
        pred[5, 5] = True
        res = mt.dti_binary(pred, truth, known=known)
        self.assertEqual(res["n_truth"], 0)
        self.assertEqual(res["n_emitted"], 0)

    def test_marginal_accept_is_exact(self):
        # one truth pixel, no dots yet: a dot at distance 1 (k = 2/3) pays at any s.
        self.assertTrue(mt.marginal_accept(2 / 3, 2 / 3, T=0.0, M=0.0, n=0, K=100.0))
        # a far dot (k = 0) never pays.
        self.assertFalse(mt.marginal_accept(0.0, 0.0, T=10.0, M=0.0, n=50, K=100.0))


class TestZoneInvariants(unittest.TestCase):
    def test_levels_are_multiples_of_three(self):
        self.assertTrue(all(s % 3 == 0 for s in zn.LEVELS))

    def test_marginal_bar_matches_definition(self):
        for s in (0.1, 0.26, 0.3195):
            self.assertAlmostEqual(zn.marginal_bar(s), 0.2 * s / (1 - 0.2 * s))

    def test_tile_respects_allowed_mask(self):
        allowed = np.zeros((40, 40), bool)
        allowed[5:35, 5:35] = True
        density = np.zeros((4, 4))
        density[1, 1] = 1.0
        em, level, rec = zn.tile_to_budget(density, allowed, np.array([3, 3, 3, 3, 3]),
                                           budget=1000, min_px_per_block=0.5)
        self.assertTrue(em.sum() > 0)
        self.assertFalse((em & ~allowed).any())
        self.assertEqual(rec["dots"], int(em.sum()))
        self.assertLessEqual(int(em.sum()), 1000)

    def test_lattice_phase_is_stable(self):
        shape = (13, 13)
        m = zn.lattice_mask(shape, 3)
        rows, cols = np.nonzero(m)
        self.assertTrue(np.all(rows % 3 == zn.PHASE))
        self.assertTrue(np.all(cols % 3 == zn.PHASE))


class TestEmissionInvariants(unittest.TestCase):
    def test_budget_emission_respects_masks_and_budget(self):
        import replication_run_lane as R
        rng = np.random.default_rng(3)
        allowed = np.zeros((60, 60), bool)
        allowed[5:55, 5:55] = True
        catalogue = np.zeros((60, 60), bool)
        catalogue[20:40, 20:40] = True
        allowed &= ~catalogue
        shares = rng.random((6, 6))
        shares /= shares.sum()
        em, level, rec = R._budget_emission(shares, 10_000.0, np.array([3, 3, 3, 3, 3]),
                                            allowed, 150, 0.5)
        self.assertFalse((em & ~allowed).any(), "dot outside the allowed mask")
        self.assertFalse((em & catalogue).any(), "dot on the mapped catalogue")
        self.assertLessEqual(int(em.sum()), 150)
        self.assertEqual(rec["dots"], int(em.sum()))
        self.assertTrue(set(np.unique(level)) <= {0, 3, 6, 9, 12})


if __name__ == "__main__":
    unittest.main()
