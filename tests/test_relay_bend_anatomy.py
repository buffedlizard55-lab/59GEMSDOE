"""Unit and regression tests for multi-scale host-bend, two-host relay, and sense-transition features."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from gems59.relay_bend_anatomy import (
    ALL_RELAY_BEND_FEATURES,
     build_relay_bend_geometry,
    host_bend_fields,
    kinematic_transition_field,
    two_host_frame,
)
from gems59.strand_orientation import magnetic_geometry


def test_two_host_frame_matches_bruteforce_oracle():
    vis = np.zeros((32, 36), bool)
    vis[5:12, 6] = True       # component 1
    vis[5:12, 26] = True      # component 2
    vis[24, 10:22] = True     # component 3
    th = two_host_frame(vis)
    comp, ncomp = ndi.label(vis, structure=np.ones((3, 3), bool))
    assert ncomp == 3
    assert not (th["c1"] == th["c2"]).any()
    assert (th["d2"] >= th["d1"] - 1e-5).all()

    # Brute-force check on every cell: d2 must equal min distance to vis & (comp != c1)
    d_by_cid = {}
    for cid in range(1, ncomp + 1):
        d_by_cid[cid] = ndi.distance_transform_edt(~(vis & (comp != cid)))
    expected_d2 = np.zeros_like(th["d2"])
    for cid in range(1, ncomp + 1):
        sel = th["c1"] == cid
        expected_d2[sel] = d_by_cid[cid][sel]
    np.testing.assert_allclose(th["d2"], expected_d2, rtol=1e-5, atol=1e-5)


def test_two_host_frame_rejects_fewer_than_two_components():
    vis = np.zeros((20, 20), bool)
    with pytest.raises(ValueError, match="undefined"):
        two_host_frame(vis)
    vis[5:10, 5] = True
    with pytest.raises(ValueError, match="at least two"):
        two_host_frame(vis)


def test_host_bend_fields_distinguish_bent_from_straight_trace():
    straight = np.zeros((60, 60), bool)
    straight[10:50, 30] = True
    bs = host_bend_fields(straight, fine_px=2.0, coarse_px=8.0)
    # Interior of a straight trace (away from tips) has near-zero multi-scale turning
    assert float(bs["turn"][18:42, 30].max()) < 0.06

    bent = np.zeros((60, 60), bool)
    bent[10:31, 20] = True
    for i in range(20):
        bent[30 + i, 20 + i] = True
    bb = host_bend_fields(bent, fine_px=2.0, coarse_px=8.0)
    # Near the bend vertex (30, 20), multi-scale turning must be strongly positive
    assert float(bb["turn"][28:35, 19:25].max()) > 0.45


def test_kinematic_transition_is_visible_only_and_peaks_at_mixed_sense():
    vis = np.zeros((50, 50), bool)
    vis[15:35, 18] = True
    vis[15:35, 32] = True
    sense = np.zeros((50, 50), np.int8)
    sense[15:35, 18] = 2  # RL strike-slip
    sense[15:35, 32] = 1  # Normal
    tr_mixed = kinematic_transition_field(vis, sense, smooth_px=8.0)
    assert float(tr_mixed[25, 25]) > 0.5

    # Pure normal everywhere -> zero transition
    sense_pure = np.where(vis, 1, 0).astype(np.int8)
    tr_pure = kinematic_transition_field(vis, sense_pure, smooth_px=8.0)
    assert float(tr_pure.max()) == 0.0

    # Changing sense outside visible mask must NOT change the output (no leakage)
    sense_perturbed = sense.copy()
    sense_perturbed[~vis] = 3
    tr_pert = kinematic_transition_field(vis, sense_perturbed, smooth_px=8.0)
    np.testing.assert_array_equal(tr_mixed, tr_pert)


def test_build_relay_bend_geometry_hidden_invariance(tmp_path: Path):
    vis = np.zeros((48, 48), bool)
    vis[10:38, 12] = True
    vis[10:38, 34] = True
    hidden_a = np.zeros_like(vis)
    hidden_a[24, 23] = True
    hidden_b = ~vis
    sense = np.zeros(vis.shape, np.int8)
    sense[10:38, 12] = 2
    sense[10:38, 34] = 1
    yy, xx = np.indices(vis.shape)
    scarp = magnetic_geometry((xx + 0.5 * yy).astype(np.float32), np.ones_like(vis))
    grid = SimpleNamespace(shape=vis.shape, footprint=np.ones_like(vis))

    ga = build_relay_bend_geometry(grid, vis, hidden_a, ~vis, sense, scarp, tmp_path, "a")
    gb = build_relay_bend_geometry(grid, vis, hidden_b, ~vis, sense, scarp, tmp_path, "b")
    assert ga.X.shape[1] == len(ALL_RELAY_BEND_FEATURES) == 22
    assert np.isfinite(ga.X).all()
    np.testing.assert_array_equal(ga.X, gb.X)
    assert not np.array_equal(ga.y, gb.y)
    # Midpoint (24, 23) between two parallel traces at col 12 and col 34 has relay_facing > 0.9
    sel = (ga.rows == 24) & (ga.cols == 23)
    i_facing = ALL_RELAY_BEND_FEATURES.index("relay_facing")
    i_ratio = ALL_RELAY_BEND_FEATURES.index("relay_ratio")
    assert float(ga.X[sel, i_facing][0]) > 0.95
    assert float(ga.X[sel, i_ratio][0]) == pytest.approx(11.0 / 22.0, abs=1e-4)


def test_session5_relay_bend_holdout_and_surface_receipts():
    import hashlib
    import json
    root = Path(__file__).resolve().parents[1]
    holdout = json.loads((root / "evidence/relay_bend_holdout.json").read_text())
    canary = json.loads((root / "evidence/relay_bend_canary.json").read_text())
    uniq = json.loads((root / "evidence/relay_bend_surface_uniqueness.json").read_text())
    card = json.loads((root / "evidence/run_card_current.json").read_text())

    assert holdout["canary_clean"] is True
    assert len(canary["features"]) == 22
    assert all(v["discriminative_auc_max"] <= 0.90 and not v["leakage_flag"] for v in canary["features"].values())

    # E2 (relay_bend_anatomy) must have strictly positive paired 95% CIs vs distance_only, anatomy, and bend_anatomy
    for key in ("distance_only", "anatomy", "bend_anatomy"):
        diff = holdout["paired_differences"][key]
        assert diff["delta"] > 0.02
        assert diff["ci95"][0] > 0.0

    # The Session-5 surface is superseded. Its own receipts must still describe
    # *it*, and the current card must describe the Session-7 release instead.
    session5 = json.loads((root / "evidence/run_card_session5.json").read_text())
    tif_path = root / session5["file"]
    assert tif_path.is_file()
    sha = hashlib.sha256(tif_path.read_bytes()).hexdigest()
    assert sha == uniq["candidate_file_sha256"]
    assert uniq["registry_rasters_checked"] == 695
    assert uniq["byte_unique_among_checked"] is True
    assert uniq["pixel_unique_among_checked"] is True
    assert session5["okay_to_submit"] is False and session5["verdict"] == "negative"
    assert card["raster_sha256"] != sha, 'the current card must not point at the Session-5 surface'
    assert card["correlation_overlap_vs_registry"]["worst_spearman"] <= 0.90
    assert card["correlation_overlap_vs_registry"]["worst_reverse_overlap"] <= 0.70
    assert card["submission_slots_used"] == 0

