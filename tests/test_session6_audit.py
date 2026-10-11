"""Session-6 registry audit: synthetic unit tests plus pins on committed evidence.

CI has no .cache/registry (818 MB, gitignored). The available synthetic tests exercise the
helper functions directly; archived Session-6 evidence-pin tests are skipped when their receipts
are absent from this merged checkout rather than recreating evidence.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    spec = importlib.util.spec_from_file_location("session6_registry_audit", ROOT / "scripts" / "session6_registry_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit()


def test_profile_counts_finite_positive_cells_inside_footprint_only():
    fp = np.zeros((4, 4), bool)
    fp[:2, :] = True                       # 8 footprint cells
    a = np.zeros((4, 4), np.float32)
    a[0, 1:] = 0.5                         # 3 positive in footprint
    a[3, 0] = 0.9                          # 1 positive OUTSIDE footprint
    a[1, 1] = np.nan                       # non-finite is never a dot
    p = audit.profile_array(a, fp)
    assert p["positive_in_footprint"] == 3
    assert p["positive_outside_footprint"] == 1
    assert p["footprint_fraction_positive"] == pytest.approx(3 / 8)
    assert p["dense"] is False
    assert p["all_finite_values_in_0_1"] is True


def test_dense_flag_triggers_at_half_footprint():
    fp = np.ones((2, 5), bool)             # 10 footprint cells
    a = np.zeros((2, 5), np.float32)
    a[:, :] = 0.3                          # 10 positive
    a[0, 0] = 0.0                          # 9 positive -> 0.9, dense
    assert audit.profile_array(a, fp)["dense"] is True
    a[0, 1] = 0.0                          # 8 positive -> 0.8
    a[0, 2] = 0.0                          # 7 positive
    a[0, 3] = 0.0                          # 6 positive
    a[0, 4] = 0.0                          # 5 positive -> exactly 0.5, dense (>= 0.5)
    assert audit.profile_array(a, fp)["footprint_fraction_positive"] == pytest.approx(0.5)
    assert audit.profile_array(a, fp)["dense"] is True
    a[1, 0] = 0.0                          # 4 positive -> 0.4, not dense
    assert audit.profile_array(a, fp)["dense"] is False


def test_containment_reports_removed_cells_and_catalogue_distance():
    cat = np.zeros((9, 9), bool)
    cat[4, :] = True                       # a mapped trace along row 4
    base = np.zeros((9, 9), bool)
    base[2, 4] = True                      # distance 2 px to the trace
    base[4, 0] = True                      # on the trace (distance 0)
    base[7, 7] = True                      # far from the trace (distance 3)
    target = base.copy()
    target[4, 0] = False                   # removed on the trace
    target[2, 4] = False                   # removed at 2 px
    out = audit.containment(base, target, cat)
    assert out["target_subset_of_base"] is True
    assert out["added_vs_base"] == 0
    assert out["removed_vs_base"] == 2
    assert out["removed_on_catalogue"] == 1
    assert out["removed_distance_px_min"] == pytest.approx(0.0)
    assert out["removed_distance_px_max"] == pytest.approx(2.0)


def test_containment_detects_added_cells():
    cat = np.zeros((3, 3), bool)
    base = np.zeros((3, 3), bool)
    target = np.zeros((3, 3), bool)
    target[1, 1] = True
    assert audit.containment(base, target, cat)["target_subset_of_base"] is False


def test_git_blob_sha1_matches_git_object_format(tmp_path):
    import subprocess
    f = tmp_path / "x.bin"
    f.write_bytes(b"gems59\n")
    expected = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True, check=True).stdout.strip()
    assert audit.git_blob_sha1(f) == expected


@pytest.mark.skipif(
    not (ROOT / "evidence" / "registry_profile_session6.json").is_file(),
    reason="the merge does not include the archived Session-6 profile receipt; do not fabricate it",
)
def test_committed_profile_summary_is_pinned():
    p = json.loads((ROOT / "evidence" / "registry_profile_session6.json").read_text())
    assert p["registry_index"] == "evidence/registry_refreshed_20261010T2001.json"
    s = p["summary"]
    assert s["indexed"] == 696
    assert s["fetched_and_verified"] == 696
    assert s["errors"] == 0
    assert s["sha256_pin_mismatches"] == 0
    assert s["git_blob_sha1_mismatches"] == 0
    assert s["footprint_cells"] == 5167373
    assert s["dense_rasters_ge_50pct_footprint"] == 53


@pytest.mark.skipif(
    not (ROOT / "evidence" / "session6_mechanism_and_witness.json").is_file(),
    reason="the merge does not include the archived Session-6 witness receipt; do not fabricate it",
)
def test_committed_witness_is_universal_blocker_and_mechanism_is_subset():
    m = json.loads((ROOT / "evidence" / "session6_mechanism_and_witness.json").read_text())
    w = m["witness_saturation"]
    assert w["uncovered_allowed_pixels"] == 0
    assert w["universal_overlap_blocker"] is True
    assert w["witness_sha256"] == "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"
    c = m["owner_reported_pair_containment"]
    assert c["base_positive"] == 40199 and c["target_positive"] == 37654
    assert c["target_subset_of_base"] is True
    assert c["added_vs_base"] == 0 and c["removed_vs_base"] == 2545
    assert c["removed_on_catalogue"] == 0
    assert c["removed_distance_px_min"] >= 1.0 and c["removed_distance_px_max"] <= 2.0 + 1e-9


def test_h59j_literal_gate_fails_and_current_card_stays_unsubmittable():
    audit = json.loads((ROOT / "evidence" / "uniqueness_h59j.json").read_text())
    gate = audit["strict_literal_gate"]
    assert audit["registry_rasters_checked"] == 695
    assert audit["complete_accessible_scan"] is True
    assert audit["unique"] is False
    assert gate["literal_forward_overlap_hits"] == 86
    assert gate["worst_literal_forward_overlap"] == 1.0
    assert gate["passes"] is False
    assert gate["latest_known_index_rasters"] == 696
    assert gate["latest_known_index_scanned_against_candidate"] is False

    card = json.loads((ROOT / "evidence" / "run_card_current.json").read_text())
    registry = card["correlation_overlap_vs_registry"]
    assert card["okay_to_submit"] is False
    assert card["submission_slots_used"] == 0
    assert card["approved_for_weekly_slot"] is False
    assert card["verdict"] == "negative"
    assert registry["leg1_literal_passed"] is False
    assert registry["strict_literal_gate_passes"] is False
    assert registry["strict_literal_forward_overlap_hits"] == 86
    assert registry["unique_under_literal_protocol"] is False
    assert registry["latest_known_index_scanned_against_candidate"] is False
    assert card["registry_gate_phases"]["surface_before_placement"]["status"] == "NOT VERIFIED"
    assert card["registry_gate_phases"]["final_dots"]["status"] == "STOP"

    ledger = (ROOT / "docs" / "research" / "irregularities.md").read_text()
    assert "IR-59J-05" in ledger and "DUPLICATE / STOP" in ledger
