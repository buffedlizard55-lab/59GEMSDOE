"""Regression pins for the literal-gate witness evidence (Session 7).

Reads only committed evidence; does not need the 19 MB witness or training features.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence" / "literal_gate_witness_verification_20261010.json"
WITNESS_SHA = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"


def _load():
    return json.loads(EVID.read_text())


def test_witness_is_pinned_and_blocks_every_candidate():
    d = _load()
    w = d["witness"]
    assert w["sha256"] == WITNESS_SHA
    assert w["allowed_cells"] == w["allowed_cells_covered_by_positive"] == 5167373
    assert w["outside_footprint_positive"] == 0
    assert w["universal_overlap_blocker"] is True


def test_gemsdoe32_is_exact_prune_of_base():
    s = _load()["gemsdoe32_subset_check"]
    assert s["candidate_is_subset_of_base"] is True
    assert s["base_positive"] - s["candidate_positive"] == s["removed_vs_base"] == 2545
    assert 1.4 <= s["removed_dist_to_catalogue_px_min"] <= s["removed_dist_to_catalogue_px_max"] <= 2.0
    assert s["kept_dist_to_catalogue_px_min"] > 2.0


def test_no_score_is_claimed_in_witness_evidence():
    d = _load()
    assert "HOLDOUT-DTI of any candidate" in d["not_claimed"]
    assert "leaderboard placement or private score" in d["not_claimed"]
    assert d["evidence_class"].startswith("REGISTRY-MEASUREMENT")
