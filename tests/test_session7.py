import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT/'evidence/h6_1_proximity_pruning_holdout.json').is_file()
    or not (ROOT/'docs/data/h6_1_proximity_pruning_holdout.json').is_file(),
    reason="H6-1 Session-7 receipts are not present in this checkout",
)
def test_h6_1_negative_receipt_is_fail_closed_and_public_copy_matches():
    source = ROOT / "evidence/h6_1_proximity_pruning_holdout.json"
    public = ROOT / "docs/data/h6_1_proximity_pruning_holdout.json"
    assert source.read_bytes() == public.read_bytes()
    report = json.loads(source.read_text())
    assert report["experiment_id"] == "H6-1"
    assert report["submission_slots_used"] == 0
    assert report["verdict"] == "negative"
    assert report["promotion_rule_passed"] is False
    assert report["scores"]["proximity_prune"]["evaluator_version"] == "gems59-pooled-hide-v2"
    assert report["scores"]["proximity_prune"]["withheld_positive_pixels"] == 11321
    assert report["paired_differences"]["random_prune"]["ci95"][0] <= 0
    assert report["distance_feature_canary"]["discriminative_auc_max"] <= 0.90
    limitation = report["holdout_scope_limitation"]
    assert limitation["nearest_visible_to_any_withheld_truth_px"] > 3.0
    assert "inconclusive" in limitation["implication"].lower()
    for fold in report["per_fold"]:
        assert fold["truth_used_for_pruning"] is False
        assert fold["removed"] >= 0
        assert fold["arms"]["proximity_prune"]["n_emitted"] == fold["allocated"] - fold["removed"]
        assert fold["arms"]["random_prune"]["n_emitted"] == fold["allocated"] - fold["removed"]


@pytest.mark.skipif(
    not (ROOT/'evidence/run_card_session7_h6_1.json').is_file()
    or not (ROOT/'docs/data/run_card_session7_h6_1.json').is_file(),
    reason="H6-1 Session-7 run card is not present in this checkout",
)
def test_session7_run_card_has_no_fabricated_submission_clearance():
    source = ROOT / "evidence/run_card_session7_h6_1.json"
    public = ROOT / "docs/data/run_card_session7_h6_1.json"
    assert source.read_bytes() == public.read_bytes()
    card = json.loads(source.read_text())
    assert card["verdict"] == "negative"
    assert card["okay_to_download"] is False
    assert card["okay_to_submit"] is False
    assert card["raster_sha256"] is None
    assert card["validator_output"] is None
    assert card["submission_name"] is None
    assert card["submission_slots_used"] == 0
