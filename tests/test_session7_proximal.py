"""S7-1 matched proximal-prune intervention and fail-closed gate invariants."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_session7_proximal import (
    RADIUS_PX,
    build_run_card,
    load_historical_reference,
    load_registry_witness,
    pruned_arms,
    verify_pinned_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pruned_arms_remove_same_number_and_are_deterministic():
    base = np.zeros((9, 9), bool)
    base[1, 1] = base[2, 7] = base[5, 5] = base[7, 2] = True
    distance = np.full(base.shape, 6.0, np.float32)
    distance[1, 1] = 1.0
    distance[5, 5] = 2.0

    arms, n_removed = pruned_arms(base, distance, seed=202610100)
    repeat, repeated_count = pruned_arms(base, distance, seed=202610100)

    assert RADIUS_PX == 2.0
    assert n_removed == repeated_count == 2
    assert arms["base"].sum() == 4
    assert arms["proximal_prune"].sum() == arms["matched_random_prune"].sum() == 2
    assert np.array_equal(arms["proximal_prune"], repeat["proximal_prune"])
    assert np.array_equal(arms["matched_random_prune"], repeat["matched_random_prune"])
    assert not (arms["proximal_prune"] & (distance <= RADIUS_PX)).any()
    assert np.all(arms["proximal_prune"] <= arms["base"])
    assert np.all(arms["matched_random_prune"] <= arms["base"])


def test_pruned_arms_noop_when_no_proximal_dots():
    base = np.zeros((5, 5), bool)
    base[1, 1] = base[3, 3] = True
    distance = np.full(base.shape, 4.0, np.float32)
    arms, n_removed = pruned_arms(base, distance, seed=17)
    assert n_removed == 0
    assert np.array_equal(arms["base"], arms["proximal_prune"])
    assert np.array_equal(arms["base"], arms["matched_random_prune"])


def test_session7_preregistration_is_lane_scoped_and_names_one_candidate():
    slate = json.loads((ROOT / "evidence/session7_proximal_hypotheses.json").read_text())
    assert slate["validation_plan"]["candidate"] == "S7-1"
    assert len(slate["ranked"]) == 4
    assert slate["ranked"][0]["id"] == "S7-1"
    assert "BLOCKED / NOT VIABLE NOW" in slate["ranked"][3]["data_status"]
    assert slate["validation_plan"]["promotion_rule"]
    audit = slate["pre_run_gate_audit"]["historical_comparator"]
    assert audit["promotion_comparator_usable"] is False


def test_source_pins_and_universal_registry_witness_are_revalidated():
    hashes = verify_pinned_inputs()
    assert len(hashes) == 3
    witness = load_registry_witness()
    assert witness["universal_support_coverage"] == 1.0
    assert witness["uncovered_allowed_pixels"] == 0
    assert witness["forward_dot_overlap_implied"] == 1.0
    assert witness["stop_before_production_placement"] is True
    assert witness["candidate_unique_scan_performed"] is False


def test_historical_holdout_is_not_promotion_comparable_after_hash_drift():
    reference = load_historical_reference(seed=20)
    assert reference["evidence_class"] == "HOLDOUT-DTI (historical reference; not a new score)"
    assert reference["dti"] == 0.14139139369438386
    assert reference["same_version_string"] is True
    assert reference["same_split_and_seed"] is True
    assert reference["evaluator_hash_mismatches"] == ["evaluate_holdout.py"]
    assert reference["comparable_for_promotion"] is False


def test_run_card_keeps_unrun_artifact_gates_null_and_note_short():
    witness = load_registry_witness()
    card = build_run_card(
        score=None,
        ci95=None,
        withheld_positive_pixels=11321,
        status="NEGATIVE HOLDOUT ATTEMPT — not run",
        witness=witness,
    )
    assert card["holdout_dti"]["dti"] is None
    assert card["tiff_sha256"] is None
    assert card["validator"]["status"].startswith("NOT RUN")
    assert card["registry_correlation_overlap"]["max_spearman_correlation"] is None
    assert len(card["submission_note"]) <= 140
    assert card["verdict"] == "negative"


def test_three_pass_review_log_is_complete_and_fail_closed():
    review = json.loads((ROOT / "evidence/session7_review_passes.json").read_text())
    assert len(review["passes"]) == 3
    assert all(item["status"].startswith("PASS") for item in review["passes"])
    assert review["final_decision"]["independent_replication"] is False
    assert review["final_decision"]["no_rerun_after_discovery"] is True
    assert review["final_decision"]["new_tiff_sha256"] is None
    assert review["final_decision"]["okay_to_download"] is False
    assert review["final_decision"]["okay_to_submit"] is False
    assert review["final_decision"]["submission_slots_used"] == 0


def test_saved_session7_receipt_math_gates_and_provenance_are_consistent():
    report = json.loads((ROOT / "evidence/session7_proximal_holdout.json").read_text())
    assert report["evidence_class"] == "HOLDOUT-DTI"
    assert report["evaluator_version"] == "gems59-pooled-hide-v2"
    assert report["split"]["withheld_positive_pixels"] == 11321
    assert report["metric"] == {
        "name": "pooled DTI",
        "alpha": 0.2,
        "beta": 0.8,
        "triangular_kernel_radius_m": 300.0,
    }
    for arm in ("base", "proximal_prune", "matched_random_prune"):
        score = report["scores"][arm]
        assert score["evidence_class"] == "HOLDOUT-DTI"
        assert score["evaluator_version"] == report["evaluator_version"]
        assert score["withheld_positive_pixels"] == 11321
        assert np.isclose(
            score["dti"],
            score["tpw"] / (score["tpw"] + 0.2 * score["fpw"] + 0.8 * score["fnw"]),
            atol=1e-12,
        )
    delta = report["paired_differences"]["matched_random_prune"]
    assert np.isclose(delta["delta"],
                      report["scores"]["proximal_prune"]["dti"]
                      - report["scores"]["matched_random_prune"]["dti"], atol=1e-12)
    assert delta["ci95"][0] <= 0 <= delta["ci95"][1]
    assert report["holdout_promotion_condition_met"] is False
    assert report["historical_holdout_reference"]["comparable_for_promotion"] is False
    assert report["leakage_canary"]["clean"] is True
    assert max(v["discriminative_auc_max"] for v in report["leakage_canary"]["features"].values()) < 0.90
    assert len(report["pipeline_implementation_sha256"]) == 11
    assert "src/gems59/grid.py" in report["pipeline_implementation_sha256"]
    assert report["provenance_receipt_correction"]["runner_sha256_at_holdout_execution"] == report["pipeline_implementation_sha256"]["scripts/run_session7_proximal.py"]
    cross_branch = report["cross_branch_context"]
    assert cross_branch["prior_experiment_pr"].endswith("/pull/27")
    assert cross_branch["prior_h6_1_promotion_rule_passed"] is False
    assert cross_branch["prior_h6_1_proximity_minus_random_ci95"][0] <= 0 <= cross_branch["prior_h6_1_proximity_minus_random_ci95"][1]
    assert "do not pool" in cross_branch["interpretation"]
    prior = json.loads((ROOT / "evidence/h6_1_proximity_pruning_holdout.json").read_text())
    assert np.isclose(cross_branch["prior_h6_1_proximity_minus_random"], prior["paired_differences"]["random_prune"]["delta"])
    assert cross_branch["prior_h6_1_proximity_minus_random_ci95"] == prior["paired_differences"]["random_prune"]["ci95"]
    current_best = cross_branch["current_comparable_best_after_main_fetch"]
    assert current_best["evaluator_hashes_match"] is True
    assert current_best["paired_cross_model_test_performed"] is False
    assert current_best["dti"] > current_best["s7_1_proximal_dti"]
    assert report["implementation_sha256"] == report["evaluator_implementation_hashes"]
    assert report["environment_sha256"]["requirements-lock.txt"]
    assert report["runtime_environment"]["python"] == "3.11.2"
    assert report["candidate_geoTIFF_sha256"] is None
    assert report["full_production_surface_built"] is False
    assert report["production_dots_generated"] is False
    assert report["registry_gate"]["candidate_unique_scan_performed"] is False
    assert report["validator_output"]["status"].startswith("NOT RUN")
    assert report["okay_to_download"] is False and report["okay_to_submit"] is False
    assert report["submission_slots_used"] == 0
    assert report["run_card"]["verdict"] == "negative"
    assert len(report["run_card"]["submission_note"]) <= 140
