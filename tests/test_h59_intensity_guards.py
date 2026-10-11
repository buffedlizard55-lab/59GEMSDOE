"""Guard regressions: incomplete requested methods cannot start new H59 runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_run_card_and_registry_receipts_preserve_stop_status():
    card = json.loads((ROOT / "evidence/run_card.json").read_text())
    registry = json.loads((ROOT / "evidence/uniqueness_h59i.json").read_text())
    assert card["verdict"] == "negative"
    assert card["holdout_result"]["evidence_class"].startswith("HOLDOUT-DTI")
    assert card["holdout_result"]["withheld_positive_pixels"] == 11321
    assert card["raster"]["download_status"].endswith("NOT CLEARED TO SUBMIT")
    assert card["submission"]["cleared_to_submit"] is False
    assert card["submission"]["weekly_slot_used"] is False
    assert len(card["submission"]["note"]) <= 140
    gate = registry["strict_literal_gate"]
    assert gate["historical_registry_rows_expected"] == 695
    assert gate["registry_rows_checked"] == 695
    assert gate["overlap_hit_count"] == 85
    assert gate["latest_index_rows_expected"] == 696
    assert gate["latest_index_rows_checked_against_candidate"] == 0
    assert gate["passes"] is False
    witness = gate["prior_witness_claim"]
    assert witness["latest_index_entry_present"] is True
    assert witness["raster_available_locally"] is False
    assert witness["verification_receipts_present"] is False
    assert witness["universal_coverage_confirmed"] is None
    assert witness["relied_on_for_current_stop"] is False


def test_target_group_requirement_is_explicitly_not_implemented():
    env = subprocess.run(
        [PYTHON, "-c", "import sys; sys.path.insert(0, 'src'); "
         "from gems59.intensity import TARGET_GROUP_BACKGROUND_IMPLEMENTED; "
         "print(TARGET_GROUP_BACKGROUND_IMPLEMENTED)"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert env.stdout.strip() == "False"


def test_holdout_runner_stops_before_any_experiment_without_target_group():
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts/run_intensity_lane.py"), "--seed", "20"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "target-group background is not implemented" in result.stderr
    assert "No new experiment may run" in result.stderr


def test_submission_builder_stops_before_output_without_target_group(tmp_path):
    output = tmp_path / "candidate.tif"
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts/build_submission_intensity.py"),
         "--gamma", "0", "--out", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "target-group background is not implemented" in result.stderr
    assert not output.exists()
