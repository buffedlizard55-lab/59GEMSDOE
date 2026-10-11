"""Session 7 regression checks: the verification receipts must match recomputation."""
from __future__ import annotations

import json

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import session7_verify as S7  # noqa: E402


def test_download_is_float32_unit_interval_and_zero_filled_outside_footprint():
    d = S7.check_download()
    assert d["meta"]["dtype"] == "float32" and d["meta"]["count"] == 1
    assert d["meta"]["crs"] == "EPSG:32611" and d["meta"]["shape"] == [3730, 3292]
    assert d["in_unit_interval"] is True and d["n_nan"] == 0
    # Known deviation from the official convention (IR-S7-03): documented, not hidden.
    assert d["outside_footprint_is_nan"] is False


def test_bridged_template_and_labels_are_flagged_not_trusted():
    t = S7.check_template()
    assert t["sample_positive_cells"] == t["labels_positive_cells"] == 60988
    assert t["sample_positives_equal_label_positives"] is True
    assert S7.check_duplicate_labels()["byte_identical"] is True


def test_0_2778_is_subset_of_0_2708_and_removed_dots_sit_within_2px_of_catalogue():
    r = S7.check_0_2778_pair()
    assert r["kept_is_subset_of_0_2708"] is True
    assert r["removed_cells"] == 2545
    assert r["removed_all_within_2px"] is True and r["kept_none_within_2px"] is True


def test_holdout_receipts_recompute_and_collar_blind_zone_is_reported():
    h = S7.check_holdout()
    assert h["max_abs_recompute_error"] < 1e-12
    assert h["fold_arm_receipts"] == 20
    assert h["collar_blind_zone"] is True
    assert abs(h["nearest_visible_to_truth_px_min"] - 10 ** 0.5) < 1e-9


@pytest.mark.skipif(
    not (ROOT / "data/official/training_features.tif").exists()
    or not (ROOT / "evidence/feature_cache.json").exists(),
    reason="training_features.tif / band cache are gitignored and absent in CI; run scripts/prepare_data.py --fetch --cache-bands to check pins",
)
def test_pins_match_and_evidence_file_is_current():
    p = S7.check_pins()
    assert p["training_features_match"] is True and p["band12_cache_match"] is True
    saved = json.loads((ROOT / "evidence/session7_verification.json").read_text())
    assert saved["download"]["sha256"] == S7.check_download()["sha256"]
    assert saved["holdout"]["collar_blind_zone"] is True
