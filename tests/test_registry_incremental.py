import pytest

from gems59.uniqueness import merge_registry_audits, summarize_registry_rows


def _row(sha, name, overlap=0.1, rho=0.0, jac=0.1, **flags):
    return {
        "repo": "fixture",
        "submission": name,
        "file": name,
        "sha256": sha,
        "my_dots_within_3px_of_theirs": overlap,
        "their_dots_within_3px_of_mine": overlap,
        "spearman_full_footprint": rho,
        "jaccard_dot_sets": jac,
        "identical_bytes": False,
        "identical_decoded_predictions": False,
        "duplicate_by_rho": rho > 0.90,
        "duplicate_by_overlap": overlap > 0.70,
        "duplicate_by_jaccard": jac > 0.50,
        **flags,
    }


def _audit(rows, *, complete=True, candidate_file="file-sha", candidate_pixels="pixel-sha"):
    return summarize_registry_rows(
        rows,
        registry_rasters_expected=len(rows),
        complete_accessible_scan=complete,
        source_errors=[],
        candidate_meta={
            "my_file": "candidate.tif",
            "my_dots": 10,
            "candidate_decoded_sha256": candidate_pixels,
            "candidate_file_sha256": candidate_file,
            "candidate_rank_variation": True,
        },
        scope="fixture",
    )


def test_incremental_merge_reuses_prior_rows_and_applies_literal_gate():
    prior = _audit([_row("old-sha", "old.tif")])
    delta = _audit([_row("new-sha", "new.tif", overlap=0.8)])
    result = merge_registry_audits(
        prior, delta, registry_rasters_expected=2,
        scope="pinned prior plus complete newly scanned delta",
    )
    assert result["complete_accessible_scan"]
    assert result["registry_rasters_expected"] == result["registry_rasters_checked"] == 2
    assert result["duplicate_count"] == 1
    assert result["worst_dot_overlap"] == 0.8
    assert result["stop_required"] and not result["unique"]
    assert result["incremental_extension"] == {
        "base_rasters": 1,
        "newly_discovered_rasters": 1,
        "deduplicated_rasters": 2,
    }


def test_incremental_merge_deduplicates_same_immutable_bytes():
    prior = _audit([_row("same-sha", "old-path.tif")])
    delta = _audit([_row("same-sha", "new-alias.tif")])
    result = merge_registry_audits(prior, delta, registry_rasters_expected=1)
    assert result["registry_rasters_checked"] == 1
    assert result["rows"][0]["sources"] == ["new-alias.tif", "old-path.tif"]

    conflicting = _audit([_row("same-sha", "conflicting.tif", overlap=0.2)])
    with pytest.raises(ValueError, match="conflicting registry measurements"):
        merge_registry_audits(prior, conflicting, registry_rasters_expected=1)


def test_incremental_merge_fails_closed_on_mismatched_candidate_or_incomplete_delta():
    prior = _audit([_row("old-sha", "old.tif")])
    other_candidate = _audit([_row("new-sha", "new.tif")], candidate_file="different-file")
    with pytest.raises(ValueError, match="different candidate"):
        merge_registry_audits(prior, other_candidate, registry_rasters_expected=2)

    other_thresholds = _audit([_row("new-sha", "new.tif")])
    other_thresholds["jaccard_limit"] += 0.01
    with pytest.raises(ValueError, match="different uniqueness thresholds"):
        merge_registry_audits(prior, other_thresholds, registry_rasters_expected=2)

    incomplete = _audit([_row("new-sha", "new.tif")], complete=False)
    with pytest.raises(ValueError, match="delta is incomplete"):
        merge_registry_audits(prior, incomplete, registry_rasters_expected=2)


def test_incremental_summary_cannot_certify_missing_registry_rows():
    meta = {
        "my_file": "candidate.tif",
        "my_dots": 10,
        "candidate_decoded_sha256": "pixels",
        "candidate_file_sha256": "file",
        "candidate_rank_variation": True,
    }
    result = summarize_registry_rows(
        [_row("one", "one.tif")],
        registry_rasters_expected=2,
        complete_accessible_scan=True,
        source_errors=[], candidate_meta=meta, scope="incomplete fixture",
    )
    assert not result["complete_accessible_scan"]
    assert not result["unique"]
    assert result["stop_required"]
