#!/usr/bin/env python3
"""Apply the literal registry rule to historical measurements, fail closed.

This interpretation-only step neither fetches rasters nor reruns measurements.
It retains the historical forward-overlap values, withdraws the invalid stride-3
rho values, and cannot certify the later 696-row inventory. The 85 measured
forward-overlap firings already exceed the 0.70 limit; incomplete scope also
blocks clearance. Dense surfaces and support-asymmetric peers are not exempted.
Representation-aware summaries remain diagnostic only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70


def main() -> int:
    path = ROOT / "evidence/uniqueness_h59i.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    historical_rho = data.pop(
        "worst_spearman_full_footprint",
        data.get("historical_worst_stride_rho_estimate"),
    )
    data["historical_worst_stride_rho_estimate"] = historical_rho
    data["exact_worst_spearman_full_footprint"] = None
    data["worst_rho_submission"] = None
    data["rank_correlation_method"] = (
        "NOT VERIFIED: historical per-raster rho values withdrawn because the screen used "
        "stride-3 sampling and incorrect tied-rank handling; no exact rerun was performed"
    )
    for row in rows:
        row.pop("spearman_full_footprint", None)
        row.pop("historical_stride_rho_estimate", None)
        row.pop("rank_correlation_status", None)
    checked = [row for row in rows if "error" not in row]
    errors = [row for row in rows if "error" in row]
    # The historical screen sampled every third cell and assigned tied values
    # ordinal ranks, so its rho column is NOT an exact Spearman measurement.
    # Do not use it for a pass/fail decision or relabel it as full-footprint rho.
    overlap_hits = [row for row in checked
                    if row.get("my_dots_within_3px_of_theirs", 0.0) > OVERLAP_LIMIT]
    identical_hits = [row for row in checked
                      if row.get("identical_decoded_predictions")]
    literal_hits = {row.get("submission", row.get("sha256")): row
                    for row in [*overlap_hits, *identical_hits]}
    historical_expected = int(data.get(
        "historical_registry_rasters_expected",
        data.get("registry_rasters_expected", 0),
    ))
    latest_index_path = ROOT / "evidence/registry_refreshed_20261010T2001.json"
    latest_index = json.loads(latest_index_path.read_text(encoding="utf-8")) if latest_index_path.exists() else {}
    latest_expected = int(latest_index.get("n_unique_grid_rasters", 0))
    prior_witness_sha = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"
    prior_witness_rows = [row for row in latest_index.get("rasters", [])
                          if row.get("sha256") == prior_witness_sha]
    prior_witness = prior_witness_rows[0] if len(prior_witness_rows) == 1 else {}
    witness_candidates = [ROOT / prior_witness.get("cache_file", ".cache/registry/missing-witness.tif")]
    if prior_witness.get("sources"):
        source_path = prior_witness["sources"][0].split(":", 1)[-1]
        witness_candidates.append(ROOT / source_path)
    witness_receipts = [
        ROOT / "evidence/literal_gate_witness_verification_20261010.json",
        ROOT / "evidence/uniqueness_saturation_certificate.json",
        ROOT / "evidence/session7_registry_precheck.json",
    ]
    witness_claim = dict(
        evidence_class="PRIOR-SESSION CLAIM (not independently verified in this checkout)",
        source=(prior_witness.get("sources", [None])[0] if prior_witness else None),
        sha256=prior_witness.get("sha256", prior_witness_sha),
        latest_index_entry_present=bool(prior_witness),
        raster_available_locally=any(path.is_file() for path in witness_candidates),
        verification_receipts_present=all(path.is_file() for path in witness_receipts),
        universal_coverage_confirmed=None,
        relied_on_for_current_stop=False,
        status=("NOT VERIFIED: witness TIFF and/or its verification receipts are absent; do not rely on the "
                "universal-blocker claim. The 85 retained historical literal overlap firings independently "
                "require STOP."),
    )
    # A 695-raster historical audit cannot certify the later 696-raster index.
    complete = bool(
        data.get("complete_accessible_scan")
        and historical_expected > 0
        and len(checked) == historical_expected
        and not errors
        and not data.get("missing_or_invalid", 0)
        and latest_expected == historical_expected
    )
    gate_pass = bool(complete and not literal_hits and data.get("my_dots", 0) > 0)

    worst_overlap = max(checked, key=lambda r: r.get("my_dots_within_3px_of_theirs", -1.0), default=None)
    data["strict_literal_gate"] = dict(
        evidence_class="REGISTRY-MEASUREMENT (literal protocol interpretation)",
        support_definition="finite prediction value > 0 within the valid footprint",
        rho_limit=RHO_LIMIT,
        forward_overlap_limit=OVERLAP_LIMIT,
        overlap_radius_px=3.0,
        rank_correlation_method="NOT VERIFIED: the historical screen used stride-3 sampling and incorrect tie ranking; exact shared-gate rho was not rerun",
        historical_registry_rows_expected=historical_expected,
        latest_index_rows_expected=latest_expected or None,
        latest_index_rows_checked_against_candidate=0,
        latest_index_file=(str(latest_index_path.relative_to(ROOT)) if latest_index else None),
        registry_rows_checked=len(checked),
        registry_snapshot_scope=data.get("scope"),
        complete_accessible_scan=complete,
        rho_hit_count=None,
        exact_rank_correlation_rerun=False,
        overlap_hit_count=len(overlap_hits),
        identical_prediction_hit_count=len(identical_hits),
        total_unique_hit_count=len(literal_hits),
        worst_overlap=(worst_overlap.get("my_dots_within_3px_of_theirs") if worst_overlap else None),
        worst_overlap_submission=(worst_overlap.get("submission") if worst_overlap else None),
        worst_rank_correlation=None,
        rank_correlation_submission=None,
        historical_stride_rho_estimate=historical_rho,
        historical_stride_rho_is_operative=False,
        passes=gate_pass,
        stop_required=not gate_pass,
        decision=("PASS (indexed inventory only; not organizer-complete)" if gate_pass
                  else "DUPLICATE / INCOMPLETE — STOP; no slot or production export"),
        representation_aware_exemptions_applied=False,
        prior_witness_claim=witness_claim,
        note=(
            "The previous representation-aware interpretation was unauthorized by the standing "
            "protocol. It is retained only as a diagnostic; it does not override any literal hit."
        ),
    )
    data["historical_registry_snapshot"] = data.get("scope")
    data["historical_registry_scan_complete"] = bool(data.get("complete_accessible_scan"))
    data["historical_registry_rasters_expected"] = historical_expected
    data["registry_rasters_expected"] = latest_expected or historical_expected
    data["registry_rasters_checked"] = len(checked)
    data["complete_accessible_scan"] = complete
    data["invalid_measurement_rows"] = len(errors)
    data["historical_rows_unmeasured"] = max(0, historical_expected - len(checked))
    data["latest_index_rows_checked_against_candidate"] = 0
    data["latest_index_rows_unscanned"] = latest_expected
    data["missing_or_invalid"] = len(errors) + max(0, historical_expected - len(checked))
    data["scope"] = (
        f"Final-dot overlap rows measured against historical {historical_expected}-raster scan; "
        f"the later {latest_expected or 'unknown'}-raster inventory has not been scanned against this candidate. "
        "The 85 historical literal overlap firings alone trigger the required stop."
    )
    data["unique"] = gate_pass
    data["registry_duplicate_evidence"] = bool(literal_hits)
    data["verdict_registry_leg"] = (
        "literal gate pass within the dated accessible index; not organizer-complete"
        if gate_pass else "DUPLICATE / STOP under the literal protocol"
    )
    data["verdict"] = "negative" if not gate_pass else "eligible-for-separate-selector-only"
    data["strict_duplicate_rows"] = sorted(literal_hits)
    data["reviewed_interpretation_utc"] = "2026-10-11"
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf-8")

    public_path = ROOT / "docs/data/uniqueness_h59i.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(
        json.dumps({key: value for key, value in data.items() if key != "rows"},
                   indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    card_path = ROOT / "evidence/run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["registry_comparison"] = data["strict_literal_gate"]
    card["correlation_overlap_vs_registry"] = dict(
        exact_spearman="not verified; historical stride/tie implementation withdrawn",
        historical_stride_rho_estimate=historical_rho,
        literal_forward_overlap_hits=len(overlap_hits),
        worst_dot_overlap=(worst_overlap.get("my_dots_within_3px_of_theirs") if worst_overlap else None),
        worst_overlap_peer=(worst_overlap.get("submission") if worst_overlap else None),
        latest_known_index_rows=latest_expected or None,
        latest_known_index_scanned=False,
        strict_gate_passes=gate_pass,
    )
    card["submission"]["download_status"] = "DOWNLOAD-VALID FOR INSPECTION ONLY"
    card["submission"]["cleared_to_submit"] = False
    card["submission"]["submit_status"] = (
        "NO — literal 3-px registry overlap gate fails; latest known 696-raster index was not scanned "
        "against this candidate; historical HOLDOUT-DTI is below the comparator and exploratory only; "
        "target-group background is not implemented; no slot spent"
    )
    card["submission"]["weekly_slot_used"] = False
    card["submission"]["weekly_slot_authorized"] = False
    card["download_status"] = "DOWNLOAD-VALID FOR INSPECTION ONLY; NOT CLEARED TO SUBMIT"
    card["verdict"] = "negative"
    card["decision"] = "STOP — no selector promotion and no weekly slot recommendation"
    for relative in ("evidence/run_card.json", "docs/data/run_card.json"):
        (ROOT / relative).write_text(json.dumps(card, indent=1, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print(json.dumps({
        "complete_accessible_scan": complete,
        "rows_checked": len(checked),
        "overlap_hits": len(overlap_hits),
        "exact_rank_correlation_rerun": False,
        "historical_stride_rho_estimate": data["strict_literal_gate"]["historical_stride_rho_estimate"],
        "worst_overlap": data["strict_literal_gate"]["worst_overlap"],
        "strict_gate_passes": gate_pass,
        "decision": data["strict_literal_gate"]["decision"],
    }, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
