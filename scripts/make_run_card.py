#!/usr/bin/env python3
"""Assemble an auditable H59-I run card from existing receipts; never fit or submit."""
from __future__ import annotations

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    exp1 = read("evidence/intensity_exp1_holdout.json")
    exp2 = read("evidence/intensity_exp2_utility_table.json")
    exp3 = read("evidence/intensity_exp3_final.json")
    profile = read("evidence/intensity_dot_profile.json")
    receipt = read("downloads/gems59-h59i-intensity-lattice-20261011T000000Z.json")
    uniqueness = read("evidence/uniqueness_h59i.json")
    registry = read("evidence/registry_refreshed_20261010T2001.json")
    leaderboard = read("evidence/leaderboard_snapshot_2026-10-11.json")
    data_prep = read("evidence/data_preparation.json")
    bank = read("evidence/feature_bank.json")
    if not all((exp1, exp2, exp3, receipt, uniqueness, registry, leaderboard)):
        raise SystemExit("missing one or more evidence receipts; run card not written")

    scores = exp3["pooled"]["scores"]
    final = scores["final_c2"]
    note = "H59-I: presence-only MaxEnt/Poisson intensity, effort covariates, 1 km new-fault-density zoning, holdout break-even lattice."
    best_ref = exp3["published_reference"]["relay_bend_anatomy_dti"]
    historical_scan = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        audit_file="evidence/uniqueness_h59i.json",
        historical_rows_expected=uniqueness.get("historical_registry_rasters_expected",
                                                uniqueness.get("registry_rasters_expected")),
        historical_rows_checked=uniqueness.get("registry_rasters_checked"),
        historical_snapshot=uniqueness.get("historical_registry_snapshot", uniqueness.get("scope")),
        literal_forward_overlap_hits=sum(
            row.get("my_dots_within_3px_of_theirs", 0.0) > 0.70
            for row in uniqueness.get("rows", []) if "error" not in row),
        worst_forward_overlap=uniqueness.get("worst_dot_overlap"),
        worst_overlap_peer=uniqueness.get("worst_overlap_submission"),
        exact_full_footprint_rank_correlation="NOT VERIFIED; historical stride/tie implementation withdrawn",
        historical_stride_rho_estimate_not_operative=uniqueness.get(
            "historical_worst_stride_rho_estimate",
            uniqueness.get("worst_spearman_full_footprint")),
        latest_known_index="evidence/registry_refreshed_20261010T2001.json",
        latest_known_index_utc=registry.get("generated_utc"),
        latest_known_index_rasters=registry.get("n_unique_grid_rasters"),
        latest_index_scanned_against_candidate=False,
        candidate_surface_gate_was_run=False,
        candidate_final_dot_gate_passes=False,
        strict_gate_passes=False,
        result="DUPLICATE / STOP; no submit clearance",
        scope=registry.get("scope"),
    )
    card = dict(
        schema="gems59.run-card.v2",
        evidence_class="RUN-CARD (local research; no organizer submission receipt)",
        session="59GEMSDOE / H59-I / arena/a4652e26-59gemsdoe",
        lane="Presence-only point-process intensity, new-fault-density zoning, break-even lattice",
        hypothesis=(
            "Geophysical and catalogue-proxy covariates may rank where additional mapped-fault pixels "
            "occur within 1 km blocks. This historical prototype did not implement the requested "
            "target-group background, so the hypothesis is not fully tested."
        ),
        mechanism=(
            "Historical H59-I: MaxEnt-form logistic fit on 60,988 catalogue presences vs 120,000 "
            "uniform background cells, 13 geophysical/cat-proxy covariates, 1 km new-fault-density "
            "zoning, holdout-derived 4 px spacing and 2 px catalogue collar. Mapping-effort covariates "
            "were trace density and relief; no independent target-group background was available. "
            "The prior H59-I run also used the full-catalogue prevalence in its case-control correction; "
            "the runner has been corrected to use visible-only prevalence, but no rerun was performed."
        ),
        named_non_fault_mimic=(
            "Mapping-effort bias / survey footprint: access, map-sheet boundaries, and survey flight-line "
            "levelling can create correlated traces or geophysical lineaments without a fault."
        ),
        requested_target_group_background=dict(
            status="NOT IMPLEMENTED",
            reason=("No pinned independent same-effort non-target fault/mapping inventory exists; "
                    "using the response catalogue itself would be circular."),
            requirement=("Obtain a defensible independent mapping-effort source, pin and grid-check it, "
                         "derive it split-safely, then rerun the leakage canary and spatial holdout."),
        ),
        holdout_result=dict(
            evidence_class="HOLDOUT-DTI (historical exploratory run; not a leaderboard score)",
            evaluator_version=exp3["evaluator_version"],
            split=exp3["split"],
            alpha=exp3["pooled"]["alpha"], beta=exp3["pooled"]["beta"],
            triangular_radius_m=exp3["pooled"]["triangular_radius_m"],
            selected_arm="final_c2",
            dti=final["dti"], ci95=final["ci95"],
            withheld_positive_pixels=final["withheld_positive_pixels"],
            best_comparable_holdout_reference=best_ref,
            reference_ci95=exp3["published_reference"]["relay_bend_anatomy_ci"],
            beats_reference=bool(final["dti"] > best_ref),
            protocol_caveat=("Historical fit used a case-control prior computed from the full catalogue; "
                             "the repaired visible-only prior has not been rerun. The requested target-group "
                             "background is also absent. Treat this as exploratory, not selection evidence."),
            old_collar0=dict(dti=scores["final_c0"]["dti"], ci95=scores["final_c0"]["ci95"]),
            old_3px_baseline=dict(dti=scores["base_s3"]["dti"], ci95=scores["base_s3"]["ci95"]),
        ),
        leakage_canary=dict(
            evidence_class="LEAKAGE-CANARY (AUC; historical exploratory receipt)",
            features_tested=len(exp1.get("canary", {})),
            maximum_auc=max((row.get("auc_max", 0.0) or 0.0)
                            for row in exp1.get("canary", {}).values()),
            any_feature_over_0_90=any(row.get("leakage_flag", False)
                                      for row in exp1.get("canary", {}).values()),
            caveat="Canary does not repair the full-catalogue case-control prior or supply a target-group background.",
        ),
        zoning_utility=dict(
            evidence_class=exp2.get("evidence_class"),
            evaluator_version=exp2.get("evaluator_version"),
            break_even_min_density=exp2.get("break_even_min_density"),
            selected_spacings={name: row.get("selected")
                               for name, row in exp2.get("utility_table", {}).items()},
            caveat=exp2.get("selection_caveat"),
        ),
        registry_comparison=historical_scan,
        raster=dict(
            file=receipt.get("file"),
            sha256=receipt.get("sha256"),
            bytes=receipt.get("bytes"),
            positive_pixels=receipt.get("validator", {}).get("emitted_positive_pixels"),
            download_status="DOWNLOAD-VALID FOR INSPECTION ONLY; NOT CLEARED TO SUBMIT",
            validator_result=receipt.get("validator"),
        ),
        leaderboard_context=dict(
            evidence_class=leaderboard.get("label"),
            source=leaderboard.get("source"),
            fetched=leaderboard.get("fetched"),
            rows=leaderboard.get("rows"),
            note=("These are participant-level public-board values, not H59-I file receipts and not private-set "
                  "scores. H33-2-B2 file-to-score attribution is unverified; the owner-maintained H33 page "
                  "calls it UNSCORED and says no organizer score exists for artifacts in that repository."),
            h33_audit_projection=dict(
                evidence_class="PROJECTION (not a score)",
                field="predicted_leaderboard_dti",
                value=0.27467315976932155,
                source="GEMSDOE32 owner-maintained audit at commit 3ae286af599fdf86dbb0f333591f8273efcee806",
            ),
        ),
        submission=dict(
            name="gems59-h59i-intensity-lattice-20261011",
            note=note,
            note_chars=len(note),
            download_status="DOWNLOAD-VALID FOR INSPECTION ONLY",
            cleared_to_submit=False,
            submit_status=("NO — strict literal registry overlap gate fails; latest known index was not scanned "
                           "against this candidate; holdout is below comparator and exploratory only; "
                           "target-group background is unimplemented."),
            weekly_slot_used=False,
            weekly_slot_authorized=False,
        ),
        review_passes=[
            dict(pass_number=1, scope="Evidence provenance",
                 result="Leaderboard values separated from file attribution; owner projection is not called a score."),
            dict(pass_number=2, scope="Scientific / holdout validity",
                 result="Historical result retained with leakage and target-group limitations; no new experiment run."),
            dict(pass_number=3, scope="Registry / delivery governance",
                 result="85 literal overlap firings trigger STOP; current index incomplete for candidate; download-only."),
        ],
        limitations=[
            "Target-group background is not implemented; no independent same-effort source is pinned.",
            "The historical holdout case-control prior used full-catalogue prevalence; corrected runner not rerun.",
            "Historical final-dot comparison found literal overlap failures on the 695-raster scan; latest known 696-raster index was not scanned against this candidate.",
            "The previous rho statistic was stride-sampled with incorrect tie ranks; exact full-footprint rank correlation is not reported.",
            "No pre-placement surface gate was performed for the historical candidate; gated scripts now stop before new placement.",
            "A prior-session universal-support-witness claim is not verified here: its TIFF and three referenced verification receipts are absent from this checkout; the current STOP does not rely on it.",
            "The old feature cache is full-catalogue geometry; holdout lanes must rebuild catalogue features from visible faults. The active intensity runner does so for its trace-density planes.",
        ],
        experiments_used=3,
        experiment_budget="3 experiments or 2 hours; exhausted for this run. No competition slot used.",
        verdict="negative",
        decision="STOP — no selector promotion and no weekly slot recommendation",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        data_preparation=data_prep,
        feature_bank=dict(
            source_sha256=(bank or {}).get("source_sha256"),
            bank_shape=(bank or {}).get("bank_shape"),
            cache_build_receipt="evidence/feature_bank.json",
            catalogue_geometry_warning=(bank or {}).get("holdout_warning"),
        ),
    )
    # Direct field names required by the standing machine-readable run-card brief.
    card["mechanism_summary"] = card["mechanism"]
    card["holdout_dti_ci95"] = dict(
        evidence_class=card["holdout_result"]["evidence_class"],
        evaluator_version=card["holdout_result"]["evaluator_version"],
        withheld_positive_pixels=card["holdout_result"]["withheld_positive_pixels"],
        dti=card["holdout_result"]["dti"], ci95=card["holdout_result"]["ci95"],
    )
    card["correlation_overlap_vs_registry"] = dict(
        exact_spearman="not verified", literal_forward_overlap_hits=historical_scan["literal_forward_overlap_hits"],
        worst_dot_overlap=historical_scan["worst_forward_overlap"],
        strict_gate_passes=False,
    )
    card["raster_sha256"] = receipt.get("sha256")
    card["validator_output"] = receipt.get("validator")
    card["submission_name"] = card["submission"]["name"]
    card["submission_note"] = card["submission"]["note"]

    for relative in ("evidence/run_card.json", "docs/data/run_card.json"):
        (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / relative).write_text(json.dumps(card, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"verdict": card["verdict"], "holdout_dti": final["dti"],
                      "strict_registry_pass": False, "download_status": card["raster"]["download_status"],
                      "submit_clearance": card["submission"]["cleared_to_submit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
