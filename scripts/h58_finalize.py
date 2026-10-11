#!/usr/bin/env python3
"""Assemble the H58 run card from measured evidence files. Strict, no fallbacks.

Refuses to write a card when a required receipt is missing, when the raster hash
drifted after its audit, or when the gate summaries do not describe the bytes on
disk.  A verdict of ``promote`` is *never* emitted by this script: it can only
report what the gates and the paired holdout measured, and set
``okay_to_submit`` from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
DOCS = ROOT / "docs"


def load(name: Path) -> dict:
    if not name.is_file():
        raise SystemExit(f"required evidence missing: {name}")
    return json.loads(name.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default=str(EVID / "h58_build.json"))
    ap.add_argument("--holdout", default=str(EVID / "h58_holdout.json"))
    ap.add_argument("--structure", default=str(EVID / "h58_structure.json"))
    ap.add_argument("--canary", default=str(EVID / "h58_canary.json"))
    ap.add_argument("--gates-dots", default=str(EVID / "h58_uniqueness_final_dots.json"))
    ap.add_argument("--gates-surface", default=str(EVID / "h58_uniqueness_surface_before_placement.json"))
    ap.add_argument("--profile", default=str(EVID / "h58_registry_profile.json"))
    ap.add_argument("--live", default=str(EVID / "live_submission_patterns.json"))
    a = ap.parse_args()
    build = load(Path(a.build))
    hold = load(Path(a.holdout))
    struct = load(Path(a.structure))
    can = load(Path(a.canary))
    gd = load(Path(a.gates_dots))
    gs = load(Path(a.gates_surface))
    prof = load(Path(a.profile))
    pat = load(Path(a.live))
    if hold.get("status", "").startswith("negative"):
        raise SystemExit("holdout experiment is negative for a different reason (canary); refusing")
    path = ROOT / build["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != build["raster_sha256"] or digest != gd.get("candidate_file_sha256"):
        raise SystemExit(f"raster hash drifted after its audit: {digest}")
    if digest != gs.get("candidate_file_sha256"):
        raise SystemExit("surface audit describes different bytes than the delivered raster's build record")
    arm = build["arm"]
    dot_score = hold["scores"][arm]
    surf_score = hold["soft_surface_holdout"]["scores"][arm]
    gain = hold["candidate_gain_over_reference"] or {}
    max_auc = max(v["discriminative_auc_max"] for v in can["features"].values())
    flags = list(can["flagged"])
    receipt = load(path.with_suffix(".json"))
    for probe, other in ((gd, gs), (gs, gd)):
        if probe["rows"] and len(probe["rows"]) != probe["registry_rasters_expected"]:
            raise SystemExit("gate rows do not cover the registry manifest")
        if probe["complete_accessible_scan"] is not bool(other["complete_accessible_scan"]):
            raise SystemExit("the two gate phases disagree about registry completeness")
    if receipt["approved_for_weekly_slot"]:
        raise SystemExit("writer receipt claims slot approval")
    if not build["validator"]["all_checks_passed"]:
        raise SystemExit("delivered raster fails the local preflight")
    if bool(gd["unique"]) != (not gd["stop_required"]):
        raise SystemExit("gate flags contradict each other")
    if receipt["sha256"] != digest or receipt["zip_sha256"] != build["zip_sha256"]:
        raise SystemExit("writer receipt does not match the bytes on disk")
    card = dict(
        hypothesis=("Secondary strands around a mapped fault are predicted by a *fitted* damage-zone "
                    "envelope whose half-width scales sub-linearly with mapped-component length "
                    "(displacement proxy), modulated by distance x host-strike structure, with "
                    "handed (sense-conditioned) radial obliquity tested rather than assumed, and "
                    "with dots kept off the fully-penalised catalogue flank."),
        mechanism=("Visible-only distance/length/strike geometry feeds a histogram-boosted intensity "
                   "fitted on a buffered whole-component hide-and-recover draw; W(L)=w0*(L/L0)**gamma "
                   "is estimated by bin-wise quantile regression with a bin bootstrap; the greedy "
                   "allocator accepts a dot only under the exact marginal-DTI test; the dot budget is "
                   "shrunk by the fitted fraction of withheld positives inside the kept zone."),
        named_non_fault_process_that_could_mimic_it=(
            "Map-trace fragmentation and mapping effort (which set the connected-component length used "
            "as the displacement proxy), lithologic contacts, magnetised dikes, erosional range-front "
            "scarps and digitisation vertices all produce oblique near-fault lineaments without "
            "secondary fault slip."),
        holdout_dot_dti={**dot_score, "representation": "holdout binary allocation", "arm": arm,
                         "split_version": hold["split_version"], "draw_seed": hold["draw_seed"]},
        holdout_surface_dti={**surf_score, "representation": "soft pre-placement surface", "arm": arm},
        paired_gain_candidate_minus_reference={**gain, "retained": hold["candidate_retained"]},
        flank_policy_effect_on_holdout=hold["flank_policy_effect"],
        width_law=struct["width_law_all_folds"],
        structure_measurements=dict(
            withheld_positive_pixels=struct["withheld_positive_pixels"],
            base_rate=struct["base_rate"],
            withheld_distance_quantiles_px=struct["structure"]["distance_positive_quantiles_px"],
            domain_distance_quantiles_px=struct["structure"]["distance_domain_quantiles_px"],
            withheld_distance_enrichment=struct["withheld_distance_enrichment"],
            relative_strike_median_withheld_deg=struct["relative_strike"]["median_withheld"],
            relative_strike_median_visible_null_deg=struct["relative_strike"]["median_visible"],
            obliquity_enrichment=struct["obliquity_histogram"]["enrichment"],
            side_asymmetry_by_sense_record=struct["side_asymmetry_by_sense_record"],
            fraction_pos_in_kept_zone=struct["fraction_pos_in_kept_zone"]),
        leakage_canary=dict(max_discriminative_auc=max_auc, flags=flags, canary_evidence_class=can["evidence_class"],
                            threshold=0.90, features=len(can["features"]),
                            evidence_class=can["evidence_class"]),
        correlation_overlap_vs_registry=dict(
            final_dots={k: gd[k] for k in ("registry_rasters_expected", "registry_rasters_checked",
                                           "complete_accessible_scan", "worst_spearman_full_footprint",
                                           "worst_rho_submission", "worst_dot_overlap",
                                           "worst_overlap_submission", "worst_jaccard_dot_sets",
                                           "duplicate_count", "byte_unique_among_checked",
                                           "pixel_unique_among_checked", "unique",
                                           "candidate_rank_variation", "jaccard_diagnostic_only")},
            surface_before_placement={k: gs[k] for k in ("worst_spearman_full_footprint",
                                                         "worst_dot_overlap", "worst_jaccard_dot_sets",
                                                         "duplicate_count", "unique")},
            registry_profile={k: prof[k] for k in ("rasters_indexed", "read", "missing", "sha_mismatch",
                                                   "universal_overlap_blockers", "allowed_pixels")}),
        surface_before_placement=dict(checked=True,
                                      protocol_pass=bool(gs["unique"]),
                                      note=("the fitted intensity is dense by construction, so the "
                                            "literal 3-px support rule cannot separate it from any "
                                            "dense prior; this is reported, not exempted")),
        final_dots=dict(status="generated", dots=build["dots"], cap=build["dot_cap"],
                        protocol_pass=bool(gd["unique"]),
                        stop_required=bool(gd["stop_required"]),
                        overlap_limit=gd["overlap_limit"], rho_limit=gd["rho_limit"],
                        byte_unique=bool(gd["byte_unique_among_checked"]),
                        pixel_unique=bool(gd["pixel_unique_among_checked"]),
                        duplicate_count=int(gd["duplicate_count"]),
                        registry_scope="all hash-pinned grid rasters present in this workspace"),
        raster_sha256=digest, validator_output=build["validator"],
        file=str(Path(build["file"])), zip_file=str(Path(build["file"]).with_suffix(".zip")),
        surface_array_sha256=gs["candidate_decoded_sha256"],
        dots_array_sha256=gd["candidate_decoded_sha256"],
        receipt_file=build.get("receipt_file"),
        submission_name=build["submission_name"], submission_note=build["submission_note"],
        submission_note_chars=build["submission_note_chars"],
        live_submission_evidence=dict(
            evidence_class=pat["evidence_class"], scored_rasters=len(
                [r for r in pat["rows"] if r.get("owner_reported_score") and r.get("dots")]),
            rank_correlations={k: v["spearman_vs_owner_reported_score"]
                               for k, v in pat["rank_correlations"].items()},
            rank_correlation_p={k: v["p_value"] for k, v in pat["rank_correlations"].items()},
            best_vs_base=pat["best_vs_base"], caveats=pat["caveats"],
            use=("shape of the support and the ~10k-per-quadrant budget rule only; these are "
                 "owner-reported numbers attached to files by filename, never a forecast")),
        # Aliases the merged site QA reads: holdout_dti is the soft-surface reading
        # (the pre-placement representation), sha256 is the delivered bytes, and the
        # content-distinctness boolean is stated separately from the literal gate so a
        # reader cannot confuse "not a copy" with "gate cleared".
        holdout_dti={**surf_score, "representation": "soft pre-placement surface", "arm": arm},
        sha256=digest,
        content_distinct_from_all_audited_rasters=bool(
            gd["byte_unique_among_checked"] and gd["pixel_unique_among_checked"]),
        evaluator_version=hold["evaluator_version"],
        experiments_used=3, submission_slots_used=0,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        okay_to_download=True,
        okay_to_submit=bool(gd["unique"]) and not gd["stop_required"],
        verdict=("promote-candidate" if (gd["unique"] and not gd["stop_required"]) else "negative"),
        verdict_reason=(
            "Gate cleared: unique against every accessible prior raster."
            if gd["unique"] else
            f"Literal drift gate not cleared: {gd['duplicate_count']} of {gd['registry_rasters_checked']} "
            f"accessible priors trip a threshold ({prof['universal_overlap_blockers']} priors have "
            "positive support on every allowed cell, which makes the 0.70 rule unsatisfiable for any "
            "non-empty candidate). Byte- and pixel-identity checks pass against all of them."),
        limitations=[
            "HOLDOUT-DTI measures recovery of mapped catalogue geometry, not discovery of unpublished "
            "expert faults; the repository's own record is that this instrument does not rank live scores.",
            "The 3 px context collar makes withheld positives structurally impossible within 2 px of a "
            "visible trace, so the holdout cannot price the catalogue-flank suppression the live metric "
            "does; the suppression is therefore justified by organiser thread 11516, not by this CI.",
            "Mapped-component length is a noisy displacement proxy; gamma describes that proxy, not "
            "measured throw.",
            "No geophysical band was used this session: the 19-band feature cache is gitignored and was "
            "not restored, so scarp/magnetic corroboration is untested here.",
            "Scores of sibling repositories are OWNER-REPORTED prompt text; no organiser receipt ties a "
            "score to a file hash.",
        ],
        evidence=dict(plan="evidence/h58_experiment_plan.json", structure="evidence/h58_structure.json",
                      holdout="evidence/h58_holdout.json", canary="evidence/h58_canary.json",
                      build="evidence/h58_build.json", uniqueness_final_dots="evidence/h58_uniqueness_final_dots.json",
                      uniqueness_surface="evidence/h58_uniqueness_surface_before_placement.json",
                      registry_profile="evidence/h58_registry_profile.json",
                      live_patterns="evidence/live_submission_patterns.json"),
    )
    for dest in (EVID / "run_card_current.json", DOCS / "data" / "run_card.json"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: card[k] for k in ("verdict", "okay_to_download", "okay_to_submit",
                                           "submission_name")}, indent=2))
    print(f"final-dots gate: unique={gd['unique']} worst_rho={gd['worst_spearman_full_footprint']} "
          f"worst_overlap={gd['worst_dot_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
