"""Write the lane's single JSON run card from the receipts on disk.

Protocol item 5: one card per release with hypothesis, mechanism, named mimic,
holdout DTI + CI, registry correlation/overlap, raster sha256, validator output,
submission name + note, and a promote/negative verdict.  Nothing here is a
competition score; the labels say which evidence class each number belongs to.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EV = ROOT / "evidence"
DOWN = ROOT / "docs" / "downloads"


def load(name: str):
    return json.loads((EV / name).read_text())


def main() -> int:
    sub = json.loads(next(DOWN.glob("*.tif")).with_suffix(".json").read_text())
    exp1 = load("exp1_catalogue_holdout.json")
    exp2 = load("exp2_sgmc_novelty.json")
    exp3 = load("exp3_canary_robustness.json")
    emit = load("emit_receipt.json")
    uni = load("uniqueness_lane_v1.json")["summary"]
    pooled = exp1["pooled"]["scores"]
    # the deployed arm of exp1 is the design-projection arm ("table0"); the receipt
    # identifies the comparable control instead of the candidate, so resolve it here
    cand = "table0" if "table0" in pooled else sorted(pooled)[0]
    lane = pooled[cand]
    prox = pooled[exp1["pooled"]["best_comparable_control"]]
    uni_literal = uni["literal_leg"]
    uni_repr = uni["representation_aware_leg"]
    dense = [r for r in json.loads((EV / "uniqueness_lane_v1.json").read_text())["rows"]
             if r.get("overlap_class") == "dense-support"]

    dec = json.loads((EV / "decile_calibration_lane1ac8.json").read_text())
    em = np.asarray(np.load(ROOT / ".cache" / "field" / "emitted.npy"), bool)
    grid = None
    try:
        from gems59.grid import load_grid  # noqa: PLC0415
        from scipy import ndimage as ndi  # noqa: PLC0415
        grid = load_grid(ROOT / "data" / "bridge")
        dist = ndi.distance_transform_edt(~grid.catalogue)
        d = dist[em]
        live_profile = dict(
            evidence_class="MEASURED",
            dots=int(em.sum()),
            median_distance_to_catalogue_px=float(np.median(d)),
            fraction_within_2px_of_catalogue=float((d <= 2).mean()),
            fraction_on_catalogue=float((em & grid.catalogue).sum() / max(int(em.sum()), 1)),
            note="the owner-reported live structure (n=15) is: log-dot-count -0.810, dots within "
                 "2 px of the catalogue -0.544, median distance to the catalogue +0.766; this "
                 "profile is reported so the release can be compared with that structure, not "
                 "because it is a score")
    except Exception as exc:  # pragma: no cover - defensive
        live_profile = dict(evidence_class="MEASURED", error=str(exc))

    card = {
        "release": "lane v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "intensity: 1 km new-fault-density zoning + presence-only point-process "
                "intensity with target-group background; sparse 3 px lattice above break-even",
        "hypothesis": (
            "Unmapped fault density is predictable at 1 km block scale from the same "
            "geophysical and topographic evidence that predicts mapped faults, once the "
            "mapping process (observer effort) is absorbed by a target-group background; "
            "deploying only the evidence terms and re-deriving the block break-even turns "
            "that density into a sparse dot field that is not a re-prediction of the "
            "catalogue."),
        "mechanism": (
            "Presence-only logistic (log) intensity fitted on thinned catalogue pixels vs "
            "random background; presence thinning 2 px; 150,000 background points; ridge "
            f"lambda selected by spatial CV; deployment zeroes the intercept and drops "
            f"neglog_dist_catalogue / cat_density_1km (EVIDENCE_FAMILY). Block shares -> "
            "expected scored-fault pixels per 1 km block -> absolute-density strata "
            "(edges 2.5/5/10/20 px) -> spacing table -> lattice sites inside the "
            "study-area footprint minus the mapped catalogue."),
        "named_non_fault_process_that_could_mimic_it": (
            "Map-scale geophysical texture that correlates with structure without being a "
            "fault: lithologic and cover contacts, magnetic dike swarms, erosional and "
            "range-front lineaments, and flight-line acquisition artefacts; plus the "
            "observer process itself -- where the region was mapped well (and at what "
            "scale) controls where new faults can be recognised, which the effort "
            "covariates only partially absorb."),
        "experiments_used": 3,
        "submission_slots_used": 0,
        "evidence_holdout_catalogue": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": exp1["pooled"]["evaluator_version"],
            "split_version": exp1.get("split_version") or exp1["holdout"].get("split_version"),
            "withheld_positive_pixels": exp1["holdout"]["withheld_positive_pixels"],
            "dti": lane["dti"], "ci95": lane["ci95"],
            "arm": cand,
            "dots": exp1["arms"][cand]["dots"],
            "controls": {k: pooled[k]["dti"] for k in pooled if k != cand},
            "control_cis": {k: pooled[k]["ci95"] for k in pooled},
            "note": "Catalogue hide-and-recover; withheld pixels are a subset of the mapped "
                    "catalogue, so this frame can rank emission rules but cannot rank field "
                    "quality for the real (off-catalogue) target. Not an organizer score.",
        },
        "evidence_independent_frame": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": exp2["pooled"]["evaluator_version"],
            "frame": "SGMC novelty = SGMC faults >3 px from the mapped catalogue (61,664 px); "
                     "a different public catalogue, NOT organizer truth",
            "arms": {k: dict(dti=v["dti"], ci95=v["ci95"], tpw=v["tpw"], fpw=v["fpw"])
                     for k, v in exp2["pooled"]["scores"].items()},
            "paired_difference": (exp2["pooled"].get("paired_differences")
                                  or exp2["pooled"].get("paired_difference")),
        },
        "leakage_canary": {
            "evidence_class": "HOLDOUT-DTI",
            "threshold": exp3["canary_threshold"],
            "worst": exp3["worst_canary"],
            "flags": exp3["canary_flags"],
        },
        "robustness": {
            "evidence_class": "HOLDOUT-DTI",
            "span": exp3["robustness_span"],
            "arms": {k: v["pooled_dti"] for k, v in exp3["robustness"].items()},
            "note": "seeds 59/101/202 x design-projection multiplier 0.6/1.0/1.6",
        },
        "correlation_overlap_vs_registry": {
            "evidence_class": "REGISTRY-MEASUREMENT",
            "rasters_compared": uni["registry_rasters_compared"],
            "literal_leg": {
                "worst_rho_dot_field": _pick(uni_literal["worst_rho_dot_field"]),
                "worst_rho_surface": _pick(uni_literal["worst_rho_surface"]),
                "dense_support_priors": len(dense),
                "note": "overlap leg is 1.0 by construction against dense-support priors; "
                        "reported with an explicit exclusion note, not exempted (IR-59-REPR-01)",
            },
            "representation_aware_leg": {
                "n_sparse_peers": uni_repr["n_sparse_peers"],
                "worst_rho_dot_field": _pick(uni_repr["worst_rho_dot_field"]),
                "worst_overlap_ours_inside_theirs": _pick(uni_repr["worst_overlap_ours_inside_theirs"]),
                "worst_overlap_theirs_inside_ours": _pick(uni_repr["worst_overlap_theirs_inside_ours"]),
            },
            "comparable_peer_tier": uni.get("comparable_peer_tier"),
            "limit_rho": uni.get("rho_limit"),
            "limit_overlap": uni.get("overlap_limit"),
            "saturation_note": uni.get("saturation_note"),
            "verdict": uni["verdict"],
        },
        "surface_before_placement": {
            "evidence_class": "MEASURED",
            "blocks_selected": emit["blocks_selected"],
            "blocks_total": emit["blocks_total"],
            "blocks_per_level": emit["blocks_per_level"],
            "predicted_mass_share_inside_selected": emit["predicted_new_fault_mass_share_inside_selected"],
            "design": emit["design"],
            "note": "shares are relative intensity; the absolute conversion uses a stated design "
                    "projection (g_total), which is sensitivity-tested, never scored",
        },
        "final_dots": {
            "evidence_class": "MEASURED",
            "dots": emit["dots"],
            "dots_on_catalogue": emit["dots_on_catalogue"],
            "dots_outside_footprint": emit["dots_outside_footprint"],
            "spacing_table": emit["table"],
            "lattice_phase": 1,
            "sha256_of_dot_bitmask": hashlib.sha256(
                np.packbits(np.asarray(np.load(ROOT / ".cache" / "field" / "emitted.npy"),
                                       bool)).tobytes()).hexdigest(),
        },
        "raster_sha256": sub["sha256"],
        "file": sub["file"],
        "zip_file": sub["zip_file"],
        "bytes": sub["bytes"],
        "validator_output": {
            "local_validation": sub["validator"],
            "template_report_ok": sub["template_report"].get("ok"),
            "template_report_problems": sub["template_report"].get("problems"),
            "status": sub["status"],
        },
        "submission_name": sub["submission_name"],
        "submission_note": sub["note"],
        "submission_note_chars": sub["note_chars"],
        "decile_calibration": dec,
        "okay_to_download": True,
        "okay_to_submit": False,
        "promotion_requires": "a frame whose truth matches the scored target (a collar-band "
                              "truth), and a release that beats both trivial controls on it; "
                              "owner selector step (protocol item 6); 0 slots used in-session",
        "verdict": "negative",
        "verdict_reason": (
            "DOWNLOAD OK: format-validated (all-finite, values in [0,1], CRS/shape/transform "
            "match the sample submission, 0 dots on the mapped catalogue, 0 outside the "
            "footprint), unique against the registry on every meaningful test -- worst rank "
            "correlation over all 710 priors 0.1142 against a 0.90 limit, worst 3 px overlap "
            "against a comparable prior 0.5129 against a 0.70 limit, and against the only other "
            "intensity-lane release 0.1235 (rank correlation 0.0435). "
            "SUBMIT: NO. The release's own instruments dominate it on both frames available in "
            "this repository, so it is not cleared for a competition slot: "
            "(1) catalogue hide-and-recover frame -- release 0.03339 [0.02282, 0.04447] vs "
            "proximity-only control 0.23490 and uniform-same-count 0.03468, i.e. the release is "
            "the worst of the three; "
            "(2) decile calibration on the same frame -- the release's densest decile of 1 km "
            "blocks carries 0.0495 of the withheld positives for 0.1014 of the footprint area, "
            "a lift of 0.49, i.e. below chance, while the catalogue-distance and "
            "catalogue-density rankings reach lifts of 8.46 and 8.26 (random-ranking "
            "calibration 0.92, so the method is sound); "
            "(3) off-catalogue frame (SGMC novelty) -- release 0.04399 [0.03231, 0.05601] vs "
            "proximity 0.03914 [0.03301, 0.04594], paired difference +0.00485 "
            "[-0.00814, +0.01893] not significant, and a uniform spread of the same number of "
            "dots scores 0.09765, i.e. more than twice the release. "
            "Reading: the evidence-only field is not merely catalogue-free by construction, it "
            "is anti-aligned with the trunk corridors at 1 km block scale (lift < 1 on a frame "
            "whose truth sits on the corridors), and it is beaten by spreading the same budget "
            "over the footprint when the truth is spread. Both frames therefore reject the "
            "emission rule, not just the ranking. The lane's premise is not supported at this "
            "scale by this evidence. "
            "What was NOT tested here and is the next step: a frame whose truth is a band 1-3 km "
            "from the traces (the scored target is newly mapped geometry beside existing "
            "systems -- thread 11536 -- and the owner-reported structure puts winning dots at "
            "~19 px median distance), and an emission that spreads its budget inside the "
            "corridors rather than tiling the top 10 % of blocks. Findings and the frame "
            "diagnosis are recorded as IR-59-FRAME-01."),
        "live_profile": live_profile,
        "limitations": [
            "No organizer receipt: every score here is a holdout proxy or owner-reported context.",
            "Catalogue holdout over-rewards dots (withheld catalogue pixels ~30x denser than the "
            "estimated new-fault population); the tie-break to the smaller budget is a declared "
            "policy, not a correction.",
            "SGMC novelty is a different public catalogue, not the organizer's new faults.",
            "USGS reliability-of-location class cannot be joined to the grid (IR-59-EFFORT-02); "
            "no roads dataset was available in any clone.",
            "Feature stack verified by transport hashes, not by an official provenance receipt.",
            "Target-group background is implemented as uniform background plus effort covariates "
            "(relief, acquisition block); a true target-group background was not available "
            "(IR-59-EFFORT-02).",
            "Both holdout frames' truths differ structurally from the scored target: one sits ON "
            "the traces, the other is a different catalogue spread over the region "
            "(IR-59-FRAME-01). Neither can rank a catalogue-free field.",
        ],
        "evidence_files": sorted(
            name for name in (
                "verify_grid.json", "planes.json", "fit_full.json",
                "exp1_catalogue_holdout.json", "family_comparison.json",
                "exp2_sgmc_novelty.json", "exp3_canary_robustness.json",
                "emit_receipt.json", "uniqueness_lane_v1.json",
                "data_preparation_lane1ac8.json", "registry_refreshed_lane1ac8.json",
                "site_inventory.json", "sibling_release_overlap_lane1ac8.json")
            if (EV / name).is_file()),
    }
    out = EV / "run_card_lane_v1.json"
    out.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verdict=card["verdict"], okay_to_download=card["okay_to_download"],
                          okay_to_submit=card["okay_to_submit"],
                          dti=round(lane["dti"], 5), ci=[round(x, 5) for x in lane["ci95"]],
                          worst_rho_surface=(uni_literal["worst_rho_surface"] or {}).get("rho_surface"),
                          worst_overlap_sparse=(uni_repr["worst_overlap_ours_inside_theirs"] or {}
                                                ).get("ours_inside_theirs_3px")), indent=2))
    return 0


def _pick(row):
    if row is None:
        return None
    out = {}
    for k in ("rho_dot_field", "rho_surface", "ours_inside_theirs_3px",
              "theirs_inside_ours_3px", "repo", "sha256", "sources"):
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, float) and not np.isfinite(v):
            v = None          # constant raster -> undefined rank correlation
        out[k] = v
    return out


if __name__ == "__main__":
    raise SystemExit(main())
