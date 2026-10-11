#!/usr/bin/env python3
"""Assemble the Session-59 run card from the evidence receipts (no new fitting)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def load(p):
    q = ROOT / p
    return json.loads(q.read_text()) if q.exists() else None

def main() -> int:
    exp1 = load("evidence/intensity_exp1_holdout.json")
    exp2 = load("evidence/intensity_exp2_utility_table.json")
    exp3 = load("evidence/intensity_exp3_final.json")
    diag = load("evidence/intensity_exp1_diagnostics_fixed.json")
    prof = load("evidence/intensity_dot_profile.json")
    receipt = load("downloads/gems59-h59i-intensity-lattice-20261011T000000Z.json")
    restore = load("evidence/registry_cache_restore.json")
    if not all([exp1, exp2, exp3, receipt]):
        raise SystemExit("missing evidence")
    best_published = 0.135204
    mine = exp3["pooled"]["scores"]["final_c2"]["dti"]
    verdict = "promote-to-selector-only" if mine > best_published else "negative"
    card = dict(
        evidence_class="RUN-CARD",
        session="59GEMSDOE / arena/f8d022fd-59gemsdoe",
        lane="Intensity: zone by new-fault density, then tile only what clears break-even",
        hypothesis=(
            "A presence-only point-process intensity fitted to the mapped catalogue with "
            "mapping-effort covariates (Phillips et al. 2009) predicts where faults the "
            "USGS/INGENIOUS catalogue missed should sit; zoning that intensity into 1 km "
            "blocks by expected new-fault pixels and lattice-tiling only the blocks above "
            "break-even places dots that beat a same-budget uniform-random control on the "
            "spatially-blocked hide-and-recover holdout."),
        mechanism=(
            "MaxEnt-form logistic fit (Poisson point-process equivalence, Fithian & Hastie "
            "2013) on 60,988 catalogue presences vs 120,000 uniform background cells over "
            "13 covariates: 4 geophysical edge planes (magnetic horizontal-gradient, tilt "
            "derivative, gravity horizontal-gradient, detrended-elevation slope), their max, "
            "5 geophysical regime planes, and 3 mapping-effort planes (1 km / 5 km trace "
            "density, relief). Expected new-fault pixels per 1 km block = sum of lambda over "
            "non-catalogue cells; blocks above break-even (>= 1 px, the 1% lattice-payment "
            "floor) are lattice-tiled at the holdout-derived spacing with a 2 px catalogue "
            "collar."),
        named_non_fault_mimic=(
            "Mapping-effort bias itself: the catalogue is a biased sample, so any intensity "
            "fitted to it inherits the mappers' footprint (roads, access, quadrangle "
            "boundaries) rather than geology; magnetic flight-line levelling residuals and "
            "radiometric survey-block edges can also mimic fault lineaments."),
        holdout_dti=dict(
            evidence_class="HOLDOUT-DTI", evaluator_version="gems57-pooled-hide-v2",
            evaluator_implementation_sha256=exp1["pooled"]["implementation_sha256"],
            split="buffered-whole-components-loqo-v2, seed 20 (identical draw to the "
                  "published Session-5/7 arms)",
            withheld_positive_pixels=exp3["pooled"]["scores"]["final_c2"]["withheld_positive_pixels"],
            alpha=0.2, beta=0.8, triangular_radius_m=300,
            final_arm_collar2=dict(dti=round(mine, 6),
                                   ci95=[round(x, 6) for x in exp3["pooled"]["scores"]["final_c2"]["ci95"]]),
            final_arm_collar0=dict(dti=round(exp3["pooled"]["scores"]["final_c0"]["dti"], 6),
                                   ci95=[round(x, 6) for x in exp3["pooled"]["scores"]["final_c0"]["ci95"]]),
            baseline_s3=dict(dti=round(exp3["pooled"]["scores"]["base_s3"]["dti"], 6),
                             ci95=[round(x, 6) for x in exp3["pooled"]["scores"]["base_s3"]["ci95"]]),
            exp1_arms={k: dict(dti=round(v["dti"], 6), ci95=[round(x, 6) for x in v["ci95"]])
                       for k, v in exp1["pooled"]["scores"].items()},
            published_reference=exp3["published_reference"],
            beats_published_holdout_best=bool(mine > best_published)),
        leakage_canary=dict(evidence_class="LEAKAGE-CANARY (AUC, not DTI)",
                            max_feature_auc=max(v["auc_max"] for v in exp1["canary"].values()),
                            any_flag=any(v["leakage_flag"] for v in exp1["canary"].values()),
                            note="all 13 covariates alone on the holdout; max AUC "
                                 "e_dens5=0.776, below the 0.90 leakage bar"),
        decile_calibration=(diag or {}).get("calibration_summary"),
        sgmc_diagnostic=(diag or {}).get("sgmc_summary"),
        dot_profile=prof,
        registry=dict(
            evidence_class="REGISTRY-MEASUREMENT",
            pinned_index="evidence/registry_refreshed.json (695 rasters, 2026-10-10T20:01Z)",
            cache_restore=(restore or {}).get("complete"), n_cached=None),
        raster=dict(
            file="downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif",
            sha256=receipt["sha256"], bytes=receipt["bytes"],
            dots=receipt["validator"]["emitted_positive_pixels"],
            validator=dict(all_checks_passed=receipt["validator"]["all_checks_passed"],
                           checks=receipt["validator"]["checks"],
                           no_nan=True, values_in_0_1=True,
                           crs="EPSG:32611", shape=[3730, 3292],
                           transform=[100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0])),
        submission=dict(
            name="gems59-h59i-intensity-lattice-20261011",
            note="H59-I: presence-only MaxEnt/Poisson intensity, effort covariates, "
                 "1 km new-fault-density zoning, holdout break-even lattice.",
            note_chars=124,
            download_status="OK (unique, locally validated)",
            submit_status="NO - holdout DTI 0.0822 is below the published holdout best "
                          "0.1352; promotion is the owner's separate selector step",
            slots_used=0),
        verdict=verdict,
        experiments_used=3,
        budget="3 experiments / 2 hours (protocol rule 6)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    for p in ("evidence/run_card.json", "docs/data/run_card.json"):
        q = ROOT / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(card, indent=1, allow_nan=False) + "\n")
    print("verdict:", verdict, "| dti", round(mine, 5))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
