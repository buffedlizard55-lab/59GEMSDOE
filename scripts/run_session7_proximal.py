#!/usr/bin/env python3
"""One preregistered S7-1 proximal-pruning test on the shared buffered holdout.

This experiment reuses the shared geometry builder, feature-only leakage
canaries, weighted model, greedy allocator and ``evaluate_holdout`` instrument.
It never writes a prediction raster or spends a competition slot. The official
19-band feature TIFF is not read or rebuilt; this geometry-only hypothesis does
not require it. The two-pixel cutoff is fixed in the pre-registration, not swept.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

from gems59 import evaluate_holdout as EH
from gems59.anatomy import (
    fold_geometry,
    measure_withheld,
    package_holdout_structure,
    relative_strike_distribution,
)
from gems59.emit import greedy_allocate
from gems59.faultzone import trace_sense_raster
from gems59.fitting import canary, fit_model, predict_surface
from gems59.grid import load_grid
from gems59.holdout import FOLD_NAMES, buffered_component_draw
from gems59.metric import ALPHA, BETA, R_M

CSV = ROOT / "data/external/trace_segments_utm11.csv"
HISTORICAL_REFERENCE_PATH = ROOT / "evidence/relay_bend_holdout.json"
RADIUS_PX = 2.0
PINNED_INPUTS = {
    "data/bridge/existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/bridge/sample_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
    "data/external/trace_segments_utm11.csv": "c5edf9df3e0e1413c52784884fc38e5b9ca54627b27c6f8baf7713e908f48513",
}
HISTORICAL_REFERENCE_ARM = "relay_bend_sense_transition"
PIPELINE_SOURCE_PATHS = (
    "scripts/run_session7_proximal.py",
    "src/gems59/anatomy.py",
    "src/gems59/evaluate_holdout.py",
    "src/gems59/holdout.py",
    "src/gems59/metric.py",
    "src/gems59/spatial.py",
    "src/gems59/fitting.py",
    "src/gems59/emit.py",
    "src/gems59/network.py",
    "src/gems59/faultzone.py",
    "src/gems59/grid.py",
)
ENVIRONMENT_PIN_PATHS = ("requirements-lock.txt",)


def load_historical_reference(seed: int) -> dict:
    """Load, verify and classify the saved best; never treat a version string as a pin."""
    payload = json.loads(HISTORICAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "HOLDOUT-DTI":
        raise ValueError("historical comparator is not labelled HOLDOUT-DTI")
    record = payload.get("scores", {}).get(HISTORICAL_REFERENCE_ARM)
    if not isinstance(record, dict):
        raise ValueError(f"missing historical comparator arm: {HISTORICAL_REFERENCE_ARM}")
    current_hashes = EH.implementation_hashes()
    saved_hashes = payload.get("implementation_sha256", {})
    mismatches = sorted(name for name, digest in current_hashes.items()
                        if saved_hashes.get(name) != digest)
    same_split = (payload.get("split_version") == "buffered-whole-components-loqo-v2"
                  and payload.get("seed") == seed)
    same_version = payload.get("evaluator_version") == EH.VERSION
    return {
        "evidence_class": "HOLDOUT-DTI (historical reference; not a new score)",
        "arm": HISTORICAL_REFERENCE_ARM,
        "dti": float(record["dti"]),
        "ci95": [float(x) for x in record["ci95"]],
        "withheld_positive_pixels": int(record["withheld_positive_pixels"]),
        "evaluator_version": payload.get("evaluator_version"),
        "saved_evaluator_implementation_sha256": saved_hashes,
        "current_evaluator_implementation_sha256": current_hashes,
        "evaluator_hash_mismatches": mismatches,
        "same_version_string": bool(same_version),
        "same_split_and_seed": bool(same_split),
        "comparable_for_promotion": bool(same_version and same_split and not mismatches),
        "source": "evidence/relay_bend_holdout.json",
        "source_sha256": sha256(HISTORICAL_REFERENCE_PATH),
        "comparison_caveat": (
            "Historical point estimate only. No paired terms are available. It is not a promotion comparator "
            "unless the evaluator implementation hashes, split and seed all match exactly."
        ),
    }


def build_run_card(*, score: float | None, ci95: list[float] | None,
                   withheld_positive_pixels: int | None, status: str,
                   witness: dict, verdict: str = "negative") -> dict:
    """Stable compact deliverable; nulls mean the corresponding gate was not run."""
    return {
        "hypothesis": "S7-1: fixed 2 px visible-catalogue pruning beats equal-count random pruning.",
        "mechanism": "Suppress potentially redundant predictions within 200 m of visible mapped faults while preserving the same fitted anatomy and matching the removed-dot count with random pruning.",
        "non_fault_mimic": "Digitization offsets, map generalization and raster alignment errors can create an apparent near-trace redundancy band.",
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI" if score is not None else status,
            "evaluator_version": EH.VERSION,
            "withheld_positive_pixels": withheld_positive_pixels,
            "dti": score,
            "ci95": ci95,
        },
        "registry_correlation_overlap": {
            "candidate_scan_status": "NOT RUN — no production candidate surface or final dots were built.",
            "max_spearman_correlation": None,
            "max_candidate_dot_overlap_with_prior": None,
            "universal_support_witness": witness,
        },
        "tiff_sha256": None,
        "validator": {
            "status": "NOT RUN — no candidate GeoTIFF was written.",
            "no_nan_inside_footprint": None,
            "range_0_1": None,
            "crs_shape_transform_match": None,
        },
        "submission_name": None,
        "submission_note": "Not generated: uniqueness gate blocks every candidate; do not download or submit.",
        "verdict": verdict,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pipeline_source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in PIPELINE_SOURCE_PATHS}


def environment_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in ENVIRONMENT_PIN_PATHS}


def runtime_environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "rasterio": package_version("rasterio"),
        "threadpoolctl": package_version("threadpoolctl"),
    }


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_pinned_inputs() -> dict:
    """Fail closed if any checked-in grid/source bytes differ from their pins."""
    actual = {}
    for relative, expected in PINNED_INPUTS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned holdout input missing: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise ValueError(f"pinned holdout input hash mismatch: {relative}")
    return actual


def load_registry_witness() -> dict:
    """Revalidate the historical universal witness used for the no-placement stop."""
    cert_path = ROOT / "evidence/uniqueness_saturation_certificate.json"
    check_path = ROOT / "evidence/session7_registry_precheck.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    check = json.loads(check_path.read_text(encoding="utf-8"))
    indexed = json.loads((ROOT / "evidence/registry_refreshed_20261010T2001.json").read_text(encoding="utf-8"))
    matches = [r for r in indexed["rasters"] if r.get("sha256") == cert.get("witness_sha256")]
    if (not cert.get("universal_overlap_blocker") or cert.get("covered_allowed_fraction") != 1.0
            or cert.get("uncovered_allowed_pixels") != 0 or len(matches) != 1
            or not check.get("witness_recheck", {}).get("repo_17_main_unchanged_since_index")):
        raise ValueError("universal uniqueness blocker could not be revalidated; fail closed")
    return {
        "evidence_class": "REGISTRY-MEASUREMENT (historical indexed witness; no candidate scan)",
        "index": "evidence/registry_refreshed_20261010T2001.json",
        "index_generated_utc": indexed["generated_utc"],
        "indexed_rasters": indexed["n_unique_grid_rasters"],
        "freshness_check": "evidence/session7_registry_precheck.json",
        "checked_utc": check["checked_utc"],
        "source": cert["source"],
        "sha256": cert["witness_sha256"],
        "universal_support_coverage": float(cert["covered_allowed_fraction"]),
        "uncovered_allowed_pixels": int(cert["uncovered_allowed_pixels"]),
        "forward_dot_overlap_implied": 1.0,
        "overlap_limit": float(cert["overlap_limit"]),
        "candidate_unique_scan_performed": False,
        "stop_before_production_placement": True,
        "changed_public_main_heads_since_index": check["public_main_head_check"]["changed_since_index"],
        "open_pull_request_heads_not_content_scanned": check["open_pull_request_heads"]["heads"],
    }


def subset(g, keep: np.ndarray, key: str):
    return SimpleNamespace(
        X=g.X[keep], y=g.y[keep], rows=g.rows[keep], cols=g.cols[keep],
        key=key, feature_names=g.feature_names,
    )


def pruned_arms(base: np.ndarray, distance_to_visible: np.ndarray, *, seed: int):
    """Return base, <=2 px prune and exactly matched random-prune masks."""
    base = np.asarray(base, bool)
    if base.ndim != 2 or distance_to_visible.shape != base.shape:
        raise ValueError("base dots and distance raster must share a 2-D grid")
    proximal = base & (distance_to_visible <= RADIUS_PX)
    n_remove = int(proximal.sum())
    candidate = base & ~proximal
    flat_base = np.flatnonzero(base)
    if n_remove > flat_base.size:
        raise AssertionError("cannot remove more random points than were emitted")
    rng = np.random.default_rng(seed)
    removed = rng.choice(flat_base, size=n_remove, replace=False) if n_remove else np.empty(0, np.int64)
    matched_random = base.copy()
    matched_random.ravel()[removed] = False
    if candidate.sum() != matched_random.sum():
        raise AssertionError("proximal and random ablations must have identical dot counts")
    return {"base": base, "proximal_prune": candidate,
            "matched_random_prune": matched_random}, n_remove


def run(*, seed: int = 20, per_quadrant_cap: int = 10000,
        minutes: float = 110, draws: int = 1000, out: Path | None = None) -> dict:
    if seed < 0 or per_quadrant_cap < 1 or not 0 < minutes <= 120:
        raise ValueError("invalid seed, cap or wall-clock budget (maximum 120 minutes)")
    start = time.monotonic()
    deadline = start + minutes * 60

    def check_time(stage: str):
        if time.monotonic() > deadline:
            raise TimeoutError(f"two-hour session budget reached during {stage}")

    input_hashes = verify_pinned_inputs()
    registry_witness = load_registry_witness()
    historical_reference = load_historical_reference(seed)
    check_time("pinned input / gate audit")
    if not CSV.is_file():
        raise FileNotFoundError(f"recorded-sense bridge input missing: {CSV}")
    grid = load_grid()
    folds, quadrant = buffered_component_draw(grid, seed=seed)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    domain = np.logical_or.reduce([f["region"] for f in folds]) & ~folds[0]["masked_known"]
    sense = trace_sense_raster(CSV, grid.shape, grid.transform)
    full = fold_geometry(grid, visible, hidden, domain, f"s7-proximal-seed-{seed}",
                         sense_src=sense)
    if full.X.shape[1] != len(full.feature_names):
        raise AssertionError("shared geometry returned inconsistent feature metadata")
    geoms = [subset(full, quadrant[full.rows, full.cols] == q, FOLD_NAMES[q])
             for q in range(len(FOLD_NAMES))]
    if any(not g.y.any() for g in geoms):
        raise ValueError("each spatial test quadrant must contain withheld positives")
    withheld_positive_pixels = int(full.y.sum())
    historical_reference["same_withheld_positive_count"] = (
        historical_reference["withheld_positive_pixels"] == withheld_positive_pixels)
    historical_reference["comparable_for_promotion"] = bool(
        historical_reference["comparable_for_promotion"]
        and historical_reference["same_withheld_positive_count"])
    check_time("geometry build")

    # Every model feature is canaried before fitting or trusting it.
    canaries = canary(geoms, feature_names=full.feature_names)
    flagged = [name for name, rec in canaries.items()
               if rec["discriminative_auc_max"] is None
               or rec["discriminative_auc_max"] > 0.90]
    relative = relative_strike_distribution(grid, visible, hidden)
    structure = package_holdout_structure(
        relative,
        withheld_positive_pixels=int(full.y.sum()),
        distance_positive_quantiles_px=np.quantile(
            full.X[full.y == 1, 0], [0.1, 0.5, 0.9, 0.95, 0.99]).tolist(),
        distance_domain_quantiles_px=np.quantile(
            full.X[:, 0], [0.1, 0.5, 0.9, 0.95, 0.99]).tolist(),
    )
    structure["holdout_distance_bins"] = measure_withheld(full)["distance"]
    structure["interpretation"] = (
        "Relative-strike and distance distributions describe withheld catalogue geometry versus visible faults; "
        "they are not model features derived from hidden values and are not a DTI score."
    )
    check_time("leakage canary and holdout structure")
    if flagged:
        report = {
            "schema": "gems59.session7-proximal-holdout.v1",
            "evidence_class": "NEGATIVE HOLDOUT ATTEMPT",
            "status": "stopped before fitting: unresolved feature-alone canary",
            "evaluator_version": EH.VERSION,
            "withheld_positive_pixels": withheld_positive_pixels,
            "evaluator_implementation_hashes": EH.implementation_hashes(),
            "pipeline_implementation_sha256": pipeline_source_hashes(),
            "environment_sha256": environment_hashes(),
            "runtime_environment": runtime_environment(),
            "input_sha256": input_hashes,
            "historical_reference_comparability": historical_reference,
            "leakage_canary": {"features": canaries, "flagged_features": flagged,
                                "threshold": 0.90, "rule": "max(AUC,1-AUC)"},
            "structure": structure,
            "fold_holdout_surfaces_built": False,
            "full_production_surface_built": False,
            "production_dots_generated": False,
            "candidate_geoTIFF_sha256": None,
            "registry_gate": registry_witness,
            "submission_slots_used": 0,
            "okay_to_download": False,
            "okay_to_submit": False,
            "score_projection": None,
            "verdict": "negative",
        }
        report["run_card"] = build_run_card(
            score=None, ci95=None, withheld_positive_pixels=withheld_positive_pixels,
            status="NEGATIVE HOLDOUT ATTEMPT — canary stop", witness=registry_witness)
        save(out or ROOT / "evidence/session7_proximal_holdout.json", report)
        return report

    # The four quadrant geometries hold copied slices; release the all-domain matrix.
    del full
    terms = {"base": None, "proximal_prune": None, "matched_random_prune": None}
    raw_details = []
    distance_to_visible = ndi.distance_transform_edt(~visible).astype(np.float32)
    for q, fold in enumerate(folds):
        check_time(f"fold {FOLD_NAMES[q]} fit")
        train = [g for j, g in enumerate(geoms) if j != q]
        test = geoms[q]
        train_positive_distances = np.concatenate([g.X[g.y == 1, 0] for g in train])
        if train_positive_distances.size == 0:
            raise ValueError(f"no training-fold positives for {FOLD_NAMES[q]}")
        fitted_zone_px = float(np.quantile(train_positive_distances, 0.90))
        zone_fraction = float(np.mean(train_positive_distances <= fitted_zone_px))
        cap = max(1, int(per_quadrant_cap * zone_fraction))
        train_pixels = sum(int(g.y.size) for g in train)
        train_positives = sum(int(g.y.sum()) for g in train)
        train_prevalence = train_positives / max(train_pixels, 1)
        scored_region = fold["region"] & ~fold["masked_known"]
        k_estimate = train_prevalence * int(scored_region.sum())

        with threadpool_limits(limits=2):
            model, scale, _ = fit_model(train, seed=seed + q)
            p = predict_surface(model, scale, test, grid.shape)
        outside_zone = test.X[:, 0] > fitted_zone_px
        p[test.rows[outside_zone], test.cols[outside_zone]] = 0.0
        allowed_zone = scored_region.copy()
        allowed_zone[test.rows[outside_zone], test.cols[outside_zone]] = False
        check_time(f"fold {FOLD_NAMES[q]} allocation")
        alloc = greedy_allocate(
            p, allowed_zone, k_truth=k_estimate, floor=0.015,
            max_dots=cap, candidate_cap=400_000,
        )
        if np.any(alloc.emitted & ~allowed_zone):
            raise AssertionError("allocator emitted outside the training-fitted zone")
        arms, n_removed = pruned_arms(
            alloc.emitted, distance_to_visible, seed=202_610_100 + q,
        )
        fold_record = {
            "fold": FOLD_NAMES[q],
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": EH.VERSION,
            "withheld_positive_pixels": int(fold["truth"].sum()),
            "training_fitted_zone_px": fitted_zone_px,
            "training_positive_pixels": int(train_positive_distances.size),
            "training_positive_pixels_in_zone": int(np.count_nonzero(train_positive_distances <= fitted_zone_px)),
            "training_fraction_within_zone": zone_fraction,
            "training_zone_quantile_caveat": "The zone is the training-positive 90th percentile, so the fraction is approximately 0.90 by construction; this is not an adaptive test-fold support estimate.",
            "training_base_rate": train_prevalence,
            "training_k_estimate_for_emission": k_estimate,
            "allowed_emission_pixels_inside_fitted_zone": int(allowed_zone.sum()),
            "allocation_restricted_to_training_fitted_zone": True,
            "test_truth_used_for_emission": False,
            "dot_cap_from_training_only": cap,
            "base_dots": int(alloc.n_dots),
            "proximal_dots_removed_at_2px": n_removed,
            "post_prune_dot_count": int(arms["proximal_prune"].sum()),
        }
        for name, dots in arms.items():
            check_time(f"fold {FOLD_NAMES[q]} evaluation")
            score, block_terms = EH.evaluate(
                dots.astype(np.float32), fold, grid.footprint,
                block_side=200, origin=(0, 0), global_shape=grid.shape,
            )
            if terms[name] is None:
                terms[name] = block_terms.copy()
            else:
                terms[name] += block_terms
            fold_record[name] = {
                "evidence_class": "HOLDOUT-DTI",
                "evaluator_version": EH.VERSION,
                "dti": float(score["dti"]),
                "withheld_positive_pixels": int(score["n_truth"]),
                "emitted_pixels": int(score["n_emitted"]),
            }
        raw_details.append(fold_record)
        print(
            f"[S7-1] {FOLD_NAMES[q]}: removed={n_removed:,}; "
            f"dots={arms['proximal_prune'].sum():,}; "
            f"prox={fold_record['proximal_prune']['dti']:.6f}; "
            f"random={fold_record['matched_random_prune']['dti']:.6f}",
            flush=True,
        )
        del p, model, alloc, arms

    pooled = EH.pooled_summary(
        terms, draws=draws, seed=20261010, candidate="proximal_prune",
    )
    candidate = pooled["scores"]["proximal_prune"]
    random_delta = pooled["paired_differences"]["matched_random_prune"]
    beats_historical_reference = candidate["dti"] > historical_reference["dti"]
    current_best_comparable = historical_reference["comparable_for_promotion"]
    beats_current_comparable_best = (
        candidate["dti"] > historical_reference["dti"] if current_best_comparable else None
    )
    beats_random = random_delta["ci95"][0] > 0.0
    promote_holdout = bool(current_best_comparable and beats_current_comparable_best
                           and beats_random and not flagged)
    report = {
        "schema": "gems59.session7-proximal-holdout.v1",
        "evidence_class": "HOLDOUT-DTI",
        "hypothesis_id": "S7-1 / prior slate H6-1",
        "hypothesis": "At fixed per-fold dot count, removing emitted dots within 2 px of the visible catalogue improves DTI more than an equal-count random removal.",
        "named_non_fault_process_that_could_mimic_it": "Digitization offsets, map generalization and trace-raster alignment error can mimic a real near-fault redundancy band.",
        "evaluator_version": EH.VERSION,
        "evaluator_implementation_hashes": EH.implementation_hashes(),
        "pipeline_implementation_sha256": pipeline_source_hashes(),
        "environment_sha256": environment_hashes(),
        "runtime_environment": runtime_environment(),
        "input_sha256": input_hashes,
        "pre_registration_sha256": sha256(ROOT / "evidence/session7_hypotheses.json"),
        "registry_gate_evidence_sha256": {
            "evidence/uniqueness_saturation_certificate.json": sha256(ROOT / "evidence/uniqueness_saturation_certificate.json"),
            "evidence/session7_registry_precheck.json": sha256(ROOT / "evidence/session7_registry_precheck.json"),
        },
        "historical_reference_comparability": historical_reference,
        "split": {
            "version": "buffered-whole-components-loqo-v2",
            "seed": seed,
            "withholding": "whole 8-connected catalogue components, label-blind eroded quadrants, shared 3 px context collar",
            "withheld_positive_pixels": withheld_positive_pixels,
            "spatial_bootstrap_clusters": pooled["bootstrap"]["clusters"],
            "block_side_px": 200,
            "block_side_m": 20000,
            "per_fold": [f["receipt"] for f in folds],
        },
        "metric": {
            "name": "pooled DTI",
            "alpha": ALPHA,
            "beta": BETA,
            "triangular_kernel_radius_m": R_M,
        },
        "leakage_canary": {
            "evidence_class": "LEAKAGE-CANARY (AUC, not DTI)",
            "features": canaries,
            "flagged_features": flagged,
            "rule": "max(AUC, 1-AUC) > 0.90 is leakage until resolved",
            "clean": not flagged,
        },
        "withheld_structure": structure,
        "intervention": {
            "radius_px": RADIUS_PX,
            "radius_m": RADIUS_PX * 100,
            "criterion": "Euclidean distance to nearest visible fault <= 2 px",
            "prune_threshold_swept": False,
            "paired_random_control": "same number of base dots removed independently within each quadrant, fixed RNG seed 202610100+fold index",
            "dot_budget": "base cap = 10,000 per quadrant times the TRAINING-only share of withheld positives inside the fitted zone; K estimate uses training prevalence only",
            "zone_budget_caveat": "The fitted zone is the training-positive 90th percentile, so the training share is approximately 0.90 by construction. This shrinks the cap by about 10% but does not adapt to test-fold support shift; no test labels determine emission.",
        },
        **pooled,
        "per_fold": raw_details,
        "historical_holdout_reference": historical_reference,
        "current_comparable_best_available": current_best_comparable,
        "current_comparable_best_reference": historical_reference if current_best_comparable else None,
        "candidate_beats_historical_reference_absolute_value_descriptive_only": beats_historical_reference,
        "candidate_beats_current_comparable_best_absolute_value": beats_current_comparable_best,
        "paired_difference_vs_matched_random_is_positive_at_95pct": beats_random,
        "holdout_promotion_condition_met": promote_holdout,
        "submission_slots_used": 0,
        "fold_holdout_surface_predictions_built": True,
        "fold_holdout_dots_generated": True,
        "full_production_surface_built": False,
        "production_dots_generated": False,
        "candidate_surface_built": False,
        "candidate_surface_semantics": "False means no full-catalogue production surface; per-fold prediction arrays were built only for HOLDOUT-DTI.",
        "candidate_geoTIFF_sha256": None,
        "validator_output": {
            "status": "NOT RUN — no candidate TIFF was written because the verified universal-support witness blocks every nonempty allowable candidate under the unchanged literal overlap rule.",
            "no_nan_inside_footprint": None,
            "range_0_1": None,
            "crs_shape_transform_match": None,
        },
        "registry_gate": {
            **registry_witness,
            "evidence_class": "REGISTRY-MEASUREMENT (historical full scan plus current HEAD freshness check; not a candidate scan)",
            "preplacement_candidate_surface_scan": "NOT RUN — stopped before production-surface construction",
            "final_dot_scan": "NOT APPLICABLE — no production dots generated",
            "universal_support_certificate": "evidence/uniqueness_saturation_certificate.json",
            "literal_result": "STOP: any nonempty candidate has 1.0 forward 3-px overlap with the witness, exceeding the 0.70 limit",
            "thresholds_changed": False,
            "scope_note": "The 696-raster full index is stale for four public-main heads and four open PR heads were recorded but not content-scanned. This does not weaken the stop: the hash-verified witness remains in unchanged 17GEMSDOE main and saturates all allowed pixels. Private/unlinked rasters remain uncertified.",
        },
        "submission_name": None,
        "submission_note": "Not generated: the literal pre-placement uniqueness gate is globally blocked; do not download or submit.",
        "submission_note_chars": len("Not generated: the literal pre-placement uniqueness gate is globally blocked; do not download or submit."),
        "verdict": "negative",
        "okay_to_download": False,
        "okay_to_submit": False,
        "score_projection": None,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - start,
    }
    if report["submission_note_chars"] > 140:
        raise AssertionError("submission note exceeds the 140-character protocol limit")
    report["run_card"] = build_run_card(
        score=candidate["dti"], ci95=candidate["ci95"],
        withheld_positive_pixels=candidate["withheld_positive_pixels"],
        status="HOLDOUT-DTI", witness=registry_witness, verdict="negative")
    report["run_card"].update({
        "historical_reference_comparable_for_promotion": current_best_comparable,
        "paired_random_control_lower_95pct_bound_positive": beats_random,
        "holdout_promotion_condition_met": promote_holdout,
        "verdict_reason": "No candidate TIFF may be built or promoted under the unchanged universal-overlap stop; historical best implementation hashes also differ.",
    })
    save(out or ROOT / "evidence/session7_proximal_holdout.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--per-quadrant-cap", type=int, default=10000)
    parser.add_argument("--minutes", type=float, default=110)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/session7_proximal_holdout.json")
    args = parser.parse_args()
    report = run(seed=args.seed, per_quadrant_cap=args.per_quadrant_cap,
                 minutes=args.minutes, draws=args.draws, out=args.out)
    result = report.get("scores", {})
    print(json.dumps({
        "evidence_class": report.get("evidence_class"),
        "evaluator_version": report.get("evaluator_version"),
        "scores": result,
        "paired_differences": report.get("paired_differences", {}),
        "canary_clean": report.get("leakage_canary", {}).get("clean"),
        "candidate_beats_historical_reference_absolute_value_descriptive_only": report.get("candidate_beats_historical_reference_absolute_value_descriptive_only"),
        "current_comparable_best_available": report.get("current_comparable_best_available"),
        "holdout_promotion_condition_met": report.get("holdout_promotion_condition_met"),
        "run_card": report.get("run_card"),
        "submission_slots_used": report.get("submission_slots_used"),
        "verdict": report.get("verdict"),
    }, indent=2, allow_nan=False))
    return 0 if report.get("evidence_class") == "HOLDOUT-DTI" else 3


if __name__ == "__main__":
    raise SystemExit(main())
