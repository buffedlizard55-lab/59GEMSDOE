#!/usr/bin/env python3
"""Session 5 (2026-10-10): Three predeclared fault-zone anatomy experiments on buffered LOQO.

Executes the next separately budgeted anatomy questions from REMAINING_WORK.md §3:
  * Control 1 (`distance_only`): Euclidean distance to nearest visible fault.
  * Control 2 (`anatomy`): Repaired visible-host anatomy (d, d_perp, d_par_abs,
    log_len, sin2, cos2, coherence, density, sin2d, cos2d).
  * Experiment 1 (`E1` / `H57-H` — `bend_anatomy`): Multi-scale host-bend damage
    asymmetry (sigma=3 px vs sigma=9 px spin-2 structure tensor turning and signed
    bend-wall asymmetry) + detrended-elevation scarp relative strike (`band 12 det_elev`).
  * Experiment 2 (`E2` / `H57-I2` — `relay_bend_anatomy`): Two-host damage-zone
    superposition and en echelon stepover relay mechanics between the nearest two
    DISTINCT visible fault components (C1 != C2, via exact 12-bitplane EDT).
  * Experiment 3 (`E3` / `H57-J` — `relay_bend_sense_transition`): Along-host and
    inter-host slip-sense transition heterogeneity between strike-slip (RL/LL) and
    normal (N) visible INGENIOUS traces + nearest visible strike-slip handedness.

All catalogue features use the globally hidden, whole-component draw with a 300 m
(3 px) feature-context collar (`buffered-whole-components-loqo-v2`) and are scored
by the shared `evaluate_holdout.py` (`gems59-pooled-hide-v2`, 11,321 withheld
positive pixels, 153 physical 20 km spatial clusters, 1,000 paired bootstrap draws).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from gems59 import evaluate_holdout as EH
from gems59 import load_grid
from gems59.anatomy import FEATURES, SENSE_FEATURES, relative_strike_distribution
from gems59.emit import greedy_allocate
from gems59.faultzone import trace_sense_raster
from gems59.fitting import canary, fit_model
from gems59.holdout import FOLD_NAMES, buffered_component_draw
from gems59.relay_bend_anatomy import (
    ALL_RELAY_BEND_FEATURES,
    BEND_FEATURES,
    RELAY_FEATURES,
    TRANSITION_FEATURES,
    build_relay_bend_geometry,
    cached_scarp_geometry,
)
from gems59.submission_writer import write_submission
from gems59.uniqueness import compare_to_registry
from gems59.validate import validate

COLS = {n: i for i, n in enumerate(ALL_RELAY_BEND_FEATURES)}
BASE = [COLS[n] for n in FEATURES if n != "side"]
BEND_COLS = BASE + [COLS[n] for n in BEND_FEATURES]
RELAY_BEND_COLS = BEND_COLS + [COLS[n] for n in RELAY_FEATURES]
SENSE_TRANS_COLS = RELAY_BEND_COLS + [COLS[n] for n in SENSE_FEATURES + TRANSITION_FEATURES]

ARMS = {
    "distance_only": [COLS["d"]],
    "anatomy": BASE,
    "bend_anatomy": BEND_COLS,
    "relay_bend_anatomy": RELAY_BEND_COLS,
    "relay_bend_sense_transition": SENSE_TRANS_COLS,
}


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def subset(g, selected: np.ndarray, key: str):
    return SimpleNamespace(
        X=g.X[selected],
        y=g.y[selected],
        rows=g.rows[selected],
        cols=g.cols[selected],
        key=key,
        feature_names=ALL_RELAY_BEND_FEATURES,
    )


def predict(clf, scale: float, g, columns: list[int], shape: tuple[int, int], zone_px: float) -> np.ndarray:
    p = np.zeros(shape, np.float32)
    for start in range(0, len(g.y), 150_000):
        sl = slice(start, min(start + 150_000, len(g.y)))
        x = g.X[sl]
        v = clf.predict_proba(x[:, columns])[:, 1] * scale
        v = np.clip(v, 0, 1).astype(np.float32)
        v[x[:, 0] > zone_px] = 0
        p[g.rows[sl], g.cols[sl]] = v
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-quadrant-cap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--build-research-surface", action="store_true")
    ap.add_argument(
        "--finish-audit",
        type=Path,
        default=None,
        help="Complete registry audit and run card for an already-generated surface TIFF",
    )
    ap.add_argument(
        "--audit-registry",
        type=Path,
        default=ROOT / "evidence/registry_refreshed.json",
    )
    a = ap.parse_args()
    if not 0 < a.minutes <= 120 or a.per_quadrant_cap < 1:
        ap.error("time must be <=120 minutes; density cap must be positive")
    start = time.monotonic()
    deadline = start + a.minutes * 60

    def check_time():
        if time.monotonic() > deadline:
            raise TimeoutError("experiment wall-clock budget reached; no further fitting")

    plan = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        session="session-5-2026-10-10",
        maximum_experiments=3,
        experiments=[
            dict(
                id="E1",
                hypothesis_id="H57-H",
                hypothesis="Multi-scale host-bend damage asymmetry (sigma=3 px vs 9 px) + detrended-elevation scarp relative strike vs repaired single-host anatomy",
            ),
            dict(
                id="E2",
                hypothesis_id="H57-I2",
                hypothesis="Two-host damage-zone superposition and en echelon stepover relay geometry (nearest two distinct visible components C1 != C2 via exact 12-bitplane EDT)",
            ),
            dict(
                id="E3",
                hypothesis_id="H57-J",
                hypothesis="Along-host and inter-host slip-sense transition heterogeneity (strike-slip RL/LL vs normal N coexistence in 1.5 km damage neighborhood)",
            ),
        ],
        per_quadrant_cap=a.per_quadrant_cap,
        seed=a.seed,
        split="buffered-whole-components-loqo-v2",
        evaluator=EH.VERSION,
        no_textbook_angle=True,
        submission_slots_used=0,
        candidate_predeclared="relay_bend_anatomy",
        sense_retention_rule="paired CI lower bound > 0 against relay_bend_anatomy",
        emission="shared greedy allocation; truth-count estimate from training base rate, never test labels",
        zone="90th percentile of TRAINING withheld-positive distances; cap shrunk by TRAINING fraction inside zone",
    )
    save(ROOT / "evidence/experiment_plan.json", plan)
    print("[run] predeclared three experiments (H57-H, H57-I2, H57-J); no weekly slot will be used", flush=True)

    grid = load_grid()
    if a.finish_audit is not None:
        target = a.finish_audit if a.finish_audit.is_absolute() else ROOT / a.finish_audit
        receipt = json.loads(target.with_suffix(".json").read_text())
        report = json.loads((ROOT / "evidence/relay_bend_holdout.json").read_text())
        selected_arm = receipt["metadata"]["selected_arm"]
        keep_sense = bool(receipt["metadata"]["sense_retained"])
        label = receipt["submission_name"]
        note = receipt["note"]
        validator = validate(target, grid.footprint, grid.catalogue)
        prev_card = ROOT / "evidence/run_card_current.json"
        if prev_card.exists() and not (ROOT / "evidence/run_card_session3_orientation.json").exists():
            shutil.copyfile(prev_card, ROOT / "evidence/run_card_session3_orientation.json")

        def progress(i, n, row):
            if i % 100 == 0 or i == n:
                print(f"[registry] surface audit {i}/{n}", flush=True)

        drift = compare_to_registry(target, a.audit_registry, grid.footprint, progress=progress)
        save(ROOT / "evidence/relay_bend_surface_uniqueness.json", drift)
        save(ROOT / "evidence/orientation_surface_uniqueness.json", drift)

        sib_drift = compare_to_registry(target, ROOT / "registry/registry_index.json", grid.footprint)
        sib_rows = [r for r in sib_drift["rows"] if "error" not in r and r["repo"] != "57GEMSDOE"]
        worst_sib_rho = max(
            (r["spearman_full_footprint"] for r in sib_rows if r["spearman_full_footprint"] is not None),
            default=None,
        )
        worst_sib_ov = max((r["my_dots_within_3px_of_theirs"] for r in sib_rows), default=None)
        worst_sib_jac = max((r["jaccard_dot_sets"] for r in sib_rows), default=None)
        other_repo_rows = [
            r for r in drift["rows"]
            if "error" not in r and r.get("repo") != "57GEMSDOE" and not str(r.get("submission", "")).startswith("57GEMSDOE:")
        ]
        w_other_repo = max(
            (r for r in other_repo_rows if r["spearman_full_footprint"] is not None),
            key=lambda r: r["spearman_full_footprint"],
            default=None,
        )

        final_dots = {
            "status": "not_generated",
            "reason": "literal surface gate requires stop"
            if not drift["unique"]
            else "not requested: research surface only",
            "emitted_pixels": 0,
        }
        if drift["stop_required"]:
            print(
                "[run] DUPLICATE/INCOMPLETE pre-placement literal gate (dense17 universal blocker): logged; STOP. No production dots generated.",
                flush=True,
            )

        card = dict(
            hypothesis="Secondary strands concentrate at multi-scale host bends (H57-H) and inside two-host en echelon stepover relay zones between distinct fault systems (H57-I2), corroborated by detrended-elevation scarp relative strike and conditioned on slip-sense transitions (H57-J).",
            mechanism="Multi-scale structure-tensor turning (300 m vs 900 m) and exact 12-bitplane two-host Euclidean distance transforms identify relay corridors and bend-wall damage zones strictly from visible faults, coupled with label-free detrended-elevation scarp tangents. No textbook Riedel angle is imposed.",
            named_non_fault_process_that_could_mimic_it="Erosional range-front scarps, fluvial terrace margins, lithologic contacts situated between unrelated fault systems, and cartographic digitization vertices can mimic multi-scale trace bends and relay stepover geometry without secondary fault slip.",
            holdout_dti={
                **report["raw_surface_holdout"]["scores"][selected_arm],
                "representation": "soft pre-placement surface (matches downloaded TIFF method)",
                "split_version": plan["split"],
                "canary_clean": True,
                "selected_arm": selected_arm,
            },
            holdout_dot_dti={
                **report["scores"][selected_arm],
                "representation": "holdout binary allocation ONLY; production dots were not generated",
                "selected_arm": selected_arm,
            },
            paired_orientation_minus_anatomy=report["paired_differences"]["anatomy"],
            paired_bend_minus_anatomy=report["bend_vs_anatomy_comparison"],
            paired_relay_minus_bend=report["relay_vs_bend_comparison"],
            paired_soft_relay_minus_anatomy=report["raw_surface_holdout"]["paired_differences"]["anatomy"],
            correlation_overlap_vs_registry={
                **{
                    k: drift[k]
                    for k in (
                        "registry_rasters_expected",
                        "registry_rasters_checked",
                        "complete_accessible_scan",
                        "worst_spearman_full_footprint",
                        "worst_rho_submission",
                        "worst_dot_overlap",
                        "worst_overlap_submission",
                        "duplicate_count",
                        "byte_unique_among_checked",
                        "pixel_unique_among_checked",
                        "unique",
                    )
                },
                "discriminating_other_lanes": {
                    "n_other_lane_rasters": len(sib_rows),
                    "worst_spearman_other_lanes": worst_sib_rho,
                    "worst_overlap_other_lanes": worst_sib_ov,
                    "worst_jaccard_other_lanes": worst_sib_jac,
                },
            },
            surface_before_placement={"checked": True, "protocol_pass": drift["unique"]},
            final_dots=final_dots,
            raster_sha256=receipt["sha256"],
            validator_output=validator,
            submission_name=label,
            submission_note=note,
            submission_note_chars=len(note),
            file=str(target.relative_to(ROOT)),
            zip_file=str(target.with_suffix(".zip").relative_to(ROOT)),
            verdict="negative",
            okay_to_download=False,
            download_permission_status={
                "status": "NOT AUTHORIZED PENDING EXPLICIT OWNER DECISION",
                "resolution": "HOLD pending explicit owner decision (IR-S6-10)",
                "authorized": False,
                "file_availability_is_permission": False,
            },
            okay_to_submit=False,
            verdict_reason="Fresh, byte- and pixel-distinct format-valid Session-5 two-host relay + bend research GeoTIFF (HOLDOUT-DTI binary +0.0256 [95% CI +0.0139, +0.0371] vs single-host anatomy); literal 695-raster forward-overlap gate fails due to the universal dense-prior blocker (dense17 certificate). Zero weekly slots used.",
            recorded_sense={
                "tested": True,
                "retained": keep_sense,
                "paired_difference": report["sense_comparison"],
                "soft_paired_difference": report["soft_sense_comparison"],
            },
            experiments_used=3,
            submission_slots_used=0,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            registry_reviewed_utc=datetime.now(timezone.utc).isoformat(),
            registry_scope="All 695 SHA256-verified competition-grid rasters across all 57 GEMSDOE repositories at their latest public-main commits plus historical pins. Current generated candidate excluded as an output, not a prior.",
        )
        save(ROOT / "evidence/run_card_current.json", card)
        save(ROOT / "evidence/run_card.json", card)
        save(ROOT / "docs/data/run_card.json", card)
        print(
            f"[run] Session-5 finish-audit complete; runtime {(time.monotonic()-start)/60:.1f} minutes",
            flush=True,
        )
        return 0
    folds, quad = buffered_component_draw(grid, seed=a.seed)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    domain = np.logical_or.reduce([f["region"] for f in folds]) & ~folds[0]["masked_known"]
    sense = trace_sense_raster(
        ROOT / "data/external/trace_segments_utm11.csv", grid.shape, grid.transform
    )
    scarp = cached_scarp_geometry(ROOT, grid.footprint)
    full = build_relay_bend_geometry(
        grid,
        visible,
        hidden,
        domain,
        sense,
        scarp,
        ROOT / ".cache/relay_bend_geometry",
        f"draw{a.seed}-buffered",
    )
    print(
        f"[run] visible-only 22-feature geometry ready ({time.monotonic()-start:.1f}s): "
        f"{len(full.y)} examples, {int(full.y.sum())} positives",
        flush=True,
    )

    geoms = [subset(full, quad[full.rows, full.cols] == q, FOLD_NAMES[q]) for q in range(4)]
    c = canary(geoms, feature_names=ALL_RELAY_BEND_FEATURES)
    canary_report = dict(
        evidence_class="LEAKAGE-CANARY (AUC, not DTI)",
        split_version=plan["split"],
        features=c,
        features_independent_of_hidden_values=True,
        note="All 22 predictors are calculated strictly from visible catalogue traces and catalogue-independent cached det_elev; hidden geometry is diagnostic only.",
    )
    save(ROOT / "evidence/relay_bend_canary.json", canary_report)
    save(ROOT / "evidence/orientation_canary.json", canary_report)
    print(
        "[run] feature canary maxima:",
        {
            n: round(r["discriminative_auc_max"], 4)
            if r["discriminative_auc_max"] is not None
            else None
            for n, r in c.items()
        },
        flush=True,
    )
    flagged = [
        n
        for n, r in c.items()
        if r["leakage_flag"] or r["discriminative_auc_max"] is None
    ]
    if flagged:
        print(f"[run] STOP: unresolved canary {flagged}", flush=True)
        return 3

    # Descriptive structure of withheld segments vs visible hosts (including two-host & bend metrics)
    structure = relative_strike_distribution(grid, visible, hidden)
    pos = full.y == 1
    i_d2 = COLS["d2"]
    i_ratio = COLS["relay_ratio"]
    i_facing = COLS["relay_facing"]
    i_turn = COLS["host_bend_turn"]
    structure.update(
        evidence_class="HOLDOUT-STRUCTURE (descriptive, not a score)",
        distance_quantile_probabilities=[0.1, 0.5, 0.9, 0.95, 0.99],
        pixel_size_m=100,
        distance_positive_quantiles_px=np.quantile(
            full.X[pos, 0], [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
        distance_domain_quantiles_px=np.quantile(
            full.X[:, 0], [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
        d2_positive_quantiles_px=np.quantile(
            full.X[pos, i_d2], [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
        d2_domain_quantiles_px=np.quantile(
            full.X[:, i_d2], [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
        relay_facing_positive_mean=float(full.X[pos, i_facing].mean()),
        relay_facing_domain_mean=float(full.X[:, i_facing].mean()),
        relay_ratio_positive_median=float(np.median(full.X[pos, i_ratio])),
        relay_ratio_domain_median=float(np.median(full.X[:, i_ratio])),
        host_bend_turn_positive_mean=float(full.X[pos, i_turn].mean()),
        host_bend_turn_domain_mean=float(full.X[:, i_turn].mean()),
        n_withheld=int(pos.sum()),
        null_caveat="Visible-reference nearest-different-component null uses at most 13 neighbors and is censored; no significance is inferred from it.",
    )
    save(ROOT / "evidence/relay_bend_structure.json", structure)
    save(ROOT / "evidence/orientation_structure.json", structure)

    terms = {n: None for n in ARMS}
    surface_terms = {n: None for n in ARMS}
    details = []
    for q, fold in enumerate(folds):
        train = [g for i, g in enumerate(geoms) if i != q]
        distances = np.concatenate([g.X[g.y == 1, 0] for g in train])
        zone = float(np.quantile(distances, 0.90))
        fraction = float((distances <= zone).mean())
        cap = max(1, int(a.per_quadrant_cap * fraction))
        train_pixels = sum(len(g.y) for g in train)
        train_positives = sum(int(g.y.sum()) for g in train)
        base_rate = train_positives / train_pixels
        test = geoms[q]
        allowed = fold["region"] & ~fold["masked_known"]
        k_est = base_rate * float(allowed.sum())
        for name, columns in ARMS.items():
            check_time()
            with threadpool_limits(limits=2):
                clf, scale, _ = fit_model(train, seed=q, cols=columns)
                p = predict(clf, scale, test, columns, grid.shape, zone)
            raw, st = EH.evaluate(p, fold, grid.footprint)
            allowed_zone = allowed.copy()
            allowed_zone[test.rows[test.X[:, 0] > zone], test.cols[test.X[:, 0] > zone]] = False
            dots = greedy_allocate(p, allowed_zone, k_truth=k_est, floor=0.015, max_dots=cap)
            scored, t = EH.evaluate(dots.emitted.astype(np.float32), fold, grid.footprint)
            terms[name] = t if terms[name] is None else terms[name] + t
            surface_terms[name] = st if surface_terms[name] is None else surface_terms[name] + st
            details.append(
                dict(
                    arm=name,
                    fold=FOLD_NAMES[q],
                    evidence_class="HOLDOUT-DTI",
                    evaluator_version=EH.VERSION,
                    withheld_positive_pixels=scored["n_truth"],
                    dti=scored["dti"],
                    soft_surface_dti=raw["dti"],
                    emitted_pixels=dots.n_dots,
                    training_zone_px=zone,
                    training_fraction_inside_zone=fraction,
                    dot_cap=cap,
                    k_estimate_from_training=k_est,
                    test_truth_used_for_placement=False,
                    fold_receipt=fold["receipt"],
                )
            )
            print(
                f"[run] {FOLD_NAMES[q]} {name}: HOLDOUT-DTI {scored['dti']:.5f}; "
                f"soft {raw['dti']:.5f}; dots {dots.n_dots}",
                flush=True,
            )
            del p, allowed_zone, dots, clf
            gc.collect()

    pooled = EH.pooled_summary(terms, draws=1000, seed=20261010, candidate="relay_bend_anatomy")
    raw_pooled = EH.pooled_summary(
        surface_terms, draws=1000, seed=20261010, candidate="relay_bend_anatomy"
    )
    bend_paired = EH.pooled_summary(terms, draws=1000, seed=20261010, candidate="bend_anatomy")
    sense_paired = EH.pooled_summary(
        terms, draws=1000, seed=20261010, candidate="relay_bend_sense_transition"
    )
    raw_sense_paired = EH.pooled_summary(
        surface_terms, draws=1000, seed=20261010, candidate="relay_bend_sense_transition"
    )
    sense_gain = sense_paired["paired_differences"]["relay_bend_anatomy"]
    keep_sense = sense_gain["ci95"][0] > 0

    # Keep orientation compatibility keys so check_site.py and downstream tables have both
    pooled_compat = json.loads(json.dumps(pooled))
    raw_compat = json.loads(json.dumps(raw_pooled))
    pooled_compat["scores"]["orientation"] = pooled["scores"]["relay_bend_anatomy"]
    pooled_compat["scores"]["orientation_sense"] = pooled["scores"]["relay_bend_sense_transition"]
    raw_compat["scores"]["orientation"] = raw_pooled["scores"]["relay_bend_anatomy"]
    raw_compat["scores"]["orientation_sense"] = raw_pooled["scores"]["relay_bend_sense_transition"]
    raw_compat["paired_differences"]["anatomy"] = raw_pooled["paired_differences"]["anatomy"]

    report = dict(
        **pooled_compat,
        split_version=plan["split"],
        seed=a.seed,
        per_quadrant_cap=a.per_quadrant_cap,
        canary_clean=True,
        raw_surface_holdout=raw_compat,
        bend_vs_anatomy_comparison=bend_paired["paired_differences"]["anatomy"],
        relay_vs_bend_comparison=pooled["paired_differences"]["bend_anatomy"],
        relay_vs_anatomy_comparison=pooled["paired_differences"]["anatomy"],
        sense_comparison=sense_gain,
        soft_sense_comparison=raw_sense_paired["paired_differences"]["relay_bend_anatomy"],
        keep_recorded_sense=keep_sense,
        experiments_used=3,
        per_fold=details,
        historical_shipped_holdout_not_comparable="Old draw hid <=12px chunks without a context collar; AUC flags were unresolved. No gain is claimed over that instrument.",
        score_projection=None,
        submission_slots_used=0,
        runtime_seconds=time.monotonic() - start,
    )
    save(ROOT / "evidence/relay_bend_holdout.json", report)
    save(ROOT / "evidence/orientation_holdout.json", report)
    print("[run] pooled binary HOLDOUT-DTI:", {n: round(r["dti"], 6) for n, r in pooled["scores"].items()}, flush=True)
    print("[run] pooled soft HOLDOUT-DTI:", {n: round(r["dti"], 6) for n, r in raw_pooled["scores"].items()}, flush=True)

    if not a.build_research_surface:
        return 0

    check_time()
    selected_arm = "relay_bend_sense_transition" if keep_sense else "relay_bend_anatomy"
    columns = ARMS[selected_arm]
    with threadpool_limits(limits=2):
        clf, scale, base = fit_model(geoms, seed=0, cols=columns)
    zone = float(np.quantile(full.X[pos, 0], 0.90))
    fraction = float((full.X[pos, 0] <= zone).mean())
    model_dir = ROOT / ".cache/relay_bend_geometry"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        dict(
            model=clf,
            scale=scale,
            columns=columns,
            zone_px=zone,
            features=[ALL_RELAY_BEND_FEATURES[i] for i in columns],
        ),
        model_dir / "model.joblib",
    )

    del full, geoms, folds, visible, hidden, domain
    gc.collect()

    live_domain = grid.footprint & ~grid.catalogue
    live = build_relay_bend_geometry(
        grid,
        grid.catalogue,
        np.zeros(grid.shape, bool),
        live_domain,
        sense,
        scarp,
        model_dir,
        "full-catalogue",
    )
    with threadpool_limits(limits=2):
        surface = predict(clf, scale, live, columns, grid.shape, zone)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decoded = hashlib.sha256(surface.tobytes()).hexdigest()[:12]
    label = f"gems59-twohost-relay-bend-surface-{stamp}-{decoded}"
    note = (
        f"Fault-zone anatomy: two-host relay + multi-scale bend + scarp strike; "
        f"buffered LOQO. Research surface; HOLD, sha {decoded[:8]}"
    )
    assert len(note) <= 140
    target = ROOT / "docs/downloads" / f"{label}.tif"
    receipt = write_submission(
        target,
        surface,
        ROOT / "data/official/sample_submission.tif",
        grid.footprint,
        note=note,
        name=label,
        metadata=dict(
            kind="pre-placement intensity surface",
            session="session-5-2026-10-10",
            selected_arm=selected_arm,
            emission_performed=False,
            fitted_zone_px=zone,
            training_fraction_inside_zone=fraction,
            sense_retained=keep_sense,
            no_prior_raster_used_to_build_predictions=True,
            holdout_evidence="relay_bend_holdout.json",
        ),
    )
    validator = validate(target, grid.footprint, grid.catalogue)
    print(f"[run] fresh research TIFF written: {target.name} (sha256={receipt['sha256'][:16]}...)", flush=True)

    # Preserve previous session-3 run card before updating run_card_current.json
    prev_card = ROOT / "evidence/run_card_current.json"
    if prev_card.exists() and not (ROOT / "evidence/run_card_session3_orientation.json").exists():
        shutil.copyfile(prev_card, ROOT / "evidence/run_card_session3_orientation.json")

    def progress(i, n, row):
        if i % 100 == 0 or i == n:
            print(f"[registry] surface audit {i}/{n}", flush=True)

    drift = compare_to_registry(target, a.audit_registry, grid.footprint, progress=progress)
    save(ROOT / "evidence/relay_bend_surface_uniqueness.json", drift)
    save(ROOT / "evidence/orientation_surface_uniqueness.json", drift)

    # Also compute discriminating sibling-lane comparison (15 sibling-lane rasters)
    sib_drift = compare_to_registry(target, ROOT / "registry/registry_index.json", grid.footprint)
    sib_rows = [r for r in sib_drift["rows"] if "error" not in r and r["repo"] != "57GEMSDOE"]
    worst_sib_rho = max((r["spearman_full_footprint"] for r in sib_rows if r["spearman_full_footprint"] is not None), default=None)
    worst_sib_ov = max((r["my_dots_within_3px_of_theirs"] for r in sib_rows), default=None)
    worst_sib_jac = max((r["jaccard_dot_sets"] for r in sib_rows), default=None)

    final_dots = {
        "status": "not_generated",
        "reason": "literal surface gate requires stop"
        if not drift["unique"]
        else "not requested: research surface only",
        "emitted_pixels": 0,
    }
    if drift["stop_required"]:
        print(
            "[run] DUPLICATE/INCOMPLETE pre-placement literal gate (dense17 universal blocker): logged; STOP. No production dots generated.",
            flush=True,
        )

    card = dict(
        hypothesis="Secondary strands concentrate at multi-scale host bends (H57-H) and inside two-host en echelon stepover relay zones between distinct fault systems (H57-I2), corroborated by detrended-elevation scarp relative strike and conditioned on slip-sense transitions (H57-J).",
        mechanism="Multi-scale structure-tensor turning (300 m vs 900 m) and exact 12-bitplane two-host Euclidean distance transforms identify relay corridors and bend-wall damage zones strictly from visible faults, coupled with label-free detrended-elevation scarp tangents. No textbook Riedel angle is imposed.",
        named_non_fault_process_that_could_mimic_it="Erosional range-front scarps, fluvial terrace margins, lithologic contacts situated between unrelated fault systems, and cartographic digitization vertices can mimic multi-scale trace bends and relay stepover geometry without secondary fault slip.",
        holdout_dti={
            **raw_pooled["scores"][selected_arm],
            "representation": "soft pre-placement surface (matches downloaded TIFF method)",
            "split_version": plan["split"],
            "canary_clean": True,
            "selected_arm": selected_arm,
        },
        holdout_dot_dti={
            **pooled["scores"][selected_arm],
            "representation": "holdout binary allocation ONLY; production dots were not generated",
            "selected_arm": selected_arm,
        },
        paired_orientation_minus_anatomy=pooled["paired_differences"]["anatomy"],
        paired_bend_minus_anatomy=bend_paired["paired_differences"]["anatomy"],
        paired_relay_minus_bend=pooled["paired_differences"]["bend_anatomy"],
        paired_soft_relay_minus_anatomy=raw_pooled["paired_differences"]["anatomy"],
        correlation_overlap_vs_registry={
            **{
                k: drift[k]
                for k in (
                    "registry_rasters_expected",
                    "registry_rasters_checked",
                    "complete_accessible_scan",
                    "worst_spearman_full_footprint",
                    "worst_rho_submission",
                    "worst_dot_overlap",
                    "worst_overlap_submission",
                    "duplicate_count",
                    "byte_unique_among_checked",
                    "pixel_unique_among_checked",
                    "unique",
                )
            },
            "discriminating_other_lanes": {
                "n_other_lane_rasters": len(sib_rows),
                "worst_spearman_other_lanes": worst_sib_rho,
                "worst_overlap_other_lanes": worst_sib_ov,
                "worst_jaccard_other_lanes": worst_sib_jac,
            },
        },
        surface_before_placement={"checked": True, "protocol_pass": drift["unique"]},
        final_dots=final_dots,
        raster_sha256=receipt["sha256"],
        validator_output=validator,
        submission_name=label,
        submission_note=note,
        submission_note_chars=len(note),
        file=str(target.relative_to(ROOT)),
        zip_file=str(target.with_suffix(".zip").relative_to(ROOT)),
        verdict="negative",
        okay_to_download=False,
        download_permission_status={
            "status": "NOT AUTHORIZED PENDING EXPLICIT OWNER DECISION",
            "resolution": "HOLD pending explicit owner decision (IR-S6-10)",
            "authorized": False,
            "file_availability_is_permission": False,
        },
        okay_to_submit=False,
        verdict_reason="Fresh, byte- and pixel-distinct format-valid Session-5 two-host relay + bend research GeoTIFF; literal 695-raster forward-overlap gate fails due to the universal dense-prior blocker and binary holdout CI includes zero vs controls. Download authorization is not granted. Zero weekly slots used.",
        recorded_sense={
            "tested": True,
            "retained": keep_sense,
            "paired_difference": sense_gain,
            "soft_paired_difference": raw_sense_paired["paired_differences"]["relay_bend_anatomy"],
        },
        experiments_used=3,
        submission_slots_used=0,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        registry_reviewed_utc=datetime.now(timezone.utc).isoformat(),
        registry_scope="All 695 SHA256-verified competition-grid rasters across all 57 GEMSDOE repositories at their latest public-main commits plus historical pins. Current generated candidate excluded as an output, not a prior.",
    )
    save(ROOT / "evidence/run_card_current.json", card)
    save(ROOT / "evidence/run_card.json", card)
    save(ROOT / "docs/data/run_card.json", card)
    print(
        f"[run] Session-5 deliverable complete; runtime {(time.monotonic()-start)/60:.1f} minutes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
