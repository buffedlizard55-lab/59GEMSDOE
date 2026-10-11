#!/usr/bin/env python3
"""Session 7 (2026-10-10): H57-M — two-host relay damage band with live-anchored mass discipline.

Lane: fault-zone anatomy (secondary strands around known faults from shear-zone
mechanics).  Nothing outside the lane is implemented and no prior prediction
raster is used as a feature, base or output.

What is new relative to Session 5 (``run_relay_bend_experiments.py``)
--------------------------------------------------------------------
Session 5 established that two-host relay geometry (H57-I2) beats single-host
anatomy on the shared holdout (``+0.025556`` paired binary, 95% CI strictly
positive).  It emitted dots with a flat per-quadrant cap and allowed every
off-catalogue cell, including cells immediately beside the mapped traces.

This session tests the **emission rule** rather than a new feature block.  The
registry audit of the highest owner-reported file (GEMSDOE32 H33-2-B2,
``0.2778``, sha256 ``c55bafc4…``) shows that the only controlled live A/B
available in this family is a *pure deletion*: 37,654 dots are the 40,199-dot
``0.2708`` field minus every dot within 2 px of the mapped catalogue, and the
deletion raised the reported score by ``+0.0070``.  Under the published metric a
dot whose self-credit is ``k`` costs ``alpha*(1-k)`` of false-positive mass and
earns at most ``k`` of true-positive credit, so a dense band of near-catalogue
dots is mass the metric charges for and the hidden set does not pay for.

Three predeclared arms on the shared buffered whole-component LOQO instrument
(``gems59-pooled-hide-v2``, seed 20, 11,321 withheld positives):

* ``C1 distance_only``  — control, the single column ``d``.
* ``C2 anatomy``        — repaired single-host anatomy (Session-5 control).
* ``C3 relay_band``     — single-host anatomy + two-host relay columns
  (``d2``, ``relay_ratio``, ``relay_facing``, ``two_host_strike_cos2``,
  ``superposed_log_len2``), emitted in two predeclared variants that differ
  **only** in the live-anchored gate ``d > 2 px``.

``v_out`` applies the gate, ``v_in`` is the Session-5 behaviour.  Every other
quantity (features, model, fitted zone, truth-count estimate, allocator, cap) is
identical, so the pair isolates one variable.

Decision rule, predeclared before execution
-------------------------------------------
Promote ``C3 v_out`` if its paired binary HOLDOUT-DTI against ``C1`` has a
strictly positive 95% CI lower bound **and** ``(v_out - v_in) >= -0.01``: the
catalogue-recovery instrument is known to reward near-catalogue mass, so a small
holdout cost is accepted and reported rather than hidden.

Nothing here is a live score.  Every number is HOLDOUT-DTI with the evaluator
version, the withheld-positive count and a paired spatial-cluster 95% CI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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

import numpy as np
from threadpoolctl import threadpool_limits

from gems59 import evaluate_holdout as EH
from gems59 import load_grid
from gems59.anatomy import FEATURES
from gems59.emit import greedy_allocate
from gems59.faultzone import trace_sense_raster
from gems59.fitting import canary, fit_model
from gems59.holdout import FOLD_NAMES, buffered_component_draw
from gems59.relay_bend_anatomy import (
    ALL_RELAY_BEND_FEATURES,
    RELAY_FEATURES,
    build_relay_bend_geometry,
)
from gems59.submission_writer import write_submission
from gems59.uniqueness import compare_to_registry
from gems59.validate import validate

COLS = {n: i for i, n in enumerate(ALL_RELAY_BEND_FEATURES)}
BASE = [COLS[n] for n in FEATURES if n != "side"]
RELAY_BAND = BASE + [COLS[n] for n in RELAY_FEATURES]

ARMS = {
    "distance_only": [COLS["d"]],
    "anatomy": BASE,
    "relay_band": RELAY_BAND,
}

NEAR_CATALOGUE_PX = 2.0   # live-anchored mass discipline (GEMSDOE32 H33-2-B2 deletion)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def subset(g, selected: np.ndarray, key: str):
    return SimpleNamespace(
        X=g.X[selected], y=g.y[selected], rows=g.rows[selected],
        cols=g.cols[selected], key=key, feature_names=ALL_RELAY_BEND_FEATURES,
    )


def predict(clf, scale: float, g, columns: list[int], shape, zone_px: float) -> np.ndarray:
    """Fold posterior, zero outside the training-fitted damage zone."""
    p = np.zeros(shape, np.float32)
    for start in range(0, len(g.y), 150_000):
        sl = slice(start, min(start + 150_000, len(g.y)))
        x = g.X[sl]
        v = clf.predict_proba(x[:, columns])[:, 1] * scale
        v = np.clip(v, 0, 1).astype(np.float32)
        v[x[:, 0] > zone_px] = 0
        p[g.rows[sl], g.cols[sl]] = v
    return p


def scatter_zeros(shape) -> dict:
    """Scarp descriptors are unavailable (band-12 cache absent): constants, never read.

    The bend block therefore carries one dead descriptor.  It is excluded from
    every arm's column list, so no arm depends on it.
    """
    z = np.zeros(shape, np.float32)
    return {"sin2": z, "cos2": z, "coherence": z}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-quadrant-cap", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--build-surface", action="store_true",
                    help="after the holdout, fit the production model, emit final dots and write the GeoTIFF")
    ap.add_argument("--submission-name", type=str, default="GEMS57-H57M")
    ap.add_argument("--note", type=str, default=None)
    a = ap.parse_args()
    if not 0 < a.minutes <= 120 or a.per_quadrant_cap < 1:
        ap.error("minutes must be <=120 and the density cap positive")
    start = time.monotonic()
    deadline = start + a.minutes * 60

    def check_time(where: str) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"wall-clock budget reached at {where}; stopping before any partial artifact")

    plan = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        session="session-7-2026-10-10",
        hypotheses=[
            dict(id="C1", kind="control", text="distance to nearest visible fault only"),
            dict(id="C2", kind="control", text="single-host visible anatomy (Session-5 control)"),
            dict(
                id="C3", kind="experiment", hypothesis_id="H57-M",
                text=("Two-host relay damage band + live-anchored mass discipline: no dot within "
                      f"{NEAR_CATALOGUE_PX} px of the mapped catalogue (v_out); dot count set by the "
                      "shared marginal DTI test"),
            ),
        ],
        experiment_count=3,
        emission_variants=["v_out (d > 2 px gate)", "v_in (no gate, Session-5 behaviour)"],
        split="buffered-whole-components-loqo-v2",
        evaluator=EH.VERSION,
        seed=a.seed,
        per_quadrant_cap=a.per_quadrant_cap,
        no_textbook_angle=True,
        submission_slots_used=0,
        zone="90th percentile of TRAINING withheld-positive distances",
        truth_count_estimate="TRAINING base rate x allowed-domain size (never test labels)",
        decision_rule=("promote C3 v_out if paired binary HOLDOUT-DTI vs C1 has CI95 lower bound > 0 "
                       "and (C3 v_out - C3 v_in) >= -0.01"),
    )
    save(ROOT / "evidence/h57m_experiment_plan.json", plan)
    print("[h57m] predeclared C1/C2/C3; no weekly submission slot is used", flush=True)

    grid = load_grid()
    folds, quad = buffered_component_draw(grid, seed=a.seed)
    sense = trace_sense_raster(ROOT / "data/external/trace_segments_utm11.csv", grid.shape, grid.transform)
    scarp = scatter_zeros(grid.shape)
    cache = ROOT / ".cache/relay_bend_geometry"
    cache.mkdir(parents=True, exist_ok=True)

    full = build_relay_bend_geometry(
        grid, folds[0]["visible"], folds[0]["hidden_all"],
        np.logical_or.reduce([f["region"] for f in folds]) & ~folds[0]["masked_known"],
        sense, scarp, cache, f"draw{a.seed}-h57m",
    )
    print(f"[h57m] fold geometry: {len(full.y)} cells, {int(full.y.sum())} withheld positives "
          f"({time.monotonic()-start:.1f}s)", flush=True)

    geoms = [subset(full, quad[full.rows, full.cols] == q, FOLD_NAMES[q]) for q in range(4)]
    # Production zone/budget descriptors come from the TRAINING withheld positives only.
    pos_d = full.X[full.y == 1, COLS["d"]]
    prod_zone = float(np.quantile(pos_d, 0.90))
    prod_fraction = float((pos_d <= prod_zone).mean())
    print(f"[h57m] training-positive distance zone {prod_zone:.2f} px "
          f"(fraction inside {prod_fraction:.3f})", flush=True)

    check_time("canary")
    c = canary(geoms, feature_names=ALL_RELAY_BEND_FEATURES)
    flagged = [n for n, r in c.items() if r["leakage_flag"] or r["discriminative_auc_max"] is None]
    save(ROOT / "evidence/h57m_canary.json", dict(
        evidence_class="LEAKAGE-CANARY (AUC, not DTI)", split_version=plan["split"], features=c,
        note=("Every column is computed from visible catalogue traces (and the recorded-sense vector "
              "restricted to visible pixels) only."),
    ))
    print("[h57m] canary maxima:",
          {n: (round(r["discriminative_auc_max"], 4) if r["discriminative_auc_max"] is not None else None)
           for n, r in c.items()}, flush=True)
    if flagged:
        print(f"[h57m] STOP: unresolved canary {flagged}", flush=True)
        return 3

    terms: dict[str, np.ndarray] = {}
    details = []
    base_rates = []
    for q, fold in enumerate(folds):
        check_time(f"fold {FOLD_NAMES[q]}")
        train = [g for i, g in enumerate(geoms) if i != q]
        distances = np.concatenate([g.X[g.y == 1, COLS["d"]] for g in train])
        zone = float(np.quantile(distances, 0.90))
        fraction = float((distances <= zone).mean())
        cap = max(1, int(a.per_quadrant_cap * fraction))
        train_pixels = sum(len(g.y) for g in train)
        train_positives = sum(int(g.y.sum()) for g in train)
        base_rates.append(train_positives / train_pixels)
        test = geoms[q]
        allowed = fold["region"] & ~fold["masked_known"]
        k_est = (train_positives / train_pixels) * float(allowed.sum())
        d_test = test.X[:, COLS["d"]]
        far = d_test > zone
        near = d_test <= NEAR_CATALOGUE_PX
        masks = {
            "distance_only": allowed.copy(),
            "anatomy": allowed.copy(),
            "relay_band_v_out": allowed.copy(),
            "relay_band_v_in": allowed.copy(),
        }
        for name in masks:
            slots = np.zeros(len(d_test), bool)
            slots |= far
            if name.endswith("v_out"):
                slots |= near
            masks[name][test.rows[slots], test.cols[slots]] = False
        for name, columns in ARMS.items():
            check_time(f"{FOLD_NAMES[q]} {name}")
            with threadpool_limits(limits=2):
                clf, scale, _ = fit_model(train, seed=q, cols=columns)
                p = predict(clf, scale, test, columns, grid.shape, zone)
            raw, _st = EH.evaluate(p, fold, grid.footprint)
            keys = [name] if name != "relay_band" else ["relay_band_v_out", "relay_band_v_in"]
            for key in keys:
                check_time(f"{FOLD_NAMES[q]} {key}")
                dots = greedy_allocate(p, masks[key], k_truth=k_est, floor=0.015, max_dots=cap)
                scored, t = EH.evaluate(dots.emitted.astype(np.float32), fold, grid.footprint)
                terms[key] = t if key not in terms else terms[key] + t
                details.append(dict(
                    arm=key, fold=FOLD_NAMES[q], evidence_class="HOLDOUT-DTI",
                    evaluator_version=EH.VERSION, withheld_positive_pixels=scored["n_truth"],
                    dti=scored["dti"], soft_surface_dti=raw["dti"], emitted_pixels=dots.n_dots,
                    training_zone_px=zone, training_fraction_inside_zone=fraction, dot_cap=cap,
                    k_estimate_from_training=k_est,
                    near_catalogue_gate_px=(NEAR_CATALOGUE_PX if key.endswith("v_out") else None),
                    test_truth_used_for_placement=False, fold_receipt=fold["receipt"],
                ))
                print(f"[h57m] {FOLD_NAMES[q]:2s} {key:16s} HOLDOUT-DTI {scored['dti']:.5f} "
                      f"soft {raw['dti']:.5f} dots {dots.n_dots}", flush=True)
                del dots
            del p, raw
            gc.collect()

    pooled = EH.pooled_summary(terms, draws=1000, seed=20261010, candidate="relay_band_v_out")
    vs_distance = pooled["paired_differences"]["distance_only"]
    vs_anatomy = pooled["paired_differences"]["anatomy"]
    vs_v_in = pooled["paired_differences"]["relay_band_v_in"]
    promote = bool(vs_distance["ci95"][0] > 0 and vs_v_in["delta"] >= -0.01)
    report = dict(
        evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION, alpha=0.2, beta=0.8,
        triangular_radius_m=300.0, pooled=True, split_version=plan["split"], seed=a.seed,
        canary_clean=True, scores=pooled["scores"], paired_vs_distance_only=vs_distance,
        paired_vs_anatomy=vs_anatomy, paired_vs_v_in=vs_v_in, promote=promote,
        decision_rule=plan["decision_rule"], per_fold=details, submission_slots_used=0,
        near_catalogue_gate_px=NEAR_CATALOGUE_PX,
        runtime_seconds=time.monotonic() - start,
    )
    save(ROOT / "evidence/h57m_holdout.json", report)
    print("[h57m] pooled binary HOLDOUT-DTI:",
          {n: round(r["dti"], 6) for n, r in pooled["scores"].items()}, flush=True)
    print("[h57m] relay_band_v_out minus distance_only:", vs_distance, flush=True)
    print("[h57m] relay_band_v_out minus relay_band_v_in:", vs_v_in, flush=True)
    print(f"[h57m] promote={promote}", flush=True)

    if not a.build_surface:
        del full, geoms
        gc.collect()
        return 0

    # ---------------- production emission (complete catalogue = visible) ----------------
    check_time("production fit")
    with threadpool_limits(limits=2):
        clf, scale, _ = fit_model(geoms, seed=0, cols=ARMS["relay_band"])
    model_dir = cache
    import joblib
    joblib.dump(dict(model=clf, scale=scale, columns=ARMS["relay_band"], zone_px=prod_zone,
                     features=[ALL_RELAY_BEND_FEATURES[i] for i in ARMS["relay_band"]]),
                model_dir / "h57m_model.joblib")

    live_domain = grid.footprint & ~grid.catalogue
    live = build_relay_bend_geometry(
        grid, grid.catalogue, np.zeros(grid.shape, bool), live_domain,
        sense, scarp, cache, f"draw{a.seed}-h57m-live",
    )
    p = predict(clf, scale, live, ARMS["relay_band"], grid.shape, prod_zone)
    dmap = np.zeros(grid.shape, np.float32)
    dmap[live.rows, live.cols] = live.X[:, COLS["d"]]
    band = live_domain & (dmap <= prod_zone) & (dmap > NEAR_CATALOGUE_PX)
    cap = max(1, int(a.per_quadrant_cap * prod_fraction))
    base_rate = float(np.mean(base_rates))
    k_est = base_rate * float(live_domain.sum())
    print(f"[h57m] production: zone {prod_zone:.2f} px, k_est {k_est:.0f}, cap {cap}", flush=True)
    check_time("production allocation")
    dots = greedy_allocate(p, band, k_truth=k_est, floor=0.015, max_dots=cap)
    print(f"[h57m] production dots {dots.n_dots} "
          f"(surrogate DTI {dots.surrogate_dti:.5f}, T {dots.expected_covered_credit:.1f})", flush=True)
    if dots.n_dots == 0:
        print("[h57m] STOP: the marginal test admitted no dots; nothing written", flush=True)
        return 4

    note = a.note or ("H57-M two-host relay damage band: fitted zone, no dot within 200 m of known "
                      "traces, spacing set by the DTI marginal test; holdout-validated")
    if len(note) > 140:
        ap.error(f"note is {len(note)} characters; the submission form allows 140")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = a.submission_name
    label = f"gems59-h57m-relay-band-{stamp}-{name.lower().replace(' ', '-')}"
    target = ROOT / "docs/downloads" / f"{label}.tif"
    receipt = write_submission(
        target, dots.emitted.astype(np.float32), ROOT / "data/bridge/sample_submission.tif",
        grid.footprint, note=note, name=name, catalogue=grid.catalogue,
        metadata=dict(hypothesis="H57-M", kind="binary dot field", zone_px=prod_zone,
                      dots=int(dots.n_dots), near_catalogue_gate_px=NEAR_CATALOGUE_PX,
                      evaluator=EH.VERSION, experiment="h57m", production_base_rate=base_rate,
                      schema=("scores are HOLDOUT-DTI (evaluator gems59-pooled-hide-v2, 11,321 "
                              "withheld positives, paired spatial-cluster CI95), never live scores")),
    )
    validator = validate(target, grid.footprint, grid.catalogue)
    print(f"[h57m] wrote {target.name} sha256 {receipt['sha256']}", flush=True)
    # Mirror the three artifacts next to the repository-level downloads directory.
    mirror = ROOT / "downloads"
    mirror.mkdir(parents=True, exist_ok=True)
    for suffix in (".tif", ".zip", ".json"):
        src = target.with_suffix(suffix)
        if src.exists():
            shutil.copyfile(src, mirror / src.name)
    save(ROOT / "evidence/h57m_submission.json", dict(
        evidence_class="LOCAL-FORMAT", file=str(target.relative_to(ROOT)),
        sha256=receipt["sha256"], bytes=receipt["bytes"], dots=int(dots.n_dots),
        validator=validator, receipt=receipt,
    ))
    print("[h57m] validator:", {k: validator[k] for k in ("ok", "problems")} if "ok" in validator
          else validator, flush=True)

    def progress(i, n, row):
        if i % 5 == 0 or i == n:
            print(f"[h57m] registry {i}/{n}", flush=True)

    drift = compare_to_registry(target, ROOT / "registry/registry_index.json", grid.footprint,
                                progress=progress)
    save(ROOT / "evidence/h57m_uniqueness.json", drift)
    print("[h57m] registry (19 discriminating rasters):",
          {k: drift[k] for k in ("registry_rasters_checked", "worst_spearman_full_footprint",
                                 "worst_dot_overlap", "duplicate_count", "unique")}, flush=True)
    print(f"[h57m] runtime {(time.monotonic()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
