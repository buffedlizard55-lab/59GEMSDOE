#!/usr/bin/env python3
"""Build the Session-59 intensity-lane submission GeoTIFF.

Fits the presence-only point-process intensity on the **full** catalogue (no
holdout at inference time), zones the footprint into 1 km blocks by expected
new-fault density, and tiles only the blocks the holdout-derived utility table
clears, with its per-class spacing and catalogue collar.  Packaging goes
through the shared fail-closed writer (all-finite, [0,1], zeros outside the
footprint, zero positive mass on the mapped catalogue, single-TIFF ZIP).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import intensity as INT                            # noqa: E402
from gems59.gates import lane_uniqueness_report, write_report  # noqa: E402
from gems59.grid import load_grid                              # noqa: E402
from gems59.submission_writer import write_submission          # noqa: E402

GEO_ONLY = tuple(n for n, _, _ in INT.EDGE_FEATURES) + tuple(n for n, _, _ in INT.REGIME_FEATURES) \
    + ("g_edge_max",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, required=True,
                    help="Phillips effort-correction exponent chosen on the holdout")
    ap.add_argument("--collar", type=int, default=0, choices=(0, 2),
                    help="catalogue-flank collar (px) chosen on the holdout")
    ap.add_argument("--break-even", type=float, default=1.0)
    ap.add_argument("--utility", default="evidence/intensity_exp2_utility_table.json")
    ap.add_argument("--geo-only", action="store_true",
                    help="fit on geophysical covariates only (no effort features)")
    ap.add_argument("--out", default="downloads/gems59-h59i-intensity-lattice.tif")
    ap.add_argument("--registry", default="evidence/registry_refreshed_20261010T2001.json",
                    help="pinned complete accessible raster inventory; missing cache rows fail closed")
    ap.add_argument("--name", default="gems59-h59i-intensity-lattice-20261011")
    ap.add_argument("--note", default=(
        "H59-I intensity lane: MaxEnt/Poisson point-process intensity with effort "
        "covariates, 1 km new-fault-density zoning, holdout-derived break-even tiling."))
    a = ap.parse_args()
    start = time.monotonic()
    if not INT.TARGET_GROUP_BACKGROUND_IMPLEMENTED:
        raise SystemExit(
            "STOP: target-group background is not implemented; no candidate construction "
            "or submission export is permitted."
        )

    # Fail before constructing anything unless the selected held-out candidate
    # clears the comparable published reference. This is HOLDOUT evidence, not a
    # portal score and not permission to consume a competition slot.
    holdout_path = ROOT / "evidence/intensity_exp3_final.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    if (holdout.get("evidence_class") != "HOLDOUT-DTI"
            or holdout.get("evaluator_version") != "gems57-pooled-hide-v2"):
        raise SystemExit("holdout gate missing/mismatched; refusing submission construction")
    if holdout.get("protocol_qualification", {}).get("qualified") is not True:
        raise SystemExit("HOLDOUT-DTI is exploratory or lacks required leakage/target-group qualification")
    holdout_score = holdout["pooled"]["scores"].get("final_c2")
    reference = holdout.get("published_reference", {}).get("relay_bend_anatomy_dti")
    if not holdout_score or reference is None:
        raise SystemExit("holdout gate lacks final-collar score or comparable reference")
    if (holdout_score.get("withheld_positive_pixels") != 11321
            or holdout_score["dti"] <= reference):
        raise SystemExit(
            "HOLDOUT GATE FAILED — selected H59-I final_c2 did not beat the comparable "
            f"holdout reference ({holdout_score.get('dti'):.6f} vs {reference:.6f}); "
            "no TIF was created and no slot was used."
        )
    if a.gamma != 0.0 or a.collar != 2 or a.geo_only:
        raise SystemExit("build settings differ from the holdout-selected H59-I arm (gamma=0, collar=2, full features)")

    data_dir = ROOT / "data/official"
    grid = load_grid(data_dir)
    fp = grid.footprint
    cat = grid.catalogue
    prior = float(cat.sum() / fp.sum())
    registry_path = Path(a.registry)
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    if not registry_path.is_file():
        raise SystemExit(f"strict lane gate requires the registry inventory: {registry_path}")
    registry_index = json.loads(registry_path.read_text(encoding="utf-8"))

    def require_lane_gate(values: np.ndarray, phase: str) -> None:
        report = lane_uniqueness_report(
            values,
            fp,
            sample=data_dir / "sample_submission.tif",
            registry_index=registry_index,
            phase=phase,
        )
        receipt_path = ROOT / f"evidence/intensity_{phase}_gate.json"
        write_report(receipt_path, report)
        if not report["ok"]:
            raise SystemExit(
                f"STRICT {phase.upper()} REGISTRY GATE FAILED — STOP before output. "
                f"See {receipt_path.relative_to(ROOT)}; no submission slot or TIFF was created."
            )

    names = GEO_ONLY if a.geo_only else INT.FEATURE_NAMES

    utility = json.loads((ROOT / a.utility).read_text())
    table = utility["utility_table"]
    spacings = {}
    for cn, row in table.items():
        if row["selected"] == "none":
            continue
        spacings[cn] = float(row["selected"][1:])     # "s3"/"s4"/"s6" -> px
    if not spacings:
        raise SystemExit("utility table tiles nothing; refusing to build an empty submission")
    print(f"[build] utility spacings {spacings} gamma={a.gamma} collar={a.collar}", flush=True)

    planes = INT.covariate_planes(grid, cat, ROOT / "out/features")
    rows, cols = np.nonzero(cat)
    fit, _ = INT.fit_intensity(planes, (rows, cols), fp, names=names, prior=prior, seed=59)
    lam = INT.predict_logit(planes, fit.names, fit.beta, fit.intercept, fp)
    lam = INT.effort_correct(lam, planes, a.gamma, fp)
    require_lane_gate(lam, "surface")
    n_b, n_b_cell = INT.zone_new_fault_density(lam, cat)
    sel = n_b >= a.break_even
    vals = n_b[sel]
    edges = np.quantile(vals, [0.25, 0.5, 0.75])
    cls_block = np.digitize(n_b, edges) + 1
    cls_block[~sel] = 0
    cls_names = list(table)
    tiled = {cn: (cls_block == i + 1) for i, cn in enumerate(cls_names)
             if cn in spacings}
    print(f"[build] blocks above break-even {int(sel.sum())}; tiled classes {list(tiled)}",
          flush=True)

    elig = fp & ~cat
    if a.collar:
        elig &= ~ndi.binary_dilation(cat, iterations=a.collar)
    anchors = INT.block_anchors(lam, elig)
    dots = np.zeros(fp.shape, bool)
    for cn, mask in tiled.items():
        dots |= INT.tile_blocks(n_b_cell, lam, elig, spacings[cn],
                                classes=mask.astype(np.int32), anchors=anchors)
    dots &= fp
    pred = dots.astype(np.float32)
    pred[~fp] = 0.0
    print(f"[build] dots {int(pred.sum())} ({time.monotonic()-start:.0f}s)", flush=True)
    require_lane_gate(pred, "dots")

    receipt = write_submission(
        ROOT / a.out, pred, data_dir / "sample_submission.tif", fp,
        note=a.note, name=a.name, catalogue=cat,
        metadata=dict(
            lane="intensity", hypothesis="H59-I", gamma=a.gamma, collar_px=a.collar,
            break_even_min_density=a.break_even, utility_spacings=spacings,
            n_blocks_above_break_even=int(sel.sum()),
            expected_new_fault_pixels=float(n_b.sum()),
            intensity_names=list(fit.names),
            intensity_beta={n: round(float(b), 6) for n, b in zip(fit.names, fit.beta)},
            intensity_intercept=round(float(fit.intercept), 6),
            prior_mapped_fraction=prior, evaluator="gems57-pooled-hide-v2",
            generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k in ("file", "sha256", "bytes", "note_chars")}, indent=1))
    print(f"[done] {(time.monotonic()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
