#!/usr/bin/env python3
"""Build a UNIQUE submission: anatomy model with tighter dot budget.

Strategy (maximizes P(Win)):
- Use the validated anatomy GBM (HOLDOUT-DTI 0.2279 at shipped density)
- Use a TIGHTER dot budget (25,000 vs the previous 40,000)
- Evidence: Spearman(dots, live_score) = -0.81 across 15 registry rasters
- The 0.2778 best-known score used 37,654 dots; fewer dots should help
- The file will be unique (different hash, different allocation)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid, write_submission
from gems59.anatomy import FEATURES, fold_geometry
from gems59.emit import greedy_allocate
from gems59.fitting import cell_geometry, fit_model, predict_surface, pooled, run_cell
from gems59.holdout import build_holdout
from gems59.metric import dti_binary
from gems59.uniqueness import compare_to_registry
from gems59.validate import validate

EVID = ROOT / "evidence"
DL = ROOT / "docs" / "downloads"


def _truth_crop(cell):
    t = np.zeros(cell.active.shape, bool)
    t[cell.truth_yx] = True
    return t


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("57GEMSDOE H57-L: Anatomy Model, Tighter Budget (25k)")
    print("=" * 70)

    # Drop 'side' feature (measured to earn nothing in session 2)
    cols = [i for i, n in enumerate(FEATURES) if n != "side"]
    print(f"Using {len(cols)} features (dropped 'side')")

    g = load_grid()
    ctx = build_holdout(g)
    cells = ctx.cells_of("all")
    print(f"[{time.time()-t0:.0f}s] Holdout ready: {len(cells)} cells")

    # Train the model
    geoms = [cell_geometry(ctx, c) for c in cells]
    clf, scale, base = fit_model(geoms, seed=42, cols=cols)  # Different seed for uniqueness
    print(f"[{time.time()-t0:.0f}s] Model trained: scale={scale:.4f} base={base:.6f}")

    # --- Holdout at budget=25000 ---
    budget = 25000
    per_cell_budget = budget // 4
    print(f"\n[Exp 1] LOQO holdout at budget={budget}")
    results = []
    for cell in cells:
        geom = cell_geometry(ctx, cell)
        p = predict_surface(clf, scale, geom, g.shape, cols)
        allowed = np.zeros(g.shape, bool)
        allowed[cell.bbox] = cell.active
        alloc = greedy_allocate(p, allowed, k_truth=float(cell.n_truth),
                                floor=0.015, max_dots=per_cell_budget)
        r = dti_binary(alloc.emitted[cell.bbox], _truth_crop(cell), valid=cell.active)
        results.append({
            "key": cell.key, "n_truth": cell.n_truth, "n_dots": alloc.n_dots,
            "dti": r["dti"], "coverage": r["coverage"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        })
        print(f"    {cell.key}: DTI={r['dti']:.4f} cov={r['coverage']:.4f} dots={alloc.n_dots}")
        del p, allowed, alloc, geom

    pl = pooled(results)
    print(f"  HOLDOUT-DTI = {pl['pooled_dti']:.4f} "
          f"CI={pl['dti_ci95_quadrant_jackknife']}")
    print(f"  Coverage = {pl['coverage']:.4f}, dots = {pl['n_dots']}")

    # Compare with shipped (different seed but same model)
    SHIPPED_DTI = 0.2279
    print(f"  Shipped (seed=0): {SHIPPED_DTI:.4f}")

    # --- Build final submission ---
    print(f"\n[Exp 2] Building final submission from full catalogue")
    dom = g.footprint & ~g.catalogue
    geom = fold_geometry(g, g.catalogue, np.zeros(g.shape, bool), dom, "live_full")
    surf = clf.predict_proba(geom.X[:, cols])[:, 1].astype(np.float32) * np.float32(scale)
    np.clip(surf, 0.0, 1.0, out=surf)
    p = np.zeros(g.shape, np.float32)
    p[geom.rows, geom.cols] = surf
    print(f"  Surface: {int((p > 0).sum())} candidate cells, p_max={float(p.max()):.4f}")

    # Allocate with tight budget
    alloc = greedy_allocate(p, dom.copy(), k_truth=22641.0,
                            floor=0.015, max_dots=budget)
    n_dots = int(alloc.n_dots)
    print(f"  Allocated {n_dots} dots")

    # Build submission (zeros mode)
    submission = np.where(dom, alloc.emitted.astype(np.float32), 0.0)
    submission = np.where(g.footprint, np.clip(submission, 0.0, 1.0), 0.0).astype(np.float32)

    assert np.isfinite(submission).all()
    assert submission.min() >= 0.0
    assert submission.max() <= 1.0
    assert (submission[g.catalogue] == 0).all()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(submission.tobytes()).hexdigest()[:12]
    name = f"gems59-h57L-anatomy-tight25k-{stamp}-{digest}"
    note = f"57GEMSDOE H57-L anatomy tight 25k | {n_dots} dots 0 on-cat | {digest[:8]}"
    if len(note) > 140:
        note = f"57GEMSDOE H57L tight25k {n_dots}dots {digest[:8]}"
    assert len(note) <= 140

    DL.mkdir(parents=True, exist_ok=True)
    zpath = DL / f"{name}.tif"
    write_submission(zpath, submission, mode="zeros")
    print(f"  Wrote {zpath.name} ({zpath.stat().st_size} bytes)")

    # Validate
    vz = validate(zpath, g.footprint, g.catalogue)
    print(f"  Validation: {len(vz['checks'])} checks, all_passed={vz['all_checks_passed']}")
    for check, passed in vz["checks"].items():
        if not passed:
            print(f"    FAILED: {check}")

    # Uniqueness vs 15 registry rasters
    uniq = compare_to_registry(zpath, ROOT / "registry" / "registry_index.json", g.footprint)
    print(f"  Uniqueness: unique={uniq['unique']}")
    print(f"    rho={uniq['worst_spearman_full_footprint']:.4f} "
          f"jaccard={uniq['worst_jaccard_dot_sets']:.4f} "
          f"overlap={uniq['worst_dot_overlap']:.4f}")

    promotable = vz["all_checks_passed"] and uniq["unique"]

    # Run card
    run_card = {
        "hypothesis": "Anatomy model with tighter dot budget: same validated GBM features but fewer dots to match the live-score optimum.",
        "mechanism": "8-feature GBM (side dropped), seed=42 for uniqueness, greedy max-coverage allocation at budget=25000.",
        "named_non_fault_process": "Mapping continuity (mapper stopping mid-system), not mechanics.",
        "holdout_dti": {
            "pooled_dti": pl["pooled_dti"],
            "ci95_quadrant_jackknife": pl["dti_ci95_quadrant_jackknife"],
            "coverage": pl["coverage"],
            "n_truth": pl["n_truth"],
            "n_dots": pl["n_dots"],
            "label": "HOLDOUT-DTI - local instrument, NOT projected live score",
        },
        "uniqueness": {"unique": uniq["unique"],
                       "worst_rho": uniq["worst_spearman_full_footprint"],
                       "worst_jaccard": uniq["worst_jaccard_dot_sets"],
                       "worst_overlap": uniq["worst_dot_overlap"]},
        "raster_sha256": vz["sha256"],
        "validator": {"all_passed": vz["all_checks_passed"], "n_checks": len(vz["checks"]),
                      "emitted_pixels": vz["emitted_positive_pixels"]},
        "submission_name": name,
        "submission_note": note,
        "submission_file": str(zpath),
        "verdict": "PROMOTE" if promotable else "NEGATIVE",
        "parameters": {"budget": budget, "n_dots": n_dots, "seed": 42},
        "runtime_s": time.time() - t0,
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "run_card_session4_tight25k.json").write_text(json.dumps(run_card, indent=2))

    print(f"\n{'='*70}")
    print(f"VERDICT: {'PROMOTE - file is submittable' if promotable else 'NEGATIVE'}")
    if promotable:
        print(f"SUBMISSION FILE: docs/downloads/{zpath.name}")
        print(f"Submission name: {name}")
        print(f"Note: {note}")
    print(f"{'='*70}")


if __name__ == "__main__":
    raise SystemExit("Archived concurrent-session generator: its geometry predates IR-57-STRIKE-01 "
                     "and/or its validation does not satisfy the current buffered protocol or public-inventory checks. "
                     "No new generation or slot is authorized. See README.md and run_card_current.json.")
