#!/usr/bin/env python3
"""Build a UNIQUE submission using multi-fault interaction zone intensity.

H57-K: Multi-fault stress interaction model.
- Analytical (no learned tree)
- Cumulative damage-zone overlap from ALL nearby faults
- Fault tip stress concentration (LEFM)
- Fault bend curvature
- Top-k allocation (different from greedy max-coverage)
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
from gems59.holdout import build_holdout
from gems59.metric import dti_binary, dti_from_components
from gems59.uniqueness import compare_to_registry
from gems59.validate import validate
from gems59.network import STRUCT3

EVID = ROOT / "evidence"
DL = ROOT / "docs" / "downloads"


def _component_weights(catalogue: np.ndarray, length_scale: float = 0.5) -> np.ndarray:
    """Weight each fault pixel by its component's L^length_scale. Vectorized."""
    comp, ncomp = ndi.label(catalogue, structure=STRUCT3)
    sizes = np.bincount(comp.ravel(), minlength=ncomp + 1).astype(np.float64)
    weights = np.zeros(ncomp + 1, np.float64)
    weights[1:] = np.power(sizes[1:], length_scale)
    # Map component label -> weight for each pixel
    w = weights[comp.ravel()].reshape(catalogue.shape).astype(np.float32)
    w[~catalogue] = 0.0
    return w


def _tip_field(catalogue: np.ndarray, tip_enhance: float = 2.0,
               tip_r: float = 8.0) -> np.ndarray:
    """LEFM-inspired tip stress field: 1/sqrt(r) capped, within tip_r."""
    cat_int = catalogue.astype(np.int16)
    n_neighbors = ndi.convolve(cat_int, np.ones((3, 3), np.int16), mode="constant") - cat_int
    tips = catalogue & (n_neighbors <= 1)
    if not tips.any():
        return np.zeros(catalogue.shape, np.float32)
    d_tips = ndi.distance_transform_edt(~tips).astype(np.float32)
    field = np.where(
        (d_tips > 0) & (d_tips <= tip_r),
        tip_enhance / np.sqrt(np.maximum(d_tips, 0.5)),
        0.0
    ).astype(np.float32)
    return field


def _bend_field(catalogue: np.ndarray) -> np.ndarray:
    """Fault bend curvature enhancement."""
    from gems59.network import local_strike
    _, coherence = local_strike(catalogue, smooth_px=3.0)
    bend_weight = np.where(catalogue & (coherence < 0.5) & np.isfinite(coherence),
                           (1.0 - coherence) * 2.0, 0.0).astype(np.float32)
    if not bend_weight.any():
        return np.zeros(catalogue.shape, np.float32)
    bend_kernel = np.ones((5, 5), np.float32) / 25.0
    return ndi.convolve(bend_weight, bend_kernel, mode="constant")


def multi_fault_stress(catalogue: np.ndarray, footprint: np.ndarray,
                      r_max: float = 15.0,
                      tip_enhance: float = 2.0,
                      length_scale: float = 0.5) -> np.ndarray:
    """Analytical multi-fault interaction intensity."""
    # Weighted fault density convolution
    w = _component_weights(catalogue, length_scale)
    weighted_cat = catalogue.astype(np.float32) * w

    r = int(np.ceil(r_max))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dd = np.sqrt(yy.astype(np.float64)**2 + xx.astype(np.float64)**2)
    kernel = np.maximum(1.0 - dd / r_max, 0.0).astype(np.float32)

    stress = ndi.convolve(weighted_cat, kernel, mode="constant")

    # Add tip and bend enhancements
    stress += _tip_field(catalogue, tip_enhance, tip_r=min(8.0, r_max * 0.5))
    stress += _bend_field(catalogue)

    # Zero out on catalogue and outside footprint
    stress[catalogue] = 0.0
    stress[~footprint] = 0.0

    return stress.astype(np.float32)


def calibrate_surface(stress: np.ndarray, base_rate: float) -> np.ndarray:
    """Convert stress to probability in [0,1] with moment matching."""
    eligible = stress > 0
    if not eligible.any():
        return stress.copy()

    vals = stress[eligible]
    threshold_high = np.percentile(vals, max(0, 100 * (1 - 3 * base_rate)))
    threshold_low = np.percentile(vals, max(0, 100 * (1 - 10 * base_rate)))

    scale = 1.0 / max(threshold_high - threshold_low, 1e-6)
    prob = np.clip((stress - threshold_low) * scale, 0.0, 1.0).astype(np.float32)

    mean_p = prob[eligible].mean()
    if mean_p > 0:
        adj = base_rate / mean_p
        prob = np.clip(prob * adj, 0.0, 1.0).astype(np.float32)

    return prob


def topk_allocation(prob: np.ndarray, allowed: np.ndarray, budget: int) -> np.ndarray:
    """Top-k allocation: place budget dots at highest-probability pixels."""
    eligible = allowed & (prob > 0)
    if not eligible.any():
        return np.zeros_like(prob, bool)

    vals = prob[eligible]
    ys, xs = np.nonzero(eligible)

    if len(ys) <= budget:
        dots = np.zeros_like(prob, bool)
        dots[ys, xs] = True
        return dots

    order = np.argsort(-vals, kind="stable")[:budget]
    dots = np.zeros_like(prob, bool)
    dots[ys[order], xs[order]] = True
    return dots


def holdout_evaluate(grid, budget=30000, r_max=15.0, tip_enhance=2.0,
                     length_scale=0.5, mode="all") -> dict:
    """LOQO holdout evaluation."""
    ctx = build_holdout(grid)
    cells = ctx.cells_of(mode)
    results = []

    for cell in cells:
        visible = ctx.visible(cell.key)
        stress = multi_fault_stress(visible, grid.footprint,
                                   r_max=r_max, tip_enhance=tip_enhance,
                                   length_scale=length_scale)
        active = np.zeros(grid.shape, bool)
        active[cell.bbox] = cell.active
        eligible = active & ~visible
        base_rate = cell.n_truth / max(int(eligible.sum()), 1)
        prob = calibrate_surface(stress, base_rate)
        dots = topk_allocation(prob, eligible, budget=budget // 4)
        truth = np.zeros(cell.active.shape, bool)
        truth[cell.truth_yx] = True
        r = dti_binary(dots[cell.bbox], truth, valid=cell.active)
        results.append({
            "key": cell.key, "n_truth": cell.n_truth, "n_dots": int(dots.sum()),
            "dti": r["dti"], "coverage": r["coverage"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        })
        print(f"    {cell.key}: DTI={r['dti']:.4f} cov={r['coverage']:.4f} "
              f"dots={int(dots.sum())} truth={cell.n_truth}")

    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    n_truth = sum(r["n_truth"] for r in results)
    pooled_dti = dti_from_components(tp, fp, fn)
    coverage = tp / max(n_truth, 1)

    quads = sorted({r["key"].split("_")[1] for r in results})
    jk = []
    for q in quads:
        rs = [r for r in results if r["key"].split("_")[1] != q]
        jk.append(dti_from_components(
            sum(r["tp"] for r in rs),
            sum(r["fp"] for r in rs),
            sum(r["fn"] for r in rs)))
    jk = np.array(jk)
    m = len(quads)
    jk_se = float(np.sqrt((m - 1) / m * ((jk - jk.mean()) ** 2).sum())) if m > 1 else 0.0

    return {
        "pooled_dti": pooled_dti, "coverage": coverage,
        "n_truth": n_truth, "n_dots": sum(r["n_dots"] for r in results),
        "ci95": [float(pooled_dti - 1.96 * jk_se), float(pooled_dti + 1.96 * jk_se)],
        "jk_drop": {q: float(v) for q, v in zip(quads, jk)},
        "per_cell": results, "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("57GEMSDOE H57-K: Multi-Fault Interaction Zone Model")
    print("=" * 70)

    g = load_grid()
    print(f"[{time.time()-t0:.0f}s] Grid: {g.shape}, {int(g.catalogue.sum())} cat px")

    # --- Experiment 1: LOQO holdout at budget=30000 ---
    print(f"\n[Exp 1] LOQO holdout, budget=30000, r_max=15")
    r1 = holdout_evaluate(g, budget=30000, r_max=15.0)
    print(f"  HOLDOUT-DTI = {r1['pooled_dti']:.4f} CI={r1['ci95']}")
    print(f"  Coverage = {r1['coverage']:.4f}, dots = {r1['n_dots']}")

    SHIPPED_DTI = 0.2279
    beats = r1["pooled_dti"] > SHIPPED_DTI
    print(f"  Shipped anatomy: {SHIPPED_DTI:.4f} → {'BEATS' if beats else 'DOES NOT BEAT'}")

    # --- Experiment 2: Try r_max=10 for faster/smaller zones ---
    print(f"\n[Exp 2] LOQO holdout, budget=30000, r_max=10")
    r2 = holdout_evaluate(g, budget=30000, r_max=10.0)
    print(f"  HOLDOUT-DTI = {r2['pooled_dti']:.4f} CI={r2['ci95']}")

    # Pick best
    if r2["pooled_dti"] > r1["pooled_dti"]:
        best, best_r = r2, 10.0
    else:
        best, best_r = r1, 15.0
    print(f"  Best: r_max={best_r:.0f} DTI={best['pooled_dti']:.4f}")

    # --- Build final submission ---
    print(f"\n[Exp 3] Building final submission with r_max={best_r:.0f}")
    stress = multi_fault_stress(g.catalogue, g.footprint, r_max=best_r)
    base_rate = 0.002339
    prob = calibrate_surface(stress, base_rate)

    eligible = g.footprint & ~g.catalogue
    budget = 30000
    dots = topk_allocation(prob, eligible, budget=budget)
    n_dots = int(dots.sum())
    print(f"  Allocated {n_dots} dots")

    # Continuous probability at dot locations
    submission = np.zeros(g.shape, np.float32)
    submission[dots] = prob[dots]
    submission = np.where(g.footprint, np.clip(submission, 0.0, 1.0), 0.0).astype(np.float32)

    assert np.isfinite(submission).all()
    assert submission.min() >= 0.0
    assert submission.max() <= 1.0
    assert (submission[g.catalogue] == 0).all()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(submission.tobytes()).hexdigest()[:12]
    name = f"gems59-h57k-interaction-{stamp}-{digest}"
    note = f"57GEMSDOE H57-K interaction | {n_dots} dots 0 on-cat | {digest[:8]}"
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

    # Uniqueness
    uniq = compare_to_registry(zpath, ROOT / "registry" / "registry_index.json", g.footprint)
    print(f"  Uniqueness: unique={uniq['unique']}")
    print(f"    rho={uniq['worst_spearman_full_footprint']:.4f} "
          f"jaccard={uniq['worst_jaccard_dot_sets']:.4f} "
          f"overlap={uniq['worst_dot_overlap']:.4f}")

    promotable = vz["all_checks_passed"] and uniq["unique"]

    # Run card
    run_card = {
        "hypothesis": "Multi-fault stress interaction: secondary strands at damage-zone overlaps, fault tips (LEFM), and bends.",
        "mechanism": "Analytical stress field: cumulative kernel from all components (L^0.5 weighted), tip 1/sqrt(r), bend coherence. Top-k allocation.",
        "named_non_fault_process": "Drainage/ridge lineaments following regional stress fabric.",
        "holdout_dti": {
            "pooled_dti": best["pooled_dti"],
            "ci95": best["ci95"],
            "coverage": best["coverage"],
            "n_truth": best["n_truth"],
            "label": "HOLDOUT-DTI - local instrument, NOT projected live score",
        },
        "shipped_comparison": {"shipped_dti": SHIPPED_DTI, "beats": beats},
        "uniqueness": {"unique": uniq["unique"],
                       "worst_rho": uniq["worst_spearman_full_footprint"],
                       "worst_jaccard": uniq["worst_jaccard_dot_sets"],
                       "worst_overlap": uniq["worst_dot_overlap"]},
        "raster_sha256": vz["sha256"],
        "validator": {"all_passed": vz["all_checks_passed"], "n_checks": len(vz["checks"])},
        "submission_name": name,
        "submission_note": note,
        "submission_file": str(zpath),
        "verdict": "PROMOTE" if promotable else "NEGATIVE",
        "parameters": {"r_max": best_r, "budget": budget, "n_dots": n_dots},
        "runtime_s": time.time() - t0,
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "run_card_session4_interaction.json").write_text(json.dumps(run_card, indent=2))

    print(f"\n{'='*70}")
    print(f"VERDICT: {'PROMOTE' if promotable else 'NEGATIVE'}")
    if promotable:
        print(f"SUBMISSION FILE: docs/downloads/{zpath.name}")
        print(f"Name: {name}")
        print(f"Note: {note}")
    print(f"{'='*70}")


if __name__ == "__main__":
    raise SystemExit("Archived concurrent-session generator: its geometry predates IR-57-STRIKE-01 "
                     "and/or its validation does not satisfy the current buffered protocol or public-inventory checks. "
                     "No new generation or slot is authorized. See README.md and run_card_current.json.")
