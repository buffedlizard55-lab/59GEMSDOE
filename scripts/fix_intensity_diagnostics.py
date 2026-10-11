#!/usr/bin/env python3
"""Re-derive the Exp-1 descriptive tables with the corrected block aggregation.

IR-59-DIAG-01: ``run_intensity_lane.py`` computed the decile-calibration and
SGMC diagnostics by sum-pooling the *per-cell expansion* of the block totals,
which multiplies every reported block density by the 10x10 cells per block.
The ranking (and therefore every decile assignment and every DTI) is
unaffected; only the reported mean densities were 100x too large.  This script
re-fits the same four folds (same seeds, same planes) and rewrites the two
descriptive tables with correct block sums.  It fits nothing new and scores
nothing: no experiment budget is consumed.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import intensity as INT                            # noqa: E402
from gems59.grid import load_grid                              # noqa: E402
from gems59.holdout import FOLD_NAMES, buffered_component_draw  # noqa: E402


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    start = time.monotonic()
    grid = load_grid()
    fp = grid.footprint
    cat = grid.catalogue
    prior = float(cat.sum() / fp.sum())
    folds, quad = buffered_component_draw(grid, seed=20)
    visible = folds[0]["visible"]
    masked_known = folds[0]["masked_known"]
    planes = INT.covariate_planes(grid, visible, ROOT / "out/features")
    import rasterio
    with rasterio.open(ROOT / "data/external/sgmc_faults_100m.tif") as s:
        sgmc_only = ((s.read(1) > 0) & fp) & ~cat

    calib_rows, sgmc_rows = [], []
    for q, fold in enumerate(folds):
        train_mask = visible & (quad != q) & fp
        rows, cols = np.nonzero(train_mask)
        with threadpool_limits(limits=2):
            fit, _ = INT.fit_intensity(planes, (rows, cols), fp, prior=prior, seed=59 + q)
            lam = INT.predict_logit(planes, fit.names, fit.beta, fit.intercept, fp)
            n_b, _ = INT.zone_new_fault_density(lam, masked_known)
        truth = fold["truth"]
        pred_block = n_b.ravel()
        truth_block = INT.block_reduce(truth.astype(np.float32)).ravel()
        sel = (pred_block > 0) | (truth_block > 0)
        p, o = pred_block[sel], truth_block[sel]
        edges = np.quantile(p, np.linspace(0, 1, 11))
        edges[0] -= 1e-9
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        for d in range(10):
            m = idx == d
            if m.sum():
                calib_rows.append(dict(
                    fold=FOLD_NAMES[q], decile=d + 1, n_blocks=int(m.sum()),
                    mean_pred_new_fault_px=float(p[m].mean()),
                    sum_withheld=float(o[m].sum()), mean_withheld=float(o[m].mean())))
        sgmc_rows.append(dict(fold=FOLD_NAMES[q], **INT.sgmc_diagnostic(n_b, sgmc_only, fp)))
        print(f"[fix] fold {FOLD_NAMES[q]} done ({time.monotonic()-start:.0f}s)", flush=True)
        del lam, n_b
        gc.collect()

    save(ROOT / "evidence/intensity_exp1_diagnostics_fixed.json", dict(
        evidence_class="HOLDOUT-STRUCTURE (calibration, not a score)",
        fixes="IR-59-DIAG-01: block densities re-aggregated from per-block sums",
        evaluator="gems57-pooled-hide-v2", split_seed=20, withheld_positive_pixels=11321,
        calibration_rows=calib_rows, sgmc_offcatalogue_diagnostic=sgmc_rows,
        calibration_summary={}, sgmc_summary={}))
    # pooled summary across folds for the two headline statements
    by_dec = {}
    for r in calib_rows:
        by_dec.setdefault(r["decile"], []).append(r)
    summary = {}
    for d, rows in sorted(by_dec.items()):
        n = sum(r["n_blocks"] for r in rows)
        sw = sum(r["sum_withheld"] for r in rows)
        summary[d] = dict(n_blocks=n, sum_withheld=sw,
                          mean_pred=float(np.mean([r["mean_pred_new_fault_px"] for r in rows])),
                          mean_withheld=float(np.mean([r["mean_withheld"] for r in rows])))
    tot_w = sum(v["sum_withheld"] for v in summary.values())
    top3 = sum(v["sum_withheld"] for k, v in summary.items() if k >= 8)
    d2 = json.loads((ROOT / "evidence/intensity_exp1_diagnostics_fixed.json").read_text())
    d2["calibration_summary"] = dict(
        by_decile=summary, total_withheld_pixels=tot_w,
        share_withheld_in_top3_deciles=float(top3 / tot_w) if tot_w else None,
        note="withheld-segment pixel counts by decile of predicted 1 km new-fault density, "
             "pooled over the four folds")
    d2["sgmc_summary"] = dict(
        spearman_by_fold={r["fold"]: round(r["spearman_pred_vs_sgmc_only"], 4) for r in sgmc_rows},
        note="Spearman(predicted new-fault density, SGMC-minus-catalogue block counts)")
    save(ROOT / "evidence/intensity_exp1_diagnostics_fixed.json", d2)
    print("[fix] top-3-decile share of withheld pixels:",
          d2["calibration_summary"]["share_withheld_in_top3_deciles"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
