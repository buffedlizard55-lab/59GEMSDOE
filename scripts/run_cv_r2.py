#!/usr/bin/env python3
"""Session-2 experiment: redundancy ablation + H57-D interaction feature.

Experiment 1 (of this session's 3-experiment budget)
-----------------------------------------------------
REMAINING_WORK.md item 3 asked for the ablation that was never run: does the
en echelon geometry (``d_perp`` x ``d_par_abs``) still pay for itself when it
is removed from the FULL shipped feature set (rather than added to distance
alone)?  Variant ``no_rielder`` answers that.

Experiment 2
------------
H57-D (strike-selective gap filling): ``sin2``/``cos2`` are rank-degenerate as
marginals (AUC exactly 0.5000), so orientation selectivity is only usable in
interaction.  Two explicit interaction features were added to
:mod:`gems59.anatomy` (``sin2d``, ``cos2d`` = the cyclic strike encoding times
distance -- no hard-coded angle).  Variant ``gated`` measures them.

Variants (column indices refer to the extended 11-feature
:data:`gems59.anatomy.FEATURES`):

* ``shipped8``     -- exactly the feature set of the shipped session-1
  submission (control; must reproduce cv_all.json's ``no_side`` within
  numerical noise).
* ``no_rielder``   -- shipped8 minus ``d_perp``/``d_par_abs`` (ablation).
* ``gated``        -- shipped8 plus ``sin2d``/``cos2d`` (H57-D).
* ``anatomy_full`` -- all 11 features (upper reference).

Every number printed is a HOLDOUT-DTI instrument reading (leave-one-
quadrant-out, pooled, quadrant-jackknife CI).  No live score is projected.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid                                    # noqa: E402
from gems59.anatomy import FEATURES                              # noqa: E402
from gems59.fitting import (canary, cell_geometry, fit_model,    # noqa: E402
                            pooled, run_cell)
from gems59.holdout import FOLD_NAMES, build_holdout             # noqa: E402

EVID = ROOT / "evidence"
IDX = {n: i for i, n in enumerate(FEATURES)}

SHIPPED8 = ["d", "d_perp", "d_par_abs", "log_len", "sin2", "cos2", "coherence", "density"]

VARIANTS = {
    "shipped8":     [IDX[n] for n in SHIPPED8],
    "no_rielder":   [IDX[n] for n in SHIPPED8 if n not in ("d_perp", "d_par_abs")],
    "gated":        [IDX[n] for n in SHIPPED8 + ["sin2d", "cos2d"]],
    "anatomy_full": list(range(len(FEATURES))),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "detached"])
    ap.add_argument("--max-dots", type=int, default=120_000)
    ap.add_argument("--floor", type=float, default=0.015)
    ap.add_argument("--canary-only", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    ctx = build_holdout(load_grid())
    cells = ctx.cells_of(a.mode)
    print(f"[{time.time()-t0:5.1f}s] holdout ready: {len(cells)} cells, mode={a.mode}")

    # ---- 1. leakage canary over ALL features (incl. the two new ones) -------
    geoms_canary = []
    for c in cells[:4]:                      # 4 folds x 1 draw is enough for AUC
        geoms_canary.append(cell_geometry(ctx, c))
    can = canary(geoms_canary)
    print(f"\n[{time.time()-t0:5.1f}s] LEAKAGE CANARY (single-feature AUC, "
          f"{len(geoms_canary)} folds, {len(FEATURES)} features)")
    for name, r in can.items():
        flag = "  <== LEAKAGE FLAG" if r["leakage_flag"] else ""
        print(f"  {name:10s} raw_mean={r['auc_mean']:.4f} "
              f"discriminative_mean={r['discriminative_auc_mean']:.4f} "
              f"discriminative_max={r['discriminative_auc_max']:.4f}{flag}")
    geoms_canary.clear()
    gc.collect()
    if a.canary_only:
        EVID.mkdir(exist_ok=True)
        (EVID / f"canary_r2_{a.mode}.json").write_text(json.dumps(can, indent=2))
        print(f"wrote evidence/canary_r2_{a.mode}.json")
        return

    # ---- 2. leave-one-quadrant-out CV --------------------------------------
    out = {"mode": a.mode, "n_cells": len(cells), "n_features": len(FEATURES),
           "canary": can, "variants": {}}
    for vname, cols in VARIANTS.items():
        t1 = time.time()
        results = []
        for qi, q in enumerate(FOLD_NAMES):
            test = [c for c in cells if c.key.split("_")[1] == f"fold{q}"]
            train = [c for c in cells if c.key.split("_")[1] != f"fold{q}"]
            geoms = []
            for c in train:
                geoms.append(cell_geometry(ctx, c))
            clf, scale, base = fit_model(geoms, seed=qi, cols=cols)
            geoms.clear()
            gc.collect()
            for c in test:
                results.append(run_cell(ctx, c, clf, scale, cols=cols,
                                        max_dots=a.max_dots, floor=a.floor))
            del clf
            gc.collect()
        pl = pooled(results)
        out["variants"][vname] = {"cols": [FEATURES[i] for i in cols],
                                  "pooled": pl, "per_cell": results}
        ci = pl["dti_ci95_quadrant_jackknife"]
        print(f"[{time.time()-t1:5.1f}s] {vname:14s} HOLDOUT-DTI pooled={pl['pooled_dti']:.4f} "
              f"CI95(jackknife)=[{ci[0]:.4f},{ci[1]:.4f}] coverage={pl['coverage']:.4f} "
              f"dots={pl['n_dots']} tp={pl['tp']:.0f} fp={pl['fp']:.0f} K={pl['n_truth']}",
              flush=True)

    EVID.mkdir(exist_ok=True)
    (EVID / f"cv_r2_{a.mode}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote evidence/cv_r2_{a.mode}.json   total {time.time()-t0:.1f}s")

    sh = out["variants"]["shipped8"]["pooled"]["pooled_dti"]
    nr = out["variants"]["no_rielder"]["pooled"]["pooled_dti"]
    gt = out["variants"]["gated"]["pooled"]["pooled_dti"]
    print(f"\nablation  no_rielder vs shipped8: {nr - sh:+.4f} DTI")
    print(f"H57-D     gated      vs shipped8: {gt - sh:+.4f} DTI")


if __name__ == "__main__":
    main()
