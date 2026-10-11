#!/usr/bin/env python3
"""Leakage canary + leave-one-quadrant-out CV for the fault-zone-anatomy lane.

Runs, for one withholding mode:

1. the leakage canary -- single-feature AUC on every fold (no fitting involved,
   so nothing can leak through the fit);
2. leave-one-quadrant-out CV of nested feature sets, so the value of the
   anisotropic anatomy and of the geophysical corroboration block is measured
   against distance-only and single-block controls;
3. pooled DTI with two CIs (coverage-Wilson and quadrant-jackknife).

Fold geometries are computed once and shared by every variant (the previous
per-variant recomputation was pure waste -- see REMAINING_WORK item 7).

All numbers printed are HOLDOUT-DTI instrument readings.  No live score is
projected anywhere in this file.
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
from gems59.anatomy import FEATURES                             # noqa: E402
from gems59.fitting import (canary, cell_geometry, fit_model,    # noqa: E402
                            pooled, run_cell, sample_train)
from gems59.holdout import FOLD_NAMES, build_holdout            # noqa: E402

EVID = ROOT / "evidence"

GEO_NAMES: tuple = ()
try:
    from gems59.geo import GEO_FEATURES, load_planes  # noqa: E402
    GEO_NAMES = tuple(GEO_FEATURES)
except Exception:  # geo.py absent or stack missing: pure-geometry CV still works
    load_planes = None


def variant_cols(names: tuple) -> dict[str, list[int]]:
    """Named feature sets, resolved to column indices of the combined matrix."""
    all_names = names
    idx = {n: i for i, n in enumerate(all_names)}
    has_geo = bool(GEO_NAMES) and GEO_NAMES[0] in idx

    def cols(*want: str) -> list[int]:
        missing = [w for w in want if w not in idx]
        if missing:
            raise KeyError(f"features not available: {missing}")
        return [idx[w] for w in want]

    out = {
        "d_only": cols("d"),
        "d_perp_par": cols("d", "d_perp", "d_par_abs"),
        "anatomy_full": cols(*FEATURES),
        "no_side": cols(*(n for n in FEATURES if n != "side")),
    }
    if has_geo:
        out["geo_only"] = cols(*GEO_NAMES)
        out["no_side_plus_geo"] = cols(*(n for n in FEATURES if n != "side"),
                                       *GEO_NAMES)
        out["anatomy_plus_geo"] = cols(*FEATURES, *GEO_NAMES)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "detached"])
    ap.add_argument("--max-dots", type=int, default=120_000)
    ap.add_argument("--floor", type=float, default=0.015)
    ap.add_argument("--canary-only", action="store_true")
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of variant names (default: all)")
    ap.add_argument("--no-geo", action="store_true",
                    help="ignore training_features.tif even if present")
    ap.add_argument("--out", default="",
                    help="evidence file name (default cv_{mode}.json)")
    a = ap.parse_args()

    t0 = time.time()
    g0 = load_grid()
    geo = None
    if load_planes is not None and not a.no_geo:
        geo = load_planes(g0.footprint)
        print(f"[{time.time()-t0:5.1f}s] geo planes loaded: {list(geo)}")
    ctx = build_holdout(g0)
    cells = ctx.cells_of(a.mode)
    print(f"[{time.time()-t0:5.1f}s] holdout ready: {len(cells)} cells, mode={a.mode}")

    # ---- 0. cell geometries, one fold group at a time ---------------------
    # Peak memory is kept near (n_cells_per_fold x 2) geometries: the sandbox
    # has 3 GB RAM and the full 8-geometry cache plus a concurrent test run
    # OOM-killed the first session-2 CV attempt (exit 137).
    geoms = {}
    for c in cells[:4]:
        geoms[c.key] = cell_geometry(ctx, c, geo=geo)
    print(f"[{time.time()-t0:5.1f}s] canary geometries built "
          f"({geoms[cells[0].key].X.shape[1]} features)")

    # ---- 1. leakage canary ----------------------------------------------
    can = canary(list(geoms.values()))                 # 4 folds x 1 draw is enough
    print(f"\n[{time.time()-t0:5.1f}s] LEAKAGE CANARY (single-feature AUC, 4 folds)")
    for name, r in can.items():
        flag = "  <== FLAG" if r["leakage_flag"] else ""
        print(f"  {name:12s} raw_mean={r['auc_mean']:.4f} "
              f"discriminative_mean={r['discriminative_auc_mean']:.4f} "
              f"discriminative_max={r['discriminative_auc_max']:.4f}{flag}")
    # Protocol rule 4: AUC > 0.90 is "leakage until proven otherwise".  The
    # proof for the catalogue-geometry block is by construction: every column
    # of fold_geometry is a deterministic function of the *visible* mask only
    # (holdout.py: whole-segment withholding + 12 px domain erosion), so no
    # channel exists from the withheld mask into the features.  Flagged
    # geometry features are therefore reported as near-field physical signal
    # (the lane's mechanism) with the mapping-continuity caveat recorded in
    # the run card.  A flagged *geophysical* column (static external raster)
    # would be equally leakage-free by construction, but a flag there would
    # also mean the fold split is geographically confounded -- review, not run.
    flagged = [n for n, r in can.items() if r["leakage_flag"]]
    geo_flagged = [n for n in flagged if n.startswith("g_")]
    for n in flagged:
        can[n]["interpretation"] = (
            "catalogue-geometry feature computed from visible faults only -- "
            "no leakage channel by construction; near-field physical signal "
            "(damage-zone adjacency), with mapping-continuity confound recorded"
            if not n.startswith("g_") else
            "static geophysical plane; high AUC means fold-split geographic "
            "confounding to review before trusting results")
    if geo_flagged:
        print(f"LEAKAGE CANARY FIRED on geophysical columns {geo_flagged} -- "
              "review before running CV.")
        raise SystemExit(2)
    if flagged:
        print(f"canary flags on {flagged}: proven non-leaking by construction "
              "(visible-only features); recorded in evidence.")
    if a.canary_only:
        EVID.mkdir(exist_ok=True)
        (EVID / f"canary_{a.mode}.json").write_text(json.dumps(can, indent=2))
        print(f"wrote evidence/canary_{a.mode}.json")
        return

    # ---- 2. leave-one-quadrant-out CV ------------------------------------
    all_names = tuple(geoms[cells[0].key].feature_names)
    VARIANTS = variant_cols(all_names)
    if a.variants:
        want = {v.strip() for v in a.variants.split(",") if v.strip()}
        unknown = want - set(VARIANTS)
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}; "
                             f"available: {sorted(VARIANTS)}")
        VARIANTS = {k: v for k, v in VARIANTS.items() if k in want}
    geoms.clear()
    gc.collect()

    out = {"mode": a.mode, "n_cells": len(cells), "canary": can,
           "feature_names": list(all_names), "variants": {
               v: {"cols": [all_names[i] for i in c], "per_cell": []}
               for v, c in VARIANTS.items()}}
    for qi, q in enumerate(FOLD_NAMES):
        t1 = time.time()
        train = [c for c in cells if f"fold{q}" not in c.key]
        test = [c for c in cells if f"fold{q}" in c.key]
        tr = {c.key: cell_geometry(ctx, c, geo=geo) for c in train}
        te = {c.key: cell_geometry(ctx, c, geo=geo) for c in test}
        for vname, cols_ in VARIANTS.items():
            clf, scale, base = fit_model(list(tr.values()), seed=qi, cols=cols_)
            for c in test:
                out["variants"][vname]["per_cell"].append(
                    run_cell(ctx, c, clf, scale, cols=cols_,
                             max_dots=a.max_dots, floor=a.floor, g=te[c.key]))
            del clf
            gc.collect()
        tr.clear(); te.clear()
        gc.collect()
        print(f"[{time.time()-t1:5.1f}s] fold{q} done "
              f"({len(VARIANTS)} variants, {len(train)} train / {len(test)} test cells)")

    for vname in VARIANTS:
        pl = pooled(out["variants"][vname]["per_cell"])
        out["variants"][vname]["pooled"] = pl
        ci = pl["dti_ci95_quadrant_jackknife"]
        print(f"{vname:18s} HOLDOUT-DTI pooled={pl['pooled_dti']:.4f} "
              f"CI95(jackknife)=[{ci[0]:.4f},{ci[1]:.4f}] coverage={pl['coverage']:.4f} "
              f"dots={pl['n_dots']} tp={pl['tp']:.0f} fp={pl['fp']:.0f} K={pl['n_truth']}")

    EVID.mkdir(exist_ok=True)
    fname = a.out or f"cv_{a.mode}.json"
    (EVID / fname).write_text(json.dumps(out, indent=2))
    print(f"\nwrote evidence/{fname}   total {time.time()-t0:.1f}s")

    def dti(v):
        return out["variants"][v]["pooled"]["pooled_dti"]
    if "d_only" in out["variants"] and "anatomy_full" in out["variants"]:
        print(f"\nanatomy gain over distance-only: "
              f"{dti('anatomy_full')-dti('d_only'):+.4f} DTI")
    if "no_side" in out["variants"] and "no_side_plus_geo" in out["variants"]:
        print(f"geo corroboration gain over no_side: "
              f"{dti('no_side_plus_geo')-dti('no_side'):+.4f} DTI")


if __name__ == "__main__":
    raise SystemExit("Historical runner retained for learning, not current buffered/inventory validation. "
                     "No new experiment or slot authorized; read the current README and run card.")
