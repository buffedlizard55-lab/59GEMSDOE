#!/usr/bin/env python3
"""HOLDOUT-DTI of the shipped configuration at the SHIPPED dot density.

Follows scripts/run_sense_experiment.py (IR-57-SHIP-01 / IR-57-CAP-01): mode
``all``, leave-one-quadrant-out over the 4 quadrants, pooled DTI
alpha=0.2 beta=0.8 R=3 px, quadrant-jackknife CI, per-cell cap = 10,000
(= 40,000 live cap / 4 quadrant cells).  Runs on the CORRECTED strike frame
(IR-57-STRIKE-01 fix is in gems59.anatomy).  Variants are the corrected-frame
candidates that were measured at run_cv density in evidence/cv_all.json:

* ``d_only``       distance to nearest visible fault
* ``d_perp_par``   the shipped lean offset frame (d, d_perp, d_par_abs)
* ``no_side``      the 8-feature anatomy set (side dropped)

Every number is a HOLDOUT-DTI instrument reading.  No live score is projected.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid                                             # noqa: E402
from gems59.anatomy import FEATURES                                      # noqa: E402
from gems59.fitting import (canary, cell_geometry, fit_model, pooled,    # noqa: E402
                            run_cell)
from gems59.holdout import FOLD_NAMES, build_holdout                     # noqa: E402

EVID = ROOT / "evidence"
PER_CELL_CAP = 10_000          # shipped density: 40,000 live cap / 4 quadrants
FLOOR = 0.015
IDX = {n: i for i, n in enumerate(FEATURES)}
VARIANTS = {
    "d_only": [IDX["d"]],
    "d_perp_par": [IDX["d"], IDX["d_perp"], IDX["d_par_abs"]],
    "no_side": [IDX[n] for n in FEATURES if n != "side"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=PER_CELL_CAP)
    ap.add_argument("--out", type=Path, default=EVID / "cv_shipped_density.json")
    a = ap.parse_args()

    t0 = time.time()
    grid = load_grid()
    ctx = build_holdout(grid)
    cells = ctx.cells_of("all")
    print(f"[{time.time()-t0:6.1f}s] holdout ready: {len(cells)} cells (mode=all)")

    can_geoms = [cell_geometry(ctx, c) for c in cells[:4]]
    can = canary(can_geoms)
    can_geoms.clear(); gc.collect()
    print(f"\n[{time.time()-t0:6.1f}s] LEAKAGE CANARY (discriminative AUC)")
    for name, r in can.items():
        flag = "  <== LEAKAGE FLAG" if r["leakage_flag"] else ""
        print(f"  {name:12s} disc_mean={r['discriminative_auc_mean']:.4f} "
              f"disc_max={r['discriminative_auc_max']:.4f}{flag}", flush=True)

    results = {v: [] for v in VARIANTS}
    per_quadrant = {v: {} for v in VARIANTS}
    for qi, q in enumerate(FOLD_NAMES):
        test = [c for c in cells if c.key.split("_")[1] == f"fold{q}"]
        train = [c for c in cells if c.key.split("_")[1] != f"fold{q}"]
        g_train = [cell_geometry(ctx, c) for c in train]
        g_test = {c.key: cell_geometry(ctx, c) for c in test}
        for vname, cols in VARIANTS.items():
            clf, scale, _ = fit_model(g_train, seed=qi, cols=cols)
            rq = []
            for c in test:
                rq.append(run_cell(ctx, c, clf, scale, cols=cols, max_dots=a.cap,
                                   floor=FLOOR, g=g_test[c.key]))
            results[vname].extend(rq)
            per_quadrant[vname][q] = pooled(rq)
            print(f"[{time.time()-t0:6.1f}s] fold {q} {vname:12s} "
                  f"DTI={per_quadrant[vname][q]['pooled_dti']:.4f} "
                  f"dots={per_quadrant[vname][q]['n_dots']}", flush=True)
            del clf
        del g_train, g_test
        gc.collect()

    out = {
        "evidence_class": "HOLDOUT-DTI (local instrument reading, NOT a projected live score)",
        "evaluator": "gems52-pooled-hide-v1 alpha=0.2 beta=0.8 R=3px, leave-one-quadrant-out, mode=all",
        "per_cell_cap": a.cap,
        "floor": FLOOR,
        "strike_frame": "corrected (IR-57-STRIKE-01 fix active)",
        "variants": {},
    }
    for vname in VARIANTS:
        p = pooled(results[vname])
        out["variants"][vname] = {
            "cols": [FEATURES[i] for i in VARIANTS[vname]],
            "per_quadrant": per_quadrant[vname],
            "pooled": p,
        }
        print(f"\n== {vname} =="
              f"\n   pooled_dti {p['pooled_dti']:.4f}"
              f"\n   ci95 jackknife {p['dti_ci95_quadrant_jackknife']}"
              f"\n   coverage {p['coverage']:.4f}"
              f"\n   dots {p['n_dots']}")
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    raise SystemExit("Historical runner retained for learning, not current buffered/inventory validation. "
                     "No new experiment or slot authorized; read the current README and run card.")
