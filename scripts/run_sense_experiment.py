#!/usr/bin/env python3
"""Experiment (session 2026-10-09): recorded sense of slip + the SHIPPED dot density.

Two questions, one run, same folds:

Q1  (E-A) What is the HOLDOUT-DTI of the configuration that actually shipped?
    The shipped file has 35,341 dots over the whole footprint, i.e. ~10,000 per
    fold cell (4 quadrants).  Every earlier holdout number was measured at the
    per-cell cap of run_cv.py (120,000), which produced ~68,000 dots per draw --
    roughly twice the shipped density.  Here the per-cell cap is set to the shipped
    share (10,000) and the number is measured, not assumed.

Q2  (E-B) Does recorded sense of slip add holdout DTI at that density?
    ``no_side`` (the shipped 8 features) vs ``no_side+sense`` (plus
    ``sense_sgn`` and ``sense_side`` from the INGENIOUS ``sense`` column).

Protocol: mode ``all``, leave-one-quadrant-out (4 folds), both draws per quadrant
in the test set, the leakage canary on every feature (incl. the sense features),
pooled DTI alpha=0.2 beta=0.8 R=3 px, quadrant-jackknife CI.  Every number is a
HOLDOUT-DTI instrument reading.  No live score is projected.

Paired comparison: per-quadrant DTI for both variants, the mean difference and its
spread over the four quadrants.  The sense claim is only promoted if the paired
difference is positive in >= 3 of 4 quadrants AND the leakage canary is clean.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid                                             # noqa: E402
from gems59.anatomy import FEATURES, SENSE_FEATURES                      # noqa: E402
from gems59.faultzone import trace_sense_raster                          # noqa: E402
from gems59.fitting import (canary, cell_geometry, fit_model, pooled,    # noqa: E402
                            run_cell)
from gems59.holdout import FOLD_NAMES, build_holdout                     # noqa: E402

EVID = ROOT / "evidence"
CSV = ROOT / "data" / "external" / "trace_segments_utm11.csv"
EVALUATOR_VERSION = "gems59-loqo-pooled-v1"
EVALUATOR_SOURCES = (
    "scripts/run_sense_experiment.py",
    "src/gems59/fitting.py",
    "src/gems59/metric.py",
    "src/gems59/holdout.py",
    "src/gems59/anatomy.py",
    "src/gems59/network.py",
    "src/gems59/emit.py",
    "src/gems59/faultzone.py",
    "src/gems59/grid.py",
)
EVALUATOR_INPUTS = (
    "data/bridge/existing_faults.tif",
    "data/bridge/sample_submission.tif",
    "data/external/trace_segments_utm11.csv",
)
PER_CELL_CAP = 10_000          # shipped density: 40,000 live cap / 4 quadrants
FLOOR = 0.015
IDX = {n: i for i, n in enumerate(FEATURES + SENSE_FEATURES)}
NO_SIDE = [IDX[n] for n in FEATURES if n != "side"]
VARIANTS = {
    "no_side": NO_SIDE,
    "no_side_plus_sense": NO_SIDE + [IDX["sense_sgn"], IDX["sense_side"]],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluator_provenance() -> dict:
    """Pin the implementation and data inputs for runs produced from now on."""
    return {
        "version": EVALUATOR_VERSION,
        "implementation": "gems59.fitting.pooled + gems59.metric.dti_binary; "
                          "visible-only fold geometry and greedy allocation",
        "implementation_sha256": {
            name: _sha256(ROOT / name) for name in EVALUATOR_SOURCES
        },
        "input_sha256": {
            name: _sha256(ROOT / name) for name in EVALUATOR_INPUTS
        },
        "versioning_note": "Applies to this run only; does not attest historical evidence lacking these hashes.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=PER_CELL_CAP)
    ap.add_argument("--out", type=Path, default=EVID / "exp_sense_loqo_all.json")
    a = ap.parse_args()

    t0 = time.time()
    grid = load_grid()
    sense_src = trace_sense_raster(CSV, grid.shape, grid.transform)
    print(f"[{time.time()-t0:6.1f}s] sense raster: N={int((sense_src==1).sum())} "
          f"RL={int((sense_src==2).sum())} LL={int((sense_src==3).sum())} px")
    ctx = build_holdout(grid)
    cells = ctx.cells_of("all")
    print(f"[{time.time()-t0:6.1f}s] holdout ready: {len(cells)} cells (mode=all)")

    # ---- leakage canary: every feature alone, incl. the two sense features -------
    can_geoms = [cell_geometry(ctx, c, sense_src) for c in cells[:4]]
    can = canary(can_geoms)
    can_geoms.clear(); gc.collect()
    print(f"\n[{time.time()-t0:6.1f}s] LEAKAGE CANARY (discriminative AUC, 4 folds)")
    for name, r in can.items():
        flag = "  <== LEAKAGE FLAG" if r["leakage_flag"] else ""
        print(f"  {name:12s} disc_mean={r['discriminative_auc_mean']:.4f} "
              f"disc_max={r['discriminative_auc_max']:.4f}{flag}")

    # ---- leave-one-quadrant-out ----------------------------------------------------
    results = {v: [] for v in VARIANTS}
    per_quadrant = {v: {} for v in VARIANTS}
    for qi, q in enumerate(FOLD_NAMES):
        test = [c for c in cells if c.key.split("_")[1] == f"fold{q}"]
        train = [c for c in cells if c.key.split("_")[1] != f"fold{q}"]
        g_train = [cell_geometry(ctx, c, sense_src) for c in train]
        g_test = {c.key: cell_geometry(ctx, c, sense_src) for c in test}
        for vname, cols in VARIANTS.items():
            clf, scale, _ = fit_model(g_train, seed=qi, cols=cols)
            rq = []
            for c in test:
                rq.append(run_cell(ctx, c, clf, scale, cols=cols, max_dots=a.cap,
                                   floor=FLOOR, g=g_test[c.key]))
            results[vname].extend(rq)
            per_quadrant[vname][q] = pooled(rq)
            print(f"[{time.time()-t0:6.1f}s] fold {q} {vname:20s} "
                  f"DTI={per_quadrant[vname][q]['pooled_dti']:.4f} "
                  f"dots={per_quadrant[vname][q]['n_dots']}", flush=True)
            del clf
        del g_train, g_test
        gc.collect()

    out = {
        "evidence_class": "HOLDOUT-DTI (local instrument reading, NOT a projected live score)",
        "evaluator": evaluator_provenance(),
        "evaluator_description": "gems59 pooled DTI alpha=0.2 beta=0.8 R=3px, leave-one-quadrant-out, mode=all",
        "per_cell_cap": a.cap,
        "shipped_density_note": "35,341 dots shipped over the whole footprint (~10,000 per quadrant cell)",
        "floor": FLOOR,
        "features_no_side": [FEATURES[i] for i in NO_SIDE],
        "canary": can,
        "variants": {},
        "paired_no_side_plus_sense_minus_no_side": None,
        "runtime_s": round(time.time() - t0, 1),
    }
    for vname in VARIANTS:
        pl = pooled(results[vname])
        out["variants"][vname] = {
            "pooled": pl,
            "per_quadrant_dti": {q: per_quadrant[vname][q]["pooled_dti"] for q in FOLD_NAMES},
            "withheld_positive_pixels": pl["n_truth"],
            "label": "HOLDOUT-DTI",
        }
        ci = pl["dti_ci95_quadrant_jackknife"]
        print(f"\n{vname:20s} HOLDOUT-DTI pooled={pl['pooled_dti']:.4f} "
              f"CI95(quadrant jackknife)=[{ci[0]:.4f},{ci[1]:.4f}] "
              f"coverage={pl['coverage']:.4f} dots={pl['n_dots']} K={pl['n_truth']}")
    diffs = [per_quadrant["no_side_plus_sense"][q]["pooled_dti"] - per_quadrant["no_side"][q]["pooled_dti"]
             for q in FOLD_NAMES]
    pos = int(sum(d > 0 for d in diffs))
    out["paired_no_side_plus_sense_minus_no_side"] = {
        "per_quadrant_diff": dict(zip(FOLD_NAMES, diffs)),
        "mean_diff": float(np.mean(diffs)), "sd_diff": float(np.std(diffs, ddof=1)),
        "n_quadrants_positive": pos,
        "canary_clean": not any(r["leakage_flag"] for r in can.values()),
        "promote_rule": "positive in >=3 of 4 quadrants AND canary clean",
        "promote": bool(pos >= 3 and not any(r["leakage_flag"] for r in can.values())),
    }
    print(f"\npaired diff (sense - no_sense) per quadrant: "
          + ", ".join(f"{q}={d:+.4f}" for q, d in zip(FOLD_NAMES, diffs))
          + f"  mean={np.mean(diffs):+.4f}  positive={pos}/4")
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit("Retired legacy/unbuffered runner: not current validation. "
                     "Reproduce the declared buffered experiment with scripts/run_orientation_experiments.py; "
                     "no extra experiment or slot is authorized by this command.")
