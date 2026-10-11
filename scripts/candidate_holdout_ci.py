#!/usr/bin/env python3
"""HOLDOUT-DTI with a 95% spatial-cluster CI for ONE candidate raster.

Why this exists
---------------
`scripts/exp_live_calibration.py` scores every file in a directory and reports the
point estimate only, and `scripts/run_holdout.py` is a five-arm experiment with
its own emission budgets.  The protocol requires every holdout number to carry a
95% CI, and the run card needs one for the single file that is being shipped.

Method (all shared tools, no private metric)
--------------------------------------------
* the shared hide-and-recover draw `gems59.holdout.buffered_component_draw`
  (seed 20, the same draw every other arm in this repository uses);
* the shared evaluator `gems59.evaluate_holdout.evaluate`, called once per fold
  on full-grid masks, so every fold's block terms sit on the same global 200 px
  (20 km) block grid;
* per-block terms from the same physical block are merged across folds BEFORE
  resampling (template rule), then `gems59.evaluate_holdout.pooled_summary`
  draws 1000 percentile resamples of whole physical blocks.

Output is labelled HOLDOUT-DTI.  It is a local instrument, never an organizer
score, and it is *not* a promotion gate for this session: the instrument is
measured to be rank-inverted against the public leaderboard
(`evidence/h59j_live_calibration.json`, rho = -0.437, p = 0.033; IR-59J-01).

Usage: python scripts/candidate_holdout_ci.py <file.tif> [--out evidence/x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import evaluate_holdout as EH            # noqa: E402
from gems59.grid import load_grid                    # noqa: E402
from gems59.holdout import buffered_component_draw   # noqa: E402

BLOCK_SIDE = 200          # 20 km physical clusters, the shared bootstrap unit
DRAWS = 1000
BOOTSTRAP_SEED = 520817   # SEED + 7, the seed run_holdout.py uses for pooling


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--out", default="evidence/h59j_candidate_ci.json")
    a = ap.parse_args()

    t0 = time.monotonic()
    grid = load_grid()
    fp = grid.footprint
    folds, draw_aux = buffered_component_draw(grid, seed=a.seed)
    withheld = int(sum(f["truth"].sum() for f in folds))
    print(f"[holdout] {len(folds)} folds, withheld={withheld} px "
          f"({time.monotonic()-t0:.0f}s)", flush=True)

    path = ROOT / a.file
    with rasterio.open(path) as src:
        pred = src.read(1).astype(np.float32)
    pred = np.where(np.isfinite(pred), pred, 0.0)

    terms = None
    fold_rows = []
    for fold in folds:
        result, t = EH.evaluate(pred, fold, fp, block_side=BLOCK_SIDE)
        terms = t if terms is None else terms + t
        fold_rows.append(dict(fold=int(fold["fold"]), name=fold.get("name"),
                              dti=float(result["dti"]), n_truth=int(result["n_truth"]),
                              tp=float(result["tp"]), fp=float(result["fp"]),
                              fn=float(result["fn"])))
        print(f"    fold {fold['fold']}: dti {result['dti']:.6f} "
              f"(truth {int(result['n_truth'])})", flush=True)
        del result, t

    summary = EH.pooled_summary({"candidate": terms}, draws=DRAWS,
                                seed=BOOTSTRAP_SEED, candidate="candidate")
    score = dict(summary["scores"]["candidate"])
    score["evidence_class"] = "HOLDOUT-DTI"
    score["holdout_split"] = "buffered-whole-components-loqo-v2"
    score["seed"] = a.seed
    score["per_fold"] = fold_rows
    score["note"] = ("local instrument with a conditional spatial-cluster CI; "
                     "not an organizer score and not a promotion gate (IR-59J-01)")

    payload = dict(
        evidence_class="HOLDOUT-DTI",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        candidate=a.file,
        evaluator_version=EH.VERSION,
        evaluator_hashes=EH.implementation_hashes(),
        draw_aux=dict(kind=type(draw_aux).__name__,
                      shape=[int(v) for v in np.shape(draw_aux)],
                      dtype=str(np.asarray(draw_aux).dtype),
                      nonzero_cells=int(np.count_nonzero(draw_aux))),
        draw_per_fold=[dict(fold=int(f["fold"]), name=f.get("name"),
                            n_held=int(f.get("n_held", 0)),
                            withheld_positive_pixels=int(f["truth"].sum()))
                       for f in folds],
        withheld_positive_pixels=withheld,
        alpha=0.2, beta=0.8, triangular_radius_m=300.0,
        spatial_cluster_side_px=BLOCK_SIDE,
        bootstrap=summary["bootstrap"],
        score=score,
        labels=dict(holdout_dti="HOLDOUT-DTI (local instrument) -- NOT an organizer score"),
    )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: score[k] for k in ("dti", "ci95", "withheld_positive_pixels")}, indent=1))
    print(f"[done] {(time.monotonic()-t0)/60:.1f} min -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
