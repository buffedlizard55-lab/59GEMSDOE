#!/usr/bin/env python3
"""Regenerate ``evidence/registry_budget.json`` from the in-repo registry.

Two kinds of content, both labelled:

1. **ORGANIZER-CONFIRMED live evidence** — for every registry raster that
   carries an owner-reported live score, the dot count (positive finite
   pixels, counted here from the raster bytes) and the live score (copied
   from a submission-page receipt by the task owner).  From these:
   Spearman(dot count, live score) and the dot-count bands.  This is the
   evidence behind the 40,000-dot live budget cap (IR-57-BUDGET-01).

2. **Synthetic illustration** — the DTI-vs-coverage curve, measured with the
   EXACT metric (:func:`gems59.metric.dti_binary`) on synthetic fields:
   ``N`` cells, ``K`` truth cells uniform at random; ``m = round(c*K)`` dots
   on distinct truth cells (guaranteeing coverage ``c``) plus ``n - m``
   uniform-random dots, ``n = round(r*K)``.  Averaged over ``TRIALS`` seeded
   trials per (coverage, n/K) cell.  ``N`` and ``K`` are the session-1
   illustration parameters (a ~2.56 M-cell field at the ~0.00221 base rate);
   they are stated, not measured.  This curve is an illustration of the
   metric's arithmetic, NOT a score projection.

The Spearman in (1) is computed over registry rasters with a live score only;
this repository's own earlier builds carry no live score and are excluded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.metric import dti_binary  # noqa: E402

OUT = ROOT / "evidence"

# session-1 synthetic-illustration parameters (stated, not measured)
GRID_PX = 2_560_000
N_TRUTH_CELLS = 5_660
COVERAGES = (0.1, 0.2, 0.2778, 0.3, 0.4, 0.5, 0.7, 1.0)
N_OVER_K = (0.5, 1.0, 2.0, 4.0, 6.0)
TRIALS = 40
SEED = 20261009


def main() -> None:
    t0 = time.time()
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    rows = []
    for e in idx:
        score = e.get("owner_reported_score")
        if score is None:
            continue                      # no live score -> excluded from the fit
        p = ROOT / e["file"]
        with rasterio.open(p) as ds:
            a = ds.read(1)
        n_dots = int((np.nan_to_num(a, nan=0.0) > 0).sum())
        rows.append({"submission": e["submission"], "n_dots": n_dots,
                     "live_score": float(score), "repo": e["repo"]})
    rows.sort(key=lambda r: r["n_dots"])
    rho, pval = spearmanr([r["n_dots"] for r in rows], [r["live_score"] for r in rows])

    def band(lo, hi):
        sel = [r for r in rows if lo <= r["n_dots"] < hi]
        return {"n": len(sel),
                "mean_live": float(np.mean([r["live_score"] for r in sel])) if sel else None,
                "best": float(np.max([r["live_score"] for r in sel])) if sel else None}

    # exact expected kernel credit of a random pixel: sum_k * K / N
    from gems59.metric import OFF_K
    sum_kernel = float(np.sum(OFF_K))
    random_hit_prob = sum_kernel * N_TRUTH_CELLS / GRID_PX

    shape = (1600, GRID_PX // 1600)
    valid = np.ones(shape, bool)
    curve = []
    for c in COVERAGES:
        for r in N_OVER_K:
            n = int(round(r * N_TRUTH_CELLS))
            m = min(n, int(round(c * N_TRUTH_CELLS)))
            vals = []
            for t in range(TRIALS):
                rng = np.random.default_rng(SEED + 977 * t + int(1000 * c) + int(10 * r))
                truth = np.zeros(GRID_PX, bool)
                truth[rng.choice(GRID_PX, N_TRUTH_CELLS, replace=False)] = True
                dots = np.zeros(GRID_PX, bool)
                ti = np.flatnonzero(truth)
                dots[rng.choice(ti, m, replace=False)] = True
                if n - m > 0:
                    dots[rng.choice(GRID_PX, n - m, replace=False)] = True
                res = dti_binary(dots.reshape(shape), truth.reshape(shape), valid)
                vals.append(res["dti"])
                del truth, dots, ti
            curve.append({"coverage": c, "n_over_K": r,
                          "dti": float(np.mean(vals)),
                          "dti_sd": float(np.std(vals)), "trials": TRIALS})
        print(f"[budget] coverage {c}: done ({time.time()-t0:.0f}s)", flush=True)

    # a perfect prediction scores exactly 1.0 (T = M = n = K)
    rng = np.random.default_rng(SEED)
    truth = np.zeros(GRID_PX, bool)
    truth[rng.choice(GRID_PX, N_TRUTH_CELLS, replace=False)] = True
    perfect = dti_binary(truth.reshape(shape), truth.reshape(shape), valid)

    payload = {
        "n_rasters": len(rows),
        "spearman_dots_vs_live": float(rho),
        "p_value": float(pval),
        "rows": rows,
        "band_lt_50k": band(0, 50_000),
        "band_ge_50k": band(50_000, 10**12),
        "band_35k_46k": band(35_000, 46_000),
        "grid_px": GRID_PX,
        "n_truth_cells": N_TRUTH_CELLS,
        "base_rate": N_TRUTH_CELLS / GRID_PX,
        "sum_kernel": sum_kernel,
        "random_hit_prob": float(random_hit_prob),
        "dti_vs_coverage_curve": curve,
        "perfect_prediction_dti": float(perfect["dti"]),
        "curve_method": ("exact metric on synthetic fields: m=round(c*K) dots on distinct "
                         "random truth cells + n-m uniform-random dots, n=round(r*K); "
                         f"{TRIALS} seeded trials per cell; N and K are stated illustration "
                         "parameters, not measurements"),
        "evidence_class": ("rows: ORGANIZER-CONFIRMED live scores vs counted dots | "
                           "curve: synthetic illustration of the metric arithmetic, "
                           "never a score projection"),
        "runtime_s": time.time() - t0,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "registry_budget.json").write_text(json.dumps(payload, indent=1))
    print(f"[budget] Spearman(dots, live) = {rho:+.4f} (p = {pval:.5f}) over {len(rows)} "
          f"rasters; perfect-prediction DTI = {perfect['dti']:.6f}")
    print("[budget] wrote evidence/registry_budget.json", flush=True)


if __name__ == "__main__":
    main()
