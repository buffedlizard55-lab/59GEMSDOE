#!/usr/bin/env python3
"""Build the release candidate: fit the leak-free intensity, tune emission on the
holdout, then emit the deployment raster.

Channel set
-----------
``dist_visible_fault``, ``visible_fault_dens_5k`` and ``visible_fault_dens_20k``
are **excluded**.  They are exactly zero (or maximal) on the catalogue pixels that
form the training presence set, so a model that sees them separates the training
data perfectly and then scores AUC 0.500 on withheld segments -- measured, see
``evidence/leakage_channel_ablation.json``.  They are leakage, not signal.

Arms
----
* ``with_trace`` - every remaining channel (includes ``trace_density``, a smooth
  30 px kernel density of mapped-trace centroids);
* ``no_trace``   - the same minus ``trace_density``, to show how much of the
  measured skill comes from the geophysics rather than from where mapping has
  already happened.

Emission
--------
Swept on the holdout (spacing x break-even bar), the DTI-optimal configuration is
carried to the deployment field by matching the *scaled dot budget*
N_deploy = N*_holdout * G_assumed / G_holdout, because the organizer's metric is
scale-free in |G| only through the ratio N/|G|.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59 import emit, features, grid, holdout, intensity, metric, pipeline  # noqa: E402
from gems59 import submission  # noqa: E402

WORK = ROOT / "work"
EVIDENCE = ROOT / "evidence"

LEAKY = ("dist_visible_fault", "visible_fault_dens_5k", "visible_fault_dens_20k")
CH_WITH_TRACE = [c for c in features.CHANNELS if c not in LEAKY]
CH_NO_TRACE = [c for c in CH_WITH_TRACE if c != "trace_density"]

SPACING_GRID = (2.5, 3.0, 4.0, 5.0)
TAU_GRID = (0.03, 0.045, 0.056, 0.07, 0.09, 0.12, 0.16, 0.22)


def fit_field(stack, presence, support, channels, spec, rng):
    idx = [features.CHANNELS.index(c) for c in channels]
    X, y, _ = intensity.sample_design(stack, presence, support, idx, spec, rng)
    m = HistGradientBoostingClassifier(
        max_iter=spec.max_iter, learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes, min_samples_leaf=spec.min_samples_leaf,
        class_weight="balanced", random_state=spec.seed).fit(X, y)
    del X, y
    H, W = support.shape
    out = np.zeros((H, W), np.float32)
    for r0 in range(0, H, 64):
        r1 = min(H, r0 + 64)
        blk = np.stack([stack[i, r0:r1] for i in idx])
        out[r0:r1] = m.predict_proba(blk.reshape(len(idx), -1).T)[:, 1].reshape(r1 - r0, W)
    out[~support] = 0.0
    lam = np.clip(out, 1e-9, 1 - 1e-9)
    return (lam / (1 - lam)).astype(np.float32)      # odds -> intensity


def sweep(delta, support, truth, g_ratio):
    """Sweep spacing x break-even bar; return the grid and the DTI-optimal cell."""
    grid_rows = []
    best = {"dti": -1.0}
    for s in SPACING_GRID:
        for tau in TAU_GRID:
            sep = np.full(delta.shape, s, np.float32)
            dots = emit.place_dots(delta, support, tau, spacing_map=sep, max_dots=200_000)
            n = int(dots.sum())
            if n == 0:
                continue
            d = metric.dti(dots.astype(np.float32), truth)
            row = {"spacing": s, "tau": tau, "n_dots": n, "dti": round(float(d), 5),
                   "n_dots_scaled": int(round(n * g_ratio))}
            grid_rows.append(row)
            if d > best["dti"]:
                best = row
    return grid_rows, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h59a")
    ap.add_argument("--max-iter", type=int, default=140)
    ap.add_argument("--n-background", type=int, default=200_000)
    ap.add_argument("--g-assumed", type=float, default=40_000.0)
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--flank-px", type=int, default=2)
    ap.add_argument("--arm", default="with_trace")
    ap.add_argument("--skip-final", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    stack, fp = features.build_stack()[0], grid.read_footprint()
    cat = grid.read_labels() & fp
    spec = intensity.PPSpec(max_iter=a.max_iter, n_background=a.n_background)
    rng = np.random.default_rng(spec.seed)
    channels = CH_WITH_TRACE if a.arm == "with_trace" else CH_NO_TRACE
    lab, nseg = holdout.segments(cat)
    blocks = holdout.assign_blocks_random(lab, 4, seed=7)
    log = {"tag": a.tag, "arm": a.arm, "channels": channels, "n_channels": len(channels),
           "excluded_as_leakage": list(LEAKY), "spec": spec.__dict__,
           "g_assumed": a.g_assumed, "flank_px": a.flank_px, "folds": []}

    # ---------------- holdout: tune the emission -------------------------
    grids, bests = [], []
    for k in [int(x) for x in a.folds.split(",")]:
        w, vis, ta = holdout.fold_masks(lab, blocks, k, fp, buffer_px=3)
        features.set_visible_channels(stack, cat & ~w, fp)
        sup = holdout.emission_support(fp, cat & ~w, flank_px=a.flank_px)
        presence = cat & ~w & ta
        lam = fit_field(stack, presence, fp, channels, spec, rng)
        # holdout truth size is |withheld|; deployment is assumed g_assumed
        g_hold = float((w & sup).sum())
        delta = pipeline.calibrate_delta(lam, sup, g_hold)
        rows, best = sweep(delta, sup, w & sup, 1.0)
        grids.append(rows)
        bests.append({**best, "fold": k, "g_holdout": g_hold})
        print(f"[holdout {k}] truth={int(g_hold)} best spacing={best['spacing']} "
              f"tau={best['tau']} n={best['n_dots']} DTI={best['dti']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        log["folds"].append({"fold": k, "g_holdout": g_hold, "best": best, "grid": rows})

    mean_dti = float(np.mean([b["dti"] for b in bests]))
    mean_n = float(np.mean([b["n_dots"] for b in bests]))
    mean_g = float(np.mean([b["g_holdout"] for b in bests]))
    spacing_star = float(np.median([b["spacing"] for b in bests]))
    g_ratio = a.g_assumed / mean_g
    n_target = int(round(mean_n * g_ratio))
    log["holdout"] = {"mean_best_dti": mean_dti, "mean_best_n_dots": mean_n,
                      "mean_g_holdout": mean_g, "spacing_star": spacing_star,
                      "g_ratio": g_ratio, "n_target_deployment": n_target}
    print(json.dumps(log["holdout"], indent=1), flush=True)
    (EVIDENCE).mkdir(exist_ok=True)
    (EVIDENCE / f"emission_sweep_{a.tag}_{a.arm}.json").write_text(json.dumps(log, indent=1) + "\n")

    if a.skip_final:
        return 0

    # ---------------- deployment: full catalogue, tuned emission ----------
    features.set_visible_channels(stack, cat, fp)
    sup = holdout.emission_support(fp, cat, flank_px=a.flank_px)
    lam = fit_field(stack, cat & fp, fp, channels, spec, rng)
    delta = pipeline.calibrate_delta(lam, sup, a.g_assumed)

    # binary-search the bar that lands on the scaled budget
    lo, hi = 1e-4, float(delta.max()) + 1e-6
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        d = emit.place_dots(delta, sup, mid, spacing_map=np.full(delta.shape, spacing_star, np.float32),
                            max_dots=400_000)
        if int(d.sum()) > n_target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-7:
            break
    tau_star = 0.5 * (lo + hi)
    dots = emit.place_dots(delta, sup, tau_star,
                           spacing_map=np.full(delta.shape, spacing_star, np.float32),
                           max_dots=400_000)
    n_dots = int(dots.sum())
    np.save(WORK / f"dots_{a.tag}.npy", dots)
    np.save(WORK / f"delta_{a.tag}.npy", delta)
    log["deployment"] = {"tau_star": float(tau_star), "spacing": spacing_star,
                         "n_dots": n_dots, "support_px": int(sup.sum()),
                         "dti_target_implied": float(metric.breakeven_delta(mean_dti))}
    print(json.dumps(log["deployment"], indent=1), flush=True)
    (EVIDENCE / f"build_{a.tag}_{a.arm}.json").write_text(json.dumps(log, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
