#!/usr/bin/env python3
"""Diagnostic: does the fitted intensity have skill, and how sharp is it?

Reports, for the held-out fault segments, the ranking AUC of each field, the
precision/recall at the top of the ranking, and the concentration of the credit
field Delta (which is what the break-even rule thresholds on).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gems59 import emit, features, grid, holdout, intensity, metric, pipeline

def auc(score, pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    p = score[pos]; p = p[np.isfinite(p)]; n = score[neg]; n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0: return float("nan")
    if p.size > 20000: p = rng.choice(p, 20000, replace=False)
    if n.size > 200000: n = rng.choice(n, 200000, replace=False)
    a = np.concatenate([p, n]); o = np.argsort(a, kind="stable"); s = a[o]
    tied = np.empty(a.size); i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j+1] == s[i]: j += 1
        tied[o[i:j+1]] = 0.5*(i+j)+1.0; i = j+1
    rp = tied[:p.size].sum()
    return float((rp - p.size*(p.size+1)/2.0)/(p.size*n.size))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--n-background", type=int, default=200000)
    ap.add_argument("--target", default="catalogue")
    ap.add_argument("--g-truth", type=float, default=40000.0)
    a = ap.parse_args()
    t0 = time.time()
    stack, fp = features.build_stack()[0], grid.read_footprint()
    cat = grid.read_labels() & fp
    with rasterio.open(grid.DATA / "external/derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = (s.read(1) > 0) & fp
    truth_full = {"catalogue": cat, "offcatalogue": sgmc & ~cat}[a.target]
    lab, n = holdout.segments(truth_full)
    blocks = holdout.assign_blocks_random(lab, a.folds, seed=7)
    withheld, visible, train_area = holdout.fold_masks(lab, blocks, a.fold, fp, buffer_px=3)
    features.set_visible_channels(stack, cat & ~withheld, fp)
    sup = holdout.emission_support(fp, cat & ~withheld, flank_px=2)
    print(f"segments={n} withheld_px={withheld.sum()} visible_px={(cat&~withheld).sum()} sup={sup.sum()}")
    spec = intensity.PPSpec(max_iter=a.max_iter, n_background=a.n_background)
    presence = truth_full & ~withheld & train_area
    t = time.time(); model = intensity.fit_pp(stack, fp, presence, spec=spec, area_mask=train_area)
    print("fit %.0fs  %s" % (time.time()-t, model.meta))
    t = time.time(); lam_new, lam_ref, b = intensity.effort_gap(model, stack, fp, area_mask=train_area)
    print("predict %.0fs" % (time.time()-t))
    ev = sup & ~withheld
    pos = withheld & sup
    neg = ev
    out = {"target": a.target, "fold": a.fold, "g_truth": a.g_truth,
           "segments": int(n), "withheld_px": int(withheld.sum())}
    for nm, f in (("lam_ref", lam_ref), ("lam_new", lam_new), ("b_hat", b)):
        out[nm] = {"auc": round(auc(f, pos, neg), 4),
                   "auc_inv": round(1 - auc(f, pos, neg), 4)}
    for nm, f in (("lam_ref", lam_ref), ("lam_new", lam_new)):
        d = pipeline.calibrate_delta(f, sup, a.g_truth)
        v = d[sup]
        out[nm + "_delta"] = {"max": float(v.max()), "mean": float(v.mean()),
                              "p999": float(np.quantile(v, 0.999)),
                              "p99": float(np.quantile(v, 0.99)),
                              "p95": float(np.quantile(v, 0.95)),
                              "n_ge_tau": {str(tau): int((v >= tau).sum())
                                           for tau in (0.02, 0.04, 0.056, 0.08, 0.12, 0.2)}}
        # top-k precision / recall against withheld truth
        idx = np.argpartition(-v, 50000)[:50000]
        rr, cc = np.nonzero(sup)
        sel = np.zeros(sup.shape, bool); sel[rr[idx], cc[idx]] = True
        out[nm + "_top50k"] = {"precision_3px": round(float((metric.kernel_max(sel.astype(np.float32))[withheld] > 0).mean()), 4),
                               "credit": round(float(metric.dti_terms(sel.astype(np.float32), withheld)["tp"]), 1)}
    out["seconds"] = round(time.time()-t0, 1)
    print(json.dumps(out, indent=1))
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / f"diagnose_{a.target}_{a.fold}.json").write_text(json.dumps(out, indent=1)+"\n")
if __name__ == "__main__":
    main()
