#!/usr/bin/env python3
"""Choose the final emitter from the cached credit field and write the release raster.

The *budget* is taken from OWNER-REPORTED live evidence (the three ~38-40k dot
artifacts the owner scored 0.2778 / 0.2750 / 0.2376) rather than from our own
holdout, whose truth population is 3x smaller and not calibrated to a live score.
The *spacing* is taken from our own holdout sweep (2.5-3.0 px was optimal in both
folds).  Both choices are recorded in the run card.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from gems59 import emit, grid, holdout, metric, pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", default="work/delta_h59a.npy")
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--nms-half", type=int, default=None)
    ap.add_argument("--flank-px", type=int, default=2)
    ap.add_argument("--tag", default="h59a")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fp = grid.read_footprint()
    cat = grid.read_labels() & fp
    sup = holdout.emission_support(fp, cat, flank_px=a.flank_px)
    delta = np.load(ROOT / a.delta)
    print(f"support {sup.sum():,}  delta max {delta[sup].max():.4f} mean {delta[sup].mean():.4f}")
    sep = np.full(delta.shape, a.spacing, np.float32)
    lo, hi = 1e-6, float(delta[sup].max()) + 1e-6
    best = None
    for _ in range(30):
        mid = 0.5*(lo+hi)
        d = emit.place_dots(delta, sup, mid, spacing_map=sep, max_dots=400000, nms_half=a.nms_half)
        n = int(d.sum())
        if best is None or abs(n - a.budget) < abs(best[1] - a.budget):
            best = (d.copy(), n, mid)
        if n > a.budget: lo = mid
        else: hi = mid
        if abs(hi-lo) < 1e-8: break
    dots, n, tau = best
    print(f"chosen tau={tau:.6g} spacing={a.spacing} nms_half={a.nms_half} dots={n}")
    # separation audit
    lab, nl = ndi.label(dots, np.ones((3,3),int))
    print(f"connected components of the dot mask: {nl} (dots {n})")
    # exact minimum-separation audit on a subsample of dots
    rr, cc = np.nonzero(dots)
    step = max(1, len(rr)//1500)
    rr, cc = rr[::step], cc[::step]
    dmin = []
    for r, c in zip(rr, cc):
        r0, c0 = max(0, r-8), max(0, c-8)
        sub = dots[r0:r+9, c0:c+9].copy()
        sub[r-r0, c-c0] = False                     # drop the dot itself
        ys, xs = np.nonzero(sub)
        if ys.size:
            dmin.append(float(np.hypot(ys-(r-r0), xs-(c-c0)).min()))
    print(f"min pairwise separation over {len(dmin)} sampled dots: min={min(dmin):.2f} px  p1={np.percentile(dmin,1):.2f} px")
    out = Path(a.out or (ROOT / f"work/dots_final_{a.tag}_s{a.spacing}_n{n}.npy"))
    np.save(out, dots)
    info = {"dots_file": str(out), "n_dots": n, "spacing": a.spacing, "nms_half": a.nms_half, "tau": float(tau),
            "budget": a.budget, "flank_px": a.flank_px, "support_px": int(sup.sum()),
            "delta_max": float(delta[sup].max()), "delta_mean": float(delta[sup].mean()),
            "on_catalogue": int((dots & cat).sum()),
            "within_2px_of_catalogue": int((dots & ndi.binary_dilation(cat, np.ones((3,3),int), 2)).sum()),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(info, indent=1))
    (ROOT/"evidence"/f"emission_final_{a.tag}.json").write_text(json.dumps(info, indent=1)+"\n")
if __name__ == "__main__":
    main()
