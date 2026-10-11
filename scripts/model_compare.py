#!/usr/bin/env python3
"""Score existing dot sets against the session-4 model surface.

The emission's `surrogate_dti` is computed against the model's own probability
surface, so on its own it is not interpretable.  This script scores every
registry raster that has an OWNER-REPORTED live score on the *same* surface with
the *same* formula, which makes the number comparable: it says how much credit
the session-4 model thinks each of those already-submitted dot fields collects.

DTI here is MODEL-SURROGATE, not HOLDOUT-DTI and not a score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59.grid import load_grid     # noqa: E402
from gems59.metric import OFF_DX, OFF_DY, OFF_K  # noqa: E402

ALPHA, BETA, G = 0.2, 0.8, 14088.747191011289


def truth_credit(p: np.ndarray, dots: np.ndarray) -> float:
    """T = sum over cells of p[cell] * best kernel weight reaching that cell."""
    dy, dx, kw = OFF_DY, OFF_DX, OFF_K
    H, W = p.shape
    best = np.zeros(p.shape, np.float32)
    ys, xs = np.nonzero(dots)
    for ddy, ddx, w in zip(dy, dx, kw):
        ny, nx = ys + ddy, xs + ddx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        np.maximum.at(best, (ny[ok], nx[ok]), np.float32(w))
    return float((p * best).sum())


def main() -> None:
    grid = load_grid()
    p = np.load(ROOT / "out/h57k_surface_f32.npy")
    p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    sub = json.loads((ROOT / "evidence" / "h57k_submission.json").read_text())

    rows = []
    for entry in [
        ("THIS SESSION — h57-k damage zone", sub["file"], None),
        ("h33-h33-2-b2 (0.2778)", "registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif", 0.2778),
        ("h36-1-rung30-blind-r1 (0.2710)", "registry/rasters/GEMSDOE28__h36-1-rung30-blind-r1.tif", 0.2710),
        ("d28-poisson300m-offcat (0.2600)", "registry/rasters/GEMSDOE30__d28-poisson300m-offcat-44090.tif", 0.2600),
        ("h19-5-powerlaw-multiline (0.1922)", "registry/rasters/19GEMSDOE__h19-5-powerlaw-budget-multiline.tif", 0.1922),
        ("r13-lattice-s5 (0.0904)", "out/live_cache/live_00904_r13lattice.tif", 0.0904),
    ]:
        f = ROOT / entry[1]
        if not f.exists():
            continue
        import rasterio
        with rasterio.open(f) as s:
            a = s.read(1)
        dots = np.isfinite(a) & (a > 0)
        n = int(dots.sum())
        T = truth_credit(p, dots)
        dti = T / (ALPHA * n + BETA * G)
        rows.append(dict(label=entry[0], n_dots=n, model_T=T,
                         model_surrogate_dti=dti, owner_reported_score=entry[2]))
    out = dict(
        schema="gems59.model-compare.v1",
        note=("model_T is the p-weighted truth-centred credit each dot field "
              "collects on the session-4 surface; model_surrogate_dti applies "
              "the no-redundancy form T/(0.2n + 0.8G) with the live-anchored "
              "G = 14,089. MODEL-SURROGATE, not a score."),
        rows=rows)
    (ROOT / "evidence" / "h57k_model_compare.json").write_text(json.dumps(out, indent=1))
    print(f"{'dot field':40s} {'dots':>8s} {'model T':>9s} {'surr.DTI':>9s} "
          f"{'owner-reported':>15s}")
    for r in rows:
        o = r["owner_reported_score"]
        print(f"{r['label'][:40]:40s} {r['n_dots']:8,d} {r['model_T']:9.1f} "
              f"{r['model_surrogate_dti']:9.4f} {(f'{o:.4f}' if o else '—'):>15s}")


if __name__ == "__main__":
    main()
