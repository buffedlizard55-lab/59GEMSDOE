"""Uniqueness check of a candidate raster against a folder of registry rasters.

Two surface measures, as in the parallel-run protocol:
  * Spearman rank correlation of the two rasters over a fixed random sample of in-footprint cells
  * share of the candidate's positive cells lying within 3 px (300 m) of a registry positive cell

Usage:
  PYTHONPATH=src python scripts/check_uniqueness.py candidate.tif registry_dir/ [--rho-max 0.90 --overlap-max 0.70]

A candidate is flagged DUPLICATE if rho > 0.90 or overlap > 70% against any registry raster
(threshold values from the protocol; they are parameters, not established constants).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage, stats

SAMPLE = 200_000


def load(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def compare(cand: np.ndarray, ref: np.ndarray, seed: int = 0) -> dict:
    valid = np.isfinite(cand) & np.isfinite(ref)
    idx = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(SAMPLE, idx.size), replace=False)
    rho = stats.spearmanr(cand.ravel()[pick], ref.ravel()[pick]).statistic
    cpos = cand > 0
    rpos = ref > 0
    if cpos.sum() == 0 or rpos.sum() == 0:
        overlap = 0.0
    else:
        d = ndimage.distance_transform_edt(~rpos)
        overlap = float(np.mean(d[cpos] <= 3.0))
    return {"spearman_rho": float(rho), "dots_within_3px_share": overlap,
            "candidate_positive_cells": int(cpos.sum()), "registry_positive_cells": int(rpos.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("registry", type=Path)
    ap.add_argument("--rho-max", type=float, default=0.90)
    ap.add_argument("--overlap-max", type=float, default=0.70)
    args = ap.parse_args()
    cand = load(args.candidate)
    results = {}
    for ref_path in sorted(args.registry.glob("*.tif")):
        if ref_path.resolve() == args.candidate.resolve():
            continue
        ref = load(ref_path)
        if ref.shape != cand.shape:
            results[ref_path.name] = {"skipped": f"shape {ref.shape} != {cand.shape}"}
            continue
        r = compare(cand, ref)
        r["duplicate_flag"] = bool(r["spearman_rho"] > args.rho_max or r["dots_within_3px_share"] > args.overlap_max)
        results[ref_path.name] = r
    flagged = [k for k, v in results.items() if v.get("duplicate_flag")]
    print(json.dumps({"candidate": args.candidate.name, "flagged_duplicates": flagged, "comparisons": results}, indent=2))


if __name__ == "__main__":
    main()
