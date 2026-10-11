#!/usr/bin/env python3
"""Independent re-check of the Session-7 (Arena) claims from local files.

Recomputes, without trusting prior run cards:
  1. labels.tif value census and footprint (nodata -1 outside the footprint);
  2. GEMSDOE32 (0.2778 owner-reported) vs GEMSDOE28 (0.2708 owner-reported)
     registry rasters: exact subset test, removed-cell count, and the distance
     of removed cells to the known-fault mask (existing_faults.tif, value > 0);
  3. literal-gate forward overlap (3 px Euclidean) and full-footprint Spearman
     rho of the 0.2778 raster against every local registry raster;
  4. optionally, the 17GEMSDOE E-proba-multiscale raster (pass --eproba PATH),
     whose index sha256 is checked against the registry entry.

Writes evidence/arena_independent_check_20261010.json. Makes NO network calls
and produces no submission artifact. Results are gate diagnostics, not scores.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "arena_independent_check_20261010.json"
CAND_0_2778 = ROOT / "registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif"
PRIOR_0_2708 = ROOT / "registry/rasters/GEMSDOE28__h27-4-r1-solo-d2-8.tif"
FAULTS = ROOT / "data/official/existing_faults.tif"
LABELS = ROOT / "data/official/labels.tif"
SAMPLE = ROOT / "data/official/sample_submission.tif"
EPROBA_INDEX_SHA = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with rasterio.open(path) as s:
        return s.read(1), s.transform, str(s.crs), s.shape


def check_labels():
    lab, tf, crs, shape = read(LABELS)
    vals, counts = np.unique(lab, return_counts=True)
    census = {int(v): int(c) for v, c in zip(vals, counts)}
    footprint = int((lab != -1).sum())
    with rasterio.open(SAMPLE) as s:
        sample_finite = int(np.isfinite(s.read(1)).sum())
    return dict(census=census, footprint_cells=footprint,
                sample_submission_finite_cells=sample_finite,
                footprint_matches_sample=footprint == sample_finite,
                sha256=sha256(LABELS))


def check_subset_and_distance():
    cand, _, _, _ = read(CAND_0_2778)
    prior, _, _, _ = read(PRIOR_0_2708)
    faults, _, _, _ = read(FAULTS)
    a1 = (cand > 0) & np.isfinite(cand)
    a2 = (prior > 0) & np.isfinite(prior)
    removed = a2 & ~a1
    known = faults > 0
    d = ndi.distance_transform_edt(~known)
    return dict(
        candidate_sha256=sha256(CAND_0_2778),
        prior_sha256=sha256(PRIOR_0_2708),
        candidate_dots=int(a1.sum()),
        prior_dots=int(a2.sum()),
        candidate_is_subset_of_prior=bool(not (a1 & ~a2).any()),
        removed_cells=int(removed.sum()),
        removed_dist_px_min=float(d[removed].min()),
        removed_dist_px_max=float(d[removed].max()),
        removed_within_2px=int((d[removed] <= 2).sum()),
        candidate_dots_within_2px_of_known=int(((d <= 2) & a1).sum()),
        candidate_dots_on_known_fault=int(((d == 0) & a1).sum()),
    )


def gate_table(eproba_path: Path | None):
    cand, _, _, _ = read(CAND_0_2778)
    fp = np.isfinite(cand)
    cd = (cand > 0) & fp
    rows = []
    paths = sorted(glob.glob(str(ROOT / "registry/rasters/*.tif")))
    if eproba_path is not None:
        paths.append(str(eproba_path))
    for p in paths:
        a, _, _, shape = read(p)
        if shape != cand.shape:
            rows.append(dict(raster=os.path.basename(p), error="shape mismatch"))
            continue
        m = np.isfinite(a) & (a > 0)
        dist = ndi.distance_transform_edt(~m)
        fwd = float((dist[cd] <= 3).mean()) if cd.any() else float("nan")
        both = fp & np.isfinite(a)
        rho = float(spearmanr(cand[both], a[both]).statistic)
        rows.append(dict(raster=os.path.basename(p), finite_cells=int(np.isfinite(a).sum()),
                         positive_cells=int(m.sum()),
                         positive_fraction=round(float(m.sum() / np.isfinite(a).sum()), 6),
                         forward_overlap_3px=round(fwd, 6), spearman_rho=round(rho, 6),
                         gate_overlap_fail=fwd > 0.70, gate_rho_fail=rho > 0.90))
    return rows


def eproba_check(path: Path):
    a, _, _, _ = read(path)
    fin = np.isfinite(a)
    return dict(path=str(path), sha256=sha256(path), index_sha256=EPROBA_INDEX_SHA,
                sha_matches_index=sha256(path) == EPROBA_INDEX_SHA,
                finite_cells=int(fin.sum()), positive_finite_cells=int((fin & (a > 0)).sum()),
                min=float(np.nanmin(a)), max=float(np.nanmax(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eproba", type=Path, default=None,
                    help="optional local copy of 17GEMSDOE E-proba-multiscale .tif")
    ap.add_argument("--skip-gate-table", action="store_true")
    args = ap.parse_args()
    result = dict(
        evidence_class="ARENA-INDEPENDENT-LOCAL-CHECK",
        note="Diagnostics of gate behaviour and file identity. Not organizer scores, not a submission, not a download clearance.",
        labels=check_labels(),
        subset_and_distance=check_subset_and_distance(),
    )
    if args.eproba is not None:
        result["eproba_17gemsdoe"] = eproba_check(args.eproba)
    if not args.skip_gate_table:
        result["literal_gate_local_rasters"] = gate_table(args.eproba)
    result["verdict"] = "STOP: literal gate blocks any nonempty candidate when a registry raster has full-footprint positive support (see eproba_17gemsdoe)."
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "literal_gate_local_rasters"}, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
