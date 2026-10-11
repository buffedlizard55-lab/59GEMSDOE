#!/usr/bin/env python3
"""Reproduce the literal-gate 'universal overlap blocker' from raw rasters.

Read-only. No candidate is generated, no slot is used, nothing is submitted.

Checks, for the competition grid (EPSG:32611, 3730 x 3292, 100 m):
1. The certificate witness (17GEMSDOE E-proba-multiscale) matches its pinned SHA256.
2. Its positive-finite support covers every allowed (finite-footprint) cell.
   If so, any nonempty candidate restricted to the footprint has 3 px forward
   overlap 1.0 with it, and cannot pass the inherited 0.70 gate.
3. For every local registry raster, the fraction of allowed cells that are
   positive-finite (a per-raster view of how widespread full support is).

Usage:
  .venv/bin/python scripts/verify_literal_gate_witness.py --witness PATH_TO_WITNESS_TIF
Writes evidence/literal_gate_witness_verification_20261010.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
WITNESS_SHA256 = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"
SAMPLE = ROOT / "data" / "official" / "sample_submission.tif"
REGISTRY = ROOT / "registry" / "rasters"
OUT = ROOT / "evidence" / "literal_gate_witness_verification_20261010.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def support(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (a > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--witness", required=True, type=Path)
    args = ap.parse_args()

    with rasterio.open(SAMPLE) as s:
        allowed = np.isfinite(s.read(1))
        sample_transform, sample_crs, sample_shape = s.transform, s.crs.to_string(), s.shape

    w_sha = sha256(args.witness)
    if w_sha != WITNESS_SHA256:
        raise SystemExit(f"witness SHA256 mismatch: {w_sha}")

    with rasterio.open(args.witness) as s:
        if s.transform != sample_transform or s.crs.to_string() != sample_crs or s.shape != sample_shape:
            raise SystemExit("witness grid does not match the competition grid")
        w = s.read(1)
    w_pos = support(w)
    witness_row = {
        "file": args.witness.name,
        "sha256": w_sha,
        "sha256_matches_certificate": True,
        "allowed_cells": int(allowed.sum()),
        "positive_cells": int(w_pos.sum()),
        "allowed_cells_covered_by_positive": int((w_pos & allowed).sum()),
        "allowed_fraction_covered": float((w_pos & allowed).sum() / allowed.sum()),
        "outside_footprint_positive": int((w_pos & ~allowed).sum()),
        "nan_cells": int(np.isnan(w).sum()),
        "min": float(np.nanmin(w)),
        "max": float(np.nanmax(w)),
    }
    witness_row["universal_overlap_blocker"] = bool(
        witness_row["allowed_fraction_covered"] == 1.0 and witness_row["outside_footprint_positive"] == 0
    )

    local = []
    for p in sorted(REGISTRY.glob("*.tif")):
        with rasterio.open(p) as s:
            a = s.read(1)
            same_grid = s.transform == sample_transform and s.shape == sample_shape
        pos = support(a)
        local.append({
            "file": p.name,
            "sha256": sha256(p),
            "same_grid": bool(same_grid),
            "positive_cells": int(pos.sum()),
            "allowed_fraction_covered": float((pos & allowed).sum() / allowed.sum()) if same_grid else None,
            "outside_footprint_positive": int((pos & ~allowed).sum()) if same_grid else None,
        })

    # Owner-reported 0.2778 file vs its nested base (file-level mechanism check only).
    from scipy.ndimage import distance_transform_edt
    g32_p = REGISTRY / "GEMSDOE32__h33-2-b2-zeros.tif"
    base_p = REGISTRY / "GEMSDOE28__h27-4-r1-solo-d2-8.tif"
    cat_p = ROOT / "data" / "official" / "existing_faults.tif"
    with rasterio.open(g32_p) as s:
        g32 = support(s.read(1))
    with rasterio.open(base_p) as s:
        base = support(s.read(1))
    with rasterio.open(cat_p) as s:
        cat = s.read(1) > 0
    dcat = distance_transform_edt(~cat)
    removed = base & ~g32
    alpha, owner_score = 0.2, 0.2778
    subset = {
        "files": {"candidate": g32_p.name, "nested_base": base_p.name,
                  "candidate_sha256": sha256(g32_p), "base_sha256": sha256(base_p)},
        "candidate_positive": int(g32.sum()),
        "base_positive": int(base.sum()),
        "candidate_is_subset_of_base": bool(not (g32 & ~base).any()),
        "removed_vs_base": int(removed.sum()),
        "removed_dist_to_catalogue_px_min": float(dcat[removed].min()),
        "removed_dist_to_catalogue_px_max": float(dcat[removed].max()),
        "kept_dist_to_catalogue_px_min": float(dcat[g32].min()),
        "owner_reported_score_used_only_for_bar": owner_score,
        "dti_bar_k_gt": alpha * owner_score,
        "dti_bar_equivalent_distance_px_lt": 3.0 * (1.0 - alpha * owner_score),
        "note": "Metric bar from src/gems59/metric.py: a non-redundant dot is worth keeping only if k > alpha*DTI. "
                "This is a mechanism check on files, not a score; owner-reported 0.2778 has no organizer receipt.",
    }

    record = {
        "evidence_class": "REGISTRY-MEASUREMENT (reproduced from raw rasters; not a score)",
        "session": "7 (2026-10-10)",
        "rule": "inherited literal support: finite and > 0 pixels are dots; forward 3 px overlap > 0.70 blocks",
        "competition_grid": {"crs": sample_crs, "shape": list(sample_shape), "allowed_cells": int(allowed.sum())},
        "witness": witness_row,
        "conclusion": (
            "Any nonempty candidate confined to the allowed footprint has forward 3 px overlap 1.0 with the witness, "
            "so the literal gate cannot be passed by any candidate. Changing this requires an explicit owner protocol "
            "revision (support definition), not a different candidate."
            if witness_row["universal_overlap_blocker"] else
            "Witness does not cover all allowed cells; the universal blocker is NOT reproduced."
        ),
        "gemsdoe32_subset_check": subset,
        "local_registry_rasters": local,
        "not_claimed": [
            "organizer acceptance or rejection of any file",
            "leaderboard placement or private score",
            "HOLDOUT-DTI of any candidate",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("conclusion",)}, indent=2))
    print("witness", witness_row["allowed_fraction_covered"], "blocker", witness_row["universal_overlap_blocker"])
    print("local rasters checked:", len(local), "| full-footprint in local set:",
          sum(1 for r in local if r["allowed_fraction_covered"] == 1.0))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
