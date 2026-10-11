#!/usr/bin/env python3
"""Session 7 (2026-10-10): recompute the measured facts behind the Session-7 status.

Every number written to ``evidence/session7_recompute_arena.json`` is measured here from
local files. Nothing is a score. Owner-reported leaderboard values are copied as
labels only. Run from the repository root with the locked environment:

    python scripts/verify_session7.py

Inputs (all local, sha256 recorded in the output):
  * registry/rasters/GEMSDOE28__h27-4-r1-solo-d2-8.tif  (owner-reported 0.2708 file)
  * registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif      (owner-reported 0.2778 file)
  * data/official/existing_faults.tif                  (known-fault catalogue, bridge bytes)
  * data/bridge/sample_submission.tif                  (grid/footprint reference)
  * .cache/registry/374c88b1da2b803e2ed160de467fd71ae114010d.tif
        (17GEMSDOE E-proba-multiscale, fetched from GitHub blob; sha256 pinned below)
  * evidence/feature_cache.json                       (19 band hashes, committed)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PRIOR_27 = ROOT / "registry/rasters/GEMSDOE28__h27-4-r1-solo-d2-8.tif"
PRIOR_33 = ROOT / "registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif"
CATALOGUE = ROOT / "data/official/existing_faults.tif"
SAMPLE = ROOT / "data/bridge/sample_submission.tif"
EPROBA = ROOT / ".cache/registry/374c88b1da2b803e2ed160de467fd71ae114010d.tif"
EPROBA_SHA = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"
FEATURE_CACHE = ROOT / "evidence/feature_cache.json"
OUT = ROOT / "evidence/session7_recompute_arena.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def main() -> int:
    for p in (PRIOR_27, PRIOR_33, CATALOGUE, SAMPLE):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    if not EPROBA.exists():
        raise SystemExit("missing 17GEMSDOE E-proba cache; fetch from GitHub blob first")
    if sha256(EPROBA) != EPROBA_SHA:
        raise SystemExit("E-proba cache sha256 mismatch")

    a27 = read(PRIOR_27)
    a33 = read(PRIOR_33)
    cat_raw = read(CATALOGUE)
    sample = read(SAMPLE)
    eprob = read(EPROBA)

    # --- 1. Owner-reported 0.2778 vs 0.2708 file containment -------------------
    pos27 = np.isfinite(a27) & (a27 > 0)
    pos33 = np.isfinite(a33) & (a33 > 0)
    cat = cat_raw == 1  # catalogue encoding: 1 = mapped fault (values -1, 0, 1 observed)
    dist_cat = ndi.distance_transform_edt(~cat)
    removed = pos27 & ~pos33
    added = pos33 & ~pos27
    containment = dict(
        evidence_class="RASTER-MEASUREMENT (owner-reported labels only)",
        lower_label="GEMSDOE28 h27-4-r1-solo-d2-8 (OWNER-REPORTED 0.2708)",
        higher_label="GEMSDOE32 h33-2-b2-zeros (OWNER-REPORTED 0.2778)",
        lower_sha256=sha256(PRIOR_27),
        higher_sha256=sha256(PRIOR_33),
        lower_positive_cells=int(pos27.sum()),
        higher_positive_cells=int(pos33.sum()),
        removed_cells=int(removed.sum()),
        added_cells=int(added.sum()),
        higher_is_subset_of_lower=bool(added.sum() == 0),
        removed_distance_to_catalogue_px_min=float(dist_cat[removed].min()) if removed.any() else None,
        removed_distance_to_catalogue_px_max=float(dist_cat[removed].max()) if removed.any() else None,
        removed_on_catalogue_cells=int((removed & cat).sum()),
        unique_positive_values_lower=[float(v) for v in np.unique(a27[pos27])],
        unique_positive_values_higher=[float(v) for v in np.unique(a33[pos33])],
        causal_link_to_score="NOT ESTABLISHED: no organizer receipt; the two scores are owner-reported",
    )

    # --- 2. 17GEMSDOE E-proba saturation of the literal support rule -----------
    finite = np.isfinite(eprob)
    positive = finite & (eprob > 0)
    fp = sample.astype(np.float64)
    allowed = np.isfinite(fp) & ~cat
    covered = positive & allowed
    saturation = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        witness_sha256=sha256(EPROBA),
        witness_source="17GEMSDOE:docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif",
        nan_cells=int((~finite).sum()),
        finite_positive_cells=int(positive.sum()),
        finite_positive_fraction_of_grid=float(positive.mean()),
        allowed_cells=int(allowed.sum()),
        allowed_cells_with_positive_support=int(covered.sum()),
        allowed_coverage_fraction=float(covered.sum() / allowed.sum()),
        literal_gate_overlap_limit=0.70,
        implication=(
            "Under the literal finite>0 support rule, any nonempty candidate placed in the "
            "allowed domain has forward overlap 1.0 with this witness, so the literal "
            "gate cannot pass. The rule must be ruled by the owner (IR-S7-02)."
        ),
    )

    # --- 3. Feature-band hashes reproduce the committed manifest ---------------
    manifest = json.loads(FEATURE_CACHE.read_text())
    band_check = []
    for row in manifest["bands"]:
        path = ROOT / row["cache_file"]
        if path.exists():
            band_check.append(dict(band=row["band"], matches=sha256(path) == row["sha256"]))
        else:
            band_check.append(dict(band=row["band"], matches=None, note="cache file absent"))
    features = dict(
        evidence_class="DATA-MEASUREMENT",
        source_sha256=manifest["source_sha256"],
        bands_checked=len(band_check),
        bands_matching=sum(1 for b in band_check if b["matches"] is True),
        per_band=band_check,
    )

    out = dict(
        evidence_class="SESSION-7 VERIFICATION (measurements, not scores)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/verify_session7.py",
        containment_0_2778_vs_0_2708=containment,
        e_proba_saturation=saturation,
        feature_band_hash_reproduction=features,
        download_status="NO (HOLD, IR-S6-10 unchanged)",
        submission_status="NO (literal uniqueness gate unsatisfiable; owner ruling required)",
        submission_slots_used=0,
    )
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(
        subset=containment["higher_is_subset_of_lower"],
        removed=containment["removed_cells"],
        removed_dist=[containment["removed_distance_to_catalogue_px_min"],
                      containment["removed_distance_to_catalogue_px_max"]],
        eproba_positive_fraction=saturation["finite_positive_fraction_of_grid"],
        allowed_coverage=saturation["allowed_coverage_fraction"],
        bands_matching=features["bands_matching"],
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
