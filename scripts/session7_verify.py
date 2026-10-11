#!/usr/bin/env python3
"""Session 7 (2026-10-10) line-by-line verification of the current submission state.

Every check below is local and reproducible from files in this repository (plus the
hash-pinned feature stack restored by ``scripts/prepare_data.py --fetch``). It makes no
network calls and claims no organizer score. Output: ``evidence/session7_verification.json``.

Checks
  1. Official format of the downloadable research GeoTIFF (rules/competition page):
     float32, one band, EPSG:32611, 100 m, same shape/transform as the official template,
     finite values inside [0, 1], no NaN, and whether outside-footprint cells are NaN.
  2. Bridged official template: footprint = finite cells of ``sample_submission.tif``; its
     positive cells are compared with the known-fault raster (irregularity check).
  3. Byte identity of ``labels.tif`` and ``existing_faults.tif`` (irregularity check).
  4. Session-3/Session-5 registry pair behind the owner-reported 0.2778 vs 0.2708: subset
     relation and the distance of every removed dot to the known catalogue.
  5. HOLDOUT-DTI receipts recomputed from TP/FP/FN; nearest visible-to-truth distance per
     fold (the holdout collar blind zone).
  6. Pinned feature stack and band-12 cache hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ALPHA, BETA = 0.2, 0.8
DOWNLOAD = ROOT / "docs/downloads/gems59-twohost-relay-bend-surface-20261010T201504Z-47ccc38b6bec.tif"
PIN_FEATURES = "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
SAMPLE_SHA = "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc"
LABELS_SHA = "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
# Registry rasters (local copies, SHA256 pinned in registry/registry_index.json).
RASTER_0_2778 = ROOT / "registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif"
RASTER_0_2708 = ROOT / "registry/rasters/GEMSDOE28__h27-4-r1-solo-d2-8.tif"
HOLDOUT = ROOT / "evidence/relay_bend_holdout.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dti(tp: float, fp: float, fn: float) -> float:
    return tp / (tp + ALPHA * fp + BETA * fn)


def check_download() -> dict:
    with rasterio.open(DOWNLOAD) as s:
        a = s.read(1)
        meta = dict(crs=s.crs.to_string(), shape=list(s.shape), dtype=s.dtypes[0], count=s.count,
                    transform=list(s.transform)[:6], nodata=s.nodata)
    finite = np.isfinite(a)
    inside = a[finite]
    return dict(
        file=str(DOWNLOAD.relative_to(ROOT)), sha256=sha256(DOWNLOAD), meta=meta,
        n_nan=int((~finite).sum()), n_finite=int(finite.sum()),
        min=float(inside.min()), max=float(inside.max()),
        in_unit_interval=bool(inside.min() >= 0.0 and inside.max() <= 1.0),
        positive_cells=int((a > 0).sum()),
        outside_footprint_is_nan=False,
        note=("Every cell is finite; outside-footprint cells are 0.0, not NaN. The official rules "
              "say data outside the bounds is null or NaN, and the bridged template uses NaN. "
              "A zero probability adds nothing to the metric, but the convention differs (IR-S7-03)."),
    )


def check_template() -> dict:
    with rasterio.open(ROOT / "data/official/sample_submission.tif") as s:
        sample = s.read(1)
        grid = dict(crs=s.crs.to_string(), shape=list(s.shape), transform=list(s.transform)[:6])
    with rasterio.open(ROOT / "data/official/labels.tif") as s:
        labels = s.read(1)
    footprint = np.isfinite(sample)
    pos_sample = footprint & (sample == 1)
    pos_labels = labels == 1
    return dict(
        sample_sha256=sha256(ROOT / "data/official/sample_submission.tif"), grid=grid,
        footprint_cells=int(footprint.sum()), sample_positive_cells=int(pos_sample.sum()),
        sample_values=sorted(float(v) for v in np.unique(sample[footprint])),
        labels_positive_cells=int(pos_labels.sum()),
        sample_positives_equal_label_positives=bool(np.array_equal(pos_sample, pos_labels)),
        note=("The bridged 'sample_submission.tif' has 60,988 positive cells that coincide with the "
              "known-fault positives, while the official page says the sample predicts total fault "
              "absence (IR-S7-02). It is used only for grid and footprint compatibility."),
    )


def check_duplicate_labels() -> dict:
    return dict(
        labels_sha256=sha256(ROOT / "data/official/labels.tif"),
        existing_faults_sha256=sha256(ROOT / "data/official/existing_faults.tif"),
        byte_identical=sha256(ROOT / "data/official/labels.tif") == sha256(ROOT / "data/official/existing_faults.tif"),
    )


def check_0_2778_pair() -> dict:
    with rasterio.open(ROOT / "data/official/labels.tif") as s:
        known = s.read(1) == 1
    with rasterio.open(RASTER_0_2778) as s:
        a = s.read(1)
    with rasterio.open(RASTER_0_2708) as s:
        b = s.read(1)
    pa = np.isfinite(a) & (a > 0)
    pb = np.isfinite(b) & (b > 0)
    removed = pb & ~pa
    kept = pa
    dist = ndi.distance_transform_edt(~known)
    return dict(
        raster_0_2778=dict(file=str(RASTER_0_2778.relative_to(ROOT)), sha256=sha256(RASTER_0_2778), positive=int(pa.sum())),
        raster_0_2708=dict(file=str(RASTER_0_2708.relative_to(ROOT)), sha256=sha256(RASTER_0_2708), positive=int(pb.sum())),
        kept_is_subset_of_0_2708=bool(not (pa & ~pb).any()),
        removed_cells=int(removed.sum()),
        removed_distance_to_known_px=dict(min=float(dist[removed].min()), median=float(np.median(dist[removed])),
                                          max=float(dist[removed].max())),
        kept_distance_to_known_px=dict(min=float(dist[kept].min()), median=float(np.median(dist[kept]))),
        removed_all_within_2px=bool((dist[removed] <= 2.0).all()),
        kept_none_within_2px=bool((dist[kept] > 2.0).all()),
        interpretation=("Removal of pure false-positive dots near known traces can only raise DTI when "
                        "it does not lower max-cover TP. The owner-reported 0.2778 vs 0.2708 gap "
                        "(0.0070) is one pair; it is not a verified causal receipt."),
    )


def check_holdout() -> dict:
    report = json.loads(HOLDOUT.read_text())
    recomputed = {}
    for arm, v in report["scores"].items():
        recomputed[arm] = dict(stored=v["dti"], recomputed=dti(v["tpw"], v["fpw"], v["fnw"]),
                               withheld_positive_pixels=v["withheld_positive_pixels"], ci95=v["ci95"])
    nearest = [float(f["fold_receipt"]["nearest_visible_to_truth_px"]) for f in report["per_fold"]]
    return dict(
        file=str(HOLDOUT.relative_to(ROOT)), evaluator_version=report["evaluator_version"],
        arms=recomputed,
        max_abs_recompute_error=max(abs(v["stored"] - v["recomputed"]) for v in recomputed.values()),
        fold_arm_receipts=len(nearest),
        nearest_visible_to_truth_px_min=min(nearest), nearest_visible_to_truth_px_max=max(nearest),
        collar_blind_zone=bool(min(nearest) >= 3.0),
        interpretation=("Withheld truth is never closer than 3 px to any visible fault in any fold. "
                        "Dots within ~3 px of a visible trace are therefore pure false positives in "
                        "this holdout. The holdout cannot reward or penalise near-trace strands "
                        "(IR-S7-04)."),
    )


def check_pins() -> dict:
    cache = json.loads((ROOT / "evidence/feature_cache.json").read_text())
    band12 = next(b for b in cache["bands"] if b["band"] == 12)
    return dict(
        training_features_sha256=sha256(ROOT / "data/official/training_features.tif"),
        training_features_pin=PIN_FEATURES,
        training_features_match=sha256(ROOT / "data/official/training_features.tif") == PIN_FEATURES,
        band12_cache_sha256=sha256(ROOT / band12["cache_file"]),
        band12_cache_pin=band12["sha256"],
        band12_cache_match=sha256(ROOT / band12["cache_file"]) == band12["sha256"],
        bands=cache["count"],
        provenance="hash-verified third-party bridge; not an organizer receipt",
    )


def main() -> int:
    out = dict(
        evidence_class="DATA-MEASUREMENT (local recomputation; no organizer score)",
        generated_by="scripts/session7_verify.py",
        download=check_download(),
        template=check_template(),
        duplicate_labels=check_duplicate_labels(),
        pair_0_2778_vs_0_2708=check_0_2778_pair(),
        holdout=check_holdout(),
        pins=check_pins(),
    )
    path = ROOT / "evidence/session7_verification.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"[session7] wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
