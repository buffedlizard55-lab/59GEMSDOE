"""Build the cached rank/edge feature bank used by the H59 intensity lane.

The input is the hash-pinned 19-band competition feature raster restored by
``scripts/prepare_data.py``.  Its checksum verifies byte identity with the
public GitHub bridge pin; it is not an independent DrivenData provenance
receipt.  Outputs are ignored by Git under ``out/features/``.

This prepares (1) two catalogue-independent transforms per supplied band,
(2) optional full-catalogue geometry caches retained for older exploratory
scripts.  Holdout evaluation must never use the latter as-is: every catalogue-
derived feature must be rebuilt from the fold's visible faults only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
FEATURE = REPO / "data" / "official" / "training_features.tif"
SAMPLE = REPO / "data" / "official" / "sample_submission.tif"
CATALOGUE = REPO / "data" / "official" / "existing_faults.tif"
OUT = REPO / "out" / "features"
EXPECTED_SHA256 = "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
EXPECTED_BYTES = 418_912_844
NODATA = -3.4028234663852886e+38
RANK_MAX = 254
NODATA_U8 = 255


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_u8(values: np.ndarray) -> np.ndarray:
    """Average-rank values to uint8 0..254; tied values receive tied ranks."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    if sorted_values.size > 1:
        ranks *= RANK_MAX / (sorted_values.size - 1)
    return np.clip(np.rint(ranks), 0, RANK_MAX).astype(np.uint8)


def _read_footprint() -> np.ndarray:
    with rasterio.open(SAMPLE) as source:
        return np.isfinite(source.read(1))


def build_feature_bank(*, force: bool = False, root: Path = REPO) -> dict:
    """Build and verify the feature bank; return its small JSON receipt."""
    root = Path(root)
    feature = root / "data/official/training_features.tif"
    sample = root / "data/official/sample_submission.tif"
    catalogue_path = root / "data/official/existing_faults.tif"
    output = root / "out/features"
    receipt_path = root / "evidence/feature_bank.json"
    bank_path = output / "bank_u8.npy"
    geo_path = output / "geo_f32.npy"
    tip_path = output / "dtip_f32.npy"

    if not feature.is_file():
        raise FileNotFoundError(
            f"Missing {feature}; run `python scripts/prepare_data.py --fetch` first."
        )
    nbytes = feature.stat().st_size
    digest = sha256(feature)
    if nbytes != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise ValueError(
            "Refusing to build features from an unverified source: "
            f"bytes={nbytes} sha256={digest}"
        )
    output.mkdir(parents=True, exist_ok=True)

    if (not force and bank_path.is_file() and geo_path.is_file() and tip_path.is_file()
            and receipt_path.is_file()):
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_shape = (38, 3730, 3292)
        cached = np.load(bank_path, mmap_mode="r")
        valid_cache = (
            existing.get("source_sha256") == digest
            and tuple(existing.get("bank_shape", ())) == expected_shape
            and cached.shape == expected_shape
            and cached.dtype == np.uint8
        )
        del cached
        if valid_cache:
            return {**existing, "cache_reused": True}

    footprint = _read_footprint()
    if footprint.shape != (3730, 3292):
        raise ValueError(f"reference footprint has unexpected shape {footprint.shape}")
    footprint_cells = int(footprint.sum())
    info = []

    with rasterio.open(feature) as source, rasterio.open(sample) as reference:
        if (source.shape != reference.shape or source.count != 19
                or source.crs != reference.crs or source.transform != reference.transform):
            raise ValueError("feature stack does not match the pinned 19-band reference grid")
        if tuple(source.res) != (100.0, 100.0):
            raise ValueError(f"unexpected feature pixel size {source.res}")
        bank = np.lib.format.open_memmap(
            bank_path, mode="w+", dtype=np.uint8,
            shape=(2 * source.count, source.height, source.width),
        )
        for band in range(source.count):
            values = source.read(band + 1).astype(np.float32, copy=False)
            bad = (~np.isfinite(values)) | (values <= NODATA / 10.0)
            valid = (~bad) & footprint
            valid_values = values[valid]
            q = np.full(values.shape, NODATA_U8, np.uint8)
            z = np.full(values.shape, NODATA_U8, np.uint8)
            if valid_values.size > 100:
                q[valid] = _rank_u8(valid_values)
                median = float(np.median(valid_values))
                filled = np.where(bad, median, values).astype(np.float32, copy=False)
                mean = ndi.uniform_filter(filled, size=9, mode="nearest")
                mean_sq = ndi.uniform_filter(filled * filled, size=9, mode="nearest")
                std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)) + 1e-6
                local_z = (filled - mean) / std
                z[valid] = _rank_u8(local_z[valid])
                band_name = (source.descriptions[band] or f"band_{band + 1}").split(" - ")[0]
                band_median = median
            else:
                band_name = source.descriptions[band] or f"band_{band + 1}"
                band_median = None
            bank[2 * band] = q
            bank[2 * band + 1] = z
            info.append(dict(
                band=band + 1,
                name=band_name,
                nodata_cells=int(bad.sum()),
                nodata_in_footprint=int((bad & footprint).sum()),
                median=band_median,
            ))
            del values, bad, valid, valid_values, q, z
            if "filled" in locals():
                del filled, mean, mean_sq, std, local_z
        bank.flush()
        del bank

    # These geometry planes are retained for legacy exploratory scripts. They
    # must not be consumed by a spatial holdout: a holdout must rebuild them from
    # that fold's visible mask. The current intensity model uses only bank_u8.
    with rasterio.open(catalogue_path) as source:
        catalogue = (source.read(1) > 0) & footprint
    if catalogue.shape != footprint.shape:
        raise ValueError("catalogue raster does not match reference grid")
    distance = ndi.distance_transform_edt(~catalogue).astype(np.float32)
    geo = np.lib.format.open_memmap(
        geo_path, mode="w+", dtype=np.float32,
        shape=(4, catalogue.shape[0], catalogue.shape[1]),
    )
    yy, xx = np.mgrid[-15:16, -15:16]
    kernel15 = ((yy * yy + xx * xx) <= 225).astype(np.float32)
    yy, xx = np.mgrid[-5:6, -5:6]
    kernel5 = ((yy * yy + xx * xx) <= 25).astype(np.float32)
    geo[0] = distance
    geo[1] = np.log1p(distance)
    geo[2] = ndi.convolve(catalogue.astype(np.float32), kernel5, mode="constant")
    geo[3] = ndi.convolve(catalogue.astype(np.float32), kernel15, mode="constant")
    geo.flush()
    del distance, geo

    labels, n_components = ndi.label(catalogue, structure=np.ones((3, 3), bool))
    tips = np.zeros(catalogue.shape, bool)
    # Crop each component before erosion. The previous full-grid-per-component
    # loop performed O(n_components * grid_cells) work and could take hours.
    for component_id, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        local_labels = labels[slices]
        component = local_labels == component_id
        if int(component.sum()) < 2:
            tips[slices] |= component
            continue
        eroded = ndi.binary_erosion(component, structure=np.ones((3, 3), bool), border_value=0)
        tips[slices] |= component & ~eroded
    tip_distance = ndi.distance_transform_edt(~tips).astype(np.float32)
    np.save(tip_path, tip_distance)
    del labels, tips, tip_distance, catalogue

    receipt = dict(
        evidence_class="FEATURE-CACHE-MEASUREMENT",
        schema="gems59.feature-bank.v2",
        source_sha256=digest,
        source_bytes=nbytes,
        source_provenance=(
            "hash-verified public GitHub bridge; organizer origin is not independently authenticated"
        ),
        bands=[row["name"] for row in info],
        bank_shape=[38, 3730, 3292],
        footprint_cells=footprint_cells,
        rank_max=RANK_MAX,
        nodata_code=NODATA_U8,
        per_band=info,
        geo_names=["dcat", "log1p_dcat", "cat_dens_5", "cat_dens_15"],
        extra=["dtip_f32.npy (distance to nearest mapped-fragment tip; full-catalogue only)"],
        full_catalogue_connected_components=int(n_components),
        holdout_warning=(
            "geo_f32.npy and dtip_f32.npy were derived from the full catalogue and are not valid "
            "inside a hide-and-recover fold; rebuild from visible faults only."
        ),
        cache_files={
            "bank_u8.npy": str(bank_path.relative_to(root)),
            "geo_f32.npy": str(geo_path.relative_to(root)),
            "dtip_f32.npy": str(tip_path.relative_to(root)),
        },
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return {**receipt, "cache_reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild existing ignored caches")
    args = parser.parse_args()
    result = build_feature_bank(force=args.force)
    print(json.dumps({key: value for key, value in result.items() if key != "per_band"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
