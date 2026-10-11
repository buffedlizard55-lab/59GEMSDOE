"""Writer and validator for the GEMS competition GeoTIFF.

Official requirements (DrivenData problem page, "Submission format", fetched 2026-10-11):
  * same projected CRS as the training data: EPSG:32611 (UTM 11N)
  * same resolution as the training data: 100 m
  * same bounds as the training data; data outside the bounds is null or NaN
  * single layer, dtype float32, values in [0, 1]

The portal error "Predicted values must be in range [0, 1]" is the one this module is built to
prevent. The footprint is the set of cells where the official template is finite. Inside it we write
finite values in [0, 1]; outside it we write NaN, as the official text requires for data outside the
bounds. Non-finite values inside the footprint, the -3.4028e38 sentinel, and out-of-range values are
refused before anything is written.

The template (CRS, transform, shape) is read from an official file that the user places in data/
(sample_submission.tif or labels.tif). Nothing is hard-coded here, because the origin of the grid
is not stated on any page we could reach and must not be guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

SENTINEL_NODATA = -3.4028234663852886e38  # the float32 sentinel present in the competition's inputs
NOTE_MAX_CHARS = 140  # protocol: note of at most 140 characters
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$")


class SubmissionError(ValueError):
    """Raised when a raster would be rejected by the portal."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_template(template_path: Path) -> dict:
    """Read CRS, transform, shape and band count from the official template raster."""
    with rasterio.open(template_path) as src:
        return {
            "crs": src.crs.to_string() if src.crs else None,
            "epsg": src.crs.to_epsg() if src.crs else None,
            "transform": tuple(src.transform)[:6],
            "height": src.height,
            "width": src.width,
            "res": tuple(src.res),
            "count": src.count,
            "path": str(template_path),
        }


def footprint_mask(template_path: Path) -> np.ndarray:
    """Cells where the official template is finite. The official rules say data outside the
    training-data bounds must be null or NaN, so the template's finite cells define the footprint.
    If the template has no NaN at all, the footprint is the whole grid."""
    with rasterio.open(template_path) as src:
        band = src.read(1)
    return np.isfinite(band)


def check_name_and_note(name: str, note: str) -> None:
    if not NAME_RE.match(name):
        raise SubmissionError(f"name {name!r} must be 3-81 chars of letters, digits, '.', '_' or '-'")
    if len(note) > NOTE_MAX_CHARS:
        raise SubmissionError(f"note is {len(note)} chars; the limit is {NOTE_MAX_CHARS}")
    if any(ord(c) > 126 or ord(c) < 32 for c in note):
        raise SubmissionError("note must be printable ASCII")


def validate_array(values: np.ndarray, template: dict, footprint: np.ndarray | None = None) -> dict:
    """Check an in-memory prediction array against the official format. Raises on failure.

    If `footprint` is given, only in-footprint cells must be finite and in [0, 1].
    Outside the footprint the official text ("data outside the bounds is null or
    nan", problem page 967, fetched 2026-10-11) *permits* null/NaN; it does not
    require it, and a value of 0 is itself a legal confidence in [0, 1].  Both
    byte patterns have been accepted by the portal: of the 26 scored rasters in
    `.cache/scored/`, 18 carry NaN outside the valid region and 8 are all-finite
    with zeros (including the family's best recorded score, 0.2778).  So
    outside-footprint cells are accepted as either NaN or a finite value in
    [0, 1]; inf, the -3.4028e38 sentinel, and finite values outside [0, 1] are
    still refused, and nothing inside the footprint is relaxed.
    """
    problems = []
    if values.ndim != 2:
        problems.append(f"expected a 2-D single-band array, got shape {values.shape}")
    elif (values.shape[0], values.shape[1]) != (template["height"], template["width"]):
        problems.append(
            f"shape {values.shape} does not match template {(template['height'], template['width'])}"
        )
    if values.dtype != np.float32:
        problems.append(f"dtype {values.dtype} must be float32")
    if not problems:
        fp = np.ones(values.shape, dtype=bool) if footprint is None else footprint
        inside = values[fp]
        if not np.all(np.isfinite(inside)):
            problems.append("contains NaN or inf inside the footprint")
        if np.any(inside == np.float32(SENTINEL_NODATA)) or np.any(values == np.float32(SENTINEL_NODATA)):
            problems.append("contains the -3.4028e38 sentinel")
        if np.isinf(values).any():
            problems.append("contains infinite values (inf is refused everywhere)")
        if footprint is not None:
            outside = values[~fp]
            finite_out = np.isfinite(outside)
            if finite_out.any():
                lo_out = float(outside[finite_out].min())
                hi_out = float(outside[finite_out].max())
                if lo_out < 0.0 or hi_out > 1.0:
                    problems.append(
                        f"finite values outside the template footprint are out of [0, 1]: "
                        f"min={lo_out} max={hi_out}")
        if inside.size and (float(inside.min()) < 0.0 or float(inside.max()) > 1.0):
            problems.append(
                f"values must lie in [0, 1]; found min={float(inside.min())} max={float(inside.max())}"
            )
    if problems:
        raise SubmissionError("; ".join(problems))
    fp = np.ones(values.shape, dtype=bool) if footprint is None else footprint
    inside = values[fp].astype(np.float64)
    outside_vals = values[~fp]
    finite_out_mask = np.isfinite(outside_vals)
    stats = {
        "outside_footprint_cells": int(np.count_nonzero(~fp)),
        "outside_footprint_nan_cells": int(np.count_nonzero(~finite_out_mask)),
        "outside_footprint_finite_cells": int(np.count_nonzero(finite_out_mask)),
        "outside_footprint_nonzero_cells": int(
            np.count_nonzero(finite_out_mask & (outside_vals > 0))),
        "outside_footprint_convention": (
            "nan" if not finite_out_mask.any() else
            "zero" if not np.any(finite_out_mask & (outside_vals != 0)) else "mixed"),
        "min": float(inside.min()),
        "max": float(inside.max()),
        "mean": float(inside.mean()),
        "positive_cells": int(np.count_nonzero(inside > 0)),
        "all_zero": bool(np.count_nonzero(inside) == 0),
    }
    return stats


def write_submission(
    values: np.ndarray,
    out_path: Path,
    template_path: Path,
    name: str,
    note: str,
) -> dict:
    """Write a validated single-band float32 GeoTIFF plus a JSON receipt next to it."""
    check_name_and_note(name, note)
    template = read_template(template_path)
    if template["count"] != 1:
        # the template is only used for georeferencing; the output is always single band
        pass
    if template["epsg"] != 32611:
        raise SubmissionError(f"template CRS is {template['crs']}, expected EPSG:32611")
    if abs(template["res"][0]) != 100.0 or abs(template["res"][1]) != 100.0:
        raise SubmissionError(f"template resolution is {template['res']}, expected 100 m")

    values = np.ascontiguousarray(values, dtype=np.float32)
    fp = footprint_mask(template_path)
    # NaN outside the footprint (the official template's own pattern; zeros inside-only
    # arrays are written through unchanged, which the portal also accepts - see validate_array)
    if np.isnan(values[~fp]).all():
        pass
    else:
        values = np.where(fp, values, np.float32(np.nan)).astype(np.float32)
    stats = validate_array(values, template, footprint=fp)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_path) as src:
        crs = src.crs
        transform = Affine(*template["transform"])
    profile = {
        "driver": "GTiff",
        "height": template["height"],
        "width": template["width"],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float("nan") if not fp.all() else None,
        "compress": "deflate",
        "tiled": False,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, "fault_probability")

    receipt = validate_file(out_path, template_path)
    receipt.update({"name": name, "note": note, "note_chars": len(note), "array_stats": stats})
    receipt_path = out_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def validate_file(path: Path, template_path: Path) -> dict:
    """Re-read a GeoTIFF from disk and check it against the official format and the template."""
    path = Path(path)
    template = read_template(template_path)
    with rasterio.open(path) as src:
        problems = []
        if src.count != 1:
            problems.append(f"band count {src.count} must be 1")
        if src.dtypes[0] != "float32":
            problems.append(f"dtype {src.dtypes[0]} must be float32")
        if src.crs is None or src.crs.to_epsg() != 32611:
            problems.append(f"CRS {src.crs} must be EPSG:32611")
        if (src.height, src.width) != (template["height"], template["width"]):
            problems.append(
                f"shape {(src.height, src.width)} != template {(template['height'], template['width'])}"
            )
        if tuple(src.transform)[:6] != tuple(template["transform"]):
            problems.append(f"transform {tuple(src.transform)[:6]} != template {template['transform']}")
        if src.nodata is not None and not math.isnan(src.nodata):
            problems.append(f"nodata tag {src.nodata} is not NaN/None")
        values = src.read(1).astype(np.float32, copy=False)
        fp = footprint_mask(template_path)
        stats = validate_array(values, template, footprint=fp) if not problems else None
    if problems:
        raise SubmissionError("; ".join(problems))
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "crs": "EPSG:32611",
        "shape": [template["height"], template["width"]],
        "transform": list(template["transform"]),
        "dtype": "float32",
        "nodata": "NaN" if stats and stats.get("outside_footprint_nan_cells") else None,
        "validator": "gems_dti.submission.validate_file v2",
        "checks_passed": [
            "single_band",
            "dtype_float32",
            "crs_epsg_32611",
            "shape_matches_template",
            "transform_matches_template",
            "all_finite_in_footprint",
            "no_sentinel",
            "range_0_1",
            "outside_footprint_nan_or_zero",
            "outside_footprint_in_0_1",
        ],
        "stats": stats,
        "all_zero_warning": bool(stats and stats["all_zero"]),
    }
