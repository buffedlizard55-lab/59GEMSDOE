#!/usr/bin/env python3
"""Write the release GeoTIFF and validate it by re-reading the bytes.

The portal rejects a raster whose values are not inside [0, 1].  A file that
carries NaN outside the bounds is read by the scorer as "data outside the bounds
is null or nan" (which the rules allow) but the portal's range check is applied
to the whole array, so a NaN there makes the upload fail with

    "Predicted values must be in range [0, 1]"

This writer therefore emits **0.0 outside the footprint** (all cells finite) by
default, which satisfies both the range check and the "null outside the bounds"
rule, and it re-reads and re-validates the file it just wrote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59 import grid  # noqa: E402

CANON = {"crs": "EPSG:32611", "width": 3292, "height": 3730,
         "transform": (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def validate(path: Path, footprint: np.ndarray | None = None) -> dict:
    """Every check the portal and the rules impose, run against the bytes on disk."""
    with rasterio.open(path) as src:
        if src.count != 1:
            return {"ok": False, "reason": f"count={src.count}, expected 1"}
        a = src.read(1)
        prof = src.profile
    checks = {}
    checks["single_band"] = src.count == 1
    checks["dtype_float32"] = a.dtype == np.float32
    checks["crs_epsg32611"] = (src.crs is not None and src.crs.to_epsg() == 32611)
    checks["shape_3730x3292"] = a.shape == (3730, 3292)
    checks["transform_matches"] = tuple(float(x) for x in prof["transform"][:6]) == CANON["transform"]
    checks["all_finite"] = bool(np.isfinite(a).all())
    checks["min_ge_0"] = float(np.nanmin(a)) >= 0.0
    checks["max_le_1"] = float(np.nanmax(a)) <= 1.0
    if footprint is not None:
        inside = np.where(footprint, a, 0.0)
        checks["no_nan_inside_footprint"] = bool(np.isfinite(inside).all())
        checks["inside_range_0_1"] = bool(inside.min() >= 0.0 and inside.max() <= 1.0)
        checks["outside_is_zero"] = bool((a[~footprint] == 0).all())
    n_pos = int((a > 0).sum())
    checks["n_positive"] = n_pos
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"ok": ok, "checks": checks, "n_positive": n_pos,
            "bytes": int(Path(path).stat().st_size), "sha256": sha256_file(path),
            "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}


def write(path: Path, values: np.ndarray, footprint: np.ndarray,
          outside: str = "zeros", compress: str = "lzw",
          allow_overwrite: bool = False, receipt: dict | None = None) -> dict:
    path = Path(path)
    if path.exists() and not allow_overwrite:
        raise SystemExit(f"refusing to overwrite an existing release: {path}")
    v = np.asarray(values, dtype=np.float32)
    v = np.where(footprint, np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0), 0.0)
    if outside == "nan":
        v = np.where(footprint, v, np.nan)
    prof = grid.canonical_profile()
    prof["compress"] = compress
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(v, 1)
    rep = validate(path, footprint)
    rep["path"] = str(path)
    rep["outside_footprint_policy"] = outside
    rep["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if receipt:
        rep["build"] = receipt
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=False, help="npy file with the float grid")
    ap.add_argument("--dots", required=False, help="npy boolean dot mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outside", default="zeros", choices=["zeros", "nan"])
    ap.add_argument("--allow-overwrite", action="store_true")
    a = ap.parse_args()
    fp = grid.read_footprint()
    if a.values:
        v = np.load(a.values)
    elif a.dots:
        v = np.load(a.dots).astype(np.float32)
    else:
        raise SystemExit("need --values or --dots")
    rep = write(Path(a.out), v, fp, outside=a.outside, allow_overwrite=a.allow_overwrite)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
