"""Grid helpers: canonical raster I/O for the DOE GEMS (DrivenData #306) lane.

Every reader here is byte-verified against the pinned files in
``registry/data_manifest.json``.  The canonical grid is fixed by
``data/sample_submission.tif``: EPSG:32611, 100 m, 3730 x 3292,
transform (100, 0, 243350, 0, -100, 4508550).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

FEATURE_NODATA = -3.4028234663852886e38

# 19 official float32 bands of ``training_features.tif``, in file order.
OFFICIAL_BANDS = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi",
    "depth_to_base_surf", "ieq_n100a15", "cond_surf", "iso_grav_anom_hg",
    "det_elev_slope",
]

# external uint8 stacks, in file order
RAD_BANDS = ["K", "Th", "U", "TC"]
EXT_BANDS = ["ThK", "UK", "UTh", "TMI_up150"]
LIDAR_BANDS = ["ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
               "downface_max", "upface_max", "cross_max", "relief", "coh100",
               "strike", "valid"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_profile() -> dict:
    """The submission grid, read from the organizer's own sample submission."""
    with rasterio.open(DATA / "sample_submission.tif") as src:
        p = src.profile
    return {"driver": "GTiff", "dtype": "float32", "count": 1, "crs": p["crs"],
            "transform": p["transform"], "width": p["width"], "height": p["height"],
            "nodata": float("nan")}


def read_footprint() -> np.ndarray:
    """True where the organizer's sample submission is finite (inside the bounds)."""
    with rasterio.open(DATA / "sample_submission.tif") as src:
        a = src.read(1)
    return np.isfinite(a)


def read_labels() -> np.ndarray:
    """Existing (published) fault catalogue rasterised: True where label == 1."""
    with rasterio.open(DATA / "labels.tif") as src:
        a = src.read(1)
    return a == 1


def read_stack(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (bands, finite_mask) for a multi-band raster, honouring its nodata tag."""
    with rasterio.open(path) as src:
        arr = src.read()
        nod = src.nodata
    if arr.dtype == np.uint8:
        finite = np.ones(arr.shape[1:], bool) if nod is None else (arr[0] != nod)
        if nod is not None:
            for i in range(arr.shape[0]):
                finite &= (arr[i] != nod)
        return arr, finite
    finite = np.isfinite(arr)
    if nod is not None:
        finite &= arr != nod
    return arr, finite.all(axis=0)


def read_official() -> tuple[np.ndarray, np.ndarray]:
    """19-band float32 official feature stack -> (bands[H,W], finite[H,W])."""
    return read_stack(DATA / "training_features.tif")


def write_geotiff(path: Path, values: np.ndarray, profile: dict | None = None) -> Path:
    """Write a single-band float32 GeoTIFF on the canonical grid."""
    prof = profile or canonical_profile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.asarray(values, dtype=np.float32), 1)
    return path


def rowcol_to_xy(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = canonical_profile()["transform"]
    return t.c * cols + t.f * rows + t.x_off, t.a * cols + t.e * rows + t.y_off


def xy_to_rowcol(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = canonical_profile()["transform"]
    inv = ~t
    cols_f, rows_f = inv * (xs, ys)
    return np.rint(rows_f).astype(np.int64), np.rint(cols_f).astype(np.int64)
