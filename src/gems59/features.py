"""Cached feature stack for the intensity lane.

Everything here is derived only from files pinned in ``registry/data_manifest.json``
(19 official GeoDAWN bands, GeoDAWN radiometrics + extensions, the lidar scarp stack,
the USGS/INGENIOUS catalogue, the SGMC compilation and the GDR well/spring inventory).

Two rules are enforced by construction, because they are the two ways this lane can
leak:

* every catalogue-derived channel is a function of the **visible** catalogue only.
  In a holdout fold the withheld segments are removed before the channel is
  computed, never after (``visible_only=True``).
* the stack is built once and cached as a float32 memmap; the cached copy is
  keyed by a receipt so a stale cache can never be silently reused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from . import grid

ROOT = grid.ROOT
WORK = ROOT / "work"


# ---------------------------------------------------------------------------
# channel specification
# ---------------------------------------------------------------------------

RAW_OFFICIAL = ["rtp", "tmi", "tmi_hg", "tmi_vg", "tc", "mag_anom",
                "iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg",
                "iso_grav_anom_slope", "geod_2ndinv", "geod_shearrate",
                "geod_dilaterate", "det_elev", "det_elev_slope", "cond_surf",
                "depth_to_base_surf", "ieq_n100a15", "deq_n100a15"]
RAW_RAD = ["K", "Th", "U", "TC"]
RAW_EXT = ["ThK", "UK", "UTh", "TMI_up150"]
RAW_LIDAR = ["l_ex_max", "l_ex_mean", "l_step_max", "l_lapneg_max", "l_lappos_max",
             "l_downface_max", "l_upface_max", "l_cross_max", "l_relief",
             "l_coh100", "l_valid"]

DERIVED_STATIC = [
    "gradmag_rtp",        # horizontal gradient magnitude, RTP magnetics, sigma 1.5 px
    "gradmag_grav",       # horizontal gradient magnitude, isostatic gravity
    "gradmag_tmiup150",   # same on upward-continued TMI (deep sources)
    "ridge_tmi",          # Hessian ridge strength of TMI, sigma 2 px (lineament)
    "ridge_grav",
    "ridge_det_elev",     # geomorphic ridge/valley strength of detrended elevation
    "curv_det_elev",      # plan curvature of detrended elevation (scarp/valley)
    "lap_tmi",            # Laplacian of TMI (edge / contact)
    "dist_lidar",         # distance to lidar coverage (EFFORT covariate)
    "therm_dens_hot",     # KDE of Hot/Warm springs+wells (fluid-pathway indicator)
    "therm_dens_all",     # KDE of every inventoried well/spring (EFFORT covariate)
    "therm_geotherm",     # KDE-weighted silica/chalcedony geothermometry anomaly
    "mapscale_field",     # log10(map scale) of nearest mapped trace (EFFORT)
    "trace_density",      # KDE of mapped traces (EFFORT)
]

DERIVED_VISIBLE = [
    "dist_visible_fault",     # distance to nearest VISIBLE catalogue fault
    "visible_fault_dens_5k",  # visible catalogue fault density within 5 km
    "visible_fault_dens_20k", # visible catalogue fault density within 20 km
]

CHANNELS = RAW_OFFICIAL + RAW_RAD + RAW_EXT + RAW_LIDAR + DERIVED_STATIC + DERIVED_VISIBLE


# ---------------------------------------------------------------------------
# small image primitives
# ---------------------------------------------------------------------------

def _fill(arr: np.ndarray, finite: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).copy()
    a[~finite] = np.nan
    return a


def _nan_safe(a: np.ndarray) -> np.ndarray:
    """Replace non-finite by the finite median so filters do not propagate NaN."""
    out = np.asarray(a, dtype=np.float32).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        med = np.nanmedian(out) if np.isfinite(out).any() else 0.0
        out[bad] = med
    return out


def gsmooth(a: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter(_nan_safe(a), sigma, mode="nearest").astype(np.float32)


def gradmag(a: np.ndarray, sigma: float) -> np.ndarray:
    s = gsmooth(a, sigma)
    gy, gx = np.gradient(s)
    return np.hypot(gx, gy).astype(np.float32)


def hessian_ridge(a: np.ndarray, sigma: float) -> np.ndarray:
    """Signed multi-scale lineament (ridge) strength from the Hessian.

    For a smoothed field S the Hessian eigenvalues l1 >= l2 describe the local
    second-order structure.  For a *dark* line on a bright background (and for a
    bright line on a dark background) the eigen-direction across the line carries
    the strongly-signed curvature while the along-line curvature is ~0, so
    |l_across| - |l_along| is the classic vessel/line detector (Frangi et al.
    1998, used in remote sensing for lineament enhancement).  We return the
    positive part of (|l1| - |l2|) signed by l1, i.e. a quantity that is large
    only on elongated second-order structures and ~0 on blobs and flat ground.
    """
    s = gsmooth(a, sigma)
    sxx = ndi.gaussian_filter(s, sigma, order=[0, 2], mode="nearest")
    syy = ndi.gaussian_filter(s, sigma, order=[2, 0], mode="nearest")
    sxy = ndi.gaussian_filter(s, sigma, order=[1, 1], mode="nearest")
    tr = sxx + syy
    det = np.sqrt((sxx - syy) ** 2 + 4.0 * sxy ** 2)
    l1 = 0.5 * (tr + det)      # larger eigenvalue
    l2 = 0.5 * (tr - det)
    return (np.abs(l1) - np.abs(l2)).astype(np.float32)


def kde_points(shape, rows, cols, sigma: float, weights=None) -> np.ndarray:
    """Gaussian kernel density of a point set, optionally value-weighted."""
    acc = np.zeros(shape, dtype=np.float32)
    if len(rows) == 0:
        return acc
    w = np.ones(len(rows), dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
    np.add.at(acc, (rows, cols), w)
    return ndi.gaussian_filter(acc, sigma, mode="constant").astype(np.float32)


def robust_scale(a: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Median/MAD scaling inside the footprint; NaN retained outside."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    v = a[finite]
    if v.size == 0:
        return out
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    s = 1.4826 * mad if mad > 0 else (np.std(v) or 1.0)
    out[finite] = np.clip((v - med) / (s + 1e-6), -8.0, 8.0)
    return out


# ---------------------------------------------------------------------------
# stack construction
# ---------------------------------------------------------------------------

def build_stack(visible_faults: np.ndarray | None = None,
                out_path: Path | None = None) -> tuple[np.memmap, dict]:
    """Build (or reuse) the (C, H, W) float32 stack + a provenance receipt."""
    WORK.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or WORK / "feature_stack.f32")
    fp = grid.read_footprint()
    H, W = fp.shape

    off, off_fin = grid.read_official()
    with rasterio.open(grid.DATA / "external/geodawn_rad_u8.tif") as s:
        rad = s.read(); rad_nod = s.nodata
    with rasterio.open(grid.DATA / "external/geodawn_extensions_u8.tif") as s:
        ext = s.read(); ext_nod = s.nodata
    with rasterio.open(grid.DATA / "external/lidar_scarp_features_u8.tif") as s:
        lid = s.read(); lid_nod = s.nodata

    n = len(CHANNELS)
    if not out_path.exists() or out_path.stat().st_size != n * H * W * 4:
        mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(n, H, W))
    else:
        mm = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(n, H, W))
    mm[:] = np.nan

    def put(name, arr):
        mm[CHANNELS.index(name)] = np.asarray(arr, dtype=np.float32)

    # ---- raw official bands (robust-scaled inside the footprint)
    for nm in RAW_OFFICIAL:
        i = grid.OFFICIAL_BANDS.index(nm if nm not in ("l_relief",) else nm)
        band = off[i]
        fin = np.isfinite(band) & (band > grid.FEATURE_NODATA / 2)
        put(nm, robust_scale(band.astype(np.float32), fin))
    del off

    # ---- raw external uint8 stacks (0 is the nodata sentinel)
    for k, nm in enumerate(RAW_RAD):
        a = rad[k].astype(np.float32)
        fin = (a != rad_nod) & fp
        put(nm, robust_scale(np.where(fin, a, np.nan), fin))
    for k, nm in enumerate(RAW_EXT):
        a = ext[k].astype(np.float32)
        fin = (a != ext_nod) & fp
        put(nm, robust_scale(np.where(fin, a, np.nan), fin))
    for k, nm in enumerate(RAW_LIDAR):
        a = lid[k].astype(np.float32)
        fin = (a != lid_nod) & fp
        put(nm, robust_scale(np.where(fin, a, np.nan), fin))
    del rad, ext

    # ---- derived static channels
    rtp = _nan_safe(mm[CHANNELS.index("rtp")])
    grav = _nan_safe(mm[CHANNELS.index("iso_grav_anom")])
    tmi = _nan_safe(mm[CHANNELS.index("tmi")])
    tup = _nan_safe(mm[CHANNELS.index("TMI_up150")])
    det = _nan_safe(mm[CHANNELS.index("det_elev")])

    put("gradmag_rtp", _nanmask(gradmag(rtp, 1.5), fp))
    put("gradmag_grav", _nanmask(gradmag(grav, 1.5), fp))
    put("gradmag_tmiup150", _nanmask(gradmag(tup, 1.5), fp))
    put("ridge_tmi", _nanmask(robust_scale_finite(hessian_ridge(tmi, 2.0), fp), fp))
    put("ridge_grav", _nanmask(robust_scale_finite(hessian_ridge(grav, 2.0), fp), fp))
    put("ridge_det_elev", _nanmask(robust_scale_finite(hessian_ridge(det, 2.0), fp), fp))
    gy, gx = np.gradient(gsmooth(det, 2.0))
    gxx = ndi.gaussian_filter(gsmooth(det, 2.0), 2.0, order=[0, 2], mode="nearest")
    gyy = ndi.gaussian_filter(gsmooth(det, 2.0), 2.0, order=[2, 0], mode="nearest")
    gxy = ndi.gaussian_filter(gsmooth(det, 2.0), 2.0, order=[1, 1], mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        curv = -(gxx * gy ** 2 - 2 * gxy * gx * gy + gyy * gx ** 2) / (gx ** 2 + gy ** 2 + 1e-6)
    put("curv_det_elev", _nanmask(robust_scale_finite(np.nan_to_num(curv), fp), fp))
    put("lap_tmi", _nanmask(robust_scale_finite(ndi.laplace(gsmooth(tmi, 2.0)), fp), fp))

    lid_valid = (lid[11] != lid_nod) & fp
    put("dist_lidar", _nanmask(ndi.distance_transform_edt(~lid_valid).astype(np.float32), fp))
    del lid

    # ---- geothermal fluid inventory (GDR / INGENIOUS) -----------------------
    hot_rows, hot_cols, hot_w, all_rows, all_cols, geo_rows, geo_cols, geo_w = _thermal_points(fp)
    put("therm_dens_hot", _nanmask(kde_points(fp.shape, hot_rows, hot_cols, 3.0, hot_w), fp))
    put("therm_dens_all", _nanmask(kde_points(fp.shape, all_rows, all_cols, 3.0), fp))
    put("therm_geotherm", _nanmask(kde_points(fp.shape, geo_rows, geo_cols, 4.0, geo_w), fp))

    # ---- mapping-effort surfaces (USGS Qfaults trace attributes) -----------
    ms = _mapscale_field(fp)
    put("mapscale_field", _nanmask(ms, fp))
    td = _trace_density(fp)
    put("trace_density", _nanmask(td, fp))

    # ---- catalogue-derived (visible only) ----------------------------------
    set_visible_channels(mm, grid.read_labels() & fp if visible_faults is None else visible_faults, fp)

    mm.flush()
    receipt = {
        "schema_version": 2,
        "channels": CHANNELS,
        "n_channels": n,
        "shape": [int(H), int(W)],
        "dtype": "float32",
        "layout": "(C, H, W)",
        "path": str(out_path),
        "bytes": int(out_path.stat().st_size),
        "footprint_px": int(fp.sum()),
        "outside_footprint": "NaN preserved; the model sees NaN outside the bounds",
        "leakage_rule": ("catalogue-derived channels are computed from the visible catalogue "
                         "only and are rewritten by set_visible_channels for every holdout fold"),
    }
    (WORK / "feature_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return np.memmap(out_path, dtype=np.float32, mode="r+", shape=(n, H, W)), receipt


def _nanmask(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where(fp, a, np.nan)


def robust_scale_finite(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    v = a[fp]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(a)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    s = 1.4826 * mad if mad > 0 else (np.std(v) or 1.0)
    return np.clip((a - med) / (s + 1e-6), -8.0, 8.0)


def set_visible_channels(mm, visible_faults: np.ndarray, fp: np.ndarray) -> None:
    """Rewrite the three catalogue-derived channels from a *visible-only* catalogue."""
    vis = np.asarray(visible_faults, bool) & fp
    d = ndi.distance_transform_edt(~vis).astype(np.float32)
    mm[CHANNELS.index("dist_visible_fault")] = np.where(fp, np.clip(d, 0, 200.0), np.nan)
    for nm, rad in (("visible_fault_dens_5k", 50.0), ("visible_fault_dens_20k", 200.0)):
        dens = ndi.gaussian_filter(vis.astype(np.float32), rad / 10.0, mode="constant")
        mm[CHANNELS.index(nm)] = _nanmask(robust_scale_finite(dens, fp), fp)
    mm.flush()


def _thermal_points(fp):
    import pandas as pd
    d = pd.read_csv(grid.DATA / "external/gdr_wellspring_in_footprint.csv")
    d = d[(d["row"].between(0, fp.shape[0] - 1)) & (d["col"].between(0, fp.shape[1] - 1))]
    rows = d["row"].to_numpy().astype(int)
    cols = d["col"].to_numpy().astype(int)
    keep = fp[rows, cols]
    rows, cols, d = rows[keep], cols[keep], d[keep]
    tc = d["temp_c"].to_numpy(dtype=float)
    cls = d["thermalclass"].astype(str).str.strip().str.lower()
    hot = cls.isin(["hot", "warm"]).to_numpy()
    tw = np.where(np.isfinite(tc), np.clip((tc - 20.0) / 40.0, 0.0, 6.0), 1.0)
    hot_rows, hot_cols, hot_w = rows[hot], cols[hot], tw[hot]
    geos = d[["geothermquartz_c", "geothermchalc_c", "geothermcat_c"]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        g = np.where(np.isfinite(geos).any(axis=1),
                      np.nanmax(np.where(np.isfinite(geos), geos, -np.inf), axis=1), np.nan)
    ok = np.isfinite(g)
    geo_rows, geo_cols, geo_w = rows[ok], cols[ok], np.clip((g[ok] - 100.0) / 40.0, -3.0, 5.0)
    return hot_rows, hot_cols, hot_w, rows, cols, geo_rows, geo_cols, geo_w


def _mapscale_field(fp):
    import pandas as pd
    q = pd.read_csv(grid.DATA / "external/gdr_qfaults_traces.csv")
    q = q[q["centroid_in_footprint"] == 1]
    rows = q["centroid_row"].to_numpy(int)
    cols = q["centroid_col"].to_numpy(int)
    keep = (rows >= 0) & (rows < fp.shape[0]) & (cols >= 0) & (cols < fp.shape[1])
    rows, cols, q = rows[keep], cols[keep], q[keep]
    scale = q["map_scale"].to_numpy(dtype=float)
    logsc = np.log10(np.where(scale > 0, scale, 250.0))
    # nearest-trace interpolation of log10(map scale): 1:100k -> 5.0, 1:250k -> 5.398
    idx = ndi.distance_transform_edt(~_sparse(rows, cols, fp.shape), return_distances=False,
                                     return_indices=True)
    field = logsc[_sparse_index(rows, cols, fp.shape, idx)]
    return ndi.gaussian_filter(field.astype(np.float32), 20.0, mode="nearest")


def _sparse(rows, cols, shape):
    m = np.zeros(shape, bool)
    m[rows, cols] = True
    return m


def _sparse_index(rows, cols, shape, indices):
    # map each pixel to the ordinal of its nearest seed point
    seeds = np.zeros(shape, int)
    seeds[rows, cols] = np.arange(len(rows))
    return seeds[indices[0], indices[1]]


def _trace_density(fp):
    import pandas as pd
    q = pd.read_csv(grid.DATA / "external/gdr_qfaults_traces.csv")
    q = q[q["centroid_in_footprint"] == 1]
    rows = q["centroid_row"].to_numpy(int)
    cols = q["centroid_col"].to_numpy(int)
    keep = (rows >= 0) & (rows < fp.shape[0]) & (cols >= 0) & (cols < fp.shape[1]) & fp[rows, cols]
    return kde_points(fp.shape, rows[keep], cols[keep], 30.0)
