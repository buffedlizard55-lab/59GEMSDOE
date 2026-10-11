"""Covariate planes for the presence-only fault-intensity lane (59GEMSDOE).

Lane: presence-only point-process intensity fitted to the mapped catalogue,
mapping-effort bias corrected with a *target-group background* (Phillips,
Dudik & Schapire 2009, Ecological Applications 19(1):181-197,
doi:10.1890/08-2025.1), zoned at 1 km, then tiled only where the lattice clears
break-even with a spacing taken from a utility table re-derived on our own
holdout.  Presence-only/PPP equivalence (Fithian & Hastie 2013, Annals of
Applied Statistics 7(4):1917-1939, doi:10.1214/13-AOAS667).

Three covariate families, all on the competition grid (EPSG:32611, 100 m):

**Evidence (E)** - where faults are, physically.  Multi-scale analytic-signal
amplitude, tilt angle, theta map and total horizontal derivative of the two
potential fields in the provided stack, plus the stack's own derivatives and
the seismicity / conductivity / basement layers.  Operators cited:

* Nabighian, M. N. (1972), Geophysics 37(3):507-517, doi:10.1190/1.1440276
  -- analytic signal of 2-D magnetic bodies.
* Roest, W. R., Verhoef, J., Pilkington, M. (1992), Geophysics 57(1):116-125,
  doi:10.1190/1.1443174 -- 3-D analytic-signal amplitude.
* Miller, H. G., Singh, V. (1994), J. Applied Geophysics 32(2-3):213-217,
  doi:10.1016/0926-9851(94)90022-1 -- tilt angle.
* Verduzco, B. et al. (2004), The Leading Edge 23(2):116-119,
  doi:10.1190/1.1651454 -- total horizontal derivative.
* Wijns, C., Perez, C., Kowalczyk, P. (2005), Geophysics 70(4):L39-L43,
  doi:10.1190/1.1988184 -- theta map.

**Structure (S)** - proximity to the *visible* mapped catalogue.  Derived only
from visible cells, so a hide-and-recover split cannot leak withheld geometry.

**Effort (M)** - mapping effort / observer bias.  Reliability-of-location class
(FCODE2023: 1 = well constrained, 2 = moderately, 3 = inferred, 4 = undefined)
and mapped scale (MAPSCALE) from the official USGS Quaternary Fault and Fold
Database attribute table as compiled by INGENIOUS (field definitions file
``qfaults_8_README_fielddefinitions_qfaults_ingenious_nad83conus117_2023-06-27.txt``),
plus acquisition-block coverage read from the feature stack's own nodata
sentinel.  Distance to roads is *not* obtainable from this sandbox (see
``docs/irregularities.md`` IR-59-EFFORT-01) and is replaced by the relief
exposure proxy, whose sign the fit is free to choose.

Memory policy: 19 bands x 12.28 M px x 4 B = 933 MB will not fit alongside the
other work, so every plane is written to ``.cache/cov`` and re-read lazily by
name.  No function here holds more than one plane plus its working copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "cov"
SENTINEL = np.float32(-3.4028234663852886e38)

B = dict(mag_anom=1, rtp=2, tmi_hg=3, geod_2ndinv=4, grav_slope=5, tc=6,
         geod_shear=7, geod_dilat=8, tmi_vg=9, deq=10, grav_vg=11, det_elev=12,
         grav=13, tmi=14, depth_base=15, ieq=16, cond_surf=17, grav_hg=18,
         det_elev_slope=19)

#: E + S + M keys used by the fitted model, in fixed order (the model is
#: identified by this list; changing it changes the receipt hash).
COVARIATES = [
    "as_rtp_1.5", "as_rtp_3.0", "as_rtp_6.0",
    "theta_rtp_3.0", "tilt_rtp_3.0",
    "thd_grav_3.0", "thd_grav_8.0",
    "band_tc", "band_det_elev_slope", "band_ieq", "neg_deq",
    "band_cond_surf", "band_depth_base", "band_geod_2ndinv",
    "relief_10",
    "neglog_dist_catalogue", "neglog_dist_tips", "cat_density_1km",
    "neglog_dist_sliprated", "sliprate_max_1km", "acq_block",
]

#: The effort family (target-group-background covariates).  Reported separately
#: because the lane's whole point is that these explain *observer* behaviour.
EFFORT = ["neglog_dist_sliprated", "sliprate_max_1km", "acq_block",
          "cat_density_1km", "relief_10"]
#: The structural family (proximity to the visible mapped catalogue).
STRUCTURE = ["neglog_dist_catalogue", "neglog_dist_tips", "cat_density_1km"]


def load(name: str, cache: Path = CACHE) -> np.ndarray:
    """Load one cached plane by name (float32, competition grid)."""
    return np.load(Path(cache) / f"{name}.npy", mmap_mode=None)


def names(cache: Path = CACHE) -> list[str]:
    return sorted(p.stem for p in Path(cache).glob("*.npy"))


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.array(a, np.float32, copy=True)
    a[a <= SENTINEL] = np.nan
    bad = ~np.isfinite(a)
    if bad.any():
        idx = ndi.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
        a = a[tuple(idx)]
    return a


def vertical_derivative(u: np.ndarray) -> np.ndarray:
    """Vertical derivative of a harmonic field by 2-D wavenumber multiplication.

    For a potential field measured on a plane, d/dz in the Fourier domain is
    multiplication by the radial wavenumber ``|k|`` (upward continuation
    derivative).  This is the standard exact operator; it is why we do not have
    to substitute the stack's ambiguous ``tc`` band ("tilt angle *or* total
    curvature" in its own tag) for the tilt computation.
    """
    fy = np.fft.fftfreq(u.shape[0])[:, None]
    fx = np.fft.rfftfreq(u.shape[1])[None, :]
    k = (2.0 * np.pi * np.hypot(fx, fy)).astype(np.float32)
    spec = np.fft.rfft2(np.asarray(u, np.float32), axes=(0, 1))
    return np.fft.irfft2(spec * k, s=u.shape, axes=(0, 1)).astype(np.float32)


def analytic_signal(u: np.ndarray, sigma: float) -> np.ndarray:
    """AS amplitude ``hypot(THDR, VDR)`` of a smoothed field (edge locator)."""
    s = ndi.gaussian_filter(u, sigma, mode="nearest")
    gy, gx = np.gradient(s)
    return np.hypot(np.hypot(gx, gy), vertical_derivative(s))


def total_horizontal_derivative(u: np.ndarray, sigma: float) -> np.ndarray:
    gy, gx = np.gradient(ndi.gaussian_filter(u, sigma, mode="nearest"))
    return np.hypot(gx, gy)


def tilt_angle(u: np.ndarray, sigma: float) -> np.ndarray:
    """Tilt angle ``atan2(VDR, THDR)`` in [-pi/2, pi/2] (Miller & Singh 1994)."""
    s = ndi.gaussian_filter(u, sigma, mode="nearest")
    gy, gx = np.gradient(s)
    thdr = np.hypot(gx, gy)
    vdr = vertical_derivative(s)
    return np.arctan2(vdr, thdr + np.float32(1e-30)).astype(np.float32)


def theta_map(u: np.ndarray, sigma: float) -> np.ndarray:
    """Theta map ``arccos(THDR / AS)`` (Wijns et al. 2005)."""
    s = ndi.gaussian_filter(u, sigma, mode="nearest")
    gy, gx = np.gradient(s)
    thdr = np.hypot(gx, gy)
    asb = np.hypot(thdr, vertical_derivative(s))
    return np.arccos(np.clip(thdr / (asb + 1e-30), -1.0, 1.0)).astype(np.float32)


def _disk(radius_px: int) -> np.ndarray:
    r = int(radius_px)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def local_relief(elev: np.ndarray, radius_px: int = 5) -> np.ndarray:
    """Half the max-min elevation range in a 1 km-diameter window (radius 5 px).

    Deliberately morphological, not a Gaussian difference: a linear smoother is
    sign-symmetric, so ``smooth(-z) == -smooth(z)`` and a "range" built from two
    Gaussian passes is identically zero (that bug was caught by measurement, not
    by review -- see IR-59-COV-01).
    """
    fp = _disk(radius_px)
    hi = ndi.maximum_filter(elev, footprint=fp, mode="nearest")
    lo = ndi.minimum_filter(elev, footprint=fp, mode="nearest")
    return (0.5 * (hi - lo)).astype(np.float32)


def build_evidence(feature_path: Path, *, cache: Path = CACHE,
                   force: bool = False) -> dict:
    """Build and cache every E plane from the 19-band stack, one band at a time."""
    import rasterio

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    wanted = [k for k in COVARIATES if k.startswith(("as_", "tilt_", "theta_",
                                                     "thd_", "band_", "neg_deq",
                                                     "relief_"))]
    have = {p.stem for p in cache.glob("*.npy")}
    missing = [k for k in wanted if k not in have]
    written = []
    if not missing and not force and "acq_block" in have:
        return {"built": [], "reused": sorted(wanted), "cache": str(cache)}
    with rasterio.open(feature_path) as src:
        shape = src.shape

        def band(name: str) -> np.ndarray:
            return _clean(src.read(B[name]))

        if "acq_block" not in have or force:
            # Validity must be measured on the RAW bytes: _clean() fills nodata
            # by nearest neighbour, so testing finiteness after the fill returns
            # True everywhere (IR-59-COV-01).
            valid = np.ones(shape, bool)
            for n in range(1, src.count + 1):
                raw = src.read(n)
                valid &= np.isfinite(raw) & (raw > SENTINEL)
            np.save(cache / "acq_block.npy", valid.astype(np.float32))
            written.append("acq_block")

        for name, fn in (
            ("as_rtp_1.5", lambda: analytic_signal(band("rtp"), 1.5)),
            ("as_rtp_3.0", lambda: analytic_signal(band("rtp"), 3.0)),
            ("as_rtp_6.0", lambda: analytic_signal(band("rtp"), 6.0)),
            ("theta_rtp_3.0", lambda: theta_map(band("rtp"), 3.0)),
            ("tilt_rtp_3.0", lambda: tilt_angle(band("rtp"), 3.0)),
            ("thd_grav_3.0", lambda: total_horizontal_derivative(band("grav"), 3.0)),
            ("thd_grav_8.0", lambda: total_horizontal_derivative(band("grav"), 8.0)),
            ("band_tc", lambda: band("tc")),
            ("band_det_elev_slope", lambda: band("det_elev_slope")),
            ("band_ieq", lambda: band("ieq")),
            ("neg_deq", lambda: -band("deq")),
            ("band_cond_surf", lambda: band("cond_surf")),
            ("band_depth_base", lambda: band("depth_base")),
            ("band_geod_2ndinv", lambda: band("geod_2ndinv")),
            ("relief_10", lambda: local_relief(band("det_elev"), 5)),
        ):
            if (cache / f"{name}.npy").exists() and not force:
                continue
            np.save(cache / f"{name}.npy", np.asarray(fn(), np.float32))
            written.append(name)
    return {"built": written, "reused": sorted(wanted), "cache": str(cache)}


# --------------------------------------------------------------------------
# Structure planes: from the *visible* catalogue only.
# --------------------------------------------------------------------------
def tip_pixels(catalogue: np.ndarray) -> np.ndarray:
    """Trace end cells: catalogue cells with <= 2 catalogue cells in their 3x3."""
    counts = ndi.convolve(catalogue.astype(np.uint8), np.ones((3, 3), np.uint8),
                          mode="constant")
    return catalogue & (counts <= 2)


def structure_planes(visible: np.ndarray, block: int = 10,
                     cache: Path | None = None, tag: str = "") -> dict:
    """Distance / tip-distance / 1 km-density planes from the visible catalogue.

    ``cache``/``tag`` allow holdout folds to be cached independently; without a
    tag the planes are returned in memory only.
    """
    if cache is not None and tag:
        p = Path(cache) / f"struct_{tag}"
        p.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, fn in (("neglog_dist_catalogue",
                          lambda: -np.log1p(ndi.distance_transform_edt(~visible))),
                         ("neglog_dist_tips",
                          lambda: -np.log1p(ndi.distance_transform_edt(~tip_pixels(visible)))),
                         ("cat_density_1km",
                          lambda: ndi.uniform_filter(visible.astype(np.float32),
                                                     size=block) * block * block)):
            f = p / f"{name}.npy"
            if not f.exists():
                np.save(f, np.asarray(fn(), np.float32))
            out[name] = np.load(f)
        return out
    return {
        "neglog_dist_catalogue": -np.log1p(
            ndi.distance_transform_edt(~visible)).astype(np.float32),
        "neglog_dist_tips": -np.log1p(
            ndi.distance_transform_edt(~tip_pixels(visible))).astype(np.float32),
        "cat_density_1km": (ndi.uniform_filter(visible.astype(np.float32),
                                               size=block) * block * block).astype(np.float32),
    }


# --------------------------------------------------------------------------
# Effort planes: reliability of location, mapped scale, acquisition block.
# --------------------------------------------------------------------------
def rasterise_segments(segments: list[dict], shape: tuple[int, int],
                       transform) -> dict[str, np.ndarray]:
    """Rasterise QFaults/INGENIOUS segment attributes onto the competition grid.

    Each segment paints its attribute along a 100 m Bresenham step between its
    two endpoints.  Where segments disagree the conservative (numerically
    larger, i.e. less reliable) FCODE wins.
    """
    fcode = np.full(shape, np.nan, np.float32)
    mapscale = np.full(shape, np.nan, np.float32)
    inv = ~transform
    h, w = shape
    for seg in segments:
        c0 = inv * (seg["x0"], seg["y0"])
        c1 = inv * (seg["x1"], seg["y1"])
        col0, row0 = int(np.floor(c0[0])), int(np.floor(c0[1]))
        col1, row1 = int(np.floor(c1[0])), int(np.floor(c1[1]))
        steps = max(abs(col1 - col0), abs(row1 - row0), 1)
        f, s = seg.get("fcode"), seg.get("mapscale")
        for i in range(steps + 1):
            t = i / steps
            c = int(round(col0 + t * (col1 - col0)))
            r = int(round(row0 + t * (row1 - row0)))
            if not (0 <= r < h and 0 <= c < w):
                continue
            if f is not None and np.isfinite(f):
                cur = fcode[r, c]
                fcode[r, c] = f if not np.isfinite(cur) else max(cur, f)
            if s is not None and np.isfinite(s):
                cur = mapscale[r, c]
                mapscale[r, c] = s if not np.isfinite(cur) else min(cur, s)
    return {"fcode": fcode, "mapscale": mapscale}


def effort_planes(acquisition: np.ndarray) -> dict:
    """Effort planes that are actually available from verified bytes.

    Only ``acq_block`` ships: the acquisition coverage read from the feature
    stack's own nodata sentinel (where the airborne survey has data at all).
    The FCODE2023 reliability class and MAPSCALE are in the official attribute
    table but cannot be placed on the grid without a join key (IR-59-EFFORT-02),
    and no placeholder plane is emitted in their stead.
    """
    return {"acq_block": np.asarray(acquisition, np.float32)}


def trace_planes(segments: list[dict], shape: tuple[int, int], transform,
                 cache: Path | None = None) -> dict:
    """Effort/evidence planes from the official trace attribute table.

    ``data/external/trace_segments_utm11.csv`` (INGENIOUS compilation of the
    Quaternary Fault and Fold Database, EPSG:32611 coordinates, one row per
    trace segment, fields ``slip_mm_yr`` and ``sense``) is the only official
    attribute table in this checkout carrying **both** geometry and attributes.
    It supplies

    * ``neglog_dist_sliprated`` -- distance to the nearest trace segment with a
      recorded slip rate;
    * ``sliprate_max_1km`` -- maximum recorded slip rate within a 1 km window.

    A recorded slip rate is a proxy for how well the fault is characterised in
    the official compilation, which is the effort notion the lane asks for under
    "USGS reliability-of-location class".  FCODE2023 itself could **not** be
    joined: the attribute table keys on ``NUM`` (feature id) while the geometry
    table keys on an anonymous ``record_id``, and the two share no column
    (IR-59-EFFORT-02).  No join is invented here.

    Fully vectorised: every segment's sample points are built with numpy and
    scattered in one pass, because a per-segment Python loop over 84,331 rows is
    minutes of wall time for no benefit.
    """
    h, w = shape
    inv = ~transform
    xy0 = np.array([[inv * (seg["x0"], seg["y0"]), inv * (seg["x1"], seg["y1"])]
                    for seg in segments], np.float64)          # (n, 2, 2) col,row
    col0 = np.floor(xy0[:, 0, 0]).astype(np.int64)
    row0 = np.floor(xy0[:, 0, 1]).astype(np.int64)
    col1 = np.floor(xy0[:, 1, 0]).astype(np.int64)
    row1 = np.floor(xy0[:, 1, 1]).astype(np.int64)
    slip = np.array([float(seg.get("slip_mm_yr") or 0.0) for seg in segments], np.float64)
    slip[~np.isfinite(slip)] = 0.0

    steps = np.maximum.reduce([np.abs(col1 - col0), np.abs(row1 - row0),
                               np.ones_like(col0)])
    counts = steps + 1
    total = int(counts.sum())
    seg_id = np.repeat(np.arange(len(segments), dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    offs = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    t = offs / np.repeat(steps, counts)
    rr = np.round(row0[seg_id] + t * (row1 - row0)[seg_id]).astype(np.int64)
    cc = np.round(col0[seg_id] + t * (col1 - col0)[seg_id]).astype(np.int64)
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc, ss = rr[ok], cc[ok], slip[seg_id][ok]
    flat = rr * w + cc
    slip_raster = np.zeros(h * w, np.float32)
    np.maximum.at(slip_raster, flat, ss.astype(np.float32))
    slip_raster = slip_raster.reshape(h, w)
    rated = slip_raster > 0
    d_rated = ndi.distance_transform_edt(~rated).astype(np.float32)
    local_max = ndi.maximum_filter(slip_raster, footprint=_disk(5), mode="nearest")
    planes = {
        "neglog_dist_sliprated": (-np.log1p(d_rated)).astype(np.float32),
        "sliprate_max_1km": np.log1p(local_max).astype(np.float32),
        "trace_sample_points": np.int64(total),
        "trace_rated_pixels": np.int64(rated.sum()),
    }
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        for k, val in planes.items():
            if k.startswith("trace_"):
                continue
            np.save(Path(cache) / f"{k}.npy", np.asarray(val, np.float32))
    return planes


def to_rank(array: np.ndarray, *, n_bins: int | None = None) -> np.ndarray:
    """Global percentile rank (monotone, leak-free, unit-free) -> float32 [0,1]."""
    from scipy.stats import rankdata

    a = np.asarray(array, np.float32)
    finite = np.isfinite(a)
    out = np.zeros(a.shape, np.float32)
    if finite.any():
        v = a[finite]
        r = rankdata(v, method="average") / max(v.size, 1)
        out[finite] = r.astype(np.float32)
    return out
