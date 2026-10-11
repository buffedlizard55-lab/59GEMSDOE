"""Session-59 intensity lane (H59-I): presence-only point-process intensity,
mapping-effort bias correction, 1 km new-fault-density zoning, break-even tiling.

Scientific basis (references verified against publisher records; see
``evidence/source_checks.json``)
--------------------------------------------------------------
* Presence-only **point-process intensity**: the catalogue is treated as a
  point pattern; the intensity ``lambda(x)`` is fitted by logistic regression
  on presence/background cells.  MaxEnt and Poisson point-process models are
  equivalent in form (Fithian & Hastie, *Annals of Applied Statistics* 2013).
* **Sample-selection (mapping-effort) bias** is handled per Phillips et al.
  (*Ecological Applications* 2009): the catalogue is a biased sample of the
  fault population, so effort covariates enter the intensity and the
  background stands for where mapping occurred (target-group background).
* The zoning unit is the **1 km block**, which deliberately exceeds the
  ~400 m catalogue registration error quoted in the lane brief (Hermant et
  al., as quoted in the standing competition brief; not independently
  re-verified here).
* Break-even tiling uses the exact marginal-DTI algebra of
  :mod:`gems59.metric` (alpha = 0.2, beta = 0.8, 300 m triangular kernel):
  a dot pays when its realised kernel credit exceeds ``alpha * DTI``.

Fold discipline (parallel-run protocol rule 2)
----------------------------------------------
Every catalogue-derived surface is rebuilt from the **visible** mask of the
holdout draw only.  The geophysical planes come from the hash-pinned 19-band
GeoDAWN stack and are catalogue-independent.  No withheld pixel value, and no
withheld-segment identity, is used to fit, zone or tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression

from .grid import Grid

BLOCK_PX = 10                      # 1 km zoning block at the 100 m grid
SPACINGS = (3.0, 4.0, 6.0)         # candidate lattice spacings (px = 100 m)
DENSITY_CLASSES = 4                # quartiles of block new-fault density
EFFORT_WINDOWS_PX = (10, 50)       # 1 km and 5 km mapping-effort windows
N_BACKGROUND = 120_000             # uniform background cells for the MaxEnt fit
MAX_ITER = 400
C_REG = 1.0

# 1-based band numbers in training_features.tif (verified against the pinned
# 19-band stack; see evidence/feature_bank.json).
BAND = {
    "mag_anom": 1, "rtp": 2, "tmi_hg": 3, "geod_2ndinv": 4,
    "iso_grav_anom_slope": 5, "tc": 6, "geod_shearrate": 7,
    "geod_dilaterate": 8, "tmi_vg": 9, "deq_n100a15": 10,
    "iso_grav_anom_vg": 11, "det_elev": 12, "iso_grav_anom": 13,
    "tmi": 14, "depth_to_base_surf": 15, "ieq_n100a15": 16,
    "cond_surf": 17, "iso_grav_anom_hg": 18, "det_elev_slope": 19,
}

# z = local-contrast (edge) transform, q = global quantile-rank transform.
EDGE_FEATURES = (
    ("g_tmihg_z", BAND["tmi_hg"], "z"),            # magnetic horizontal-gradient ridge
    ("g_tc_z", BAND["tc"], "z"),                   # tilt-derivative edge
    ("g_gravhg_z", BAND["iso_grav_anom_hg"], "z"),  # gravity-gradient edge
    ("g_slope_z", BAND["det_elev_slope"], "z"),    # detrended-elevation scarp
)
REGIME_FEATURES = (
    ("g_tmi_q", BAND["tmi"], "q"),                 # magnetic regime level
    ("g_cond_q", BAND["cond_surf"], "q"),          # conductivity (alteration) regime
    ("g_base_q", BAND["depth_to_base_surf"], "q"), # basement-depth regime
    ("g_strain_q", BAND["geod_2ndinv"], "q"),      # geodetic second invariant
    ("g_deq_q", BAND["deq_n100a15"], "q"),         # earthquake-distance regime
)
EFFORT_FEATURES = ("e_dens1", "e_dens5", "e_relief")
# e_block_id / e_block_edge (derived GeoDAWN acquisition-block proxy) are NOT
# used: IR-59-EFFORT-02 measured that the k-means proxy on windowed magnetic
# statistics is not spatially contiguous (one 5.16 M-cell connected component,
# 369 k boundary cells), so it carries no survey-block information.  The
# official block polygons are on USGS ScienceBase (DOI 10.5066/P93LGLVQ), which
# this sandbox's egress allow-list cannot reach.
FEATURE_NAMES = tuple(n for n, _, _ in EDGE_FEATURES) + tuple(n for n, _, _ in REGIME_FEATURES) \
    + ("g_edge_max",) + EFFORT_FEATURES
GAMMA_CANDIDATES = (0.0, 0.5, 1.0)   # Phillips effort-correction exponent sweep

NODATA_U8 = 255


def _u8_to_unit(a: np.ndarray, fill: float = 0.5) -> np.ndarray:
    """uint8 quantile/contrast rank -> float32 in [0,1]; 255 (nodata) -> fill."""
    out = np.full(a.shape, fill, np.float32)
    ok = a != NODATA_U8
    out[ok] = (a[ok].astype(np.float32) / 254.0)
    return out


def _robust_z(a: np.ndarray, mask: np.ndarray, clip: float = 10.0) -> np.ndarray:
    v = a[mask]
    v = v[np.isfinite(v)]
    out = np.zeros(a.shape, np.float32)
    if v.size == 0:
        return out
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        mad = max(float(np.std(v)), 1e-6)
    z = (a.astype(np.float64) - med) / mad
    np.clip(z, -clip, clip, out=z)
    out[mask] = z[mask].astype(np.float32)
    return out


def load_bank(features_dir: Path):
    """Memmap the cached rank bank (2*19, H, W) uint8 built by prep_feature_stack."""
    return np.load(Path(features_dir) / "bank_u8.npy", mmap_mode="r")


def band_slice(bank, band: int, transform: str) -> np.ndarray:
    """q (even) or z (odd) rank plane for a 1-based band, as float32 [0,1]."""
    idx = 2 * (band - 1) + (0 if transform == "q" else 1)
    return _u8_to_unit(np.asarray(bank[idx]))


# ---------------------------------------------------------------------------
# Effort surfaces (Phillips et al. 2009 sample-selection bias covariates)
# ---------------------------------------------------------------------------

def mapping_effort_surfaces(footprint: np.ndarray, mapped: np.ndarray) -> dict[str, np.ndarray]:
    """Mapping-effort covariates derived from the *visible* mapped-fault mask.

    ``mapped`` must be the holdout-visible mask (or the full catalogue outside
    a holdout).  Trace-length density in 1 km / 5 km windows is the direct
    record of where mapping happened; relief (detrended-elevation rank) is the
    accessibility proxy.
    """
    m = np.asarray(mapped, bool) & np.asarray(footprint, bool)
    out = {}
    for name, r in zip(("e_dens1", "e_dens5"), EFFORT_WINDOWS_PX):
        # separable box filter (fast); the effort surface is a smooth proxy,
        # a disk kernel changes nothing material at these scales
        dens = ndi.uniform_filter(m.astype(np.float32), size=2 * r + 1, mode="constant")
        out[name] = np.log1p(dens * 100.0).astype(np.float32)  # % of window mapped
    return out


def relief_surface(footprint: np.ndarray, det_elev_q: np.ndarray) -> np.ndarray:
    """Relief/accessibility covariate: global quantile rank of detrended elevation."""
    return det_elev_q.astype(np.float32)


def acquisition_block_proxy(footprint: np.ndarray, mag: np.ndarray, *, k: int = 10,
                            window: int = 41, seed: int = 59) -> dict[str, np.ndarray]:
    """Derived GeoDAWN acquisition-block proxy from windowed magnetic statistics.

    The GeoDAWN survey was flown in acquisition blocks with their own line
    spacing, direction and micro-levelling, so block membership shows up as
    discontinuities in the local mean/variance of the magnetic field.  The
    official block polygons live on USGS ScienceBase (DOI 10.5066/P93LGLVQ),
    which this sandbox cannot reach (egress allow-list), so blocks are
    *derived* here by k-means on 4.1 km windowed (mean, sd) of the magnetic
    anomaly.  This is a declared proxy, not the organiser's survey geometry.
    """
    from sklearn.cluster import KMeans

    bad = ~np.isfinite(mag) | (mag <= -3.4e38 / 10.0)
    b = np.where(bad, 0.0, mag).astype(np.float64)
    cnt = ndi.uniform_filter((~bad).astype(np.float64), size=window, mode="constant")
    m1 = ndi.uniform_filter(b, size=window, mode="constant") / np.maximum(cnt, 1e-6)
    m2 = ndi.uniform_filter(b * b, size=window, mode="constant") / np.maximum(cnt, 1e-6)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    ok = np.asarray(footprint, bool) & (cnt > 0.5) & np.isfinite(m1) & np.isfinite(sd)
    feats = np.stack([m1[ok], sd[ok]], axis=1)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(feats)
    labels = np.zeros(mag.shape, np.int32)
    labels[ok] = km.labels_ + 1
    # distance to the nearest block boundary (px), 0 inside a single block
    boundary = np.zeros(mag.shape, bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    d_edge = ndi.distance_transform_edt(~boundary).astype(np.float32)
    d_edge[~ok] = 0.0
    return dict(e_block_id=labels.astype(np.float32),
                e_block_edge=np.log1p(d_edge).astype(np.float32),
                block_boundary=boundary, block_labels=labels)


# ---------------------------------------------------------------------------
# Covariate assembly and the MaxEnt / Poisson point-process fit
# ---------------------------------------------------------------------------

@dataclass
class IntensityFit:
    model: LogisticRegression
    names: tuple
    beta: np.ndarray
    intercept: float
    n_presence: int
    n_background: int


def covariate_planes(grid: Grid, visible: np.ndarray, features_dir: Path) -> dict[str, np.ndarray]:
    """All intensity covariates as full-grid float32 planes.

    ``visible`` is the mapped-fault mask the effort surfaces are built from
    (holdout-visible inside a fold, the full catalogue otherwise).
    """
    fp = grid.footprint
    bank = load_bank(features_dir)
    planes: dict[str, np.ndarray] = {}
    for name, band, tr in EDGE_FEATURES:
        planes[name] = band_slice(bank, band, tr)
    edge_stack = np.stack([planes[n] for n, _, _ in EDGE_FEATURES])
    planes["g_edge_max"] = edge_stack.max(axis=0)
    del edge_stack
    for name, band, tr in REGIME_FEATURES:
        planes[name] = band_slice(bank, band, tr)
    planes.update(mapping_effort_surfaces(fp, visible))
    planes["e_relief"] = relief_surface(fp, band_slice(bank, BAND["det_elev"], "q"))
    del bank
    return planes


def gather(planes: dict[str, np.ndarray], names: tuple, idx: tuple[np.ndarray, np.ndarray],
           block_dummies: int = 0) -> np.ndarray:
    """(n, len(names)+block_dummies) design matrix at the given cell indices."""
    cols = [planes[n][idx] for n in names if n != "e_block_id"]
    X = np.stack(cols, axis=1).astype(np.float32)
    if block_dummies and "e_block_id" in names:
        ids = planes["e_block_id"][idx].astype(np.int64)
        D = np.zeros((X.shape[0], block_dummies), np.float32)
        for j in range(block_dummies):
            D[:, j] = (ids == (j + 2)).astype(np.float32)  # block 1 is the reference
        X = np.hstack([X, D])
    return X


def fit_intensity(planes: dict[str, np.ndarray], presence_idx, footprint: np.ndarray,
                  *, names=FEATURE_NAMES, n_background: int = N_BACKGROUND,
                  seed: int = 59, prior: float | None = None) -> tuple[IntensityFit, np.ndarray]:
    """MaxEnt-form (== Poisson point-process) logistic fit on presence/background.

    Presence cells are the mapped catalogue; background cells are drawn
    uniformly over the footprint, which stands for the region of inference.
    Mapping-effort bias is corrected by the effort covariates themselves
    (Phillips et al. 2009 allow either effort covariates or a target-group
    background; the catalogue *is* the effort record here, so a
    catalogue-density-weighted background would be circular and is not used --
    see evidence/irregularities_intensity.json IR-59-EFFORT-01).

    ``prior`` (the mapped-fault fraction of the footprint) applies the standard
    case-control (priority-score) correction so the returned probabilities are
    calibrated as cell fractions rather than as the artificial 1:2
    presence:background sampling rate; pass ``None`` to skip it.
    """
    rng = np.random.default_rng(seed)
    rows, cols = presence_idx
    pres = np.flatnonzero(footprint.ravel())
    bg = rng.choice(pres, size=min(n_background, pres.size), replace=False)
    brow, bcol = np.unravel_index(bg, footprint.shape)
    Xp = gather(planes, names, (rows, cols))
    Xb = gather(planes, names, (brow, bcol))
    X = np.vstack([Xp, Xb])
    y = np.concatenate([np.ones(len(Xp), np.float32), np.zeros(len(Xb), np.float32)])
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    clf = LogisticRegression(C=C_REG, max_iter=MAX_ITER, solver="lbfgs")
    clf.fit(Xs, y)
    intercept = float(clf.intercept_[0] - float((clf.coef_[0] * mu / sd).sum()))
    if prior is not None and 0.0 < prior < 1.0:
        # case-control correction: logit p_true = logit p_fit + logit(prior) - log(n_p/n_bg)
        intercept += float(np.log(prior / (1.0 - prior)) - np.log(len(Xp) / len(Xb)))
    fit = IntensityFit(model=clf, names=tuple(names),
                       beta=(clf.coef_[0] / sd).astype(np.float64),
                       intercept=intercept,
                       n_presence=int(len(Xp)), n_background=int(len(Xb)))
    return fit, np.stack([mu, sd])


def effort_correct(lam: np.ndarray, planes: dict[str, np.ndarray], gamma: float,
                   footprint: np.ndarray) -> np.ndarray:
    """Phillips-style mapping-effort bias correction of the fitted intensity.

    ``lam_true(x) = lam_obs(x) / ehat(x)**gamma`` with ``ehat`` the 1 km-window
    mapping-density covariate normalised to unit mean inside the footprint, so
    the correction moves probability mass from well-mapped to under-mapped
    terrain while preserving the total.  ``gamma = 0`` is the uncorrected
    control.
    """
    if gamma == 0.0:
        return lam
    e = planes["e_dens1"].astype(np.float64)
    vals = e[footprint]
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    ehat = np.clip((e - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    ehat = 0.05 + 0.95 * ehat
    ehat *= float(vals.mean() / max(ehat[footprint].mean(), 1e-9))  # unit mean in footprint
    return (lam / np.power(ehat, gamma)).astype(np.float32)


def sgmc_diagnostic(n_b: np.ndarray, sgmc_only: np.ndarray, footprint: np.ndarray,
                    *, block: int = BLOCK_PX, n_deciles: int = 10) -> dict:
    """Rank agreement between predicted new-fault density and independent
    off-catalogue fault pixels (SGMC-minus-catalogue), by decile of prediction.

    ``n_b`` is the per-block array from :func:`zone_new_fault_density` (block
    sums, NOT the per-cell expansion -- summing the expansion would count
    every cell's block total once per cell).

    Diagnostic only: the SGMC compilation is a different, also-biased mapping
    of the same terrain, and IR-57-INSTR-01 measured that proxy coverage cannot
    rank placements.  This tests whether the zoning points at *any* real
    off-catalogue faults, nothing more.
    """
    fp_block = block_reduce(np.asarray(footprint, np.float32), block) > 0
    pred = np.where(fp_block.ravel(), np.asarray(n_b, np.float32).ravel(), 0.0)
    obs = block_reduce(np.where(footprint, sgmc_only, 0.0).astype(np.float32), block).ravel()
    keep = (pred > 0) | (obs > 0)
    pred, obs = pred[keep], obs[keep]
    if pred.size < n_deciles or pred.std() == 0:
        return {"note": "degenerate diagnostic", "n_blocks": int(pred.size)}
    edges = np.quantile(pred, np.linspace(0, 1, n_deciles + 1))
    edges[0] -= 1e-9
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_deciles - 1)
    rows = []
    for d in range(n_deciles):
        m = idx == d
        rows.append(dict(decile=d + 1, n_blocks=int(m.sum()),
                         mean_pred=float(pred[m].mean()), mean_sgmc_only=float(obs[m].mean()),
                         total_sgmc_only=float(obs[m].sum())))
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred, obs).statistic)
    return dict(n_blocks=int(pred.size), spearman_pred_vs_sgmc_only=rho, deciles=rows,
                note="HOLDOUT-STRUCTURE diagnostic; not a score; SGMC proxy bias per IR-57-INSTR-01")


def predict_logit(planes: dict[str, np.ndarray], names, beta, intercept,
                  footprint: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """lambda(x) = sigmoid(x.beta + intercept) over the whole footprint."""
    logit = np.full(footprint.shape, intercept, np.float32)
    names_c = [n for n in names if n != "e_block_id"]
    flat_idx = np.flatnonzero(footprint.ravel())
    for start in range(0, flat_idx.size, chunk):
        sel = flat_idx[start:start + chunk]
        r, c = np.unravel_index(sel, footprint.shape)
        X = gather(planes, names_c, (r, c))
        logit[r, c] = X.astype(np.float64) @ beta + intercept
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# 1 km zoning by new-fault density, and break-even tiling
# ---------------------------------------------------------------------------

def block_reduce(a: np.ndarray, block: int = BLOCK_PX) -> np.ndarray:
    """Sum-pool to blocks; trailing partial blocks are dropped."""
    h, w = a.shape
    hh, ww = (h // block) * block, (w // block) * block
    return a[:hh, :ww].reshape(hh // block, block, ww // block, block).sum(axis=(1, 3))


def block_expand(b: np.ndarray, shape: tuple, block: int = BLOCK_PX) -> np.ndarray:
    hh, ww = b.shape[0] * block, b.shape[1] * block
    out = np.zeros(shape, np.float32)
    out[:hh, :ww] = np.repeat(np.repeat(b, block, axis=0), block, axis=1)
    return out


def zone_new_fault_density(lam: np.ndarray, known: np.ndarray,
                           block: int = BLOCK_PX) -> tuple[np.ndarray, np.ndarray]:
    """Expected new-fault pixels per 1 km block.

    ``new_b = sum_{x in b, x not mapped} lambda(x)``: the catalogue cells of a
    block are treated as already-mapped (completeness), so only the unmapped
    part of a block contributes to its new-fault density.  Returns the
    per-block array (block sums) and the per-cell expanded copy (each cell
    carries its block's total; use the per-block array for any aggregation).
    """
    eligible = ~np.asarray(known, bool)
    contrib = np.where(eligible, lam, 0.0).astype(np.float32)
    n_b = block_reduce(contrib, block)
    return n_b, block_expand(n_b, lam.shape, block)


def block_anchors(lam: np.ndarray, allowed: np.ndarray,
                  block: int = BLOCK_PX) -> np.ndarray:
    """(n_by, n_bx, 2) argmax-lambda allowed cell per block (-1 if none)."""
    h, w = lam.shape
    hh, ww = (h // block) * block, (w // block) * block
    cand = np.where(allowed[:hh, :ww], lam[:hh, :ww], -np.inf)
    cb = cand.reshape(hh // block, block, ww // block, block)
    anchors = np.full((hh // block, ww // block, 2), -1, np.int32)
    for by in range(hh // block):
        for bx in range(ww // block):
            sub = cb[by, :, bx, :]
            if not np.isfinite(sub).any():
                continue
            ay, ax = np.unravel_index(int(np.argmax(sub)), sub.shape)
            anchors[by, bx] = (ay, ax)
    return anchors


def tile_blocks(n_b_cell: np.ndarray, lam: np.ndarray, allowed: np.ndarray,
                spacing: float, *, block: int = BLOCK_PX,
                classes: np.ndarray | None = None,
                anchors: np.ndarray | None = None) -> np.ndarray:
    """Lattice-tile every block whose density clears the class table.

    ``classes`` maps block id -> density class (0 = below break-even, not
    tiled).  The lattice of each tiled block is anchored at the block's
    highest-lambda allowed cell, with the given spacing in pixels.
    Fixes IR-59-LATTICE-MOD10: uses non-wrapping offsets ``ay + k*s`` inside
    ``[0, block)`` rather than modulo-``block`` wraparound (which placed
    adjacent wrapped points 1 px apart when ``s = 3, block = 10``).
    """
    dots = np.zeros(lam.shape, bool)
    h, w = lam.shape
    hh, ww = (h // block) * block, (w // block) * block
    if classes is None:
        return dots
    if anchors is None:
        anchors = block_anchors(lam, allowed, block)
    s = int(round(spacing))
    if s < 1:
        raise ValueError("spacing must be >= 1 px")
    ks = np.arange(-block, block + 1) * s
    for by in range(hh // block):
        row_cls = classes[by]
        if not (row_cls > 0).any():
            continue
        for bx in range(ww // block):
            if int(row_cls[bx]) <= 0:
                continue
            r0, c0 = by * block, bx * block
            ay, ax = int(anchors[by, bx, 0]), int(anchors[by, bx, 1])
            if ay < 0:
                continue
            sub_ok = allowed[r0:r0 + block, c0:c0 + block]
            ys = ay + ks[(ay + ks >= 0) & (ay + ks < block)]
            xs = ax + ks[(ax + ks >= 0) & (ax + ks < block)]
            gy, gx = np.meshgrid(ys, xs, indexing="ij")
            sel = sub_ok[gy, gx]
            dots[r0:r0 + block, c0:c0 + block][gy[sel], gx[sel]] = True
    return dots


def density_classes(n_b: np.ndarray, *, n_classes: int = DENSITY_CLASSES,
                    min_density: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Assign each block a density class 1..n_classes by quantile of n_b.

    Blocks at or below ``min_density`` (the break-even floor) get class 0.
    Returns (class array, the quantile edges).
    """
    vals = n_b[n_b > min_density]
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1)[1:-1]) if vals.size else np.array([])
    cls = np.digitize(n_b, edges, right=True) + 1
    cls[n_b <= min_density] = 0
    return cls.astype(np.int32), edges


# ---------------------------------------------------------------------------
# Session H59-I2: Leave-One-Component-Out (LOCO) Two-Host Presence-Only IPP,
# USGS/INGENIOUS QFaults effort join (IR-59-EFFORT-02), and cross-block
# corridor-support lattice tiling (IR-59-LATTICE-MOD10).
# ---------------------------------------------------------------------------

LOCO_GEO_NAMES = (
    tuple(f"g_{n}" for n, _, _ in EDGE_FEATURES)
    + ("g_max_edge",)
    + tuple(f"g_{n}" for n, _, _ in REGIME_FEATURES)
    + ("g_scarp_rel_cos2",)
)
LOCO_EFFORT_NAMES = (
    "e_dens_5km_loco",
    "e_dens_10km_loco",
    "e_scarp_relief",
    "e_fcode_mod",
    "e_fcode_inf",
    "e_scale_250k",
)
LOCO_STRUCT_NAMES = (
    "s_log_d1",
    "s_log_d1_sq",
    "s_log_dperp",
    "s_log_dperp_sq",
    "s_log_dpar",
    "s_log_len1",
    "s_log_len2",
    "s_coh1",
    "s_turn1",
    "s_log_d2",
    "s_relay_ratio",
    "s_relay_facing",
    "s_relay_int",
    "s_d1_x_len1",
    "s_strike_cos2",
)
LOCO_FEATURE_NAMES = LOCO_GEO_NAMES + LOCO_EFFORT_NAMES + LOCO_STRUCT_NAMES
LOCO_EFFORT_SLICE = slice(len(LOCO_GEO_NAMES), len(LOCO_GEO_NAMES) + len(LOCO_EFFORT_NAMES))
LOCO_LINEAR_INDICES = tuple(
    i for i, n in enumerate(LOCO_FEATURE_NAMES)
    if n not in {"g_scarp_rel_cos2", "s_log_d1_sq", "s_log_dperp_sq", "s_turn1", "s_relay_int", "s_d1_x_len1"}
)


def _triangular_kernel_300m() -> np.ndarray:
    yy, xx = np.mgrid[-3:4, -3:4]
    return np.maximum(0.0, 1.0 - np.hypot(yy, xx) / 3.0).astype(np.float32)


def kernel_expected_intensity(lam: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Convolve unmapped point-process intensity with the 300 m (3 px) triangular kernel."""
    return ndi.convolve(np.where(eligible, lam, 0.0).astype(np.float32),
                        _triangular_kernel_300m(), mode="constant")


def tile_blocks_corridor(
    lam: np.ndarray,
    eligible: np.ndarray,
    classes: np.ndarray,
    spacing_map: dict[int, float | None],
    *,
    cell_quantile: float = 0.10,
    block: int = BLOCK_PX,
    score: np.ndarray | None = None,
) -> np.ndarray:
    """Tile 1 km blocks above break-even along their kernel-intensity corridor.

    Blocks with ``classes[by, bx] <= 0`` (or mapped to ``None`` in ``spacing_map``)
    receive zero dots. Surviving blocks are visited in descending order of block
    peak expected kernel credit ``(K_tri * lam)``, and candidate cells inside each
    block are placed greedily at minimum Euclidean spacing ``s(class)`` (in pixels),
    enforced both within and across 1 km block boundaries.
    """
    dots = np.zeros(lam.shape, bool)
    elig = np.asarray(eligible, bool)
    if score is None:
        score = kernel_expected_intensity(lam, elig)
    hh, ww = classes.shape
    tiled_cell = block_expand((classes > 0).astype(np.float32), lam.shape, block) > 0
    active = score[elig & tiled_cell]
    floor = float(np.quantile(active, cell_quantile)) if (active.size and cell_quantile > 0.0) else 0.0

    block_order = []
    for by in range(hh):
        row_c = classes[by]
        if not (row_c > 0).any():
            continue
        r0 = by * block
        for bx in range(ww):
            c = int(row_c[bx])
            if c <= 0 or c not in spacing_map or spacing_map[c] is None:
                continue
            c0 = bx * block
            m = float(score[r0:r0 + block, c0:c0 + block].max())
            if m >= floor and m > 0.0:
                block_order.append((m, by, bx, c))
    block_order.sort(reverse=True)

    h, w = lam.shape
    blocked = np.zeros(lam.shape, bool)
    offsets: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for sp in set(v for v in spacing_map.values() if v is not None):
        sp_f = float(sp)
        r = int(np.ceil(sp_f))
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        m = (oy * oy + ox * ox) < (sp_f * sp_f - 1e-6)
        offsets[sp_f] = (oy[m], ox[m])

    for _, by, bx, c in block_order:
        sp_f = float(spacing_map[c])  # type: ignore[arg-type]
        oy, ox = offsets[sp_f]
        r0, c0 = by * block, bx * block
        sub_ok = elig[r0:r0 + block, c0:c0 + block]
        if not sub_ok.any():
            continue
        sub_s = np.where(sub_ok, score[r0:r0 + block, c0:c0 + block], -1.0)
        flat = np.argsort(sub_s.ravel())[::-1]
        for idx in flat:
            val = float(sub_s.ravel()[idx])
            if val < floor or val <= 0.0:
                break
            ly, lx = divmod(int(idx), block)
            gy, gx = r0 + ly, c0 + lx
            if blocked[gy, gx]:
                continue
            dots[gy, gx] = True
            ny = np.clip(gy + oy, 0, h - 1)
            nx = np.clip(gx + ox, 0, w - 1)
            blocked[ny, nx] = True
    return dots


def qfaults_effort_rasters(
    grid: Grid,
    visible: np.ndarray,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize USGS/INGENIOUS FCODE2023 and MAPSCALE onto visible fault pixels.

    Resolves IR-59-EFFORT-02: ``record_id`` in ``data/external/trace_segments_utm11.csv``
    is the 0-based row index of ``data/external/qfault_attributes.csv``.
    """
    import csv
    from .faultzone import _csv_pixels

    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    attr_path = base / "data/external/qfault_attributes.csv"
    seg_path = base / "data/external/trace_segments_utm11.csv"
    with attr_path.open() as f:
        attrs = list(csv.DictReader(f))
    tr, points = _csv_pixels(seg_path, grid.shape, grid.transform)
    fcode_raw = np.zeros(grid.shape, np.int8)
    mscale_raw = np.zeros(grid.shape, np.int16)
    for (rr, cc), rid in zip(points, tr["record_id"].values):
        a = attrs[int(rid)]
        if a.get("FCODE2023"):
            fcode_raw[rr, cc] = int(a["FCODE2023"])
        if a.get("MAPSCALE"):
            mscale_raw[rr, cc] = int(a["MAPSCALE"])
    vis = np.asarray(visible, bool)
    fcode_vis = np.where(vis, ndi.maximum_filter(fcode_raw, size=3), 0).astype(np.int8)
    mscale_vis = np.where(vis, ndi.maximum_filter(mscale_raw, size=3), 0).astype(np.int16)
    return fcode_vis, mscale_vis


def loco_structural_frame(
    grid: Grid,
    visible: np.ndarray,
    root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Build Leave-One-Component-Out (LOCO) two-host structural & effort frame.

    For off-catalogue pixels ``x not in visible``, ``(d1, iy1, ix1, c1)`` and
    ``(d2, iy2, ix2, c2)`` come directly from ``two_host_frame(visible)``.
    For presence pixels ``x in visible`` (belonging to component ``C1(x)``),
    ``(d1, iy1, ix1, c1)`` is replaced in-place with the nearest *other*
    visible component ``C2(x) != C1(x)``, and ``(d2, iy2, ix2, c2)`` is replaced
    with the second-nearest *other* visible component ``C3(x) not in {C1, C2}``
    found via ``cKDTree`` over ``visible``.  Regional effort densities
    ``e_dens5`` and ``e_dens10`` subtract the self-component pixel contribution
    ``min(|C1(x)|, L)`` on ``visible`` so no feature leaks self-component identity.
    """
    from scipy.spatial import cKDTree
    from .relay_bend_anatomy import two_host_frame, host_bend_fields, cached_scarp_geometry

    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    vis = np.asarray(visible, bool)
    fcode_vis, mscale_vis = qfaults_effort_rasters(grid, vis, base)

    th = two_host_frame(vis)
    d1, iy1, ix1, c1 = th["d1"], th["iy1"], th["ix1"], th["c1"]
    d2, iy2, ix2, c2 = th["d2"], th["iy2"], th["ix2"], th["c2"]
    comp_len = th["comp_len"]
    del th

    vy, vx = np.nonzero(vis)
    vc1 = c1[vy, vx].copy()
    vc2 = c2[vy, vx].copy()
    tree = cKDTree(np.column_stack([vy, vx]))
    dists, idxs = tree.query(np.column_stack([vy, vx]), k=min(256, len(vy)), workers=2)
    cnbr = vc1[idxs]
    m3 = (cnbr != vc1[:, None]) & (cnbr != vc2[:, None])
    has3 = m3.any(axis=1)
    i3 = np.argmax(m3, axis=1)
    row_idx = np.arange(len(vy))
    vd3 = np.where(has3, dists[row_idx, i3], d2[vy, vx] * 1.5).astype(np.float32)
    viy3 = np.where(has3, vy[idxs[row_idx, i3]], iy2[vy, vx]).astype(np.int32)
    vix3 = np.where(has3, vx[idxs[row_idx, i3]], ix2[vy, vx]).astype(np.int32)
    vc3 = np.where(has3, cnbr[row_idx, i3], vc2).astype(np.int32)
    del tree, dists, idxs, cnbr, m3, has3, i3

    raw_count5 = ndi.uniform_filter(vis.astype(np.float32), size=51, mode="constant") * (51.0 * 51.0)
    self_in_51 = np.zeros(grid.shape, np.float32)
    self_in_51[vy, vx] = np.minimum(comp_len[vc1], 51.0)
    e_dens5 = np.log1p(np.maximum(0.0, raw_count5 - self_in_51) * (100.0 / (51.0 * 51.0))).astype(np.float32)
    del raw_count5, self_in_51

    raw_count10 = ndi.uniform_filter(vis.astype(np.float32), size=101, mode="constant") * (101.0 * 101.0)
    self_in_101 = np.zeros(grid.shape, np.float32)
    self_in_101[vy, vx] = np.minimum(comp_len[vc1], 101.0)
    e_dens10 = np.log1p(np.maximum(0.0, raw_count10 - self_in_101) * (100.0 / (101.0 * 101.0))).astype(np.float32)
    del raw_count10, self_in_101, vc1, vc2

    d1[vy, vx] = d2[vy, vx]
    iy1[vy, vx] = iy2[vy, vx]
    ix1[vy, vx] = ix2[vy, vx]
    c1[vy, vx] = c2[vy, vx]
    d2[vy, vx] = vd3
    iy2[vy, vx] = viy3
    ix2[vy, vx] = vix3
    c2[vy, vx] = vc3
    del vd3, viy3, vix3, vc3

    hb = host_bend_fields(vis)
    strike_vis = np.where(np.isfinite(hb["strike_fine"]), hb["strike_fine"], 0.0).astype(np.float32)
    coh_vis = np.where(np.isfinite(hb["coh_fine"]), np.clip(hb["coh_fine"], 0.0, 1.0), 0.0).astype(np.float32)
    turn_vis = hb["turn"].astype(np.float32)
    del hb

    scarp = cached_scarp_geometry(base, grid.footprint)
    return dict(
        d1=d1, iy1=iy1, ix1=ix1, c1=c1,
        d2=d2, iy2=iy2, ix2=ix2, c2=c2,
        comp_len=comp_len,
        strike_vis=strike_vis, coh_vis=coh_vis, turn_vis=turn_vis,
        fcode_vis=fcode_vis, mscale_vis=mscale_vis,
        e_dens5=e_dens5, e_dens10=e_dens10,
        scarp_cos2=scarp["cos2"], scarp_sin2=scarp["sin2"], scarp_coh=scarp["coherence"],
    )


def extract_loco_features(
    frame: dict[str, np.ndarray],
    bank,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Extract the 31 LOCO geophysical, effort, and structural covariates at ``(rows, cols)``."""
    from .network import offset_components
    g_cols = []
    edges = []
    for _, bidx, tr_type in EDGE_FEATURES:
        p = 2 * (bidx - 1) + (0 if tr_type == "q" else 1)
        u = bank[p, rows, cols]
        v = np.where(u != 255, u.astype(np.float32) / 254.0, 0.5)
        g_cols.append(v)
        edges.append(v)
    g_cols.append(np.maximum.reduce(edges))
    for _, bidx, tr_type in REGIME_FEATURES:
        p = 2 * (bidx - 1) + (0 if tr_type == "q" else 1)
        u = bank[p, rows, cols]
        v = np.where(u != 255, u.astype(np.float32) / 254.0, 0.5)
        g_cols.append(v)

    ay1 = frame["iy1"][rows, cols]
    ax1 = frame["ix1"][rows, cols]
    ay2 = frame["iy2"][rows, cols]
    ax2 = frame["ix2"][rows, cols]
    s1 = frame["strike_vis"][ay1, ax1]
    s2 = frame["strike_vis"][ay2, ax2]
    th2 = np.radians(2.0 * s1)
    sc_rel = frame["scarp_coh"][rows, cols] * (
        frame["scarp_cos2"][rows, cols] * np.cos(th2)
        + frame["scarp_sin2"][rows, cols] * np.sin(th2)
    )
    g_cols.append(sc_rel.astype(np.float32))

    u_rel = bank[2 * (BAND["det_elev"] - 1), rows, cols]
    e_rel = np.where(u_rel != 255, u_rel.astype(np.float32) / 254.0, 0.5)
    fc_h = frame["fcode_vis"][ay1, ax1]
    ms_h = frame["mscale_vis"][ay1, ax1]
    e_cols = [
        frame["e_dens5"][rows, cols],
        frame["e_dens10"][rows, cols],
        e_rel,
        (fc_h == 2).astype(np.float32),
        (fc_h >= 3).astype(np.float32),
        (ms_h >= 250).astype(np.float32),
    ]

    dd1 = frame["d1"][rows, cols]
    dd2 = frame["d2"][rows, cols]
    dy1 = (rows - ay1).astype(np.float32)
    dx1 = (cols - ax1).astype(np.float32)
    d_par1, d_perp1, _ = offset_components(dy1, dx1, s1)
    dy2 = (rows - ay2).astype(np.float32)
    dx2 = (cols - ax2).astype(np.float32)
    n1 = np.maximum(dd1, 1e-3)
    n2 = np.maximum(dd2, 1e-3)
    rfacing = np.clip(-(dy1 * dy2 + dx1 * dx2) / (n1 * n2), -1.0, 1.0).astype(np.float32)
    rratio = (dd1 / np.maximum(dd1 + dd2, 1e-3)).astype(np.float32)
    ld1 = np.log1p(dd1)
    ldp = np.log1p(d_perp1)
    lda = np.log1p(np.abs(d_par1))
    ll1 = np.log1p(frame["comp_len"][frame["c1"][rows, cols]])
    ll2 = np.log1p(frame["comp_len"][frame["c2"][rows, cols]])
    ld2 = np.log1p(dd2)
    s_cols = [
        ld1, ld1 * ld1,
        ldp, ldp * ldp,
        lda,
        ll1, ll2,
        frame["coh_vis"][ay1, ax1],
        frame["turn_vis"][ay1, ax1],
        ld2,
        rratio,
        rfacing,
        rfacing * rratio,
        ld1 * ll1,
        np.cos(np.radians(2.0 * (s1 - s2))).astype(np.float32),
    ]
    return np.column_stack(g_cols + e_cols + s_cols).astype(np.float32)


def fit_loco_intensity(
    frame: dict[str, np.ndarray],
    bank,
    pres_mask: np.ndarray,
    tgb_mask: np.ndarray,
    *,
    col_indices: tuple[int, ...] | None = None,
    n_bg: int = 120_000,
    seed: int = 59,
    c_reg: float = 1.0,
) -> dict:
    """Fit target-group presence-only MaxEnt/Poisson IPP via case-control logistic GLM."""
    prow, pcol = np.nonzero(pres_mask)
    rng = np.random.default_rng(seed)
    bg_all = np.flatnonzero(tgb_mask.ravel())
    bg_pick = rng.choice(bg_all, size=min(n_bg, bg_all.size), replace=False)
    brow, bcol = np.unravel_index(bg_pick, pres_mask.shape)

    Xp = extract_loco_features(frame, bank, prow, pcol)
    Xb = extract_loco_features(frame, bank, brow, bcol)
    if col_indices is not None:
        Xp = Xp[:, col_indices]
        Xb = Xb[:, col_indices]
        names = tuple(LOCO_FEATURE_NAMES[i] for i in col_indices)
    else:
        names = LOCO_FEATURE_NAMES

    X = np.vstack([Xp, Xb])
    y = np.concatenate([np.ones(len(Xp), np.float32), np.zeros(len(Xb), np.float32)])
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    clf = LogisticRegression(C=c_reg, max_iter=400, solver="lbfgs").fit((X - mu) / sd, y)
    beta_raw = (clf.coef_[0] / sd).astype(np.float64)
    prior_tgb = float(pres_mask.sum() / max(int(tgb_mask.sum()), 1))
    sample_ratio = len(Xp) / max(len(Xb), 1)
    prior_shift = float(np.log(prior_tgb / (1.0 - prior_tgb)) - np.log(sample_ratio))
    intercept_raw = float(clf.intercept_[0] - float((clf.coef_[0] * mu / sd).sum())) + prior_shift
    eff_idx = [i for i, n in enumerate(names) if n.startswith("e_")]
    eff_ref = Xp[:, eff_idx].mean(axis=0).astype(np.float64) if eff_idx else np.zeros(0, np.float64)
    return dict(
        names=names,
        col_indices=col_indices,
        beta=beta_raw,
        intercept=intercept_raw,
        std_coef={n: float(c) for n, c in zip(names, clf.coef_[0])},
        eff_idx=eff_idx,
        eff_ref=eff_ref,
        n_pos=int(len(Xp)),
        n_bg=int(len(Xb)),
        prior_tgb=prior_tgb,
    )


def predict_loco_intensity(
    frame: dict[str, np.ndarray],
    bank,
    model: dict,
    eval_mask: np.ndarray,
    *,
    gamma: float = 0.0,
    chunk: int = 250_000,
) -> np.ndarray:
    """Predict presence-only intensity ``lambda(x)`` on ``eval_mask`` with effort discount ``gamma``."""
    beta = model["beta"].copy()
    intercept = float(model["intercept"])
    eff_idx = model["eff_idx"]
    if eff_idx and gamma != 0.0:
        beta[eff_idx] *= (1.0 - float(gamma))
        intercept += float(gamma) * float(np.dot(model["beta"][eff_idx], model["eff_ref"]))
    col_indices = model["col_indices"]
    c_rows, c_cols = np.nonzero(eval_mask)
    lam = np.zeros(eval_mask.shape, np.float32)
    for start in range(0, len(c_rows), chunk):
        sl = slice(start, min(start + chunk, len(c_rows)))
        Xc = extract_loco_features(frame, bank, c_rows[sl], c_cols[sl])
        if col_indices is not None:
            Xc = Xc[:, col_indices]
        logit = np.clip(Xc.astype(np.float64) @ beta + intercept, -40.0, 40.0)
        lam[c_rows[sl], c_cols[sl]] = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    return lam


def decile_calibration_blocks(
    n_b: np.ndarray,
    truth: np.ndarray,
    footprint: np.ndarray,
) -> dict:
    """1 km block decile calibration of predicted new-fault density vs withheld truth."""
    from scipy.stats import rankdata

    fp_b = block_reduce(footprint.astype(np.float32)) > 0
    truth_b = block_reduce(truth.astype(np.float32))
    comp_truth, _ = ndi.label(truth, structure=np.ones((3, 3), int))
    p = n_b[fp_b]
    o = truth_b[fp_b]
    ranks = (rankdata(p, method="ordinal") - 1) / max(len(p), 1)
    dec = np.clip((ranks * 10).astype(int), 0, 9)
    rows = []
    for d in range(10):
        m = dec == d
        mask_grid = np.zeros_like(fp_b)
        mask_grid[fp_b] = m
        cell_m = block_expand(mask_grid.astype(np.float32), footprint.shape) > 0
        segs = set(np.unique(comp_truth[cell_m & truth])) - {0}
        rows.append(dict(
            decile=d + 1,
            n_blocks=int(m.sum()),
            pred_sum=round(float(p[m].sum()), 2),
            mean_pred_new_fault_px=round(float(p[m].mean()), 5),
            sum_withheld_px=int(round(float(o[m].sum()))),
            mean_withheld_px=round(float(o[m].mean()), 5),
            withheld_segments=int(len(segs)),
        ))
    tot_w = float(o.sum())
    return dict(
        evidence_class="HOLDOUT-STRUCTURE (calibration, not a score)",
        n_footprint_blocks=int(fp_b.sum()),
        total_withheld_pixels=int(round(tot_w)),
        share_withheld_top1_decile=round(float(o[dec == 9].sum() / max(tot_w, 1.0)), 5),
        share_withheld_top2_deciles=round(float(o[dec >= 8].sum() / max(tot_w, 1.0)), 5),
        share_withheld_top3_deciles=round(float(o[dec >= 7].sum() / max(tot_w, 1.0)), 5),
        monotone_withheld_pixels=bool(all(rows[i]["sum_withheld_px"] <= rows[i + 1]["sum_withheld_px"] for i in range(9))),
        rows=rows,
    )

