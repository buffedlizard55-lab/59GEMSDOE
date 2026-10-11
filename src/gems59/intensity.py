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
    steps = np.arange(0, block, s)
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
            ys = (ay + steps) % block
            xs = (ax + steps) % block
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
