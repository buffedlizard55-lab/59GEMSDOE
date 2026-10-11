"""Hide-and-recover holdout, spatially blocked (the shared instrument).

Protocol fixed by the lane brief:

* withhold **whole fault segments** -- 8-connected components of the catalogue
  raster, never individual pixels;
* a **buffer** (30 px) around every withheld component is removed from training
  so the model cannot read the immediate neighbourhood of what it must predict;
* every **catalogue-derived feature channel is recomputed from the visible
  catalogue only**, before any fitting;
* the **visible catalogue is masked pixel-exactly** from the emission support, so
  no dot can be placed on a fault the fold was allowed to see;
* scoring is **pooled** across folds with the organizer's metric
  (alpha 0.2, beta 0.8, 300 m triangular kernel).

Honesty note, carried into every report: the truth used here is the *published
catalogue*, which is precisely the population that is NOT in the competition's
test set.  This instrument measures generalisation to unseen fault geometry; it
cannot measure discovery of the off-catalogue population the prize scores.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import features, metric


def segments(mask: np.ndarray, min_size: int = 1) -> tuple[np.ndarray, int]:
    """8-connected components of a fault mask (whole fault segments)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if min_size > 1:
        sizes = np.bincount(lab.ravel())
        drop = np.where(sizes < min_size)[0]
        for d in drop:
            lab[lab == d] = 0
        lab, n = ndi.label(lab > 0, structure=np.ones((3, 3), int))
    return lab, n


def assign_blocks(lab: np.ndarray, shape, nx: int = 2, ny: int = 2,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Assign each segment to one of ``nx*ny`` spatial blocks by its centroid.

    Blocks are the quadrants of the grid, so a fold's training data and its
    withheld segments are separated in space, not at random.
    """
    n = int(lab.max())
    cents = np.array(ndi.center_of_mass(lab > 0, lab, range(1, n + 1)))
    H, W = shape
    bx = np.minimum((cents[:, 0] / (H / ny)).astype(int), ny - 1)
    by = np.minimum((cents[:, 1] / (W / nx)).astype(int), nx - 1)
    return bx * nx + by


def fold_masks(lab: np.ndarray, blocks: np.ndarray, k: int, footprint: np.ndarray,
               buffer_px: int = 30):
    """Return (withheld, visible, train_area) for fold ``k``."""
    withheld_ids = np.where(blocks == k)[0] + 1
    withheld = np.isin(lab, withheld_ids) & footprint
    visible = (lab > 0) & footprint & ~withheld
    buf = ndi.binary_dilation(withheld, structure=np.ones((3, 3), int),
                              iterations=buffer_px)
    train_area = footprint & ~withheld & ~buf
    return withheld, visible, train_area


def emission_support(footprint: np.ndarray, visible: np.ndarray, flank_px: int = 2):
    """Pixels where a dot may be placed: footprint minus the visible catalogue
    minus a flank buffer around it (dots on mapped faults cannot score)."""
    sup = footprint & ~visible
    if flank_px > 0:
        sup &= ~ndi.binary_dilation(visible, structure=np.ones((3, 3), int),
                                    iterations=flank_px)
    return sup


def pooled_dti(preds: list[np.ndarray], truths: list[np.ndarray]) -> dict:
    """The organizer's metric applied to all folds pooled into one raster pair."""
    P = np.zeros_like(preds[0], dtype=np.float32)
    T = np.zeros(truths[0].shape, bool)
    for p, t in zip(preds, truths):
        P = np.maximum(P, p)
        T |= t
    t = metric.dti_terms(P, T)
    return {**t, "dti": metric.dti(P, T), "n_truth_total": int(T.sum()),
            "n_pred_total": int((P > 0).sum())}


def bootstrap_ci(preds, truths, n_boot: int = 400, seed: int = 20261011,
                 block: int = 64) -> tuple[float, float]:
    """95 % CI for pooled DTI by a spatial block bootstrap over the truth pixels.

    Resampling whole 6.4 km tiles (rather than pixels) respects the strong
    spatial autocorrelation of fault pixels, which a pixel bootstrap would ignore
    and which would make the interval far too narrow.
    """
    P = np.zeros_like(preds[0], dtype=np.float32)
    T = np.zeros(truths[0].shape, bool)
    for p, t in zip(preds, truths):
        P = np.maximum(P, p)
        T |= t
    H, W = T.shape
    # hoisted out of the bootstrap loop: these are the two full-grid dilations
    # and they do not depend on the resample.
    M_full = metric.kernel_max(P)
    fp_full = float((P * (1.0 - metric.kernel_max(T.astype(np.float32)))).sum())
    nbh, nbw = int(np.ceil(H / block)), int(np.ceil(W / block))
    gr = np.minimum(np.arange(H)[:, None] // block, nbh - 1) * nbw
    gc = np.arange(W)[None, :] // block
    tile = (gr + gc)
    rows, cols = np.nonzero(T)
    tid = tile[rows, cols]
    uniq, inv = np.unique(tid, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.size).astype(float)
    rng = np.random.default_rng(seed)
    base = metric.dti(P, T)
    vals = []
    m_full = M_full[rows, cols].astype(np.float64)
    for _ in range(n_boot):
        mult = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size))
        wts = (mult / counts)[inv]      # each truth pixel weighted by its tile's draw count
        tp = float((wts * m_full).sum())
        fn = float((wts * (1.0 - m_full)).sum())
        vals.append(tp / (tp + metric.ALPHA * fp_full + metric.BETA * fn + metric.EPS))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float(base)


def assign_blocks_random(lab: np.ndarray, n_folds: int = 4, seed: int = 7) -> np.ndarray:
    """Assign whole segments to folds at random (segment-level, spatially scattered).

    Used for the deployment-matched holdout: the deployed model sees the full
    published catalogue and is asked for faults that are *not* in it, so the
    withheld segments must be removed while the rest of the catalogue stays
    visible everywhere.  A quadrant block (``assign_blocks``) instead removes
    every catalogue pixel from a whole quarter of the map, which is a much
    harder and less representative problem; both are reported.
    """
    n = int(lab.max())
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n)
