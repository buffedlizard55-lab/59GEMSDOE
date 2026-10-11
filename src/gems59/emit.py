"""Zone-then-tile emission: 1 km density zoning, break-even gating, spacing by utility.

The organizer's metric prices every emitted pixel against the kernel weight it
earns, and ``metric.breakeven_delta`` shows the bar is exactly ``alpha * DTI``.
That makes the emission problem a *fractional program* over the emitted set S:

    maximise   sum_{x in S} Delta(x)          (Delta = kernel-smoothed truth density)
    subject to the |S| penalty,  optimum  <=>  emit x iff  Delta(x) >= alpha * DTI*

so the only free design choices are (a) how good Delta is, (b) the self-consistent
value of DTI*, and (c) how the dots are spaced, which is *not* free: two dots
that both sit within 300 m of the same truth pixel earn the metric only once
(TP_w takes a max over emitted pixels), so crowding is pure loss.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import metric


# ---------------------------------------------------------------------------
# the kernel-smoothed credit field
# ---------------------------------------------------------------------------

def credit_field(q: np.ndarray) -> np.ndarray:
    """Delta(x) = sum_{y} q(y) * k(|x - y|): expected kernel weight of a dot at x.

    ``q`` is the (unnormalised) predicted density of truth pixels.  This is a
    *sum* convolution with the triangular kernel, not a max: for sparse truth the
    expected value of ``max_g k(d) * 1{g is truth}`` equals the sum of the
    per-pixel expectations.
    """
    ker = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            d = np.hypot(i - 2, j - 2)
            ker[i, j] = max(0.0, 1.0 - d / metric.RADIUS_PX)
    return ndi.convolve(np.asarray(q, np.float32), ker, mode="constant", cval=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1 km zoning
# ---------------------------------------------------------------------------

def block_sum(a: np.ndarray, block: int = 10) -> np.ndarray:
    """Sum of ``a`` over block x block tiles; the trailing partial tiles are dropped."""
    H, W = a.shape
    h, w = H // block * block, W // block * block
    return a[:h, :w].reshape(h // block, block, w // block, block).sum(axis=(1, 3))


def block_index_map(shape, block: int = 10) -> np.ndarray:
    """Per-pixel index of its 1 km block (-1 outside the aligned area)."""
    H, W = shape
    nb = (H // block, W // block)
    idx = np.arange(nb[0] * nb[1]).reshape(nb)
    out = np.full(shape, -1, dtype=np.int32)
    out[:nb[0] * block, :nb[1] * block] = np.repeat(np.repeat(idx, block, axis=0), block, axis=1)
    return out


def zone_by_density(delta: np.ndarray, block: int = 10,
                    edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """1 km block sums of Delta, and the per-pixel block id."""
    bs = block_sum(delta, block)
    return bs, block_index_map(delta.shape, block)


# ---------------------------------------------------------------------------
# spacing / placement
# ---------------------------------------------------------------------------

def _disk_offsets(radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(np.ceil(radius_px))
    dr, dc = [], []
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if np.hypot(i, j) <= radius_px + 1e-9:
                dr.append(i)
                dc.append(j)
    return np.asarray(dr), np.asarray(dc)


def nonmax_peaks(delta: np.ndarray, half: int) -> np.ndarray:
    """Pixels that are the strict maximum of Delta inside a (2*half+1)^2 window.

    A square window is used because ``ndi.maximum_filter(size=...)`` is separable
    and therefore fast; a pixel surviving it is at Chebyshev distance >= half
    from every other survivor, hence at Euclidean distance >= half, so the
    minimum-separation contract holds (conservatively at the window corners).
    """
    if half <= 0:
        return np.ones(delta.shape, bool)
    mx = ndi.maximum_filter(delta, size=2 * half + 1, mode="constant", cval=-np.inf)
    return delta >= mx          # ties kept; the greedy pass below breaks them


def place_dots(delta: np.ndarray, support: np.ndarray, tau: float,
               spacing_map: np.ndarray | None = None, block_ids: np.ndarray | None = None,
               block_ok: np.ndarray | None = None, spacing_by_block: np.ndarray | None = None,
               max_dots: int = 400_000, nms_half: int | None = None) -> np.ndarray:
    """Greedy emission: highest Delta first, honouring a per-pixel minimum spacing.

    ``spacing_map`` gives, per pixel, the minimum separation enforced between dots
    (derived from the 1 km block's predicted density, see ``pipeline.spacing_from_density``);
    a block with ``block_ok[b] == False`` is not tiled at all because it does not
    clear break-even.  Returns a boolean dot mask.

    Implementation: separable non-maximum suppression first reduces the candidate
    set to at most one pixel per spacing window, then a short greedy sweep over
    that reduced set stamps the exact inhibition disks.  This is O(N) in numpy
    instead of a Python loop over every above-threshold pixel.
    """
    H, W = delta.shape
    cand = support & np.isfinite(delta) & (delta >= tau)
    if block_ok is not None and block_ids is not None:
        cand &= block_ok[np.clip(block_ids, 0, None)]
    if not cand.any():
        return np.zeros((H, W), bool)

    sep = (np.full((H, W), 3.0, dtype=np.float32) if spacing_map is None
           else np.asarray(spacing_map, np.float32))
    # one maximum_filter pass per distinct window size actually requested
    # (a constant spacing map costs one pass, not one per integer up to the max)
    # ``nms_half`` decouples the pre-filter window from the inhibition radius: the
    # window only decides how many local maxima are offered to the greedy pass,
    # while the disk radius (``sep``) is the actual minimum-separation contract.
    if nms_half is not None:
        halfs = np.array([int(nms_half)])
    else:
        halfs = np.unique(np.ceil(sep[cand])).astype(int) if cand.any() else np.array([3])
    peaks = np.zeros((H, W), bool)
    sub_all = np.where(cand, delta, -np.inf)
    for h in halfs:
        h = int(max(h, 1))
        if nms_half is None:
            # with the default window every pixel is offered at the first window
            # size that covers its requested separation
            want = cand & (sep <= h + 1e-6)
            if not want.any():
                continue
            sub = np.where(want, sub_all, -np.inf)
        else:
            sub = sub_all
        peaks |= nonmax_peaks(sub, h)
    peaks &= cand
    if not peaks.any():
        return np.zeros((H, W), bool)

    # greedy over the surviving peaks only (a few 10^4 of them)
    pr, pc = np.nonzero(peaks)
    order = np.lexsort((pc, pr, -delta[pr, pc]))
    pr, pc = pr[order], pc[order]
    placed = np.zeros((H, W), bool)
    taken = np.zeros((H, W), bool)
    n = 0
    for r, c in zip(pr, pc):
        if n >= max_dots:
            break
        if taken[r, c]:
            continue
        placed[r, c] = True
        n += 1
        rad = float(sep[r, c])
        if rad >= 1.0:
            dr, dc = _disk_offsets(rad)
            rr, ccx = r + dr, c + dc
            ok = (rr >= 0) & (rr < H) & (ccx >= 0) & (ccx < W)
            taken[rr[ok], ccx[ok]] = True
    return placed


def lattice_in_block(delta: np.ndarray, support: np.ndarray, tau: float,
                     block: int = 10, spacing: float = 3.0,
                     block_ok: np.ndarray | None = None) -> np.ndarray:
    """Reference emitter: a literal square lattice inside every block that clears
    break-even.  Kept so the greedy placement can be compared against the plain
    lattice the lane paragraph describes."""
    H, W = delta.shape
    out = np.zeros((H, W), bool)
    nbh, nbw = H // block, W // block
    for bi in range(nbh):
        for bj in range(nbw):
            if block_ok is not None and not block_ok[bi * nbw + bj]:
                continue
            r0, c0 = bi * block, bj * block
            rr = np.arange(r0, r0 + block)
            cc = np.arange(c0, c0 + block)
            gr, gc = np.meshgrid(rr, cc, indexing="ij")
            m = ((rr[None, :] % max(int(spacing), 1)) == 0) if False else None
            sel_r = (rr - r0) % max(int(round(spacing)), 1) == 0
            sel_c = (cc - c0) % max(int(round(spacing)), 1) == 0
            m = sel_r[:, None] & sel_c[None, :]
            sub = np.zeros((block, block), bool)
            sub[m] = True
            sub &= support[r0:r0 + block, c0:c0 + block]
            sub &= delta[r0:r0 + block, c0:c0 + block] >= tau
            out[r0:r0 + block, c0:c0 + block] = sub
    return out


def emission(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Score an emitted prediction with the organizer's metric."""
    return {**metric.dti_terms(pred, truth), "dti": metric.dti(pred, truth)}
