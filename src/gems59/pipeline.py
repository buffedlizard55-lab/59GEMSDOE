"""One shared pipeline: intensity -> 1 km zoning -> break-even tiling.

Both the holdout instrument and the released artifact call these functions, so
the thing that was measured is byte-for-byte the thing that ships (lane rule 2:
no private forks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from . import emit, features, intensity, metric

BLOCK = 10  # 1 km


@dataclass
class EmissionSpec:
    """Everything the emission rule needs.  Defaults are the pre-declared ones."""
    g_truth: float = 40_000.0       # assumed number of test-set fault pixels
    dti_target: float = 0.28        # starting point for the self-consistent bar
    flank_px: int = 2               # keep dots this far off the mapped catalogue
    spacing_dense: float = 3.0      # px, in the densest zone
    spacing_sparse: float = 5.0     # px, in the sparsest zone that still clears the bar
    min_sep_default: float = 3.0
    self_consistent_iters: int = 6
    max_dots: int = 120_000


def fit_field(stack, fp, presence, spec: intensity.PPSpec | None = None,
              area_mask: np.ndarray | None = None):
    model = intensity.fit_pp(stack, fp, presence, spec=spec, area_mask=area_mask)
    lam_new, lam_true, b_hat = intensity.effort_gap(model, stack, fp, area_mask=area_mask)
    return {"model": model, "lam_true": lam_true, "lam_new": lam_new, "b_hat": b_hat}


def calibrate_delta(lam: np.ndarray, support: np.ndarray, g_truth: float) -> np.ndarray:
    """Delta(x) = G * (q_hat (*) k)(x): expected kernel weight of a dot at x.

    ``q_hat`` is the predicted density normalised over the emission support and
    ``k`` is the organizer's triangular 300 m kernel.  The scale is fixed by
    ``g_truth``, the assumed number of test-set fault pixels, because Delta is a
    *count* of truth pixels weighted by the kernel, not a probability.
    """
    v = np.where(support & np.isfinite(lam), np.maximum(lam, 0.0), 0.0).astype(np.float64)
    tot = v.sum()
    if tot <= 0:
        return np.zeros_like(v, dtype=np.float32)
    q = v / tot
    return (g_truth * emit.credit_field(q.astype(np.float32))).astype(np.float32)


def zone_table(delta: np.ndarray, support: np.ndarray, block: int = BLOCK):
    """1 km block predicted new-fault pixel count + the per-pixel block id."""
    d = np.where(support, delta, 0.0).astype(np.float64)
    # ravel: block ids are flat, so every consumer indexes these with bmap directly
    bs = emit.block_sum(d, block).ravel().astype(np.float32)
    bmap = emit.block_index_map(delta.shape, block)
    return bs, bmap


def spacing_from_density(block_counts: np.ndarray, es: EmissionSpec,
                         tau: float, block: int = BLOCK) -> np.ndarray:
    """Utility-derived spacing: tight where the block is dense, wider as density falls.

    A block's expected credit for one uniformly placed dot is 3*n/100 (the
    triangular kernel's perpendicular cross-section integrates to 3).  A dot is
    only worth emitting while its *marginal* credit clears the bar tau, so the
    number of dots a block can absorb scales with its density and the spacing
    between them widens as the density falls.  The absolute scale is set by the
    holdout utility sweep in ``scripts/evaluate_holdout.py``; the shape
    (spacing ~ n^(-1/2), i.e. a fixed number of dots along the trace per unit
    length) is what the algebra gives.
    """
    n = np.maximum(block_counts, 0.0)
    area = float(block * block)
    # dots a block can carry at break-even: every dot must earn >= tau, and the
    # block's total credit is 3n, so at most 3n/tau dots fit before the marginal
    # one stops paying.  Spacing = sqrt(area / n_dots), clamped.
    n_dots = np.where(n > 0, np.minimum(3.0 * n / max(tau, 1e-9), area), 0.0)
    s = np.where(n_dots > 1, np.sqrt(area / np.maximum(n_dots, 1.0)), es.min_sep_default)
    s = np.clip(s, 1.0, 12.0)
    return s.astype(np.float32).ravel()   # 1-D, indexed by block id


def emit_from_delta(delta: np.ndarray, support: np.ndarray, es: EmissionSpec,
                    block_counts: np.ndarray | None = None,
                    bmap: np.ndarray | None = None,
                    spacing: np.ndarray | None = None) -> dict:
    """Zone, gate at break-even, then tile greedily at the zone's spacing."""
    if block_counts is None:
        block_counts, bmap = zone_table(delta, support)
    nb = block_counts.size
    tau0 = metric.breakeven_delta(es.dti_target)
    bar_per_block = metric.breakeven_trace_px_per_km_block(es.dti_target)

    trace = {  # record every intermediate decision
        "tau_initial": float(tau0),
        "breakeven_trace_px_per_km_block": float(bar_per_block),
        "g_truth_assumed": float(es.g_truth),
    }
    tau = tau0
    dots = None
    for it in range(es.self_consistent_iters):
        block_ok = (block_counts >= bar_per_block)
        if spacing is None:
            sp = spacing_from_density(block_counts, es, tau)
        else:
            sp = spacing
        sep_map = np.where(bmap >= 0, sp[np.clip(bmap, 0, nb - 1)], es.min_sep_default)
        dots = emit.place_dots(delta, support, tau, spacing_map=sep_map,
                               block_ids=bmap, block_ok=block_ok, max_dots=es.max_dots)
        n = int(dots.sum())
        # self-consistent DTI on this (unlabelled) emission cannot be evaluated
        # without truth, so the caller passes the achieved value; here we only
        # iterate on the *count* target via the caller's feedback.
        trace.setdefault("iterations", []).append(
            {"iter": it, "tau": float(tau), "n_dots": n,
             "blocks_total": int(nb), "blocks_above_bar": int(block_ok.sum())})
        break  # a single pass; the caller re-runs with an updated dti_target
    return {"dots": dots, "tau": float(tau), "block_counts": block_counts,
            "bmap": bmap, "block_ok": (block_counts >= bar_per_block),
            "spacing": sp if spacing is None else None, "trace": trace}
