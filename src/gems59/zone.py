"""1 km zoning plus lattice tiling whose spacing comes from a utility table.

Lane (59GEMSDOE): *zone by new-fault density, then tile only what clears
break-even.*  The arithmetic behind "break-even" is the metric's own marginal
condition, derived -- not assumed -- in :mod:`gems59.metric`:

    adding a dot of realised kernel credit ``k`` raises the DTI iff
    ``k > alpha * s / (1 - alpha * s)``   (``s`` = the index being scored)

with ``alpha = 0.2``.  At ``s = 0.3195`` the bar is ``0.0683``; a dot must sit
within ``0.0683 * 300 m = 280 m`` of a scored truth pixel to pay for itself.
For a square lattice of spacing ``s`` px over ground whose fault density (scored
truth pixels per pixel) is ``d``, the expected credit per dot is
``0.617 * d * s^2`` -- the factor 0.617 is the mean of ``1 - rho/s`` over a
square Voronoi cell (the expected distance from a random point to the nearest
lattice site is ``0.3826 s``).  So a lattice pays where

    d > 0.0683 / (0.617 * s^2)  ->  s = 3 px: d > 1.23 %;  s = 6 px: 0.31 %;
    s = 9 px: 0.14 %;  s = 12 px: 0.077 %.

That is the lane's "a lattice pays only above roughly 1-2 % density", and it is
why the spacing must widen as density falls.  The numbers are *re-derived on our
own holdout* in ``scripts/run_lane.py`` rather than trusted from this docstring;
the table that ships is whatever the holdout selected.

Nesting: every spacing is a multiple of 3 px, so the coarse lattices are exact
subsets of the fine one.  A block's level therefore means "keep the lattice
sites this level would keep", with no phase bookkeeping.
"""
from __future__ import annotations

import numpy as np

#: Candidate lattice spacings, px (multiples of the 3 px = 300 m metric kernel).
LEVELS = (3, 6, 9, 12)
PHASE = 1  # lattice sites at (row % s == PHASE) and (col % s == PHASE)


def lattice_mask(shape: tuple[int, int], spacing: int, phase: int = PHASE) -> np.ndarray:
    """Global lattice at one spacing."""
    if spacing not in LEVELS:
        raise ValueError(f"spacing {spacing} not in {LEVELS}")
    r = (np.arange(shape[0]) % spacing) == phase
    c = (np.arange(shape[1]) % spacing) == phase
    return r[:, None] & c[None, :]


def tile(shape: tuple[int, int], level_of_block: np.ndarray, block: int = 10,
         phase: int = PHASE) -> np.ndarray:
    """Keep, in every block, the lattice sites of that block's chosen level.

    ``level_of_block`` is an integer array of block spacings, ``0`` meaning
    "do not tile this block".  Because every level is a multiple of the finest
    level, a site kept by a coarse level is also a site of the fine lattice, so
    the union needs no phase correction.
    """
    levels = np.asarray(level_of_block)
    if levels.ndim != 2:
        raise ValueError("level_of_block must be 2-D")
    ny, nx = levels.shape
    if np.any(~np.isin(levels, (0, *LEVELS))):
        raise ValueError(f"unknown lattice levels in {np.unique(levels)}")
    h, w = shape
    r = np.arange(h)
    c = np.arange(w)
    rows = (np.minimum(r // block, ny - 1)).astype(np.int64)
    cols = (np.minimum(c // block, nx - 1)).astype(np.int64)
    s_map = levels[rows[:, None], cols[None, :]].astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok_r = np.where(s_map > 0, (r[:, None] % np.maximum(s_map, 1)) == phase, False)
        ok_c = np.where(s_map > 0, (c[None, :] % np.maximum(s_map, 1)) == phase, False)
    return np.asarray(ok_r & ok_c, bool)


#: Density bin edges in *expected scored-fault pixels per 1 km block*.  The
#: first edge is the lane's break-even: a 3 px lattice pays only where the block
#: is expected to hold more than about 1-2 new-fault pixels.  Absolute bins, not
#: quantiles: with most blocks empty, quantile strata collapse onto the top
#: stratum and the utility table would degenerate to a single spacing.
DENSITY_EDGES = (2.5, 5.0, 10.0, 20.0)
BREAK_EVEN_PX_PER_BLOCK = 1.4


def density_px_per_block(shares: np.ndarray, expected_total_px: float) -> np.ndarray:
    """Convert population *shares* to expected scored-fault pixels per block.

    ``expected_total_px`` is a **design projection** of how many scored new-fault
    pixels exist in the study area, not a score and not a measurement.  It enters
    only through this conversion, and the shipped design's sensitivity to it is
    reported in ``evidence/exp3_canary_robustness.json``.
    """
    return np.asarray(shares, np.float64) * float(expected_total_px)


def stratum_of_density(density: np.ndarray,
                       edges: tuple[float, ...] = DENSITY_EDGES) -> np.ndarray:
    """Density stratum index in ``[0, len(edges)]``; 0 is the break-even bin."""
    return np.searchsorted(np.asarray(edges, np.float64),
                           np.asarray(density, np.float64), side="right").astype(np.int64)


def select_blocks(density: np.ndarray, min_px_per_block: float = BREAK_EVEN_PX_PER_BLOCK
                  ) -> tuple[np.ndarray, float]:
    """Boolean block mask for blocks clearing the lattice break-even."""
    d = np.asarray(density, np.float64)
    return d >= float(min_px_per_block), float(min_px_per_block)


def level_map(density: np.ndarray, selected: np.ndarray, table: np.ndarray,
              *, edges: tuple[float, ...] = DENSITY_EDGES) -> np.ndarray:
    """Map each selected block to its lattice level from ``table``.

    ``table[stratum]`` is a spacing in :data:`LEVELS`; ``0`` excludes the block.
    Returns the integer level array (0 = not tiled).
    """
    table = np.asarray(table, np.int64)
    if table.shape != (len(edges) + 1,):
        raise ValueError(f"table must have {len(edges) + 1} entries")
    strat = stratum_of_density(density, edges)
    return np.where(np.asarray(selected, bool), table[strat], 0).astype(np.int64)


def expected_credit_per_dot(density_px_per_px: float, spacing: int) -> float:
    """Analytic expectation used to sanity-check the holdout-derived table."""
    return 0.617 * float(density_px_per_px) * float(spacing) ** 2


def marginal_bar(index: float, alpha: float = 0.2) -> float:
    """The metric's own break-even credit for one more non-redundant dot."""
    return alpha * float(index) / (1.0 - alpha * float(index))


def tile_to_budget(density: np.ndarray, allowed: np.ndarray, table: np.ndarray, *,
                   budget: int, min_px_per_block: float = BREAK_EVEN_PX_PER_BLOCK,
                   block: int = 10, edges: tuple[float, ...] = DENSITY_EDGES
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tile the highest-density blocks first, stopping at ``budget`` dots.

    Ordering is by predicted density (descending), so the dots that are added
    first are the ones the model believes clear break-even most comfortably.
    ``budget`` is the emission's break-even stopping rule in the metric's own
    terms: a dot pays only while its realised credit exceeds
    ``0.2*s/(1-0.2*s)``, so beyond that point extra dots can only lower the
    index.  Returns the dot mask, the level map and a receipt.
    """
    d = np.asarray(density, np.float64)
    sel, cut = select_blocks(d, min_px_per_block)
    level = level_map(d, sel, np.asarray(table, np.int64), edges=edges)
    sites = tile(allowed.shape, level, block=block) & np.asarray(allowed, bool)
    order = np.argsort(-d.ravel(), kind="stable")
    ny, nx = level.shape
    out = np.zeros(allowed.shape, bool)
    n = 0
    used_blocks = 0
    for bidx in order:
        if n >= budget:
            break
        if not sel.ravel()[bidx]:
            continue
        by, bx = divmod(int(bidx), nx)
        r0, c0 = by * block, bx * block
        blk = sites[r0:r0 + block, c0:c0 + block]
        c = int(blk.sum())
        if c == 0:
            continue
        out[r0:r0 + block, c0:c0 + block] = blk
        n += c
        used_blocks += 1
    receipt = dict(dots=int(out.sum()), budget=int(budget), cut_share=float(cut),
                   blocks_selected=int(sel.sum()), blocks_used=int(used_blocks),
                   blocks_available=int((sel & (level > 0)).sum()),
                   spacing_histogram={int(s): int((level == s).sum())
                                      for s in np.unique(level) if s > 0})
    return out & np.asarray(allowed, bool), level, receipt
