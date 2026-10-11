"""Exact re-implementation of the organizer's distance-weighted Tversky index (DW-Tversky).

Source (official, retrieved 2026-10-11):
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
"Performance metric" / "Mathematical representation".

    k(d)   = max(1 - d/R, 0),  R = 300 m  (= 3 px at 100 m)
    TP_w   = sum_{g in G} max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FP_w   = sum_{x: p(x) > 0} p(x) [ 1 - max_{g in G} k(d(x,g)) ]
    FN_w   = sum_{g in G} [ 1 - max_{x: d(x,g) <= R} p(x) k(d(x,g)) ]
    DTI    = TP_w / (TP_w + a*FP_w + b*FN_w + eps),   a = 0.2, b = 0.8

This module is verified three ways (tests/test_metric.py):
  1. against the organizer's own worked example (TP=3.00, FP=1.89, FN=2.00 -> 0.60);
  2. against a dense O(N*|G|) brute-force evaluation on small random grids;
  3. for the algebraic identity TP_w + FN_w == |G| (which holds because
     FN_w is defined with the same max term), giving the closed form
        DTI = TP_w / (0.2*TP_w + 0.2*FP_w + 0.8*|G|).
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.2
BETA = 0.8
RADIUS_PX = 3.0          # 300 m support / 100 m pixels
EPS = 1e-12


def kernel_offsets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dr, dc, k) for every pixel offset with k(d) > 0 inside the 300 m support."""
    offs = []
    r = int(np.ceil(RADIUS_PX))
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            d = float(np.hypot(dr, dc))
            if d < RADIUS_PX:
                offs.append((dr, dc, 1.0 - d / RADIUS_PX))
    a = np.array(offs, dtype=float)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]


DR, DC, KW = kernel_offsets()


def _shift(src: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """out[r, c] = src[r + dr, c + dc] with ``fill`` outside the array."""
    out = np.full(src.shape, fill, dtype=src.dtype)
    H, W = src.shape
    r0, r1 = max(0, -dr), min(H, H - dr)
    c0, c1 = max(0, -dc), min(W, W - dc)
    if r0 >= r1 or c0 >= c1:
        return out
    out[r0:r1, c0:c1] = src[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def kernel_max(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """max-plus dilation: out[g] = max over offsets o of arr[g + o] * k(|o|).

    float32 throughout: 25 shifts over 12.3 M pixels move ~2.5 GB, so the dtype
    halves the memory traffic.  Verified identical to the float64 path to within
    float32 epsilon by tests/test_metric.py::test_fast_matches_bruteforce.
    """
    a = np.asarray(arr, dtype=np.float32)
    out = np.zeros(a.shape, dtype=np.float32)
    buf = np.empty(a.shape, dtype=np.float32)
    for dr, dc, kw in zip(DR, DC, KW):
        buf = _shift(a, dr, dc, np.float32(fill))
        buf *= np.float32(kw)
        np.maximum(out, buf, out=out)
    return out


def dti_terms(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Exact TP_w, FP_w, FN_w of the organizer's metric.

    ``pred``  : float array, predicted confidence in [0, 1].
    ``truth`` : boolean array, ground-truth fault pixels (the *new* fault set).
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    g_n = int(truth.sum())
    if g_n == 0:
        # No positives: only the false-positive mass is defined.
        delta = kernel_max(np.where(truth, 1.0, 0.0))       # 0 everywhere
        fp = float((pred * (1.0 - delta)).sum())
        return {"tp": 0.0, "fp": fp, "fn": 0.0, "n_truth": 0, "n_pred": int((pred > 0).sum())}

    # M[g] = max_{x: d(x,g) <= R} p(x) k(d(x,g))   -> dilation of the prediction
    M = kernel_max(pred)[truth]
    tp = float(M.sum())
    fn = float((1.0 - M).sum())

    # delta[x] = max_{g in G} k(d(x,g))            -> dilation of the truth indicator
    delta = kernel_max(truth.astype(np.float64))
    fp = float((pred * (1.0 - delta)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "n_truth": g_n, "n_pred": int((pred > 0).sum())}


def dti(pred: np.ndarray, truth: np.ndarray, alpha: float = ALPHA, beta: float = BETA) -> float:
    t = dti_terms(pred, truth)
    denom = t["tp"] + alpha * t["fp"] + beta * t["fn"] + EPS
    return float(t["tp"] / denom)


# --------------------------------------------------------------------------
# Analytic results derived from the organizer's equations (used to size budgets)
# --------------------------------------------------------------------------

def breakeven_delta(dti_value: float, alpha: float = ALPHA, beta: float = BETA) -> float:
    """Minimum expected kernel weight a candidate pixel must have to pay for itself.

    Derivation (exact, from the organizer's equations).  Write A = TP_w,
    F = FP_w, G = |G|.  Because FN_w = G - TP_w,

        DTI = A / (alpha*A + alpha*F + beta*G)   with alpha - beta = -0.6 for the
                                                 published alpha=0.2, beta=0.8, so
        DTI = A / ((1-beta)*A + alpha*F + beta*G) = A / (0.2A + 0.2F + 0.8G)
            = A / D,   D := 0.2A + 0.2F + 0.8G.

    Emitting mass v at a pixel whose kernel weight (against the truth) is the
    random variable delta raises A by v*delta and F by v*(1-delta), so

        dDTI/dv  =  [ delta*(D - 0.2A) - 0.2A*(1-delta) ] / D^2
                 =  [ delta*(0.2F + 0.8G) - 0.2A*(1-delta) ] / D^2  >= 0
                 <=>  delta*(1 - alpha*DTI) >= alpha*DTI*(1 - delta)      (divide by D)
                 <=>  delta >= alpha*DTI.

    So the bar is exactly **alpha times the score you are getting**.  At the
    group's incumbent 0.2778 that is 0.0556; at the public board leader 0.3774
    it is 0.0755.  For a single dot in a 1 km block that contains L px of fault
    trace, E[delta] = 3L/100, so break-even is L = 100*alpha*DTI/3 = 1.85 px
    per 100 px block at DTI 0.2778 -- the "~1-2 % density" rule of thumb.
    """
    return float(alpha * dti_value)


def breakeven_trace_px_per_km_block(dti_value: float, alpha: float = ALPHA,
                                    block_px: int = 100) -> float:
    """Fault-trace pixels a 1 km (10x10 px) block must hold before a random dot pays.

    E[delta] for a uniformly placed dot = (integral of k over the block)/block
    area = 3*L/block_px for a line-like trace of L px (the perpendicular
    cross-section of the triangular kernel sums to 3).
    """
    return block_px * breakeven_delta(dti_value, alpha) / 3.0


def dti_from_hits(effective_hits: float, n_dots: int, n_truth: int,
                  alpha: float = ALPHA, beta: float = BETA) -> float:
    """Closed form DTI = TP_w / (alpha*TP_w + alpha*FP_w + beta*|G|).

    Valid when no truth pixel is claimed by more than one predicted pixel
    (the "sparse dots" regime), where TP_w = sum_x delta(x) over emitted dots
    and FP_w = n_dots - TP_w.
    """
    g = float(n_truth)
    denom = alpha * effective_hits + alpha * (n_dots - effective_hits) + beta * g
    return float(effective_hits / max(denom, EPS))


def brute_force_dti(pred: np.ndarray, truth: np.ndarray,
                    alpha: float = ALPHA, beta: float = BETA) -> float:
    """Dense O(|G| * N) reference implementation used only to test ``dti``."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    H, W = pred.shape
    gr, gc = np.nonzero(truth)
    pr, pc = np.nonzero(pred > 0)
    if len(gr) == 0:
        return 0.0
    tp = 0.0
    fn = 0.0
    for r, c in zip(gr, gc):
        best = 0.0
        for r2, c2 in zip(pr, pc):
            d = np.hypot(r - r2, c - c2)
            if d < RADIUS_PX:
                best = max(best, pred[r2, c2] * (1.0 - d / RADIUS_PX))
        tp += best
        fn += 1.0 - best
    fp = 0.0
    pr_all, pc_all = np.nonzero(pred)
    for r, c in zip(pr_all, pc_all):
        best = 0.0
        for r2, c2 in zip(gr, gc):
            d = np.hypot(r - r2, c - c2)
            if d < RADIUS_PX:
                best = max(best, 1.0 - d / RADIUS_PX)
        fp += pred[r, c] * (1.0 - best)
    return float(tp / (tp + alpha * fp + beta * fn + EPS))
