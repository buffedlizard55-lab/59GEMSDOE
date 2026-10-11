"""Distance-weighted Tversky index (DTI) exactly as written in the official GEMS problem page.

Source (official, fetched 2026-10-11):
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/  ("Performance metric")

Official definitions, with p(x) in [0,1] the prediction and g the ground truth raster, R = 300 m = 3 px:

  k(d)        = max(1 - d/R, 0)                      triangular kernel
  TP_w        = sum_{g in G}  max_{x: d(x,g)<=R} p(x) k(d(x,g))
  FP_w        = sum_{x: p(x)>0} p(x) [1 - max_{g in G} k(d(x,g))]
  FN_w        = sum_{g in G}  [1 - max_{x: d(x,g)<=R} p(x) k(d(x,g))]
  DTI(a,b)    = TP_w / (TP_w + a*FP_w + b*FN_w + eps),   a = 0.2, b = 0.8

Implementation notes (things the official text does NOT pin down, flagged for review):
  * eps is not stated publicly; we default to 1e-9 (negligible for non-empty truth).
  * Known USGS/INGENIOUS fault pixels are excluded from evaluation. Staff confirmed this in
    the DrivenData forum on 2026-09-16 ("Pixels corresponding to known USGS/INGENIOUS faults are
    masked / excluded from evaluation"). Pass that mask as `ignore`.
  * Distances are Euclidean between pixel centres on a 100 m grid.

Everything here is a pure function of arrays, so it can be unit-tested on hand-computed cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ALPHA = 0.2  # penalty on false positives (official)
BETA = 0.8  # penalty on false negatives (official)
R_METERS = 300.0  # kernel support (official)
PIXEL_METERS = 100.0  # training-data resolution (official)
R_PX = R_METERS / PIXEL_METERS  # = 3.0


@dataclass(frozen=True)
class DTIResult:
    dti: float
    tp_w: float
    fp_w: float
    fn_w: float
    n_truth: int
    n_eval_pred_positive: int


def _offsets_within(radius_px: float):
    """All integer offsets (dy, dx) with Euclidean length <= radius, with kernel weight."""
    r = int(np.ceil(radius_px))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dy, dx))
            if d <= radius_px + 1e-12:
                out.append((dy, dx, max(1.0 - d / radius_px, 0.0)))
    return out


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return b with b[y, x] = a[y+dy, x+dx], zero where the source is outside the grid."""
    h, w = a.shape
    b = np.zeros_like(a)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys0 < ys1 and xs0 < xs1:
        b[ys0:ys1, xs0:xs1] = a[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx]
    return b


def dti(
    pred: np.ndarray,
    truth: np.ndarray,
    ignore: np.ndarray | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    radius_px: float = R_PX,
    eps: float = 1e-9,
) -> DTIResult:
    """Compute the distance-weighted Tversky index of `pred` against binary `truth`.

    pred   : float array in [0,1]; NaN is rejected (the portal rejects non-finite values).
    truth  : bool/0-1 array, same shape.
    ignore : optional bool array; pixels where True are excluded from every term
             (known-fault mask, see module docstring).
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("pred contains NaN/inf; the portal and this metric require finite values")
    if pred.min() < 0.0 or pred.max() > 1.0:
        raise ValueError("pred must lie in [0, 1]")

    if ignore is not None:
        ignore = np.asarray(ignore).astype(bool)
        pred = np.where(ignore, 0.0, pred)
        truth = truth & ~ignore

    n_truth = int(truth.sum())

    # FP_w: distance from every pixel to nearest truth pixel (0 on truth itself).
    if n_truth > 0:
        d_nearest_px = ndimage.distance_transform_edt(~truth)
        k_nearest = np.maximum(1.0 - d_nearest_px / radius_px, 0.0)
    else:
        k_nearest = np.zeros_like(pred)
    fp_w = float(np.sum(pred[pred > 0] * (1.0 - k_nearest[pred > 0])))

    # TP_w / FN_w: for each truth pixel g, best kernel-weighted prediction within R.
    best = np.zeros_like(pred)
    for dy, dx, w in _offsets_within(radius_px):
        # best[g] = max over offsets o of p(g+o) * k(|o|); the kernel is symmetric so direction is irrelevant
        np.maximum(best, _shift(pred, dy, dx) * w, out=best)
    best_at_truth = best[truth]
    tp_w = float(best_at_truth.sum())
    fn_w = float((1.0 - best_at_truth).sum())

    denom = tp_w + alpha * fp_w + beta * fn_w + eps
    value = tp_w / denom if denom > 0 else 0.0
    return DTIResult(
        dti=float(value),
        tp_w=tp_w,
        fp_w=fp_w,
        fn_w=fn_w,
        n_truth=n_truth,
        n_eval_pred_positive=int(np.count_nonzero(pred > 0)),
    )


def break_even_kernel_credit(dti_value: float, alpha: float = ALPHA) -> float:
    """Special case: break-even kernel credit for an ISOLATED dot aimed at ONE truth pixel.

    Exact general rule (derived in docs/research/utility-derivation.md, verified numerically in
    scripts/utility_table.py): with S = TP_w + FP_w and the exact identity TP_w + FN_w = G,
    DTI = TP_w / (alpha*S + beta*G) and therefore

        sign(d DTI) = sign(dTP_w - alpha * DTI * dS),   dS = dTP_w + dFP_w.

    For a dot whose only effect is on one truth pixel at kernel credit k, dTP = k and dS = 1, so the
    dot pays iff k > alpha * DTI. This function returns that single-pixel threshold.
    """
    return alpha * float(dti_value)


def break_even_distance_px(dti_value: float, alpha: float = ALPHA, radius_px: float = R_PX) -> float:
    """Distance (px) at which the single-pixel break-even credit k* = alpha*DTI is reached."""
    k_star = break_even_kernel_credit(dti_value, alpha)
    return radius_px * (1.0 - k_star)
