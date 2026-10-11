"""Presence-only point-process intensity with a target-group background.

Theory (all citations verified against the published records by the sources
table in ``docs/sources.html``):

* Presence-only methods estimate a *relative* intensity, not a probability of
  presence.  Phillips, Dudik & Schapire (2009), "Sample selection bias and
  presence-only distribution models", Ecological Applications 19(1):181-197,
  doi:10.1890/08-2025.1, show that when occurrences are a *biased sample* of the
  target group, a model fitted against a general background estimates the
  biased sample's intensity, and the bias is corrected by drawing the
  background from the **target group** -- the localities the sampling process
  could actually have visited.  That is what "target-group background" (TGB)
  means here.
* MaxEnt and the inhomogeneous Poisson point-process (IPP) likelihood are
  equivalent in form for this estimator: Fithian & Hastie (2013), "Bias
  correction in species distribution models: pooling survey and collection
  data", Annals of Applied Statistics 7(4):1917-1939, doi:10.1214/13-AOAS667,
  and Renner & Warton (2013), "Equivalence of MAXENT and Poisson point process
  models", Biometrics 69(1):274-281, doi:10.1111/j.1541-0420.2012.01824.x.
  We therefore fit a ridge-regularised logistic regression -- the standard
  practical form of the equivalence -- and read ``exp(eta(x))`` as the relative
  intensity.
* Spatial autocorrelation is handled by *thinning* the presence sample
  (minimum separation) rather than by pretending the pixels are independent.

Implemented with numpy only (no sklearn dependency), so the fit is auditable
line by line and reproducible from the pinned receipts.

Output convention used by the rest of the lane: :func:`fit_intensity` returns
``beta`` on the covariates listed in ``covariates.COVARIATES``, and
:func:`relative_intensity` returns a float32 field normalised so that its sum
over the supplied domain equals 1.  A block's share of that sum is then that
block's share of the fault population -- a *relative* quantity, with no
absolute score claim anywhere in the module.
"""
from __future__ import annotations

import numpy as np

MAX_ITER = 60
TOL = 1e-8


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
                 w: np.ndarray | None = None, max_iter: int = MAX_ITER,
                 tol: float = TOL) -> dict:
    """Ridge-regularised logistic regression by damped Newton (IRLS).

    ``X`` must already include the intercept column.  ``w`` are per-row
    quadrature weights (the target-group background weights).  The penalty is
    applied to every coefficient except the intercept.  Returns the coefficients
    and the convergence trace; raises if Newton fails to converge.
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X / y shape mismatch")
    n, p = X.shape
    w = np.ones(n, np.float64) if w is None else np.asarray(w, np.float64).ravel()
    if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite and positive")
    pen = np.full(p, float(l2), np.float64)
    pen[0] = 0.0                       # intercept unpenalised
    beta = np.zeros(p, np.float64)
    trace = []
    for it in range(int(max_iter)):
        eta = X @ beta
        mu = _sigmoid(eta)
        r = w * (y - mu)
        g = X.T @ r - pen * beta
        h = (X.T * (w * mu * (1.0 - mu))) @ X
        h[np.diag_indices_from(h)] += pen
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h, g, rcond=None)[0]
        # damped step: halve until the penalised deviance does not increase
        dev0 = _deviance(X, y, w, beta, pen)
        t = 1.0
        for _ in range(30):
            cand = beta + t * step
            dev1 = _deviance(X, y, w, cand, pen)
            if np.isfinite(dev1) and dev1 <= dev0:
                break
            t *= 0.5
        beta = beta + t * step
        trace.append(float(dev1))
        if np.abs(t * step).max() < tol * (1.0 + np.abs(beta).max()):
            break
    else:
        raise RuntimeError("Newton did not converge; refusing to return a fit")
    return {"beta": beta, "iterations": it + 1, "deviance": float(dev1),
            "deviance_trace": trace}


def _deviance(X, y, w, beta, pen) -> float:
    eta = np.clip(X @ beta, -500.0, 500.0)
    ll = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
    return float(-2.0 * ll + np.sum(pen * beta * beta))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney), used for the leakage canary only."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, bool)
    n1 = int(label.sum())
    n0 = int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    # average ties
    s_sorted = score[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def thin_points(mask: np.ndarray, min_sep_px: int, seed: int = 0) -> np.ndarray:
    """Greedy spatial thinning of a presence mask to a minimum separation.

    Reduces the spatial autocorrelation that inflates the effective sample size
    of a pixel-wise presence sample.  Order is randomised with a fixed seed so
    the result is reproducible.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.empty(0, np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(ys.size)
    cell = int(max(1, min_sep_px))
    grid_owner = {}
    keep = []
    for i in order:
        key = (int(ys[i]) // cell, int(xs[i]) // cell)
        if key in grid_owner:
            continue
        grid_owner[key] = True
        keep.append(i)
    return order[np.sort(np.asarray(keep, np.int64))]


def relative_intensity(eta: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """``exp(eta)`` renormalised to sum to 1 over ``domain`` (float32)."""
    e = np.exp(np.clip(eta, -60.0, 60.0)).astype(np.float64)
    total = float(e[np.asarray(domain, bool)].sum())
    if total <= 0:
        raise ValueError("empty or zero-mass intensity domain")
    out = np.zeros(e.shape, np.float32)
    out[domain] = (e[domain] / total).astype(np.float32)
    return out


def block_shares(field: np.ndarray, block: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Sum a relative-intensity field onto ``block`` x ``block`` (1 km) blocks.

    Returns ``(shares, bounds)`` where ``shares`` is the array of block sums and
    ``bounds`` the pixel edges.  Block sums are shares of the fault population
    (they sum to 1 over the footprint), a relative quantity only.
    """
    a = np.asarray(field, np.float64)
    h, w = a.shape
    ny, nx = int(np.ceil(h / block)), int(np.ceil(w / block))
    pad_y, pad_x = ny * block - h, nx * block - w
    if pad_y or pad_x:
        a = np.pad(a, ((0, pad_y), (0, pad_x)))
    b = a.reshape(ny, block, nx, block).sum(axis=(1, 3))
    return b, (ny, nx)


def block_of_pixel(shape: tuple[int, int], block: int = 10) -> tuple[np.ndarray, tuple[int, int]]:
    """Row/column block index for every pixel (no padding of the data itself)."""
    h, w = shape
    ny, nx = int(np.ceil(h / block)), int(np.ceil(w / block))
    r = (np.arange(h) // block).astype(np.int32)
    c = (np.arange(w) // block).astype(np.int32)
    return (np.repeat(r[:, None], w, axis=1), np.repeat(c[None, :], h, axis=0)), (ny, nx)
