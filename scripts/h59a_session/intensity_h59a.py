"""Presence-only point-process intensity with an explicit mapping-effort correction.

Why a point process
-------------------
The prediction target is a *density of new-fault pixels per unit area*, not a
per-pixel posterior, because the organizer's metric prices each emitted pixel by
the kernel weight it earns within 300 m (``metric.breakeven_delta``).  The object
that answers "how much fault lies within 300 m of here" is a point-process
intensity.

Model
-----
Mapped fault pixels are a realisation of an inhomogeneous Poisson point process

    lambda_obs(x) = lambda(x) * b(x) * rho

* ``lambda(x)`` - true fault intensity, driven by the geophysical channels X;
* ``b(x)``      - detection probability, driven by mapping-effort channels E;
* ``rho``       - a scalar rate that is not of interest.

Fithian & Hastie (*Annals of Applied Statistics* **7(4)**, 1917-1939, 2013;
doi:10.1214/13-AOAS667) prove that the IPP and MaxEnt give the *exact same*
estimate of that density.  Presence-only records are biased by where surveyors
looked, which is why the effort channels enter the fit explicitly rather than
being ignored: Phillips, Dudik, Elith, Graham, Lehmann, Leathwick & Ferrier
(*Ecological Applications* **19(1)**, 181-197, 2009; doi:10.1890/07-2153.1) show
that background data should carry the *same* sampling bias as the occurrence
records (the target-group / biased-background argument).

Estimation
----------
We fit the discriminant of the process with a **case-control (logistic) design**
and convert back to intensities with the odds.  Prentice & Pyke (*Biometrika*
**66(3)**, 403-411, 1979; doi:10.1093/biomet/66.3.403) show a case-control
logistic fit recovers the odds ratio consistently, and Fithian & Hastie show the
IPP log-intensity differs from the logistic linear predictor by the (constant) log
exposure -- **but only in their *infinitely weighted* logistic regression**: they
are explicit that ordinary logistic regression "in general yields a different
estimate in finite samples" and that under misspecification the two may have
substantially different asymptotic limits.  We therefore use

    lambda(x)  ∝  p(x) / (1 - p(x))              (odds of the fitted discriminant)

as an *approximation* to the IPP intensity, not as the exact IPP estimate.  Two
things make the approximation acceptable here and both are stated rather than
assumed: (i) the emission rule renormalises the field, so only its *shape* is
used, and (ii) the released arm is validated on the holdout against the same
emission ladder, so the comparison is like-for-like.  An exact fit would use
infinitely weighted logistic regression (or a Poisson GBM with quadrature
weights); that is queued work, not shipped.

Effort correction by counterfactual clamping
-------------------------------------------
The bias channels are *clamped* to a common "fully mapped" reference and the model
is re-evaluated on the counterfactual design:

    lambda_obs(x) = intensity under the effort actually present
    lambda_ref(x) = intensity the same geophysics would show if the area had been
                    mapped as thoroughly as the best-mapped part of the region

Since lambda_obs = lambda * b and lambda_ref = lambda * b_max (with b_max scaled to 1),

    lambda_hat(x) = lambda_ref(x)                         true fault intensity
    b_hat(x)      = min(1, lambda_obs / lambda_ref)       detection probability
    lambda_new(x) = lambda_hat * (1 - b_hat) = lambda_ref - lambda_obs   (>= 0)

i.e. **the gap intensity is simply the counterfactual minus the observed
intensity**: the faults that are really there minus the ones the catalogue already
has.  This is the "mapping gap" as an identified operation on the fitted process
rather than a hand-made set difference of two rasters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features

EFFORT_CHANNELS = ["dist_lidar", "mapscale_field", "trace_density", "therm_dens_all"]

# Reference ("fully mapped") value for each effort channel.  Direction is fixed a
# priori from what the channel measures, never fitted:
#   dist_lidar      0.0   -> inside lidar coverage (the area has been scanned)
#   mapscale_field  2.0   -> log10(100 000); 1:100k is the finer of the two scales present
#   trace_density   p95   -> as densely fault-mapped as the best-covered 5 %
#   therm_dens_all  p95   -> as heavily drilled/spring-inventoried as the best 5 %
EFFORT_REFERENCE = {"dist_lidar": ("value", 0.0),
                    "mapscale_field": ("value", 2.0),
                    "trace_density": ("quantile", 0.95),
                    "therm_dens_all": ("quantile", 0.95)}

GEOPHYS_CHANNELS = [c for c in features.CHANNELS if c not in EFFORT_CHANNELS]


@dataclass
class PPSpec:
    max_iter: int = 200
    learning_rate: float = 0.10
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2: float = 1.0
    n_background: int = 200_000
    seed: int = 20261011


@dataclass
class PPModel:
    model: HistGradientBoostingClassifier
    spec: PPSpec
    channels: list[str]
    effort_pos: list[int]
    effort_ref: np.ndarray
    meta: dict = field(default_factory=dict)


def _odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    return (p / (1.0 - p)).astype(np.float32)


def sample_design(stack, presence: np.ndarray, support: np.ndarray,
                  chan_idx: list[int], spec: PPSpec, rng: np.random.Generator,
                  counterfactual: bool = False, effort_pos=(), effort_ref=None):
    pres_idx = np.flatnonzero((presence & support).ravel())
    sup_idx = np.flatnonzero(support.ravel())
    n_bg = int(min(spec.n_background, sup_idx.size))
    bg_idx = rng.choice(sup_idx, size=n_bg, replace=False)
    idx = np.concatenate([pres_idx, bg_idx])
    H, W = support.shape
    rows, cols = np.unravel_index(idx, (H, W))
    C = len(chan_idx)
    X = np.empty((idx.size, C), np.float32)
    for k, ci in enumerate(chan_idx):
        X[:, k] = stack[ci, rows, cols]
    y = np.concatenate([np.ones(pres_idx.size, np.int8), np.zeros(bg_idx.size, np.int8)])
    if counterfactual:
        for j, ci in enumerate(effort_pos):
            X[:, j] = effort_ref[j]
    return X, y, idx


def fit_pp(stack, footprint: np.ndarray, presence: np.ndarray, spec: PPSpec | None = None,
           area_mask: np.ndarray | None = None, seed: int | None = None) -> PPModel:
    spec = spec or PPSpec()
    if seed is not None:
        object.__setattr__(spec, "seed", seed)
    rng = np.random.default_rng(spec.seed)
    support = footprint if area_mask is None else (footprint & area_mask)
    chan = list(features.CHANNELS)
    effort_pos = [chan.index(c) for c in EFFORT_CHANNELS]

    effort_ref = np.empty(len(effort_pos), np.float32)
    for j, ci in enumerate(effort_pos):
        kind, val = EFFORT_REFERENCE[chan[ci]]
        if kind == "value":
            effort_ref[j] = np.float32(val)
        else:
            v = stack[ci][support & np.isfinite(stack[ci])]
            effort_ref[j] = np.float32(np.quantile(v, val)) if v.size else np.float32(0.0)

    X, y, _ = sample_design(stack, presence, support, list(range(len(chan))), spec, rng)
    m = HistGradientBoostingClassifier(
        max_iter=spec.max_iter, learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes, min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2, random_state=spec.seed, early_stopping=False,
        class_weight="balanced")
    m.fit(X, y)
    return PPModel(model=m, spec=spec, channels=chan, effort_pos=effort_pos,
                   effort_ref=effort_ref,
                   meta={"n_presence": int(y.sum()), "n_background": int((y == 0).sum()),
                         "effort_reference": {chan[ci]: float(effort_ref[j])
                                              for j, ci in enumerate(effort_pos)}})


def _predict_proba_chunked(model: PPModel, stack, support: np.ndarray,
                           counterfactual: bool, chunk_rows: int = 96) -> np.ndarray:
    H, W = support.shape
    C = len(model.channels)
    out = np.zeros((H, W), np.float32)
    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        blk = np.empty((C, r1 - r0, W), np.float32)
        for k in range(C):
            blk[k] = stack[k, r0:r1]
        X = blk.reshape(C, -1).T
        if counterfactual:
            for j, ci in enumerate(model.effort_pos):
                X[:, ci] = model.effort_ref[j]
        p = model.model.predict_proba(X)[:, 1]
        out[r0:r1] = p.reshape(r1 - r0, W).astype(np.float32)
    out[~support] = 0.0
    return out


def effort_gap(model: PPModel, stack, footprint: np.ndarray,
               area_mask: np.ndarray | None = None, chunk_rows: int = 96):
    """Return (lambda_new, lambda_ref, b_hat); see the module docstring."""
    support = footprint if area_mask is None else (footprint & area_mask)
    lam_obs = _odds(_predict_proba_chunked(model, stack, support, False, chunk_rows))
    lam_ref = _odds(_predict_proba_chunked(model, stack, support, True, chunk_rows))
    lam_ref = np.maximum(lam_ref, lam_obs)                 # detection <= 1 by construction
    b = np.where(lam_ref > 0, np.clip(lam_obs / np.maximum(lam_ref, 1e-12), 0.0, 1.0), 0.0)
    lam_new = np.maximum(lam_ref - lam_obs, 0.0).astype(np.float32)
    lam_new[~support] = 0.0
    return (lam_new.astype(np.float32), lam_ref.astype(np.float32), b.astype(np.float32))
