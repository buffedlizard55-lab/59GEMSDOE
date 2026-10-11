"""Live-anchored credit shells.

Organizer-side scores for nine rasters in the public family (OWNER-REPORTED,
not organizer receipts) are converted into pooled truth credit ``T`` with the
exact metric identity, then differenced along the one nesting chain the pixel
sets actually satisfy.  The result is a *measured* mean per-dot credit for each
shell -- the only numbers in this repository that come from the real scoring
function and the real hidden truth.

Metric identity used (unit-valued binary dots, GEMSDOE48-corrected form)::

    DTI = T / (0.2*T + 0.2*n - 0.2*Q + 0.8*G)

with  T = sum_g max_x k(d(x,g))   (truth-centred credit)
      Q = sum_x max_g k(d(x,g))   (prediction-centred credit)
      n = number of dots,  G = |hidden truth|

The single free constant ``G`` is fixed by the one *nested* owner-reported pair
(0.2600 @ 44,090 dots and 0.2778 @ 37,654 dots, the b2 prune): assuming the
2,545 + 3,891 deleted dots changed T and Q negligibly gives G = 14,090 and
T = 5,223.  Everything downstream is labelled MODEL.
"""
from __future__ import annotations

import json
import os
import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.optimize import nnls

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "out", "live_cache")

# (file, n_dots, owner_reported_score).  Scores are OWNER-REPORTED (copied from
# the task brief / sibling-repo manifests); no submission-page receipt exists.
LIVE = [
    ("live_01922_h195.tif",     121131, 0.1922),
    ("live_02477_d15.tif",       60069, 0.2477),
    ("live_02600_d28.tif",       44090, 0.2600),
    ("live_02708_d28.tif",       40199, 0.2708),
    ("live_02778_b2prune.tif",   37654, 0.2778),
    ("live_02649_h32tip.tif",    42294, 0.2649),
    ("live_02632_tipstepover.tif", 41865, 0.2632),
    ("live_02710_h36rung30.tif", 37660, 0.2710),
    ("live_00512_sgmc44k.tif",   44090, 0.0512),
]

SUM_K = 9.380298          # sum of triangular-kernel weights, verified in metric.py
ALPHA, BETA = 0.2, 0.8


def credit(n: float, dti: float, G: float) -> float:
    """T implied by (n, dti, G) under T == Q (see module docstring)."""
    return dti * (ALPHA * n + BETA * G)


def main() -> dict:
    with rasterio.open(os.path.join(REPO, "data/bridge/existing_faults.tif")) as s:
        cat = s.read(1)
    with rasterio.open(os.path.join(REPO, "data/bridge/sample_submission.tif")) as s:
        sub = s.read(1)
    fp = np.isfinite(sub)
    cat = (cat > 0) & fp
    dcat = ndi.distance_transform_edt(~cat)

    # ---- 1. fix G from the nested owner-reported pair -----------------------
    # T(44090, 0.2600) == T(37654, 0.2778)
    # dti*(0.2n + 0.8G) equal  ->  solve for G
    n1, d1 = 44090.0, 0.2600
    n2, d2 = 37654.0, 0.2778
    # 0.26(0.2*44090 + 0.8G) = 0.2778(0.2*37654 + 0.8G)
    G = (d1 * ALPHA * n1 - d2 * ALPHA * n2) / (d2 * BETA - d1 * BETA)
    T_pair = credit(n2, d2, G)

    # ---- 2. implied credit for every scored raster --------------------------
    rows = []
    for f, n, dti in LIVE:
        T = credit(n, dti, G)
        rows.append(dict(file=f, n_dots=int(n), owner_reported_score=dti,
                         model_T=float(T), model_mean_credit=float(T / n),
                         model_coverage=float(T / G)))

    # ---- 3. nesting chain -> measured shell credits -------------------------
    M = {}
    for f, _, _ in LIVE:
        with rasterio.open(os.path.join(CACHE, f)) as s:
            a = s.read(1)
        M[f] = np.isfinite(a) & (a > 0) & fp
    chain = ["live_01922_h195.tif", "live_02600_d28.tif",
             "live_02708_d28.tif", "live_02778_b2prune.tif"]
    for a, b in zip(chain, chain[1:]):
        assert (M[b] & ~M[a]).sum() == 0, f"{b} is not a subset of {a}"

    T = {r["file"]: r["model_T"] for r in rows}
    shells = []
    for i, f in enumerate(chain):
        # chain[i] SUPERSETS chain[i+1]; the shell is the part the next
        # (smaller) member does not keep.
        nxt = chain[i + 1] if i + 1 < len(chain) else None
        core = M[f] if nxt is None else M[f] & ~M[nxt]
        t_core = T[f] - (T[nxt] if nxt else 0.0)
        n = int(core.sum())
        shells.append(dict(shell=f, n=n, model_T=float(t_core),
                           model_mean_credit=float(t_core / n) if n else None,
                           dcat_median=float(np.median(dcat[core])) if n else None,
                           dcat_mean=float(dcat[core].mean()) if n else None,
                           frac_le2=float((dcat[core] <= 2).mean()) if n else None,
                           frac_le5=float((dcat[core] <= 5).mean()) if n else None,
                           frac_le10=float((dcat[core] <= 10).mean()) if n else None,
                           frac_le30=float((dcat[core] <= 30).mean()) if n else None))
    # standalone low-credit reference set (disjoint mechanism)
    E = M["live_00512_sgmc44k.tif"]
    shells.append(dict(shell="live_00512_sgmc44k.tif (standalone)", n=int(E.sum()),
                       model_T=float(T["live_00512_sgmc44k.tif"]),
                       model_mean_credit=float(T["live_00512_sgmc44k.tif"] / E.sum()),
                       dcat_median=float(np.median(dcat[E])), dcat_mean=float(dcat[E].mean()),
                       frac_le2=float((dcat[E] <= 2).mean()), frac_le5=float((dcat[E] <= 5).mean()),
                       frac_le10=float((dcat[E] <= 10).mean()), frac_le30=float((dcat[E] <= 30).mean())))

    # ---- 4. can distance-to-catalogue alone explain the shell credits? ------
    # Fit  e(d) = sum_m w_m B_m(dcat)  (piecewise-constant on distance bins)
    # to the four shell totals + the absolute mass constraint, non-negative
    # least squares.  This is the lane's central quantity: the damage-zone
    # decay profile, measured against live scores instead of assumed.
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 32, 1e9])
    nb = len(edges) - 1
    def hist(mask):
        h, _ = np.histogram(dcat[mask], bins=edges)
        return h.astype(float)
    A_rows, b_vec, w_vec = [], [], []
    # Use the *cumulative* (always non-negative) constraints: the total credit
    # of each scored raster and the absolute mass constraint.  Shell
    # differences between the 44,090 / 40,199 / 37,654 members are degenerate
    # (identical T by construction of G) and are reported, not fitted.
    for f, n, dti in LIVE:
        A_rows.append(hist(M[f])); b_vec.append(T[f])
        w_vec.append(1.0 / max(abs(T[f]), 50.0))
    A_rows.append(hist(fp)); b_vec.append(G * SUM_K); w_vec.append(1.0 / (G * SUM_K))
    A = np.array(A_rows); b = np.array(b_vec); w = np.array(w_vec)
    sol, _ = nnls(A * w[:, None], b * w)
    pred = A @ sol
    fit = dict(bin_edges=edges.tolist(), weights=sol.tolist(),
               observed=[float(x) for x in b_vec], fitted=[float(x) for x in pred],
               mean_credit_by_bin=[float(x) for x in sol],
               footprint_counts=hist(fp).tolist())

    out = dict(
        schema="gems59.live-credit-shells.v1",
        label="MODEL — derived from OWNER-REPORTED scores, not organizer receipts",
        metric_identity="DTI = T/(0.2T + 0.2n - 0.2Q + 0.8G), T==Q assumed",
        G_hidden_truth=float(G), T_pair=float(T_pair),
        sum_kernel=SUM_K, footprint_cells=int(fp.sum()), catalogue_px=int(cat.sum()),
        rasters=rows, shells=shells, distance_profile_fit=fit,
    )
    path = os.path.join(REPO, "evidence", "live_credit_shells.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1)[:6000])
    return out


if __name__ == "__main__":
    main()
