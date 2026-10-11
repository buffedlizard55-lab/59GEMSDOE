"""Experiment H57-K : does the official geophysics predict *live-anchored* credit?

The catalogue hide-and-recover holdout is a weak instrument (the sibling
repository measured rho = +0.14 between its ranking and the owner-reported live
scores).  This experiment replaces it with a label set that comes from the real
scoring function:

* ``scripts/live_credit_shells.py`` converts nine owner-reported scores into a
  pooled credit ``T`` per raster and fixes the hidden-truth mass ``G``.
* The 0.2778 winner ``b2prune`` is a *selected* sample: its 37,654 dots carry a
  mean per-dot credit of 0.1387, 5.4x the footprint mean of 0.0256.  Under the
  biased-sampling model ``p(x | selected) proportional to p_bg(x) * g(e(x))`` the
  density ratio selected-vs-background is monotone in the local credit density
  ``e(x) = (p_truth (*) k)(x)``.
* A pure distance-to-catalogue model is *falsified* first (it cannot separate
  ``b2prune`` from ``sgmc44k``: they have nearly identical distance
  distributions and a 6x credit ratio), so whatever separates them is the
  mechanism worth fitting.

Design
------
1. Density-ratio classifier, selected (b2prune) vs a footprint background sample
   that excludes every live-scored raster (so the contrast is not circular).
2. Spatially blocked 4-quadrant cross-validation: every cell is scored by a model
   that never saw its quadrant -> out-of-fold score on the whole grid.
3. Aggregate calibration: bin the out-of-fold score, then fit non-negative
   per-bin credit weights to the ten aggregate constraints (nine scored rasters
   + the absolute mass sum), validated leave-one-raster-out.
4. Leakage canary on every single feature (protocol rule 4).
"""
from __future__ import annotations

import json
import os
import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "out")
CACHE = os.path.join(REPO, "out", "live_cache")
ALPHA, BETA = 0.2, 0.8
SUM_K = 9.380298
RANK_MAX, NODATA_U8 = 254, 255
SEED = 57

LIVE = [
    ("live_01922_h195.tif", 121131, 0.1922),
    ("live_02477_d15.tif", 60069, 0.2477),
    ("live_02600_d28.tif", 44090, 0.2600),
    ("live_02708_d28.tif", 40199, 0.2708),
    ("live_02778_b2prune.tif", 37654, 0.2778),
    ("live_02649_h32tip.tif", 42294, 0.2649),
    ("live_02632_tipstepover.tif", 41865, 0.2632),
    ("live_02710_h36rung30.tif", 37660, 0.2710),
    ("live_00512_sgmc44k.tif", 44090, 0.0512),
]
POS_FILE = "live_02778_b2prune.tif"


def load_grid():
    with rasterio.open(os.path.join(REPO, "data/bridge/sample_submission.tif")) as s:
        sub = s.read(1)
    fp = np.isfinite(sub)
    with rasterio.open(os.path.join(REPO, "data/bridge/existing_faults.tif")) as s:
        cat = (s.read(1) > 0) & fp
    return fp, cat


def feature_names():
    info = json.load(open(os.path.join(REPO, "evidence", "feature_bank.json")))
    names = []
    for b in info["bands"]:
        names.append(b + "__q")
    for b in info["bands"]:
        names.append(b + "__z")
    names += info["geo_names"] + ["dtip"]
    return names


def score_grid(models, quads, block_rows=300):
    """Out-of-fold HistGradientBoosting score for every footprint cell.

    ``models[q]`` scores quadrant ``q`` and was trained without it.
    """
    bank = np.load(os.path.join(OUT, "features", "bank_u8.npy"), mmap_mode="r")
    geo = np.load(os.path.join(OUT, "features", "geo_f32.npy"), mmap_mode="r")
    dtip = np.load(os.path.join(OUT, "features", "dtip_f32.npy"), mmap_mode="r")
    H, W = bank.shape[1], bank.shape[2]
    s = np.zeros((H, W), np.float32)
    nb = bank.shape[0]
    for r0 in range(0, H, block_rows):
        r1 = min(r0 + block_rows, H)
        cols = [bank[i, r0:r1].reshape(-1, 1).astype(np.float32) for i in range(nb)]
        for i in range(geo.shape[0]):
            cols.append(geo[i, r0:r1].reshape(-1, 1).astype(np.float32))
        cols.append(dtip[r0:r1].reshape(-1, 1).astype(np.float32))
        X = np.concatenate(cols, axis=1)
        X[X >= NODATA_U8] = np.nan
        out = np.full(X.shape[0], np.nan, np.float32)
        for q, mdl in models.items():
            sel = quads[r0:r1].reshape(-1) == q
            if sel.any() and q in models:
                out[sel] = mdl.predict_proba(X[sel])[:, 1].astype(np.float32)
        s[r0:r1] = out.reshape(r1 - r0, W)
        del X, cols, out
    return s


def main():
    shells = json.load(open(os.path.join(REPO, "evidence", "live_credit_shells.json")))
    G = shells["G_hidden_truth"]
    fp, cat = load_grid()
    H, W = fp.shape
    yy = np.arange(H)[:, None] // (H // 2)
    xx = np.arange(W)[None, :] // (W // 2)
    quads = np.clip(yy * 2 + xx, 0, 3).astype(np.int8)     # 4 spatial blocks
    names = feature_names()

    masks = {}
    for f, n, dti in LIVE:
        with rasterio.open(os.path.join(CACHE, f)) as s:
            a = s.read(1)
        masks[f] = np.isfinite(a) & (a > 0) & fp
    T_obs = {f: dti * (ALPHA * n + BETA * G) for f, n, dti in LIVE}

    rng = np.random.default_rng(SEED)
    scored_union = np.zeros(fp.shape, bool)
    for m in masks.values():
        scored_union |= m
    pos = masks[POS_FILE]
    bg_pool = np.flatnonzero((fp & ~scored_union & ~cat).ravel())
    bg = rng.choice(bg_pool, size=220_000, replace=False)

    # ---------------------------------------------------------------- sample
    bank = np.load(os.path.join(OUT, "features", "bank_u8.npy"), mmap_mode="r")
    geo = np.load(os.path.join(OUT, "features", "geo_f32.npy"), mmap_mode="r")
    dtip = np.load(os.path.join(OUT, "features", "dtip_f32.npy"), mmap_mode="r")
    nf = bank.shape[0] + geo.shape[0] + 1

    def X_of(idx):
        cols = [bank[i].reshape(-1)[idx].astype(np.float32) for i in range(bank.shape[0])]
        for i in range(geo.shape[0]):
            cols.append(geo[i].reshape(-1)[idx].astype(np.float32))
        cols.append(dtip.reshape(-1)[idx].astype(np.float32))
        X = np.stack(cols, axis=1)
        X[X >= NODATA_U8] = np.nan
        return X

    pos_idx = np.flatnonzero(pos.ravel())
    y_pos = np.ones(pos_idx.size, np.int8)
    y_bg = np.zeros(bg.size, np.int8)
    X = np.vstack([X_of(pos_idx), X_of(bg)])
    y = np.concatenate([y_pos, y_bg])
    q = np.concatenate([quads.reshape(-1)[pos_idx], quads.reshape(-1)[bg]])
    del X_of

    # ------------------------------------------------ leakage canary (rule 4)
    canary = {}
    for j, nm in enumerate(names):
        v = X[:, j]
        ok = np.isfinite(v)
        aucs = []
        for qq in (0, 1, 2, 3):
            m = (q == qq) & ok
            if y[m].sum() < 10 or (~y[m].astype(bool)).sum() < 10:
                continue
            aucs.append(roc_auc_score(y[m], v[m]))
        disc = [max(a, 1.0 - a) for a in aucs]
        canary[nm] = dict(auc_per_fold=[float(a) for a in aucs],
                          discriminative_auc_mean=float(np.mean(disc)) if disc else None,
                          discriminative_auc_max=float(np.max(disc)) if disc else None,
                          leakage_flag=bool(np.max(disc) > 0.90) if disc else False)

    # --------------------------------------------- spatially blocked density ratio
    models, fold_auc = {}, {}
    for qq in (0, 1, 2, 3):
        tr = q != qq
        te = q == qq
        mdl = HistGradientBoostingClassifier(
            max_depth=8, max_iter=400, learning_rate=0.06, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=SEED)
        mdl.fit(X[tr], y[tr])
        models[qq] = mdl
        p = mdl.predict_proba(X[te])[:, 1]
        fold_auc[f"quad{qq}"] = float(roc_auc_score(y[te], p))
    del X, y, q

    # ------------------------------------------- out-of-fold score on the grid
    s = score_grid(models, quads)
    s[~fp] = np.nan
    auc_mean = float(np.mean(list(fold_auc.values())))
    print("blocked fold AUC:", {k: round(v, 4) for k, v in fold_auc.items()})
    np.save(os.path.join(OUT, "oof_score_f32.npy"), s)

    # ------------------------------------------------ aggregate calibration
    fin = np.isfinite(s)
    cuts = np.quantile(s[fin], np.linspace(0, 1, 9)[1:-1])
    binidx = np.full(s.shape, -1, np.int16)
    binidx[fin] = np.searchsorted(cuts, s[fin])
    nbin = len(cuts) + 1

    def counts(mask):
        b = binidx[mask]
        return np.bincount(b[b >= 0], minlength=nbin).astype(float)

    rows, rhs, wts = [], [], []
    for f, _, _ in LIVE:
        rows.append(counts(masks[f])); rhs.append(T_obs[f])
        wts.append(1.0 / max(T_obs[f], 50.0))
    rows.append(counts(fp)); rhs.append(G * SUM_K); wts.append(1.0 / (G * SUM_K))
    A = np.array(rows); b = np.array(rhs); w = np.array(wts)
    sol, _ = nnls(A * w[:, None], b * w)
    fitted = A @ sol

    # leave-one-raster-out check of the calibration
    loo = {}
    for k, (f, _, _) in enumerate(LIVE):
        keep = [i for i in range(len(LIVE)) if i != k] + [len(LIVE)]
        sk, _ = nnls(A[keep] * w[keep][:, None], b[keep] * w[keep])
        loo[f] = dict(observed=float(b[k]), loo_predicted=float(A[k] @ sk),
                      rel_error=float((A[k] @ sk - b[k]) / b[k]))
    loo["__mass__"] = dict(observed=float(b[-1]), loo_predicted=float((A[-1] @ (
        lambda: nnls(A[:-1] * w[:-1][:, None], b[:-1] * w[:-1])[0])())),
        rel_error=float(((A[-1] @ nnls(A[:-1] * w[:-1][:, None], b[:-1] * w[:-1])[0]) - b[-1]) / b[-1]))

    out = dict(
        schema="gems59.exp-live-credit-geophysics.v1",
        label="MODEL — anchored on OWNER-REPORTED scores, not organizer receipts",
        G_hidden_truth=float(G), positive_set=POS_FILE,
        n_positive=int(pos.sum()), n_background=int(bg.size),
        blocked_fold_auc=fold_auc, blocked_auc_mean=auc_mean,
        leakage_canary=canary,
        calibration=dict(n_bins=nbin, credit_per_cell=sol.tolist(),
                         observed=[float(x) for x in b], fitted=[float(x) for x in fitted],
                         rel_resid=[float((fitted[i] - b[i]) / b[i]) for i in range(len(b))],
                         raster_order=[f for f, _, _ in LIVE] + ["__footprint_mass__"]),
        leave_one_raster_out=loo,
    )
    path = os.path.join(REPO, "evidence", "exp5_live_credit_geophysics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("AUC mean %.4f" % auc_mean)
    print("calibration rel resid", [round(x, 3) for x in out["calibration"]["rel_resid"]])
    print("LOO rel err", {k: round(v["rel_error"], 3) for k, v in loo.items()})
    print("credit per cell by bin", [round(x, 5) for x in sol])
    top = sorted(canary.items(), key=lambda kv: -(kv[1]["discriminative_auc_mean"] or 0))[:8]
    print("top single-feature discriminative AUC:", [(k, round(v["discriminative_auc_mean"], 3)) for k, v in top])
    return out


if __name__ == "__main__":
    main()
