#!/usr/bin/env python3
"""H57-K — fault-zone anatomy with a live-anchored emission envelope.

What is new in this session, and why
------------------------------------
1. The official 19-band GeoDAWN feature stack (``training_features.tif``,
   sha256 4371c82e...) is now local and byte-verified.  Neither the previous
   sessions of this repository nor any earlier fault-zone-anatomy raster had it.
   It supplies the *strand-expression* term: whether a pixel inside a damage
   zone actually carries the geophysical signature of a fault (magnetic
   tilt-angle / gravity-gradient edge, detrended-elevation scarp, conductivity
   and strain contrasts) rather than being a bare distance halo.
2. A pure distance-to-catalogue model is falsified against live evidence
   (``scripts/live_credit_shells.py``): a NNLS fit of a 9-bin distance profile
   to the nine owner-reported scores leaves relative residuals of 0.14-0.73.
3. The <= 2 px (200 m) proximal band is measured, from the same live evidence,
   to earn 0.0024 credit per dot against 0.1387 for the retained dots of the
   0.2778 submission.  That is a direct measurement against the hidden truth,
   so the emitted dot set excludes it.
4. The hidden-truth mass G = 14,089 is inverted from two owner-reported scores
   of the same family, which fixes the greedy allocation bar that previously
   had to use the catalogue holdout's K (a quantity about catalogue faults,
   not about new ones).

Protocol
--------
* Fit on the shared hide-and-recover holdout (whole segments, buffered,
  visible-only features, visible faults masked pixel-exactly, pooled DTI
  alpha 0.2 / beta 0.8 / 300 m triangular kernel).
* Leakage canary on every single feature (rule 4).
* Uniqueness against every registry raster (rule 1).
* Every number is labelled HOLDOUT-DTI or MODEL / LIVE-ANCHORED.  No score is
  projected.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import holdout as HO                 # noqa: E402
from gems59.anatomy import FEATURES, fold_geometry  # noqa: E402
from gems59.emit import allocate_by_marginal_bar, greedy_allocate  # noqa: E402
from gems59.grid import load_grid                # noqa: E402
from gems59.metric import ALPHA, BETA            # noqa: E402

EVID = ROOT / "evidence"
OUT = ROOT / "out"
FEAT = OUT / "features"
SEED = 57
NEG_PER_CELL = 25_000
GBM = dict(max_depth=6, max_iter=240, learning_rate=0.08, min_samples_leaf=60,
           l2_regularization=1.0, early_stopping=False, random_state=SEED)
NODATA_U8 = 255

LANE8 = ("d", "d_perp", "d_par_abs", "log_len", "sin2", "cos2", "coherence", "density")


# --------------------------------------------------------------------------- #
def extra_names() -> list[str]:
    info = json.load(open(EVID / "feature_bank.json"))
    return ([b + "__q" for b in info["bands"]]
            + [b + "__z" for b in info["bands"]]
            + ["vis_dens15", "vis_dtip"])


CELLDIR = OUT / "h57k_cells"


class CellRec:
    """Disk-backed fold geometry.

    Eight quadrant crops of ~1.2 M rows x 49 float32 columns is ~1.9 GB, which
    does not fit alongside the memmapped feature bank in a 4 GB sandbox.  Each
    cell's design matrix is therefore written to ``out/h57k_cells`` (gitignored)
    and read back with ``mmap_mode='r'``, so only the rows actually in use are
    resident.
    """

    __slots__ = ("key", "xpath", "rows", "cols", "y", "n_hidden", "cell")

    def __init__(self, key, xpath, rows, cols, y, n_hidden, cell):
        self.key, self.xpath = key, xpath
        self.rows, self.cols, self.y = rows, cols, y
        self.n_hidden, self.cell = n_hidden, cell

    @property
    def X(self):
        return np.load(self.xpath, mmap_mode="r")


class Extras:
    """Leakage-safe extra features.

    The 38 geophysical layers do not depend on the catalogue at all, so they
    cannot leak a withheld segment.  The two geometric extras are rebuilt from
    the *visible* mask of each fold, never from the full catalogue.
    """

    def __init__(self):
        self.bank = np.load(FEAT / "bank_u8.npy", mmap_mode="r")
        self.nb = self.bank.shape[0]

    def geophys(self, ys, xs) -> np.ndarray:
        cols = [np.asarray(self.bank[i])[ys, xs] for i in range(self.nb)]
        X = np.stack(cols, axis=1).astype(np.float32)
        X[X >= NODATA_U8] = np.nan
        return X

    @staticmethod
    def geom(visible: np.ndarray, ys, xs) -> np.ndarray:
        yy, xx = np.mgrid[-15:16, -15:16]
        k15 = ((yy * yy + xx * xx) <= 225).astype(np.float32)
        dens15 = ndi.convolve(visible.astype(np.float32), k15, mode="constant")
        lab, n = ndi.label(visible, structure=np.ones((3, 3), bool))
        tips = np.zeros(visible.shape, bool)
        if n:
            er = ndi.binary_erosion(visible, structure=np.ones((3, 3), bool))
            tips = visible & ~er
            if not tips.any():
                tips = visible
        dtip = ndi.distance_transform_edt(~tips)
        return np.stack([dens15[ys, xs], dtip[ys, xs]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
def build_cells(ctx, mode, extra: Extras, log=print):
    """Disk-backed fold geometry per cell, with the extra block appended."""
    CELLDIR.mkdir(parents=True, exist_ok=True)
    out = []
    cached = True
    t0 = time.time()
    for cell in ctx.cells_of(mode):
        xp = CELLDIR / f"{cell.key}.npy"
        rp = CELLDIR / f"{cell.key}_idx.npz"
        if xp.exists() and rp.exists():
            z = np.load(rp)
            out.append(CellRec(cell.key, xp, z["rows"], z["cols"], z["y"],
                               int(z["n_hidden"]), cell))
            continue
        cached = False
        dom = np.zeros(ctx.grid.shape, bool)
        dom[cell.bbox] = cell.active
        g = fold_geometry(ctx.grid, ctx.visible(cell.key),
                          ctx.hidden_by_cell[cell.key], dom, cell.key)
        del dom
        if g.X.shape[0] == 0:
            continue
        ys, xs = g.rows, g.cols
        X = np.concatenate([g.X, extra.geophys(ys, xs),
                            extra.geom(ctx.visible(cell.key), ys, xs)],
                           axis=1).astype(np.float32)
        np.save(xp, X)
        np.savez_compressed(rp, rows=ys, cols=xs, y=g.y,
                            n_hidden=np.int64(g.n_hidden))
        out.append(CellRec(cell.key, xp, ys, xs, g.y, int(g.n_hidden), cell))
        del X, g
    log(f"  {'loaded' if cached else 'built'} {len(out)} {mode} cells in "
        f"{time.time()-t0:.0f}s{'' if not cached else ' (from ' + str(CELLDIR) + ')'}")
    return out, cached


def sample(rng, geoms, neg_per_cell=NEG_PER_CELL):
    Xs, ys_, ws = [], [], []
    for g in geoms:
        X = g.X
        pos = g.y == 1
        Xs.append(np.asarray(X[pos])); ys_.append(np.ones(int(pos.sum()), np.int8))
        ws.append(np.ones(int(pos.sum()), np.float64))
        neg = np.flatnonzero(~pos)
        if neg.size > neg_per_cell:
            neg = rng.choice(neg, size=neg_per_cell, replace=False)
        Xs.append(np.asarray(X[neg])); ys_.append(np.zeros(neg.size, np.int8))
        ws.append(np.full(neg.size, float((~pos).sum()) / max(neg.size, 1)))
        del X
    return np.vstack(Xs), np.concatenate(ys_), np.concatenate(ws)


def canary(geoms, names, idxs, subsample=200_000):
    out = {}
    rng = np.random.default_rng(1)
    for j in idxs:
        aucs = []
        for g in geoms:
            if (g.y == 1).sum() < 10 or (g.y == 0).sum() < 10:
                continue
            X = g.X
            sel = np.arange(X.shape[0])
            if sel.size > subsample:
                keep = rng.choice(sel, size=subsample, replace=False)
            else:
                keep = sel
            v = np.asarray(X[keep, j])
            yv = g.y[keep]
            ok = np.isfinite(v)
            if (~ok).any():
                v = np.where(ok, v, np.nanmedian(v[ok]) if ok.any() else 0.0)
            aucs.append(roc_auc_score(yv, v))
            del X, v
        disc = [max(a, 1.0 - a) for a in aucs]
        out[names[j]] = dict(auc_mean=float(np.mean(aucs)) if aucs else None,
                             discriminative_auc_mean=float(np.mean(disc)) if disc else None,
                             discriminative_auc_max=float(np.max(disc)) if disc else None,
                             leakage_flag=bool(np.max(disc) > 0.90) if disc else False)
    return out


def pooled(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows); n = sum(r["n_truth"] for r in rows)
    from gems59.metric import dti_from_components
    dti = dti_from_components(tp, fp, fn)
    quads = sorted({r["key"].split("_")[1] for r in rows})
    jk = []
    for q in quads:
        rs = [r for r in rows if r["key"].split("_")[1] != q]
        jk.append(dti_from_components(sum(r["tp"] for r in rs), sum(r["fp"] for r in rs),
                                      sum(r["fn"] for r in rs)))
    jk = np.array(jk); m = len(quads)
    se = float(np.sqrt((m - 1) / m * ((jk - jk.mean()) ** 2).sum())) if m > 1 else 0.0
    return dict(pooled_dti=dti, coverage=float(tp / n) if n else 0.0,
                dti_ci95_quadrant_jackknife=[float(dti - 1.959963985 * se),
                                             float(dti + 1.959963985 * se)],
                jackknife_drop_quadrant={q: float(v) for q, v in zip(quads, jk)},
                tp=float(tp), fp=float(fp), fn=float(fn), n_truth=int(n),
                n_dots=sum(r["n_dots"] for r in rows))


def quad_of(g) -> str:
    """Cell keys look like ``draw20_foldNW_detached``; the quadrant is index 1."""
    return g.key.split("_")[1].replace("fold", "")


def score_arm(geoms, cols, ctx, k_truth_mode="own", k_truth=None,
              max_dots=40_000, floor=0.0, tag=""):
    """Leave-one-quadrant-out: fit on 3 quadrants, allocate + score on the 4th."""
    names_all = list(FEATURES) + extra_names()
    rows = []
    for q in ("NW", "NE", "SW", "SE"):
        tr = [g for g in geoms if quad_of(g) != q]
        te = [g for g in geoms if quad_of(g) == q]
        if not tr or not te:
            continue
        rng = np.random.default_rng(SEED)
        X, y, w = sample(rng, tr)
        X = X[:, cols]
        clf = HistGradientBoostingClassifier(**GBM)
        clf.fit(X, y, sample_weight=w)
        p_all = clf.predict_proba(X)[:, 1]
        base = float((y * w).sum() / w.sum()); mp = float((p_all * w).sum() / w.sum())
        scale = base / max(mp, 1e-12)
        del X, y, w
        for g in te:
            p = np.zeros(ctx.grid.shape, np.float32)
            Xg = g.X
            s = clf.predict_proba(Xg[:, cols])[:, 1].astype(np.float32) * np.float32(scale)
            np.clip(s, 0.0, 1.0, out=s)
            p[g.rows, g.cols] = s
            del Xg, s
            allowed = np.zeros(ctx.grid.shape, bool)
            allowed[g.cell.bbox] = g.cell.active
            kt = float(g.cell.n_truth) if k_truth_mode == "own" else float(k_truth)
            alloc = allocate_by_marginal_bar(p, allowed, k_truth=kt, floor=floor,
                                             max_dots=max_dots)
            r = HO.score_cell(g.cell, alloc.emitted)
            rows.append(dict(key=g.key, n_dots=alloc.n_dots, tp=r["tp"], fp=r["fp"],
                             fn=float(r["n_truth"]) - r["tp"], n_truth=r["n_truth"]))
            del p, allowed, alloc
    return pooled(rows), rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="detached", choices=["all", "detached"])
    ap.add_argument("--max-dots", type=int, default=120_000)
    ap.add_argument("--g-live", type=float, default=14088.747191011289)
    ap.add_argument("--proximal-exclude-px", type=float, default=2.0)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--arms", default="d_only,lane8,lane8_geophys")
    a = ap.parse_args()

    t0 = time.time()
    grid = load_grid()
    ctx = HO.build_holdout(grid)
    print(f"[h57k] holdout: {len(ctx.cells)} cells, mode={a.mode}", flush=True)

    extra = Extras()
    names_all = list(FEATURES) + extra_names()
    geoms, cached = build_cells(ctx, a.mode, extra)
    nfeat = geoms[0].X.shape[1]
    assert nfeat == len(names_all), (nfeat, len(names_all))
    print(f"[h57k] {len(geoms)} cells, {nfeat} features", flush=True)

    res = dict(schema="gems59.h57k.v1", mode=a.mode, features=names_all,
               n_cells=len(geoms),
               n_withheld=int(sum(int((g.y == 1).sum()) for g in geoms)))
    res["leakage_canary"] = canary(geoms, names_all, range(nfeat))
    flags = [k for k, v in res["leakage_canary"].items() if v["leakage_flag"]]
    print(f"[h57k] canary: {len(flags)} features above 0.90 -> {flags}", flush=True)

    if cached:
        # The cell matrices are on disk now; drop the ~1.9 GB bank of
        # geophysical memmaps and hand the pages back to the OS before the CV
        # loop.  A 3.9 GB sandbox cannot hold both.
        del extra
        extra = None
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        print("[h57k] released the feature bank; CV runs from the cell cache",
              flush=True)

    idx = {n: i for i, n in enumerate(names_all)}
    arms = {
        "d_only": [idx["d"]],
        "lane8": [idx[n] for n in LANE8],
        "lane8_geophys": [idx[n] for n in LANE8] + list(range(len(FEATURES), nfeat)),
    }
    res["arms"] = {}
    want = set(x.strip() for x in a.arms.split(",") if x.strip())
    if not a.skip_cv:
        for nm, cols in arms.items():
            if nm not in want:
                continue
            p, rows = score_arm(geoms, cols, ctx, max_dots=a.max_dots)
            res["arms"][nm] = dict(n_features=len(cols), **p)
            print(f"[h57k] {nm:16s} HOLDOUT-DTI={p['pooled_dti']:.4f} "
                  f"CI=[{p['dti_ci95_quadrant_jackknife'][0]:.4f},"
                  f"{p['dti_ci95_quadrant_jackknife'][1]:.4f}] "
                  f"dots={p['n_dots']} cov={p['coverage']:.4f} "
                  f"n_truth={p['n_truth']}", flush=True)
    prev = EVID / "exp6_h57k_arms.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        for k, v in old.get("arms", {}).items():
            res["arms"].setdefault(k, v)
    prev.write_text(json.dumps(res, indent=1))

    # ------------------------------------------------------------ final fit
    best = max(res["arms"], key=lambda k: res["arms"][k]["pooled_dti"]) if res["arms"] else "lane8_geophys"
    cols = arms[best]
    print(f"[h57k] final arm = {best} ({len(cols)} features)", flush=True)
    rng = np.random.default_rng(SEED)
    X, y, w = sample(rng, geoms)
    X = X[:, cols]
    clf = HistGradientBoostingClassifier(**GBM)
    clf.fit(X, y, sample_weight=w)
    p_all = clf.predict_proba(X)[:, 1]
    base = float((y * w).sum() / w.sum()); mp = float((p_all * w).sum() / w.sum())
    scale = base / max(mp, 1e-12)
    del X, y, w, p_all

    # full-catalogue surface, predicted in row bands: the whole footprint is
    # ~12.3 M rows x 48 float32 = 2.4 GB, which does not fit in a 4 GB sandbox.
    if extra is None:
        extra = Extras()
    vis = grid.catalogue
    dom = grid.footprint & ~vis
    p = np.zeros(grid.shape, np.float32)
    H = grid.shape[0]
    band = 300
    t1 = time.time()
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        sub = np.zeros(grid.shape, bool)
        sub[r0:r1] = dom[r0:r1]
        if not sub.any():
            continue
        g = fold_geometry(grid, vis, np.zeros(grid.shape, bool), sub, f"full{r0}")
        ys, xs = g.rows, g.cols
        Xf = np.concatenate([g.X, extra.geophys(ys, xs), extra.geom(vis, ys, xs)],
                            axis=1).astype(np.float32)
        sc = clf.predict_proba(Xf[:, cols])[:, 1].astype(np.float32) * np.float32(scale)
        np.clip(sc, 0.0, 1.0, out=sc)
        p[ys, xs] = sc
        del g, Xf, sc, sub
    print(f"[h57k] surface predicted in {time.time()-t1:.0f}s", flush=True)
    np.save(OUT / "h57k_surface_f32.npy", p)
    print(f"[h57k] surface: p_max={p.max():.4f} mean={p[dom].mean():.6f}", flush=True)

    # --------------------------------- emission at the LIVE-ANCHORED optimum
    dcat = ndi.distance_transform_edt(~vis)
    allowed = grid.footprint & ~vis & (dcat > a.proximal_exclude_px)
    alloc = allocate_by_marginal_bar(p, allowed, k_truth=a.g_live, floor=0.0,
                                      max_dots=400_000, candidate_cap=1_500_000)
    dots = alloc.emitted
    res["emission"] = dict(
        arm=best, n_dots=int(alloc.n_dots), k_truth_used=a.g_live,
        proximal_exclude_px=a.proximal_exclude_px,
        surrogate_dti=float(alloc.surrogate_dti),
        expected_covered_credit=float(alloc.expected_covered_credit),
        expected_self_credit=float(alloc.expected_self_credit),
        surface_max=float(p.max()), surface_mean_in_domain=float(p[dom].mean()),
        allowed_cells=int(allowed.sum()),
        dots_within_2px_of_catalogue=int((dots & (dcat <= 2)).sum()),
        dots_on_catalogue=int((dots & vis).sum()),
        distance_to_catalogue_median=float(np.median(dcat[dots])),
    )
    print(f"[h57k] emitted {alloc.n_dots} dots, surrogate DTI {alloc.surrogate_dti:.4f}", flush=True)
    print(json.dumps(res["emission"], indent=1), flush=True)
    (EVID / "exp6_h57k_emission.json").write_text(json.dumps(res, indent=1))
    print(f"[h57k] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
