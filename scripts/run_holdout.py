#!/usr/bin/env python3
"""ARCHIVED H1/H52 experiment; not the active H57 workflow.

This legacy script fits the H1 vector-native faultzone model, sweeps its own
budget/ablations, and historically used ``gems52-pooled-hide-v1``. The shared
helper it imports has since been repaired/versioned as H57 v2; rerunning this
script will therefore not reproduce the archived H1 evaluator provenance.
Do not compare its outputs with ``scripts/run_lane.py`` / ``scripts/run_cv.py``
or overwrite the H1 archive with a current H57 score.

EXPERIMENT 2 (fault-zone anatomy lane): fit the intensity on the holdout,
sweep the dot budget, ablate every factor, run the leakage canary, and score
pooled DTI with the legacy template evaluator.

Arms (each is top-k binary dots on the fold's visible-only intensity):
  full          f(d/s(L)) * g(phi)          -- the lane's intensity
  no_phi        f(d/s(L))                   -- orientation factor ablated
  dist_only     f(d)                         -- length scaling ablated too
  uniform_halo  1 on every halo pixel        -- halo geometry alone (control)
  random        uniform random footprint    -- mass control

Every number written here is labelled HOLDOUT-DTI (evaluator version,
withheld positives, 95% CI) or MEASUREMENT.  No organizer score exists.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import faultzone as fz  # noqa: E402
from gems59 import fit as F  # noqa: E402
from gems59 import evaluate_holdout as EH  # noqa: E402
from gems59 import holdout as HO  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "evidence"
BASE = 100000
BUFFER_PX = 3
N_FOLDS = 4
PREVALENCE = 0.002
SEED = 20261009
D_MAX = F.D_MAX_PX
BUDGETS = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 120000]


def auc(score, truth, domain):
    sel = np.asarray(domain, bool)
    s = np.asarray(score, dtype=np.float64)[sel]
    t = np.asarray(truth, bool)[sel]
    n_pos = int(t.sum()); n_neg = int(t.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def halo_features(visible, seg_lab_v, seg_stats_v, valid, d_max=D_MAX):
    """Halo-restricted per-pixel geometry from visible faults only."""
    d, (iy, ix) = ndimage.distance_transform_edt(~visible, return_indices=True)
    halo = (d <= d_max) & valid & ~visible
    ys, xs = np.nonzero(halo)
    dv = d[ys, xs]
    ny, nx = iy[ys, xs], ix[ys, xs]
    sn = seg_lab_v[ny, nx].astype(np.int64)
    sn[~visible[ny, nx]] = 0
    dy = (ys - ny).astype(np.float64)
    dx = (xs - nx).astype(np.float64)
    az = np.degrees(np.arctan2(dx, dy)) % 180.0
    del d, iy, ix, dy, dx, ny, nx
    gc.collect()
    max_id = max(seg_stats_v) if seg_stats_v else 0
    strike_of = np.zeros(max_id + 1)
    vy_of = np.zeros(max_id + 1); vx_of = np.zeros(max_id + 1)
    cy_of = np.zeros(max_id + 1); cx_of = np.zeros(max_id + 1)
    span_of = np.ones(max_id + 1)
    len_of = np.zeros(max_id + 1)
    for sid, s in seg_stats_v.items():
        strike_of[sid] = s["strike"]
        vy_of[sid], vx_of[sid] = float(s["axis"][0]), float(s["axis"][1])
        cy_of[sid], cx_of[sid] = float(s["centroid"][0]), float(s["centroid"][1])
        span_of[sid] = max(s["span_px"], 1e-9)
        len_of[sid] = s["length_px"]
    phi = np.full(ys.size, np.nan)
    u = np.full(ys.size, np.nan)
    L = np.zeros(ys.size)
    has = sn > 0
    if has.any():
        snh = sn[has]
        rel = (az[has] - strike_of[snh]) % 180.0
        phi[has] = np.minimum(rel, 180.0 - rel)
        proj = vy_of[snh] * (ys[has] - cy_of[snh]) + vx_of[snh] * (xs[has] - cx_of[snh])
        u[has] = proj / span_of[snh] + 0.5
        L[has] = len_of[snh]
    del az, strike_of, vy_of, vx_of, cy_of, cx_of, span_of, len_of, sn, has
    gc.collect()
    return ys, xs, dv, phi, u, L


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with rasterio.open(DATA / "official/labels.tif") as ds:
        cat = ds.read(1) > 0
        transform = ds.transform
    with rasterio.open(DATA / "official/sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))

    print("[exp2] unified known-fault store ...", flush=True)
    ing_stats, ing_pix, tmap, _ = fz.ingenious_record_segments(
        DATA / "external/trace_segments_utm11.csv", cat.shape, transform)
    ingen_mask = tmap > 0
    seg_lab_cat, seg_stats_cat = fz.link_segments(cat & valid)
    n_cat = len(seg_stats_cat)
    seg_lab = np.zeros(cat.shape, dtype=np.int32)
    pix_lists = fz.segment_pixel_lists(seg_lab_cat, n_cat)
    for sid in range(1, n_cat + 1):
        ys, xs = pix_lists[sid - 1]
        seg_lab[ys, xs] = sid
    del seg_lab_cat, pix_lists
    gc.collect()
    for sid, (ys, xs) in ing_pix.items():
        seg_lab[ys, xs] = BASE + sid
    seg_stats = dict(seg_stats_cat)
    for sid, s in ing_stats.items():
        seg_stats[BASE + sid] = s
    known = (cat | ingen_mask) & valid
    del tmap, ingen_mask
    gc.collect()

    # ---- fitted factors from the exp1b holdout measurements ----
    pooled = np.load(OUT / "exp1b_pooled.npz")
    sgmc = np.load(OUT / "exp1b_sgmc.npz")
    d_h, phi_h, u_h, L_h = pooled["d"], pooled["phi"], pooled["u"], pooled["L"]
    f_dist = F.fit_distance_density(d_h)
    near_density = F.extend_near_band_sgmc(d_h, sgmc["d"])
    g_az = F.fit_azimuth_density(phi_h)
    tip = F.fit_tip_weight(u_h)
    scaling = F.fit_length_scaling(d_h, L_h)
    factors = dict(
        distance=dict(median_px=f_dist["median"], n=f_dist["n"],
                      hist_1px=f_dist["hist"].tolist()),
        near_band=dict(note="d<3px unmeasurable on the holdout (300 m truth buffer); "
                            "shape from SGMC off-catalogue, continuity-scaled",
                       density=near_density[:4].tolist()),
        azimuth=dict(median_deg=g_az["median"], n=g_az["n"],
                     hist_15deg=g_az["hist"].tolist()),
        tip_position=dict(frac_beyond_tip=tip["frac_beyond_tip"],
                          frac_inside=tip["frac_inside"],
                          note="descriptive; not a per-pixel weight (both classes occur "
                               "at all distances) -- kept out of the intensity, ablation "
                               "confirms"),
        length_scaling=scaling,
        sense=dict(note="holdout cannot resolve sense conditioning "
                        "(nearest-visible sense: unk=38063, N=214, RL=53, LL=9; SGMC agrees "
                        "in sign only for RL vs unk and disagrees for N) -- mechanism kept, "
                        "marginal weight, reported as unresolved",
                    holdout_n=dict(unk=38063, N=214, RL=53, LL=9)),
    )
    print(f"[exp2] fitted factors: d median {f_dist['median']:.1f}px, "
          f"phi median {g_az['median']:.0f}deg, beyond-tip {tip['frac_beyond_tip']:.3f}",
          flush=True)

    # ---- folds (same construction as exp1b) ----
    rng = np.random.default_rng(SEED)
    cat_ids = np.arange(1, n_cat + 1)
    assign = rng.integers(0, N_FOLDS, size=cat_ids.size)
    target = PREVALENCE * float(valid.sum())
    folds = []
    for f in range(N_FOLDS):
        ids = cat_ids[assign == f]
        order = rng.permutation(ids)
        keep = []
        tot = 0.0
        for sid in order:
            if tot >= target:
                break
            keep.append(int(sid))
            tot += float(seg_stats[sid]["size"])
        held = np.isin(seg_lab, keep)
        folds.append(dict(fold=f, held=held, kept=keep))

    arms = ["full", "no_phi", "dist_only", "uniform_halo", "random"]
    terms_by_arm = {a: [] for a in arms}
    budget_terms = {a: {} for a in arms}
    canary = {}
    fold_meta = []
    rng_emit = np.random.default_rng(SEED + 1)
    fp_flat = np.flatnonzero(valid.ravel())

    for f in folds:
        print(f"[exp2] fold {f['fold']} ...", flush=True)
        visible = known & ~f["held"]
        visible_cat = cat & ~f["held"] & valid
        # rebuild visible-only store (catalogue relinked + INGENIOUS attached)
        seg_lab_cat_v, seg_stats_cat_v = fz.link_segments(visible_cat & valid)
        n_cat_v = len(seg_stats_cat_v)
        seg_lab_v = np.zeros(cat.shape, dtype=np.int32)
        pl = fz.segment_pixel_lists(seg_lab_cat_v, n_cat_v)
        for sid in range(1, n_cat_v + 1):
            ys, xs = pl[sid - 1]
            seg_lab_v[ys, xs] = sid
        del seg_lab_cat_v, pl
        gc.collect()
        for sid, (ys, xs) in ing_pix.items():
            seg_lab_v[ys, xs] = BASE + sid
        seg_stats_v = dict(seg_stats_cat_v)
        for sid, s in ing_stats.items():
            seg_stats_v[BASE + sid] = s

        near_visible = ndimage.binary_dilation(visible, iterations=BUFFER_PX)
        truth = f["held"] & valid & ~near_visible
        fold_dict = dict(fold=f["fold"], mode="hide", truth=truth, visible=visible,
                         region=valid, fit=valid, boundary=near_visible,
                         n_truth=int(truth.sum()), n_held=int(f["held"].sum()))

        ys, xs, dv, phi, u, L = halo_features(visible, seg_lab_v, seg_stats_v, valid)
        sL = F.length_scale_lookup(L, scaling)
        d_eff = dv / sL
        f_d = F.density_at(d_eff, f_dist, near_density)
        g_p = F.azimuth_at(phi, g_az)
        intensity = f_d * g_p
        intensity = np.where(np.isfinite(intensity) & (intensity > 0), intensity, 0.0)

        # full-grid float32 fields per arm (halo-restricted)
        fields = {}
        base = np.zeros(valid.shape, dtype=np.float32)
        base[ys, xs] = intensity.astype(np.float32)
        fields["full"] = base
        b2 = np.zeros(valid.shape, dtype=np.float32)
        b2[ys, xs] = f_d.astype(np.float32)
        fields["no_phi"] = b2
        b3 = np.zeros(valid.shape, dtype=np.float32)
        b3[ys, xs] = F.density_at(dv, f_dist, near_density).astype(np.float32)
        fields["dist_only"] = b3
        b4 = np.zeros(valid.shape, dtype=np.float32)
        b4[ys, xs] = 1.0
        fields["uniform_halo"] = b4
        g_field = np.zeros(valid.shape, dtype=np.float32)
        g_field[ys, xs] = g_p.astype(np.float32)
        sL_field = np.zeros(valid.shape, dtype=np.float32)
        sL_field[ys, xs] = sL.astype(np.float32)

        allowed = valid & ~visible
        # leakage canary: each factor alone, AUC on the scored withheld pixels
        domain = np.zeros(valid.shape, bool)
        domain[ys, xs] = True
        ckey = "fold%d" % f["fold"]
        canary[ckey] = {
            "intensity_full": auc(base, truth, domain),
            "distance_density": auc(b3, truth, domain),
            "azimuth_density": auc(g_field, truth, domain),
            "length_scale": auc(sL_field, truth, domain),
        }
        # random arm field
        rand_field = np.zeros(valid.shape, dtype=np.float32)
        picks = rng_emit.choice(fp_flat, size=min(max(BUDGETS), fp_flat.size),
                                replace=False)
        tmp = np.zeros(valid.size, dtype=np.float32)
        tmp[picks] = 1.0
        rand_field = tmp.reshape(valid.shape)

        for arm in arms:
            field = rand_field if arm == "random" else fields[arm]
            per_budget = {}
            for b in BUDGETS:
                if arm == "random":
                    em = (field > 0).astype(np.float32)
                    if int((em > 0).sum()) > b:
                        idx = np.flatnonzero(em.ravel() > 0)
                        keep_idx = rng_emit.choice(idx, size=b, replace=False)
                        m = np.zeros(em.size, dtype=np.float32)
                        m[keep_idx] = 1.0
                        em = m.reshape(em.shape)
                else:
                    em = HO.emit_topk(field, allowed, b)
                result, terms = EH.evaluate(em, fold_dict, valid, block_side=200)
                per_budget[b] = terms
            budget_terms[arm][f"fold{f['fold']}"] = per_budget
        fold_meta.append(dict(fold=f["fold"], n_truth=int(truth.sum()),
                              halo_px=int(ys.size), allowed_px=int(allowed.sum())))
        print("    fold %d: truth %d px, halo %d px, canary full AUC %.4f" % (
            f["fold"], int(truth.sum()), int(ys.size),
            canary[ckey]["intensity_full"]), flush=True)
        del visible, visible_cat, seg_lab_v, seg_stats_v, near_visible, truth
        del fields, base, b2, b3, b4, rand_field, ys, xs, dv, phi, u, L, sL
        del f_d, g_p, intensity, d_eff, domain, allowed, g_field, sL_field
        gc.collect()

    # ---- pooled summaries per arm/budget ----
    # Contributions from the same physical block across folds are merged BEFORE
    # resampling (template rule): sum the per-fold block terms per block.
    print("[exp2] pooling ...", flush=True)
    merged = {a: {b: np.stack([budget_terms[a][f"fold{k}"][b]
                               for k in range(N_FOLDS)]).sum(axis=0)
                  for b in BUDGETS} for a in arms}
    summary = {}
    for arm in arms:
        arm_sum = {}
        for b in BUDGETS:
            stack = {arm: merged[arm][b]}
            controls = [a for a in arms if a != arm]
            for c in controls:
                stack[c] = merged[c][b]
            s = EH.pooled_summary(stack, draws=1000, seed=SEED + 7, candidate=arm)
            arm_sum[b] = s["scores"][arm]
            arm_sum[b]["paired_differences"] = s["paired_differences"]
        summary[arm] = arm_sum

    payload = dict(
        evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION,
        evaluator_implementation_hashes=EH.implementation_hashes(),
        protocol=dict(folds=N_FOLDS, fold_scheme="random whole catalogue segments",
                      buffer_px=BUFFER_PX, buffer_applied_to="truth",
                      mask_visible="pixel-exact (USGS+INGENIOUS known faults)",
                      prevalence_target=PREVALENCE, seed=SEED,
                      metric="DTI alpha=0.2 beta=0.8 R=300m triangular"),
        withheld_positive_pixels=sum(m["n_truth"] for m in fold_meta),
        fold_meta=fold_meta,
        fitted_factors=factors,
        budgets=BUDGETS,
        arms={arm: {str(b): summary[arm][b] for b in BUDGETS} for arm in arms},
        leakage_canary=canary,
        leakage_rule="AUC > 0.90 on withheld pixels = leakage until proven otherwise",
        runtime_s=time.time() - t0,
    )
    (OUT / "exp2_holdout.json").write_text(json.dumps(payload, indent=1, default=str))
    print("[exp2] wrote evidence/exp2_holdout.json", flush=True)
    for arm in arms:
        row = []
        for b in BUDGETS:
            s = summary[arm][b]
            row.append(f"{b}:{s['dti']:.4f}[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]")
        print(f"  {arm:12s} " + "  ".join(row), flush=True)
    print("  canary:", json.dumps(canary, indent=1), flush=True)


if __name__ == "__main__":
    raise SystemExit("Retired legacy/unbuffered runner: not current validation. "
                     "Reproduce the declared buffered experiment with scripts/run_orientation_experiments.py; "
                     "no extra experiment or slot is authorized by this command.")
