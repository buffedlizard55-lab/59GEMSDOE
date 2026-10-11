#!/usr/bin/env python3
"""ARCHIVED H1/H52 EXPERIMENT 1b; not the active H57 workflow.

This legacy random whole-segment measurement uses a different experimental
path and historical outputs. Keep its artifacts separate from ``run_lane.py`` /
``run_cv.py``; do not cite them as current H57 evidence or as an organizer score.

EXPERIMENT 1b (fault-zone anatomy lane): the lane's true holdout.

Whole fault SEGMENTS (linked catalogue traces + INGENIOUS record faults) are
withheld in RANDOM folds.  A spatially-blocked hide cannot see intra-system
strands: withheld segments would sit across block boundaries, tens of km from
any visible fault (measured in exp1: median 194 px).  Random whole-segment
folds keep unseen strands in the same neighbourhoods as seen ones -- exactly
the population the organizer's truth is drawn from ("newly mapped geometry of
an existing system, so splays and parallel strands count", thread 11536).

Protocol (the brief's): withhold whole segments with a 300 m buffer (hidden
pixels within 300 m of a visible fault are not scored), derive every
catalogue-based feature from the VISIBLE faults only, mask visible faults
pixel-exactly (plus INGENIOUS, which the organizer also masks).

Measurements on the scored (buffered) withheld pixels:
  d, phi, u, L*, sense of the nearest VISIBLE segment, and segment-level
  dstrike.  Cross-check population: SGMC faults captured by neither USGS nor
  INGENIOUS (real off-catalogue faults).

Outputs: evidence/exp1b_zone_distributions.json (MEASUREMENT)
         evidence/exp1b_pooled.npz, evidence/exp1b_sgmc.npz (fitting inputs)
"""
from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import faultzone as fz  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "evidence"
BASE = 100000          # INGENIOUS segment-id offset
BUFFER_PX = 3          # 300 m metric halo, applied to the TRUTH (the brief's buffer)
N_FOLDS = 4
PREVALENCE = 0.002     # template's target: |G| bracket / footprint
SEED = 20261009


def build_known(cat, valid, transform):
    """Catalogue segments + INGENIOUS record segments -> unified store."""
    seg_lab_cat, seg_stats_cat = fz.link_segments(cat & valid)
    ing_stats, ing_pix, tmap, trace_sense = fz.ingenious_record_segments(
        DATA / "external/trace_segments_utm11.csv", cat.shape, transform)
    ingen_mask = tmap > 0
    n_cat = len(seg_stats_cat)
    seg_lab = np.zeros(cat.shape, dtype=np.int32)
    pix_lists = fz.segment_pixel_lists(seg_lab_cat, n_cat)
    for sid in range(1, n_cat + 1):
        ys, xs = pix_lists[sid - 1]
        seg_lab[ys, xs] = sid
    del seg_lab_cat, pix_lists
    gc.collect()
    for sid, (ys, xs) in ing_pix.items():
        seg_lab[ys, xs] = BASE + sid          # INGENIOUS wins ties (vector truth)
    seg_stats = dict(seg_stats_cat)
    for sid, s in ing_stats.items():
        seg_stats[BASE + sid] = s
    sense_of = {sid: "unk" for sid in seg_stats_cat}
    for sid, s in ing_stats.items():
        sense_of[BASE + sid] = s["sense"]
    return seg_lab, seg_stats, sense_of, ingen_mask, n_cat, ing_stats, ing_pix


def visible_store(visible_cat, valid, ing_stats, ing_pix):
    """Rebuild catalogue segments from VISIBLE catalogue pixels only and
    re-attach the (never-hidden) INGENIOUS record segments."""
    seg_lab_cat, seg_stats_cat = fz.link_segments(visible_cat & valid)
    n_cat = len(seg_stats_cat)
    seg_lab = np.zeros(visible_cat.shape, dtype=np.int32)
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
    sense_of = {sid: "unk" for sid in seg_stats_cat}
    for sid, s in ing_stats.items():
        sense_of[BASE + sid] = s["sense"]
    return seg_lab, seg_stats, sense_of


def measure(truth, visible, seg_lab_v, seg_stats_v, sense_of, valid):
    """Distributions of truth pixels vs nearest visible segment (1-D arrays)."""
    d, (iy, ix) = ndimage.distance_transform_edt(~visible, return_indices=True)
    t = truth & valid
    ys, xs = np.nonzero(t)
    if ys.size == 0:
        del d, iy, ix
        return None
    dv = d[ys, xs]
    ny, nx = iy[ys, xs], ix[ys, xs]
    sn = seg_lab_v[ny, nx].astype(np.int64)
    sn[~visible[ny, nx]] = 0
    dy = (ys - ny).astype(np.float64)
    dx = (xs - nx).astype(np.float64)
    az = np.degrees(np.arctan2(dx, dy)) % 180.0
    del d, iy, ix, dy, dx
    gc.collect()
    phi = np.full(ys.size, np.nan)
    u = np.full(ys.size, np.nan)
    L = np.zeros(ys.size)
    se = np.array(["unk"] * ys.size, dtype=object)
    has = sn > 0
    if has.any():
        sids = sn[has]
        rel = (az[has] - np.array([seg_stats_v[s]["strike"] for s in sids])) % 180.0
        phi[has] = np.minimum(rel, 180.0 - rel)
        vy = np.array([seg_stats_v[s]["axis"][0] for s in sids])
        vx = np.array([seg_stats_v[s]["axis"][1] for s in sids])
        cy = np.array([seg_stats_v[s]["centroid"][0] for s in sids])
        cx = np.array([seg_stats_v[s]["centroid"][1] for s in sids])
        span = np.array([max(seg_stats_v[s]["span_px"], 1e-9) for s in sids])
        proj = vy * (ys[has] - cy) + vx * (xs[has] - cx)
        u[has] = proj / span + 0.5
        L[has] = np.array([seg_stats_v[s]["length_px"] for s in sids])
        se[has] = [sense_of.get(int(s), "unk") for s in sids]
    return dict(ys=ys, xs=xs, d=dv, phi=phi, u=u, L=L, sense=se)


def dstrike_of_truth(truth, valid, seg_stats_v):
    """Segment-level |strike difference| of truth segments vs nearest visible."""
    tlab, tstats = fz.link_segments(truth & valid)
    if not tstats or not seg_stats_v:
        return []
    tc = np.array([s["centroid"] for s in tstats.values()])
    vc = np.array([s["centroid"] for s in seg_stats_v.values()])
    vstrike = np.array([s["strike"] for s in seg_stats_v.values()])
    tree = cKDTree(vc)
    _, idx = tree.query(tc, k=1)
    tstrike = np.array([s["strike"] for s in tstats.values()])
    diff = np.abs(tstrike - vstrike[idx]) % 180.0
    return list(np.minimum(diff, 180.0 - diff))


def summarize(name, m):
    d = m["d"]
    inside = (m["u"] >= 0) & (m["u"] <= 1)
    return dict(
        name=name, n_px=int(d.size),
        d_px=dict(median=float(np.median(d)), p25=float(np.percentile(d, 25)),
                  p75=float(np.percentile(d, 75)), p90=float(np.percentile(d, 90)),
                  frac_le1=float((d <= 1.5).mean()), frac_le2=float((d <= 2.5).mean()),
                  frac_le3=float((d <= 3.5).mean()), frac_le5=float((d <= 5.5).mean()),
                  frac_le10=float((d <= 10.5).mean()), frac_le20=float((d <= 20.5).mean()),
                  frac_le50=float((d <= 50.5).mean()), frac_gt100=float((d > 100.5).mean())),
        d_hist_1px=[int(c) for c in np.histogram(d, bins=np.arange(0, 61, 1))[0]],
        phi_deg=dict(median=float(np.nanmedian(m["phi"])),
                     frac_le15=float(np.nanmean(m["phi"] <= 15)),
                     frac_15_45=float(np.nanmean((m["phi"] > 15) & (m["phi"] <= 45))),
                     frac_45_75=float(np.nanmean((m["phi"] > 45) & (m["phi"] <= 75))),
                     frac_gt75=float(np.nanmean(m["phi"] > 75))),
        u=dict(frac_beyond_tip=float(np.nanmean(~inside)),
               frac_inside=float(np.nanmean(inside))),
        L_near_px=dict(median=float(np.median(m["L"][m["L"] > 0])) if (m["L"] > 0).any() else None),
        sense_of_nearest_visible={k: int(v) for k, v in Counter(m["sense"]).most_common()},
    )


def main():
    OUT.mkdir(exist_ok=True)
    with rasterio.open(DATA / "official/labels.tif") as ds:
        cat = ds.read(1) > 0
        transform = ds.transform
    with rasterio.open(DATA / "official/sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))

    print("[exp1b] building unified known-fault segment store ...", flush=True)
    seg_lab, seg_stats, sense_of, ingen_mask, n_cat, ing_stats, ing_pix = \
        build_known(cat, valid, transform)
    known = (cat | ingen_mask) & valid
    print(f"  catalogue segments {n_cat}, INGENIOUS record segments {len(ing_stats)}, "
          f"known px {int(known.sum())}", flush=True)
    print("  INGENIOUS sense classes:",
          Counter(s['sense'] for s in ing_stats.values()).most_common(), flush=True)
    lens = np.array([s["length_px"] for s in seg_stats.values()])
    print(f"  segment lengths px: median {np.median(lens):.0f} "
          f"p90 {np.percentile(lens, 90):.0f} max {lens.max():.0f}", flush=True)

    # random whole-catalogue-segment folds, prevalence-thinned (template rule)
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
        folds.append(dict(fold=f, held=held, kept=keep, n_truth_raw=int(held.sum())))
        print(f"  fold {f}: held {int(held.sum())} px in {len(keep)} segments", flush=True)

    results = {"protocol": {"folds": N_FOLDS, "buffer_px": BUFFER_PX,
                            "prevalence_target": PREVALENCE, "seed": SEED,
                            "fold_scheme": "random whole catalogue segments"},
               "segments": {"catalogue": n_cat, "ingenious": len(ing_stats),
                            "length_median_px": float(np.median(lens)),
                            "length_max_px": float(lens.max())},
               "folds": []}
    pooled = {"d": [], "phi": [], "u": [], "L": [], "sense": []}
    for f in folds:
        visible = known & ~f["held"]
        visible_cat = cat & ~f["held"] & valid
        seg_lab_v, seg_stats_v, sense_v = visible_store(visible_cat, valid,
                                                         ing_stats, ing_pix)
        near_visible = ndimage.binary_dilation(visible, iterations=BUFFER_PX)
        truth = f["held"] & valid & ~near_visible
        m = measure(truth, visible, seg_lab_v, seg_stats_v, sense_v, valid)
        dsk = dstrike_of_truth(truth, valid, seg_stats_v)
        results["folds"].append(dict(
            fold=f["fold"], n_held_raw=f["n_truth_raw"], n_scored=int(truth.sum()),
            n_buffered_out=int((f['held'] & valid & near_visible).sum()),
            d_median=float(np.median(m["d"])) if m else None,
            d_frac_le3=float((m["d"] <= 3.5).mean()) if m else None,
            d_frac_le10=float((m["d"] <= 10.5).mean()) if m else None,
            d_frac_le20=float((m["d"] <= 20.5).mean()) if m else None,
            d_frac_le50=float((m["d"] <= 50.5).mean()) if m else None,
            dstrike_median=float(np.median(dsk)) if dsk else None,
            dstrike_n=len(dsk)))
        if m:
            for k in pooled:
                pooled[k].append(m[k])
            del m
        del visible, visible_cat, seg_lab_v, seg_stats_v, sense_v, near_visible, truth
        gc.collect()

    d_all = np.concatenate(pooled["d"])
    phi_all = np.concatenate(pooled["phi"])
    u_all = np.concatenate(pooled["u"])
    L_all = np.concatenate(pooled["L"])
    sense_all = np.concatenate(pooled["sense"])
    hp = summarize("holdout_pooled", dict(d=d_all, phi=phi_all, u=u_all, L=L_all,
                                           sense=sense_all))
    cond = {}
    for cls in ("RL", "LL", "N", "unk"):
        sel = sense_all == cls
        if sel.sum() < 30:
            cond[cls] = {"n": int(sel.sum()), "note": "too few for a conditioned fit"}
            continue
        dv = d_all[sel]
        cond[cls] = dict(n=int(sel.sum()), median=float(np.median(dv)),
                         frac_le3=float((dv <= 3.5).mean()),
                         frac_le10=float((dv <= 10.5).mean()),
                         frac_le20=float((dv <= 20.5).mean()))
    hp["sense_conditioned_distance"] = cond
    Lc = {}
    for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 100), (100, 1e9)):
        sel = (L_all >= lo) & (L_all < hi)
        if sel.sum() < 30:
            continue
        dv = d_all[sel]
        Lc[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = dict(
            n=int(sel.sum()), median=float(np.median(dv)),
            frac_le3=float((dv <= 3.5).mean()), frac_le10=float((dv <= 10.5).mean()),
            frac_le20=float((dv <= 20.5).mean()), frac_le50=float((dv <= 50.5).mean()))
    hp["length_conditioned_distance"] = Lc
    results["holdout_pooled"] = hp

    # cross-check: SGMC faults off the known set (real off-catalogue faults)
    with rasterio.open(DATA / "external/derived_sgmc_faults_100m.tif") as ds:
        sgmc = ds.read(1) > 0
    sgmc_off = sgmc & ~known & valid
    m = measure(sgmc_off, known, seg_lab, seg_stats, sense_of, valid)
    sm = summarize("sgmc_off_known", m)
    dsk = dstrike_of_truth(sgmc_off, valid, seg_stats)
    sm["dstrike_deg"] = dict(median=float(np.median(dsk)) if dsk else None, n=len(dsk),
                             frac_le10=float(np.mean(np.array(dsk) <= 10)) if dsk else None,
                             frac_le30=float(np.mean(np.array(dsk) <= 30)) if dsk else None)
    results["sgmc_off_known"] = sm
    np.savez_compressed(OUT / "exp1b_sgmc.npz", d=m["d"], phi=m["phi"], u=m["u"],
                        L=m["L"], sense=m["sense"].astype(str))
    del m, sgmc, sgmc_off, seg_lab, seg_stats, sense_of
    gc.collect()

    np.savez_compressed(OUT / "exp1b_pooled.npz", d=d_all, phi=phi_all, u=u_all,
                        L=L_all, sense=sense_all.astype(str))
    (OUT / "exp1b_zone_distributions.json").write_text(json.dumps(results, indent=1, default=str))
    print("[exp1b] wrote evidence/exp1b_zone_distributions.json + pooled npz files", flush=True)
    print("  HOLDOUT d: median %.1f px | <=3px %.4f  <=10px %.4f  <=20px %.4f  <=50px %.4f  >100px %.3f" % (
        hp["d_px"]["median"], hp["d_px"]["frac_le3"], hp["d_px"]["frac_le10"],
        hp["d_px"]["frac_le20"], hp["d_px"]["frac_le50"], hp["d_px"]["frac_gt100"]), flush=True)
    print("  HOLDOUT phi: median %.0f | <=15 %.3f  15-45 %.3f  45-75 %.3f  >75 %.3f" % (
        hp["phi_deg"]["median"], hp["phi_deg"]["frac_le15"], hp["phi_deg"]["frac_15_45"],
        hp["phi_deg"]["frac_45_75"], hp["phi_deg"]["frac_gt75"]), flush=True)
    print("  HOLDOUT u: beyond-tip %.3f  inside %.3f" % (
        hp["u"]["frac_beyond_tip"], hp["u"]["frac_inside"]), flush=True)
    print("  HOLDOUT sense of nearest visible:", hp["sense_of_nearest_visible"], flush=True)
    print("  SGMC-off d: median %.1f px | <=3px %.4f  <=10px %.4f  <=20px %.4f  <=50px %.4f  >100px %.3f" % (
        sm["d_px"]["median"], sm["d_px"]["frac_le3"], sm["d_px"]["frac_le10"],
        sm["d_px"]["frac_le20"], sm["d_px"]["frac_le50"], sm["d_px"]["frac_gt100"]), flush=True)
    print("  SGMC-off phi: median %.0f | <=15 %.3f  15-45 %.3f  45-75 %.3f  >75 %.3f" % (
        sm["phi_deg"]["median"], sm["phi_deg"]["frac_le15"], sm["phi_deg"]["frac_15_45"],
        sm["phi_deg"]["frac_45_75"], sm["phi_deg"]["frac_gt75"]), flush=True)
    print("  SGMC-off u: beyond-tip %.3f  inside %.3f" % (
        sm["u"]["frac_beyond_tip"], sm["u"]["frac_inside"]), flush=True)
    print("  SGMC-off dstrike:", sm["dstrike_deg"], flush=True)
    print("  length-conditioned d (holdout):", json.dumps(Lc), flush=True)


if __name__ == "__main__":
    main()
