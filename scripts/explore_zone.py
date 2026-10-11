#!/usr/bin/env python3
"""ARCHIVED H1/H52 exploration prototype; not the active H57 workflow.

Do not run as-is: this script calls ``faultzone.assign_sense_from_tracemap``,
which is not defined in the checked-out module. Its measurements and outputs
are historical and are not a current H57 score or validation.

EXPERIMENT 1 (fault-zone anatomy lane): where do withheld fault segments
sit relative to the nearest VISIBLE fault, and does the same hold for the real
off-catalogue population we can measure (SGMC faults that are captured by
neither USGS nor INGENIOUS)?

Known faults = USGS/INGENIOUS (the organizer masks both: thread 11516/11536).
The catalogue raster is USGS; the INGENIOUS vector traces (with recorded sense
of slip) are a second known population rasterized here.

Measurements (all from visible-only geometry per fold):
  d        distance of a withheld pixel to the nearest visible fault pixel
  phi      azimuth of the offset (0 = along strike, 90 = across strike)
  u        along-strike position vs the nearest visible segment (0..1 inside,
           <0/>1 beyond a tip)
  dstrike  |strike(hidden segment) - strike(nearest visible segment)|
  L*       length of the nearest visible segment (displacement proxy)
  sense    sense of slip of the nearest visible segment where recorded

Outputs: evidence/exp1_zone_distributions.json (MEASUREMENT, not a score)
         evidence/exp1_pooled.npz (raw pooled values for the fitting step)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import faultzone as fz  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "evidence"


def block_ids(shape, valid, n=4):
    h, w = shape
    rs = np.linspace(0, h, n + 1).astype(int)
    cs = np.linspace(0, w, n + 1).astype(int)
    lab = np.full(shape, -1, dtype=np.int16)
    k = 0
    for i in range(n):
        for j in range(n):
            blk = np.zeros(shape, dtype=bool)
            blk[rs[i]:rs[i + 1], cs[j]:cs[j + 1]] = True
            lab[blk & valid] = k
            k += 1
    return lab


def make_segment_folds(seg_lab, seg_stats, valid, hiddable, n_folds=4,
                       prevalence=0.002, seed=0):
    """Whole-segment hide-and-recover folds (template rules, segment units).

    Only segments flagged ``hiddable`` (pure-catalogue) may be withheld;
    INGENIOUS segments are public knowledge and stay visible in every fold.
    """
    lab = block_ids(seg_lab.shape, valid, n=4)
    fid = np.where(lab >= 0, lab % n_folds, -1).astype(np.int16)
    n_seg = len(seg_stats)
    seg_block = np.full(n_seg + 1, -1, dtype=np.int16)
    pix_lists = fz.segment_pixel_lists(seg_lab, n_seg)
    for sid in range(1, n_seg + 1):
        if not hiddable[sid]:
            continue
        ys, xs = pix_lists[sid - 1]
        votes = np.bincount(fid[ys, xs][fid[ys, xs] >= 0], minlength=n_folds)
        seg_block[sid] = int(votes.argmax()) if votes.sum() > 0 else -1
    rng = np.random.default_rng(seed)
    target = prevalence * float(valid.sum())
    folds = []
    for f in range(n_folds):
        held_ids = np.flatnonzero(seg_block == f)
        held = np.isin(seg_lab, held_ids)
        order = rng.permutation(held_ids)
        keep = np.zeros_like(held)
        kept_ids = []
        tot = 0.0
        for sid in order:
            if tot >= target:
                break
            keep |= seg_lab == sid
            kept_ids.append(int(sid))
            tot += float(seg_stats[sid]["size"])
        truth = keep & valid
        visible = (seg_lab > 0) & ~held & valid
        folds.append(dict(fold=f, truth=truth, visible=visible,
                          n_held=int(held.sum()), n_truth=int(truth.sum()),
                          n_segments=len(kept_ids), kept_ids=kept_ids))
    return folds


def measure_population(name, truth_mask, visible, seg_lab_v, seg_stats_v, valid):
    """Distance/azimuth/u/dstrike/L* distributions of truth pixels."""
    d, (iy, ix) = ndimage.distance_transform_edt(~visible, return_indices=True)
    t = truth_mask & valid
    ys, xs = np.nonzero(t)
    if ys.size == 0:
        return None
    dv = d[ys, xs]
    seg_near = seg_lab_v[iy, ix][ys, xs]
    seg_near[~visible[iy, ix][ys, xs]] = 0
    dy = ys - iy[ys, xs]
    dx = xs - ix[ys, xs]
    az = np.degrees(np.arctan2(dx, dy)) % 180.0
    phi = np.full(ys.size, np.nan)
    u = np.full(ys.size, np.nan)
    L = np.zeros(ys.size)
    for j, sid in enumerate(seg_near):
        if sid == 0:
            continue
        s = seg_stats_v[sid]
        rel = (az[j] - s["strike"]) % 180.0
        phi[j] = min(rel, 180.0 - rel)
        v = s["axis"]
        proj = v[0] * (ys[j] - s["centroid"][0]) + v[1] * (xs[j] - s["centroid"][1])
        u[j] = proj / max(s["span_px"], 1e-9) + 0.5
        L[j] = s["length_px"]
    # segment-level dstrike for truth segments
    tlab, tstats = fz.link_segments(truth_mask & valid)
    dstrike = []
    for sid, s in tstats.items():
        best, bd = 0, np.inf
        for vid, v in seg_stats_v.items():
            dd = float(np.linalg.norm(v["centroid"] - s["centroid"]))
            if dd < bd:
                bd, best = dd, vid
        if best:
            dstrike.append(fz._strike_diff(s["strike"], seg_stats_v[best]["strike"]))
    inside = (u >= 0) & (u <= 1)
    return dict(
        name=name, n_px=int(ys.size),
        d_px=dict(median=float(np.median(dv)), p25=float(np.percentile(dv, 25)),
                  p75=float(np.percentile(dv, 75)), p90=float(np.percentile(dv, 90)),
                  frac_le1=float((dv <= 1.5).mean()), frac_le2=float((dv <= 2.5).mean()),
                  frac_le3=float((dv <= 3.5).mean()), frac_le5=float((dv <= 5.5).mean()),
                  frac_le10=float((dv <= 10.5).mean()), frac_le20=float((dv <= 20.5).mean()),
                  frac_le50=float((dv <= 50.5).mean()), frac_gt100=float((dv > 100.5).mean())),
        d_hist_1px=[int(c) for c in np.histogram(dv, bins=np.arange(0, 61, 1))[0]],
        d_hist_20px=[int(c) for c in np.histogram(dv[dv > 60], bins=np.arange(60, 1100, 20))[0]],
        phi_deg=dict(median=float(np.nanmedian(phi)),
                     frac_le15=float(np.nanmean(phi <= 15)),
                     frac_15_45=float(np.nanmean((phi > 15) & (phi <= 45))),
                     frac_45_75=float(np.nanmean((phi > 45) & (phi <= 75))),
                     frac_gt75=float(np.nanmean(phi > 75))),
        u=dict(frac_beyond_tip=float(np.nanmean(~inside)),
               frac_inside=float(np.nanmean(inside)),
               median_inside=float(np.nanmedian(u[inside])) if inside.any() else None),
        L_near_px=dict(median=float(np.median(L[L > 0])) if (L > 0).any() else None,
                       frac_no_visible_within=float((seg_near == 0).mean())),
        dstrike_deg=dict(median=float(np.median(dstrike)) if dstrike else None,
                         n_segments=len(dstrike),
                         frac_le10=float(np.mean(np.array(dstrike) <= 10)) if dstrike else None,
                         frac_le30=float(np.mean(np.array(dstrike) <= 30)) if dstrike else None,
                         frac_gt60=float(np.mean(np.array(dstrike) > 60)) if dstrike else None),
    )


def trace_id_map_and_sense(csv_path, shape, transform):
    import pandas as pd
    tr = pd.read_csv(csv_path)
    inv = ~transform
    H, W = shape
    tmap = np.zeros((H, W), dtype=np.int32)
    r0, c0 = (inv * (tr.x0.values, tr.y0.values))
    r1, c1 = (inv * (tr.x1.values, tr.y1.values))
    for i, (a, b, cc, dd) in enumerate(zip(np.asarray(r0).astype(int), np.asarray(c0).astype(int),
                                          np.asarray(r1).astype(int), np.asarray(c1).astype(int)),
                                       start=1):
        npts = max(abs(b - a), abs(dd - cc)) + 1
        rr = np.linspace(a, b, npts).astype(int)
        ccc = np.linspace(cc, dd, npts).astype(int)
        ok = (rr >= 0) & (rr < H) & (ccc >= 0) & (ccc < W)
        tmap[rr[ok], ccc[ok]] = i
    sense = np.array(["unk"] + [str(s) for s in tr.sense], dtype=object)
    mask = tmap > 0
    return mask, tmap, sense, tr


def main():
    OUT.mkdir(exist_ok=True)
    with rasterio.open(DATA / "official/labels.tif") as ds:
        cat = ds.read(1) > 0
        transform = ds.transform
    with rasterio.open(DATA / "official/sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))

    print("[exp1] rasterizing INGENIOUS traces (known faults, with slip sense) ...")
    ingen_mask, trace_id_map, trace_sense, ingen_df = trace_id_map_and_sense(
        DATA / "external/trace_segments_utm11.csv", cat.shape, transform)
    known = (cat | ingen_mask) & valid
    print(f"  catalogue {int(cat.sum())} px, INGENIOUS {int(ingen_mask.sum())} px, "
          f"known union {int(known.sum())} px, overlap {int((cat & ingen_mask).sum())} px")

    print("[exp1] linking known-fault fragments into segments ...")
    seg_lab, seg_stats = fz.link_segments(known)
    pix_lists = fz.segment_pixel_lists(seg_lab, len(seg_stats))
    hiddable = np.zeros(len(seg_stats) + 1, dtype=bool)
    for sid in range(1, len(seg_stats) + 1):
        ys, xs = pix_lists[sid - 1]
        frac_cat = float(cat[ys, xs].mean())
        hiddable[sid] = frac_cat >= 0.9
    sizes = np.array([s["size"] for s in seg_stats.values()])
    lens = np.array([s["length_px"] for s in seg_stats.values()])
    print(f"  segments: {len(seg_stats)} (hiddable catalogue segments: {int(hiddable[1:].sum())}), "
          f"size median {np.median(sizes):.0f} max {sizes.max()}, "
          f"length median {np.median(lens):.0f}px max {lens.max():.0f}px")

    seg_sense = fz.assign_sense_from_tracemap(seg_lab, seg_stats, trace_id_map, trace_sense)
    print("  segment sense classes:", Counter(seg_sense[1:]).most_common())

    print("[exp1] building whole-segment hide-and-recover folds ...")
    folds = make_segment_folds(seg_lab, seg_stats, valid, hiddable)
    for f in folds:
        print(f"  fold {f['fold']}: truth {f['n_truth']} px in {f['n_segments']} segments, "
              f"visible {int(f['visible'].sum())} px")

    results = {"known_px": int(known.sum()), "catalogue_px": int(cat.sum()),
               "ingenious_px": int(ingen_mask.sum()),
               "segments": {"n": len(seg_stats), "hiddable": int(hiddable[1:].sum()),
                            "size_median": float(np.median(sizes)),
                            "length_median_px": float(np.median(lens)),
                            "length_max_px": float(lens.max())},
               "sense_classes": Counter(seg_sense[1:]).most_common(),
               "folds": []}

    pooled = {"d": [], "phi": [], "u": [], "L": [], "sense": [], "fold": []}
    for f in folds:
        seg_lab_v, seg_stats_v = fz.link_segments(f["visible"])
        sense_v = fz.assign_sense_from_tracemap(seg_lab_v, seg_stats_v, trace_id_map, trace_sense)
        m = measure_population(f"holdout_fold{f['fold']}", f["truth"], f["visible"],
                               seg_lab_v, seg_stats_v, valid)
        results["folds"].append(m)
        # pool raw values
        d, (iy, ix) = ndimage.distance_transform_edt(~f["visible"], return_indices=True)
        t = f["truth"] & valid
        ys, xs = np.nonzero(t)
        seg_near = seg_lab_v[iy, ix][ys, xs]
        seg_near[~f["visible"][iy, ix][ys, xs]] = 0
        dy = ys - iy[ys, xs]; dx = xs - ix[ys, xs]
        az = np.degrees(np.arctan2(dx, dy)) % 180.0
        phi = np.full(ys.size, np.nan); u = np.full(ys.size, np.nan)
        L = np.zeros(ys.size); se = np.array(["unk"] * ys.size, dtype=object)
        for j, sid in enumerate(seg_near):
            if sid == 0: continue
            s = seg_stats_v[sid]
            rel = (az[j] - s["strike"]) % 180.0
            phi[j] = min(rel, 180.0 - rel)
            v = s["axis"]
            proj = v[0] * (ys[j] - s["centroid"][0]) + v[1] * (xs[j] - s["centroid"][1])
            u[j] = proj / max(s["span_px"], 1e-9) + 0.5
            L[j] = s["length_px"]
            se[j] = str(sense_v[sid])
        pooled["d"].append(d[ys, xs]); pooled["phi"].append(phi); pooled["u"].append(u)
        pooled["L"].append(L); pooled["sense"].append(se)
        pooled["fold"].append(np.full(ys.size, f["fold"]))

    d_all = np.concatenate(pooled["d"]); phi_all = np.concatenate(pooled["phi"])
    u_all = np.concatenate(pooled["u"]); L_all = np.concatenate(pooled["L"])
    sense_all = np.concatenate(pooled["sense"])
    inside = (u_all >= 0) & (u_all <= 1)
    results["holdout_pooled"] = dict(
        n_px=int(d_all.size),
        d_px=dict(median=float(np.median(d_all)),
                  frac_le1=float((d_all <= 1.5).mean()), frac_le2=float((d_all <= 2.5).mean()),
                  frac_le3=float((d_all <= 3.5).mean()), frac_le5=float((d_all <= 5.5).mean()),
                  frac_le10=float((d_all <= 10.5).mean()), frac_le20=float((d_all <= 20.5).mean()),
                  frac_le50=float((d_all <= 50.5).mean()), frac_gt100=float((d_all > 100.5).mean())),
        d_hist_1px=[int(c) for c in np.histogram(d_all, bins=np.arange(0, 61, 1))[0]],
        phi_deg=dict(median=float(np.nanmedian(phi_all)),
                     frac_le15=float(np.nanmean(phi_all <= 15)),
                     frac_15_45=float(np.nanmean((phi_all > 15) & (phi_all <= 45))),
                     frac_45_75=float(np.nanmean((phi_all > 45) & (phi_all <= 75))),
                     frac_gt75=float(np.nanmean(phi_all > 75))),
        u=dict(frac_beyond_tip=float(np.nanmean(~inside)),
               frac_inside=float(np.nanmean(inside))),
        L_near_px=dict(median=float(np.median(L_all[L_all > 0])) if (L_all > 0).any() else None),
        sense_of_nearest_visible={k: int(v) for k, v in Counter(sense_all).most_common()},
    )
    cond = {}
    for cls in ("RL", "LL", "N", "unk"):
        sel = sense_all == cls
        if sel.sum() < 50:
            cond[cls] = {"n": int(sel.sum()), "note": "too few for a conditioned fit"}
            continue
        dv = d_all[sel]
        cond[cls] = dict(n=int(sel.sum()), median=float(np.median(dv)),
                         frac_le3=float((dv <= 3.5).mean()),
                         frac_le10=float((dv <= 10.5).mean()),
                         frac_le20=float((dv <= 20.5).mean()))
    results["holdout_pooled"]["sense_conditioned_distance"] = cond

    # length-conditioned distance: does the zone widen with L* (displacement proxy)?
    Lc = {}
    for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 100), (100, 1e9)):
        sel = (L_all >= lo) & (L_all < hi)
        if sel.sum() < 50:
            continue
        dv = d_all[sel]
        Lc[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = dict(
            n=int(sel.sum()), median=float(np.median(dv)),
            frac_le3=float((dv <= 3.5).mean()), frac_le10=float((dv <= 10.5).mean()),
            frac_le20=float((dv <= 20.5).mean()), frac_le50=float((dv <= 50.5).mean()))
    results["holdout_pooled"]["length_conditioned_distance"] = Lc

    print("[exp1] external cross-checks (SGMC off USGS+INGENIOUS) ...")
    with rasterio.open(DATA / "external/derived_sgmc_faults_100m.tif") as ds:
        sgmc = ds.read(1) > 0
    sgmc_off = sgmc & ~known & valid
    print(f"  SGMC px {int(sgmc.sum())}, off known (USGS+INGENIOUS) {int(sgmc_off.sum())}")
    results["sgmc_off_known"] = measure_population("sgmc_off_known", sgmc_off, known,
                                                   seg_lab, seg_stats, valid)
    sgmc_off_cat = sgmc & ~cat & valid
    results["sgmc_off_catonly"] = measure_population("sgmc_off_catonly", sgmc_off_cat,
                                                     cat & valid, seg_lab, seg_stats, valid)

    np.savez_compressed(OUT / "exp1_pooled.npz", d=d_all, phi=phi_all, u=u_all,
                        L=L_all, sense=sense_all.astype(str),
                        fold=np.concatenate(pooled["fold"]))
    (OUT / "exp1_zone_distributions.json").write_text(json.dumps(results, indent=1, default=str))
    print("[exp1] wrote evidence/exp1_zone_distributions.json + exp1_pooled.npz")
    print("  holdout pooled d: median %.1f px | <=3px %.3f  <=10px %.3f  <=20px %.3f  >100px %.3f" % (
        np.median(d_all), (d_all <= 3.5).mean(), (d_all <= 10.5).mean(),
        (d_all <= 20.5).mean(), (d_all > 100.5).mean()))
    print("  holdout pooled phi: median %.0f deg | <=15 %.3f  15-45 %.3f  45-75 %.3f  >75 %.3f" % (
        np.nanmedian(phi_all), np.nanmean(phi_all <= 15),
        np.nanmean((phi_all > 15) & (phi_all <= 45)),
        np.nanmean((phi_all > 45) & (phi_all <= 75)), np.nanmean(phi_all > 75)))
    print("  holdout pooled u: beyond-tip %.3f  inside %.3f" % (np.nanmean(~inside), np.nanmean(inside)))
    print("  sense of nearest visible:", dict(Counter(sense_all).most_common()))
    print("  length-conditioned d:", json.dumps(Lc, indent=1))


if __name__ == "__main__":
    main()
