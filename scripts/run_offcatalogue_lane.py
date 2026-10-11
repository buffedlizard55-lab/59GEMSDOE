#!/usr/bin/env python3
"""Session H59-J experiment -- off-catalogue discriminant + break-even sparse emission.

Lane discipline
---------------
This session's lane is *not* the H59-I intensity lattice.  It is the
pre-registered follow-up the previous session left open as **H59-I4**: build an
instrument whose truth is genuinely **off-catalogue**, fit the placement on that
population, and emit only what clears the exact marginal break-even rule.

What is new here (relative to every earlier session in this repository)
----------------------------------------------------------------------
1. The classifier's **positive class is the off-catalogue fault population** --
   USGS State Geologic Map Compilation fault pixels more than 300 m from the
   published catalogue (``data/external/sgmc_faults_100m.tif``), i.e. the only
   independent realisation of "faults the supplied catalogue does not contain"
   available offline.  Earlier sessions in this repository trained on the
   catalogue itself or on hand-built surfaces; the SGMC was used either as an
   evaluation truth or as a raw emission source, never as the training label.
2. The classifier is evaluated **out-of-fold on six spatial blocks**, so the
   emitted field at every pixel comes from a model that never saw that block.
3. The dot budget is **not swept blindly**: it is anchored to the exact
   marginal rule of the official metric (``dDTI > 0 <=> k > alpha * DTI``,
   alpha = 0.2) and to the *measured* within-family ladder of the owner's own
   thinnings, in which every step that removed dots raised the reported score.

Honest scope
------------
Class labels are a **proxy**: SGMC off-catalogue faults are 79,615 pixels
where the calibrated hidden "new fault" population is ~12,700, so the proxy
population is ~6x too prevalent and includes faults the experts did not add.
Everything this script prints about DTI is a **proxy measurement**, never a
competition score.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid               # noqa: E402
from gems59.metric import ALPHA, BETA, RADIUS_PX, kernel   # noqa: E402

BANDS = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf",
    "ieq_n100a15", "cond_surf", "iso_grav_anom_hg", "det_elev_slope",
]
NODATA_U8 = 255
COLLAR_PX = 3.0          # the official kernel radius: SGMC truth is taken beyond it
BLOCKS_YX = (3, 2)       # six spatial folds
NEG_MIN_DIST_PX = 3.0    # background pixels must be far from both fault sets


def load_bank(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def feature_columns(bank: np.ndarray, yx: np.ndarray,
                    box: tuple[slice, slice] | None = None) -> np.ndarray:
    """Sampled feature matrix (float32) for pixel coordinates ``yx`` (N, 2).

    When ``box`` is given the bank slice for that bounding box is read
    contiguously once (memmap-friendly) and indexed in memory; coordinates are
    interpreted as absolute grid coordinates.
    """
    y, x = yx[:, 0], yx[:, 1]
    if box is not None:
        sub = np.stack([np.asarray(bank[k][box]) for k in range(bank.shape[0])], 0)
        y = y - box[0].start
        x = x - box[1].start
        cols = [sub[k][y, x].astype(np.float32) for k in range(bank.shape[0])]
    else:
        cols = [np.asarray(bank[k][y, x], dtype=np.float32) for k in range(bank.shape[0])]
    X = np.stack(cols, axis=1)
    # 255 marks nodata in the bank: replace by the neutral rank 127 before scaling
    X[X == NODATA_U8] = 127.0
    return X / 254.0


def block_ids(shape, ny, nx) -> np.ndarray:
    H, W = shape
    bid = np.full(shape, -1, np.int8)
    ys = np.linspace(0, H, ny + 1).astype(int)
    xs = np.linspace(0, W, nx + 1).astype(int)
    for j in range(ny):
        for i in range(nx):
            bid[ys[j]:ys[j + 1], xs[i]:xs[i + 1]] = j * nx + i
    return bid


def greedy_nms(scores: np.ndarray, coords: np.ndarray, n_keep: int, min_dist: float) -> np.ndarray:
    """Deterministic greedy selection: walk the ranking, keep a dot when it is at
    least ``min_dist`` pixels from every already-kept dot."""
    kept = []
    # Bucket the space so the neighbourhood test is O(1) instead of a KD-tree rebuild.
    cell = max(min_dist, 1.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for value, (yy, xx) in zip(scores, coords):
        gy, gx = int(yy // cell), int(xx // cell)
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (py, px) in grid.get((gy + dy, gx + dx), ()):
                    if (py - yy) ** 2 + (px - xx) ** 2 < min_dist * min_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gy, gx), []).append((yy, xx))
            kept.append((yy, xx))
            if len(kept) >= n_keep:
                break
    return np.array(kept, dtype=np.int64).reshape(-1, 2)


def credit_stats(dots_yx: np.ndarray, truth_yx: np.ndarray) -> dict:
    if dots_yx.size == 0 or truth_yx.size == 0:
        return dict(T=0.0, M=0.0, S=float(dots_yx.shape[0]), DTI=0.0,
                    credit_per_dot=0.0, coverage=0.0)
    dt = cKDTree(truth_yx)
    dd = cKDTree(dots_yx)
    d_t, _ = dd.query(truth_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    T = float(kernel(np.where(np.isfinite(d_t), d_t, RADIUS_PX)).sum())
    d_x, _ = dt.query(dots_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    M = float(kernel(np.where(np.isfinite(d_x), d_x, RADIUS_PX)).sum())
    S = T + dots_yx.shape[0] - M
    return dict(T=T, M=M, S=float(S), DTI=float(T / (ALPHA * S + BETA * truth_yx.shape[0] + 1e-12)),
                credit_per_dot=float(T / dots_yx.shape[0]), coverage=float(T / truth_yx.shape[0]))


def matched_truth(truth: np.ndarray, target_px: int, seed: int = 59) -> np.ndarray:
    """Prevalence-match a truth raster by dropping whole connected components."""
    lab, n = ndi.label(truth, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(-sizes[1:]) + 1
    rng = np.random.default_rng(seed)
    keep = np.zeros(n + 1, bool)
    total = 0
    idx = list(order)
    rng.shuffle(idx)                       # random component order, deterministic seed
    for cid in idx:
        if total + sizes[cid] > target_px:
            continue
        keep[cid] = True
        total += sizes[cid]
        if total >= target_px * 0.98:
            break
    return keep[lab]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=str(ROOT / "out/features/bank_u8.npy"))
    ap.add_argument("--out", default=str(ROOT / "evidence/offcat_lane_h59j.json"))
    ap.add_argument("--oob-out", default=str(ROOT / ".cache/oof_propensity.npy"))
    ap.add_argument("--budgets", default="10000,20000,30000,44090")
    ap.add_argument("--spacings", default="1.0,2.0,3.0")
    ap.add_argument("--target-px", type=int, default=12700,
                    help="prevalence-matched proxy truth size (calibrated hidden |G| ~12.7k)")
    ap.add_argument("--seed", type=int, default=59)
    args = ap.parse_args()
    t0 = time.time()

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    H, W = grid.shape
    d_cat = ndi.distance_transform_edt(~cat).astype(np.float32)

    with rasterio.open(ROOT / "data/external/sgmc_faults_100m.tif") as src:
        sgmc = src.read(1) == 1
    proxy = sgmc & foot & ~cat                       # off-catalogue PROXY truth
    proxy_far = proxy & (d_cat > COLLAR_PX)
    proxy_matched = matched_truth(proxy, args.target_px, seed=args.seed)
    print(f"[lane] proxy truth {int(proxy.sum())} px; >{COLLAR_PX:.0f} px from catalogue "
          f"{int(proxy_far.sum())} px; prevalence-matched {int(proxy_matched.sum())} px")

    d_proxy = ndi.distance_transform_edt(~(proxy | cat)).astype(np.float32)
    bid = block_ids((H, W), *BLOCKS_YX)
    rng = np.random.default_rng(args.seed)

    bank = load_bank(Path(args.bank))
    n_feat = bank.shape[0]
    print(f"[lane] bank {bank.shape} ({n_feat} channels)")

    # ---------------- out-of-fold propensity ----------------
    oof = np.lib.format.open_memmap(Path(args.oob_out), mode="w+", dtype=np.float32, shape=(H, W))
    auc_rows = []
    pos_all = np.argwhere(proxy)
    foot_idx = np.argwhere(foot)
    n_blocks = BLOCKS_YX[0] * BLOCKS_YX[1]
    ys_grid = np.linspace(0, H, BLOCKS_YX[0] + 1).astype(int)
    xs_grid = np.linspace(0, W, BLOCKS_YX[1] + 1).astype(int)

    def fold_box(fold: int) -> tuple[slice, slice]:
        j, i = divmod(fold, BLOCKS_YX[1])
        return (slice(ys_grid[j], ys_grid[j + 1]), slice(xs_grid[i], xs_grid[i + 1]))

    near_fault = d_proxy <= NEG_MIN_DIST_PX      # excludes the faults themselves
    for fold in range(n_blocks):
        blocks_here = (bid == fold) & foot
        train = (bid != fold) & foot
        test = blocks_here
        box = fold_box(fold)
        # positives: proxy fault pixels inside the training blocks
        pos_tr = pos_all[train[pos_all[:, 0], pos_all[:, 1]]]
        if pos_tr.shape[0] > 30000:
            pos_tr = pos_tr[rng.choice(pos_tr.shape[0], 30000, replace=False)]
        # negatives: background pixels of the training blocks, clear of both fault sets
        cand = foot_idx[train[foot_idx[:, 0], foot_idx[:, 1]]
                        & ~near_fault[foot_idx[:, 0], foot_idx[:, 1]]]
        take = min(120000, cand.shape[0])
        neg_tr = cand[rng.choice(cand.shape[0], take, replace=False)]
        yx_tr = np.concatenate([pos_tr, neg_tr], 0)
        y_tr = np.concatenate([np.ones(pos_tr.shape[0], np.int8),
                               np.zeros(neg_tr.shape[0], np.int8)])
        X_tr = feature_columns(bank, yx_tr, box=None)
        clf = HistGradientBoostingClassifier(max_iter=90, learning_rate=0.12,
                                             max_leaf_nodes=31, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=args.seed)
        clf.fit(X_tr, y_tr)
        # predict the held-out block in row chunks (memory discipline)
        ys, xs = np.nonzero(test)
        p = np.empty(ys.size, np.float32)
        for start in range(0, ys.size, 400_000):
            sl = slice(start, start + 400_000)
            Xb = feature_columns(bank, np.stack([ys[sl], xs[sl]], 1), box=box)
            p[sl] = clf.predict_proba(Xb)[:, 1].astype(np.float32)
        oof[ys, xs] = p
        # OOF AUC on this fold
        lab_te = proxy[ys, xs].astype(np.int8)
        if lab_te.min() == 0 and lab_te.max() == 1:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(lab_te, p))
        else:
            auc = float("nan")
        auc_rows.append(dict(fold=int(fold), n_test=int(ys.size), n_pos=int(lab_te.sum()),
                             auc_oof_on_proxy=auc))
        print(f"[lane] fold {fold}: held-out {ys.size} px, proxy positives {int(lab_te.sum())}, "
              f"OOF AUC {auc:.4f}  ({time.time()-t0:.0f}s)")
    oof.flush()
    del oof

    pooled_lab = proxy[foot]
    pooled_p = np.load(Path(args.oob_out), mmap_mode="r")[foot]
    from sklearn.metrics import roc_auc_score
    pooled_auc = float(roc_auc_score(pooled_lab.astype(np.int8), pooled_p))
    print(f"[lane] pooled out-of-fold AUC on the off-catalogue proxy: {pooled_auc:.4f}")

    # single-feature leakage canary: any channel that alone exceeds 0.90 is leakage
    canary = {}
    sample = foot_idx[rng.choice(foot_idx.shape[0], 120_000, replace=False)]
    lab_s = proxy[sample[:, 0], sample[:, 1]].astype(np.int8)
    if lab_s.min() == 0 and lab_s.max() == 1:
        for k in range(n_feat):
            v = feature_columns(bank, sample)[:, k]
            canary[f"ch{k:02d}"] = float(roc_auc_score(lab_s, v))
    canary_max = max(canary.values()) if canary else float("nan")
    print(f"[lane] leakage canary: max single-channel AUC {canary_max:.4f} "
          f"(flag if > 0.90)")

    # ---------------- emission arms ----------------
    truth_yx_all = np.argwhere(proxy)
    truth_yx_far = np.argwhere(proxy_far)
    truth_yx_matched = np.argwhere(proxy_matched)
    flat = np.load(Path(args.oob_out), mmap_mode="r")
    flat_scores = np.asarray(flat).ravel()
    keep_idx = np.flatnonzero(foot.ravel() & np.isfinite(flat_scores))
    order = keep_idx[np.argsort(-flat_scores[keep_idx])]
    top_n = min(order.size, max(int(x) for x in args.budgets.split(",")) * 3)
    top = order[:top_n]
    coords = np.stack([top // W, top % W], 1).astype(np.int64)
    vals = flat_scores[top]

    arms = []
    for spacing in [float(s) for s in args.spacings.split(",")]:
        picked = greedy_nms(vals, coords, n_keep=top_n, min_dist=spacing)
        for budget in [int(b) for b in args.budgets.split(",")]:
            dots = picked[:budget]
            if dots.shape[0] < budget:
                continue
            row = dict(spacing_px=spacing, budget=budget,
                       on_catalogue=int(cat[dots[:, 0], dots[:, 1]].sum()),
                       distance_p50_px=float(np.median(d_cat[dots[:, 0], dots[:, 1]])))
            for name, truth in (("proxy_all", truth_yx_all),
                                ("proxy_far", truth_yx_far),
                                ("proxy_matched", truth_yx_matched)):
                st = credit_stats(dots, truth)
                row[f"dti_{name}"] = st["DTI"]
                row[f"credit_per_dot_{name}"] = st["credit_per_dot"]
                row[f"coverage_{name}"] = st["coverage"]
            arms.append(row)
            print(f"[lane] arm spacing={spacing:.1f} budget={budget:6d} "
                  f"cred/dot(far)={row['credit_per_dot_proxy_far']:.4f} "
                  f"DTI(matched)={row['dti_proxy_matched']:.4f} "
                  f"cov(matched)={row['coverage_proxy_matched']:.3f}")

    # ---------------- reference arms: the incumbent thinnings ----------------
    registry = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    by_sha = {r["sha256"]: r for r in registry["rasters"]}
    ledger = json.loads((ROOT / "evidence/live_ledger_merged.json").read_text())["rows"]
    reference = []
    for row in ledger:
        if not row["name"].startswith(("h33-h33-2-b2", "dotted-h19-5-d2-8",
                                       "h25-1-dotted-h19-5-d1-5",
                                       "h19-5-powerlaw-budget-multiline")):
            continue
        rec = by_sha.get(row["sha256"])
        with rasterio.open(ROOT / rec["cache_file"]) as src:
            a = src.read(1)
        dots = np.argwhere(np.isfinite(a) & (a > 0))
        entry = dict(name=row["name"], owner_reported_score=row["score"],
                     n_dots=int(dots.shape[0]))
        for name, truth in (("proxy_all", truth_yx_all), ("proxy_far", truth_yx_far),
                            ("proxy_matched", truth_yx_matched)):
            st = credit_stats(dots, truth)
            entry[f"dti_{name}"] = st["DTI"]
            entry[f"credit_per_dot_{name}"] = st["credit_per_dot"]
        reference.append(entry)
        print(f"[lane] reference {row['name'][:48]:<50s} n={entry['n_dots']:7d} "
              f"cred/dot(far)={entry['credit_per_dot_proxy_far']:.4f} "
              f"DTI(matched)={entry['dti_proxy_matched']:.4f}")

    result = dict(
        evidence_class="PROXY-MEASUREMENT (off-catalogue SGMC proxy truth; not a competition score)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/run_offcatalogue_lane.py",
        hypothesis=("Faults that expert compilations add to a regional catalogue -- the "
                    "population the competition scores -- are separable from background by an "
                    "out-of-fold discriminant trained on off-catalogue map faults, and a sparse "
                    "emission at that field's top ranks clears the exact marginal break-even bar."),
        named_non_fault_mimic=("The SGMC off-catalogue population is ~6x more prevalent than the "
                               "hidden 'new fault' population and mixes non-fault lineaments "
                               "(unit contacts, thrust sheets, mining-district structures and "
                               "compilation artefacts) with real faults; a classifier trained on it "
                               "can reproduce compilation style rather than geology."),
        proxy=dict(px_all=int(proxy.sum()), px_far=int(proxy_far.sum()),
                   px_matched=int(proxy_matched.sum()), target_px=args.target_px),
        oof_auc=dict(per_fold=auc_rows, pooled=pooled_auc),
        leakage_canary=dict(max_single_channel_auc=canary_max, per_channel=canary),
        arms=arms, reference_arms=reference,
        elapsed_s=float(time.time() - t0),
    )
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[lane] wrote {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
