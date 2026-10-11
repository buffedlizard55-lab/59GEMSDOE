#!/usr/bin/env python3
"""Calibration: score the registry's top rasters (and controls) on THIS lane's
holdout instrument, so the lane's HOLDOUT-DTI is comparable to the live scores
those rasters carry.  Also score every raster on the SGMC-truth instrument
(real off-catalogue faults = the closest legal proxy for the hidden truth).

Both instruments use the template metric (alpha 0.2, beta 0.8, 300 m).  All
numbers are HOLDOUT-DTI / PROXY-DTI, never organizer scores.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import faultzone as fz  # noqa: E402
from gems59 import evaluate_holdout as EH  # noqa: E402
from gems59 import holdout as HO  # noqa: E402
from gems59 import metric as M  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "evidence"
REG = ROOT / "registry" / "rasters"       # in-repo registry (was /home/user/registry)
BASE = 100000
BUFFER_PX = 3
N_FOLDS = 4
PREVALENCE = 0.002
SEED = 20261009


def _registry_files() -> dict:
    """Map a short label to an in-repo registry raster path, by sha256 first and
    by name substring second (IR-57-CALIB-01: the original hard-coded paths
    pointed at a sibling checkout that no longer exists at that location)."""
    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    by_sha = {e["sha256"]: ROOT / e["file"] for e in idx}
    by_name = {e["submission"]: ROOT / e["file"] for e in idx}
    return by_sha, by_name


# registry rasters that carry owner-reported live scores (top of the family).
# Each entry: (label, registry_index submission substring, sha256 or None)
REGISTRY = [
    ("h33-2-b2 LIVE 0.2778", "h33-h33-2-b2", "c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9"),
    ("h27-4-solo LIVE 0.2708", "h27-4-r1-solo-d2-8", "2fc94a38d77f74f4f4e1a97a83e7bb71a1ceea090515ec641e6681cc47c44c8"),
    ("h36-1-rung30 LIVE 0.2710", "h36-1-rung30-blind-r1", "7c74270ad48fa6b55163853046a6d3815967c5bf37bbda5b74ca7fa22583a32"),
    ("tip-h33d LIVE 0.2632", "h33d-analog-tip-stepover-r30", "87f857d505e23247e991ccfab2cbe9f49a04df4f9c8028dce7ea261554690757"),
    ("tip-h32-1 LIVE 0.2649", "h32-1-prethin-tip-euler-d2-8", "26748e4b4721277b093c776f7ca72920023d0ee6e9514ee38e677c8010f4068e"),
    ("h19-5 LIVE 0.1922", "h19-5-powerlaw-budget-multiline", "ef2ae808eb7179861b9f1ff753fd3ded625ace7b7c158e29cec45bbd8fa3cfaa"),
    ("d15 LIVE 0.2477", "h25-1-dotted-h19-5-d1-5", "3e0730e8d4f7db095dd80246a98cba3bc0fc9a28f436b91db27ca34df4d0d1ce"),
    ("h32 LIVE 0.2649", "h32-1-prethin-tip-euler-d2-8", "26748e4b4721277b093c776f7ca72920023d0ee6e9514ee38e677c8010f4068e"),
    ("h36 LIVE 0.2710", "h36-1-rung30-blind-r1", "7c74270ad48fa6b55163853046a6d3815967c5bf37bbda5b74ca7fa22583a32"),
    ("anderson LIVE 0.2750", "anderson-geothermal-pinn-38854", "e15891020b9c57056ac7fa874a5e506fbd6ed9314a4ea8e73af0753887a3708a"),
    ("catalogue (sample_submission)", None, None),
]


def _resolve(label, substr, sha):
    if substr is None:
        return DATA / "official/sample_submission.tif"
    by_sha, by_name = _registry_files()
    if sha and sha in by_sha:
        return by_sha[sha]
    for name, p in by_name.items():
        if substr in name:
            return p
    return None


def read_raster(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
    return np.where(np.isfinite(a), a, 0.0)


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with rasterio.open(DATA / "official/labels.tif") as ds:
        cat = ds.read(1) > 0
        transform = ds.transform
    with rasterio.open(DATA / "official/sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))
    with rasterio.open(DATA / "external/derived_sgmc_faults_100m.tif") as ds:
        sgmc = ds.read(1) > 0

    ing_stats, ing_pix, tmap, _ = fz.ingenious_record_segments(
        DATA / "external/trace_segments_utm11.csv", cat.shape, transform)
    ingen_mask = tmap > 0
    seg_lab_cat, seg_stats_cat = fz.link_segments(cat & valid)
    n_cat = len(seg_stats_cat)
    seg_lab = np.zeros(cat.shape, dtype=np.int32)
    pl = fz.segment_pixel_lists(seg_lab_cat, n_cat)
    for sid in range(1, n_cat + 1):
        ys, xs = pl[sid - 1]
        seg_lab[ys, xs] = sid
    del seg_lab_cat, pl
    gc.collect()
    for sid, (ys, xs) in ing_pix.items():
        seg_lab[ys, xs] = BASE + sid
    seg_stats = dict(seg_stats_cat)
    for sid, s in ing_stats.items():
        seg_stats[BASE + sid] = s
    known = (cat | ingen_mask) & valid
    sgmc_truth = sgmc & ~known & valid
    del tmap, ingen_mask, sgmc
    gc.collect()

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

    fold_dicts = []
    for f in folds:
        visible = known & ~f["held"]
        near_visible = ndimage.binary_dilation(visible, iterations=BUFFER_PX)
        truth = f["held"] & valid & ~near_visible
        fold_dicts.append(dict(fold=f["fold"], mode="hide", truth=truth,
                               visible=visible, region=valid, fit=valid,
                               boundary=near_visible, n_truth=int(truth.sum()),
                               n_held=int(f["held"].sum())))
        del visible, near_visible, truth
        gc.collect()

    results = {}
    for name, substr, sha in REGISTRY:
        path = _resolve(name, substr, sha)
        if path is None or not Path(path).exists():
            print(f"  MISSING {name} (substr={substr})", flush=True)
            continue
        p = read_raster(path)
        terms = []
        per_fold = []
        for fd in fold_dicts:
            result, t = EH.evaluate(p, fd, valid, block_side=200)
            terms.append(t)
            per_fold.append(dict(fold=fd["fold"], dti=result["dti"],
                                 tpw=result["tpw"], fpw=result["fpw"],
                                 fnw=result["fnw"], emitted=result["emitted"]))
        merged = np.stack(terms).sum(axis=0)
        s = EH.pooled_summary({"candidate": merged}, draws=1000, seed=SEED + 7,
                              candidate="candidate")
        # SGMC-truth instrument: mask known pixel-exactly, score vs SGMC off-known
        pm = np.where(known, 0.0, p)
        r = M.dti_exact(pm.astype(np.float32), sgmc_truth, valid=valid, known=known)
        results[name] = dict(
            file=str(path),
            holdout_dti=s["scores"]["candidate"]["dti"],
            holdout_ci95=s["scores"]["candidate"]["ci95"],
            holdout_withheld_positives=s["scores"]["candidate"]["withheld_positive_pixels"],
            holdout_tpw=s["scores"]["candidate"]["tpw"],
            holdout_fpw=s["scores"]["candidate"]["fpw"],
            holdout_fnw=s["scores"]["candidate"]["fnw"],
            per_fold=per_fold,
            sgmc_truth_dti=r["dti"], sgmc_tpw=r["tpw"], sgmc_fpw=r["fpw"],
            sgmc_fnw=r["fnw"], sgmc_n_truth=r["n_truth"],
            emitted_px=int((p > 0).sum()))
        print(f"  {name:34s} holdout DTI {results[name]['holdout_dti']:.4f} "
              f"[{results[name]['holdout_ci95'][0]:.4f},{results[name]['holdout_ci95'][1]:.4f}]  "
              f"SGMC-truth DTI {r['dti']:.4f}  emitted {int((p > 0).sum())}", flush=True)
        del p, pm
        gc.collect()

    payload = dict(
        evidence_class="HOLDOUT-DTI + PROXY-DTI (SGMC-truth)",
        evaluator_version=EH.VERSION,
        evaluator_implementation_sha256=EH.implementation_hashes(),
        protocol="same folds as exp2; registry rasters scored as-is (masked pixel-exactly "
                 "on visible known faults per fold)",
        withheld_positive_pixels=sum(fd["n_truth"] for fd in fold_dicts),
        sgmc_truth_pixels=int(sgmc_truth.sum()),
        results=results,
        runtime_s=time.time() - t0,
    )
    (OUT / "calibrate_registry.json").write_text(json.dumps(payload, indent=1, default=str))
    print("[calib] wrote evidence/calibrate_registry.json", flush=True)


if __name__ == "__main__":
    main()
