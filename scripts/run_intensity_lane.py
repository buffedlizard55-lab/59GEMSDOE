#!/usr/bin/env python3
"""Session-59 intensity lane experiments (H59-I).

Three pre-declared experiments (budget rule 6):

* **Exp 1** -- fit validity and arm selection: presence-only MaxEnt/Poisson
  point-process intensity with mapping-effort covariates, Phillips-style
  effort correction exponent gamma in {0, 0.5, 1}, a geophysics-only
  comparator and a uniform-random control; leakage canary per covariate;
  predicted-vs-withheld **decile calibration** of the 1 km zoning.
* **Exp 2** -- utility table: per density class, candidate lattice spacings
  {3, 4, 6} px plus "none" (break-even), re-derived on the same holdout.
* **Exp 3** -- catalogue-collar sweep {0, 2} px on the selected table, the
  final arm, and the comparison against the published holdout best.

Holdout: shared buffered whole-component LOQO draw, seed 20 (identical to the
published Session-5/7 arms, evaluator ``gems57-pooled-hide-v2``), pooled DTI
(alpha 0.2, beta 0.8, 300 m triangular kernel), 1,000 paired draws over the
physical 20 km spatial clusters.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import evaluate_holdout as EH                      # noqa: E402
from gems59 import intensity as INT                            # noqa: E402
from gems59.grid import load_grid                              # noqa: E402
from gems59.holdout import FOLD_NAMES, buffered_component_draw  # noqa: E402
from gems59.metric import dti_exact                            # noqa: E402

GEO_ONLY = tuple(n for n, _, _ in INT.EDGE_FEATURES) + tuple(n for n, _, _ in INT.REGIME_FEATURES) \
    + ("g_edge_max",)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def auc_discriminative(scores: np.ndarray, pos: np.ndarray) -> float | None:
    """max(AUC, 1-AUC) of one covariate separating positives from the domain."""
    if pos.sum() == 0 or (~pos).sum() == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(scores)
    n1 = float(pos.sum())
    n0 = float((~pos).sum())
    auc = (r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(max(auc, 1.0 - auc))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--break-even", type=float, default=1.0,
                    help="minimum expected new-fault pixels per 1 km block to tile")
    ap.add_argument("--experiments", type=int, default=3)
    a = ap.parse_args()
    start = time.monotonic()

    grid = load_grid()
    fp = grid.footprint
    cat = grid.catalogue
    prior = float(cat.sum() / fp.sum())

    # ---- shared holdout draw (same seed as the published Session-5 arms) ----
    folds, quad = buffered_component_draw(grid, seed=a.seed)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    masked_known = folds[0]["masked_known"]
    domain_all = np.logical_or.reduce([f["region"] for f in folds]) & ~masked_known
    n_withheld = int(sum(f["truth"].sum() for f in folds))
    print(f"[holdout] seed={a.seed} withheld positives={n_withheld} "
          f"visible={int(visible.sum())} domain={int(domain_all.sum())}", flush=True)

    # ---- covariate planes (fold-invariant; built from the visible mask) ----
    planes = INT.covariate_planes(grid, visible, ROOT / "out/features")
    print(f"[planes] ready ({time.monotonic()-start:.0f}s): {list(planes)}", flush=True)

    with __import__("rasterio").open(ROOT / "data/external/sgmc_faults_100m.tif") as s:
        sgmc_only = ((s.read(1) > 0) & fp) & ~cat

    # =====================================================================
    # EXP 1 -- fit, effort correction, arms, canary, decile calibration
    # =====================================================================
    arm_specs = {
        "intensity_g0": dict(names=INT.FEATURE_NAMES, gamma=0.0),
        "intensity_g05": dict(names=INT.FEATURE_NAMES, gamma=0.5),
        "intensity_g1": dict(names=INT.FEATURE_NAMES, gamma=1.0),
        "intensity_geo_g1": dict(names=GEO_ONLY, gamma=1.0),
    }
    terms = {n: None for n in arm_specs}
    terms["uniform_random"] = None
    per_fold, canary_rows, calib_rows, sgmc_diag_rows = [], [], [], []
    lambdas = {}

    for q, fold in enumerate(folds):
        allowed = fold["region"] & ~masked_known
        train_mask = visible & (quad != q) & fp
        rows, cols = np.nonzero(train_mask)
        with threadpool_limits(limits=2):
            for name, spec in arm_specs.items():
                fit, _ = INT.fit_intensity(planes, (rows, cols), fp, names=spec["names"],
                                           prior=prior, seed=59 + q)
                lam = INT.predict_logit(planes, fit.names, fit.beta, fit.intercept, fp)
                lam = INT.effort_correct(lam, planes, spec["gamma"], fp)
                lambdas[name] = lam
                n_b, n_b_cell = INT.zone_new_fault_density(lam, masked_known)
                classes = np.where(n_b >= a.break_even, 1, 0).astype(np.int32)
                anchors = INT.block_anchors(lam, allowed)
                dots = INT.tile_blocks(n_b_cell, lam, allowed, 3.0, classes=classes,
                                       anchors=anchors)
                res, t = EH.evaluate(dots.astype(np.float32), fold, fp)
                terms[name] = t if terms[name] is None else terms[name] + t
                per_fold.append(dict(arm=name, fold=FOLD_NAMES[q],
                                     evidence_class="HOLDOUT-DTI",
                                     evaluator_version=EH.VERSION,
                                     withheld_positive_pixels=res["n_truth"],
                                     dti=round(res["dti"], 6), dots=int(dots.sum())))
                print(f"[exp1] {FOLD_NAMES[q]} {name}: DTI {res['dti']:.5f} "
                      f"dots {int(dots.sum())} ({time.monotonic()-start:.0f}s)", flush=True)
                del lam, n_b, n_b_cell, dots
            # uniform-random control at the primary arm's dot count
            n_dots = [r["dots"] for r in per_fold if r["arm"] == "intensity_g0" and r["fold"] == FOLD_NAMES[q]][0]
            rng = np.random.default_rng(1000 + q)
            cells = np.flatnonzero(allowed.ravel())
            pick = rng.choice(cells, size=n_dots, replace=False)
            rnd = np.zeros(fp.shape, bool)
            rnd.ravel()[pick] = True
            res, t = EH.evaluate(rnd.astype(np.float32), fold, fp)
            terms["uniform_random"] = t if terms["uniform_random"] is None else terms["uniform_random"] + t
            per_fold.append(dict(arm="uniform_random", fold=FOLD_NAMES[q],
                                 evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
                                 withheld_positive_pixels=res["n_truth"],
                                 dti=round(res["dti"], 6), dots=int(rnd.sum())))
            print(f"[exp1] {FOLD_NAMES[q]} uniform_random: DTI {res['dti']:.5f} "
                  f"dots {n_dots}", flush=True)
            del rnd

        # ---- leakage canary + decile calibration on this fold ----
        truth = fold["truth"]
        pos_idx = np.flatnonzero(truth.ravel())
        dom_idx = np.flatnonzero(allowed.ravel())
        for fname in INT.FEATURE_NAMES:
            v = planes[fname].ravel()
            canary_rows.append((fname, q, auc_discriminative(v[dom_idx],
                                                             np.isin(dom_idx, pos_idx))))
        lam = lambdas["intensity_g0"]
        n_b, n_b_cell = INT.zone_new_fault_density(lam, masked_known)
        # decile calibration: blocks containing withheld truth, ranked by predicted density
        pred_block = INT.block_reduce(np.where(fp, n_b_cell, 0.0)).ravel()
        truth_block = INT.block_reduce(truth.astype(np.float32)).ravel()
        sel = (pred_block > 0) | (truth_block > 0)
        p, o = pred_block[sel], truth_block[sel]
        edges = np.quantile(p, np.linspace(0, 1, 11)); edges[0] -= 1e-9
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        for d in range(10):
            m = idx == d
            if m.sum():
                calib_rows.append(dict(fold=FOLD_NAMES[q], decile=d + 1, n_blocks=int(m.sum()),
                                       mean_pred=float(p[m].mean()), sum_withheld=float(o[m].sum()),
                                       mean_withheld=float(o[m].mean())))
        sgmc_diag_rows.append(dict(fold=FOLD_NAMES[q],
                                   **INT.sgmc_diagnostic(n_b_cell, sgmc_only, fp)))
        del lam, n_b, n_b_cell
        gc.collect()

    canary = {}
    for fname in INT.FEATURE_NAMES:
        vals = [v for n, q, v in canary_rows if n == fname and v is not None]
        canary[fname] = dict(evidence_class="LEAKAGE-CANARY (AUC, not DTI)",
                             auc_max=float(max(vals)) if vals else None,
                             leakage_flag=bool(vals and max(vals) > 0.90))
    pooled = EH.pooled_summary(terms, draws=1000, seed=20261010, candidate="intensity_g0")
    calib = dict(evidence_class="HOLDOUT-STRUCTURE (calibration, not a score)",
                 rows=calib_rows,
                 note="predicted 1 km new-fault density vs withheld-segment pixel counts by decile")
    save(ROOT / "evidence/intensity_exp1_holdout.json", dict(
        evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        per_fold=per_fold, pooled=pooled, calibration=calib, canary=canary,
        sgmc_offcatalogue_diagnostic=sgmc_diag_rows,
        prior_mapped_fraction=prior))
    print("[exp1] pooled:", json.dumps({k: round(v["dti"], 5) for k, v in pooled["scores"].items()}),
          flush=True)

    # =====================================================================
    # EXP 2 -- utility table (spacing per density class + break-even)
    # =====================================================================
    candidates = {"s3": 3.0, "s4": 4.0, "s6": 6.0, "none": None}
    cls_names = ["q1_low", "q2", "q3", "q4_high"]
    base_terms = None
    # marginal table: this class at the candidate spacing, all other classes at 3 px
    terms3 = {f"{cn}__{k}": None for cn in cls_names for k in candidates if k != "s3"}
    for q, fold in enumerate(folds):
        allowed = fold["region"] & ~masked_known
        train_mask = visible & (quad != q) & fp
        rows, cols = np.nonzero(train_mask)
        with threadpool_limits(limits=2):
            fit, _ = INT.fit_intensity(planes, (rows, cols), fp, prior=prior, seed=59 + q)
            lam = INT.predict_logit(planes, fit.names, fit.beta, fit.intercept, fp)
            n_b, n_b_cell = INT.zone_new_fault_density(lam, masked_known)
            sel = n_b >= a.break_even
            vals = n_b[sel]
            edges = np.quantile(vals, [0.25, 0.5, 0.75])
            cls_block = np.digitize(n_b, edges) + 1
            cls_block[~sel] = 0
            anchors = INT.block_anchors(lam, allowed)
            rest3 = INT.tile_blocks(n_b_cell, lam, allowed, 3.0, classes=cls_block.astype(np.int32),
                                    anchors=anchors)
            res, t = EH.evaluate(rest3.astype(np.float32), fold, fp)
            base_terms = t if base_terms is None else base_terms + t
            for ci, cn in enumerate(cls_names, start=1):
                for key, sp in candidates.items():
                    if key == "s3":
                        continue
                    if sp is not None:
                        dots = INT.tile_blocks(n_b_cell, lam, allowed, sp,
                                               classes=np.where(cls_block == ci, ci, 0).astype(np.int32),
                                               anchors=anchors) | \
                            INT.tile_blocks(n_b_cell, lam, allowed, 3.0,
                                            classes=np.where(cls_block != ci, cls_block, 0).astype(np.int32),
                                            anchors=anchors)
                    else:
                        dots = INT.tile_blocks(n_b_cell, lam, allowed, 3.0,
                                               classes=np.where(cls_block != ci, cls_block, 0).astype(np.int32),
                                               anchors=anchors)
                    res, t = EH.evaluate(dots.astype(np.float32), fold, fp)
                    terms3[f"{cn}__{key}"] = t if terms3[f"{cn}__{key}"] is None \
                        else terms3[f"{cn}__{key}"] + t
        print(f"[exp2] fold {FOLD_NAMES[q]} done ({time.monotonic()-start:.0f}s)", flush=True)
        del lam, n_b, n_b_cell, rest3
        gc.collect()

    table_pooled = EH.pooled_summary({**terms3, "base_s3": base_terms}, draws=1000,
                                     seed=20261010, candidate="base_s3")
    utility = {}
    for cn in cls_names:
        opts = {k: table_pooled["scores"][f"{cn}__{k}"]["dti"] for k in candidates if k != "s3"}
        opts["s3"] = table_pooled["scores"]["base_s3"]["dti"]  # reference value for every class
        best = max(opts, key=opts.get)
        utility[cn] = dict(candidates_dti={k: round(v, 6) for k, v in opts.items()},
                           selected=best,
                           selected_dti=round(opts[best], 6),
                           evidence_class="HOLDOUT-DTI",
                           note="class-inclusive field: this class at the candidate spacing, "
                                "all other classes at 3 px; s3 column is the all-3px baseline")
    save(ROOT / "evidence/intensity_exp2_utility_table.json", dict(
        evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        break_even_min_density=a.break_even, density_classes=cls_names,
        utility_table=utility, baseline_dti=round(table_pooled["scores"]["base_s3"]["dti"], 6),
        selection_caveat="table selected on the same holdout it is measured on; CIs are "
                         "conditional on the selected table and fitted folds"))
    print("[exp2] utility table:", json.dumps({k: v["selected"] for k, v in utility.items()}), flush=True)

    # =====================================================================
    # EXP 3 -- collar sweep on the selected table, final arm
    # =====================================================================
    selected = {cn: (None if utility[cn]["selected"] == "none"
                     else candidates[utility[cn]["selected"]]) for cn in cls_names}
    terms4 = {"final_c0": None, "final_c2": None}
    for q, fold in enumerate(folds):
        allowed = fold["region"] & ~masked_known
        train_mask = visible & (quad != q) & fp
        rows, cols = np.nonzero(train_mask)
        with threadpool_limits(limits=2):
            fit, _ = INT.fit_intensity(planes, (rows, cols), fp, prior=prior, seed=59 + q)
            lam = INT.predict_logit(planes, fit.names, fit.beta, fit.intercept, fp)
            n_b, n_b_cell = INT.zone_new_fault_density(lam, masked_known)
            sel = n_b >= a.break_even
            vals = n_b[sel]
            edges = np.quantile(vals, [0.25, 0.5, 0.75])
            cls_block = np.digitize(n_b, edges) + 1
            cls_block[~sel] = 0
            for cname, collar in (("final_c0", 0), ("final_c2", 2)):
                elig = (allowed & ~ndi.binary_dilation(masked_known, iterations=collar)) \
                    if collar else allowed
                anchors = INT.block_anchors(lam, elig)
                dots = np.zeros(fp.shape, bool)
                for ci, cn in enumerate(cls_names, start=1):
                    sp = selected[cn]
                    if sp is None:
                        continue
                    classes = np.where(cls_block == ci, ci, 0).astype(np.int32)
                    dots |= INT.tile_blocks(n_b_cell, lam, elig, sp, classes=classes,
                                            anchors=anchors)
                res, t = EH.evaluate(dots.astype(np.float32), fold, fp)
                terms4[cname] = t if terms4[cname] is None else terms4[cname] + t
                print(f"[exp3] {FOLD_NAMES[q]} {cname}: DTI {res['dti']:.5f} dots {int(dots.sum())}",
                      flush=True)
        del lam, n_b, n_b_cell
        gc.collect()

    final_pooled = EH.pooled_summary({**terms4, "base_s3": base_terms}, draws=1000,
                                     seed=20261010, candidate="final_c0")
    save(ROOT / "evidence/intensity_exp3_final.json", dict(
        evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        utility_table=utility, selected_spacings={k: v for k, v in selected.items()},
        pooled=final_pooled,
        published_reference=dict(
            source="57GEMSDOE evidence/relay_bend_holdout.json (Session 5, same evaluator "
                   "and same seed-20 draw)",
            relay_bend_anatomy_dti=0.135204, relay_bend_anatomy_ci=[0.119140, 0.153036],
            distance_only_dti=0.112725, distance_only_ci=[0.101428, 0.124446])))
    print("[exp3] pooled:", json.dumps({k: round(v["dti"], 5) for k, v in final_pooled["scores"].items()}),
          flush=True)
    print(f"[done] experiments used: 3; runtime {(time.monotonic()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
