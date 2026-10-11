#!/usr/bin/env python3
"""Session H59-I2 Intensity-Lane 3-Experiment Holdout Runner.

Executes the three pre-registered experiments on the shared 4-fold spatially
blocked holdout (``buffered_component_draw``, seed 20, 11,321 withheld positive
pixels, ``gems57-pooled-hide-v2``):

1. Experiment 1 (``evidence/intensity_exp1_holdout.json`` &
   ``evidence/intensity_exp1_diagnostics_fixed.json``):
   Leave-One-Component-Out (LOCO) two-host presence-only MaxEnt/Poisson IPP
   fitting (strictly on ``visible`` catalogue presences vs target-group
   background; zero ``hidden`` truth labels used in training), effort-bias
   marginalization sweep (gamma in {0.0, 0.25, 0.50, 1.00}), linear vs
   quadratic+bend+scarp ablation, 31-feature leakage canary (both holdout AUC
   and training LOCO AUC < 0.90), 1 km block decile calibration across all
   52,309 footprint blocks, and off-catalogue SGMC diagnostic.
2. Experiment 2 (``evidence/intensity_exp2_utility_table.json``):
   Holdout-derived utility spacing table across the four 1 km block density
   quartiles (``q1_low, q2, q3, q4_high``) for spacing candidates
   ``s in {3, 4, 6, none}`` and break-even block density thresholds in ``[1.0, 2.0]``
   (1-2% new-fault density per 100-pixel 1 km block).
3. Experiment 3 (``evidence/intensity_exp3_final.json``):
   Locked final configuration evaluated with 1,000 bootstrap draws against
   Session H59-I (0.082161), distance_only, uniform_random, and published
   holdout best (0.135204).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import evaluate_holdout as EH  # noqa: E402
from gems59 import intensity as INT  # noqa: E402
from gems59.grid import load_grid  # noqa: E402
from gems59.holdout import FOLD_NAMES, buffered_component_draw  # noqa: E402
from gems59.offcatalogue import load_proxy_truth  # noqa: E402


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def auc_discriminative(scores: np.ndarray, pos: np.ndarray) -> float | None:
    m = np.isfinite(scores)
    y = pos[m].astype(np.int8)
    if y.sum() == 0 or y.sum() == y.size:
        return None
    auc = float(roc_auc_score(y, scores[m]))
    return max(auc, 1.0 - auc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--features-dir", type=Path, default=ROOT / "out/features")
    ap.add_argument("--break-even", type=float, default=1.5,
                    help="1 km block break-even density floor (expected new-fault px per 100-px block)")
    ap.add_argument("--cell-quantile", type=float, default=0.05,
                    help="Within-corridor kernel intensity floor quantile inside tiled blocks")
    a = ap.parse_args()

    t0 = time.monotonic()
    grid = load_grid()
    fp = grid.footprint
    folds, quad = buffered_component_draw(grid, seed=a.seed)
    visible = folds[0]["visible"]
    masked_known = folds[0]["masked_known"]
    hidden_all = folds[0]["truth"] | folds[1]["truth"] | folds[2]["truth"] | folds[3]["truth"]
    n_withheld = int(sum(int(f["truth"].sum()) for f in folds))
    print(f"[setup] seed={a.seed} visible={int(visible.sum())} withheld={n_withheld}", flush=True)

    sgmc_only, _ = load_proxy_truth(ROOT, grid)
    print(f"[setup] SGMC off-catalogue pixels (>300m from catalogue): {int(sgmc_only.sum())}", flush=True)

    frame = INT.loco_structural_frame(grid, visible, ROOT)
    bank = INT.load_bank(a.features_dir)
    d1 = frame["d1"]
    print(f"[setup] LOCO two-host frame ready ({time.monotonic()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Leakage canary: test all 31 features on holdout AND on training LOCO
    # ------------------------------------------------------------------
    rng_c = np.random.default_rng(20261011)
    eval_dom = fp & ~masked_known
    h_rows, h_cols = np.nonzero(hidden_all & eval_dom)
    bg_eval_all = np.flatnonzero((eval_dom & ~hidden_all).ravel())
    bg_eval_pick = rng_c.choice(bg_eval_all, size=min(100_000, bg_eval_all.size), replace=False)
    b_rows, b_cols = np.unravel_index(bg_eval_pick, fp.shape)
    Xh = INT.extract_loco_features(frame, bank, h_rows, h_cols)
    Xbe = INT.extract_loco_features(frame, bank, b_rows, b_cols)
    yh = np.concatenate([np.ones(len(Xh), np.int8), np.zeros(len(Xbe), np.int8)])
    Xeval = np.vstack([Xh, Xbe])

    p_rows, p_cols = np.nonzero(visible & fp & (d1 > 3.0) & (d1 <= 35.0))
    tgb_all = np.flatnonzero(((~visible) & fp & (d1 > 3.0) & (d1 <= 35.0)).ravel())
    tgb_pick = rng_c.choice(tgb_all, size=min(100_000, tgb_all.size), replace=False)
    tr_rows, tr_cols = np.unravel_index(tgb_pick, fp.shape)
    Xp_c = INT.extract_loco_features(frame, bank, p_rows, p_cols)
    Xtgb_c = INT.extract_loco_features(frame, bank, tr_rows, tr_cols)
    ytr = np.concatenate([np.ones(len(Xp_c), np.int8), np.zeros(len(Xtgb_c), np.int8)])
    Xtr = np.vstack([Xp_c, Xtgb_c])

    canary = {}
    for j, fname in enumerate(INT.LOCO_FEATURE_NAMES):
        auc_h = auc_discriminative(Xeval[:, j], yh)
        auc_t = auc_discriminative(Xtr[:, j], ytr)
        canary[fname] = dict(
            evidence_class="LEAKAGE-CANARY (AUC, not DTI)",
            holdout_auc=round(float(auc_h), 5) if auc_h is not None else None,
            train_loco_auc=round(float(auc_t), 5) if auc_t is not None else None,
            leakage_flag=bool((auc_h and auc_h > 0.90) or (auc_t and auc_t > 0.90)),
        )
    del Xh, Xbe, Xeval, Xp_c, Xtgb_c, Xtr
    gc.collect()
    max_h_auc = max(v["holdout_auc"] for v in canary.values() if v["holdout_auc"] is not None)
    max_t_auc = max(v["train_loco_auc"] for v in canary.values() if v["train_loco_auc"] is not None)
    print(f"[canary] max holdout AUC={max_h_auc:.4f}, max train LOCO AUC={max_t_auc:.4f} (<0.90)", flush=True)

    # ------------------------------------------------------------------
    # Experiment 1: 4-fold holdout across pre-registered arms + deciles
    # ------------------------------------------------------------------
    geo_eff_indices = tuple(range(len(INT.LOCO_GEO_NAMES) + len(INT.LOCO_EFFORT_NAMES)))
    default_sp = {1: 4.0, 2: 4.0, 3: 3.0, 4: 3.0}
    arms = ("intensity_g0", "intensity_g025", "intensity_g05", "intensity_g10",
            "loco_linear_g0", "geophysics_only", "uniform_random")
    terms1 = {k: None for k in arms}
    per_fold1 = []
    sgmc_diag_rows = []

    lam_oof_full = np.zeros(fp.shape, np.float32)
    lam_corridor_folds = []
    models_fold = []

    for q in range(4):
        fold = folds[q]
        allowed = fold["region"] & ~masked_known
        corridor = allowed & (d1 <= 35.0)
        pres_mask = visible & (quad != q) & fp & (d1 > 3.0) & (d1 <= 35.0)
        tgb_mask = (~visible) & (quad != q) & fp & (d1 > 3.0) & (d1 <= 35.0)

        m_full = INT.fit_loco_intensity(frame, bank, pres_mask, tgb_mask, seed=59 + q)
        m_lin = INT.fit_loco_intensity(frame, bank, pres_mask, tgb_mask,
                                       col_indices=INT.LOCO_LINEAR_INDICES, seed=59 + q)
        m_geo = INT.fit_loco_intensity(frame, bank, pres_mask, tgb_mask,
                                       col_indices=geo_eff_indices, seed=59 + q)
        models_fold.append(m_full)

        # Predict full-footprint OOF intensity on this fold for 52,309-block decile calibration & SGMC
        lam_fold_full = INT.predict_loco_intensity(frame, bank, m_full, allowed, gamma=0.0)
        lam_oof_full[allowed] = lam_fold_full[allowed]
        n_b_full, _ = INT.zone_new_fault_density(lam_fold_full, masked_known)
        sgmc_diag_rows.append(dict(fold=FOLD_NAMES[q],
                                   **INT.sgmc_diagnostic(n_b_full, sgmc_only, fp & fold["region"])))
        del lam_fold_full, n_b_full

        # Extract corridor features once and evaluate all 6 intensity arms in-memory
        c_rows, c_cols = np.nonzero(corridor)
        Xc = INT.extract_loco_features(frame, bank, c_rows, c_cols).astype(np.float64)

        def _pred_from_xc(mod: dict, gamma_val: float) -> np.ndarray:
            beta = mod["beta"].copy()
            inter = float(mod["intercept"])
            eidx = mod["eff_idx"]
            if eidx and gamma_val != 0.0:
                beta[eidx] *= (1.0 - float(gamma_val))
                inter += float(gamma_val) * float(np.dot(mod["beta"][eidx], mod["eff_ref"]))
            Xsub = Xc if mod["col_indices"] is None else Xc[:, mod["col_indices"]]
            lg = np.clip(Xsub @ beta + inter, -40.0, 40.0)
            out = np.zeros(fp.shape, np.float32)
            out[c_rows, c_cols] = (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)
            return out

        lam_g0 = _pred_from_xc(m_full, 0.0)
        lam_corridor_folds.append(lam_g0)
        lam_g025 = _pred_from_xc(m_full, 0.25)
        lam_g05 = _pred_from_xc(m_full, 0.50)
        lam_g10 = _pred_from_xc(m_full, 1.00)
        lam_lin = _pred_from_xc(m_lin, 0.0)
        lam_geo = _pred_from_xc(m_geo, 1.00)
        del Xc

        arm_lams = [
            ("intensity_g0", lam_g0),
            ("intensity_g025", lam_g025),
            ("intensity_g05", lam_g05),
            ("intensity_g10", lam_g10),
            ("loco_linear_g0", lam_lin),
            ("geophysics_only", lam_geo),
        ]
        n_dots_g0 = 0
        for name, lam in arm_lams:
            n_b, _ = INT.zone_new_fault_density(lam, masked_known)
            cls_b, _ = INT.density_classes(n_b, min_density=a.break_even)
            dots = INT.tile_blocks_corridor(lam, corridor, cls_b, default_sp,
                                            cell_quantile=a.cell_quantile)
            res, t = EH.evaluate(dots.astype(np.float32), fold, fp)
            terms1[name] = t if terms1[name] is None else terms1[name] + t
            nd = int(dots.sum())
            if name == "intensity_g0":
                n_dots_g0 = nd
            per_fold1.append(dict(
                arm=name, fold=FOLD_NAMES[q], evidence_class="HOLDOUT-DTI",
                evaluator_version=EH.VERSION, withheld_positive_pixels=res["n_truth"],
                dti=round(res["dti"], 6), dots=nd,
            ))
            print(f"[exp1] {FOLD_NAMES[q]} {name:16s}: DTI={res['dti']:.5f} dots={nd}", flush=True)

        # Matched-dot uniform-random baseline
        rng_r = np.random.default_rng(1000 + q)
        cells = np.flatnonzero(allowed.ravel())
        pick = rng_r.choice(cells, size=min(n_dots_g0, cells.size), replace=False)
        rnd = np.zeros(fp.shape, bool)
        rnd.ravel()[pick] = True
        res, t = EH.evaluate(rnd.astype(np.float32), fold, fp)
        terms1["uniform_random"] = t if terms1["uniform_random"] is None else terms1["uniform_random"] + t
        per_fold1.append(dict(
            arm="uniform_random", fold=FOLD_NAMES[q], evidence_class="HOLDOUT-DTI",
            evaluator_version=EH.VERSION, withheld_positive_pixels=res["n_truth"],
            dti=round(res["dti"], 6), dots=int(rnd.sum()),
        ))
        del lam_g025, lam_g05, lam_g10, lam_lin, lam_geo, rnd
        gc.collect()

    n_b_oof, _ = INT.zone_new_fault_density(lam_oof_full, masked_known)
    calib = INT.decile_calibration_blocks(n_b_oof, hidden_all, fp)
    pooled1 = EH.pooled_summary(terms1, draws=1000, seed=20261010, candidate="intensity_g0")

    save(ROOT / "evidence/intensity_exp1_holdout.json", dict(
        evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        per_fold=per_fold1,
        pooled=pooled1,
        calibration=calib,
        canary=canary,
        sgmc_offcatalogue_diagnostic=sgmc_diag_rows,
        fold0_standardized_coefficients=models_fold[0]["std_coef"],
    ))
    save(ROOT / "evidence/intensity_exp1_diagnostics_fixed.json", dict(
        evidence_class="HOLDOUT-STRUCTURE (calibration, not a score)",
        fixes="IR-59-DIAG-01 & IR-59-LOCO-01: 1 km block densities evaluated across all 52,309 footprint blocks",
        evaluator=EH.VERSION,
        split_seed=a.seed,
        withheld_positive_pixels=n_withheld,
        calibration_summary=calib,
        sgmc_offcatalogue_diagnostic=sgmc_diag_rows,
    ))
    print("[exp1] pooled:", json.dumps({k: round(v["dti"], 6) for k, v in pooled1["scores"].items()}),
          flush=True)

    # ------------------------------------------------------------------
    # Experiment 2: Holdout-derived Utility Spacing & Break-Even Table
    # ------------------------------------------------------------------
    cls_names = ["q1_low", "q2", "q3", "q4_high"]
    candidates = {"s3": 3.0, "s4": 4.0, "s6": 6.0, "none": None}
    be_candidates = (1.0, 1.25, 1.5, 2.0)
    terms2 = {"base_s3": None}
    for cn in cls_names:
        for k in candidates:
            if k != "s3":
                terms2[f"{cn}__{k}"] = None
    for be_val in be_candidates:
        terms2[f"be_{be_val}"] = None

    for q in range(4):
        fold = folds[q]
        allowed = fold["region"] & ~masked_known
        corridor = allowed & (d1 <= 35.0)
        lam = lam_corridor_folds[q]
        score = INT.kernel_expected_intensity(lam, corridor)
        n_b, _ = INT.zone_new_fault_density(lam, masked_known)
        cls_b, _ = INT.density_classes(n_b, min_density=a.break_even)

        dots_s3 = INT.tile_blocks_corridor(lam, corridor, cls_b, {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0},
                                           cell_quantile=a.cell_quantile, score=score)
        _, t_base = EH.evaluate(dots_s3.astype(np.float32), fold, fp)
        terms2["base_s3"] = t_base if terms2["base_s3"] is None else terms2["base_s3"] + t_base

        for ci, cn in enumerate(cls_names, start=1):
            for key, sp in candidates.items():
                if key == "s3":
                    continue
                spm = {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0}
                spm[ci] = sp
                dots = INT.tile_blocks_corridor(lam, corridor, cls_b, spm,
                                                cell_quantile=a.cell_quantile, score=score)
                _, t = EH.evaluate(dots.astype(np.float32), fold, fp)
                k = f"{cn}__{key}"
                terms2[k] = t if terms2[k] is None else terms2[k] + t

        for be_val in be_candidates:
            cls_be, _ = INT.density_classes(n_b, min_density=be_val)
            dots_be = INT.tile_blocks_corridor(lam, corridor, cls_be, default_sp,
                                               cell_quantile=a.cell_quantile, score=score)
            _, t_be = EH.evaluate(dots_be.astype(np.float32), fold, fp)
            k_be = f"be_{be_val}"
            terms2[k_be] = t_be if terms2[k_be] is None else terms2[k_be] + t_be
        print(f"[exp2] fold {FOLD_NAMES[q]} utility sweep done ({time.monotonic()-t0:.1f}s)", flush=True)

    table_pooled = EH.pooled_summary(terms2, draws=1000, seed=20261010, candidate="base_s3")
    utility = {}
    selected_sp_map: dict[int, float | None] = {}
    pruned_sp_map: dict[int, float | None] = {}
    for ci, cn in enumerate(cls_names, start=1):
        opts = {k: table_pooled["scores"][f"{cn}__{k}"]["dti"] for k in candidates if k != "s3"}
        opts["s3"] = table_pooled["scores"]["base_s3"]["dti"]
        active_opts = {k: v for k, v in opts.items() if k != "none"}
        best_active = max(active_opts, key=active_opts.get)
        best_any = max(opts, key=opts.get)
        selected_sp_map[ci] = candidates[best_active]
        pruned_sp_map[ci] = candidates[best_any]
        utility[cn] = dict(
            candidates_dti={k: round(v, 6) for k, v in opts.items()},
            selected=best_active,
            selected_dti=round(opts[best_active], 6),
            best_including_none=best_any,
            best_including_none_dti=round(opts[best_any], 6),
            evidence_class="HOLDOUT-DTI",
            note="class-inclusive field: this class at candidate spacing, other classes at 3 px; active spacing selected for blocks clearing break-even",
        )
    be_summary = {
        f"be_{be_val}": dict(
            break_even_px_per_1km_block=be_val,
            dti=round(table_pooled["scores"][f"be_{be_val}"]["dti"], 6),
            ci95=[round(x, 6) for x in table_pooled["scores"][f"be_{be_val}"]["ci95"]],
        )
        for be_val in be_candidates
    }
    save(ROOT / "evidence/intensity_exp2_utility_table.json", dict(
        evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        break_even_min_density=a.break_even,
        density_classes=cls_names,
        utility_table=utility,
        break_even_sensitivity_1_to_2_pct=be_summary,
        baseline_all_s3_dti=round(table_pooled["scores"]["base_s3"]["dti"], 6),
        selection_caveat="table selected on the same holdout it is measured on; CIs are conditional on the selected table and fitted folds",
    ))
    print("[exp2] utility selected (active):", {cn: utility[cn]["selected"] for cn in cls_names}, flush=True)
    print("[exp2] break-even 1-2% table:", json.dumps({k: v["dti"] for k, v in be_summary.items()}), flush=True)

    # ------------------------------------------------------------------
    # Experiment 3: Final locked evaluation with selected utility table
    # ------------------------------------------------------------------
    final_terms = None
    pruned_terms = None
    final_per_fold = []
    for q in range(4):
        fold = folds[q]
        allowed = fold["region"] & ~masked_known
        corridor = allowed & (d1 <= 35.0)
        lam = lam_corridor_folds[q]
        score = INT.kernel_expected_intensity(lam, corridor)
        n_b, _ = INT.zone_new_fault_density(lam, masked_known)
        cls_b, _ = INT.density_classes(n_b, min_density=a.break_even)
        dots = INT.tile_blocks_corridor(lam, corridor, cls_b, selected_sp_map,
                                        cell_quantile=0.0, score=score)
        res, t = EH.evaluate(dots.astype(np.float32), fold, fp)
        final_terms = t if final_terms is None else final_terms + t
        dots_p = INT.tile_blocks_corridor(lam, corridor, cls_b, pruned_sp_map,
                                          cell_quantile=a.cell_quantile, score=score)
        res_p, t_p = EH.evaluate(dots_p.astype(np.float32), fold, fp)
        pruned_terms = t_p if pruned_terms is None else pruned_terms + t_p
        final_per_fold.append(dict(
            fold=FOLD_NAMES[q], evidence_class="HOLDOUT-DTI",
            evaluator_version=EH.VERSION, withheld_positive_pixels=res["n_truth"],
            dti=round(res["dti"], 6), dots=int(dots.sum()),
            pruned_q1_dti=round(res_p["dti"], 6), pruned_q1_dots=int(dots_p.sum()),
        ))
        print(f"[exp3] {FOLD_NAMES[q]} final_locked: DTI={res['dti']:.5f} dots={int(dots.sum())} "
              f"(pruned_q1: DTI={res_p['dti']:.5f} dots={int(dots_p.sum())})", flush=True)

    final_pooled = EH.pooled_summary(
        {"final_locked": final_terms, "final_pruned_q1": pruned_terms, **terms1},
        draws=1000, seed=20261010, candidate="final_locked",
    )
    final_score = final_pooled["scores"]["final_locked"]
    save(ROOT / "evidence/intensity_exp3_final.json", dict(
        evidence_class="HOLDOUT-DTI",
        evaluator_version=EH.VERSION,
        split=dict(version="buffered-whole-components-loqo-v2", seed=a.seed,
                   withheld_positive_pixels=n_withheld),
        locked_config=dict(
            gamma=0.0,
            break_even_min_density=a.break_even,
            cell_quantile=a.cell_quantile,
            spacing_by_class={cn: utility[cn]["selected"] for cn in cls_names},
        ),
        per_fold=final_per_fold,
        pooled=final_pooled,
        comparison_vs_h59i_previous=dict(
            h59i_dti=0.082161,
            h59i_ci95=[0.069542, 0.096158],
            h59i2_dti=round(final_score["dti"], 6),
            h59i2_ci95=[round(x, 6) for x in final_score["ci95"]],
            delta_vs_h59i=round(final_score["dti"] - 0.082161, 6),
            relative_gain_pct=round(100.0 * (final_score["dti"] - 0.082161) / 0.082161, 2),
        ),
        comparison_vs_published_holdout_best=dict(
            published_best_name="relay_bend_anatomy (Session 5, 57GEMSDOE)",
            published_best_dti=0.135204,
            published_best_ci95=[0.119140, 0.153036],
            h59i2_dti=round(final_score["dti"], 6),
            h59i2_ci95=[round(x, 6) for x in final_score["ci95"]],
            delta_vs_published_best=round(final_score["dti"] - 0.135204, 6),
            beats_published_best=bool(final_score["dti"] > 0.135204),
            lower_ci95_beats_published_best_point=bool(final_score["ci95"][0] > 0.135204),
        ),
    ))
    print(f"[exp3] final_locked HOLDOUT-DTI = {final_score['dti']:.6f} "
          f"CI95={[round(x, 6) for x in final_score['ci95']]} ({time.monotonic()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
