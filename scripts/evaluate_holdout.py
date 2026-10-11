#!/usr/bin/env python3
"""Shared holdout instrument: leakage canary + hide-and-recover fold evaluation.

    python scripts/evaluate_holdout.py --canary                 # per-channel leakage screen
    python scripts/evaluate_holdout.py --tag e1                 # 4-fold arm evaluation
    python scripts/evaluate_holdout.py --tag e1 --targets catalogue,offcatalogue

Every candidate in this repository is measured with this file and nothing else
(lane rule 2: reuse, don't rebuild; fix a shared tool once, here).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59 import emit, features, grid, holdout, intensity, metric, pipeline  # noqa: E402

EVIDENCE = ROOT / "evidence"
BLOCK = 10


def load_stack():
    return features.build_stack()[0], grid.read_footprint()


def load_masks(fp):
    cat = grid.read_labels() & fp
    with rasterio.open(grid.DATA / "external/derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = (s.read(1) > 0) & fp
    return cat, sgmc, (sgmc & ~cat)


# ---------------------------------------------------------------------------
# leakage canary (protocol rule 4)
# ---------------------------------------------------------------------------

def _auc(score, pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    p = score[pos]; p = p[np.isfinite(p)]
    n = score[neg]; n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return float("nan")
    if p.size > 20000:
        p = rng.choice(p, 20000, replace=False)
    if n.size > 80000:
        n = rng.choice(n, 80000, replace=False)
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="stable")
    s = allv[order]
    tied = np.empty(allv.size)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        tied[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rp = tied[:p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def run_canary(fold=0, buffer_px=30):
    stack, fp = load_stack()
    cat = grid.read_labels() & fp
    lab, _ = holdout.segments(cat)
    blocks = holdout.assign_blocks(lab, fp.shape, 2, 2)
    withheld, visible, train_area = holdout.fold_masks(lab, blocks, fold, fp, buffer_px)
    features.set_visible_channels(stack, visible, fp)
    eval_area = fp & ~train_area
    truth = withheld & eval_area
    neg_pool = eval_area & ~truth & ~ndi.binary_dilation(visible, np.ones((3, 3), int), 1)
    out = {}
    for i, nm in enumerate(features.CHANNELS):
        s = _auc(stack[i], truth, neg_pool)
        out[nm] = None if not np.isfinite(s) else round(float(max(s, 1 - s)), 4)
    flagged = {k: v for k, v in out.items() if v is not None and v > 0.90}
    res = {"fold": fold, "n_truth": int(truth.sum()), "n_neg": int(neg_pool.sum()),
           "per_channel_max_auc": out, "flagged_above_0.90": flagged,
           "verdict": "FLAGGED" if flagged else "clean",
           "note": ("clean != zero leakage; it means no single channel separates withheld "
                    "fault pixels from background at AUC>0.90 on this fold"),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / "leakage_canary.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ---------------------------------------------------------------------------
# emission arms
# ---------------------------------------------------------------------------

def arm_emissions(field, support, es, withheld, rng, spacing_sweep=None):
    """Every arm is emitted from the same fitted field at matched decisions."""
    arms = {}
    for name, lam in (("plain", field["lam_true"]), ("gap", field["lam_new"])):
        delta = pipeline.calibrate_delta(lam, support, es.g_truth)
        # --- self-consistent break-even: tau = alpha * DTI(tau)
        dti_star, tau, dots = es.dti_target, None, None
        for _ in range(es.self_consistent_iters):
            tau = metric.breakeven_delta(dti_star)
            out = pipeline.emit_from_delta(delta, support,
                                           pipeline.EmissionSpec(**{**es.__dict__, "dti_target": dti_star}))
            dots = out["dots"]
            d = metric.dti(dots.astype(np.float32), withheld) if withheld.any() else 0.0
            if abs(d - dti_star) < 5e-4:
                break
            dti_star = 0.5 * dti_star + 0.5 * d   # damped fixed point
        arms[name] = {"dots": dots, "tau": float(tau), "dti_target_used": float(dti_star),
                      "block_counts": out["block_counts"], "bmap": out["bmap"]}
    # --- control: uniform random pixels, count matched to `gap`
    n_ref = int(arms["gap"]["dots"].sum())
    idx = np.flatnonzero(support.ravel())
    if n_ref and idx.size:
        pick = rng.choice(idx, size=min(n_ref, idx.size), replace=False)
        rnd = np.zeros(support.size, bool); rnd[pick] = True
        arms["random"] = {"dots": rnd.reshape(support.shape), "tau": None,
                          "dti_target_used": None}
    # --- control: literal lattice inside blocks that clear the bar
    delta_gap = pipeline.calibrate_delta(field["lam_new"], support, es.g_truth)
    bc = pipeline.zone_table(delta_gap, support)[0]
    tau_g = arms["gap"]["tau"]
    ok = bc >= metric.breakeven_trace_px_per_km_block(arms["gap"]["dti_target_used"])
    lat = emit.lattice_in_block(delta_gap, support, tau_g, BLOCK, 3.0, ok)
    arms["lattice"] = {"dots": lat, "tau": float(tau_g),
                       "dti_target_used": arms["gap"]["dti_target_used"]}
    return arms


def run_folds(tag="e1", spec=None, buffer_px=30, flank_px=2, folds=(0, 1, 2, 3),
              targets=("catalogue",), es=None, prior_dti=0.28):
    es = es or pipeline.EmissionSpec(dti_target=prior_dti, flank_px=flank_px)
    stack, fp = load_stack()
    cat, sgmc, offcat = load_masks(fp)
    truth_sets = {"catalogue": cat, "offcatalogue": offcat}
    rng = np.random.default_rng(20261011)

    results = {"tag": tag, "emission_spec": es.__dict__, "pp_spec": (spec or intensity.PPSpec()).__dict__,
               "buffer_px": buffer_px, "arms": {}}
    for tgt in targets:
        truth_full = truth_sets[tgt]
        lab, _ = holdout.segments(truth_full)
        blocks = holdout.assign_blocks(lab, fp.shape, 2, 2)
        per_arm_preds, per_arm_truths, fold_rows = {}, {}, []
        for k in folds:
            t0 = time.time()
            withheld, visible, train_area = holdout.fold_masks(lab, blocks, k, fp, buffer_px)
            features.set_visible_channels(stack, cat & ~withheld, fp)
            presence = truth_full & ~withheld & train_area
            field = pipeline.fit_field(stack, fp, presence, spec=spec, area_mask=train_area)
            sup = holdout.emission_support(fp, cat & ~withheld, flank_px=es.flank_px)
            arms = arm_emissions(field, sup, es, withheld, rng)
            row = {"fold": k, "n_truth": int(withheld.sum()),
                   "n_presence_train": int(presence.sum()),
                   "seconds": round(time.time() - t0, 1)}
            for nm, a in arms.items():
                per_arm_preds.setdefault(nm, []).append(a["dots"].astype(np.float32))
                per_arm_truths.setdefault(nm, []).append(withheld)
                row[nm] = {"n_dots": int(a["dots"].sum()),
                           "dti": float(metric.dti(a["dots"].astype(np.float32), withheld)),
                           "tau": a["tau"]}
            fold_rows.append(row)
            print(f"[{tgt}] fold {k} truth={row['n_truth']} " +
                  " ".join(f"{nm}={row[nm]['n_dots']}/{row[nm]['dti']:.4f}" for nm in row if nm not in
                           ("fold", "n_truth", "n_presence_train", "seconds")) +
                  f"  ({row['seconds']}s)", flush=True)
        arm_summary = {}
        for nm in per_arm_preds:
            pooled = holdout.pooled_dti(per_arm_preds[nm], per_arm_truths[nm])
            lo, hi, base = holdout.bootstrap_ci(per_arm_preds[nm], per_arm_truths[nm])
            arm_summary[nm] = {"pooled": pooled, "ci95": [lo, hi], "ci_base": base,
                               "mean_n_dots": float(np.mean([r[nm]["n_dots"] for r in fold_rows]))}
        results["arms"][tgt] = {"folds": fold_rows, "summary": arm_summary}
    results["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / f"holdout_{tag}.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--fold-subset", default="0,1,2,3")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--n-background", type=int, default=200_000)
    ap.add_argument("--targets", default="catalogue")
    ap.add_argument("--prior-dti", type=float, default=0.28)
    ap.add_argument("--g-truth", type=float, default=40_000)
    ap.add_argument("--flank-px", type=int, default=2)
    ap.add_argument("--buffer-px", type=int, default=30)
    a = ap.parse_args()
    if a.canary:
        r = run_canary()
        print(json.dumps({k: v for k, v in r.items() if k != "per_channel_max_auc"}, indent=1))
        for k, v in sorted(r["per_channel_max_auc"].items(), key=lambda kv: -(kv[1] or 0))[:10]:
            print(f"  {k:26s} {v}")
        return 0
    spec = intensity.PPSpec(max_iter=a.max_iter, n_background=a.n_background)
    es = pipeline.EmissionSpec(dti_target=a.prior_dti, flank_px=a.flank_px, g_truth=a.g_truth)
    r = run_folds(tag=a.tag, spec=spec, folds=tuple(int(x) for x in a.fold_subset.split(",")),
                  targets=tuple(a.targets.split(",")), es=es, prior_dti=a.prior_dti,
                  buffer_px=a.buffer_px)
    for tgt, d in r["arms"].items():
        print(f"\n== {tgt} ==")
        for nm, s in d["summary"].items():
            p = s["pooled"]
            print(f"  {nm:10s} pooled DTI {p['dti']:.4f}  95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
                  f"  n_dots~{s['mean_n_dots']:.0f}  TP={p['tp']:.0f} FP={p['fp']:.0f} G={p['n_truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
