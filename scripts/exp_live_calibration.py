#!/usr/bin/env python3
"""H59-J Exp 1 -- calibrate the shared holdout instrument against the live leaderboard.

The repo's central unresolved tension (`REMAINING_WORK.md` section 1) is that no
*submitted* candidate has ever been scored on the shared hide-and-recover
holdout, so the instrument has never been validated against a real score.

This experiment closes that loop with the only labelled data that exists: the
owner's own public submission files whose *actual* public DW-Tversky score is
recorded in the file name (``GEMSDOE48/data/raw/scored/*.tif`` in the sibling
registry, ``7GEMSDOE/external/scored/*.tif``), e.g. ``h36_rung30_02710.tif`` →
0.2710.  Each file is scored with the shared evaluator
``gems59.evaluate_holdout.evaluate`` on the shared seed-20 buffered
whole-component draw, and the per-file holdout DTI is correlated (Spearman)
against the recorded live score.

Everything printed here is labelled:
  * ``HOLDOUT-DTI``  -- local instrument (evaluator version recorded).
  * ``OWNER-REPORTED`` -- score copied from the file name recorded by the owner
    in a sibling repository; not an organizer receipt (see irregularities).
  * ``NOT-AN-ORGANIZER-SCORE`` -- the holdout number is never a competition score.

Read-only: this script writes only ``evidence/h59j_live_calibration.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import evaluate_holdout as EH          # noqa: E402
from gems59.grid import load_grid                  # noqa: E402
from gems59.holdout import buffered_component_draw  # noqa: E402
from gems59 import metric                          # noqa: E402

# Owner-recorded public scores.  Evidence tier "name": the score is written into
# the *file name* by the owner in a sibling repo (GEMSDOE48 data/raw/scored/,
# data/families/, 17GEMSDOE, GEMSDOE54 registry).  Tier "brief": the score comes
# from the standing brief's site list, which cross-checks with the repos for the
# files that appear in both.  No organizer receipt is reachable from this
# session, so every score here is OWNER-REPORTED, never ORGANIZER-CONFIRMED.
LIVE = {
    "dotted_b2_prune_02778": 0.2778,
    "anderson_pinn_02750": 0.2750,
    "h36_rung30_02710": 0.2710,
    "dotted_d2_8_02708": 0.2708,
    "h32_prethin_tip_02649": 0.2649,
    "tip_stepover_r30_02632": 0.2632,
    "probe_S1_anchor_02600": 0.2600,
    "dotted_d2_8_02600": 0.2600,
    "d15_02477": 0.2477,
    "h19_5_01922": 0.1922,
    "lb_anchor_h19_5_01922": 0.1922,
    "lb_anchor_h19_4_01894": 0.1894,
    "lb_anchor_h16_1_01855": 0.1855,
    "a_verified_01563": 0.1563,
    "hedge_v2_01563": 0.1563,
    "gemsdoe1-ens12": 0.1563,
    "gemsdoe2-dual-union": 0.1560,
    "gemsdoe3-pindrop-nodes": 0.1193,
    "gemsdoe3-pindrop-ridge": 0.1152,
    "r13_lattice_s5_00904": 0.0904,
    "gemsdoe3-pindrop-discovery": 0.0830,
    "sgmc44k_00512": 0.0512,
    "gemsdoe4-combined-237f0063": 0.0343,
    "gems6-hgb88-topk03": 0.0286,
    "gemsdoe3-sgmc-gap": None,
    "c0_scored_reference": None,
}
# score written inside the file name by the owner (highest-confidence tier here)
NAME_TIER = {
    "dotted_b2_prune_02778", "h36_rung30_02710", "dotted_d2_8_02708",
    "h32_prethin_tip_02649", "tip_stepover_r30_02632", "probe_S1_anchor_02600",
    "dotted_d2_8_02600", "d15_02477", "h19_5_01922", "lb_anchor_h19_5_01922",
    "lb_anchor_h19_4_01894", "lb_anchor_h16_1_01855", "a_verified_01563",
    "hedge_v2_01563", "r13_lattice_s5_00904", "sgmc44k_00512",
}


def read_pred(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--files", default=str(ROOT / ".cache/scored"))
    ap.add_argument("--out", default="evidence/h59j_live_calibration.json")
    a = ap.parse_args()
    start = time.monotonic()

    grid = load_grid()
    fp = grid.footprint
    cat = grid.catalogue
    print(f"[grid] footprint={int(fp.sum())} catalogue={int(cat.sum())}", flush=True)

    folds, _quad = buffered_component_draw(grid, seed=a.seed)
    print(f"[holdout] draw built in {time.monotonic()-start:.0f}s "
          f"withheld={int(sum(f['truth'].sum() for f in folds))}", flush=True)

    rows = []
    for path in sorted(Path(a.files).glob("*.tif")):
        key = path.name.split("__")[0]
        pred = read_pred(path)
        terms = np.zeros(4, np.float64)
        for fold in folds:
            r, t = EH.evaluate(pred, fold, fp)
            terms += t.sum(axis=0)
        tp, fpw, fnw, ntruth = (float(x) for x in terms)
        pooled = EH.from_terms(terms.reshape(1, 4))[0]
        rows.append(dict(key=key, file=path.name, live_score=LIVE.get(key),
                         holdout_dti=float(pooled), tp=tp, fp=fpw, fn=fnw,
                         n_truth=int(ntruth),
                         emitted=int((pred > 0).sum()),
                         mean_value=float(pred[fp].mean())))
        print(f"[{key:30s}] HOLDOUT-DTI {pooled:.6f}  live={LIVE.get(key)}  "
              f"(tp={tp:.1f} fp={fpw:.1f} fn={fnw:.1f})  {time.monotonic()-start:.0f}s", flush=True)

    def corr(x, y):
        if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan"), float("nan")
        return (float(v) for v in spearmanr(x, y))
    labelled = [r for r in rows if r["live_score"] is not None]
    named = [r for r in labelled if r["key"] in NAME_TIER]
    h = [r["holdout_dti"] for r in labelled]
    s = [r["live_score"] for r in labelled]
    rho, p = corr(h, s)
    hn = [r["holdout_dti"] for r in named]
    sn = [r["live_score"] for r in named]
    rho_n, p_n = corr(hn, sn)
    pear, pp = corr([np.log10(max(r["emitted"], 1)) for r in labelled], s)
    pear_n, pp_n = corr([np.log10(max(r["emitted"], 1)) for r in named], sn)
    dens = [r["emitted"] for r in labelled]
    rho_d, p_d = corr(dens, s)
    out = dict(
        evidence_class="HOLDOUT-DTI + OWNER-REPORTED",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evaluator_version=EH.VERSION,
        evaluator_hashes=EH.implementation_hashes(),
        holdout_split="buffered-whole-components-loqo-v2",
        seed=a.seed,
        alpha=metric.ALPHA, beta=metric.BETA, kernel_radius_px=metric.RADIUS_PX,
        n_files=len(rows),
        n_labelled=len(labelled),
        spearman_holdout_vs_live=dict(rho=None if np.isnan(rho) else float(rho),
                                      p_value=None if np.isnan(p) else float(p),
                                      n=len(labelled)),
        spearman_holdout_vs_live_name_tier=dict(
            rho=None if np.isnan(rho_n) else float(rho_n),
            p_value=None if np.isnan(p_n) else float(p_n), n=len(named)),
        spearman_log_emitted_vs_live=dict(
            rho=None if np.isnan(pear) else float(pear),
            p_value=None if np.isnan(pp) else float(pp), n=len(labelled)),
        spearman_log_emitted_vs_live_name_tier=dict(
            rho=None if np.isnan(pear_n) else float(pear_n),
            p_value=None if np.isnan(pp_n) else float(pp_n), n=len(named)),
        spearman_emitted_vs_live=dict(rho=None if np.isnan(rho_d) else float(rho_d),
                                      p_value=None if np.isnan(p_d) else float(p_d),
                                      n=len(labelled)),
        rows=rows,
        labels=dict(
            holdout_dti="HOLDOUT-DTI (local instrument, evaluator "
                        f"{EH.VERSION}, draw seed {a.seed}) -- NOT an organizer score",
            live_score="OWNER-REPORTED: score recorded by the owner in a sibling-repo "
                       "file name (GEMSDOE48/data/raw/scored, 7GEMSDOE/external/scored); "
                       "no organizer receipt is reachable from this session",
        ),
        note="This is an instrument-calibration measurement, not a submission score.",
    )
    target = ROOT / a.out
    target.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"\nSpearman(holdout DTI, owner-reported live score) = {rho:+.3f} "
          f"(p={p:.4f}, n={len(labelled)})")
    print(f"  name-tier subset (n={len(named)}): rho={rho_n:+.3f} p={p_n:.4f}")
    print(f"Spearman(log10 emitted, live score)              = {pear:+.3f} (p={pp:.4f})")
    print(f"  name-tier subset (n={len(named)}): rho={pear_n:+.3f} p={pp_n:.4f}")
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
