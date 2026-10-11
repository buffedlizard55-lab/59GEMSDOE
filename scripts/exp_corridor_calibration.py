#!/usr/bin/env python3
"""H59-J Exp 2 -- calibrate a *corridor* holdout against owner-recorded live scores.

Motivation (measured, not assumed)
----------------------------------
`scripts/exp_live_calibration.py` scored 24 owner-labelled sibling rasters on the
shared hide-and-recover draw and found the local instrument is **negatively**
rank-correlated with the recorded public scores
(Spearman -0.437, p = 0.033, n = 24; name-tier subset -0.483, n = 16).
The mechanism is structural: in that draw the withheld truth *is* catalogue that
sits beside the visible catalogue, so the instrument pays for hugging mapped
traces -- the opposite of the live setting, whose truth is faults *absent* from
the catalogue (problem page 967; staff thread 11516).

This experiment builds the missing instrument family and calibrates it the only
way that is honest here: rank the same owner-labelled rasters and compare with
their recorded public scores.

Corridor variants
-----------------
Truth is restricted per fold to withheld catalogue pixels that lie at least
`corridor_px` away from any *visible* mapped trace, i.e. "faults that are not
next to a mapped trace".  Everything else -- the four quadrant blocks, the
whole-component withholding, the 12 px domain erosion, the pixel-exact mask of
the visible catalogue, the official DTI arithmetic -- is the shared template.
`corridor_px = 0` reproduces the shared instrument as a control.

Output: evidence/h59j_corridor_calibration.json  (READ-ONLY otherwise)
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid                    # noqa: E402
from gems59.holdout import buffered_component_draw    # noqa: E402
from gems59 import metric                             # noqa: E402

from exp_live_calibration import LIVE, NAME_TIER      # noqa: E402


def read_pred(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--corridors", default="0,3,6,10")
    ap.add_argument("--files", default=str(ROOT / ".cache/scored"))
    ap.add_argument("--out", default="evidence/h59j_corridor_calibration.json")
    a = ap.parse_args()
    corridors = [float(x) for x in a.corridors.split(",")]
    start = time.monotonic()

    grid = load_grid()
    fp = grid.footprint
    folds, _q = buffered_component_draw(grid, seed=a.seed)

    # per-fold distance to the *visible* mapped traces (the mapping context)
    ctx = []
    for f in folds:
        d_vis = ndi.distance_transform_edt(~f["visible"]).astype(np.float32)
        ctx.append(dict(fold=f, d_vis=d_vis,
                        active=f["region"] & fp & ~f["masked_known"]))
    print(f"[holdout] {len(folds)} folds ready in {time.monotonic()-start:.0f}s "
          f"(whole-component draw, seed {a.seed})", flush=True)

    files = sorted(Path(a.files).glob("*.tif"))
    results = {}
    for corridor in corridors:
        rows = []
        for path in files:
            key = path.name.split("__")[0]
            pred = read_pred(path)
            tp = fpw = fnw = 0.0
            ntruth = 0
            for c in ctx:
                f = c["fold"]
                truth = f["truth"] & (c["d_vis"] >= corridor) if corridor > 0 else f["truth"]
                if not truth.any():
                    continue
                valid = c["active"] & (truth | True)
                r = metric.dti_exact(pred, truth, valid=valid, known=f["masked_known"])
                tp += r["tp"]; fpw += r["fp"]; fnw += r["fn"]; ntruth += r["n_truth"]
            dti = tp / (tp + metric.ALPHA * fpw + metric.BETA * fnw + metric.EPS)
            rows.append(dict(key=key, live_score=LIVE.get(key), dti=float(dti),
                             tp=tp, fp=fpw, fn=fnw, n_truth=int(ntruth)))
        labelled = [r for r in rows if r["live_score"] is not None]
        named = [r for r in labelled if r["key"] in NAME_TIER]
        def corr(x, y):
            if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
                return float("nan"), float("nan")
            return (float(v) for v in spearmanr(x, y))
        rho, p = corr([r["dti"] for r in labelled], [r["live_score"] for r in labelled])
        rho_n, p_n = corr([r["dti"] for r in named], [r["live_score"] for r in named])
        results[f"corridor_{corridor:g}"] = dict(
            corridor_px=corridor, rows=rows,
            n_truth=int(sum(r["n_truth"] for r in rows[:1])),
            spearman_vs_live=dict(rho=None if np.isnan(rho) else float(rho),
                                  p_value=None if np.isnan(p) else float(p), n=len(labelled)),
            spearman_vs_live_name_tier=dict(rho=None if np.isnan(rho_n) else float(rho_n),
                                            p_value=None if np.isnan(p_n) else float(p_n), n=len(named)),
        )
        print(f"[corridor {corridor:g} px] withheld-scored truth={rows[0]['n_truth']}  "
              f"Spearman vs live = {rho:+.3f} (p={p:.4f}, n={len(labelled)})  "
              f"name-tier {rho_n:+.3f} (p={p_n:.4f}, n={len(named)})   "
              f"{time.monotonic()-start:.0f}s", flush=True)

    out = dict(
        evidence_class="HOLDOUT-DTI + OWNER-REPORTED",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose="instrument calibration against owner-recorded public scores",
        split="buffered-whole-components-loqo-v2 (shared) with corridor restriction "
              "of the withheld truth to pixels >= corridor_px from any visible trace",
        seed=a.seed, alpha=metric.ALPHA, beta=metric.BETA, radius_px=metric.RADIUS_PX,
        variants=results,
        labels=dict(dti="HOLDOUT-DTI variant -- NOT a competition score",
                    live_score="OWNER-REPORTED (file-name recorded by the owner in a "
                               "sibling repo, or the standing brief) -- not an organizer receipt"),
    )
    target = ROOT / a.out
    target.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
