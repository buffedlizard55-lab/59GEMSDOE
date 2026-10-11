#!/usr/bin/env python3
"""Dot-field geometry: does the winning family trace *lines*, or just fill space?

The 2026-10-11 forensic pass could not explain the owner-reported score spread with
feature-side statistics (band profile, coarse spatial histogram, credit-per-dot on
the SGMC proxy).  The spread is visible at **equal dot budget** (37,654 dots:
0.2778 vs 0.0430), so it must live in the *geometry* of the dot field.

This script measures three scale-free geometry statistics for each artifact:

* ``nn_median_px``  -- median distance to the nearest other dot (packing);
* ``nn_p10_px``     -- 10th percentile (how tight the tightest chains are);
* ``collinearity``  -- median over dots of the ratio (smaller / larger) of the
  eigenvalues of the local dot-neighbourhood covariance inside 6 px.  0.5 is a
  perfectly isotropic neighbourhood (scatter or lattice); values toward 0 mean
  the neighbourhood is elongated -- i.e. the dots form **chains**, like a mapped
  fault trace does.

Only measurements are written; no model is fitted and no threshold is applied.
The numbers are OWNER-REPORTED-score context plus REGISTRY-MEASUREMENT geometry.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
MAX_SAMPLE = 40_000
RADIUS_PX = 6.0


def geometry(dots_yx: np.ndarray, seed: int = 0) -> dict:
    if dots_yx.shape[0] > MAX_SAMPLE:
        idx = np.random.default_rng(seed).choice(dots_yx.shape[0], MAX_SAMPLE, replace=False)
        sample = dots_yx[idx]
    else:
        sample = dots_yx
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)
    nn = d[:, 1]
    neighbours = tree.query_ball_point(sample, RADIUS_PX)
    ratios = []
    for i in range(0, sample.shape[0], 7):
        p = sample[neighbours[i]] - sample[i]
        if p.shape[0] < 4:
            continue
        w = np.linalg.eigvalsh(np.cov(p.T))
        if w.sum() > 0:
            ratios.append(float(w[0] / w.sum()))
    return dict(n_dots=int(dots_yx.shape[0]), n_sampled=int(sample.shape[0]),
                nn_median_px=float(np.median(nn)), nn_p10_px=float(np.percentile(nn, 10)),
                collinearity_median=float(np.median(ratios)) if ratios else None,
                dots_with_enough_neighbours=len(ratios))


def read_dots(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1)
    return np.argwhere(np.isfinite(a) & (a > 0)).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "evidence/emission_geometry_h59j.json"))
    ap.add_argument("--fixed-budget", type=int, default=37654,
                    help="only ledger artifacts with exactly this dot count enter the comparison")
    args = ap.parse_args()

    registry = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    by_sha = {r["sha256"]: r for r in registry["rasters"]}
    ledger = json.loads((ROOT / "evidence/live_ledger_merged.json").read_text())["rows"]

    rows = []
    for row in ledger:
        rec = by_sha.get(row["sha256"])
        if rec is None:
            continue
        path = ROOT / rec["cache_file"]
        if not path.is_file():
            continue
        dots = read_dots(path)
        g = geometry(dots)
        g.update(name=row["name"], owner_reported_score=row["score"],
                 evidence_class="REGISTRY-MEASUREMENT geometry; OWNER-REPORTED score",
                 equal_budget=bool(g["n_dots"] == args.fixed_budget))
        rows.append(g)

    local = []
    for label, rel in (("H59-J shipped 30k sp3", "downloads/gems59-h59j-offcat-propen-30k-20261011.tif"),
                       ("H59-I shipped 71.1k sp4", "downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif")):
        path = ROOT / rel
        if not path.is_file():
            continue
        g = geometry(read_dots(path))
        g.update(name=label, owner_reported_score=None,
                 evidence_class="LOCAL candidate geometry; no organizer score", equal_budget=False)
        local.append(g)

    rows.sort(key=lambda r: -(r["owner_reported_score"] or -1))
    scored = [r for r in rows if r["owner_reported_score"] is not None]
    fixed = [r for r in rows if r["equal_budget"]]
    groups = {"top_ge_0_24": [r for r in scored if r["owner_reported_score"] >= 0.24],
              "mid_0_10_0_24": [r for r in scored if 0.10 <= r["owner_reported_score"] < 0.24],
              "bottom_lt_0_10": [r for r in scored if r["owner_reported_score"] < 0.10]}
    group_stats = {name: dict(
        n=len(g),
        nn_median_px=float(np.median([r["nn_median_px"] for r in g])) if g else None,
        collinearity_median=float(np.median([r["collinearity_median"] for r in g])) if g else None,
        collinearity_p90=float(np.percentile([r["collinearity_median"] for r in g], 90)) if g else None,
        names=[r["name"] for r in g][:20]) for name, g in groups.items()}
    winners, others = groups["top_ge_0_24"], groups["bottom_lt_0_10"]
    finding = dict(
        fixed_budget_px=args.fixed_budget,
        n_fixed_budget=len(fixed),
        winners_ge_0_24=[r["name"] for r in winners],
        others_lt_0_24=[r["name"] for r in others],
        winner_collinearity_median=float(np.median([r["collinearity_median"] for r in winners]))
        if winners else None,
        winner_nn_median_px=float(np.median([r["nn_median_px"] for r in winners])) if winners else None,
        other_collinearity_median=float(np.median([r["collinearity_median"] for r in others]))
        if others else None,
        other_nn_median_px=float(np.median([r["nn_median_px"] for r in others])) if others else None,
        score_groups=group_stats,
        hypothesis=("At equal dot budget the owner-reported winners are tightly packed (nn ~ 3 px) and "
                    "locally elongated (chain-like), while failures are looser or lattice-regular. If the "
                    "hidden truth is a set of line traces, chain geometry is what converts budget into "
                    "credit; this is measurable offline and is the pre-registered H59-K emission rule."),
        not_proven=("n is small (geometry is measured on all ledger artifacts with restored rasters, but "
                    "only a handful sit at the same budget), the score column is owner-reported, and "
                    "geometry is correlated with which field produced the dots. Treat as a hypothesis to "
                    "test, never as a calibrated predictor."),
    )
    out = dict(evidence_class="REGISTRY-MEASUREMENT (geometry) + OWNER-REPORTED (scores)",
               generated_utc=datetime.now(timezone.utc).isoformat(),
               script="scripts/emission_geometry.py",
               sample_cap=int(MAX_SAMPLE), neighbour_radius_px=RADIUS_PX,
               finding=finding, rows=rows, local_candidates=local)
    Path(args.out).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(f"fixed-budget artifacts: {len(fixed)} | winners {len(winners)} | others {len(others)}")
    print(f"winner median collinearity {finding['winner_collinearity_median']} "
          f"nn {finding['winner_nn_median_px']}")
    print(f"other  median collinearity {finding['other_collinearity_median']} "
          f"nn {finding['other_nn_median_px']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
