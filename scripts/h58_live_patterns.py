#!/usr/bin/env python3
"""Measure what the registry rasters actually look like, against owner-reported scores.

No model, no fitting, no score projection.  Every number is read from raster bytes
that are physically present in this workspace, so the table can be re-derived by
anyone who checks out the repository.  This is the answer to "why did the
0.2778 file score highest": it is a *file measurement*, not an organiser receipt.

Outputs ``evidence/live_submission_patterns.json``:

* per-raster support statistics (dot count, distance-to-catalogue quantiles,
  nearest-neighbour spacing, fraction of dots within 3 px of the mapped
  catalogue, minimum catalogue distance);
* the containment test between the two owner-reported best files
  (0.2708 base and 0.2778 pruned) and the exact rule that separates them;
* Spearman rank correlation between support statistics and the owner-reported
  score, with n and the caveat that method and budget change together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from gems59 import load_grid  # noqa: E402

CAT_NAME = "GEMSDOE32__h33-2-b2-zeros.tif"
BASE_NAME = "GEMSDOE28__h27-4-r1-solo-d2-8.tif"


def support(path: Path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
    m = np.isfinite(a) & (a > 0)
    return m, a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(ROOT / "registry/registry_index.json"))
    ap.add_argument("--out", default=str(ROOT / "evidence/live_submission_patterns.json"))
    a = ap.parse_args()
    grid = load_grid()
    dcat = ndi.distance_transform_edt(~grid.catalogue)
    rows = json.loads(Path(a.registry).read_text())
    stats = []
    for rec in rows:
        p = ROOT / rec["file"]
        if not p.is_file():
            stats.append(dict(submission=rec["submission"], repo=rec["repo"], error="file missing"))
            continue
        m, arr = support(p)
        m &= grid.footprint
        y, x = np.nonzero(m)
        if y.size == 0:
            stats.append(dict(submission=rec["submission"], repo=rec["repo"], dots=0))
            continue
        nn = cKDTree(np.stack([y, x], 1)).query(np.stack([y, x], 1), k=2)[0][:, 1]
        d = dcat[m]
        stats.append(dict(
            submission=rec["submission"], repo=rec["repo"],
            file=rec["file"], sha256=rec["sha256"],
            owner_reported_score=rec.get("owner_reported_score"),
            mode=rec.get("mode"), dots=int(m.sum()),
            max_value=float(arr[m].max()), min_value=float(arr[m].min()),
            median_distance_to_catalogue_px=float(np.median(d)),
            p10_distance_to_catalogue_px=float(np.percentile(d, 10)),
            fraction_within_3px_of_catalogue=float((d <= 3).mean()),
            min_distance_to_catalogue_px=float(d.min()),
            median_neighbour_spacing_px=float(np.median(nn)),
            fraction_catalogue_pixels_hit=float((grid.catalogue & m).sum() / max(grid.catalogue.sum(), 1)),
        ))
    scored = [s for s in stats if s.get("owner_reported_score") and s.get("dots")]
    corr = {}
    for key in ("dots", "median_distance_to_catalogue_px", "p10_distance_to_catalogue_px",
                "fraction_within_3px_of_catalogue", "median_neighbour_spacing_px"):
        v = np.array([s[key] for s in scored], float)
        y = np.array([s["owner_reported_score"] for s in scored], float)
        rho = spearmanr(v, y)
        corr[key] = dict(spearman_vs_owner_reported_score=float(rho.statistic),
                         p_value=float(rho.pvalue), n=int(len(scored)),
                         range=[float(v.min()), float(v.max())])
    # containment test between the owner-reported best and its alleged base
    best = ROOT / "registry/rasters" / CAT_NAME
    base = ROOT / "registry/rasters" / BASE_NAME
    pair = {}
    if best.is_file() and base.is_file():
        mb, _ = support(best)
        mb &= grid.footprint
        ab, _ = support(base)
        ab &= grid.footprint
        removed = ab & ~mb
        pair = dict(
            best_file_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
            base_file_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
            best_dots=int(mb.sum()), base_dots=int(ab.sum()),
            best_is_subset_of_base=bool((mb & ~ab).sum() == 0),
            removed_dots=int(removed.sum()), added_dots=int((mb & ~ab).sum()),
            removed_max_distance_to_catalogue_px=float(dcat[removed].max()) if removed.any() else 0.0,
            kept_min_distance_to_catalogue_px=float(dcat[mb].min()),
            equals_base_pruned_at_2px=bool(np.array_equal(mb, ab & (dcat > 2.0))),
        )
    out = dict(
        evidence_class="RASTER-MEASUREMENT (registry bytes; not a score, not a receipt)",
        generated_utc=__import__("datetime").datetime.utcnow().isoformat() + "Z",
        catalogue_pixels=int(grid.catalogue.sum()), footprint_pixels=int(grid.footprint.sum()),
        rows=stats, rank_correlations=corr, best_vs_base=pair,
        caveats=[
            "owner_reported_score is copied from the owner's prompt text, not from a "
            "submission-page receipt; no file-to-score attribution is established here.",
            "Method and dot budget change together across registries, so a rank correlation "
            "between a statistic and a reported score is confounded and cannot be read as a "
            "causal effect.",
            "Distance to catalogue is measured against the public mapped-fault raster, not "
            "against the private new-fault truth.",
        ],
    )
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=len(stats), scored=len(scored), best_vs_base=pair), indent=2))
    for key, v in corr.items():
        print(f"  {key:38s} rho={v['spearman_vs_owner_reported_score']:+.4f} "
              f"p={v['p_value']:.3g} n={v['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
