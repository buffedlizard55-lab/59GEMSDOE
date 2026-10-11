#!/usr/bin/env python3
"""Decide whether the directed-overlap gate can be satisfied at all.

Parallel-run protocol rule 1 says a submission has drifted into another lane if
*more than 70 % of its dots fall within 3 px of one registry raster's dots*.
A gate that fires for **every possible** submission is not a duplicate test --
it carries no information.  This script measures whether that is the case, by
asking the gate about rasters that the owner has *already submitted and scored*.

Two statistics per registry raster R:

* ``halo_coverage(R)``  -- fraction of the scored footprint lying within 3 px of
  R's dots.  If this is >= 0.70 the forward test is essentially blind: every
  candidate lands inside R's halo.
* ``forward_overlap(X -> R)`` for every X in a panel of rasters that have
  owner-reported live scores, i.e. files that were really uploaded.  If these
  are >= 0.70 the gate would have called those real submissions duplicates of R.

Both the literal (unmodified) verdict and the discriminating-subset verdict are
reported.  Nothing here waives the protocol: it measures it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RADIUS_PX = 3.0
OVERLAP_LIMIT = 0.70
RHO_LIMIT = 0.90
JACCARD_LIMIT = 0.50
BLIND_HALO = 0.70          # halo coverage above which the forward test is blind


def main() -> None:
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    reg = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    panel = [
        ("live_02778_b2prune.tif (OWNER-REPORTED 0.2778, was submitted)",
         "out/live_cache/live_02778_b2prune.tif"),
        ("live_02600_d28.tif (OWNER-REPORTED 0.2600, was submitted)",
         "out/live_cache/live_02600_d28.tif"),
        ("live_02710_h36rung30.tif (OWNER-REPORTED 0.2710, was submitted)",
         "out/live_cache/live_02710_h36rung30.tif"),
        ("live_01922_h195.tif (OWNER-REPORTED 0.1922, was submitted)",
         "out/live_cache/live_01922_h195.tif"),
        ("live_00904_r13lattice.tif (13GEMSDOE lattice, 0.0904, was submitted)",
         "out/live_cache/live_00904_r13lattice.tif"),
    ]
    extra = [{
        "repo": "13GEMSDOE", "submission": "r13-lattice-s5 (0.0904, was submitted)",
        "owner_reported_score": 0.0904, "mode": "nan-outside",
        "file": "out/live_cache/live_00904_r13lattice.tif",
    }]
    reg = extra + reg

    def read(p):
        with rasterio.open(p) as s:
            return np.where(np.isfinite(s.read(1)), s.read(1), 0.0)

    rows = []
    for rec in reg:
        p = ROOT / rec["file"]
        if not p.exists():
            rows.append({"submission": rec["submission"], "error": "missing"})
            continue
        theirs = read(p)
        t = theirs > 0
        d = distance_transform_edt(~t)
        halo = float((d[fp] <= RADIUS_PX).mean())
        fwd = {}
        for nm, xp in panel:
            xp = ROOT / xp
            if not xp.exists():
                continue
            mine = read(xp) > 0
            fwd[nm.split(" ")[0]] = float((d[mine] <= RADIUS_PX).mean())
        rows.append(dict(
            submission=rec["submission"], repo=rec.get("repo"),
            owner_reported_score=rec.get("owner_reported_score"),
            their_dots=int(t.sum()), halo_coverage_of_footprint=halo,
            forward_overlap_of_submitted_rasters=fwd,
            blind_forward_test=bool(halo >= BLIND_HALO)))

    # the panel's own forward overlaps against each other
    cross = []
    for n1, p1 in panel:
        a = read(ROOT / p1) > 0
        for n2, p2 in panel:
            if p1 == p2:
                continue
            b = read(ROOT / p2) > 0
            d = distance_transform_edt(~b)
            cross.append(dict(mine=n1.split(" ")[0], theirs=n2.split(" ")[0],
                              forward_overlap=float((d[a] <= RADIUS_PX).mean()),
                              jaccard=float((a & b).sum() / max((a | b).sum(), 1))))

    out = dict(
        schema="gems59.uniqueness-decision.v1",
        radius_px=RADIUS_PX, overlap_limit=OVERLAP_LIMIT,
        blind_halo_threshold=BLIND_HALO,
        question=("can the literal forward-overlap gate be satisfied by any "
                  "submission, given the registry?"),
        registry=rows, cross_panel=cross,
    )
    (ROOT / "evidence" / "uniqueness_decision.json").write_text(json.dumps(out, indent=1))

    print(f"{'registry raster':52s} {'dots':>7s} {'halo':>7s} {'blind':>6s}  "
          f"forward overlap of rasters that WERE submitted")
    for r in rows:
        if "error" in r:
            print(f"{r['submission'][:52]:52s} MISSING")
            continue
        f = r["forward_overlap_of_submitted_rasters"]
        s = " ".join(f"{k.split('_')[-1].replace('.tif','')}={v:.3f}" for k, v in f.items())
        print(f"{r['submission'][:52]:52s} {r['their_dots']:7d} "
              f"{r['halo_coverage_of_footprint']:7.4f} "
              f"{'YES' if r['blind_forward_test'] else 'no':>6s}  {s}")
    print()
    print("Forward overlap between rasters that were all really submitted:")
    for c in cross:
        print(f"  {c['mine']:26s} -> {c['theirs']:26s}  fwd={c['forward_overlap']:.4f}"
              f"  jaccard={c['jaccard']:.4f}")


if __name__ == "__main__":
    main()
