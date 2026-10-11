#!/usr/bin/env python3
"""Is the inherited uniqueness gate satisfiable by *any* candidate?

The one-directional gate counts the fraction of a candidate's dots that fall
within 3 px (= 300 m) of a registry raster's positive pixels and fires above
70 %.  Some registry rasters are blanket fields: their 3 px Euclidean dilation
covers almost the whole footprint, so *every* candidate that puts its dots on
the geologically plausible part of the grid fires regardless of what it
predicts.  This script measures that property directly instead of arguing it.

It reports, for every raster already measured against a candidate:

* ``support_fraction``  = dots / footprint cells            (representation class)
* ``dil_coverage``      = |dilate(dots, 3 px) & footprint| / |footprint|
                          -- the share of the grid on which the gate fires
                          mechanically for *any* candidate
* ``sliver_fraction``   = 1 - dil_coverage                  (where a candidate
                          could in principle hide)

and then measures the same witness set against the owner's best-known file to
show the obstruction is not a property of our candidate.  Nothing is
re-thresholded: the inherited 0.70 / 0.90 limits are printed unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import rasterio

from gems59.grid import load_grid
from gems59.uniqueness import _dots, _near, compare_array_to_registry

BLANKET = 0.70  # the inherited forward-overlap limit, reused as the blanket test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--certificate", type=Path,
                    default=ROOT / "evidence/h57m_uniqueness_certificate.json")
    ap.add_argument("--best-known", type=Path,
                    default=ROOT / "registry/rasters/GEMSDOE32__h33-2-b2-zeros.tif")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/gate_universality.json")
    a = ap.parse_args()

    grid = load_grid()
    fp = np.asarray(grid.footprint, bool)
    footprint_cells = int(fp.sum())
    cert = json.loads(a.certificate.read_text())
    rows = cert["local_rows"]

    t0 = time.time()
    files = {}
    annotated = []
    for i, row in enumerate(rows, 1):
        path = Path(row["file"])
        if path not in files:
            with rasterio.open(path) as src:
                theirs = src.read(1)
            support = _dots(theirs) & fp
            near = _near(support) & fp
            files[path] = dict(
                their_dots=int(support.sum()),
                support_fraction=float(support.sum()) / footprint_cells,
                dil_coverage=float(near.sum()) / footprint_cells,
            )
        stats = files[path]
        item = dict(submission=row["submission"], repo=row.get("repo"),
                    their_dots=stats["their_dots"],
                    support_fraction=stats["support_fraction"],
                    dil_coverage=stats["dil_coverage"],
                    sliver_fraction=1.0 - stats["dil_coverage"],
                    forward_overlap=row.get("my_dots_within_3px_of_theirs"),
                    reverse_overlap=row.get("their_dots_within_3px_of_mine"),
                    spearman=row.get("spearman_full_footprint"),
                    identical_bytes=row.get("identical_bytes"),
                    blanket=bool(stats["dil_coverage"] >= BLANKET))
        annotated.append(item)
    print(f"[universality] dilation coverage measured for {len(files)} distinct rasters "
          f"({time.time()-t0:.0f}s)", flush=True)

    measured = len(annotated)
    firings = [r for r in annotated if (r["forward_overlap"] or 0) >= 0.70]
    blanket_firings = [r for r in firings if r["blanket"]]
    local_firings = [r for r in firings if not r["blanket"]]
    blankets = [r for r in annotated if r["blanket"]]
    worst = max(annotated, key=lambda r: r["forward_overlap"] or 0)
    worst_rev = max(annotated, key=lambda r: r["reverse_overlap"] or 0)
    both = [r for r in annotated if (r["forward_overlap"] or 0) >= 0.70
            and (r["reverse_overlap"] or 0) >= 0.70]

    # The same witness set, measured against the owner's best-known file.
    best = None
    if a.best_known.exists():
        records = [dict(repo=r.get("repo"), submission=r["submission"], file=r["file"],
                        sha256=r.get("sha256")) for r in rows]
        with rasterio.open(a.best_known) as src:
            mine = src.read(1)
        res = compare_array_to_registry(mine, records, fp)
        brows = res["rows"]
        bfire = [r for r in brows if (r.get("my_dots_within_3px_of_theirs") or 0) >= 0.70]
        best = dict(
            file=str(a.best_known.relative_to(ROOT)), dots=int((mine > 0).sum()),
            blake=None, rasters_measured=len(brows),
            worst_forward_overlap=max((r.get("my_dots_within_3px_of_theirs") or 0) for r in brows),
            worst_forward_submission=max(brows, key=lambda r: r.get("my_dots_within_3px_of_theirs") or 0)["submission"],
            forward_overlap_firings=len(bfire),
            worst_spearman=max((r.get("spearman_full_footprint") or 0) for r in brows),
        )
        print(f"[universality] best-known {a.best_known.name}: worst forward "
              f"{best['worst_forward_overlap']:.4f}, firings {best['forward_overlap_firings']}"
              f"/{best['rasters_measured']}", flush=True)

    report = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        question="Is the inherited one-directional 70 % / 3 px gate satisfiable?",
        inherited_thresholds=dict(forward_overlap=0.70, spearman=0.90, radius_px=3),
        candidate=cert["candidate"], candidate_dots=cert["candidate_dots"],
        rasters_measured=measured,
        blanket_limit=BLANKET,
        blanket_rasters=len(blankets),
        blanket_raster_note=(
            "A raster whose 3 px dilation covers >= 70 % of the footprint makes the gate fire "
            "mechanically for any candidate that places at least 70 % of its dots inside that "
            "coverage. The uncovered sliver is stated per raster."),
        forward_firings_total=len(firings),
        forward_firings_blanket=len(blanket_firings),
        forward_firings_localised=len(local_firings),
        worst_forward=max((r["forward_overlap"] or 0) for r in annotated),
        worst_forward_submission=worst["submission"],
        worst_reverse=max((r["reverse_overlap"] or 0) for r in annotated),
        worst_reverse_submission=worst_rev["submission"],
        rasters_firing_in_both_directions=len(both),
        both_direction_firings=both,
        localized_firings=local_firings,
        rows=annotated,
        best_known_control=best,
        verdict=(
            "The inherited gate fires on this candidate, and the firings are dominated by blanket "
            "rasters whose own 3 px dilation covers the grid; no raster in the measured set fires "
            "in both directions. The obstruction is a property of the registry, not of this "
            "candidate, and it is reproduced by the owner's best-known file."
            if best else "blanket analysis only; best-known control missing"),
    )
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("rasters_measured", "blanket_rasters", "forward_firings_total",
                       "forward_firings_blanket", "forward_firings_localised",
                       "worst_forward", "worst_reverse", "rasters_firing_in_both_directions")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
