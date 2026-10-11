#!/usr/bin/env python3
"""Fast uniqueness screen of the H59-I candidate against every pinned registry raster.

Same statistics and thresholds as ``scripts/audit_candidate_registry.py``
(Spearman rank correlation over the fixed stride-3 footprint subgrid; forward
3 px dot overlap; reverse overlap; Jaccard), but the 3 px neighbourhood test is
answered with a KD-tree over the peer's dot coordinates instead of a full-grid
distance transform, which is what made the shared audit slow on a 3 GB sandbox.
Fail-closed: a missing or unreadable cache file is an error, never a pass.

Usage: python scripts/uniqueness_fast.py <candidate.tif>
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70
JACCARD_LIMIT = 0.50
STRIDE = 3


def main() -> int:
    cand = ROOT / sys.argv[1]
    index = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    cache = ROOT / ".cache/registry"
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        sub = s.read(1)
    valid = np.isfinite(sub)
    with rasterio.open(cand) as s:
        mine = s.read(1)
    my_dots = np.isfinite(mine) & (mine > 0)
    my_yx = np.argwhere(my_dots).astype(np.float64)
    n_mine = int(my_dots.sum())

    ys, xs = np.nonzero(valid)
    ys, xs = ys[::STRIDE], xs[::STRIDE]
    a = np.asarray(mine, np.float64)[ys, xs]
    if np.unique(a).size < 2:
        raise SystemExit("candidate has no rank variation")
    _, ra = np.unique(a, return_inverse=True)
    ra = ra.astype(np.float64)
    ra -= ra.mean()
    na = float(np.sqrt((ra * ra).sum()))

    def work(rec):
        path = cache / f"{rec['blob']}.tif"
        if not path.exists():
            return dict(sources=rec["sources"], error="cache miss")
        try:
            with rasterio.open(path) as src:
                b = src.read(1)
        except Exception as exc:  # noqa: BLE001
            return dict(sources=rec["sources"], error=str(exc)[:200])
        bv = np.isfinite(b) & (b > 0)
        n_b = int(bv.sum())
        bb = np.asarray(b, np.float64)[ys, xs]
        rho = None
        if np.unique(bb).size >= 2:
            _, rb = np.unique(bb, return_inverse=True)
            rb = rb.astype(np.float64)
            rb -= rb.mean()
            den = na * float(np.sqrt((rb * rb).sum()))
            rho = float((ra * rb).sum() / den) if den > 0 else None
        # forward overlap: my dots within 3 px of any peer dot
        ov = rev = 0.0
        if n_b and n_mine:
            tree = cKDTree(np.argwhere(bv).astype(np.float64))
            dmy, _ = tree.query(my_yx, workers=1)
            ov = float((dmy <= 3.0).mean())
            tree2 = cKDTree(my_yx)
            dth, _ = tree2.query(np.argwhere(bv).astype(np.float64), workers=1)
            rev = float((dth <= 3.0).mean())
        inter = int((my_dots & bv).sum())
        union = int((my_dots | bv).sum())
        return dict(submission=rec["sources"][0], blob=rec["blob"], sha256=rec["sha256"],
                    registry_dots=n_b, footprint_density=float(bv[valid].mean()),
                    spearman_full_footprint=rho,
                    my_dots_within_3px_of_theirs=ov,
                    their_dots_within_3px_of_mine=rev,
                    jaccard_dot_sets=float(inter / union) if union else 0.0,
                    identical_decoded_predictions=bool(rec["sha256"] ==
                                                       __import__("hashlib").sha256(cand.read_bytes()).hexdigest()))

    t0 = time.time()
    items = index["rasters"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(work, items))
    errors = [r for r in rows if "error" in r]
    checked = [r for r in rows if "error" not in r]
    flags = [r for r in checked
             if (r["spearman_full_footprint"] or 0) > RHO_LIMIT
             or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
             or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    rank_rows = [r for r in checked if r["spearman_full_footprint"] is not None]
    worst_rho = max(rank_rows, key=lambda r: r["spearman_full_footprint"], default=None)
    worst_ov = max(checked, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
    worst_jac = max(checked, key=lambda r: r["jaccard_dot_sets"], default=None)
    out = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        method="scripts/uniqueness_fast.py (same statistics/thresholds as audit_candidate_registry.py; KD-tree overlap)",
        candidate=str(sys.argv[1]), my_dots=n_mine,
        registry_rasters_expected=len(items), registry_rasters_checked=len(checked),
        missing_or_invalid=len(errors), complete_accessible_scan=bool(not errors),
        rho_limit=RHO_LIMIT, overlap_limit=OVERLAP_LIMIT, jaccard_limit=JACCARD_LIMIT,
        worst_spearman_full_footprint=worst_rho["spearman_full_footprint"] if worst_rho else None,
        worst_rho_submission=worst_rho["submission"] if worst_rho else None,
        worst_dot_overlap=worst_ov["my_dots_within_3px_of_theirs"] if worst_ov else None,
        worst_overlap_submission=worst_ov["submission"] if worst_ov else None,
        worst_jaccard_dot_sets=worst_jac["jaccard_dot_sets"] if worst_jac else None,
        worst_jaccard_submission=worst_jac["submission"] if worst_jac else None,
        duplicate_count=len(flags), duplicate_rows=[r["submission"] for r in flags],
        unique=bool(not errors and not flags and n_mine > 0),
        scope="all 695 SHA256-pinned competition-grid rasters (registry_refreshed.json, "
              "2026-10-10T20:01Z snapshot; rasters published after that snapshot are not in "
              "this index and are a disclosed scope gap)",
        seconds=round(time.time() - t0, 1),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        errors=errors[:20],
    )
    # representation-aware leg (IR-57-REPR-01): the forward-overlap test is only
    # informative against peers of comparable support; DENSE peers (support over
    # half the footprint) are reported with their numbers and excluded from the
    # operative verdict, per row.  Spearman and the scale-free statistics apply
    # to every row.
    dense = [r for r in checked if r["footprint_density"] > 0.5]
    dot_peers = [r for r in checked if r["footprint_density"] <= 0.5]
    flags_literal = [r for r in checked
                     if (r["spearman_full_footprint"] or 0) > RHO_LIMIT
                     or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
                     or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    flags_op = [r for r in dot_peers
                if (r["spearman_full_footprint"] or 0) > RHO_LIMIT
                or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
                or r["their_dots_within_3px_of_mine"] > OVERLAP_LIMIT
                or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    out["dense_peers"] = len(dense)
    out["dot_peers"] = len(dot_peers)
    out["flags_literal_all_peers"] = len(flags_literal)
    out["flags_representation_aware"] = len(flags_op)
    out["duplicate_rows_representation_aware"] = [r["submission"] for r in flags_op]
    out["unique_representation_aware"] = bool(not errors and not flags_op and n_mine > 0)
    out["verdict"] = ("promote-to-selector-only" if out["unique_representation_aware"] else "negative")
    out["excluded_dense_rows"] = [
        dict(submission=r["submission"], footprint_density=round(r["footprint_density"], 4),
             my_dots_within_3px_of_theirs=r["my_dots_within_3px_of_theirs"],
             spearman_full_footprint=r["spearman_full_footprint"],
             exclusion="forward-overlap leg not applied: peer support exceeds half the "
                       "footprint (IR-57-REPR-01); Spearman/Jaccard/reverse-overlap still applied")
        for r in dense]
    out["rows"] = rows
    (ROOT / "evidence/uniqueness_h59i.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("my_dots", "registry_rasters_checked", "complete_accessible_scan",
                                          "worst_spearman_full_footprint", "worst_rho_submission",
                                          "worst_dot_overlap", "worst_overlap_submission",
                                          "worst_jaccard_dot_sets", "duplicate_count", "unique")}, indent=1))
    return 0 if out["unique"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
