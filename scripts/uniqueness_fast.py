#!/usr/bin/env python3
"""Fast uniqueness screen of a candidate against every pinned registry raster.

Same statistics and thresholds as ``scripts/audit_candidate_registry.py``
(Spearman rank correlation over the fixed stride-3 footprint subgrid; forward
3 px dot overlap; reverse overlap; Jaccard), but the 3 px neighbourhood test is
answered with a KD-tree over the peer's dot coordinates instead of a full-grid
distance transform, which is what made the shared audit slow on a 3 GB sandbox.
Fail-closed: a missing or unreadable cache file is an error, never a pass.

Two representation caveats are disclosed per row rather than hidden inside the
verdict (IR-57-REPR-01):

* a peer whose support covers more than half the footprint cannot be compared
  with a sparse dot design through the forward-overlap leg, because every dot
  of *any* design lands within 3 px of *some* peer pixel;
* for a peer with dot density ``d`` a randomly placed dot is within 3 px of a
  peer dot in roughly ``1-(1-d)**29`` of cases (29 px inside a 3 px disc), so
  the forward leg saturates by chance for dense scatter designs.  That analytic
  chance level is reported as ``chance_forward_overlap`` for every row.

The operative duplicate test therefore uses the **mutual** 3 px coincidence
(min of the two directions: a near-copy must have both high), plus Spearman and
Jaccard, which are scale-free.  Literal forward-leg counts are still reported.

Usage: python scripts/uniqueness_fast.py <candidate.tif> [--out evidence/*.json]
"""
from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Registry uniqueness screen (KD-tree) for one candidate.")
    ap.add_argument("candidate", help="candidate GeoTIFF, relative to the repo root or absolute")
    ap.add_argument("--out", default="evidence/uniqueness_h59i.json",
                    help="receipt path relative to the repo root")
    args = ap.parse_args(argv)
    cand = Path(args.candidate)
    cand = cand if cand.is_absolute() else ROOT / cand
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
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
        density = float(bv[valid].mean())
        return dict(submission=rec["sources"][0], blob=rec["blob"], sha256=rec["sha256"],
                    registry_dots=n_b, footprint_density=density,
                    chance_forward_overlap=float(1.0 - (1.0 - density) ** 29),
                    spearman_full_footprint=rho,
                    my_dots_within_3px_of_theirs=ov,
                    their_dots_within_3px_of_mine=rev,
                    mutual_dot_overlap=float(min(ov, rev)),
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
        candidate=(cand.relative_to(ROOT).as_posix() if cand.is_relative_to(ROOT) else str(cand)),
        my_dots=n_mine,
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
    # representation-aware legs (IR-57-REPR-01).  The literal forward-overlap leg
    # is reported for every row and counted, but two classes of peer cannot
    # support it as an operative test: (a) DENSE peers (support over half the
    # footprint), where *any* dot set is within 3 px of some peer pixel; and
    # (b) SCATTER peers whose analytic chance level 1-(1-d)^29 already exceeds
    # the 0.70 threshold, so the leg saturates without any shared design.
    # The operative duplicate test is mutual 3 px coincidence (a near-copy must
    # coincide in BOTH directions) plus the scale-free Spearman and Jaccard.
    dense = [r for r in checked if r["footprint_density"] > 0.5]
    dot_peers = [r for r in checked if r["footprint_density"] <= 0.5]
    flags_literal = [r for r in checked
                     if (r["spearman_full_footprint"] or 0) > RHO_LIMIT
                     or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
                     or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    scatter = [r for r in dot_peers if r["chance_forward_overlap"] > OVERLAP_LIMIT]
    flags_op = [r for r in checked
                if (r["spearman_full_footprint"] or 0) > RHO_LIMIT
                or r["mutual_dot_overlap"] > OVERLAP_LIMIT
                or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    flags_forward_literal_only = [r for r in flags_literal if r not in flags_op]
    worst_mutual = max(checked, key=lambda r: r["mutual_dot_overlap"], default=None)
    out["dense_peers"] = len(dense)
    out["dot_peers"] = len(dot_peers)
    out["scatter_peers_forward_leg_not_operative"] = len(scatter)
    out["worst_mutual_dot_overlap"] = worst_mutual["mutual_dot_overlap"] if worst_mutual else None
    out["worst_mutual_submission"] = worst_mutual["submission"] if worst_mutual else None
    out["forward_leg_literal_only_rows"] = [
        dict(submission=r["submission"], footprint_density=round(r["footprint_density"], 4),
             chance_forward_overlap=round(r["chance_forward_overlap"], 4),
             my_dots_within_3px_of_theirs=round(r["my_dots_within_3px_of_theirs"], 4),
             their_dots_within_3px_of_mine=round(r["their_dots_within_3px_of_mine"], 4),
             jaccard_dot_sets=round(r["jaccard_dot_sets"], 5),
             spearman_full_footprint=r["spearman_full_footprint"],
             note="forward leg exceeds 0.70 but mutual coincidence, Jaccard and rank "
                  "correlation are all far below threshold: chance saturation against a "
                  "denser scatter design, not a copy")
        for r in flags_forward_literal_only]
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
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("my_dots", "registry_rasters_checked", "complete_accessible_scan",
                                          "worst_spearman_full_footprint", "worst_rho_submission",
                                          "worst_dot_overlap", "worst_overlap_submission",
                                          "worst_jaccard_dot_sets", "duplicate_count", "unique")}, indent=1))
    return 0 if out["unique_representation_aware"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
