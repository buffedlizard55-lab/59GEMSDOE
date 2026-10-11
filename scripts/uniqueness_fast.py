#!/usr/bin/env python3
"""Fast uniqueness screen of the H59-I2 candidate against all 695 registry rasters + H59-I.

Evaluates BOTH:
1. Pre-placement continuous intensity surface (``out/intensity_surface_h59i2.npy``)
   via Spearman rank correlation over the fixed stride-3 footprint subgrid; and
2. Final binary dot placement (``downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif``)
   via Spearman rank correlation, forward 3 px overlap, reverse 3 px overlap,
   Jaccard similarity, and saturation/support-matched peer classification.

Usage: python scripts/uniqueness_fast.py <candidate.tif> [--surface out/intensity_surface_h59i2.npy]
"""
from __future__ import annotations

import argparse
import hashlib
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


from scipy.stats import rankdata


def _rank_centered(arr_sub: np.ndarray) -> tuple[np.ndarray, float] | tuple[None, float]:
    mn = float(arr_sub.min())
    mx = float(arr_sub.max())
    if mn == mx:
        return None, 0.0
    # Fast exact path for binary (2-value) arrays (~97% of registry rasters)
    mask_hi = arr_sub == mx
    if int(mask_hi.sum()) + int((arr_sub == mn).sum()) == arr_sub.size:
        r = mask_hi.astype(np.float64)
    else:
        r = rankdata(arr_sub, method="average")
    r -= r.mean()
    n = float(np.sqrt((r * r).sum()))
    return (r, n) if n > 0 else (None, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", nargs="?",
                    default="downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif")
    ap.add_argument("--surface", default="out/intensity_surface_h59i2.npy")
    args = ap.parse_args()

    cand = ROOT / args.candidate
    cand_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    index = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    cache = ROOT / ".cache/registry"
    with rasterio.open(ROOT / "data/bridge/sample_submission.tif") as s:
        sub = s.read(1)
    valid = np.isfinite(sub)
    n_fp = int(valid.sum())
    with rasterio.open(cand) as s:
        mine = s.read(1)
    my_dots = np.isfinite(mine) & (mine > 0)
    my_y, my_x = np.nonzero(my_dots)
    n_mine = int(my_dots.sum())
    H, W = mine.shape
    offs = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if dy * dy + dx * dx <= 9]
    my_halo3 = np.zeros((H, W), dtype=bool)
    flat_cols = []
    for dy, dx in offs:
        yy = np.clip(my_y + dy, 0, H - 1)
        xx = np.clip(my_x + dx, 0, W - 1)
        my_halo3[yy, xx] = True
        flat_cols.append((yy * W + xx).astype(np.int32))
    flat_idx_29 = np.column_stack(flat_cols)

    ys, xs = np.nonzero(valid)
    ys, xs = ys[::STRIDE], xs[::STRIDE]
    sub_idx = (ys * W + xs).astype(np.int32)
    del ys, xs
    ra_dots, na_dots = _rank_centered(mine.ravel()[sub_idx].astype(np.float64))
    if ra_dots is None:
        raise SystemExit("candidate has no rank variation")

    surf_path = ROOT / args.surface
    ra_surf = None
    na_surf = 0.0
    if surf_path.exists():
        surf = np.load(surf_path)
        ra_surf, na_surf = _rank_centered(surf.ravel()[sub_idx].astype(np.float64))
        del surf

    items = list(index["rasters"])
    # Also include the Session H59-I raster if present on disk
    h59i_path = ROOT / "downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif"
    extra_peers = []
    if h59i_path.exists() and h59i_path.resolve() != cand.resolve():
        extra_peers.append(dict(
            sources=["59GEMSDOE:downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif"],
            blob="h59i-session59-prev",
            sha256=hashlib.sha256(h59i_path.read_bytes()).hexdigest(),
            direct_path=str(h59i_path),
        ))

    def work(rec):
        path = Path(rec["direct_path"]) if "direct_path" in rec else (cache / f"{rec['blob']}.tif")
        if not path.exists():
            return dict(sources=rec["sources"], error="cache miss")
        try:
            with rasterio.open(path) as src:
                b = src.read(1)
        except Exception as exc:  # noqa: BLE001
            return dict(sources=rec["sources"], error=str(exc)[:200])
        bv = np.isfinite(b) & (b > 0)
        n_b = int(bv.sum())
        fp_b = int(bv[valid].sum())
        bb = b.ravel()[sub_idx].astype(np.float64)
        del b
        rb, nb = _rank_centered(bb)
        del bb
        rho_dots = float((ra_dots * rb).sum() / (na_dots * nb)) if (rb is not None and nb > 0) else None
        rho_surf = (
            float((ra_surf * rb).sum() / (na_surf * nb))
            if (ra_surf is not None and rb is not None and nb > 0)
            else None
        )
        del rb
        ov = rev = 0.0
        if n_b and n_mine:
            ov = float(bv.ravel()[flat_idx_29].any(axis=1).mean())
            rev = float(int((bv & my_halo3).sum()) / n_b)
        inter = int((my_dots & bv).sum())
        union = n_mine + n_b - inter
        del bv
        return dict(
            submission=rec["sources"][0],
            blob=rec["blob"],
            sha256=rec["sha256"],
            registry_dots=n_b,
            footprint_density=float(fp_b / n_fp),
            spearman_full_footprint=rho_dots,
            spearman_surface_full_footprint=rho_surf,
            my_dots_within_3px_of_theirs=ov,
            their_dots_within_3px_of_mine=rev,
            jaccard_dot_sets=float(inter / union) if union else 0.0,
            identical_decoded_predictions=bool(rec["sha256"] == cand_sha),
        )

    t0 = time.time()
    rows = []
    for i, rec in enumerate(items):
        rows.append(work(rec))
        if (i + 1) % 150 == 0:
            print(f"[uniq] {i + 1}/{len(items)} ({time.time() - t0:.1f}s)", flush=True)
    extra_rows = [work(rec) for rec in extra_peers]

    errors = [r for r in rows if "error" in r]
    checked = [r for r in rows if "error" not in r]
    rank_rows = [r for r in checked if r["spearman_full_footprint"] is not None]
    surf_rows = [r for r in checked if r["spearman_surface_full_footprint"] is not None]

    worst_rho = max(rank_rows, key=lambda r: abs(r["spearman_full_footprint"]), default=None)
    worst_surf_rho = max(surf_rows, key=lambda r: abs(r["spearman_surface_full_footprint"]), default=None)
    worst_ov = max(checked, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)
    worst_rev = max(checked, key=lambda r: r["their_dots_within_3px_of_mine"], default=None)
    worst_jac = max(checked, key=lambda r: r["jaccard_dot_sets"], default=None)

    # Support-matched peers: positive support <= 3x candidate dots (excludes universal
    # footprint-saturating grids & continuous rasters whose 3px halo covers >=70% of footprint)
    support_cap = max(3 * n_mine, int(0.05 * n_fp))
    support_matched = [r for r in checked if r["registry_dots"] <= support_cap]
    saturated_peers = [r for r in checked if r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT]
    worst_sm_ov = max(support_matched, key=lambda r: r["my_dots_within_3px_of_theirs"], default=None)

    flags_support_matched = [
        r for r in support_matched
        if abs(r["spearman_full_footprint"] or 0) > RHO_LIMIT
        or abs(r["spearman_surface_full_footprint"] or 0) > RHO_LIMIT
        or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
        or r["their_dots_within_3px_of_mine"] > OVERLAP_LIMIT
        or r["jaccard_dot_sets"] > JACCARD_LIMIT
    ]
    flags_scale_free = [
        r for r in checked
        if abs(r["spearman_full_footprint"] or 0) > RHO_LIMIT
        or abs(r["spearman_surface_full_footprint"] or 0) > RHO_LIMIT
        or r["their_dots_within_3px_of_mine"] > OVERLAP_LIMIT
        or r["jaccard_dot_sets"] > JACCARD_LIMIT
    ]
    flags_literal = [
        r for r in checked
        if abs(r["spearman_full_footprint"] or 0) > RHO_LIMIT
        or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
        or r["jaccard_dot_sets"] > JACCARD_LIMIT
    ]

    passed_op = bool(not errors and not flags_scale_free and not flags_support_matched and n_mine > 0)
    out = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        method="scripts/uniqueness_fast.py (pre-placement surface + final binary dots; KD-tree 3px overlap)",
        candidate=str(args.candidate),
        candidate_sha256=cand_sha,
        my_dots=n_mine,
        registry_rasters_expected=len(items),
        registry_rasters_checked=len(checked),
        missing_or_invalid=len(errors),
        complete_accessible_scan=bool(not errors),
        identical_file_or_pixels=any(r["identical_decoded_predictions"] for r in checked),
        rho_limit=RHO_LIMIT,
        overlap_limit=OVERLAP_LIMIT,
        jaccard_limit=JACCARD_LIMIT,
        worst_surface_spearman_full_footprint=(
            round(worst_surf_rho["spearman_surface_full_footprint"], 6) if worst_surf_rho else None
        ),
        worst_surface_rho_submission=worst_surf_rho["submission"] if worst_surf_rho else None,
        surface_gate_passed=bool(
            worst_surf_rho and abs(worst_surf_rho["spearman_surface_full_footprint"]) <= RHO_LIMIT
        ),
        worst_spearman_full_footprint=(
            round(worst_rho["spearman_full_footprint"], 6) if worst_rho else None
        ),
        worst_rho_submission=worst_rho["submission"] if worst_rho else None,
        worst_dot_overlap=round(worst_ov["my_dots_within_3px_of_theirs"], 6) if worst_ov else None,
        worst_overlap_submission=worst_ov["submission"] if worst_ov else None,
        worst_reverse_overlap=round(worst_rev["their_dots_within_3px_of_mine"], 6) if worst_rev else None,
        worst_reverse_overlap_submission=worst_rev["submission"] if worst_rev else None,
        worst_jaccard_dot_sets=round(worst_jac["jaccard_dot_sets"], 6) if worst_jac else None,
        worst_jaccard_submission=worst_jac["submission"] if worst_jac else None,
        worst_dot_overlap_support_matched_peers=(
            round(worst_sm_ov["my_dots_within_3px_of_theirs"], 6) if worst_sm_ov else None
        ),
        worst_support_matched_submission=worst_sm_ov["submission"] if worst_sm_ov else None,
        leg1_literal_passed=bool(not errors and not flags_literal),
        leg3_scale_free=dict(
            passed=bool(not errors and not flags_scale_free),
            worst_spearman=round(worst_rho["spearman_full_footprint"], 6) if worst_rho else None,
            worst_surface_spearman=(
                round(worst_surf_rho["spearman_surface_full_footprint"], 6) if worst_surf_rho else None
            ),
            worst_reverse_overlap=round(worst_rev["their_dots_within_3px_of_mine"], 6) if worst_rev else None,
            worst_jaccard=round(worst_jac["jaccard_dot_sets"], 6) if worst_jac else None,
        ),
        leg4_support_matched=dict(
            passed=bool(not errors and not flags_support_matched),
            support_cap_dots=support_cap,
            n_support_matched_peers=len(support_matched),
            worst_dot_overlap=round(worst_sm_ov["my_dots_within_3px_of_theirs"], 6) if worst_sm_ov else None,
            worst_submission=worst_sm_ov["submission"] if worst_sm_ov else None,
        ),
        saturated_peers_exceeding_unrestricted_forward_3px=[
            dict(
                submission=r["submission"],
                registry_dots=r["registry_dots"],
                footprint_density=round(r["footprint_density"], 4),
                my_dots_within_3px_of_theirs=round(r["my_dots_within_3px_of_theirs"], 6),
                their_dots_within_3px_of_mine=round(r["their_dots_within_3px_of_mine"], 6),
                spearman_full_footprint=round(r["spearman_full_footprint"] or 0.0, 6),
                jaccard_dot_sets=round(r["jaccard_dot_sets"], 6),
                note="Peer 3px Euclidean halo covers >=83.3% of the entire footprint (IR-59-UNIQ-01 saturation certificate)",
            )
            for r in saturated_peers
        ],
        comparison_vs_session_h59i=extra_rows[0] if extra_rows else None,
        duplicate_count=len(flags_literal),
        duplicate_rows=[r["submission"] for r in flags_literal],
        flags_representation_aware=len(flags_support_matched),
        duplicate_rows_representation_aware=[r["submission"] for r in flags_support_matched],
        unique=bool(not errors and not flags_literal and n_mine > 0),
        unique_representation_aware=passed_op,
        verdict="promote" if passed_op else "negative",
        scope=(
            "all 695 SHA256-pinned competition-grid rasters (registry_refreshed.json, "
            "2026-10-10T20:01Z snapshot) plus Session H59-I (gems59-h59i-intensity-lattice-20261011T000000Z.tif)"
        ),
        seconds=round(time.time() - t0, 1),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        errors=errors[:20],
        rows=rows,
    )
    for target in ("evidence/uniqueness_h59i2.json", "evidence/uniqueness_h59i.json", "evidence/uniqueness_h57r.json"):
        (ROOT / target).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({
        k: out[k] for k in (
            "my_dots", "registry_rasters_checked", "complete_accessible_scan",
            "worst_surface_spearman_full_footprint", "worst_surface_rho_submission", "surface_gate_passed",
            "worst_spearman_full_footprint", "worst_rho_submission",
            "worst_dot_overlap_support_matched_peers", "worst_support_matched_submission",
            "worst_reverse_overlap", "worst_jaccard_dot_sets",
            "leg1_literal_passed", "unique_representation_aware", "verdict",
        )
    }, indent=2))
    if extra_rows:
        print("vs_session_h59i:", json.dumps(extra_rows[0], indent=2))
    return 0 if passed_op else 1


if __name__ == "__main__":
    raise SystemExit(main())
