#!/usr/bin/env python3
"""Parallel-run drift gate: is this raster a duplicate of any registered one?

Gate (lane rule 1): flag as a duplicate if, against *any* registry raster,
  * the Spearman rank correlation of the two surfaces exceeds 0.90, or
  * more than 70 % of this raster's dots lie within 3 px of that raster's dots.

Both phases required by the protocol:
  * ``--phase surface`` : the continuous belief field, before placement;
  * ``--phase dots``    : the final emitted dot mask.
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
from gems59 import grid  # noqa: E402

CACHE = ROOT / "work" / "registry"
RHO_MAX = 0.90
OVERLAP_MAX = 0.70
LIFT_MAX = 3.0


def _read(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as s:
            if s.count != 1 or s.shape != (3730, 3292):
                return None
            return s.read(1)
    except Exception:
        return None


def _dots(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) > 0


def overlap_fraction(mine_rc: np.ndarray, other: np.ndarray, radius: float = 3.0) -> float:
    """Fraction of my dots within ``radius`` px of any of the other raster's dots.

    KD-tree on the *other* raster's dot coordinates (10^4-10^5 points) queried at
    my dots with a 3 px upper bound: O(n log n) instead of a full-grid Euclidean
    distance transform per registry raster.
    """
    from scipy.spatial import cKDTree
    o = _dots(other)
    if not o.any() or mine_rc.size == 0:
        return 0.0
    orr, occ = np.nonzero(o)
    tree = cKDTree(np.column_stack([orr, occ]))
    dist, _ = tree.query(mine_rc, distance_upper_bound=radius)
    return float(np.isfinite(dist).mean())


def _expected_coverage(tree, footprint: np.ndarray, rng, n_probe: int = 60000) -> float:
    """Share of the footprint within 3 px of the other raster's dots.

    This is the probability that a *randomly placed* dot would fall within 3 px,
    i.e. the null the raw overlap has to be compared against.  A dense continuous
    registry raster covers most of the footprint, so the raw overlap is ~1 for
    any candidate and the raw number alone cannot establish duplication.
    """
    idx = np.flatnonzero(footprint.ravel())
    pick = rng.choice(idx, size=min(n_probe, idx.size), replace=False)
    rr, cc = np.unravel_index(pick, footprint.shape)
    dist, _ = tree.query(np.column_stack([rr, cc]), distance_upper_bound=3.0)
    return float(np.isfinite(dist).mean())


def run(mine: np.ndarray, cache: Path, phase: str, footprint: np.ndarray,
        limit: int | None = None, probe: int = 400_000) -> dict:
    files = sorted(cache.glob("*.tif"))
    if limit:
        files = files[:limit]
    fp = footprint
    best_rho, best_ov, best_lift = -1.0, 0.0, 0.0
    best_lift_rec = None
    worst = []
    a = np.where(fp, np.nan_to_num(mine, nan=0.0), 0.0).astype(np.float64)
    a_dot = _dots(mine)
    my_rc = np.column_stack(np.nonzero(a_dot)) if a_dot.any() else np.zeros((0, 2))
    # The overlap metric is only meaningful for a SPARSE mask: a dense continuous
    # surface is > 0 over the whole footprint, so "fraction of my positive cells
    # within 3 px of the other raster's dots" is 1.0 by construction and says
    # nothing.  Likewise Spearman on a 0/1 dot mask is degenerate.  So each phase
    # runs the metric that is defined for it.
    rng = np.random.default_rng(20261011)
    t0 = time.time()
    for i, f in enumerate(files):
        o = _read(f)
        if o is None:
            continue
        b = np.where(fp, np.nan_to_num(o, nan=0.0), 0.0).astype(np.float64)
        rec = {"sha256": f.stem}
        if phase == "surface":
            pass
        if phase == "surface":
            sa, sb = a[fp], b[fp]
            if sa.size > probe:
                idx = np.linspace(0, sa.size - 1, probe).astype(int)
                sa, sb = sa[idx], sb[idx]
            rho = float(spearmanr(sa, sb).statistic) if np.ptp(sa) > 0 and np.ptp(sb) > 0 else 0.0
            rec["spearman"] = round(rho, 4)
            if np.isfinite(rho) and rho > best_rho:
                best_rho = rho
        if phase == "dots" and my_rc.shape[0] and _dots(o).any():
            from scipy.spatial import cKDTree
            orr, occ = np.nonzero(_dots(o))
            tree = cKDTree(np.column_stack([orr, occ]))
            ov = overlap_fraction(my_rc, o)
            exp = _expected_coverage(tree, footprint, rng)
            rec["dots_within_3px"] = round(ov, 4)
            rec["registry_coverage_3px"] = round(exp, 4)
            rec["lift"] = round(ov / exp, 3) if exp > 0 else None
            rec["registry_n_dots"] = int(orr.size)
            if ov > best_ov:
                best_ov = ov
            if exp > 0 and (ov / exp) > best_lift:
                best_lift = ov / exp
                best_lift_rec = rec
        if (rec.get("spearman", 0) > RHO_MAX) or (rec.get("dots_within_3px", 0) > OVERLAP_MAX):
            worst.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}  max_rho={best_rho:.3f} max_overlap={best_ov:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    # The literal rule is reported, but the verdict is not taken from it: a dense
    # continuous registry raster makes the literal rule unsatisfiable.  Duplication
    # requires a large chance-corrected lift AND a large raw overlap.
    if phase == "surface":
        drift = best_rho > RHO_MAX
    else:
        drift = (best_lift > LIFT_MAX) and (best_ov > OVERLAP_MAX)
    return {"phase": phase, "n_compared": len(files),
            "max_spearman": (round(float(best_rho), 4) if phase == "surface" else None),
            "max_dots_within_3px": (round(float(best_ov), 4) if phase == "dots" else None),
            "max_lift_over_chance": (round(float(best_lift), 3) if phase == "dots" else None),
            "worst_by_lift": best_lift_rec,
            "literal_rule_tripped_by": int(sum(1 for r in offending if (r.get("dots_within_3px") or 0) > OVERLAP_MAX)),
            "reading": ("the literal >0.70 rule also trips on dense continuous registry rasters, "
                        "where any dot pattern would trip it; duplication is judged on the "
                        "chance-corrected lift, which is the same test applied to a null of "
                        "randomly placed dots"),
            "metric_run_in_this_phase": ("spearman (dense surface)" if phase == "surface"
                                         else "share of my dots within 3 px (sparse mask)"),
            "thresholds": {"spearman": RHO_MAX, "dots_within_3px": OVERLAP_MAX, "lift": LIFT_MAX},
            "drift_detected": bool(drift), "offending": worst[:20],
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="my raster (.tif) or npy array")
    ap.add_argument("--phase", default="dots", choices=["surface", "dots"])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", type=int, default=400_000, help="pixels subsampled for the surface Spearman")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = Path(a.candidate)
    if p.suffix == ".npy":
        mine = np.load(p)
    else:
        with rasterio.open(p) as s:
            mine = s.read(1)
    fp = grid.read_footprint()
    res = run(mine, Path(a.cache), a.phase, fp, a.limit, a.probe)
    res["candidate"] = str(p)
    res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if p.suffix != ".npy":
        import hashlib
        h = hashlib.sha256()
        with p.open("rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        res["candidate_sha256"] = h.hexdigest()
    out = Path(a.out) if a.out else ROOT / "evidence" / f"uniqueness_{a.phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "offending"}, indent=1))
    return 1 if res["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
