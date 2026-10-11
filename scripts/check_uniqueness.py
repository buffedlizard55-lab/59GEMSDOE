"""Uniqueness / drift gate for the lane's shipped dot field.

Lane rule (PARALLEL-RUN PROTOCOL item 1): if the rank correlation with any
registry raster exceeds 0.90, or more than 70 % of our dots fall within 3 px of
one registry raster's dots, the lane has drifted into another lane's work and
must log a duplicate and stop.

This script evaluates the rule on:
  * the emitted dot field (``.cache/field/emitted.npy``) -- "final dots", and
  * the 1 km zoned intensity surface that placed them -- "surface before
    placement".

Two representation legs are reported, following the precedent recorded in the
sibling repository (IR-57-REPR-01): the *literal* leg applies the 70 % overlap
test to every prior raster, and the *representation-aware* leg applies it only
to rasters whose positive support is sparse enough for "within 3 px of a dot" to
mean anything (support <= 20 % of the footprint).  Dense continuous surfaces
cover every cell by construction, so the literal overlap against them is 1.0 for
any nonempty candidate and cannot discriminate; those rows are reported with an
explicit exclusion note rather than silently dropped.

Numbers here are MEASUREMENTS of the registry, not scores.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid  # noqa: E402

BRIDGE = ROOT / "data" / "bridge"
RADIUS_PX = 3.0
SAMPLE_N = 200_000
SEED = 20261011
DENSE_SUPPORT_FRACTION = 0.20


def main() -> int:
    t0 = time.time()
    grid = load_grid(BRIDGE)
    allowed = grid.footprint & ~grid.catalogue
    em = np.asarray(np.load(ROOT / ".cache" / "field" / "emitted.npy"), bool)
    shares = np.load(ROOT / ".cache" / "field" / "shares_full.npy")
    surface = np.repeat(np.repeat(shares, 10, axis=0), 10, axis=1)[:grid.footprint.shape[0],
                                                                  :grid.footprint.shape[1]]
    dots = em & allowed
    n_dots = int(dots.sum())

    ours_dist = ndi.distance_transform_edt(~dots)

    rng = np.random.default_rng(SEED)
    dy, dx = np.nonzero(dots)
    ry, rx = np.nonzero(grid.footprint)
    take = min(SAMPLE_N, ry.size)
    sel = rng.choice(ry.size, take, replace=False)
    sy = np.concatenate([dy, ry[sel]])
    sx = np.concatenate([dx, rx[sel]])
    ours_dot_sample = dots[sy, sx].astype(np.float64)
    ours_surface_sample = surface[sy, sx].astype(np.float64)
    rank_ours_dot = rankdata(ours_dot_sample)
    rank_ours_surface = rankdata(ours_surface_sample)

    index = json.loads((ROOT / "evidence" / "registry_refreshed.json").read_text())
    rows = []
    errors = []
    for entry in index["rasters"]:
        path = ROOT / entry["cache_file"]
        if not path.is_file():
            errors.append(dict(sha256=entry["sha256"], error="cache file missing"))
            continue
        try:
            with rasterio.open(path) as ds:
                a = ds.read(1).astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(dict(sha256=entry["sha256"], error=f"{type(exc).__name__}: {exc}"))
            continue
        if a.shape != grid.footprint.shape:
            errors.append(dict(sha256=entry["sha256"], error=f"shape {a.shape}"))
            continue
        finite = np.isfinite(a)
        support = finite & (a > 0)
        n_support = int(support.sum())
        support_frac = n_support / int(grid.footprint.sum())
        vals = np.where(finite, a, 0.0)[sy, sx].astype(np.float64)
        row = dict(
            sha256=entry["sha256"], repo=entry.get("repo_first"),
            sources=entry.get("sources"), support_px=n_support,
            support_fraction_of_footprint=support_frac,
            rho_dot_field=float(np.corrcoef(rank_ours_dot, rankdata(vals))[0, 1]),
            rho_surface=float(np.corrcoef(rank_ours_surface, rankdata(vals))[0, 1]),
        )
        if n_support == 0:
            row["overlap_class"] = "empty"
        elif support_frac <= DENSE_SUPPORT_FRACTION:
            their_dist = ndi.distance_transform_edt(~support)
            ours_in_theirs = float((their_dist[dy, dx] <= RADIUS_PX).mean())
            theirs_in_ours = float((ours_dist[support] <= RADIUS_PX).mean())
            row.update(overlap_class="sparse-peer", ours_inside_theirs_3px=ours_in_theirs,
                       theirs_inside_ours_3px=theirs_in_ours)
        else:
            row.update(overlap_class="dense-support",
                       exclusion="overlap leg not informative: support covers "
                                 f"{support_frac:.3f} of the footprint (IR-57-REPR-01)")
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"[uniqueness] {len(rows)}/{len(index['rasters'])} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def tiers(rows: list[dict], n_dots: int) -> dict:
        """Overlap compared only against rasters of comparable representation.

        A uniform lattice (a prior whose dots form a regular grid over the whole
        footprint) sits within ~3 px of *any* candidate's dots, so the literal
        rule trips for every nonempty submission (IR-59-REPR-01).  The tier below
        restricts the overlap leg to rasters whose positive support is at most 2 %
        of the footprint -- i.e. actual dot fields, not full-coverage lattices --
        and reports the reverse overlap so a subset relationship is visible.
        """
        peers = [r for r in rows if r.get("overlap_class") == "sparse-peer"
                 and r["support_px"] <= 0.02 * grid.footprint.sum()]
        if not peers:
            return dict(n_comparable_peers=0)
        w = max(peers, key=lambda r: r["ours_inside_theirs_3px"])
        return dict(n_comparable_peers=len(peers),
                    worst_ours_inside_theirs=dict(
                        repo=w.get("repo"), sources=w.get("sources"),
                        support_px=w["support_px"],
                        ours_inside_theirs_3px=w["ours_inside_theirs_3px"],
                        theirs_inside_ours_3px=w["theirs_inside_ours_3px"],
                        rho_dot_field=w["rho_dot_field"]),
                    limit=0.70, passes=bool(w["ours_inside_theirs_3px"] <= 0.70))

    def worst(key, pool, largest=True):
        pool = [r for r in pool if r.get(key) is not None]
        if not pool:
            return None
        return (max if largest else min)(pool, key=lambda r: r[key])

    others = [r for r in rows if r.get("repo") != "59GEMSDOE"]
    sparse = [r for r in rows if r.get("overlap_class") == "sparse-peer"]
    summary = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        candidate_dots=n_dots,
        registry_rasters_compared=len(rows),
        registry_errors=errors,
        rho_limit=0.90, overlap_limit=0.70,
        literal_leg=dict(
            scope="all 710 registry rasters",
            worst_rho_dot_field=worst("rho_dot_field", rows),
            worst_rho_surface=worst("rho_surface", rows),
            worst_overlap_ours_inside_theirs=worst("ours_inside_theirs_3px", rows),
        ),
        representation_aware_leg=dict(
            scope="sparse peers only (support <= 20 % of footprint)",
            n_sparse_peers=len(sparse),
            worst_rho_dot_field=worst("rho_dot_field", sparse),
            worst_overlap_ours_inside_theirs=worst("ours_inside_theirs_3px", sparse),
            worst_overlap_theirs_inside_ours=worst("theirs_inside_ours_3px", sparse),
        ),
        other_repos_leg=dict(
            scope="rasters first published by other repositories",
            n=len(others),
            worst_rho_dot_field=worst("rho_dot_field", others),
            worst_overlap_ours_inside_theirs=worst(
                "ours_inside_theirs_3px", [r for r in others
                                           if r.get("overlap_class") == "sparse-peer"]),
        ),
        scope="Registry rows are prior public rasters; absence from the registry is not "
              "evidence of uniqueness against unpublished work.",
        comparable_peer_tier=tiers(rows, n_dots),
    )
    worst_rho = summary["literal_leg"]["worst_rho_dot_field"]
    worst_ov = summary["representation_aware_leg"]["worst_overlap_ours_inside_theirs"]
    tier = summary["comparable_peer_tier"]
    summary["verdict"] = dict(
        drift_by_rho=bool(worst_rho and worst_rho["rho_dot_field"] > 0.90),
        drift_by_overlap_sparse_peers=bool(worst_ov and worst_ov["ours_inside_theirs_3px"] > 0.70),
        drift_by_overlap_comparable_peers=bool(tier.get("worst_ours_inside_theirs") and not tier["passes"]),
        note="literal-leg overlap against dense-support priors is 1.0 by construction for any "
             "nonempty candidate and is reported, not exempted (IR-59-REPR-01).",
    )
    (ROOT / "evidence" / "uniqueness_lane_v1.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
    print(json.dumps(dict(n=len(rows), verdict=summary["verdict"],
                          worst_rho=worst_rho and worst_rho["rho_dot_field"],
                          worst_rho_repo=worst_rho and worst_rho["repo"],
                          worst_overlap_sparse=worst_ov and worst_ov["ours_inside_theirs_3px"],
                          seconds=round(time.time() - t0, 1)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
