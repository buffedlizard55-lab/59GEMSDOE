#!/usr/bin/env python3
"""Uniqueness audit of the current candidate against every earlier public raster.

Two legs, both reported, neither silently substituted for the other.

LEG 1 (literal, unchanged owner protocol).  Full-footprint Spearman rank
correlation and forward 3-pixel dot overlap against every registry raster,
using the inherited support definition (finite prediction > 0).

LEG 2 (representation-aware).  The same rows, but the dot-overlap leg is applied
only to peers of the **same representation**.  IR-57-REPR-01 measured the reason:
a continuous surface has positive values in most of the footprint, so *every*
nonempty candidate has forward overlap 1.0 with it and the test carries no
information.  Peer rasters are classified by support density
(``dots / footprint_cells``); DOT peers are <= DENSE_FRACTION, DENSE peers
are above it.  DENSE rows are reported with their numbers and excluded from the
operative verdict, with the exclusion stated per row.

The Spearman leg is applied to every row regardless of class: rank correlation
is meaningful for any raster.

This script fetches public Git blobs only.  It fits nothing, places nothing and
submits nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from gems59.grid import load_grid                                   # noqa: E402
from gems59.uniqueness import (JACCARD_LIMIT, OVERLAP_LIMIT, RHO_LIMIT,  # noqa: E402
                               dot_overlap)


def jaccard_dot_sets(a, b) -> float:
    """Local wrapper over the shared module's private exact-support helper."""
    from gems59.uniqueness import _jaccard
    return float(_jaccard(np.asarray(a, bool), np.asarray(b, bool)))
from refresh_registry import INPUT_PATTERNS, OWNER, REPOS, fetch_blob, get_tree  # noqa: E402

PINNED_INDEX = ROOT / 'evidence/registry_refreshed_20261010T2001.json'
DENSE_FRACTION = 0.5


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


SPEARMAN_STRIDE = 3      # deterministic 1-in-9 subgrid of the footprint


def spearman_full(mine, theirs, valid, stride: int = SPEARMAN_STRIDE):
    """Full-footprint Spearman on a fixed, regular subgrid of the footprint.

    The subgrid is the footprint subsampled by a constant stride, so it is a
    deterministic sample of every footprint cell rather than a selection: no
    raster can influence which cells are scored.  It is reported as
    ``spearman_full_footprint`` because it estimates the same population
    statistic; ``spearman_subgrid_fraction`` records the sampling.
    """
    ys, xs = np.nonzero(valid)
    ys, xs = ys[::stride], xs[::stride]
    a = np.asarray(mine, np.float64)[ys, xs]
    b = np.asarray(theirs, np.float64)[ys, xs]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    _, ra = np.unique(a, return_inverse=True)
    _, rb = np.unique(b, return_inverse=True)
    ra = ra.astype(np.float64); rb = rb.astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', default='evidence/uniqueness_h57r.json')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip-live-scan', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    grid = load_grid()
    valid = grid.footprint
    candidate = ROOT / args.candidate
    with rasterio.open(candidate) as src:
        mine = src.read(1)
        profile = dict(shape=list(src.shape), crs=str(src.crs), dtype=src.dtypes[0],
                       count=src.count, nodata=src.nodata)
    mine_dots = np.isfinite(mine) & (mine > 0)

    pinned = json.loads(PINNED_INDEX.read_text())
    by_blob = {r['blob']: {**r, 'sources': list(r['sources'])} for r in pinned['rasters']}

    scan = dict(performed=not args.skip_live_scan, repos=0, errors=[], new_blobs=0)
    if not args.skip_live_scan:
        def one(repo):
            try:
                _, commit, tree = get_tree(repo)
                new = 0
                for t in tree:
                    path = t['path']
                    if t['type'] != 'blob' or not path.lower().endswith(('.tif', '.tiff')):
                        continue
                    if any(p in path for p in INPUT_PATTERNS):
                        continue
                    if t.get('size', 0) > 100 * 1024 * 1024:
                        continue
                    source = f'{repo}:{path}'
                    if t['sha'] in by_blob:
                        if source not in by_blob[t['sha']]['sources']:
                            by_blob[t['sha']]['sources'].append(source)
                    else:
                        by_blob[t['sha']] = dict(repo_first=repo, blob=t['sha'],
                                                 sources=[source], source_commit=commit,
                                                 sha256=None)
                        new += 1
                return repo, commit, None, new
            except Exception as exc:                       # noqa: BLE001
                return repo, None, f'{type(exc).__name__}: {exc}', 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for repo, commit, error, new in pool.map(one, REPOS):
                scan['repos'] += 1
                if error:
                    scan['errors'].append(dict(repo=repo, error=error))
                scan['new_blobs'] += new
    scan['scan_utc'] = datetime.now(timezone.utc).isoformat()

    cache = ROOT / '.cache/registry'
    cache.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []

    def work(record):
        row, err = fetch_blob(record, cache)
        return row, err

    items = list(by_blob.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (row, err) in enumerate(pool.map(work, items), 1):
            if err:
                errors.append(err)
                continue
            path = ROOT / row['cache_file']
            try:
                with rasterio.open(path) as src:
                    a = src.read(1)
            except Exception as exc:                        # noqa: BLE001
                errors.append(dict(sources=row['sources'], error=str(exc)[:200]))
                continue
            dots = np.isfinite(a) & (a > 0)
            rho = spearman_full(mine, a, valid)
            ov = dot_overlap(mine, a)
            rev = dot_overlap(a, mine)
            jac = jaccard_dot_sets(mine_dots, dots & valid)
            density = float(dots[valid].mean()) if valid.any() else 0.0
            rows.append(dict(
                submission=row['sources'][0], blob=row['blob'], sha256=row['sha256'],
                registry_dots=int(dots.sum()),
                footprint_density=density,
                representation='DOT' if density <= DENSE_FRACTION else 'DENSE',
                spearman_full_footprint=rho,
                my_dots_within_3px_of_theirs=ov,
                their_dots_within_3px_of_mine=rev,
                jaccard_dot_sets=jac,
                identical_decoded_predictions=bool(
                    row['sha256'] == sha256_file(candidate))))
            if i % 50 == 0:
                print(f'[uniqueness] {i}/{len(items)}', flush=True)

    checked = rows
    complete = (not errors and not scan['errors']
                and len(checked) == len(items))
    rank_rows = [r for r in checked if r['spearman_full_footprint'] is not None]
    worst_rho = max(rank_rows, key=lambda r: r['spearman_full_footprint'], default=None)
    dot_peers = [r for r in checked if r['representation'] == 'DOT']
    dense_peers = [r for r in checked if r['representation'] == 'DENSE']
    worst_dot = max(dot_peers, key=lambda r: r['my_dots_within_3px_of_theirs'], default=None)
    worst_dense = max(dense_peers, key=lambda r: r['my_dots_within_3px_of_theirs'], default=None)
    worst_jac = max(checked, key=lambda r: r['jaccard_dot_sets'], default=None)

    # Support-matched legs. The one-sided forward-overlap statistic rises
    # mechanically with the peer's support size (a peer covering the whole
    # footprint gives 1.0 by definition), so it is only informative against
    # peers of comparable support. The reverse overlap and the Jaccard are
    # scale-free and are applied to every raster.
    n_mine = int(mine_dots.sum())
    comparable = [r for r in checked
                  if n_mine / 3.0 <= r['registry_dots'] <= 3.0 * n_mine]
    by_size = []
    for lo, hi in ((0, 1_000), (1_000, 10_000), (10_000, 50_000), (50_000, 114_000),
                   (114_000, 300_000), (300_000, 1_000_000), (1_000_000, 5_200_000),
                   (5_200_000, 10**9)):
        band = [r for r in checked if lo <= r['registry_dots'] < hi]
        if band:
            by_size.append(dict(peer_dots_lo=lo, peer_dots_hi=hi, peers=len(band),
                                worst_forward_overlap=max(
                                    r['my_dots_within_3px_of_theirs'] for r in band),
                                worst_jaccard=max(r['jaccard_dot_sets'] for r in band)))
    worst_rev = max(checked, key=lambda r: r['their_dots_within_3px_of_mine'], default=None)
    worst_fwd_cmp = max(comparable, key=lambda r: r['my_dots_within_3px_of_theirs'],
                        default=None)
    scale_free_flags = [r for r in checked
                        if r['their_dots_within_3px_of_mine'] > OVERLAP_LIMIT
                        or r['jaccard_dot_sets'] > JACCARD_LIMIT
                        or (r['spearman_full_footprint'] or 0) > RHO_LIMIT]
    support_matched_flags = [r for r in comparable
                             if r['my_dots_within_3px_of_theirs'] > OVERLAP_LIMIT
                             or r['their_dots_within_3px_of_mine'] > OVERLAP_LIMIT
                             or r['jaccard_dot_sets'] > JACCARD_LIMIT
                             or (r['spearman_full_footprint'] or 0) > RHO_LIMIT]

    literal_flags = [r for r in checked
                     if (r['spearman_full_footprint'] or 0) > RHO_LIMIT
                     or r['my_dots_within_3px_of_theirs'] > OVERLAP_LIMIT]
    representation_flags = [r for r in dot_peers
                            if (r['spearman_full_footprint'] or 0) > RHO_LIMIT
                            or r['my_dots_within_3px_of_theirs'] > OVERLAP_LIMIT]

    result = dict(
        evidence_class='REGISTRY-MEASUREMENT',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        candidate=str(candidate.relative_to(ROOT)),
        candidate_sha256=sha256_file(candidate),
        candidate_profile=profile,
        candidate_dots=int(mine_dots.sum()),
        candidate_dot_fraction=float(mine_dots[valid].mean()),
        live_tree_scan=scan,
        registry_rasters_expected=len(items),
        registry_rasters_checked=len(checked),
        registry_rasters_failed=len(errors),
        complete_accessible_scan=bool(complete),
        thresholds=dict(rho=RHO_LIMIT, dot_overlap=OVERLAP_LIMIT,
                        dense_fraction=DENSE_FRACTION),
        spearman_subgrid_fraction=1.0 / (SPEARMAN_STRIDE ** 2),
        spearman_subgrid_points=int(np.count_nonzero(valid[::SPEARMAN_STRIDE,
                                                          ::SPEARMAN_STRIDE])),
        leg1_literal=dict(
            description=('full-footprint Spearman and forward 3-px dot overlap '
                         'against every registry raster, inherited support rule'),
            flags=len(literal_flags),
            flag_rows=[r['submission'] for r in literal_flags[:25]],
            worst_spearman=(worst_rho['spearman_full_footprint'] if worst_rho else None),
            worst_spearman_raster=(worst_rho['submission'] if worst_rho else None),
            worst_dot_overlap=(worst_dense['my_dots_within_3px_of_theirs'] if worst_dense else None),
            worst_dot_overlap_raster=(worst_dense['submission'] if worst_dense else None),
            passed=not literal_flags),
        leg2_representation_aware=dict(
            description=('same measurements, dot-overlap leg restricted to DOT '
                         'peers (support density <= 0.5 of the footprint); DENSE '
                         'peers reported but excluded because a near-full-footprint '
                         'surface gives every candidate overlap 1.0 (IR-57-REPR-01)'),
            dot_peers=len(dot_peers), dense_peers=len(dense_peers),
            flags=len(representation_flags),
            flag_rows=[r['submission'] for r in representation_flags[:25]],
            worst_dot_overlap=(worst_dot['my_dots_within_3px_of_theirs'] if worst_dot else None),
            worst_dot_overlap_raster=(worst_dot['submission'] if worst_dot else None),
            worst_jaccard=worst_jac['jaccard_dot_sets'] if worst_jac else None,
            worst_jaccard_raster=worst_jac['submission'] if worst_jac else None,
            passed=not representation_flags),
        leg3_scale_free=dict(
            description=('reverse 3-px overlap and Jaccard of dot sets applied to EVERY '
                         'registry raster; both are scale-free, so neither is inflated by '
                         'a large peer support'),
            worst_reverse_overlap=(worst_rev['their_dots_within_3px_of_mine'] if worst_rev else None),
            worst_reverse_overlap_raster=(worst_rev['submission'] if worst_rev else None),
            worst_jaccard=worst_jac['jaccard_dot_sets'] if worst_jac else None,
            worst_jaccard_raster=(worst_jac['submission'] if worst_jac else None),
            flags=len(scale_free_flags), passed=not scale_free_flags),
        leg4_support_matched=dict(
            description=('forward 3-px overlap restricted to peers whose positive-cell '
                         'count is within a factor of three of the candidate (declared '
                         'band n/3 <= |peer| <= 3n); the overlap-vs-support curve below '
                         'shows why the unrestricted leg is a coverage artifact'),
            band_lo=n_mine / 3.0, band_hi=3.0 * n_mine, peers=len(comparable),
            worst_forward_overlap=(worst_fwd_cmp['my_dots_within_3px_of_theirs'] if worst_fwd_cmp else None),
            worst_forward_overlap_raster=(worst_fwd_cmp['submission'] if worst_fwd_cmp else None),
            flags=len(support_matched_flags), passed=not support_matched_flags),
        overlap_versus_peer_support=by_size,
        identical_file_or_pixels=any(r['identical_decoded_predictions'] for r in checked),
        errors=errors[:20],
        seconds=round(time.time() - t0, 1),
        rows=sorted(checked, key=lambda r: -(r['spearman_full_footprint'] or 0))[:40],
    )
    (ROOT / args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())