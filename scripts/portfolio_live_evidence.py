#!/usr/bin/env python3
"""Portfolio-level evidence: which properties of a published dot field track its score.

For every registry raster that carries an owner-reported live score, measure a
declared set of placement properties and report the Spearman rank correlation of
each with that score, plus the relative-strike histogram of the dot field
against its nearest mapped fault.

The live scores are OWNER-REPORTED labels copied from the task brief. They are
not organizer receipts. n = 15 and the properties are strongly collinear, so
this is evidence for a design choice, never a calibrated forecast of a score.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gems59 import load_grid                              # noqa: E402
from gems59 import offcatalogue as oc                     # noqa: E402
from gems59.network import local_strike                   # noqa: E402

ANG_EDGES = np.linspace(0.0, 90.0, 19)


def properties(dots, grid, host_strike, host_coh, dcat, proxy):
    n = int(dots.sum())
    ys, xs = np.nonzero(dots)
    d = dcat[ys, xs]
    tree = cKDTree(np.stack([ys, xs], 1))
    nnd, _ = tree.query(np.stack([ys, xs], 1), k=2)
    quarter = (ys >= grid.height // 2).astype(int) * 2 + (xs >= grid.width // 2)
    counts = np.bincount(quarter, minlength=4) / max(n, 1)
    return {
        'n_dots': n,
        'log_n_dots': float(np.log(n)),
        'on_catalogue_fraction': float(grid.catalogue[ys, xs].mean()),
        'frac_within_1px_of_catalogue': float((d <= 1).mean()),
        'frac_within_2px_of_catalogue': float((d <= 2).mean()),
        'frac_within_3px_of_catalogue': float((d <= 3).mean()),
        'distance_p25_px': float(np.percentile(d, 25)),
        'distance_p50_px': float(np.percentile(d, 50)),
        'distance_p75_px': float(np.percentile(d, 75)),
        'distance_p90_px': float(np.percentile(d, 90)),
        'mean_distance_px': float(d.mean()),
        'nearest_dot_spacing_mean_px': float(nnd[:, 1].mean()),
        'quadrant_spread_entropy': float(
            -(counts * np.log(counts + 1e-12)).sum() / np.log(4)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='evidence/portfolio_live_evidence.json')
    args = ap.parse_args()
    grid = load_grid()
    cat = grid.catalogue
    dcat, (iy, ix) = ndi.distance_transform_edt(~cat, return_indices=True)
    strike, coh = local_strike(cat, smooth_px=3.0)
    host_strike = strike[iy, ix]
    host_coh = coh[iy, ix]
    proxy, _ = oc.load_proxy_truth(ROOT, grid)

    rows = []
    for entry in json.loads((ROOT / 'registry/registry_index.json').read_text()):
        with rasterio.open(ROOT / entry['file']) as src:
            a = src.read(1)
        dots = np.isfinite(a) & (a > 0)
        if not dots.any():
            continue
        props = properties(dots, grid, host_strike, host_coh, dcat, proxy)
        ang_strike, ang_coh = local_strike(dots, smooth_px=5.0)
        ys, xs = np.nonzero(dots)
        keep = (ang_coh[ys, xs] > 0.2) & (host_coh[ys, xs] > 0.2)
        if keep.any():
            ang = oc.fold_angle(ang_strike[ys[keep], xs[keep]],
                                host_strike[ys[keep], xs[keep]])
        else:
            ang = np.zeros(0)
        hist, _ = np.histogram(ang, bins=ANG_EDGES)
        rows.append(dict(submission=entry['submission'], repo=entry['repo'],
                         owner_reported_score=entry.get('owner_reported_score'),
                         relative_strike_histogram=hist.astype(float).tolist(),
                         relative_strike_pixels=int(ang.size), **props))

    scored = [r for r in rows if r['owner_reported_score'] is not None]
    live = [r['owner_reported_score'] for r in scored]
    corr = {}
    for key in scored[0]:
        if key.startswith(('owner_reported', 'relative_strike', 'submission', 'repo')):
            continue
        values = [r[key] for r in scored]
        if np.std(values) == 0:
            continue
        rho, pval = spearmanr(live, values)
        corr[key] = dict(spearman=float(rho), p_value=float(pval))
    bins = ANG_EDGES[:-1]
    ang_corr = []
    for i, lo in enumerate(bins):
        values = [r['relative_strike_histogram'][i] / max(sum(r['relative_strike_histogram']), 1)
                  for r in scored]
        rho, pval = spearmanr(live, values)
        ang_corr.append(dict(bin_lo_deg=float(lo), bin_hi_deg=float(ANG_EDGES[i + 1]),
                             spearman=float(rho), p_value=float(pval)))

    result = dict(
        evidence_class='REGISTRY-MEASUREMENT (portfolio labels are OWNER-REPORTED)',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scored_rasters=len(scored), unlabelled_rasters=len(rows) - len(scored),
        label_provenance=('live scores are owner-reported values from the task brief, '
                          'not organizer submission receipts'),
        caveats=('n = 15; dot count, distance profile and sub-parallel fraction are '
                 'strongly collinear, so these correlations are evidence for a design '
                 'choice and not a calibrated score model'),
        property_correlations=corr,
        relative_strike_correlations=ang_corr,
        relative_strike_bin_edges_deg=ANG_EDGES.tolist(),
        rows=rows)
    (ROOT / args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('[portfolio] wrote', args.out)
    for key in ('frac_within_2px_of_catalogue', 'distance_p50_px', 'log_n_dots'):
        print(f"  {key:32s} rho={corr[key]['spearman']:+.3f} p={corr[key]['p_value']:.4f}")
    for row in ang_corr[:6]:
        print(f"  angle {row['bin_lo_deg']:.0f}-{row['bin_hi_deg']:.0f} deg  "
              f"rho={row['spearman']:+.3f} p={row['p_value']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
