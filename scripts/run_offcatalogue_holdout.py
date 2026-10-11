#!/usr/bin/env python3
"""Session 7 experiment driver for the fault-zone anatomy lane.

Experiments (the declared session budget is three):

E4  OFF-CATALOGUE INSTRUMENT + FITTED DAMAGE ZONE
    Build the off-catalogue proxy-truth instrument, measure the lane's
    structure (distance, relative strike, host length, recorded sense) and
    fit the per-band new-fault rate by leave-one-quadrant-out.

E5  DOT BUDGET ON THE OFF-CATALOGUE INSTRUMENT
    Sweep the dot budget with the shared exact greedy allocator and score
    every arm with the shared pooled DTI (alpha 0.2, beta 0.8, 300 m kernel).

E6  STRUCTURE ABLATION (distance only / +length / +sense / wider zone)
    Keep only the structure the instrument shows; report the rest as negative.

Everything is labelled HOLDOUT-* or EXTERNAL-AUXILIARY-*. Nothing here is a
competition score, and nothing here submits anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gems59 import load_grid, metric                          # noqa: E402
from gems59 import offcatalogue as oc                          # noqa: E402
from gems59.emit import greedy_allocate                        # noqa: E402
from gems59.evaluate_holdout import VERSION as EVAL_VERSION     # noqa: E402
from gems59.evaluate_holdout import implementation_hashes      # noqa: E402
from gems59.metric import ALPHA, BETA, R_M                     # noqa: E402

EVID = ROOT / 'evidence'
BUDGETS = (8_000, 12_000, 16_000, 20_000, 24_000, 28_000, 32_000, 38_000, 46_000, 60_000)
LENGTH_CENTRES = np.array([0.5, np.log1p(50), np.log1p(200), np.log1p(600),
                           np.log1p(1500), np.log1p(4000)])


def build_intensity(cov, profile, *, length_rows=None, sense_rows=None, d_max=None):
    """Per-cell new-fault probability: fitted distance profile x optional terms."""
    p = oc.profile_field(profile, cov).astype(np.float64)
    if d_max is not None:
        p[cov.d > d_max] = 0.0
    if length_rows:
        rate = {r['bin']: r['rate_ratio_to_all'] for r in length_rows}
        if rate:
            values = np.clip([rate.get(i, 1.0) for i in range(LENGTH_CENTRES.size)],
                             0.0, 8.0)
            p = p * np.interp(np.clip(cov.log_len, LENGTH_CENTRES[0], LENGTH_CENTRES[-1]),
                              LENGTH_CENTRES, values)
    if sense_rows:
        rate = {r['sense']: r['rate_ratio_to_all'] for r in sense_rows}
        if rate:
            lookup = np.ones(4, np.float64)
            lookup[1] = rate.get('N', 1.0)
            lookup[2] = rate.get('RL', 1.0)
            lookup[3] = rate.get('LL', 1.0)
            p = p * lookup[np.clip(cov.sense, 0, 3)]
    return np.clip(p, 0.0, 1.0)


def _merge_profiles(profiles):
    area = np.sum([np.asarray(p['band_area'], float) for p in profiles], axis=0)
    pos = np.sum([np.asarray(p['band_positive'], float) for p in profiles], axis=0)
    return dict(edges=profiles[0]['edges'], band_area=area.tolist(),
                band_positive=pos.tolist(),
                band_rate=(pos / np.maximum(area, 1.0)).tolist(),
                n_positive=float(pos.sum()), n_region=float(area.sum()))


def _merge_rows(domains, others, cov, fn, key):
    """Pool one structure table over the training quadrants."""
    pooled = {}
    for name in others:
        for r in fn(domains[name]['region'], domains[name]['truth'], cov)['rows']:
            slot = pooled.setdefault(r[key], [0.0, 0.0])
            slot[0] += float(r['area'])
            slot[1] += float(r['positive'])
    total_area = sum(v[0] for v in pooled.values())
    total_pos = sum(v[1] for v in pooled.values())
    base = total_pos / max(total_area, 1.0)
    return [dict({key: k}, area=int(v[0]), positive=int(v[1]),
                 rate=float(v[1] / max(v[0], 1.0)),
                 rate_ratio_to_all=float((v[1] / max(v[0], 1.0)) / max(base, 1e-12)))
            for k, v in sorted(pooled.items(), key=lambda kv: str(kv[0]))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', default='all',
                    choices=('all', 'structure', 'budget'))
    ap.add_argument('--seed', type=int, default=57)
    ap.add_argument('--budgets', default='', help='comma separated override')
    args = ap.parse_args()
    budgets = (tuple(int(v) for v in args.budgets.split(',')) if args.budgets
               else BUDGETS)
    t0 = time.time()
    EVID.mkdir(exist_ok=True)

    grid = load_grid()
    proxy, proxy_receipt = oc.load_proxy_truth(ROOT, grid)
    sense, sense_receipt = oc.load_sense(ROOT, grid)
    cov = oc.catalogue_covariates(grid, sense)
    domains = oc.fold_domains(grid, proxy)

    base = dict(
        instrument=oc.VERSION, evaluator=EVAL_VERSION,
        evaluator_implementation=implementation_hashes(),
        metric=dict(alpha=ALPHA, beta=BETA, kernel_radius_m=R_M,
                    kernel='triangular', pixel_m=100.0),
        proxy_truth=proxy_receipt, sense_source=sense_receipt,
        grid=dict(shape=list(grid.shape), crs='EPSG:32611',
                  transform=[float(v) for v in tuple(grid.transform)[:6]],
                  footprint_px=int(grid.footprint.sum()),
                  catalogue_px=int(grid.catalogue.sum())),
        folds=list(domains), seed=args.seed,
        proxy_pixels_total=int(proxy.sum()),
        evidence_class=('HOLDOUT-DTI on an off-catalogue proxy-truth instrument; '
                        'NOT a competition score and NOT calibrated to a live score'))

    profiles = {name: oc.distance_profile(d['region'], d['truth'], cov)
                for name, d in domains.items()}
    profiles['all'] = oc.distance_profile(grid.footprint, proxy, cov)

    if args.stage in ('all', 'structure'):
        structure = dict(base)
        structure['distance_profile'] = profiles
        structure['relative_strike'] = oc.relative_strike_structure(grid, proxy, cov)
        structure['sense_by_host'] = {n: oc.sense_structure(d['region'], d['truth'], cov)
                                      for n, d in domains.items()}
        structure['length_by_host'] = {n: oc.length_structure(d['region'], d['truth'], cov)
                                       for n, d in domains.items()}
        (EVID / 'offcat_structure.json').write_text(
            json.dumps(structure, indent=2, allow_nan=False) + '\n')
        print('[E4] wrote evidence/offcat_structure.json', flush=True)

    if args.stage in ('all', 'budget'):
        arms = {
            'distance_only': dict(length=False, sense=False, d_max=95.0),
            'distance_len': dict(length=True, sense=False, d_max=95.0),
            'distance_len_sense': dict(length=True, sense=True, d_max=95.0),
            'distance_len_sense_wide': dict(length=True, sense=True, d_max=140.0),
        }
        rows = []
        for name, d in domains.items():
            others = [n for n in domains if n != name]
            merged = _merge_profiles([profiles[n] for n in others])
            length_rows = _merge_rows(domains, others, cov, oc.length_structure, 'bin')
            sense_rows = _merge_rows(domains, others, cov, oc.sense_structure, 'sense')
            for arm, cfg in arms.items():
                surface = build_intensity(cov, merged, length_rows=length_rows,
                                          sense_rows=sense_rows, d_max=cfg['d_max'])
                for n in budgets:
                    alloc = greedy_allocate(surface.astype(np.float32), d['scored'],
                                            float(d['truth'].sum()), max_dots=n,
                                            candidate_cap=250_000, per_round=20_000)
                    r = metric.dti_exact(alloc.emitted.astype(np.float64), d['truth'],
                                         valid=d['scored'])
                    rows.append(dict(fold=name, arm=arm, n_dots=n,
                                     emitted=int(alloc.emitted.sum()),
                                     dti=float(r['dti']), tp=float(r['tp']),
                                     fp=float(r['fp']), fn=float(r['fn']),
                                     n_truth=int(r['n_truth']),
                                     coverage=float(r['coverage'])))
                    print(f'[E5] fold={name} arm={arm:26s} n={n:6d} '
                          f'emitted={int(alloc.emitted.sum()):6d} '
                          f'DTI={r["dti"]:.6f} cov={r["coverage"]:.4f}', flush=True)
        sweep = dict(base, rows=rows, budgets=list(budgets),
                     allocator='gems59.emit.greedy_allocate (exact two-sided marginal test)',
                     arms={k: dict(v) for k, v in arms.items()})
        (EVID / 'offcat_budget_sweep.json').write_text(
            json.dumps(sweep, indent=2, allow_nan=False) + '\n')
        print('[E5] wrote evidence/offcat_budget_sweep.json', flush=True)

    print(f'[session7] done in {time.time() - t0:.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())