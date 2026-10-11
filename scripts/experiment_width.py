#!/usr/bin/env python3
"""One preregistered H57-G experiment: visible-only length-scaled stepover.

Research artifact only: if the surface duplicate gate fires, stop BEFORE dot
placement. Do not submit the resulting continuous GeoTIFF to the competition.
Every training target is a withheld *whole catalogue segment*; fold feature
geometry is rebuilt without that segment, and fits exclude the test quadrant.
"""
from __future__ import annotations
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import subprocess
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gems59 import load_grid, write_submission
from gems59.anatomy import FEATURES, fold_geometry
from gems59.fitting import canary, cell_geometry, fit_model, pooled, run_cell
from gems59.holdout import build_holdout, FOLD_NAMES
from gems59.validate import validate
from gems59.gates import _disk, registry_coverage, lane_uniqueness_report
from gems59 import evaluate_holdout
import rasterio

CAP = 10_000
FLOOR = 0.015
BASE_COLS = [i for i, n in enumerate(FEATURES) if n != 'side']
WIDTH_COLS = BASE_COLS + [len(FEATURES)]
LATTICE = Path('/tmp/gems13-review/docs/downloads/13gems_20261001_r13-lattice-s5_v2_zerofill.tif')
CONTINUOUS = Path('/tmp/gems17-review/docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif')


def add_width(g):
    """No hidden pixels enter: both columns come from the nearest visible trace."""
    d_perp = g.X[:, FEATURES.index('d_perp')]
    log_length = g.X[:, FEATURES.index('log_len')]
    # log_len = log(1+component pixel count), so sqrt(1+L) = exp(log_len/2).
    width = (d_perp / np.exp(0.5 * log_length)).astype(np.float32)
    g.X = np.column_stack((g.X, width))
    return g


def restore_registry_witnesses():
    for repo, path in (('13GEMSDOE', LATTICE), ('17GEMSDOE', CONTINUOUS)):
        if path.exists():
            continue
        subprocess.run(['git', 'clone', '-q', '--depth', '1', '--filter=blob:limit=1m',
                        f'https://github.com/buffedlizard55-lab/{repo}.git',
                        str(path.parents[2])], check=True, timeout=600)
        if not path.exists():
            raise FileNotFoundError(f'registry witness missing after clone: {path}')


def main():
    start = time.monotonic()
    grid = load_grid()
    # Detached segments avoid the known all-mode proximity canary (>0.90 on
    # distance), recorded in exp_sense_loqo_all.json. The detached instrument
    # had every feature below 0.90 in cv_detached.json.
    ctx = build_holdout(grid, modes=('detached',))
    cells = ctx.cells_of('detached')
    first = [add_width(cell_geometry(ctx, c)) for c in cells if c.seed == 20]
    all_canary = canary(first, feature_names=FEATURES + ('length_scaled_stepover',))
    base_canary = {n: all_canary[n] for n in FEATURES}
    new_canary = all_canary['length_scaled_stepover']
    first.clear(); gc.collect()
    canary_clean = not new_canary['leakage_flag'] and not any(v['leakage_flag'] for v in base_canary.values())
    if not canary_clean:
        print('CANARY FLAGGED', {n:r['discriminative_auc_max'] for n,r in base_canary.items()}, new_canary, flush=True)
        raise RuntimeError('leakage canary flagged; refusing training and raster build')
    print('canary clean; width AUC', new_canary, flush=True)
    scores = {'baseline': [], 'H57-G': []}
    for qi, q in enumerate(FOLD_NAMES):
        test = [c for c in cells if c.key.split('_')[1] == f'fold{q}']
        train = [c for c in cells if c.key.split('_')[1] != f'fold{q}']
        tr = [add_width(cell_geometry(ctx, c)) for c in train]
        te = {c.key: add_width(cell_geometry(ctx, c)) for c in test}
        for name, cols in (('baseline', BASE_COLS), ('H57-G', WIDTH_COLS)):
            clf, scale, _ = fit_model(tr, seed=qi, cols=cols)
            for c in test:
                scores[name].append(run_cell(ctx, c, clf, scale, cols=cols,
                                             max_dots=CAP, floor=FLOOR, g=te[c.key],
                                             shared_evaluator=True))
            del clf
        tr.clear(); te.clear(); gc.collect()
        print('finished quadrant', q, 'at seconds', round(time.monotonic()-start, 1), flush=True)
    b, h = pooled(scores['baseline']), pooled(scores['H57-G'])
    paired = {q: pooled([r for r in scores['H57-G'] if r['key'].split('_')[1] == f'fold{q}'])['pooled_dti'] -
                 pooled([r for r in scores['baseline'] if r['key'].split('_')[1] == f'fold{q}'])['pooled_dti']
              for q in FOLD_NAMES}
    report = {'hypothesis':'H57-G: sqrt(length)-normalized cross-strike stepover',
              'mechanism':'Damage-zone width may grow with fault displacement; connected visible-component length is only a noisy displacement proxy.',
              'non_fault_mimic':'Map-trace fragmentation and mapping effort, not actual displacement.',
              'evidence_class':'HOLDOUT-DTI (not a live score)',
              'evaluator_version':evaluate_holdout.VERSION + ' repaired (metric.max_cover, pixel-exact mask), gems59 LOQO pooled/quadrant-jackknife',
              'shared_evaluator_hashes':evaluate_holdout.implementation_hashes(),
              'withheld_positive_pixels': h['n_truth'], 'canary': {'base': base_canary, 'width':new_canary},
              'baseline':b, 'H57-G':h, 'paired_delta':h['pooled_dti']-b['pooled_dti'],
              'paired_delta_per_quadrant':paired,
              'passes_holdout_bar':h['pooled_dti'] > b['pooled_dti'],
              'bar_note':'Same detached-mode, same-budget LOQO baseline; the earlier all-mode .2279 is not a comparable bar.',
              'runtime_s_before_surface': round(time.monotonic()-start,1)}
    # Generate independent surface from visible full catalogue, not from an
    # earlier submission. This is an opt-in length/stepover model and never
    # uses any previous prediction as an input.
    tr = [add_width(cell_geometry(ctx,c)) for c in cells]
    clf, scale, _ = fit_model(tr, seed=57, cols=WIDTH_COLS)
    tr.clear(); gc.collect()
    domain = grid.footprint & ~grid.catalogue
    geom = add_width(fold_geometry(grid, grid.catalogue, np.zeros(grid.shape,bool),
                                   domain, 'live_full_visible'))
    p = np.zeros(grid.shape, np.float32)
    p[geom.rows, geom.cols] = np.clip(clf.predict_proba(geom.X[:,WIDTH_COLS])[:,1] * scale, 0, 1).astype(np.float32)
    del geom, clf; gc.collect()
    # Check the *surface* against the two authenticated registry witnesses
    # BEFORE placing dots. Their >0 support is the literal overlap convention
    # of src/gems59/uniqueness.py. A positive continuous prior over the whole
    # footprint is a universal overlap witness, not a useful geological match.
    restore_registry_witnesses()
    index = json.loads((ROOT/'evidence'/'registry_full_index.json').read_text())
    registered_hashes = {r['sha256'] for r in index['rasters']}
    checks = []
    for prior in (LATTICE, CONTINUOUS):
        if not prior.exists():
            raise FileNotFoundError(f'restore registry witness first: {prior}')
        with rasterio.open(prior) as ds:
            a = ds.read(1)
            assert ds.shape == grid.shape and ds.crs == grid.crs and ds.transform == grid.transform
        prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
        if prior_hash not in registered_hashes:
            raise ValueError(f'prior hash not pinned in registry: {prior}')
        proposal = np.isfinite(a) & (a > 0)
        halo = binary_dilation(proposal, structure=_disk(3.))
        checks.append({'source':str(prior), 'sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
                       'prior_coverage_of_eligible_3px':registry_coverage(proposal, domain),
                       'candidate_surface_positive_3px_overlap':float((halo & (p>0)).sum()/max((p>0).sum(),1)),
                       'prior_nonzero':int(proposal.sum())})
    rank_gate = lane_uniqueness_report(p, domain, (LATTICE, CONTINUOUS),
                      sample=ROOT/'data'/'official'/'sample_submission.tif', phase='surface')
    if not rank_gate['ok']:
        raise RuntimeError('surface rank check failed or registry witness unreadable')
    # This is a newly inferred, on-disk, finite GeoTIFF for research/download.
    # It is NOT an emission raster because literal support-overlap rule fires.
    digest = hashlib.sha256(p.tobytes()).hexdigest()[:12]
    path = ROOT/'docs'/'downloads'/f'gems59-h57g-width-normalized-{digest}-RESEARCH-DO-NOT-SUBMIT.tif'
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1 or (p[~domain] != 0).any():
        raise ValueError('fail-closed research surface range/footprint check')
    write_submission(path, p, mode='zeros')
    v = validate(path, grid.footprint, grid.catalogue)
    raster_hash = v['sha256']
    report.update({'surface_witnesses':checks,
                   'surface_rank_vs_witnesses':[{'source':r['path'], 'spearman':r['spearman']} for r in rank_gate['per_prior']],
                   'sha256_distinct_from_all_644_indexed_rasters':raster_hash not in registered_hashes,
                   'index_scope_note':'Full sha256 index only; two pinned overlap witnesses suffice to fail the literal gate.', 'surface_gate_passes':all(x['candidate_surface_positive_3px_overlap'] <= .70 for x in checks),
                   'dot_placement':'STOPPED before placement; literal surface-overlap gate fired',
                   'research_raster':{'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'validator':v}, 'submission_name':path.stem,
                   'submission_note':'H57-G width-scaled stepover; research only: strict overlap gate failed; DO NOT SUBMIT',
                   'verdict':'negative; DO NOT SUBMIT (not a unique lane submission by literal gate)',
                   'runtime_s':round(time.monotonic()-start,1)})
    out = ROOT/'evidence'/'exp4_width.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('wrote',out, 'surface gate',report['surface_gate_passes'], 'HOLDOUT-DTI', h['pooled_dti'], h['dti_ci95_quadrant_jackknife'],flush=True)

if __name__ == "__main__":
    raise SystemExit("Archived concurrent-session generator: its geometry predates IR-57-STRIKE-01 "
                     "and/or its validation does not satisfy the current buffered protocol or public-inventory checks. "
                     "No new generation or slot is authorized. See README.md and run_card_current.json.")
