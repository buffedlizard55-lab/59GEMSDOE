#!/usr/bin/env python3
"""Deployment-matched diagnostic: per-channel AUC + multivariate AUC, segment folds."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from gems59 import features, grid, holdout, intensity

def auc(score, pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    p = score[pos]; p = p[np.isfinite(p)]; n = score[neg]; n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0: return float('nan')
    if p.size > 20000: p = rng.choice(p, 20000, replace=False)
    if n.size > 300000: n = rng.choice(n, 300000, replace=False)
    a = np.concatenate([p, n]); o = np.argsort(a, kind='stable'); s = a[o]
    tied = np.empty(a.size); i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j+1] == s[i]: j += 1
        tied[o[i:j+1]] = 0.5*(i+j)+1.0; i = j+1
    rp = tied[:p.size].sum()
    return float((rp - p.size*(p.size+1)/2.0)/(p.size*n.size))

t0=time.time()
stack, fp = features.build_stack()[0], grid.read_footprint()
cat = grid.read_labels() & fp
lab, nseg = holdout.segments(cat)
blocks = holdout.assign_blocks_random(lab, 4, seed=7)
w, vis, ta = holdout.fold_masks(lab, blocks, 0, fp, buffer_px=3)
features.set_visible_channels(stack, cat & ~w, fp)
sup = holdout.emission_support(fp, cat & ~w, flank_px=2)
pos = w & sup; neg = sup & ~w
print(f'segments {nseg}  withheld {w.sum()}  in-support pos {pos.sum()}  neg {neg.sum()}  ({time.time()-t0:.0f}s)')
rows = []
for i, nm in enumerate(features.CHANNELS):
    a = auc(stack[i], pos, neg); rows.append((nm, None if not np.isfinite(a) else round(max(a,1-a),4)))
rows.sort(key=lambda r: -(r[1] or 0))
print('--- per-channel max(AUC,1-AUC), deployment-matched fold 0 ---')
for nm, v in rows: print(f'  {v}  {nm}')
json.dump({'per_channel': dict(rows), 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum())},
          open(ROOT/'evidence'/'diag2_channels.json','w'), indent=1)

# multivariate: geophysics-only vs all
rng = np.random.default_rng(0)
for label, chans in (('geo_only', [c for c in features.CHANNELS if c not in ('dist_lidar','mapscale_field','trace_density','therm_dens_all')]),
                     ('all', list(features.CHANNELS)),
                     ('no_cat', [c for c in features.CHANNELS if c not in ('dist_visible_fault','visible_fault_dens_5k','visible_fault_dens_20k')])):
    idx = [features.CHANNELS.index(c) for c in chans]
    X, y, _ = intensity.sample_design(stack, cat & ~w & ta, ta, idx, intensity.PPSpec(n_background=200000, max_iter=120), rng)
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31,
                                       min_samples_leaf=40, class_weight='balanced', random_state=0).fit(X, y)
    # predict on the eval area in chunks
    H, W = fp.shape; out = np.zeros((H, W), np.float32)
    for r0 in range(0, H, 64):
        r1 = min(H, r0+64)
        blk = np.stack([stack[i, r0:r1] for i in idx])
        out[r0:r1] = m.predict_proba(blk.reshape(len(idx), -1).T)[:,1].reshape(r1-r0, W)
    a = auc(out, pos, neg)
    print(f'{label:10s} channels={len(idx):3d}  holdout AUC = {a:.4f}   ({time.time()-t0:.0f}s)')
    np.save(ROOT/f'work/field_{label}_fold0.npy', out)
