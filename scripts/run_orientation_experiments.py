#!/usr/bin/env python3
"""Three predeclared experiments; spatial LOQO; no competition-slot selection.

E1: shipped-density distance-only vs existing anatomy (repaired buffered folds).
E2: add candidate magnetic-edge strike RELATIVE to the visible host.
E3: recorded-sense ablation on E2. Same folds, fixed density, shared evaluator.

All catalogue features share a globally hidden, whole-component draw with a
300m feature-context buffer. Every feature gets a single-feature AUC canary.
The final scientific surface may be written as a held research TIFF to audit;
no final dot placement is allowed if the pre-placement registry check fails.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Avoid sandbox-wide OpenMP oversubscription even if invoked without env flags.
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from gems59 import load_grid
from gems59.anatomy import (FEATURES, SENSE_FEATURES, package_holdout_structure,
                            relative_strike_distribution)
from gems59.faultzone import trace_sense_raster
from gems59.fitting import canary, fit_model
from gems59.holdout import buffered_component_draw, FOLD_NAMES
from gems59.emit import greedy_allocate
from gems59 import evaluate_holdout as EH
from gems59.strand_orientation import (ALL_FEATURES, MAG_FEATURES, cached_magnetic_geometry, geometry)
from gems59.submission_writer import write_submission
from gems59.uniqueness import compare_to_registry, saturation_certificate
from gems59.validate import validate

COLS={n:i for i,n in enumerate(ALL_FEATURES)}
BASE=[COLS[n] for n in FEATURES if n!='side']
ARMS={'distance_only':[COLS['d']], 'anatomy':BASE,
      'orientation':BASE+[COLS[n] for n in MAG_FEATURES],
      'orientation_sense':BASE+[COLS[n] for n in MAG_FEATURES+SENSE_FEATURES]}


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def subset(g, selected, key):
    return SimpleNamespace(X=g.X[selected],y=g.y[selected],rows=g.rows[selected],
                           cols=g.cols[selected],key=key)


def predict(clf,scale,g,columns,shape,zone_px):
    p=np.zeros(shape,np.float32)
    for start in range(0,len(g.y),150_000):
        sl=slice(start,min(start+150_000,len(g.y)))
        x=g.X[sl]
        v=clf.predict_proba(x[:,columns])[:,1]*scale
        v=np.clip(v,0,1).astype(np.float32)
        v[x[:,0]>zone_px]=0
        p[g.rows[sl],g.cols[sl]]=v
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-quadrant-cap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20)
    ap.add_argument('--minutes',type=float,default=90)
    ap.add_argument('--build-research-surface',action='store_true')
    ap.add_argument('--audit-registry',type=Path,default=ROOT/'evidence/registry_refreshed.json')
    a=ap.parse_args()
    if not 0<a.minutes<=120 or a.per_quadrant_cap<1:
        ap.error('time must be <=120 minutes; density cap must be positive')
    start=time.monotonic();deadline=start+a.minutes*60
    def check_time():
        if time.monotonic()>deadline:
            raise TimeoutError('experiment wall-clock budget reached; no further fitting')
    plan=dict(generated_utc=datetime.now(timezone.utc).isoformat(), maximum_experiments=3,
        experiments=[dict(id='E1',hypothesis='distance-only vs anatomy at fixed density'),
                     dict(id='E2',hypothesis='candidate-edge relative strike inside fitted damage zone'),
                     dict(id='E3',hypothesis='recorded slip sense conditions E2')],
        per_quadrant_cap=a.per_quadrant_cap, seed=a.seed,
        split='buffered-whole-components-loqo-v2', evaluator=EH.VERSION,
        no_textbook_angle=True, submission_slots_used=0,
        candidate_predeclared='orientation', sense_retention_rule='paired CI lower bound > 0 against orientation',
        emission='existing greedy allocation; truth-count estimate from training base rate, never test labels',
        zone='90th percentile of TRAINING withheld-positive distances; cap shrunk by TRAINING fraction inside zone')
    save(ROOT/'evidence/experiment_plan.json',plan)
    print('[run] predeclared three experiments; no weekly slot will be used',flush=True)
    grid=load_grid()
    folds,quad=buffered_component_draw(grid,seed=a.seed)
    visible=folds[0]['visible'];hidden=folds[0]['hidden_all']
    domain=np.logical_or.reduce([f['region'] for f in folds]) & ~folds[0]['masked_known']
    sense=trace_sense_raster(ROOT/'data/external/trace_segments_utm11.csv',grid.shape,grid.transform)
    magnetic=cached_magnetic_geometry(ROOT,grid.footprint)
    full=geometry(grid,visible,hidden,domain,sense,magnetic,ROOT/'.cache/geometry',f'draw{a.seed}-buffered')
    print(f'[run] visible-only geometry ready: {len(full.y)} examples, {int(full.y.sum())} positives',flush=True)
    geoms=[subset(full,quad[full.rows,full.cols]==q,FOLD_NAMES[q]) for q in range(4)]
    c=canary(geoms,feature_names=ALL_FEATURES)
    save(ROOT/'evidence/orientation_canary.json',dict(evidence_class='LEAKAGE-CANARY (AUC, not DTI)',
        split_version=plan['split'], features=c, features_independent_of_hidden_values=True,
        note='All predictors are calculated from visible mask and catalogue-independent cached TMI; hidden geometry is diagnostic only.'))
    print('[run] feature canary maxima:',{n:round(r['discriminative_auc_max'],4) if r['discriminative_auc_max'] is not None else None for n,r in c.items()},flush=True)
    flagged=[n for n,r in c.items() if r['leakage_flag'] or r['discriminative_auc_max'] is None]
    if flagged:
        save(ROOT/'evidence/orientation_holdout.json',dict(evidence_class='HOLDOUT-DTI',evaluator_version=EH.VERSION,
            status='negative: unresolved leakage canary',flagged_features=flagged,
            withheld_positive_pixels=int(full.y.sum()),holdout_dti=None,ci95=None))
        print(f'[run] STOP: unresolved canary {flagged}',flush=True)
        return 3
    # Diagnostic WITHHELD strand orientations never enter the prediction matrix.
    relative=relative_strike_distribution(grid,visible,hidden)
    pos=full.y==1
    structure=package_holdout_structure(
        relative, withheld_positive_pixels=int(pos.sum()),
        distance_positive_quantiles_px=np.quantile(full.X[pos,0],[.1,.5,.9,.95,.99]).tolist(),
        distance_domain_quantiles_px=np.quantile(full.X[:,0],[.1,.5,.9,.95,.99]).tolist())
    save(ROOT/'evidence/orientation_structure.json',structure)
    terms={n:None for n in ARMS};surface_terms={n:None for n in ARMS}
    details=[];fold_models={}
    for q,fold in enumerate(folds):
        train=[g for i,g in enumerate(geoms) if i!=q]
        distances=np.concatenate([g.X[g.y==1,0] for g in train])
        zone=float(np.quantile(distances,.90))
        fraction=float((distances<=zone).mean())
        cap=max(1,int(a.per_quadrant_cap*fraction))
        train_pixels=sum(len(g.y) for g in train)
        train_positives=sum(int(g.y.sum()) for g in train)
        base_rate=train_positives/train_pixels
        test=geoms[q]
        allowed=fold['region'] & ~fold['masked_known']
        k_est=base_rate*float(allowed.sum())
        for name,columns in ARMS.items():
            check_time()
            with threadpool_limits(limits=2):
                clf,scale,_=fit_model(train,seed=q,cols=columns)
                p=predict(clf,scale,test,columns,grid.shape,zone)
            raw,st=EH.evaluate(p,fold,grid.footprint)
            allowed_zone=allowed.copy()
            allowed_zone[test.rows[test.X[:,0]>zone],test.cols[test.X[:,0]>zone]]=False
            dots=greedy_allocate(p,allowed_zone,k_truth=k_est,floor=.015,max_dots=cap)
            scored,t=EH.evaluate(dots.emitted.astype(np.float32),fold,grid.footprint)
            terms[name]=t if terms[name] is None else terms[name]+t
            surface_terms[name]=st if surface_terms[name] is None else surface_terms[name]+st
            details.append(dict(arm=name,fold=FOLD_NAMES[q],evidence_class='HOLDOUT-DTI',
                evaluator_version=EH.VERSION,withheld_positive_pixels=scored['n_truth'],
                dti=scored['dti'],soft_surface_dti=raw['dti'],emitted_pixels=dots.n_dots,
                training_zone_px=zone,training_fraction_inside_zone=fraction,dot_cap=cap,
                k_estimate_from_training=k_est,test_truth_used_for_placement=False,
                fold_receipt=fold['receipt']))
            print(f'[run] {FOLD_NAMES[q]} {name}: HOLDOUT-DTI {scored["dti"]:.5f}; soft {raw["dti"]:.5f}; dots {dots.n_dots}',flush=True)
            del p,allowed_zone,dots,clf
            gc.collect()
    pooled=EH.pooled_summary(terms,draws=1000,seed=20261009,candidate='orientation')
    raw_pooled=EH.pooled_summary(surface_terms,draws=1000,seed=20261009,candidate='orientation')
    sense_paired=EH.pooled_summary(terms,draws=1000,seed=20261009,candidate='orientation_sense')
    sense_gain=sense_paired['paired_differences']['orientation']
    keep_sense=sense_gain['ci95'][0]>0
    report=dict(**pooled, split_version=plan['split'], seed=a.seed, per_quadrant_cap=a.per_quadrant_cap,
        canary_clean=True, raw_surface_holdout=raw_pooled, sense_comparison=sense_gain,
        keep_recorded_sense=keep_sense, experiments_used=3, per_fold=details,
        historical_shipped_holdout_not_comparable='Old draw hid <=12px chunks without a context collar; AUC flags were unresolved. No gain is claimed over that instrument.',
        score_projection=None, submission_slots_used=0, runtime_seconds=time.monotonic()-start)
    save(ROOT/'evidence/orientation_holdout.json',report)
    print('[run] pooled:',{n:r['dti'] for n,r in pooled['scores'].items()},flush=True)
    if not a.build_research_surface:
        return 0
    check_time()
    name='orientation_sense' if keep_sense else 'orientation'
    columns=ARMS[name]
    with threadpool_limits(limits=2):
        clf,scale,base=fit_model(geoms,seed=0,cols=columns)
    zone=float(np.quantile(full.X[pos,0],.90))
    fraction=float((full.X[pos,0]<=zone).mean())
    joblib.dump(dict(model=clf,scale=scale,columns=columns,zone_px=zone,features=[ALL_FEATURES[i] for i in columns]),ROOT/'.cache/strand_orientation/model.joblib')
    # This TIFF is the SURFACE submitted to the pre-placement gate. It is a
    # negative research deliverable, not final-dot placement or slot promotion.
    del full,geoms,folds,visible,hidden,domain,sense
    gc.collect()
    sense=trace_sense_raster(ROOT/'data/external/trace_segments_utm11.csv',grid.shape,grid.transform)
    live_domain=grid.footprint & ~grid.catalogue
    live=geometry(grid,grid.catalogue,np.zeros(grid.shape,bool),live_domain,sense,magnetic,ROOT/'.cache/geometry','full-catalogue')
    with threadpool_limits(limits=2):
        surface=predict(clf,scale,live,columns,grid.shape,zone)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    decoded=hashlib.sha256(surface.tobytes()).hexdigest()[:12]
    label=f'gems59-relative-strand-surface-{stamp}-{decoded}'
    note='Fault-zone anatomy: learned magnetic-edge relative strike and host length; buffered LOQO. Research surface; HOLD, not slot-cleared.'
    target=ROOT/'docs/downloads'/f'{label}.tif'
    receipt=write_submission(target,surface,ROOT/'data/official/sample_submission.tif',grid.footprint,
        note=note,name=label,metadata=dict(kind='pre-placement intensity surface',emission_performed=False,
        fitted_zone_px=zone,training_fraction_inside_zone=fraction,sense_retained=keep_sense,
        no_prior_raster_used_to_build_predictions=True,holdout_evidence='orientation_holdout.json'))
    validator=validate(target,grid.footprint,grid.catalogue)
    print(f'[run] fresh research TIFF written: {target.name}',flush=True)
    # Audit EVERY accessible entry; any firing still stops production placement.
    def progress(i,n,row):
        if i%50==0:
            print(f'[registry] surface audit {i}/{n}',flush=True)
    drift=compare_to_registry(target,a.audit_registry,grid.footprint,progress=progress)
    save(ROOT/'evidence/orientation_surface_uniqueness.json',drift)
    final_dots={'status':'not_generated','reason':'literal surface gate requires stop' if not drift['unique'] else 'not requested: research surface only','emitted_pixels':0}
    if drift['stop_required']:
        print('[run] DUPLICATE/INCOMPLETE pre-placement gate: logged; STOP. No production dots generated.',flush=True)
    card=dict(hypothesis='A candidate magnetic edge\u2019s strike relative to its nearest visible fault, conditioned on distance and host length, predicts missing secondary strands better than a catalogue-only halo.',
        mechanism='Label-independent TMI derivatives supply an axial candidate tangent. Visible-only host geometry and an empirical training-distance zone drive a fitted intensity; textbook shear angles are not imposed.',
        named_non_fault_process_that_could_mimic_it='Magnetized dikes, lithologic contacts and east-west flight-line leveling residuals may form magnetic edges without fault displacement.',
        holdout_dti={**raw_pooled['scores'][name], 'representation':'soft pre-placement surface (matches downloaded TIFF method)',
            'split_version':plan['split'],'canary_clean':True},
        holdout_dot_dti={**pooled['scores'][name], 'representation':'holdout binary allocation ONLY; production dots were not generated'},
        paired_orientation_minus_anatomy=pooled['paired_differences']['anatomy'],
        correlation_overlap_vs_registry={k:drift[k] for k in ('registry_rasters_expected','registry_rasters_checked','complete_accessible_scan','worst_spearman_full_footprint','worst_rho_submission','worst_dot_overlap','worst_overlap_submission','duplicate_count','byte_unique_among_checked','pixel_unique_among_checked','unique')},
        surface_before_placement={'checked':True,'protocol_pass':drift['unique']}, final_dots=final_dots,
        raster_sha256=receipt['sha256'],validator_output=validator,
        submission_name=label,submission_note=note,submission_note_chars=len(note),
        file=str(target.relative_to(ROOT)),zip_file=str(target.with_suffix('.zip').relative_to(ROOT)),
        verdict='negative',okay_to_download=False,okay_to_submit=False,
        download_permission_status={
            'status':'NOT AUTHORIZED PENDING EXPLICIT OWNER DECISION',
            'resolution':'HOLD pending explicit owner decision (IR-S6-10)',
            'authorized':False,
            'file_availability_is_permission':False,
        },
        verdict_reason='Fresh, pixel-distinct format-valid research raster; literal overlap gate not cleared. Download authorization is not granted. No weekly slot used.',
        recorded_sense={'tested':True,'retained':keep_sense,'paired_difference':sense_gain},
        experiments_used=3,submission_slots_used=0,generated_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT/'evidence/run_card_current.json',card)
    save(ROOT/'docs/data/run_card.json',card)
    print(f'[run] NEGATIVE deliverable; runtime {(time.monotonic()-start)/60:.1f} minutes',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
