"""Regression tests for shared evaluator, buffers, rasterizer and literal gates."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import rasterio
from affine import Affine
from gems59 import evaluate_holdout as eh
from gems59.metric import dti_bruteforce, dti_exact
from gems59.holdout import buffered_component_draw, score_cell, Cell
from gems59.faultzone import rasterize_traces, trace_sense_raster, ingenious_record_segments
from gems59.strand_orientation import magnetic_geometry
from gems59.uniqueness import compare_array_to_registry, saturation_certificate, dot_overlap


def test_shared_evaluator_soft_predictions_matches_official_oracle():
    rng=np.random.default_rng(42)
    p=rng.random((20,20))*.8
    truth=np.zeros_like(p,bool);truth[4,4]=1;truth[9,10]=1
    known=np.zeros_like(truth);known[5,4]=1
    region=np.ones_like(truth);region[:,0]=0
    fold=dict(region=region,truth=truth,visible=known)
    report,terms=eh.evaluate(p,fold,np.ones_like(truth),block_side=5)
    active=region & ~known
    oracle=dti_bruteforce(np.where(active,p,0),truth & active)
    assert report['dti']==pytest.approx(oracle['dti'],abs=1e-12)
    assert eh.from_terms(terms.sum(axis=0))==pytest.approx(report['dti'],abs=1e-12)
    pooled=eh.pooled_summary({'candidate':terms,'baseline':terms.copy()},draws=50,candidate='candidate')
    assert pooled['scores']['candidate']['withheld_positive_pixels']==2
    assert pooled['paired_differences']['baseline']['ci95']==[0.0,0.0]
    json.dumps(pooled,allow_nan=False)


def test_shared_evaluator_grid_mismatch_and_no_positives_rejected():
    p=np.zeros((8,8));full=np.ones_like(p,bool)
    with pytest.raises(ValueError):
        eh.evaluate(p,dict(region=full,truth=full,visible=full),full)
    with pytest.raises(ValueError):
        eh.evaluate(p,dict(region=full,truth=np.ones((4,4)),visible=full),full)
    with pytest.raises(ValueError):
        eh.evaluate(np.full_like(p,np.nan),dict(region=full,truth=full,visible=~full),full)


def test_score_cell_has_fn_even_when_no_dots():
    active=np.ones((5,5),bool)
    c=Cell('draw20_foldNW_all',0,20,'all',(slice(None),slice(None)),active,(np.array([2]),np.array([2])),np.zeros((5,5),np.float32),1)
    r=score_cell(c,np.zeros((5,5),bool))
    assert r['fn']==1 and r['tp']==0


def test_buffered_holdout_whole_components_and_no_feature_leakage():
    fp=np.ones((100,100),bool);cat=np.zeros_like(fp)
    for y in (10,20,60,70):
        for x in (10,20,60,70):
            cat[y:y+6,x]=1
    grid=SimpleNamespace(footprint=fp,catalogue=cat)
    folds,quad=buffered_component_draw(grid,seed=20,hide_frac=.25,buffer_px=3,boundary_px=4)
    visible=folds[0]['visible'];hidden=folds[0]['hidden_all']
    from scipy.ndimage import label,distance_transform_edt
    labels,n=label(cat,np.ones((3,3)))
    for k in range(1,n+1):
        assert not (hidden[labels==k].any() and not hidden[labels==k].all())
    assert distance_transform_edt(~visible)[hidden].min()>3
    assert not (hidden & visible).any()
    assert len(folds)==4 and all(f['truth'].any() for f in folds)
    assert all(f['receipt']['evaluation_region_label_blind'] for f in folds)


def _csv(tmp_path):
    p=tmp_path/'traces.csv'
    # Two widely-separated records; row zero must NOT be assigned record one's final row.
    p.write_text('record_id,x0,y0,x1,y1,sense\n1,2.5,17.5,2.5,14.5,RL\n2,15.5,4.5,18.5,4.5,LL\n')
    return p


def test_trace_endpoint_deltas_record_index_and_sense_are_consistent(tmp_path):
    tr=Affine(1,0,0,0,-1,20)
    csv=_csv(tmp_path)
    mask,_=rasterize_traces(csv,(20,20),tr)
    assert mask[2:6,2].all() and mask[15,15:19].all()
    assert mask.sum()==8
    sense=trace_sense_raster(csv,(20,20),tr)
    assert np.all(sense[2:6,2]==2) and np.all(sense[15,15:19]==3)
    stats,pixels,_,_=ingenious_record_segments(csv,(20,20),tr)
    assert np.all(pixels[1][1]==2) and np.all(pixels[2][0]==15)
    assert stats[1]['sense']=='RL' and stats[2]['sense']=='LL'


def test_magnetic_tangent_and_missing_data_guard():
    y,x=np.indices((40,40))
    vertical=magnetic_geometry(x.astype(float),np.ones((40,40),bool))
    assert vertical['cos2'][20,20]==pytest.approx(1,abs=1e-6)
    assert vertical['sin2'][20,20]==pytest.approx(0,abs=1e-6)
    horizontal=magnetic_geometry(y.astype(float),np.ones((40,40),bool))
    assert horizontal['cos2'][20,20]==pytest.approx(-1,abs=1e-6)
    a=x.astype(float);a[20,20]=np.nan
    out=magnetic_geometry(a,np.ones((40,40),bool))
    assert not out['support'][20,20]
    assert out['log_gradient'][20,20]==0
    assert all(np.isfinite(out[n]).all() for n in ('sin2','cos2','coherence','log_gradient'))


def _tif(path,arr):
    with rasterio.open(path,'w',driver='GTiff',height=arr.shape[0],width=arr.shape[1],count=1,dtype='float32',crs='EPSG:32611',transform=Affine(100,0,0,0,-100,2000)) as dst:
        dst.write(arr.astype(np.float32),1)


def test_missing_or_empty_registry_never_certifies_uniqueness(tmp_path):
    a=np.zeros((20,20),np.float32);a[10,10]=1
    fp=np.ones_like(a,bool)
    index=tmp_path/'registry.json'
    index.write_text(json.dumps([dict(repo='test',submission='missing',file=str(tmp_path/'missing.tif'))]))
    r=compare_array_to_registry(a,index,fp)
    assert not r['unique'] and not r['complete_accessible_scan'] and r['missing_or_invalid']==1
    index.write_text('[]')
    r=compare_array_to_registry(a,index,fp)
    assert not r['unique'] and not r['complete_accessible_scan']


def test_dense_prior_is_literal_universal_blocker_not_exempted(tmp_path):
    prior=tmp_path/'dense.tif';_tif(prior,np.full((20,20),.01))
    fp=np.ones((20,20),bool)
    cert=saturation_certificate(prior,fp)
    assert cert['universal_overlap_blocker'] and cert['uncovered_allowed_pixels']==0
    p=np.zeros((20,20),np.float32);p[10,10]=1
    index=tmp_path/'registry.json'
    index.write_text(json.dumps([dict(repo='test',submission='dense',file=str(prior))]))
    r=compare_array_to_registry(p,index,fp)
    assert r['worst_dot_overlap']==1.0 and r['duplicate_count']==1 and not r['unique']
    assert r['byte_unique_among_checked'] and r['pixel_unique_among_checked']
    json.dumps(r,allow_nan=False)


def test_saturation_certificate_requires_matching_reference_grid(tmp_path):
    prior=tmp_path/'prior.tif'
    _tif(prior,np.ones((20,20),np.float32))
    fp=np.ones((20,20),bool)
    good=(fp.shape,rasterio.crs.CRS.from_epsg(32611),Affine(100,0,0,0,-100,2000))
    assert saturation_certificate(prior,fp,expected_grid=good)['universal_overlap_blocker']
    wrong=(fp.shape,good[1],Affine(100,0,100,0,-100,2000))
    with pytest.raises(ValueError,match='CRS/transform'):
        saturation_certificate(prior,fp,expected_grid=wrong)
    with pytest.raises(ValueError,match='catalogue / footprint'):
        saturation_certificate(prior,fp,np.zeros((19,20),bool),expected_grid=good)
    with pytest.raises(ValueError,match='empty or non-2D'):
        saturation_certificate(prior,np.zeros((20,20),bool),expected_grid=good)


def test_nonfinite_registry_pixels_not_dots_and_radius_is_inclusive():
    a=np.zeros((10,10));b=a.copy();a[5,5]=1;b[5,8]=1
    assert dot_overlap(a,b)==1
    b[5,8]=np.nan
    assert dot_overlap(a,b)==0


def test_valid_primary_strikes_survive_fold_geometry_and_angles_are_relative():
    from gems59.anatomy import fold_geometry
    v=np.zeros((40,40),bool);v[20,5:35]=1
    hidden=np.zeros_like(v);hidden[10,20]=1
    grid=SimpleNamespace(shape=v.shape)
    g=fold_geometry(grid,v,hidden,~v,'east-west')
    selected=(g.rows==10)&(g.cols==20)
    point=g.X[selected][0]
    assert point[6]==pytest.approx(-1,abs=1e-6)  # cos(2*90deg)
    assert abs(point[5])<1e-6                   # sin(2*90deg)
    assert point[1]==pytest.approx(10,abs=1e-5) # cross-strike north offset
    assert point[2]==pytest.approx(0,abs=1e-5)  # along-strike offset
    alternative=~hidden
    same=fold_geometry(grid,v,alternative,~v,'hidden-values-perturbed')
    np.testing.assert_array_equal(g.X,same.X)   # hidden values cannot reach ANY feature
    assert not np.array_equal(g.y,same.y)


def test_empty_visible_catalogue_is_not_an_implicit_array_boundary_fault():
    from gems59.network import nearest_frame
    with pytest.raises(ValueError,match='undefined'):
        nearest_frame(np.zeros((10,10),bool))
