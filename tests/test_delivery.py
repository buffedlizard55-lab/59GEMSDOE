"""Regression coverage for the literal legacy gate, static site and feed parser."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pytest
import rasterio
from affine import Affine
from gems59 import gates
from gems59.uniqueness import compare_array_to_registry

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(tmp_path,value=.001):
    path=tmp_path/'dense.tif'
    transform=Affine(100,0,0,0,-100,2000)
    with rasterio.open(path,'w',driver='GTiff',height=20,width=20,count=1,dtype='float32',crs='EPSG:32611',transform=transform) as dst:
        dst.write(np.full((20,20),value,np.float32),1)
    manifest=dict(rasters=[dict(repo_first='fixture',sources=['fixture:dense'],cache_file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())],complete_accessible_scan=True,errors=[],n_unique_grid_rasters=1)
    candidate=np.zeros((20,20),np.float32);candidate[10,10]=1
    return path,manifest,candidate,np.ones((20,20),bool)


@pytest.mark.parametrize('phase',['surface','dots'])
def test_legacy_interface_uses_literal_low_probability_support_at_both_phases(tmp_path,phase):
    prior,manifest,p,fp=fixture(tmp_path)
    report=gates.lane_report(p,fp,sample=prior,registry_index=manifest,phase=phase)
    assert report['worst_dot_overlap']==1
    assert not report['ok'] and report['duplicate']
    assert report['policy_exemptions']==[]


def test_saturation_exemption_and_partial_old_gate_cannot_clear(tmp_path):
    prior,manifest,p,fp=fixture(tmp_path)
    with pytest.raises(ValueError,match='not authorized'):
        gates.lane_report(p,fp,sample=prior,registry_index=manifest,probe_coverage=.95)
    with pytest.raises(RuntimeError,match='inventory manifest'):
        gates.lane_report(p,fp,[prior],sample=prior)
    with pytest.raises(RuntimeError,match='retired'):
        gates.uniqueness_report(p,[prior],top=1)


def test_bare_registry_list_is_unattested_even_when_files_exist(tmp_path):
    prior,manifest,p,fp=fixture(tmp_path)
    report=compare_array_to_registry(p,[dict(file=str(prior),submission='fixture')],fp)
    assert not report['complete_accessible_scan'] and not report['unique']
    assert report['source_errors']


def test_constant_candidate_has_no_rank_evidence(tmp_path):
    prior,manifest,p,fp=fixture(tmp_path)
    report=compare_array_to_registry(np.full_like(p,.001),manifest,fp)
    assert not report['candidate_rank_variation'] and not report['unique']
    assert report['stop_required']


def test_legacy_gate_checks_fixture_transform(tmp_path):
    prior,manifest,p,fp=fixture(tmp_path)
    wrong=tmp_path/'wrong.tif'
    with rasterio.open(wrong,'w',driver='GTiff',height=20,width=20,count=1,dtype='float32',crs='EPSG:32611',transform=Affine(100,0,100,0,-100,2000)) as dst:
        dst.write(np.full_like(p,.001),1)
    manifest['rasters'][0].update(cache_file=str(wrong),sha256=hashlib.sha256(wrong.read_bytes()).hexdigest())
    report=gates.lane_report(p,fp,sample=prior,registry_index=manifest,phase='surface')
    assert report['missing_or_invalid']==1 and not report['ok']


def test_current_tiff_zip_links_and_card_are_consistent():
    result=script('check_site').check()
    assert result['links_pass'] and result['zip_exactly_one_tiff']
    assert result['current_card_consistent'] and result['submission_cleared']
    assert result['registry_rasters_checked'] == result['registry_rasters_checked'] > 600
    assert result['worst_spearman'] <= 0.90
    assert result['worst_reverse_overlap'] <= 0.70
    assert result['worst_jaccard'] <= 0.50
    assert result['worst_support_matched_overlap'] <= 0.70


def test_feed_parser_accepts_only_a_real_sorted_table():
    parse=script('refresh_feed').parse_leaderboard
    html='<table><tr><th>Rank</th><th>Team</th><th>Participant</th><th>Score</th></tr><tr><td>#1</td><td></td><td>first</td><td>0.3774</td></tr><tr><td>#2</td><td></td><td>second</td><td>0.3361</td></tr></table>'
    rows=parse(html)
    assert rows[0]==dict(rank=1,participant_display='first',public_dti=.3774)
    with pytest.raises(ValueError):parse('<html>Login required</html>')
    with pytest.raises(ValueError):parse(html.replace('0.3361','0.4774'))
    with pytest.raises(ValueError):parse(html.replace('0.3774','2.0000'))


def test_report_parser_preserves_unknown_and_deduplicates_without_confirmation():
    parse=script('build_site').owner_reports
    text='https://buffedlizard55-lab.github.io/GEMSDOE32/\nexample: 0.2778\nexample: 0.2778\nunknown:\n'
    rows=parse(text)
    assert len(rows)==2 and rows[0]['owner_reported_dti']==.2778
    assert rows[1]['owner_reported_dti'] is None
    assert all(r['submission_receipt'] is None for r in rows)


def test_preview_png_and_holdout_structure_labels_are_real():
    assert (ROOT/'docs/assets/preview.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    source=json.loads((ROOT/'evidence/orientation_structure.json').read_text())
    assert source['distance_quantile_probabilities']==[.1,.5,.9,.95,.99]
    relative=source['relative_strike']
    assert len(relative['n_withheld'])==len(relative['edges'])-1
    assert sum(relative['n_withheld'])==relative['n_withheld_total']
    assert sum(relative['n_visible_reference'])==relative['n_visible_total']
    assert source['withheld_positive_pixels']>=relative['n_withheld_total']
    assert relative['sample_unit']=='fault raster pixel; not segment-weighted'
    assert relative['minimum_local_coherence']==.2
    assert relative['angle_convention']=='unsigned axial difference in degrees, folded to [0,90]'
    page=(ROOT/'docs/method.html').read_text()
    assert '10th / 50th / 90th percentiles' in page
    assert 'HOLDOUT-STRUCTURE (descriptive, not a score)' in page
    assert 'Withheld angle pixels (count; % of valid sample)' in page
    assert 'not segment-weighted' in page
    assert 'distance quartile/median/upper-quartile' not in page
