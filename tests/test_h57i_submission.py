"""Preserve historical H57-I format/bytes; withdraw its partial-registry clearance."""
from pathlib import Path
import hashlib
import json
import zipfile
import pytest
import numpy as np
import rasterio
from gems59 import load_grid
from gems59.validate import validate
from gems59.uniqueness import compare_to_registry

ROOT=Path(__file__).resolve().parents[1]
TIF=ROOT/'docs/downloads/gems59-h57i-iso_full-20261009T202310Z-5e393d50e59a-zeros.tif'
ALIAS=ROOT/'docs/downloads/archive/SUBMIT-THIS-gems59-h57i-iso-zeros.tif'
SHA='0025ec29647e3778043b81f5bf998e0e652ea25a8ead03339f40e9939de78b22'


@pytest.mark.skipif(not TIF.is_file(), reason="historical H57I TIFF is not included in this repository checkout")
def test_historical_tif_format_and_sha_preserved():
    grid=load_grid();v=validate(TIF,grid.footprint,grid.catalogue)
    assert v['all_checks_passed'] and v['n_nan']==0
    assert v['min']==0 and v['max']==1
    assert v['emitted_positive_pixels']==37654 and v['on_catalogue_positive_pixels']==0
    assert v['sha256']==SHA==hashlib.sha256(TIF.read_bytes()).hexdigest()
    with rasterio.open(TIF) as ds:
        assert ds.nodata is None and set(np.unique(ds.read(1)))<={0.,1.}


@pytest.mark.skipif(not TIF.is_file() or not ALIAS.is_file(), reason="historical H57I archive and alias are absent from this checkout")
def test_misleading_alias_is_archived_not_advertised():
    assert ALIAS.is_file() and ALIAS.read_bytes()==TIF.read_bytes()
    assert not (ROOT/'docs/downloads'/ALIAS.name).exists()
    assert 'SUBMIT-THIS' not in (ROOT/'docs/index.html').read_text()


@pytest.mark.skipif(not TIF.is_file() or not TIF.with_suffix('.zip').is_file(), reason="historical H57I TIFF/ZIP are absent from this checkout")
def test_historical_zip_contains_only_its_tif():
    with zipfile.ZipFile(TIF.with_suffix('.zip')) as archive:
        assert archive.namelist()==[TIF.name]
        assert archive.read(TIF.name)==TIF.read_bytes()


@pytest.mark.skipif(
    not TIF.is_file() or not (ROOT/'registry/registry_index.json').is_file()
    or not (ROOT/'evidence/h57i_clearance_withdrawal.json').is_file(),
    reason="historical H57I TIFF / partial registry / withdrawal receipt are absent from this checkout",
)
def test_partial_16_prior_check_never_clears_and_dense_witness_fails():
    grid=load_grid()
    partial=compare_to_registry(TIF,ROOT/'registry/registry_index.json',grid.footprint)
    assert not partial['unique'] and not partial['complete_accessible_scan']
    withdrawal=json.loads((ROOT/'evidence/h57i_clearance_withdrawal.json').read_text())
    assert withdrawal['candidate_file_sha256']==SHA
    assert withdrawal['worst_dot_overlap']==1 and not withdrawal['unique']


def test_current_site_and_card_agree_on_submission_permission():
    text=(ROOT/'docs/index.html').read_text();card=json.loads((ROOT/'evidence/run_card_current.json').read_text())
    # The strict H59-J literal-overlap rule requires STOP; inspection download is not clearance.
    assert 'CLEARED TO SUBMIT: NO' in text
    assert 'Submit to competition: OK' not in text
    assert card['okay_to_submit'] is False
    assert card['verdict']=='negative'
    assert card['correlation_overlap_vs_registry']['strict_literal_forward_overlap_hits']==86
    assert card['registry_gate_phases']['surface_before_placement']['status']=='NOT VERIFIED'
    assert card['submission_slots_used']==0
    assert card['approved_for_weekly_slot'] is False
    assert card['raster_sha256']!='0025ec29647e3778043b81f5bf998e0e652ea25a8ead03339f40e9939de78b22'
    assert card['file'].split('/')[-1] in text
    assert 'IR-59J-05' in (ROOT/'docs/irregularities.html').read_text()
    assert (ROOT/'docs/data/run_card.json').read_bytes()==(ROOT/'evidence/run_card_current.json').read_bytes()
    import html as _html
    note=card['submission_note'];assert 1<=len(note)<=140
    assert _html.escape(note, quote=True) in text, 'historical run-card note should be shown only as provenance'
