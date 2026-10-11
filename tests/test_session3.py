"""Negative-result artifact is independently new, valid, and explicitly blocked."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from types import SimpleNamespace

from gems59 import load_grid
from gems59.fitting import canary
from gems59.validate import validate

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'evidence' / 'exp4_width.json'
INDEX = ROOT / 'evidence' / 'registry_full_index.json'


def test_shared_canary_rejects_mislabelled_feature_columns():
    g = SimpleNamespace(X=np.zeros((30, 10), np.float32), y=np.r_[np.ones(15),np.zeros(15)])
    with pytest.raises(ValueError, match='feature names'):
        canary([g], feature_names=('wrong',))


def test_new_research_raster_is_distinct_and_fail_closed():
    e = json.loads(EXP.read_text())
    ras = e['research_raster']
    path = ROOT / ras['path']
    assert path.is_file() and 'DO-NOT-SUBMIT' in path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ras['sha256']
    assert ras['sha256'] not in {r['sha256'] for r in json.loads(INDEX.read_text())['rasters']}
    g = load_grid()
    v = validate(path, g.footprint, g.catalogue)
    assert v['all_checks_passed'] and v['n_nan'] == 0
    assert v['meta']['shape'] == [g.height,g.width]
    assert v['meta']['transform'] == list(tuple(g.transform)[:6])
    assert e['surface_gate_passes'] is False
    assert any(w['candidate_surface_positive_3px_overlap'] > .70 for w in e['surface_witnesses'])
    assert 'DO NOT SUBMIT' in e['verdict']
    assert 'DO NOT SUBMIT' in (ROOT/'docs'/'archive.html').read_text()
    assert path.name in (ROOT/'docs'/'archive.html').read_text()
    assert 'DO NOT SUBMIT' in (ROOT/'index.html').read_text()
    assert 'gems59-faultzone-anatomy-60000px' not in (ROOT/'index.html').read_text()
    card = json.loads((ROOT/'evidence'/'run_card_legacy_width.json').read_text())
    assert card['raster_sha256'] == ras['sha256']
    assert card['slot_used'] is False


def test_witnesses_are_exact_indexed_files_and_why_literal_gate_is_blocked():
    e = json.loads(EXP.read_text())
    idx = {r['sha256']:r for r in json.loads(INDEX.read_text())['rasters']}
    for w in e['surface_witnesses']:
        assert w['sha256'] in idx
        assert any(Path(s.split(':',1)[1]).name == Path(w['source']).name
                   for s in idx[w['sha256']]['sources'])
    assert max(x['prior_coverage_of_eligible_3px'] for x in e['surface_witnesses']) == 1.0
    assert e['sha256_distinct_from_all_644_indexed_rasters'] is True
    assert e['dot_placement'].startswith('STOPPED')
