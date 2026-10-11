"""Session-7 H57-R: off-catalogue proxy instrument, joint strata, submission format.

These are contract tests. They pin the pieces a reviewer must be able to trust:
the proxy derivation is reproduced from the pinned bytes, no predictor column can
come from the proxy layer, the damage-zone intensity really widens with host
length, the emitted raster is a well-formed competition submission, and the
portfolio-level evidence carries the negative result as well as the positive.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gems59 import offcatalogue as oc                      # noqa: E402
from gems59.grid import load_grid                           # noqa: E402
from gems59.validate import validate                       # noqa: E402

CARD = ROOT / 'evidence/run_card_h57r.json'


@pytest.fixture(scope='module')
def grid():
    return load_grid()


def test_predictor_sources_fail_closed():
    oc.assert_predictor_sources(['catalogue:d', 'ingenious:sense'])
    with pytest.raises(ValueError):
        oc.assert_predictor_sources(['catalogue:d', 'sgmc:proxy'])


def test_proxy_truth_is_rebuilt_from_pinned_sgmc_bytes(grid, tmp_path):
    """IR-57-SGMC-01: the derived raster is a dilation, so the proxy is rebuilt."""
    proxy, receipt = oc.load_proxy_truth(ROOT, grid)
    import rasterio
    with rasterio.open(ROOT / 'data/external/sgmc_faults_100m.tif') as src:
        sgmc = src.read(1) > 0
    assert np.array_equal(proxy, sgmc & ~grid.catalogue & grid.footprint)
    assert int((proxy & grid.catalogue).sum()) == 0
    assert 'irregularity_IR_57_SGMC_01' in receipt
    assert receipt['proxy_pixels'] > 0


def _builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'build_h57r', ROOT / 'scripts/build_h57r.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_intensity_width_grows_with_host_length():
    """Savage & Brodsky style: a longer host must spread further."""
    mod = _builder()
    canvas = np.zeros((161, 1401), bool)
    canvas[20, 20:40] = True            # 20 px host   -> class 1 (sigma 6)
    canvas[80, 20:320] = True           # 300 px host  -> class 3 (sigma 15)
    canvas[140, 20:1220] = True         # 1200 px host -> class 3 (sigma 16)
    short = np.zeros((121, 121), bool)
    short[60, 20:40] = True
    i_short, _, used_short = mod.intensity_field(short)
    _, _, used = mod.intensity_field(canvas)
    assert [c['sigma_px'] for c in used_short] == [5.0]
    assert [c['sigma_px'] for c in used] == [5.0, 15.0, 24.0]
    i_mid, _, _ = mod.intensity_field(np.pad(
        np.ones((1, 300), bool), ((60, 60), (0, 31))))
    reach_short = int((i_short > 0.05 * i_short.max()).sum())
    reach_mid = int((i_mid > 0.05 * i_mid.max()).sum())
    assert reach_mid > reach_short
    # the real catalogue must populate every displacement class
    _, _, used_real = mod.intensity_field(load_grid().catalogue)
    assert len(used_real) == len(mod.LENGTH_CLASSES)
    assert [c['sigma_px'] for c in used_real] == [c[1] for c in mod.LENGTH_CLASSES]


def test_distance_profile_is_a_rate_not_a_count(grid):
    proxy, _ = oc.load_proxy_truth(ROOT, grid)
    sense, _ = oc.load_sense(ROOT, grid)
    cov = oc.catalogue_covariates(grid, sense)
    profile = oc.distance_profile(grid.footprint, proxy, cov)
    edges = np.asarray(profile['edges'], float)
    assert edges[0] == 1.0 and edges[-1] >= 1e9
    assert all(b > 0 for b in profile['band_rate'])
    assert sum(profile['band_positive']) == profile['n_positive']
    # the nearest band must be enriched over the far band: the fitted zone is real
    assert profile['band_rate'][0] > 3.0 * profile['band_rate'][-2]


def test_relative_strike_structure_is_normalised(grid):
    proxy, _ = oc.load_proxy_truth(ROOT, grid)
    cov = oc.catalogue_covariates(grid, np.zeros(grid.shape, np.int8))
    out = oc.relative_strike_structure(grid, proxy, cov)
    assert abs(sum(out['proxy_bins']) - out['n_proxy']) < 1e-9
    assert abs(sum(out['null_bins']) - out['n_null']) < 1e-9
    assert len(out['enrichment']) == len(out['edges']) - 1
    assert out['n_proxy'] > 1000 and out['n_null'] > 1000


def test_fold_domains_are_disjoint_and_eroded(grid):
    proxy, _ = oc.load_proxy_truth(ROOT, grid)
    domains = oc.fold_domains(grid, proxy)
    names = list(domains)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not (domains[names[a]]['region'] & domains[names[b]]['region']).any()
    total = sum(int(d['truth'].sum()) for d in domains.values())
    assert total < int(proxy.sum())   # the erosion collar removes truth by design


def test_submission_is_a_valid_single_band_rasters():
    card = json.loads(CARD.read_text())
    path = ROOT / card['file']
    with rasterio.open(path) as src:
        a = src.read(1)
        assert src.count == 1
        assert str(src.crs) == 'EPSG:32611'
        assert (src.height, src.width) == (3730, 3292)
        assert tuple(src.transform)[:6] == (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0
    grid = load_grid()
    dots = a > 0
    assert int(dots.sum()) == card['portal_preflight']['positive_pixels']
    assert int((dots & grid.catalogue).sum()) == 0          # thread 11536
    assert int((dots & ~grid.footprint).sum()) == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == card['sha256']
    assert len(card['submission_note']) <= 140


def test_zip_contains_exactly_the_same_tiff():
    card = json.loads(CARD.read_text())
    import zipfile
    zpath = ROOT / card['zip_file']
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert names == [Path(card['file']).name]
        assert hashlib.sha256(z.read(names[0])).hexdigest() == card['sha256']


def test_validator_reports_every_check_true():
    card = json.loads(CARD.read_text())
    checks = card['validator']['checks']
    assert checks, 'validator produced no checks'
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}


def test_portfolio_evidence_keeps_the_negative_result():
    """The proxy-instrument anti-correlation must survive in the receipt."""
    path = ROOT / 'evidence/offcat_instrument_check.json'
    if not path.exists():
        pytest.skip('portfolio evidence not built in this run')
    data = json.loads(path.read_text())
    full = data['spearman_live_vs_proxy_full_density']
    assert full['spearman_live_vs_proxy'] < -0.5
    assert all(s['spearman_live_vs_proxy'] < 0.3 for s in data['summary'])
    assert data['conclusion'].startswith('The off-catalogue proxy cannot rank placements')


def test_every_receipt_declares_its_evidence_class():
    """Every Session-7 receipt must name what kind of evidence it is."""
    names = ('h57r_build.json', 'run_card_h57r.json', 'offcat_structure.json',
             'offcat_instrument_check.json', 'portfolio_live_evidence.json',
             'hypotheses_current.json')
    seen = 0
    for name in names:
        path = ROOT / 'evidence' / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert data.get('evidence_class'), f'{name} has no evidence_class label'
        seen += 1
    assert seen >= 4


def test_hypotheses_record_the_negative_results():
    data = json.loads((ROOT / 'evidence/hypotheses_current.json').read_text())
    assert data['negative_results_this_session'], 'negative results must be recorded'
    assert any('-0.928' in t for t in data['negative_results_this_session'])
