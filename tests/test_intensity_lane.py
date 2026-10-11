"""Session H59-I2 Intensity-Lane regression & deliverable verification tests."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from gems59 import intensity as INT
from gems59.grid import load_grid
from scripts.check_site import check_links

ROOT = Path(__file__).resolve().parents[1]


def test_tile_blocks_does_not_wrap_modulo_10():
    """IR-59-LATTICE-MOD10: spacing=3 from anchor=8 must not wrap 8+3=11 -> 1 (7,8 1px apart)."""
    n_b_cell = np.full((20, 20), 2.5, dtype=np.float32)
    lam = np.full((20, 20), 0.025, dtype=np.float32)
    lam[8, 8] = 0.90
    lam[18, 18] = 0.90
    elig = np.ones((20, 20), dtype=bool)
    classes = np.ones((2, 2), dtype=np.int32)
    dots = INT.tile_blocks(n_b_cell, lam, elig, 3.0, classes=classes)
    ys, xs = np.nonzero(dots[:10, :10])
    # Inside block (0,0) with anchor (8,8) and s=3, valid rows are [2, 5, 8], never [8, 1, 4, 7]
    assert set(ys.tolist()) <= {2, 5, 8}
    assert set(xs.tolist()) <= {2, 5, 8}
    assert not (7 in set(ys.tolist()) and 8 in set(ys.tolist()))


def test_tile_blocks_corridor_enforces_cross_block_spacing():
    lam = np.zeros((30, 30), dtype=np.float32)
    corridor = np.ones((30, 30), dtype=bool)
    lam[9, 10] = 0.8
    lam[10, 10] = 0.79  # 1 px across the 10x10 block boundary
    cls_b = np.full((3, 3), 4, dtype=np.int32)
    dots = INT.tile_blocks_corridor(lam, corridor, cls_b, {4: 3.0}, cell_quantile=0.0)
    ys, xs = np.nonzero(dots)
    coords = np.column_stack([ys, xs]).astype(np.float64)
    assert len(coords) >= 1
    if len(coords) >= 2:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        np.fill_diagonal(dist, 999.0)
        assert float(dist.min()) >= 3.0 - 1e-6


def test_qfaults_effort_rasters_joins_all_catalogue_pixels():
    grid = load_grid()
    fcode, mscale = INT.qfaults_effort_rasters(grid, grid.catalogue, ROOT)
    cat = grid.catalogue
    assert fcode.shape == cat.shape and mscale.shape == cat.shape
    matched = int((fcode[cat] > 0).sum())
    assert matched == int(cat.sum()) == 60988
    assert set(np.unique(fcode[cat]).tolist()) == {1, 2, 3}
    assert set(np.unique(mscale[cat]).tolist()) == {100, 250}


def test_promoted_h59i2_submission_and_run_card_consistency():
    card = json.loads((ROOT / "evidence/run_card.json").read_text())
    pub_card = json.loads((ROOT / "docs/data/run_card.json").read_text())
    assert card == pub_card
    assert card["verdict"] == "promote"
    assert card["submission"]["okay_to_download"] is True
    assert card["submission"]["okay_to_submit"] is True
    assert 0 < card["submission"]["note_chars"] <= 140
    assert card["submission"]["note_chars"] == len(card["submission"]["note"])

    # Holdout superiority over both Session H59-I and published all-lane best
    dti = card["holdout_dti"]["final_locked"]["dti"]
    ci_lo, ci_hi = card["holdout_dti"]["final_locked"]["ci95"]
    assert dti > 0.160 and ci_lo > 0.135204 and ci_hi > dti
    assert card["holdout_dti"]["beats_previous_h59i"] is True
    assert card["holdout_dti"]["beats_published_holdout_best"] is True

    # Leakage canary and decile calibration
    assert card["leakage_canary"]["any_flag"] is False
    assert card["leakage_canary"]["max_holdout_feature_auc"] < 0.90
    assert card["leakage_canary"]["max_train_loco_feature_auc"] < 0.90
    assert card["decile_calibration"]["n_footprint_blocks"] == 52309
    assert card["decile_calibration"]["total_withheld_pixels"] == 11321
    assert card["decile_calibration"]["share_withheld_top3_deciles"] > 0.75
    assert card["decile_calibration"]["monotone_withheld_pixels"] is True

    # Uniqueness audit
    u = card["registry"]["uniqueness"]
    assert u["registry_rasters_checked"] == 695
    assert u["complete_accessible_scan"] is True
    assert u["worst_surface_spearman_full_footprint"] <= 0.90
    assert u["worst_spearman_full_footprint"] <= 0.90
    assert u["worst_jaccard_dot_sets"] <= 0.50
    assert u["representation_aware"]["worst_forward_overlap_vs_smaller_or_equal_peers"] <= 0.70
    assert u["representation_aware"]["vs_session_h59i"]["my_dots_within_3px_of_theirs"] <= 0.70

    # GeoTIFF and single-TIFF ZIP verification
    tif = ROOT / card["raster"]["file"]
    docs_tif = ROOT / card["raster"]["docs_file"]
    zip_f = ROOT / card["raster"]["zip_file"]
    docs_zip = ROOT / card["raster"]["docs_zip_file"]
    assert tif.exists() and docs_tif.exists() and zip_f.exists() and docs_zip.exists()
    sha = hashlib.sha256(tif.read_bytes()).hexdigest()
    assert sha == card["raster"]["sha256"] == hashlib.sha256(docs_tif.read_bytes()).hexdigest()
    with zipfile.ZipFile(docs_zip) as zf:
        assert zf.namelist() == [tif.name]
        assert zf.read(tif.name) == tif.read_bytes()
    with rasterio.open(tif) as ds:
        arr = ds.read(1)
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        assert str(ds.crs) == "EPSG:32611" and ds.shape == (3730, 3292)
        assert np.isfinite(arr).all()
        assert set(np.unique(arr).tolist()) <= {0.0, 1.0}
        assert int((arr > 0).sum()) == card["raster"]["dots"] == 77727


def test_site_pages_and_links_pass():
    pages, errors = check_links(ROOT / "docs")
    assert len(pages) >= 4
    assert not errors, f"Broken links in docs/: {errors}"
    for page_name in ("index.html", "executive-summary.html"):
        html = (ROOT / "docs" / page_name).read_text()
        assert "DOWNLOAD: OK" in html
        assert "SUBMIT TO COMPETITION: OK" in html
        assert "gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif" in html
        assert "088069aec07a9150d9473f94f259bd1f7e4c5212f4ff55c1866bf41b3d1fc8e6" in html
