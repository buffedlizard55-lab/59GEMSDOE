"""Tests for the submission writer/validator. The template below is a SYNTHETIC fixture
(made-up origin, 30x40 grid) used only to exercise the code path. It is not the competition grid."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from gems_dti.submission import (
    SENTINEL_NODATA,
    SubmissionError,
    validate_array,
    validate_file,
    write_submission,
)

H, W = 30, 40


@pytest.fixture()
def template(tmp_path):
    p = tmp_path / "synthetic_template.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32611),
        transform=from_origin(0.0, 3000.0, 100.0, 100.0),
    ) as dst:
        dst.write(np.zeros((H, W), np.float32), 1)
    return p


def _good(seed=1):
    rng = np.random.default_rng(seed)
    v = rng.random((H, W)).astype(np.float32)
    v[v < 0.9] = 0.0
    return v


def test_good_submission_writes_and_revalidates(tmp_path, template):
    out = tmp_path / "sub.tif"
    receipt = write_submission(_good(), out, template, name="unit-test-sub", note="unit test only")
    assert receipt["all_zero_warning"] is False
    assert receipt["sha256"] and len(receipt["sha256"]) == 64
    with rasterio.open(out) as src:
        assert src.count == 1 and src.dtypes[0] == "float32"
        assert src.crs.to_epsg() == 32611
        assert (src.height, src.width) == (H, W)
        assert src.nodata is None
    assert json.loads(out.with_suffix(".receipt.json").read_text())["name"] == "unit-test-sub"


def test_out_of_range_is_rejected(tmp_path, template):
    v = _good()
    v[0, 0] = 1.2
    with pytest.raises(SubmissionError, match=r"\[0, 1\]"):
        write_submission(v, tmp_path / "bad.tif", template, name="bad-range", note="x")


def test_negative_is_rejected(tmp_path, template):
    v = _good()
    v[0, 0] = -0.01
    with pytest.raises(SubmissionError, match=r"\[0, 1\]"):
        write_submission(v, tmp_path / "bad.tif", template, name="bad-neg", note="x")


def test_nan_is_rejected(tmp_path, template):
    v = _good()
    v[5, 5] = np.nan
    with pytest.raises(SubmissionError, match="NaN"):
        write_submission(v, tmp_path / "bad.tif", template, name="bad-nan", note="x")


def test_sentinel_is_rejected(tmp_path, template):
    v = _good()
    v[7, 7] = np.float32(SENTINEL_NODATA)
    with pytest.raises(SubmissionError, match="sentinel"):
        write_submission(v, tmp_path / "bad.tif", template, name="bad-sentinel", note="x")


def test_wrong_dtype_and_shape_rejected():
    with pytest.raises(SubmissionError, match="float32"):
        validate_array(np.zeros((H, W), np.float64), {"height": H, "width": W})
    with pytest.raises(SubmissionError, match="shape"):
        validate_array(np.zeros((H + 1, W), np.float32), {"height": H, "width": W})


def test_note_length_and_name_limits(tmp_path, template):
    with pytest.raises(SubmissionError, match="limit is 140"):
        write_submission(_good(), tmp_path / "a.tif", template, name="ok-name", note="x" * 141)
    with pytest.raises(SubmissionError, match="name"):
        write_submission(_good(), tmp_path / "a.tif", template, name="bad name!", note="x")


def test_validate_file_detects_wrong_crs(tmp_path, template):
    bad = tmp_path / "wrong_crs.tif"
    with rasterio.open(
        bad,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=from_origin(0.0, 3000.0, 100.0, 100.0),
    ) as dst:
        dst.write(_good(), 1)
    with pytest.raises(SubmissionError, match="EPSG:32611"):
        validate_file(bad, template)


def test_all_zero_submission_is_flagged_not_rejected(tmp_path, template):
    receipt = write_submission(np.zeros((H, W), np.float32), tmp_path / "z.tif", template, name="zeros-check", note="all zero")
    assert receipt["all_zero_warning"] is True


def test_template_footprint_is_respected(tmp_path):
    """Template with NaN outside a block: writer must write NaN there, values inside."""
    tpl = tmp_path / "fp_template.tif"
    band = np.full((H, W), np.nan, np.float32)
    band[5:20, 5:30] = 0.0  # footprint
    with rasterio.open(
        tpl, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", nodata=np.nan,
        crs=CRS.from_epsg(32611), transform=from_origin(0.0, 3000.0, 100.0, 100.0),
    ) as dst:
        dst.write(band, 1)
    v = np.full((H, W), 0.7, np.float32)  # values everywhere, including outside the footprint
    receipt = write_submission(v, tmp_path / "fp.tif", tpl, name="fp-test", note="footprint test")
    with rasterio.open(tmp_path / "fp.tif") as src:
        out = src.read(1)
    assert np.isnan(out[0, 0]) and np.isnan(out[29, 39])
    assert np.all(out[5:20, 5:30] == np.float32(0.7))
    assert receipt["array_stats"]["outside_footprint_cells"] == H * W - 15 * 25
    assert receipt["array_stats"]["max"] == pytest.approx(0.7, rel=1e-6)
