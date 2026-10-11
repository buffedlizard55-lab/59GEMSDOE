"""Historical candidate format checks and the writer's fail-closed contract.

The historical raster passes local format checks but is not cleared by the
literal full-registry uniqueness gate and must not be downloaded/submitted.
The writer rejects out-of-range values, NaN, positive mass outside the footprint,
and over-long names/notes; it never silently repairs model output.
"""
import json

import numpy as np
import pytest
import rasterio

from gems59 import gates
from gems59.submission_writer import write_submission

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "docs" / "downloads"
SAMPLE = ROOT / "data" / "official" / "sample_submission.tif"
BUILD = ROOT / "evidence" / "submission_build_all.json"


def _shipped_submission() -> Path:
    """The shipped artifact is whatever the latest build receipt says.

    Resolving it from ``evidence/submission_build_all.json`` (instead of a
    hard-coded timestamped filename) is what keeps these gates from going
    stale every time the lane rebuilds -- see ``REMAINING_WORK.md`` item 7.
    """
    assert BUILD.exists(), "missing evidence/submission_build_all.json (run build_submission.py)"
    name = json.loads(BUILD.read_text())["submission_name"]
    return DL / f"{name}-zeros.tif"


def test_shipped_submission_passes_format_gate():
    SUB = _shipped_submission()
    assert SUB.exists(), f"missing submission artifact: {SUB}"
    with rasterio.open(SAMPLE) as ds:
        footprint = np.isfinite(ds.read(1))
    rep = gates.format_report(SUB, SAMPLE, footprint=footprint)
    assert rep["ok"], rep.get("problems")
    assert rep["bands"] == 1
    assert rep["dtype"] == "float32"
    assert rep["crs"] == "EPSG:32611"
    assert rep["n_nan"] == 0
    assert rep["min"] >= 0.0 and rep["max"] <= 1.0
    assert rep["mass_outside_footprint"] == 0
    rec = json.loads(SUB.with_suffix(".json").read_text())
    assert rep["n_nonzero"] == rec["validator"]["n_nonzero"] > 0
    with rasterio.open(SUB) as ds:
        vals = np.unique(ds.read(1))
    assert set(vals) <= {0.0, 1.0}


def test_shipped_submission_has_no_dots_on_known_faults():
    with rasterio.open(_shipped_submission()) as ds:
        p = ds.read(1)
    with rasterio.open(ROOT / "data" / "official" / "labels.tif") as ds:
        cat = ds.read(1)
    assert float(p[cat > 0].sum()) == 0.0


def test_shipped_submission_sha256_matches_receipt():
    import hashlib
    SUB = _shipped_submission()
    rec = json.loads(SUB.with_suffix(".json").read_text())
    got = hashlib.sha256(SUB.read_bytes()).hexdigest()
    assert got == rec["sha256"] == rec["validator"]["sha256"]
    assert rec["note_chars"] <= 140
    assert rec["promoted"] is False


def _valid_footprint():
    with rasterio.open(SAMPLE) as ds:
        ref = ds.read(1)
    return np.isfinite(ref)


def test_writer_rejects_out_of_range_prediction(tmp_path):
    valid = _valid_footprint()
    pred = np.where(valid, 2.5, 0.0).astype(np.float32)
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, SAMPLE, valid,
                         note="t", name="t")


def test_writer_rejects_nan_inside_footprint(tmp_path):
    valid = _valid_footprint()
    pred = np.where(valid, 0.5, 0.0).astype(np.float32)
    pred[valid] = np.nan
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, SAMPLE, valid,
                         note="t", name="t")


def test_writer_rejects_mass_outside_footprint(tmp_path):
    valid = _valid_footprint()
    pred = np.where(valid, 0.0, 1.0).astype(np.float32)  # positive outside footprint
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, SAMPLE, valid,
                         note="t", name="t")


def test_writer_rejects_overlong_note(tmp_path):
    valid = _valid_footprint()
    pred = np.where(valid, 0.5, 0.0).astype(np.float32)
    with pytest.raises(ValueError):
        write_submission(tmp_path / "x.tif", pred, SAMPLE, valid,
                         note="n" * 141, name="t")


def test_writer_accepts_valid_binary_prediction(tmp_path):
    valid = _valid_footprint()
    pred = np.where(valid, 0.0, 0.0).astype(np.float32)
    ys, xs = np.nonzero(valid)
    pred[ys[:100], xs[:100]] = 1.0
    out = tmp_path / "ok.tif"
    rec = write_submission(out, pred, SAMPLE, valid, note="test note", name="test")
    assert rec["validator"]["all_checks_passed"]
    assert rec["validator"]["emitted_positive_pixels"] == 100
    assert rec["approved_for_weekly_slot"] is False
    assert (tmp_path / "ok.zip").exists()
    assert (tmp_path / "ok.json").exists()
