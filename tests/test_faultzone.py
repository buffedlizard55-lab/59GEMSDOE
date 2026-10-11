"""Lane-module sanity: segment linking and halo features on the real data."""
import numpy as np
import rasterio

from gems59 import faultzone as fz

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cat_valid_transform():
    with rasterio.open(ROOT / "data" / "official" / "labels.tif") as ds:
        cat = ds.read(1) > 0
    with rasterio.open(ROOT / "data" / "official" / "sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))
        transform = ds.transform
    return cat, valid, transform


def test_link_segments_catalogue_counts():
    cat, valid, _ = _cat_valid_transform()
    seg_lab, seg_stats = fz.link_segments(cat)
    # 2,588 linked catalogue segments (Exp 1b measurement)
    assert len(seg_stats) == 2588
    assert int(cat.sum()) == sum(s["size"] for s in seg_stats.values())
    # every catalogue pixel is assigned to exactly one segment
    assert int((seg_lab[cat] > 0).sum()) == int(cat.sum())
    # segment ids are 1..n contiguous
    assert seg_lab.max() == len(seg_stats)


def test_ingenious_record_segments_counts():
    cat, valid, transform = _cat_valid_transform()
    seg_stats, seg_pix, tmap, trace_sense = fz.ingenious_record_segments(
        ROOT / "data" / "external" / "trace_segments_utm11.csv", cat.shape, transform)
    # 1,126 INGENIOUS record segments (Exp 1b measurement)
    assert len(seg_stats) == 1126
    # seg_pix holds per-trace rasterized pixels (traces overlap, so the sum
    # exceeds the unique-pixel count); sizes match the pixel lists exactly.
    assert sum(s["size"] for s in seg_stats.values()) == \
        sum(len(a[0]) for a in seg_pix.values())
    assert 0 < int((tmap > 0).sum()) <= sum(s["size"] for s in seg_stats.values())
    # recorded sense classes
    senses = {s["sense"] for s in seg_stats.values()}
    assert senses <= {"N", "RL", "LL", "unk"}
    assert "RL" in senses and "LL" in senses and "N" in senses


def test_pixel_features_geometry():
    cat, valid, _ = _cat_valid_transform()
    visible = cat & valid
    seg_lab, seg_stats = fz.link_segments(cat)
    feats = fz.pixel_features(visible, seg_lab, seg_stats)
    d = feats["d"]
    # distance is 0 exactly on the visible fault pixels
    assert (d[visible] == 0).all()
    # off-fault pixels have d >= 1 px
    off = valid & ~visible
    assert (d[off] >= 1).all()
    # phi is an angle in [0, 90] (0 = along strike, 90 = across)
    phi = feats["phi"][off]
    assert np.isfinite(phi).all()
    assert (phi >= 0).all() and (phi <= 90).all()
    # u is 0..1 inside a segment's span, beyond-tip outside
    u = feats["u"][off]
    assert np.isfinite(u).all()
    assert (u < 0).any() and (u > 1).any()   # beyond-tip pixels exist
    # L carries the nearest segment's length (6 single-pixel fragments have
    # length_px == 0 by construction — the axis of a 1-px segment is undefined)
    L = feats["L"][off]
    assert (L >= 0).all()
    n_zero_len = sum(1 for s in seg_stats.values() if s["length_px"] <= 0)
    assert n_zero_len <= 10
    assert (L > 0).sum() > 0.99 * L.size
