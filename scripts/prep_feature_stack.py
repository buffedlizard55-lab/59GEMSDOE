"""Cache the official 19-band GeoDAWN feature stack as a compact uint8 bank.

Source: DrivenData competition 306 ``training_features.tif``
(sha256 ``4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5``,
418,912,844 B, 19 float32 bands, EPSG:32611, 3730x3292, 100 m), restored from
the owner's public mirror ``buffedlizard55-lab/GEMSDOE`` @ ``c0c06ac8``
(``data/bridge/gems-geodawn-numerical-features.tif.part-000..004``) and
re-verified byte-for-byte before use.

Two transforms per band, both rank-based so they are immune to the heavy tails
and to the ``-3.4028235e+38`` nodata sentinel:

* ``q``  -- global quantile rank inside the scored footprint (0..254)
* ``z``  -- local standardised contrast ``(x - mean_9) / std_9``, then globally
  quantile-ranked (0..254).  This is the edge / anomaly detector: a narrow
  geophysical lineament (magnetic tilt-angle ridge, gravity-gradient edge,
  detrended-elevation scarp) is exactly a local high-contrast feature.

255 marks nodata.  Output is memmappable and gitignored (``out/``).
"""
from __future__ import annotations

import json
import os
import numpy as np
import rasterio
from scipy import ndimage as ndi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(REPO, "out", "features", "training_features.tif")
OUT = os.path.join(REPO, "out", "features")
NODATA = -3.4028234663852886e+38
RANK_MAX = 254
NODATA_U8 = 255


def _rank_u8(v: np.ndarray) -> np.ndarray:
    """Average-rank of ``v`` scaled to 0..RANK_MAX (ties share a rank)."""
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, np.float64)
    r[order] = np.arange(v.size, dtype=np.float64)
    # average ties
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    if sv.size > 1:
        r = r * (RANK_MAX / (sv.size - 1))
    return np.clip(np.rint(r), 0, RANK_MAX).astype(np.uint8)


def main() -> dict:
    with rasterio.open(os.path.join(REPO, "data/bridge/sample_submission.tif")) as s:
        sub = s.read(1)
    fp = np.isfinite(sub)
    n_fp = int(fp.sum())

    with rasterio.open(FEAT) as src:
        nb = src.count
        names = [src.descriptions[i].split(" - ")[0] for i in range(nb)]
        bank = np.lib.format.open_memmap(
            os.path.join(OUT, "bank_u8.npy"), mode="w+", dtype=np.uint8,
            shape=(2 * nb, src.height, src.width))
        info = []
        for i in range(nb):
            a = src.read(i + 1).astype(np.float32)
            bad = (~np.isfinite(a)) | (a <= NODATA / 10.0)
            q = np.full(a.shape, NODATA_U8, np.uint8)
            z = np.full(a.shape, NODATA_U8, np.uint8)
            if (~bad & fp).sum() > 100:
                idx = np.flatnonzero((~bad & fp).ravel())
                q.reshape(-1)[idx] = _rank_u8(a[~bad & fp])
                # local contrast on a filled copy (nodata -> footprint median)
                med = float(np.median(a[~bad & fp]))
                b = np.where(bad, med, a)
                m1 = ndi.uniform_filter(b, size=9, mode="nearest")
                m2 = ndi.uniform_filter(b * b, size=9, mode="nearest")
                sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0)) + 1e-6
                zz = (b - m1) / sd
                z.reshape(-1)[idx] = _rank_u8(zz[~bad & fp])
            bank[2 * i] = q
            bank[2 * i + 1] = z
            info.append(dict(band=i + 1, name=names[i],
                             nodata_cells=int(bad.sum()),
                             nodata_in_footprint=int((bad & fp).sum()),
                             median=float(np.median(a[~bad & fp]))))
            del a, b, m1, m2, sd, zz, q, z
        bank.flush()

    # ---- geometric (lane) features, float32 ---------------------------------
    with rasterio.open(os.path.join(REPO, "data/bridge/existing_faults.tif")) as s:
        cat = (s.read(1) > 0) & fp
    dcat = ndi.distance_transform_edt(~cat).astype(np.float32)
    geo = np.lib.format.open_memmap(
        os.path.join(OUT, "geo_f32.npy"), mode="w+", dtype=np.float32,
        shape=(4, cat.shape[0], cat.shape[1]))
    yy, xx = np.mgrid[-15:16, -15:16]
    k15 = ((yy * yy + xx * xx) <= 225).astype(np.float32)
    yy, xx = np.mgrid[-5:6, -5:6]
    k5 = ((yy * yy + xx * xx) <= 25).astype(np.float32)
    geo[0] = dcat
    geo[1] = np.log1p(dcat)
    geo[2] = ndi.convolve(cat.astype(np.float32), k5, mode="constant")
    geo[3] = ndi.convolve(cat.astype(np.float32), k15, mode="constant")
    # distance to a mapped-fault tip (end point of a catalogue fragment)
    lab, n = ndi.label(cat, structure=np.ones((3, 3), bool))
    tips = np.zeros(cat.shape, bool)
    if n:
        for j in range(1, n + 1):
            m = lab == j
            if m.sum() < 2:
                tips |= m
                continue
            ys, xs = np.nonzero(m)
            er = ndi.binary_erosion(m, structure=np.ones((3, 3), bool))
            e = m & ~er
            if e.sum() == 0:
                e = m
            tips |= e
    geo = np.lib.format.open_memmap(os.path.join(OUT, "geo_f32.npy"), mode="r+")
    dtip = ndi.distance_transform_edt(~tips).astype(np.float32)
    np.save(os.path.join(OUT, "dtip_f32.npy"), dtip)
    geo.flush()

    out = dict(schema="gems59.feature-bank.v1",
               source_sha256="4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
               source_bytes=418912844, bands=names, bank_shape=list(bank.shape),
               footprint_cells=n_fp, rank_max=RANK_MAX, nodata_code=NODATA_U8,
               per_band=info,
               geo_names=["dcat", "log1p_dcat", "cat_dens_5", "cat_dens_15"],
               extra=["dtip_f32.npy (distance to nearest mapped-fragment tip)"])
    with open(os.path.join(REPO, "evidence", "feature_bank.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_band"}, indent=1))
    return out


if __name__ == "__main__":
    main()
