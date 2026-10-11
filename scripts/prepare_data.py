#!/usr/bin/env python3
"""Restore the hash-pinned public bridge through the permitted GitHub API.

This verifies bridge bytes, NOT an independent DrivenData provenance receipt.
The large feature raster is assembled atomically from five separately hashed
parts. Downloads and band caches are ignored by Git. No login/token is read or
stored here: gh uses Arena's existing GitHub connection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_REPO = "buffedlizard55-lab/6GEMSDOE"
BRIDGE_COMMIT = "e2fe3f41c6f5dd2dcb2fc91958ee67698f114ada"
PIN = {
    "data/official/labels.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/official/existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/official/sample_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
    "data/official/training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "data/external/sgmc_faults_100m.tif": "26d142c4c93282cd94f6950ab96f22aeff59fbbea523d43d662e76fa1b161b5c",
    "data/external/derived_sgmc_faults_100m.tif": "643cbe992ef4ba37588fb469163ed8291e3ceb23d6c1f78a3cfaa462430c2da0",
    "data/external/trace_segments_utm11.csv": "c5edf9df3e0e1413c52784884fc38e5b9ca54627b27c6f8baf7713e908f48513",
    "data/external/qfault_attributes.csv": "3b8745f0086a57b8255f6679afa1bf1168c10b5573491666c75cf53217029507",
}
PARTS = [
    (94371840, "0a330f8951af6c921029e25c84a579319d2db554d62d30d894d6ddc97f98cff7"),
    (94371840, "3c98037b2c997e3bbfcdfc2d9a982e8b820a77410dd7404b05cdb80594922c50"),
    (94371840, "c375c4dbc40c59bbaece572b5e348700b75935f9a30c879c82b6417e0836f31c"),
    (94371840, "b164159e6d0cb2595bc9f63a948af2646b124114a9c5921880c7516092137320"),
    (41425484, "fa0a6f9c936fac1d6f20ca37f5929b2d60bf7a80f3d477dcab886f941aee2696"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(repo: str, source: str, ref: str, destination: Path,
              expected: str, expected_bytes: int | None = None) -> None:
    if destination.exists() and sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    endpoint = f"repos/{repo}/contents/{source}?ref={ref}"
    try:
        with temporary.open("wb") as stream:
            result = subprocess.run(
                ["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw+json"],
                stdout=stream, stderr=subprocess.PIPE, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(f"GitHub API failed for {source}: {result.stderr.decode()[:200]}")
        if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
            raise ValueError(f"download size mismatch: {source}")
        if sha256(temporary) != expected:
            raise ValueError(f"download hash mismatch: {source}; original pin was NOT replaced")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def assemble_features(root: Path) -> dict:
    cache = root / ".cache" / "source_parts"
    paths = [cache / f"gems-geodawn-numerical-features.tif.part-{i:03d}" for i in range(5)]
    def download(i):
        size, digest = PARTS[i]
        fetch_raw(BRIDGE_REPO, f"data/bridge/{paths[i].name}", BRIDGE_COMMIT,
                  paths[i], digest, size)
        return dict(part=paths[i].name, bytes=size, sha256=digest)
    with ThreadPoolExecutor(max_workers=3) as pool:
        part_receipts = list(pool.map(download, range(5)))
    cache_target = root / ".cache" / "training_features.tif"
    cache_target.parent.mkdir(parents=True, exist_ok=True)
    target = root / "data/official/training_features.tif"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_target.with_suffix(".tif.partial")
    try:
        with temporary.open("wb") as out:
            for part in paths:
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, out, length=1 << 20)
        if sha256(temporary) != PIN["data/official/training_features.tif"]:
            raise ValueError("reassembled feature stack does not match its pinned digest")
        os.replace(temporary, cache_target)
        target.unlink(missing_ok=True)
        target.symlink_to(os.path.relpath(cache_target, target.parent))
    finally:
        temporary.unlink(missing_ok=True)
    return dict(source_repo=BRIDGE_REPO, source_commit=BRIDGE_COMMIT, parts=part_receipts)


def cache_bands(root: Path) -> dict:
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    feature = root / "data/official/training_features.tif"
    cache = root / ".cache/features"
    cache.mkdir(parents=True, exist_ok=True)
    bands = []
    with rasterio.open(feature) as src, rasterio.open(root / "data/official/sample_submission.tif") as sample:
        if src.shape != sample.shape or src.crs != sample.crs or src.transform != sample.transform:
            raise ValueError("feature stack does not match the bridge reference grid")
        for number, description in enumerate(src.descriptions, 1):
            path = cache / f"band-{number:02d}.npy"
            a = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=src.shape)
            n_missing = 0
            for y in range(0, src.height, 256):
                h = min(256, src.height - y)
                b = src.read(number, window=Window(0, y, src.width, h), masked=True).filled(np.nan).astype("float32")
                # GDAL's Float32-min sentinel is not a real physical measurement.
                b[(~np.isfinite(b)) | (np.abs(b) > 1e30)] = np.nan
                n_missing += int((~np.isfinite(b)).sum())
                a[y:y+h] = b
            a.flush()
            del a
            bands.append(dict(band=number, description=description, cache_file=str(path.relative_to(root)),
                              missing_pixels=n_missing, sha256=sha256(path)))
        report = dict(evidence_class="DATA-MEASUREMENT", source_sha256=sha256(feature),
                      shape=list(src.shape), crs=str(src.crs), transform=list(src.transform)[:6],
                      count=src.count, bands=bands,
                      provenance="hash-verified third-party bridge; official origin not independently authenticated")
    (root / "evidence").mkdir(exist_ok=True)
    (root / "evidence/feature_cache.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--cache-bands", action="store_true")
    a = ap.parse_args()
    receipt = dict(evidence_class="DATA-MEASUREMENT", files=[], provenance="public bridge, not organizer receipt")
    ok = True
    for relative, digest in PIN.items():
        path = ROOT / relative
        try:
            if not path.exists() and a.fetch:
                if relative.endswith("training_features.tif"):
                    receipt["assembly"] = assemble_features(ROOT)
                elif relative.startswith("data/official/"):
                    original = "example_submission.tif" if "sample_submission" in relative else "existing_faults.tif"
                    fetch_raw(BRIDGE_REPO, f"data/bridge/{original}", BRIDGE_COMMIT, path, digest)
            matches = path.exists() and sha256(path) == digest
            row = dict(path=relative, sha256=sha256(path) if path.exists() else None,
                       expected_sha256=digest, verified=matches,
                       bytes=path.stat().st_size if path.exists() else None)
            receipt["files"].append(row)
            ok &= matches
            print(f"[data] {'OK' if matches else 'MISSING/BAD'} {relative}", flush=True)
        except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            ok = False
            receipt["files"].append(dict(path=relative, verified=False, error=str(error)))
            print(f"[data] FAILED {relative}: {error}", file=sys.stderr, flush=True)
    receipt["all_pins_verified"] = bool(ok)
    (ROOT / "evidence/data_preparation.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if ok and a.cache_bands:
        report = cache_bands(ROOT)
        print(f"[data] cached {report['count']} bands (ignored by Git)")
    print("[data] " + ("ALL PINS VERIFIED" if ok else "PIN CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
