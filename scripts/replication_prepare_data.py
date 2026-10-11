#!/usr/bin/env python3
"""Restore the hash-pinned public bridge into this repository's layout.

The 19-band feature raster is 419 MB, so it is *not* committed to Git: it is
assembled atomically from five separately hashed parts of the public bridge
repository through the permitted GitHub API (``gh``), and lives in ``.cache/``,
which is excluded from workspace snapshots.  Everything else is small enough to
keep in the tree and is simply hash-verified here.

These hashes authenticate *transport* from a public mirror, not an official
DrivenData provenance receipt.  No login or token is read or stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_REPO = "buffedlizard55-lab/6GEMSDOE"
BRIDGE_COMMIT = "e2fe3f41c6f5dd2dcb2fc91958ee67698f114ada"

PIN = {
    "data/bridge/labels.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/bridge/existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/bridge/sample_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
    "data/bridge/example_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
    "data/external/sgmc_faults_100m.tif": "26d142c4c93282cd94f6950ab96f22aeff59fbbea523d43d662e76fa1b161b5c",
    # IR-59-SGMC-01: the derived layer in the published bridge is byte-identical to
    # the base SGMC layer (it is not a "minus the mapped union" difference layer), so
    # the pin below is the hash the file actually has, not the sibling's label.
    "data/external/derived_sgmc_faults_100m_u8.tif":
        "26d142c4c93282cd94f6950ab96f22aeff59fbbea523d43d662e76fa1b161b5c",
    "data/external/trace_segments_utm11.csv":
        "c5edf9df3e0e1413c52784884fc38e5b9ca54627b27c6f8baf7713e908f48513",
}
FEATURES = ".cache/training_features.tif"
FEATURES_SHA = "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
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


def fetch_raw(source: str, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = (f"repos/{BRIDGE_REPO}/contents/{source}?ref={BRIDGE_COMMIT}")
    with temporary.open("wb") as out:
        subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw", url],
                       stdout=out, check=True, timeout=600)
    if sha256(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"hash mismatch after fetch: {source}")
    temporary.replace(destination)


def assemble_features() -> dict:
    cache = ROOT / ".cache" / "source_parts"
    receipts = []
    for i, (size, digest) in enumerate(PARTS):
        part = cache / f"training_features.tif.part-{i:03d}"
        if not (part.is_file() and part.stat().st_size == size and sha256(part) == digest):
            fetch_raw(f"data/bridge/training_features.tif.part-{i:03d}", part, digest)
        receipts.append(dict(part=part.name, bytes=size, sha256=digest))
    out = ROOT / FEATURES
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tif.assembling")
    with temporary.open("wb") as sink:
        for i in range(len(PARTS)):
            sink.write((cache / f"training_features.tif.part-{i:03d}").read_bytes())
    if sha256(temporary) != FEATURES_SHA:
        temporary.unlink(missing_ok=True)
        raise ValueError("assembled feature raster hash mismatch")
    temporary.replace(out)
    return dict(source_repo=BRIDGE_REPO, source_commit=BRIDGE_COMMIT, parts=receipts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download missing pins via gh")
    args = ap.parse_args()
    receipt = dict(evidence_class="DATA-MEASUREMENT", files=[],
                   provenance="public bridge mirror; not an organizer receipt")
    ok = True
    if args.fetch and not (ROOT / FEATURES).is_file():
        receipt["assembly"] = assemble_features()
        print("[data] assembled 19-band feature stack", flush=True)
    for relative, digest in PIN.items():
        path = ROOT / relative
        if not path.is_file() and args.fetch:
            try:
                source = relative.replace("data/bridge/", "data/bridge/")
                fetch_raw(source, path, digest)
            except Exception as error:  # pragma: no cover - network path
                ok = False
                receipt["files"].append(dict(path=relative, verified=False, error=str(error)))
                continue
        verified = path.is_file() and sha256(path) == digest
        ok &= verified
        receipt["files"].append(dict(
            path=relative, verified=verified, bytes=path.stat().st_size if path.is_file() else None,
            sha256=sha256(path) if path.is_file() else None, expected_sha256=digest))
        print(f"[data] {'OK ' if verified else 'MISSING/BAD '}{relative}", flush=True)
    features = ROOT / FEATURES
    features_ok = features.is_file() and sha256(features) == FEATURES_SHA
    receipt["feature_stack"] = dict(path=FEATURES, verified=features_ok,
                                    sha256=sha256(features) if features.is_file() else None,
                                    expected_sha256=FEATURES_SHA,
                                    bytes=features.stat().st_size if features.is_file() else None,
                                    note="419 MB; lives in .cache/, not committed")
    ok &= features_ok
    receipt["all_pins_verified"] = bool(ok)
    (ROOT / "evidence" / "data_preparation_lane1ac8.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("[data] " + ("ALL PINS VERIFIED" if ok else "PIN CHECK INCOMPLETE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
