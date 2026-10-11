#!/usr/bin/env python3
"""Write ``docs/downloads/manifest.json`` from the release receipts.

The manifest is the machine-readable answer to the standing brief's question
"is it OK to download and submit this TIF?": one row per candidate, hashes and
byte counts recomputed from the bytes on disk, and the download/submit statuses
copied from the same receipts the status pages quote.  Fail-closed: a missing
file, a hash mismatch or a missing receipt stops the build.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/downloads"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(name: str, receipt_path: str, *, superseded_by: str | None = None,
          statuses: dict | None = None) -> dict:
    receipt = json.loads((ROOT / receipt_path).read_text())
    validator = receipt.get("validator") or {}
    statuses = statuses or {}
    files = []
    for suffix, role in (("tif", "submission GeoTIFF (single band, float32, values in [0, 1])"),
                         ("zip", "ZIP containing exactly that one GeoTIFF"),
                         ("json", "writer receipt (format + validator output; not a competition score)")):
        path = DOCS / f"{name}.{suffix}"
        if not path.is_file():
            raise SystemExit(f"missing {path.name} in docs/downloads/")
        files.append(dict(name=path.name, role=role, bytes=path.stat().st_size,
                          sha256=sha256(path)))
    if files[0]["sha256"] != receipt["sha256"]:
        raise SystemExit(f"{name}: TIF sha256 does not match its receipt")
    download_status = receipt.get("download_status") or statuses.get("download_status")
    submit_status = receipt.get("submit_status") or statuses.get("submit_status")
    if not download_status or not submit_status:
        raise SystemExit(f"{name}: no download/submit status in the receipt or its status source")
    return dict(
        candidate=name, receipt=receipt_path, superseded_by=superseded_by,
        sha256=receipt["sha256"], bytes=receipt["bytes"],
        dots=receipt.get("dots") or validator.get("emitted_positive_pixels"),
        spacing_px=receipt.get("spacing_px"),
        on_catalogue_dots=receipt.get("on_catalogue_dots",
                                      validator.get("on_catalogue_positive_pixels")),
        note=receipt.get("note") or receipt.get("submission_note"), note_chars=receipt.get("note_chars"),
        validator=validator, files=files,
        download_status=download_status, submit_status=submit_status,
        evidence_class="SITE-DELIVERY MANIFEST (hashes recomputed from the bytes in this directory)",
    )


def main() -> int:
    current = entry("gems59-h59j-offcat-propen-30k-20261011", "evidence/submission_h59j.json")
    card = json.loads((ROOT / "evidence/run_card.json").read_text())
    lane_h59i = (card.get("lane_h59i") or {}).get("submission") or {}
    previous = entry("gems59-h59i-intensity-lattice-20261011T000000Z",
                     "downloads/gems59-h59i-intensity-lattice-20261011T000000Z.json",
                     superseded_by="gems59-h59j-offcat-propen-30k-20261011",
                     statuses=dict(download_status=lane_h59i.get("download_status"),
                                   submit_status=lane_h59i.get("submit_status")))
    manifest = dict(
        evidence_class="SITE-DELIVERY MANIFEST (hashes recomputed from the bytes in this directory)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        current_candidate=current["candidate"],
        how_to_submit="docs/executive-summary.html",
        status_pages="docs/index.html (download), docs/method.html (method & evidence)",
        legend=dict(
            download_status="whether the file is safe to download as a candidate TIF",
            submit_status="whether the owner's selector step has cleared it for a weekly slot; "
                          "no slot is spent by this project"),
        candidates=[current, previous],
    )
    out = DOCS / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)} with {len(manifest['candidates'])} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
