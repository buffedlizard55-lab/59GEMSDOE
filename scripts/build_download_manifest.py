#!/usr/bin/env python3
"""Write ``docs/downloads/manifest.json`` from the release receipts.

The manifest is the machine-readable answer to the standing brief's question
"is it OK to download and submit this TIF?": one row per candidate that is
published in ``docs/downloads/``, hashes and byte counts recomputed from the
bytes actually on disk, and the download/submit statuses copied from the same
receipts the status pages quote.

Four candidates exist after the 2026-10-11 merge of the parallel lanes, and the
manifest keeps one row for every one of them, current first:

* this lane's off-catalogue discriminant (``arena/fba37db3``),
* the H59-I intensity lattice (archived on this site),
* the parallel lane's ridge-coherence consensus (``arena/1ee0553d``),
* the replication lane's independent intensity run (``arena/1ac83927``).

Each row also names its uniqueness receipt, so ``scripts/check_site.py`` can
re-check the registry leg of every published file rather than only the front
one.  Fail-closed: a missing file, a hash mismatch, a missing receipt or a
missing status stops the build.
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
          statuses: dict | None = None, uniqueness_receipt: str | None = None,
          run_card: str | None = None, lane: str | None = None,
          shared_template_gate_passed: bool = True) -> dict:
    receipt_file = ROOT / receipt_path
    if not receipt_file.is_file():
        raise SystemExit(f"missing receipt {receipt_path}")
    receipt = json.loads(receipt_file.read_text())
    validator = receipt.get("validator") or {}
    statuses = statuses or {}
    note = receipt.get("note") or receipt.get("submission_note")
    dots = (receipt.get("dots") or validator.get("emitted_positive_pixels")
            or receipt.get("n_dots") or receipt.get("emitted"))
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
    if files[0]["bytes"] != receipt["bytes"]:
        raise SystemExit(f"{name}: TIF byte count does not match its receipt")
    download_status = receipt.get("download_status") or statuses.get("download_status")
    submit_status = receipt.get("submit_status") or statuses.get("submit_status")
    if not download_status or not submit_status:
        raise SystemExit(f"{name}: no download/submit status in the receipt or its status source")
    if uniqueness_receipt and not (ROOT / uniqueness_receipt).is_file():
        raise SystemExit(f"{name}: uniqueness receipt {uniqueness_receipt} is missing")
    return dict(
        candidate=name, receipt=receipt_path, superseded_by=superseded_by,
        sha256=receipt["sha256"], bytes=receipt["bytes"],
        dots=dots,
        spacing_px=receipt.get("spacing_px") or receipt.get("min_separation_px")
        or receipt.get("min_separation_px"),
        on_catalogue_dots=(receipt.get("on_catalogue_dots")
                           or validator.get("on_catalogue_positive_pixels")
                           or receipt.get("dots_on_or_within_2px_of_catalogue")),
        note=note, note_chars=receipt.get("note_chars") or (len(note) if note else None),
        validator=validator, files=files,
        download_status=download_status, submit_status=submit_status,
        uniqueness_receipt=uniqueness_receipt, run_card=run_card, lane=lane,
        shared_template_gate_passed=shared_template_gate_passed,
        evidence_class="SITE-DELIVERY MANIFEST (hashes recomputed from the bytes in this directory)",
    )


def main() -> int:
    card_path = ROOT / "evidence/run_card.json"
    card = json.loads(card_path.read_text())
    current = entry("gems59-h59j-offcat-propen-30k-20261011", "evidence/submission_h59j.json",
                    uniqueness_receipt="evidence/uniqueness_h59j_offcat.json",
                    run_card="evidence/run_card.json",
                    lane="arena/fba37db3 (off-catalogue discriminant)")
    lane_h59i = (card.get("lane_h59i") or {}).get("submission") or {}
    h59i = entry("gems59-h59i-intensity-lattice-20261011T000000Z",
                 "downloads/gems59-h59i-intensity-lattice-20261011T000000Z.json",
                 statuses=dict(download_status=lane_h59i.get("download_status"),
                               submit_status=lane_h59i.get("submit_status")),
                 uniqueness_receipt="evidence/uniqueness_h59i.json",
                 lane="arena/f8d022fd (intensity lane)")
    coherence = entry("gems59-h59j-coherence-consensus-dotted-20261011T043000Z",
                      "downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.json",
                      statuses=dict(
                          download_status="OK (format-valid; registry screen clean: no duplication "
                                          "identified over all 695 pinned rasters)",
                          submit_status="NO - not cleared for a weekly slot: the writer receipt records "
                                        "approved_for_weekly_slot=False and promoted=False; its published "
                                        "0.24-0.29 expectation is a PROJECTION, not a clearance"),
                      uniqueness_receipt="evidence/uniqueness_h59j.json",
                      run_card="docs/data/run_card.json",
                      lane="arena/1ee0553d (ridge coherence consensus)")
    replication = entry("59gemsdoe-intensity-lane-50651dots-s3-t33333-20261011T005447Z-b25dd620-zeros",
                        "docs/downloads/59gemsdoe-intensity-lane-50651dots-s3-t33333-20261011T005447Z-b25dd620-zeros.json",
                        statuses=dict(
                            download_status="OK (format-valid; registry screen clean: no duplication "
                                            "identified in its own screen)",
                            submit_status="NO - its own run card records okay_to_submit=False with verdict "
                                          "negative (evidence/run_card_lane_v1.json)"),
                        uniqueness_receipt="evidence/uniqueness_lane_v1.json",
                        run_card="evidence/run_card_lane_v1.json",
                        lane="arena/1ac83927 (intensity lane, replication)")
    h59a = entry("gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros",
                 "docs/downloads/gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.json",
                 statuses=dict(
                     download_status="OK by its own 12/12 validator, BUT the shared template gate flags its "
                                     "nodata tag (IR-H59J-15) -- download is safe; upload follows the owner's "
                                     "call like every other file here",
                     submit_status="NO clearance claimed by its own receipt (note says UNSCORED); the lane's page "
                                   "puts the slot question to the owner, who has not promoted it"),
                 uniqueness_receipt="docs/downloads/gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.json",
                 run_card="docs/h59a.html",
                 lane="arena/1ba7bc3e (intensity zone + break-even tiling, H59-A)",
                 shared_template_gate_passed=False)
    manifest = dict(
        evidence_class="SITE-DELIVERY MANIFEST (hashes recomputed from the bytes in this directory)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        current_candidate=current["candidate"],
        how_to_submit="docs/executive-summary.html",
        status_pages="docs/index.html (download), docs/method.html (method & evidence)",
        legend=dict(
            download_status="whether the file is safe to download as a candidate TIF",
            submit_status="whether the owner's selector step has cleared it for a weekly slot; "
                          "no slot is spent by this project",
            uniqueness_receipt="the registry screen that clears (or fails) the download badge for this file"),
        candidates=[current, h59i, coherence, replication, h59a],
    )
    out = DOCS / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")

    # keep the published run card's candidate list in step with the manifest --
    # the manifest is the single source of truth for "what is published", and
    # the card must not look as if only one lane's files exist.
    card["candidates"] = [
        dict(name=c["candidate"], dots=c["dots"], lane=c["lane"], receipt=c["receipt"],
             download_status=c["download_status"], submit_status=c["submit_status"],
             shared_template_gate_passed=c["shared_template_gate_passed"])
        for c in manifest["candidates"]]
    card["candidate_count"] = len(manifest["candidates"])
    card["candidate_list_note"] = (
        "Every published candidate, mirroring docs/downloads/manifest.json (the authoritative index with "
        "hashes). Rows are ordered current-first; this lane's candidate is the current one.")
    for target in ("evidence/run_card.json", "docs/data/run_card.json"):
        (ROOT / target).write_text(json.dumps(card, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)} with {len(manifest['candidates'])} candidates "
          f"(run card candidate list mirrored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
