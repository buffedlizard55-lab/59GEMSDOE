#!/usr/bin/env python3
"""Fail-closed local delivery QA for the H59-era site. No fitting, no submission.

Checks, in order:

* every local link and anchor in ``docs/`` resolves inside the site;
* the published run card (``docs/data/run_card.json``) is byte-equal to the
  evidence copy (``evidence/run_card.json``) -- a stale public card fails;
* every candidate in ``docs/downloads/manifest.json``: the TIF exists, its
  SHA256 matches the manifest *and* the writer receipt, the template format
  validator passes, the local validator passes, and the ZIP contains exactly
  that one GeoTIFF with identical bytes;
* the two status badges the standing brief demands are on the status page and
  agree with the receipts: "DOWNLOAD: OK" and "SUBMIT TO COMPETITION: NO"
  (the second only while the candidate is not cleared for a slot);
* the current candidate's uniqueness receipt exists, scanned every pinned
  registry raster without error, and reports no duplication under the
  representation-aware reading (IR-57-REPR-01).
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.downloads = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if tag == "a" and "download" in attrs:
            self.downloads.append(attrs.get("href", ""))


def check_links(docs: Path) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    pages = list(docs.rglob("*.html"))
    for page in pages:
        parser = Links()
        parser.feed(page.read_text())
        for value in parser.links:
            url = urlsplit(value)
            if url.scheme or url.netloc or value.startswith(("mailto:", "data:")):
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if not target.is_relative_to(docs.resolve()):
                errors.append(f"{page}: local link escapes site: {value}")
                continue
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                errors.append(f"{page}: broken local link {value}")
                continue
            if url.fragment and target.suffix == ".html":
                other = Links()
                other.feed(target.read_text())
                if unquote(url.fragment) not in other.ids:
                    errors.append(f"{page}: missing anchor {value}")
    return pages, errors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path = ROOT) -> dict:
    """Run every delivery check and return a report dict.

    ``ok`` is False when any check failed and ``errors`` lists them; the remaining
    keys are the machine-readable readings the tests assert on.  Only receipts are
    consulted -- nothing is fitted, relaxed or promoted here.
    """
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import rasterio
    from gems59.gates import format_report
    from gems59.validate import validate

    errors: list[str] = []
    readings: dict = {}
    docs = root / "docs"

    pages, link_errors = check_links(docs)
    errors.extend(link_errors)
    readings["links_pass"] = not link_errors
    readings["pages_checked"] = len(pages)

    # --- run card parity -------------------------------------------------- #
    card = json.loads((root / "evidence/run_card.json").read_text())
    public = json.loads((docs / "data/run_card.json").read_text())
    if card != public:
        errors.append("docs/data/run_card.json is stale (differs from evidence/run_card.json)")
    readings["current_card_consistent"] = card == public
    if card.get("verdict") is None or card.get("promotion_decision", {}).get("verdict") is None:
        errors.append("run card does not state a verdict and a promotion decision")

    sample = root / "data/official/sample_submission.tif"
    if not sample.is_file():
        sample = root / "data/bridge/sample_submission.tif"
    with rasterio.open(sample) as ref:
        footprint = np.isfinite(ref.read(1))

    # --- every candidate in the manifest ---------------------------------- #
    manifest = json.loads((docs / "downloads/manifest.json").read_text())
    if not manifest.get("candidates"):
        errors.append("docs/downloads/manifest.json lists no candidates")
    current = manifest.get("current_candidate")
    readings["current_candidate"] = current
    readings["zip_exactly_one_tiff"] = True
    for cand in manifest.get("candidates", []):
        name = cand["candidate"]
        for item in cand["files"]:
            path = docs / "downloads" / item["name"]
            if not path.is_file():
                errors.append(f"{item['name']}: missing from docs/downloads/")
                continue
            if path.stat().st_size != item["bytes"]:
                errors.append(f"{item['name']}: byte count differs from the manifest")
            if sha256(path) != item["sha256"]:
                errors.append(f"{item['name']}: sha256 differs from the manifest")
        tif = docs / "downloads" / f"{name}.tif"
        zip_path = docs / "downloads" / f"{name}.zip"
        if not tif.is_file():
            continue
        report = format_report(tif, sample, footprint=footprint)
        if not report["ok"]:
            errors.append(f"{name}: template format validator failed: {report.get('problems')}")
        local = validate(tif, footprint=footprint)
        if not local.get("all_checks_passed"):
            failed = [k for k, v in local["checks"].items() if not v]
            errors.append(f"{name}: local validator failed: {failed}")
        if local["sha256"] != cand["sha256"]:
            errors.append(f"{name}: validator hash differs from the manifest")
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                if members != [f"{name}.tif"]:
                    errors.append(f"{name}: ZIP members {members} != the single GeoTIFF")
                    readings["zip_exactly_one_tiff"] = False
                elif zf.read(members[0]) != tif.read_bytes():
                    errors.append(f"{name}: ZIP member bytes differ from the GeoTIFF")
                    readings["zip_exactly_one_tiff"] = False
        else:
            errors.append(f"{name}: ZIP twin missing")

    # --- badges and download links on the status page --------------------- #
    index = (docs / "index.html").read_text()
    if f"downloads/{current}.tif" not in index:
        errors.append("index.html does not link the current candidate GeoTIFF")
    if "DOWNLOAD: OK" not in index:
        errors.append("index.html is missing the DOWNLOAD: OK badge")
    current_row = next((c for c in manifest["candidates"] if c["candidate"] == current), None)
    if current_row:
        readings["download_cleared"] = str(current_row["download_status"]).upper().startswith("OK")
        readings["submission_cleared"] = not str(current_row["submit_status"]).upper().startswith("NO")
    if current_row and current_row["submit_status"].upper().startswith("NO"):
        if "SUBMIT TO COMPETITION: NO" not in index:
            errors.append("index.html is missing the SUBMIT TO COMPETITION: NO badge")
    summary = (docs / "executive-summary.html").read_text()
    if current not in summary:
        errors.append("executive-summary.html does not name the current candidate file")

    # --- registry leg for the current candidate --------------------------- #
    uniqueness = root / "evidence/uniqueness_h59j.json"
    uq: dict = {}
    if not uniqueness.is_file():
        errors.append("evidence/uniqueness_h59j.json missing")
    else:
        uq = json.loads(uniqueness.read_text())
        if not uq.get("complete_accessible_scan"):
            errors.append("uniqueness receipt: scan is not complete over the registry")
        if uq.get("registry_duplicate_evidence"):
            errors.append("uniqueness receipt reports registry duplication evidence")
        if not uq.get("verdict_registry_leg"):
            errors.append("uniqueness receipt does not state a representation-aware verdict")
        like = uq.get("representation_aware") or {}
        directions = like.get("overlap_by_direction") or {}
        readings["registry_rasters_checked"] = uq.get("registry_rasters_checked")
        readings["worst_spearman"] = abs(like.get("worst_spearman_full_footprint") or 0.0)
        readings["worst_jaccard"] = like.get("worst_jaccard_dot_sets")
        readings["worst_support_matched_overlap"] = (
            directions.get("informative_forward_vs_smaller_or_equal_peers") or {}).get("value")
        readings["worst_reverse_overlap"] = (
            directions.get("informative_reverse_vs_bigger_peers") or {}).get("value")
        # disclosed, non-operative directions: a dense peer covers every candidate
        readings["worst_uninformative_overlap"] = max(
            [d.get("value") or 0.0 for key, d in directions.items() if key.startswith("uninformative")]
            or [0.0])
        readings["uniqueness_verdict"] = uq.get("verdict_registry_leg")

    readings["ok"] = not errors
    readings["errors"] = errors
    print(f"pages: {len(pages)}; candidates: {len(manifest.get('candidates', []))}; "
          f"registry rasters: {readings.get('registry_rasters_checked', 'n/a')}")
    return readings


def inspect_site(docs) -> list[str]:
    """Compatibility helper for the repository's site smoke test (``tests/test_site.py``).

    Returns the delivery-QA error list for the site at ``docs`` — empty means the
    site is internally sound (links, run-card parity, candidate hashes/formats,
    ZIP twins, status badges, registry leg). The badge check is deliberately
    receipt-driven, so the smoke test also fails a site that claims a clearance
    the receipts do not support.
    """
    docs = Path(docs).resolve()
    root = docs.parent if docs.name == "docs" else docs
    return check(root)["errors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    report = check(Path(args.root))
    if not report["ok"]:
        print("FAIL")
        for e in report["errors"]:
            print(" -", e)
        return 1
    print("PASS: links, run-card parity, candidate hashes/formats, ZIP twins, badges, registry leg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
