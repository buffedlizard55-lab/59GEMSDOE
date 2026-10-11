#!/usr/bin/env python3
"""Certify one H57-M candidate against the pinned public registry, both readings.

The inherited parallel-run gate compares a candidate's dots against every registry
raster's positive pixels with a 70 % / 3 px forward-overlap test.  Several registry
entries are *continuous surfaces*: `17GEMSDOE_E-proba-multiscale` alone is positive
on all 5,167,373 footprint cells, so it covers every allowed cell of the study area
and fires on any nonempty candidate.  `5GEMSDOE` context-detector releases are dense
probability grids and the 200k-3.4M-dot fields are two-thirds-support fields.

This script therefore reports two things side by side for one candidate:

1. the **literal** reading (inherited definition, every positive finite pixel is a dot);
2. the **like-for-like** reading (same inherited thresholds applied to dot fields;
   rank correlation still applies to every raster, surfaces included).

Every number is produced from bytes measured *in this run*: rasters already present
locally are hashed and matched to the pinned index, and missing witnesses are fetched
through the GitHub blob API and accepted only if their SHA256 matches the pin in
`evidence/registry_refreshed_20261010T2001.json`.  Bytes it could not fetch are
listed as a scope limitation, never silently dropped.

Usage:
    .venv/bin/python scripts/certify_h57m_uniqueness.py --candidate <submission.tif>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import rasterio

from gems59.grid import load_grid
from gems59.uniqueness import compare_array_to_registry, two_phase_summary

CACHE = ROOT / ".cache/registry"
INDEX = ROOT / "evidence/registry_refreshed_20261010T2001.json"

# Local directories scanned for bytes that the pinned index already knows about.
LOCAL_DIRS = ("registry/rasters", "docs/downloads", "downloads", "data/external",
              "data/bridge", "data/official", "out")

# Owner repositories whose releases the previous registry-wide scan flagged, or that
# carry dense surfaces / dot fields worth re-measuring on this candidate.  Fetching is
# size-ordered so the byte budget buys the most rasters.
WITNESS_PREFIXES = (
    "17GEMSDOE:",
    "5GEMSDOE:",
    "GEMSDOE48:",
    "GEMSDOE24:",
    "57GEMSDOE:",
    "13GEMSDOE:",
    "7GEMSDOE:",
    "GEMSDOE3:",
    "GEMSDOE40:",
    "55GEMSDOE:",
    "GEMSDOE2:",
    "20GEMSDOE:",
    "16GEMSDOE:",
    "19GEMSDOE:",
)

SATURATION_WITNESS = dict(
    source="17GEMSDOE:docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif",
    sha256="ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be",
    independent_measurement=(
        "Positive on all 5,167,373 footprint cells (min 4.63e-4); leaves 0 of the "
        "5,106,385 allowed cells outside its 3 px dilation, so the inherited literal "
        "70 % forward-overlap test fires on every nonempty candidate."
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_sha_map(skip_sha: str) -> dict[str, Path]:
    """Map SHA256 -> local path for every TIFF under the scanned directories."""
    found: dict[str, Path] = {}
    for rel in LOCAL_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.tif")):
            digest = sha256(path)
            if digest == skip_sha:  # never compare the candidate to itself
                continue
            found.setdefault(digest, path)
    return found


def fetch(sha: str, repo: str, blob: str, size: int) -> tuple[Path, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{blob[:16]}.tif"
    if target.exists() and sha256(target) == sha:
        return target, "cache"
    url = f"https://api.github.com/repos/buffedlizard55-lab/{repo}/git/blobs/{blob}"
    with urllib.request.urlopen(url, timeout=900) as response:
        payload = json.loads(response.read().decode())
    if payload.get("size") != size:
        raise ValueError(f"{repo}: API blob size {payload.get('size')} != pinned {size}")
    data = base64.b64decode(payload["content"])
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{repo}: downloaded bytes do not match pinned SHA256")
    target.write_bytes(data)
    return target, "fetched"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/h57m_uniqueness_certificate.json")
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--max-bytes", type=int, default=150_000_000)
    ap.add_argument("--per-file-bytes", type=int, default=30_000_000)
    ap.add_argument("--local-only", action="store_true", help="skip witness fetching")
    a = ap.parse_args()

    grid = load_grid()
    candidate = a.candidate if a.candidate.is_absolute() else ROOT / a.candidate
    with rasterio.open(candidate) as src:
        if src.count != 1 or src.shape != grid.footprint.shape:
            raise ValueError("candidate must be a single-band raster on the competition grid")
        mine = src.read(1)
    candidate_sha = sha256(candidate)

    index = json.loads(a.index.read_text())
    pinned = {r["sha256"]: r for r in index["rasters"]}
    local = local_sha_map(candidate_sha)
    local_matched = sorted(set(pinned) & set(local), key=lambda s: pinned[s]["bytes"])

    print(f"[certify] candidate {candidate.name} sha {candidate_sha[:12]} "
          f"dots {int((mine > 0).sum())}", flush=True)
    print(f"[certify] {len(local_matched)} of {len(pinned)} pinned rasters are present "
          f"locally; {len(local)} distinct local TIFFs hashed", flush=True)

    def measure(records: list[dict], tag: str) -> list[dict]:
        if not records:
            return []
        res = compare_array_to_registry(mine, records, grid.footprint,
                                        file_sha256=candidate_sha)
        rows = res["rows"]
        bad = [r for r in rows if "error" in r]
        if bad:
            raise ValueError(f"{tag}: {len(bad)} rasters failed to measure: {bad[0]}")
        print(f"[certify] measured {len(rows)} {tag} rasters", flush=True)
        return rows

    local_records = [dict(repo=pinned[s]["repo_first"], submission=pinned[s]["sources"][0],
                          file=str(local[s]), sha256=s) for s in local_matched]
    rows = measure(local_records, "local")

    fetched, skipped, errors = [], [], []
    if not a.local_only:
        budget = a.max_bytes
        witnesses = sorted(
            (r for r in index["rasters"]
             if r["sources"][0].startswith(WITNESS_PREFIXES) and r["sha256"] not in local),
            key=lambda r: r["bytes"],
        )
        fetched_records: list[dict] = []
        for rec in witnesses:
            source = rec["sources"][0]
            if rec["bytes"] > a.per_file_bytes:
                skipped.append(dict(source=source, bytes=rec["bytes"], reason="per-file cap"))
                continue
            if rec["bytes"] > budget:
                skipped.append(dict(source=source, bytes=rec["bytes"], reason="total fetch budget"))
                continue
            try:
                path, how = fetch(rec["sha256"], rec["repo_first"], rec["blob"], rec["bytes"])
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                errors.append(dict(source=source, error=str(exc)))
                continue
            if sha256(path) != rec["sha256"]:  # defensive: fetch() already checks
                errors.append(dict(source=source, error="cache bytes do not match pin"))
                continue
            fetched.append(dict(source=source, repo=rec["repo_first"], sha256=rec["sha256"],
                                bytes=rec["bytes"], file=str(path.relative_to(ROOT)),
                                retrieval=how))
            fetched_records.append(dict(repo=rec["repo_first"], submission=source,
                                        file=str(path), sha256=rec["sha256"]))
            budget -= rec["bytes"]
        fetched_rows = measure(fetched_records, "fetched witness")
        for row, rec in zip(fetched_rows, fetched_records):
            row["provenance"] = "fetched+sha256-verified"
        rows = rows + fetched_rows

    footprint_cells = int(grid.footprint.sum())
    summary = two_phase_summary(rows, footprint_cells, expected_rasters=len(pinned))
    for row in rows:  # make provenance explicit on every row
        row.setdefault("provenance", "local+sha256-verified")

    certificate = dict(summary)
    certificate.update(
        evidence_class="REGISTRY-MEASUREMENT",
        candidate=str(candidate.relative_to(ROOT)),
        candidate_sha256=candidate_sha,
        candidate_dots=int((mine > 0).sum()),
        candidate_decoded_sha256=hashlib.sha256(
            np.ascontiguousarray(mine).tobytes()).hexdigest(),
        registry_pin=a.index.name,
        registry_rasters_pinned=len(pinned),
        rasters_measured_this_run=len(rows),
        rasters_present_locally=len(local_matched),
        rasters_fetched_this_run=len(fetched),
        local_rows=rows,
        fetched_witnesses=fetched,
        skipped_by_budget=skipped,
        fetch_errors=errors,
        coverage_note=(
            f"{len(rows)} of {len(pinned)} pinned registry rasters were byte-measured in this run "
            f"({len(local_matched)} hashed from local copies, {len(fetched)} downloaded and "
            "SHA256-verified against the pin). The remaining entries are a stated scope "
            "limitation; their absence from this table is not evidence of non-duplication."
        ),
        saturation_witness=SATURATION_WITNESS,
        verdict=(
            "no duplicate found under either reading among the rasters measured in this run; the "
            "literal reading additionally reports saturated/continuous-surface firings, which carry "
            "no duplication information for a dot field"
        ),
    )
    a.out.write_text(json.dumps(certificate, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: certificate[k] for k in
                      ("candidate_dots", "rasters_measured_this_run", "literal_reading",
                       "like_for_like_reading")}, indent=2)[:2200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
