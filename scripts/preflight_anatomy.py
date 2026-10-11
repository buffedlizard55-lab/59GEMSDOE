#!/usr/bin/env python3
"""Fail-closed pre-placement gate: an immutable prior can block ALL nonempty dots.

This is not a new submission generator and does not alter the 70% rule. One
universal witness is enough to stop; no private definition of a soft 'dot'.
The witness is pinned in the existing indexed public registry AND checked at
its current GitHub main path. This proves an impossibility for the literal
rule, not that every private or future submission is in the index.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import SHA256_EXISTING_FAULTS, SHA256_SAMPLE_SUBMISSION, load_grid
from gems59.uniqueness import OVERLAP_LIMIT, saturation_certificate
from refresh_registry import OWNER, fetch_blob

INDEX = ROOT / "evidence/registry_refreshed_20261010T2001.json"
WITNESS_SHA = "ab0a0a62eecf066a82713b09dd49f0f638a91fa3dd81f54cc34ae89afa3872be"
WITNESS_SOURCE = "17GEMSDOE:docs/downloads/17GEMSDOE_E-proba-multiscale_20260930T044527Z.tif"
OUT = ROOT / "evidence/preflight_anatomy.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    reference = {
        "data/bridge/sample_submission.tif": SHA256_SAMPLE_SUBMISSION,
        "data/bridge/existing_faults.tif": SHA256_EXISTING_FAULTS,
    }
    for name, digest in reference.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"bridge reference pin mismatch: {name}")
    grid = load_grid()
    index = json.loads(INDEX.read_text())
    recs = [r for r in index["rasters"] if r["sha256"] == WITNESS_SHA]
    if len(recs) != 1 or WITNESS_SOURCE not in recs[0]["sources"]:
        raise ValueError("pinned witness absent/ambiguous in indexed public registry")
    record = recs[0]
    repo, path = WITNESS_SOURCE.split(":", 1)
    endpoint = f"repos/{OWNER}/{repo}/contents/{path}?ref=main"
    remote = json.loads(subprocess.run(["gh", "api", endpoint], check=True,
                                       capture_output=True, timeout=90).stdout)
    if remote["sha"] != record["blob"] or remote["size"] != record["bytes"]:
        raise ValueError("prior witness has changed/vanished on current public main")
    cache = ROOT / ".cache/registry"
    cache.mkdir(parents=True, exist_ok=True)
    _, issue = fetch_blob(record, cache)
    if issue is not None:
        raise ValueError(f"prior witness grid mismatch: {issue}")
    witness = cache / f"{record['blob']}.tif"
    if sha(witness) != WITNESS_SHA:
        raise ValueError("prior witness SHA256 mismatch")
    certificate = saturation_certificate(
        witness, grid.footprint, grid.catalogue,
        expected_grid=(grid.shape, grid.crs, grid.transform),
    )
    blocked = bool(certificate["universal_overlap_blocker"] and 1.0 > OVERLAP_LIMIT)
    return {
        "evidence_class": "REGISTRY-MEASUREMENT; not a HOLDOUT-DTI or organizer score",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": remote["html_url"], "source_blob_sha1": record["blob"],
        "registry_index": str(INDEX.relative_to(ROOT)),
        "registry_index_sha256": sha(INDEX), "registry_entries_in_pinned_index": len(index["rasters"]),
        "reference_sha256": reference, "witness_sha256": WITNESS_SHA,
        "certificate": certificate,
        "literal_preplacement_gate": "STOP" if blocked else "UNRESOLVED — run full registry gate",
        "implication": ("Every nonempty prediction supported within the scored footprint and "
                        "off the mapped catalogue has 100% of its positive pixels within "
                        "3 px of this earlier positive raster. No production dots may be "
                        "placed under the current literal rule." if blocked else
                        "This witness alone does not establish a universal block; it does "
                        "not certify uniqueness against the rest of the registry."),
        "scope": "One SHA/grid-verified public-main prior; suffices for STOP only. "
                 "Not an organizer submission registry or a full rescan.",
    }


def main() -> int:
    try:
        result = run()
    except Exception as exc:
        result = {"evidence_class": "PREFLIGHT ERROR; not clearance",
                  "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "literal_preplacement_gate": "STOP — source or grid unverifiable",
                  "error": f"{type(exc).__name__}: {exc}"}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["literal_preplacement_gate"] == "UNRESOLVED — run full registry gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
