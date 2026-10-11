#!/usr/bin/env python3
"""Build a scoped index of public owner-repository rasters.

This indexes grid-matching GeoTIFFs discoverable in the listed, accessible
GEMSDOE sibling repositories and historical pins. It is not a complete
organizer registry or proof of complete competition coverage: private,
unlinked, external, and otherwise inaccessible rasters may be absent. The
uniqueness gate can only compare against this explicitly scoped inventory.

The rasters themselves are cached OUTSIDE the repository (default
``/tmp/gems_registry_cache``) so Git does not grow.

Two stages (both idempotent):

    python scripts/scan_gemsdoe_registry.py clone    # shallow, blob-size-limited clones
    python scripts/scan_gemsdoe_registry.py index    # extract + scoped index -> evidence/registry_full_index.json

Then ``scripts/check_uniqueness_full.py`` runs the literal checks against the
indexed public owner-repository inventory; the filename does not imply a full
organizer registry.

Clones use ``--filter=blob:limit=3m`` so the 420 MB feature stack in the bridge
repos is never downloaded; only submission-sized rasters are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "buffedlizard55-lab"
# Repo list taken from the owner's submission list in BRIEF.md (GEMSDOE, GEMSDOE2 ..
# GEMSDOE54, 5GEMSDOE .. 8GEMSDOE, 11GEMSDOE .. 20GEMSDOE, 55GEMSDOE .. 57GEMSDOE).
# Repos that do not exist on github.com are recorded as unreachable, not guessed.
REPOS = (
    ["GEMSDOE", "GEMSDOE2", "GEMSDOE3", "GEMSDOE4", "5GEMSDOE", "6GEMSDOE", "7GEMSDOE",
     "8GEMSDOE", "GEMSDOE9", "GEMSDOE10", "11GEMSDOE", "12GEMSDOE", "13GEMSDOE", "14GEMSDOE",
     "15GEMSDOE", "16GEMSDOE", "17GEMSDOE", "18GEMSDOE", "19GEMSDOE", "20GEMSDOE"]
    + [f"GEMSDOE{i}" for i in range(21, 55)]
    + ["55GEMSDOE", "56GEMSDOE", "57GEMSDOE"]
)
SELF = "57GEMSDOE"   # this repo: its own file is excluded from the registry it is checked against
GRID = (3730, 3292)
MAX_BLOB = 3 * 1024 * 1024          # clone-time filter: larger blobs are fetched on demand
MAX_LAZY_BLOB = 60 * 1024 * 1024    # on-demand fetch ceiling for large submission-style rasters
# Large rasters that are feature stacks, multi-part splits, DEM/aux layers or diagnostics
# are NOT submissions and are not fetched (bandwidth). Every exclusion is counted in the index.
LAZY_EXCLUDE = ("part-", "numerical-features", "fixture", "source_mirrors", "external/", "aux",
                "diagnostics", "pilot", "audit_sources", "radiometric", "topo_u8", "geodawn",
                "lidar_scarp", "h52_scarp")

DEFAULT_CLONES = Path("/tmp/gems_scan")
DEFAULT_CACHE = Path("/tmp/gems_registry_cache")
INDEX_OUT = ROOT / "evidence" / "registry_full_index.json"


def cmd_clone(clones: Path) -> None:
    clones.mkdir(parents=True, exist_ok=True)
    status = {}
    for r in REPOS:
        d = clones / r
        if (d / "HEAD").exists() or (d / ".git").exists():
            status[r] = "present"
            continue
        p = subprocess.run(
            ["git", "clone", "-q", "--no-checkout", "--filter=blob:limit=3m", "--depth", "1",
             f"https://github.com/{OWNER}/{r}.git", str(d)],
            capture_output=True, text=True, timeout=600)
        status[r] = "cloned" if p.returncode == 0 else "unreachable: " + p.stderr.strip()[:160]
        print(f"{r}: {status[r]}", flush=True)
    (clones / "clone_status.json").write_text(json.dumps(status, indent=2))


def _git(d: Path, *args: str, binary: bool = True) -> bytes:
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, check=True).stdout


def cmd_index(clones: Path, cache: Path) -> None:
    import numpy as np
    import rasterio

    cache.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    skipped = {"not_grid": 0, "too_large_or_missing": 0, "self": 0}
    for r in REPOS:
        if r == SELF:
            continue
        d = clones / r
        if not d.exists():
            continue
        try:
            listing = _git(d, "ls-tree", "-r", "-l", "HEAD").decode(errors="replace").splitlines()
        except subprocess.CalledProcessError:
            continue
        for line in listing:
            meta, path = line.split("\t", 1)
            parts = meta.split()
            if len(parts) < 4 or not path.lower().endswith((".tif", ".tiff")):
                continue
            blob, size = parts[2], parts[3]
            if size == "-":
                skipped["too_large_or_missing"] += 1
                continue
            if int(size) > MAX_BLOB:
                if int(size) > MAX_LAZY_BLOB:
                    skipped["over_lazy_ceiling"] = skipped.get("over_lazy_ceiling", 0) + 1
                    continue
                if any(k in path for k in LAZY_EXCLUDE):
                    skipped["lazy_excluded_nonsubmission"] = skipped.get("lazy_excluded_nonsubmission", 0) + 1
                    continue
                skipped["lazy_fetched"] = skipped.get("lazy_fetched", 0) + 1
            data = _git(d, "cat-file", "-p", blob)
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen:
                seen[sha]["sources"].append(f"{r}:{path}")
                continue
            local = cache / f"{sha[:16]}.tif"
            local.write_bytes(data)
            with rasterio.open(local) as ds:
                ok = (ds.height, ds.width) == GRID and str(ds.crs) == "EPSG:32611"
            if not ok:
                local.unlink()
                skipped["not_grid"] += 1
                continue
            seen[sha] = {"sha256": sha, "bytes": len(data), "blob": blob, "cache_file": str(local),
                         "sources": [f"{r}:{path}"], "repo_first": r}
    # provenance-only self check: our own shipped files are listed but not compared
    recs = sorted(seen.values(), key=lambda x: x["repo_first"])
    out = {
        "owner": OWNER,
        "repos_scanned": [r for r in REPOS if r != SELF and (clones / r).exists()],
        "repos_unreachable_or_missing": [r for r in REPOS if not (clones / r).exists()],
        "grid": {"shape": list(GRID), "crs": "EPSG:32611"},
        "max_blob_bytes": MAX_BLOB,
        "n_unique_grid_rasters": len(recs),
        "skipped": skipped,
        "rasters": recs,
    }
    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    INDEX_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"unique grid rasters: {len(recs)}  skipped: {skipped}  -> {INDEX_OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["clone", "index"])
    ap.add_argument("--clones", type=Path, default=DEFAULT_CLONES)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    a = ap.parse_args()
    if a.stage == "clone":
        cmd_clone(a.clones)
    else:
        cmd_index(a.clones, a.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
