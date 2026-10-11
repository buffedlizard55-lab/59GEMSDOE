#!/usr/bin/env python3
"""Restore the pinned public registry raster cache from GitHub blobs and verify hashes.

The registry index (``evidence/registry_refreshed.json``) pins every accessible
sibling-repository prediction raster by SHA256 plus its git blob id and the commit
that contained it.  The raster bytes themselves are gitignored, so a fresh sandbox
has the manifest but not the files.  This tool refetches each blob through the
authenticated GitHub API, decodes it, and *only* writes the cache file if the
SHA256 equals the pinned value.  A mismatch or a missing blob is recorded as an
error and the affected raster is reported, never silently skipped.

Nothing here scores, fits or promotes anything; it restores read-only evidence.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(repo: str, blob: str, timeout: int = 180) -> bytes:
    """Return the decoded blob bytes, or raise."""
    out = subprocess.run(
        ["gh", "api", f"repos/buffedlizard55-lab/{repo}/git/blobs/{blob}",
         "--jq", ".content"],
        capture_output=True, text=True, timeout=timeout,
    )
    if out.returncode != 0:
        raise RuntimeError(f"gh api failed rc={out.returncode}: {out.stderr.strip()[:200]}")
    raw = out.stdout.strip()
    if not raw:
        raise RuntimeError("empty blob payload")
    return base64.b64decode(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(ROOT / "evidence/registry_refreshed.json"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "evidence/registry_cache_restore.json"))
    a = ap.parse_args()
    if shutil.which("gh") is None:
        print("gh not available", file=sys.stderr)
        return 2
    index = json.loads(Path(a.index).read_text())
    cache = ROOT / ".cache/registry"
    cache.mkdir(parents=True, exist_ok=True)
    rasters = index["rasters"]
    t0 = time.time()
    done = {"verified": 0, "cached": 0, "errors": []}

    def work(rec):
        dest = ROOT / rec["cache_file"]
        if dest.exists() and sha256_bytes(dest.read_bytes()) == rec["sha256"]:
            return ("cached", rec, None)
        try:
            data = fetch(rec["repo_first"], rec["blob"])
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return ("error", rec, str(exc)[:300])
        digest = sha256_bytes(data)
        if digest != rec["sha256"]:
            return ("error", rec, f"sha256 mismatch: got {digest} pinned {rec['sha256']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return ("verified", rec, None)

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for i, (kind, rec, err) in enumerate(pool.map(work, rasters), 1):
            if kind == "error":
                done["errors"].append({"repo": rec["repo_first"], "blob": rec["blob"],
                                       "sources": rec["sources"], "error": err})
            else:
                done[kind] += 1
            if i % 25 == 0:
                print(f"[restore] {i}/{len(rasters)} ok={done['verified']+done['cached']} "
                      f"err={len(done['errors'])} {time.time()-t0:.0f}s", flush=True)
    receipt = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        index=a.index, index_rasters=len(rasters),
        restored=done["verified"], already_cached=done["cached"],
        errors=done["errors"], n_errors=len(done["errors"]),
        complete=len(done["errors"]) == 0,
        seconds=round(time.time() - t0, 1),
        note="Cache restore only; no scoring, fitting or promotion performed here.",
    )
    Path(a.out).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "errors"}, indent=2))
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
