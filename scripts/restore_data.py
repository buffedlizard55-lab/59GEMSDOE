#!/usr/bin/env python3
"""Restore + SHA-256 verify every file pinned in registry/data_manifest.json.

All bytes come from public GitHub blobs (``gh api``), never from drivendata.org.
Every file is verified against its pinned sha256 and byte count; a mismatch is a
hard failure (fail-closed), per the repository's no-hallucination rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "registry" / "data_manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_fetch(repo: str, ref: str, path: str, dest: Path, tries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err = None
    for attempt in range(tries):
        try:
            with tmp.open("wb") as out:
                subprocess.run(
                    ["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
                     "-H", "Accept: application/vnd.github.raw"],
                    stdout=out, check=True)
            if tmp.stat().st_size > 0:
                tmp.replace(dest)
                return
            last_err = "zero-byte download"
        except subprocess.CalledProcessError as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"fetch failed {repo}@{ref}:{path}: {last_err}")


def restore_entry(entry: dict, root: Path, only_ids: set[str] | None) -> dict:
    if only_ids and entry["id"] not in only_ids:
        return None
    dest = root / entry["dest"]
    if dest.exists() and sha256_file(dest) == entry["sha256"] and dest.stat().st_size == entry["bytes"]:
        return {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
                "status": "present", "bytes": dest.stat().st_size, "sha256": entry["sha256"],
                "provenance": entry.get("provenance", "")}
    if "parts" in entry:
        tmp = dest.with_suffix(".assembling")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as out:
            for i, part in enumerate(entry["parts"]):
                p = root / "_parts" / f"{entry['id']}-{i:03d}"
                gh_fetch(entry["repo"], entry["ref"], part, p)
                with p.open("rb") as src:
                    shutil.copyfileobj(src, out, 1 << 22)
                p.unlink()
        got = sha256_file(tmp)
        if got != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"HASH MISMATCH {entry['id']}: {got} != {entry['sha256']}")
        tmp.replace(dest)
    else:
        gh_fetch(entry["repo"], entry["ref"], entry["path"], dest)
        got = sha256_file(dest)
        if got != entry["sha256"]:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"HASH MISMATCH {entry['id']}: {got} != {entry['sha256']}")
    return {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
            "status": "restored", "bytes": dest.stat().st_size, "sha256": entry["sha256"],
            "provenance": entry.get("provenance", "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="core,external", help="comma list: core,external,scored,all")
    ap.add_argument("--only", default="", help="comma list of entry ids")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--target-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    root = Path(args.target_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups = {g.strip() for g in args.group.split(",")}
    only = {o.strip() for o in args.only.split(",") if o.strip()}

    results = []
    for e in man["files"]:
        if "all" not in groups and e["group"] not in groups:
            continue
        r = restore_entry(e, root, only or None)
        if r:
            results.append(r)
            print(f"{r['status']:>8}  {e['dest']:70s}  {r['bytes']:>12,}  sha256={r['sha256'][:12]}", flush=True)

    receipt = {"schema_version": 1, "verified_files_count": len(results),
               "storage_root": str(root), "files": results}
    out = Path(args.manifest).parent.parent / "data" / "restore_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified": len(results), "dest": str(root)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
