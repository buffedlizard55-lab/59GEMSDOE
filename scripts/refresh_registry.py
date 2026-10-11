#!/usr/bin/env python3
"""Download every reachable hash-pinned prior GEMSDOE raster for the drift gate.

Index: registry/registry_index_20261010T2001.json -- 696 unique grid rasters across
57 sibling repositories, each pinned by sha256.  The index records the branch the
bytes were found on and many of those branches no longer resolve, so this script
first lists each repository's default-branch tree (one API call per repository),
matches registry entries to a path that actually exists there, and only then
downloads.  Every file is re-hashed; a mismatch or short read is a hard failure,
and reachable/total counts are written to ``evidence`` so the coverage of the
drift gate is auditable rather than assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "registry"
OWNER = "buffedlizard55-lab"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gh(args) -> tuple[int, bytes]:
    r = subprocess.run(["gh", "api"] + args, capture_output=True)
    return r.returncode, r.stdout


def default_branch(repo: str) -> str:
    rc, out = gh([f"repos/{OWNER}/{repo}", "--jq", ".default_branch"])
    return out.decode().strip() or "main" if rc == 0 else "main"


def tree_paths(repo: str, branch: str) -> dict:
    rc, out = gh([f"repos/{OWNER}/{repo}/git/trees/{branch}?recursive=1", "--jq",
                  '.tree[] | select(.type=="blob") | "\\(.path)\\t\\(.size)"'])
    if rc != 0:
        return {}
    d = {}
    for line in out.decode().splitlines():
        if "\t" in line:
            p, s = line.rsplit("\t", 1)
            try:
                d[p] = int(s)
            except ValueError:
                pass
    return d


def fetch(repo: str, path: str, dest: Path, branch: str, tries: int = 2) -> bool:
    tmp = dest.with_suffix(".part")
    for _ in range(tries):
        with tmp.open("wb") as out:
            r = subprocess.run(["gh", "api", f"repos/{OWNER}/{repo}/contents/{path}?ref={branch}",
                                "-H", "Accept: application/vnd.github.raw"], stdout=out)
        if r.returncode == 0 and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return True
        time.sleep(0.5)
    tmp.unlink(missing_ok=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "registry/registry_index_20261010T2001.json"))
    ap.add_argument("--build-map", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = json.loads(Path(a.index).read_text())["rasters"]

    mapp = ROOT / "work" / "registry_path_map.json"
    if a.build_map or not mapp.exists():
        repos = sorted({e["sources"][0].split(":")[0] for e in idx if e["sources"]})
        trees = {}
        for i, r in enumerate(repos):
            b = default_branch(r)
            trees[r] = {"branch": b, "paths": tree_paths(r, b)}
            print(f"  tree {i+1}/{len(repos)} {r} ({b}) {len(trees[r]['paths'])} blobs", flush=True)
        mapp.write_text(json.dumps(trees))

    trees = json.loads(mapp.read_text())
    todo = [e for e in idx[a.start:a.start + a.count]
            if not (CACHE / (e["sha256"] + ".tif")).exists()]
    ok = bad = unreachable = 0
    t0 = time.time()
    for i, e in enumerate(todo):
        dest = CACHE / (e["sha256"] + ".tif")
        hit = None
        for src in e["sources"]:
            if ":" not in src:
                continue
            repo, path = src.split(":", 1)
            t = trees.get(repo)
            if t and path in t["paths"] and t["paths"][path] == e["bytes"]:
                hit = (repo, path, t["branch"])
                break
        if hit is None:
            unreachable += 1
            continue
        if not fetch(hit[0], hit[1], dest, hit[2]):
            bad += 1
            continue
        if sha(dest) != e["sha256"] or dest.stat().st_size != e["bytes"]:
            dest.unlink(missing_ok=True)
            bad += 1
            continue
        ok += 1
        if (ok + bad) % 50 == 0:
            print(f"  {i+1}/{len(todo)} ok={ok} bad={bad} unreachable={unreachable} "
                  f"{time.time()-t0:.0f}s", flush=True)
    out = {"ok": ok, "bad": bad, "unreachable_on_default_branch": unreachable,
           "already_cached": len(idx) - len(todo), "index_total": len(idx),
           "cache_dir": str(CACHE), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "registry_cache_status.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
