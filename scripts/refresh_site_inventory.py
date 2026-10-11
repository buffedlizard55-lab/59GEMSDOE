#!/usr/bin/env python3
"""Refresh homepage/README presence for every currently pinned public-main tree.

Commit pins come from the complete 57-repository public-main scan in
``registry_refreshed.json``. Trees are re-requested only when their commit differs
from the existing site inventory. This updates navigational evidence only; it
does not claim that a live Pages deployment is current or manually reviewed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from refresh_registry import OWNER

REGISTRY = ROOT / "evidence" / "registry_refreshed.json"
INVENTORY = ROOT / "evidence" / "site_inventory.json"


def fetch_tree(repo: str, commit: str) -> dict:
    proc = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{repo}/git/trees/{commit}?recursive=1"],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    result = json.loads(proc.stdout)
    if result.get("truncated"):
        raise ValueError(f"recursive Git tree truncated for {repo}@{commit}")
    return result


def refresh() -> dict:
    registry = json.loads(REGISTRY.read_text())
    previous = json.loads(INVENTORY.read_text())
    if not registry.get("complete_accessible_scan"):
        raise ValueError("site inventory requires a complete current public-main commit scan")
    snapshots = {row["repo"]: row["commit"] for row in registry.get("snapshots", [])}
    old = {row["repo"]: row for row in previous.get("repos", [])}
    if len(snapshots) != 57 or set(snapshots) != set(old):
        raise ValueError("expected exactly the same 57 repositories in both manifests")
    changed = {repo: sha for repo, sha in snapshots.items() if old[repo].get("commit") != sha}
    trees: dict[str, dict] = {}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_tree, repo, sha): (repo, sha) for repo, sha in changed.items()}
        for future in as_completed(futures):
            repo, sha = futures[future]
            try:
                trees[repo] = future.result()
            except Exception as exc:
                errors.append({"repo": repo, "commit": sha, "error": str(exc)[:240]})
    if errors or len(trees) != len(changed):
        raise RuntimeError(f"site inventory refresh incomplete: {errors}")

    refreshed = []
    for repo in sorted(snapshots):
        sha = snapshots[repo]
        prior = old[repo]
        if repo in trees:
            paths = {item.get("path", "") for item in trees[repo].get("tree", []) if item.get("type") == "blob"}
            index_files = [path for path in ("docs/index.html", "index.html") if path in paths]
            readme_available = any(path.rsplit("/", 1)[-1].casefold() == "readme.md" and "/" not in path for path in paths)
        else:
            index_files = list(prior.get("index_files", []))
            readme_available = bool(prior.get("readme_available"))
        refreshed.append({
            "repo": repo,
            "commit": sha,
            "website": f"https://{OWNER}.github.io/{repo}/",
            "index_files": index_files,
            "readme_available": readme_available,
        })

    now = datetime.now(timezone.utc).isoformat()
    result = {
        "generated_utc": now,
        "evidence_class": "PUBLIC-MAIN-SITE-INVENTORY",
        "commit_source": "complete current public-main snapshots in registry_refreshed.json",
        "repositories_checked": len(refreshed),
        "trees_refreshed": len(trees),
        "tree_errors": [],
        "scope": (
            "Homepage-file and root README presence in pinned public GitHub trees only; "
            "does not verify deployed GitHub Pages freshness, rendered content, or manual page review."
        ),
        "repos": refreshed,
    }
    INVENTORY.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = {
        "repositories_checked": len(refreshed),
        "trees_refreshed": len(trees),
        "current_main_commits": len(snapshots),
        "index_files_found": sum(bool(row["index_files"]) for row in refreshed),
        "errors": 0,
        "generated_utc": now,
    }
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    refresh()
