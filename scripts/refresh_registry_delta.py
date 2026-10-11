#!/usr/bin/env python3
"""Extend a complete raster uniqueness audit using only newly changed Git blobs.

The baseline full audit is already SHA256-pinned and complete. This script
checks the latest public-main commit for every listed sibling repository, scans
only trees whose commits changed, fetches only previously unseen TIFF blobs via
the authenticated GitHub CLI/API allowlist, validates their competition grid,
and compares each new grid raster with the current research surface using the
shared ``gems59.uniqueness`` implementation. The same candidate, thresholds,
and previous per-raster measurements are required for an incremental merge.

It does not fit a model, evaluate DTI, place production dots, or submit anything.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gems59 import load_grid
from gems59.uniqueness import compare_to_registry, merge_registry_audits
from refresh_registry import INPUT_PATTERNS, OWNER, REPOS, fetch_blob

DELTA_OUT = ROOT / "evidence" / "registry_live_delta.json"
DELTA_AUDIT_OUT = ROOT / "evidence" / "orientation_surface_delta_uniqueness.json"


def gh_json(endpoint: str) -> dict:
    proc = subprocess.run(
        ["gh", "api", endpoint], capture_output=True, text=True, check=True, timeout=90
    )
    return json.loads(proc.stdout)


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def current_commits(workers: int = 6) -> tuple[dict[str, str], list[dict]]:
    commits: dict[str, str] = {}
    errors: list[dict] = []

    def one(repo: str):
        proc = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{repo}/commits/main", "--jq", ".sha"],
            capture_output=True, text=True, timeout=90,
        )
        if proc.returncode:
            return repo, None, proc.stderr.strip()[:240]
        return repo, proc.stdout.strip(), None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for repo, sha, error in pool.map(one, REPOS):
            if error or not sha:
                errors.append({"repo": repo, "error": error or "empty main commit SHA"})
            else:
                commits[repo] = sha
    return commits, errors


def current_open_pr_heads() -> tuple[list[dict], list[dict]]:
    """List same-repository PRs targeting main, pinned to immutable head SHAs."""
    proc = subprocess.run(
        [
            "gh", "pr", "list", "--repo", f"{OWNER}/57GEMSDOE", "--state", "open",
            "--limit", "1000", "--json",
            "number,baseRefName,headRefName,headRefOid,headRepository,headRepositoryOwner",
        ],
        capture_output=True, text=True, timeout=90,
    )
    if proc.returncode:
        return [], [{"error": proc.stderr.strip()[:240] or "failed to list open pull requests"}]
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return [], [{"error": f"invalid open pull request JSON: {exc}"}]
    if not isinstance(rows, list):
        return [], [{"error": "open pull request response is not a list"}]

    heads: list[dict] = []
    errors: list[dict] = []
    for row in rows:
        if row.get("baseRefName") != "main":
            continue
        owner = (row.get("headRepositoryOwner") or {}).get("login")
        repo = (row.get("headRepository") or {}).get("name")
        if owner != OWNER or repo != "57GEMSDOE":
            errors.append({
                "pr": row.get("number"),
                "error": "open main-target PR head is not in the indexed owner repository",
            })
            continue
        commit = row.get("headRefOid")
        if not isinstance(commit, str) or len(commit) != 40:
            errors.append({"pr": row.get("number"), "error": "missing immutable PR head commit"})
            continue
        heads.append({
            "pr": int(row["number"]),
            "repo": "57GEMSDOE",
            "branch": row.get("headRefName"),
            "commit": commit,
        })
    return sorted(heads, key=lambda row: row["pr"]), errors


def open_pr_trees(heads: list[dict], workers: int = 5) -> tuple[dict[int, list], list[dict]]:
    trees: dict[int, list] = {}
    errors: list[dict] = []

    def one(head: dict):
        result = gh_json(
            f"repos/{OWNER}/57GEMSDOE/git/trees/{head['commit']}?recursive=1"
        )
        if result.get("truncated"):
            raise ValueError("GitHub returned a truncated recursive PR tree")
        return int(head["pr"]), result.get("tree", [])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, head): head for head in heads}
        for future in as_completed(futures):
            head = futures[future]
            try:
                pr, tree = future.result()
                trees[pr] = tree
            except Exception as exc:
                errors.append({
                    "pr": head["pr"], "commit": head["commit"],
                    "error": str(exc)[:240],
                })
    return trees, errors


def changed_trees(changed: dict[str, str], workers: int = 5) -> tuple[dict[str, list], list[dict]]:
    trees: dict[str, list] = {}
    errors: list[dict] = []

    def one(repo: str, commit: str):
        result = gh_json(f"repos/{OWNER}/{repo}/git/trees/{commit}?recursive=1")
        if result.get("truncated"):
            raise ValueError("GitHub returned a truncated recursive tree")
        return repo, result.get("tree", [])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, repo, commit): (repo, commit) for repo, commit in changed.items()}
        for future in as_completed(futures):
            repo, commit = futures[future]
            try:
                got_repo, tree = future.result()
                trees[got_repo] = tree
            except Exception as exc:
                errors.append({"repo": repo, "commit": commit, "error": str(exc)[:240]})
    return trees, errors


def _baseline_is_consistent(registry: dict, audit: dict, candidate: Path) -> None:
    records = registry.get("rasters", [])
    rows = audit.get("rows", [])
    if not registry.get("complete_accessible_scan") or not audit.get("complete_accessible_scan"):
        raise ValueError("incremental extension requires a complete prior inventory and comparison")
    if len(records) != len(rows) or len(records) != registry.get("n_unique_grid_rasters"):
        raise ValueError("prior inventory and uniqueness row counts disagree")
    if audit.get("registry_rasters_expected") != len(rows) or audit.get("registry_rasters_checked") != len(rows):
        raise ValueError("prior uniqueness audit is not a full exact scan")
    if audit.get("missing_or_invalid") or audit.get("source_errors"):
        raise ValueError("prior uniqueness audit has missing or invalid rows")
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if candidate_sha != audit.get("candidate_file_sha256"):
        raise ValueError("baseline audit belongs to different candidate bytes")
    by_source = {source: record["sha256"] for record in records for source in record.get("sources", [])}
    audited_hashes = set()
    for row in rows:
        if "error" in row or by_source.get(row.get("submission")) != row.get("sha256"):
            raise ValueError(f"cannot map prior measurement to immutable registry record: {row.get('submission')}")
        audited_hashes.add(row["sha256"])
    if audited_hashes != {r["sha256"] for r in records}:
        raise ValueError("prior audit does not cover the exact registry content-hash set")


def refresh(root: Path = ROOT, *, workers: int = 5) -> dict:
    registry_path = root / "evidence" / "registry_refreshed.json"
    base_audit_path = root / "evidence" / "orientation_surface_uniqueness.json"
    candidate_card = json.loads((root / "evidence" / "run_card_current.json").read_text())
    candidate = root / candidate_card["file"]
    registry = json.loads(registry_path.read_text())
    baseline = json.loads(base_audit_path.read_text())
    _baseline_is_consistent(registry, baseline, candidate)

    commits, commit_errors = current_commits(workers=max(workers, 6))
    open_pr_heads, open_pr_discovery_errors = current_open_pr_heads()
    old_commits = {row["repo"]: row["commit"] for row in registry["snapshots"]}
    changed = {repo: sha for repo, sha in commits.items() if old_commits.get(repo) != sha}
    old_open_pr_heads = registry.get("open_pr_heads_checked", [])
    old_open_refs = {(int(row["pr"]), row["commit"]) for row in old_open_pr_heads}
    current_open_refs = {(int(row["pr"]), row["commit"]) for row in open_pr_heads}
    open_pr_heads_changed = old_open_refs != current_open_refs
    if not changed and not commit_errors and not open_pr_discovery_errors and not open_pr_heads_changed:
        # Idempotent repeat: preserve the dated measurement instead of replacing
        # it with an empty delta when neither main nor any open PR head changed.
        result = {
            "evidence_class": "REGISTRY-DELTA-NO-OP",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "current_main_commits_checked": len(commits),
            "changed_repositories": [],
            "current_open_pr_heads_checked": open_pr_heads,
            "new_grid_rasters_added": 0,
            "existing_audit_rasters_checked": baseline["registry_rasters_checked"],
            "complete_accessible_scan": True,
            "surface_protocol_pass": baseline["unique"],
            "preserved_historical_delta_receipt": "evidence/registry_live_delta.json",
        }
        print(json.dumps(result, indent=2))
        return result
    trees, tree_errors = changed_trees(changed, workers=workers)
    pr_trees, pr_tree_errors = open_pr_trees(open_pr_heads, workers=workers)

    old_by_blob = {record["blob"]: record for record in registry["rasters"]}
    candidate_blob = git_blob_sha1(candidate)
    discovered: dict[str, dict] = {}
    aliases_added: dict[str, list[str]] = {}
    excluded_candidate_sources: list[str] = []
    excluded_inputs: list[dict] = []
    too_large: list[dict] = []

    tree_sources = [
        (repo, commits[repo], tree)
        for repo, tree in trees.items()
    ]
    tree_sources.extend(
        (f"57GEMSDOE-PR{pr}", next(row["commit"] for row in open_pr_heads if row["pr"] == pr), tree)
        for pr, tree in pr_trees.items()
    )
    for repo, commit, tree in tree_sources:
        for item in tree:
            path = item.get("path", "")
            if item.get("type") != "blob" or not path.lower().endswith((".tif", ".tiff")):
                continue
            if any(pattern in path for pattern in INPUT_PATTERNS):
                excluded_inputs.append({"repo": repo, "path": path, "reason": "input/auxiliary source raster under existing registry policy"})
                continue
            blob = item["sha"]
            source = f"{repo}:{path}"
            # The exact current candidate bytes are never treated as an earlier
            # prediction, regardless of which public tree exposes that blob.
            if blob == candidate_blob:
                excluded_candidate_sources.append(source)
                continue
            if blob in old_by_blob:
                if source not in old_by_blob[blob]["sources"]:
                    old_by_blob[blob]["sources"].append(source)
                    aliases_added.setdefault(blob, []).append(source)
                continue
            if blob not in discovered:
                size = int(item.get("size", 0))
                if size > 100 * 1024 * 1024:
                    too_large.append({"repo": repo, "path": path, "blob": blob, "size": size})
                    continue
                discovered[blob] = {
                    "repo_first": "57GEMSDOE" if repo.startswith("57GEMSDOE-PR") else repo,
                    "blob": blob,
                    "sources": [source],
                    "source_commit": commit,
                }
            elif source not in discovered[blob]["sources"]:
                discovered[blob]["sources"].append(source)

    cache = root / ".cache" / "registry"
    cache.mkdir(parents=True, exist_ok=True)
    new_records: list[dict] = []
    skipped_grid: list[dict] = []
    fetch_errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_blob, record, cache): record for record in discovered.values()}
        for future in as_completed(futures):
            source_record = futures[future]
            try:
                row, exclusion = future.result()
                if exclusion:
                    skipped_grid.append(exclusion)
                else:
                    row["sources"] = sorted(set(row["sources"]))
                    new_records.append(row)
            except Exception as exc:
                fetch_errors.append({"sources": source_record["sources"], "blob": source_record["blob"], "error": str(exc)[:240]})

    # A fresh scan must not count multiple byte-identical aliases twice.
    existing_by_sha = {row["sha256"]: row for row in registry["rasters"]}
    deduplicated_new: dict[str, dict] = {}
    for row in new_records:
        digest = row["sha256"]
        if digest in existing_by_sha:
            prior = existing_by_sha[digest]
            prior["sources"] = sorted(set(prior["sources"] + row["sources"]))
            aliases_added.setdefault(prior["blob"], []).extend(row["sources"])
        elif digest in deduplicated_new:
            got = deduplicated_new[digest]
            got["sources"] = sorted(set(got["sources"] + row["sources"]))
        else:
            deduplicated_new[digest] = row

    scan_complete = not (
        commit_errors or open_pr_discovery_errors or tree_errors or pr_tree_errors
        or too_large or fetch_errors
    )
    combined = registry["rasters"] + list(deduplicated_new.values())
    combined_by_sha = {row["sha256"]: row for row in combined}
    combined = sorted(combined_by_sha.values(), key=lambda row: row["sha256"])
    scan_complete = scan_complete and len(combined) == len(registry["rasters"]) + len(deduplicated_new)

    # Validate old and new manifests separately; merge their exact per-raster metrics.
    scope = (
        "SHA256-pinned indexed public owner-repository inventory extended with current public-main trees for all 57 listed repositories "
        "and currently open main-target PR heads in 57GEMSDOE; only newly seen Git blobs were downloaded and compared. "
        "This is not a complete organizer registry. The exact current candidate bytes are excluded as the candidate's own output. "
        "Private, unlinked, external, inaccessible and later-written files may be absent."
    )
    delta = None
    if deduplicated_new:
        mini = {
            "complete_accessible_scan": bool(scan_complete),
            "errors": fetch_errors + tree_errors + commit_errors + too_large,
            "n_unique_grid_rasters": len(deduplicated_new),
            "rasters": list(deduplicated_new.values()),
        }
        grid = load_grid(root / "data" / "bridge")
        delta = compare_to_registry(candidate, mini, grid.footprint)
        delta["phase"] = "surface"
        delta["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
        delta["submission_slots_used"] = int(candidate_card.get("submission_slots_used", 0))
        merged_audit = merge_registry_audits(
            baseline, delta,
            registry_rasters_expected=len(combined),
            complete_accessible_scan=scan_complete,
            scope=scope,
        )
    else:
        # A repeated refresh with no new valid grid raster is an idempotent no-op,
        # not an error and not a false zero-row completeness claim.
        merged_audit = dict(baseline)
        merged_audit["scope"] = scope
        merged_audit["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
        merged_audit["complete_accessible_scan"] = bool(
            scan_complete and baseline.get("complete_accessible_scan")
        )
    merged_audit["aggregation"] = dict(
        method="incremental exact extension of a complete prior compare_to_registry result",
        base_registry_rasters=baseline["registry_rasters_checked"],
        new_registry_rasters=(delta["registry_rasters_checked"] if delta is not None else 0),
        base_candidate_sha256=baseline["candidate_file_sha256"],
        current_candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        latest_public_main_commits_checked=len(commits),
        changed_public_main_commits=len(changed),
        current_open_pr_heads_checked=len(open_pr_heads),
        changed_open_pr_heads=sum(
            (int(row["pr"]), row["commit"]) not in old_open_refs
            for row in open_pr_heads
        ),
    )

    now = datetime.now(timezone.utc).isoformat()
    extension = {
        "evidence_class": "REGISTRY-DELTA-MEASUREMENT",
        "generated_utc": now,
        "owner": OWNER,
        "candidate": candidate.name,
        "candidate_sha256": merged_audit["candidate_file_sha256"],
        "prior_rasters_reused": baseline["registry_rasters_checked"],
        "new_unique_blobs_examined": len(discovered),
        "new_grid_rasters_added": len(deduplicated_new),
        "new_grid_raster_rows": deduplicated_new,
        "new_raster_comparison_file": (DELTA_AUDIT_OUT.name if delta is not None else None),
        "new_raster_comparison_summary": ({
            "registry_rasters_checked": delta["registry_rasters_checked"],
            "worst_spearman_full_footprint": delta["worst_spearman_full_footprint"],
            "worst_dot_overlap": delta["worst_dot_overlap"],
            "duplicate_count": delta["duplicate_count"],
            "unique": delta["unique"],
        } if delta is not None else None),
        "changed_repositories": sorted(changed),
        "current_public_main_snapshots": [
            {"repo": repo, "commit": sha} for repo, sha in sorted(commits.items())
        ],
        "current_main_commit_errors": commit_errors,
        "current_open_pr_heads": open_pr_heads,
        "previous_open_pr_heads_checked": old_open_pr_heads,
        "open_pr_head_discovery_errors": open_pr_discovery_errors,
        "open_pr_tree_errors": pr_tree_errors,
        "changed_open_pr_heads": [
            row for row in open_pr_heads if (int(row["pr"]), row["commit"]) not in old_open_refs
        ],
        "closed_or_replaced_open_pr_heads": [
            row for row in old_open_pr_heads if (int(row["pr"]), row["commit"]) not in current_open_refs
        ],
        "changed_tree_errors": tree_errors,
        "fetch_errors": fetch_errors,
        "too_large_unchecked_tiffs": too_large,
        "non_grid_tiffs_examined": skipped_grid,
        "input_auxiliary_tiffs_excluded_by_existing_policy": excluded_inputs,
        "candidate_output_excluded_from_its_own_prior_registry": sorted(excluded_candidate_sources),
        "new_aliases_for_prior_blobs": {k: sorted(set(v)) for k, v in sorted(aliases_added.items())},
        "complete_accessible_scan": bool(scan_complete and merged_audit["complete_accessible_scan"]),
        "surface_protocol_pass": bool(merged_audit["unique"]),
        "current_audit_raster_count": merged_audit["registry_rasters_checked"],
        "uniqueness_audit_file": "orientation_surface_uniqueness.json",
        "submission_slots_used": 0,
    }

    if not extension["complete_accessible_scan"]:
        raise ValueError("incremental scan incomplete; current audit cannot be promoted")

    previous_backup = root / "evidence" / "orientation_surface_uniqueness_20261009.json"
    if not previous_backup.exists():
        shutil.copy2(base_audit_path, previous_backup)
    (root / "evidence" / "registry_refreshed.json").write_text(
        json.dumps({
            **registry,
            "generated_utc": now,
            "snapshots": [{"repo": repo, "commit": sha} for repo, sha in sorted(commits.items())],
            "repos_scanned": sorted(commits),
            "repos_unreachable_or_missing": [],
            "errors": [],
            "n_unique_grid_rasters": len(combined),
            "complete_accessible_scan": True,
            "rasters": combined,
            "latest_main_commit_scan_utc": now,
            "latest_open_pr_head_scan_utc": now,
            "open_pr_heads_checked": open_pr_heads,
            "incremental_extension": {
                "base_registry_rasters": len(registry["rasters"]),
                "added_unique_grid_rasters": len(deduplicated_new),
                "changed_repositories": sorted(changed),
                "open_pr_heads_checked": open_pr_heads,
                "open_pr_head_rows_replaced": old_open_pr_heads,
                "current_candidate_excluded_by_sha256": merged_audit["candidate_file_sha256"],
            },
            "scope": (
                "Historical immutable grid-raster pins union latest checked public-main trees for all 57 listed repositories and currently open main-target PR heads in 57GEMSDOE; "
                "this is an indexed public owner-repository inventory, not a complete organizer registry. The exact current candidate bytes are excluded from its own check. "
                "Private, unlinked, external, inaccessible and later-written rasters may be absent."
            ),
        }, indent=2, allow_nan=False) + "\n"
    )
    (root / "evidence" / "orientation_surface_uniqueness.json").write_text(
        json.dumps(merged_audit, indent=2, allow_nan=False) + "\n"
    )
    (root / "evidence" / "uniqueness_current_review.json").write_text(
        json.dumps(merged_audit, indent=2, allow_nan=False) + "\n"
    )
    run_card = dict(candidate_card)
    old_gate = run_card["correlation_overlap_vs_registry"]
    for key in (
        "registry_rasters_expected", "registry_rasters_checked", "complete_accessible_scan",
        "worst_spearman_full_footprint", "worst_rho_submission", "worst_dot_overlap",
        "worst_overlap_submission", "duplicate_count", "byte_unique_among_checked",
        "pixel_unique_among_checked", "unique",
    ):
        old_gate[key] = merged_audit[key]
    run_card["registry_reviewed_utc"] = now
    run_card["registry_scope"] = merged_audit["scope"]
    run_card["surface_before_placement"] = {"checked": True, "protocol_pass": merged_audit["unique"]}
    if not merged_audit["unique"]:
        run_card["final_dots"] = {
            "status": "not_generated",
            "reason": "literal surface gate requires stop",
            "emitted_pixels": 0,
        }
    (root / "evidence" / "run_card_current.json").write_text(
        json.dumps(run_card, indent=2, allow_nan=False) + "\n"
    )
    DELTA_OUT.write_text(json.dumps(extension, indent=2, allow_nan=False) + "\n")
    if delta is not None:
        DELTA_AUDIT_OUT.write_text(json.dumps(delta, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "current_main_repositories_checked": len(commits),
        "changed_repositories": sorted(changed),
        "new_grid_rasters_added": len(deduplicated_new),
        "final_registry_rasters_checked": merged_audit["registry_rasters_checked"],
        "worst_spearman_full_footprint": merged_audit["worst_spearman_full_footprint"],
        "worst_dot_overlap": merged_audit["worst_dot_overlap"],
        "duplicate_count": merged_audit["duplicate_count"],
        "complete_accessible_scan": merged_audit["complete_accessible_scan"],
        "surface_protocol_pass": merged_audit["unique"],
        "production_final_dots": 0,
        "submission_slots_used": 0,
        "delta_file": str(DELTA_OUT.relative_to(root)),
    }, indent=2))
    return extension


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        parser.error("workers must be in 1..12")
    refresh(workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
