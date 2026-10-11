#!/usr/bin/env python3
"""Refresh accessible sibling prediction rasters via immutable GitHub blob IDs.

No off-allowlist raw host, /tmp paths or partial-registry clearance. The old
inventory is unioned with current main trees, including this repo's earlier
artifacts. A missing source/blob is an error, not evidence of uniqueness.
Large source stacks/DEM tiles are excluded by path; any ambiguous single-band
competition-grid TIFF is conservatively kept, including dense soft fields.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OWNER = "buffedlizard55-lab"
REPOS = ["GEMSDOE", "GEMSDOE2", "GEMSDOE3", "GEMSDOE4", "5GEMSDOE", "6GEMSDOE", "7GEMSDOE", "8GEMSDOE", "GEMSDOE9", "GEMSDOE10", "11GEMSDOE", "12GEMSDOE", "13GEMSDOE", "14GEMSDOE", "15GEMSDOE", "16GEMSDOE", "17GEMSDOE", "18GEMSDOE", "19GEMSDOE", "20GEMSDOE"] + [f"GEMSDOE{i}" for i in range(21, 55)] + ["55GEMSDOE", "56GEMSDOE", "57GEMSDOE"]
# Inputs cannot be treated as earlier predictions. Dense predictions are NOT excluded.
INPUT_PATTERNS = (".part-", "numerical-features", "training_features", "existing_faults", "sample_submission", "example_submission", "labels.tif", "/external/", "/raw/", "/bridge/", "/source_mirrors/", "/aux/", "/audit_sources/", "lidar_scarp", "topo_u8", "h52_scarp")


def gh_json(endpoint):
    p = subprocess.run(["gh", "api", endpoint], check=True, capture_output=True, timeout=90)
    return json.loads(p.stdout)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda: s.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def get_tree(repo):
    # /commits/main resolves to an immutable commit; the tree endpoint then pins the scan.
    c = gh_json(f"repos/{OWNER}/{repo}/commits/main")
    x = gh_json(f"repos/{OWNER}/{repo}/git/trees/{c['sha']}?recursive=1")
    if x.get('truncated'):
        raise ValueError(f"truncated GitHub tree: {repo}")
    return repo, c['sha'], x['tree']


def fetch_blob(record, cache):
    import rasterio
    blob = record['blob']
    target = cache / f"{blob}.tif"
    temporary = target.with_suffix('.tif.partial')
    if not target.exists():
        try:
            with temporary.open('wb') as f:
                p = subprocess.run(['gh', 'api', f"repos/{OWNER}/{record['repo_first']}/git/blobs/{blob}", '-H', 'Accept: application/vnd.github.raw+json'], stdout=f, stderr=subprocess.PIPE, timeout=180)
            if p.returncode:
                raise RuntimeError(p.stderr.decode()[:180])
            # Verify the Git object itself, not only its name or transport URL.
            actual = hashlib.sha1(b'blob ' + str(temporary.stat().st_size).encode() + b'\0' + temporary.read_bytes()).hexdigest()
            if actual != blob:
                raise ValueError('immutable Git blob hash mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(target)
    if record.get('sha256') and digest != record['sha256']:
        raise ValueError('historical SHA256 pin mismatch')
    with rasterio.open(target) as src:
        if src.count != 1 or src.shape != (3730, 3292) or src.crs is None or src.crs.to_epsg() != 32611:
            return None, dict(sources=record['sources'], reason='not a single-band competition-grid prediction')
        from affine import Affine
        if src.transform != Affine(100, 0, 243350, 0, -100, 4508550):
            return None, dict(sources=record['sources'], reason='different transform')
    return {**record, 'sha256': digest, 'bytes': target.stat().st_size, 'cache_file': str(target.relative_to(ROOT))}, None


def refresh(root=ROOT, workers=4):
    cache = root / '.cache/registry'
    cache.mkdir(parents=True, exist_ok=True)
    by_blob = {}
    for old_path in (root / 'evidence/registry_full_index.json', root / 'evidence/registry_refreshed.json'):
        if not old_path.exists():
            continue
        for record in json.loads(old_path.read_text()).get('rasters', []):
            blob = record['blob']
            if blob in by_blob:
                by_blob[blob]['sources'] = sorted(set(by_blob[blob]['sources'] + record['sources']))
            else:
                by_blob[blob] = {**record, 'sources': list(record['sources'])}
    snapshots, errors, excluded, reviews = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get_tree, r): r for r in REPOS}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                repo, commit, tree = future.result()
                snapshots.append(dict(repo=repo, commit=commit))
                reviews.append(dict(repo=repo, commit=commit,
                    website=f'https://{OWNER}.github.io/{repo}/',
                    index_files=[t['path'] for t in tree if t['path'] in ('docs/index.html', 'index.html')],
                    readme_available=any(t['path']=='README.md' for t in tree)))
                for t in tree:
                    path = t['path']
                    if t['type'] != 'blob' or not path.lower().endswith(('.tif', '.tiff')):
                        continue
                    if any(p in path for p in INPUT_PATTERNS):
                        excluded.append(dict(repo=repo, path=path, reason='input/auxiliary source raster, not submission'))
                        continue
                    if t.get('size', 0) > 100 * 1024 * 1024:
                        errors.append(dict(repo=repo, source=path, error='ambiguous TIFF exceeds API raw limit; not silently skipped'))
                        continue
                    source = f'{repo}:{path}'
                    if t['sha'] in by_blob:
                        if source not in by_blob[t['sha']]['sources']:
                            by_blob[t['sha']]['sources'].append(source)
                    else:
                        by_blob[t['sha']] = dict(repo_first=repo, blob=t['sha'], sources=[source], source_commit=commit)
                print(f'[registry] inventory {repo}', flush=True)
            except Exception as e:
                errors.append(dict(repo=repo, error=str(e)[:240]))
                print(f'[registry] ERROR {repo}: {str(e)[:200]}', flush=True)
    seen = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_blob, r, cache): r for r in by_blob.values()}
        for i, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            try:
                row, exclusion = future.result()
                if exclusion:
                    excluded.append(exclusion)
                else:
                    if row['sha256'] in seen:
                        prior = seen[row['sha256']]
                        prior['sources'] = sorted(set(prior['sources'] + row['sources']))
                    else:
                        seen[row['sha256']] = row
            except Exception as e:
                errors.append(dict(sources=rec['sources'], blob=rec['blob'], error=str(e)[:240]))
            if i % 50 == 0:
                print(f'[registry] fetched/verified {i}/{len(by_blob)}', flush=True)
    report = dict(evidence_class='REGISTRY-MEASUREMENT', generated_utc=datetime.now(timezone.utc).isoformat(),
        owner=OWNER, repos_scanned=sorted(x['repo'] for x in snapshots), snapshots=sorted(snapshots, key=lambda x:x['repo']),
        repos_unreachable_or_missing=[x for x in errors if 'repo' in x], errors=errors,
        grid=dict(shape=[3730,3292], crs='EPSG:32611', transform=[100,0,243350,0,-100,4508550]),
        n_unique_grid_rasters=len(seen), complete_accessible_scan=not errors,
        skipped=excluded, rasters=sorted(seen.values(), key=lambda r:r['sha256']),
        scope='Historical pinned inventory union latest public main trees including this repository; inaccessible/private/unlinked rasters cannot be certified.')
    (root / 'evidence/registry_refreshed.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (root / 'evidence/site_inventory.json').write_text(json.dumps(dict(generated_utc=report['generated_utc'], repos=sorted(reviews,key=lambda r:r['repo']), errors=errors), indent=2, allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=4)
    a=p.parse_args()
    r=refresh(workers=a.workers)
    print(json.dumps({k:r[k] for k in ('n_unique_grid_rasters','complete_accessible_scan','errors')}, indent=2))
    return 0 if r['complete_accessible_scan'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
