#!/usr/bin/env python3
"""Literal parallel-run gates for the H58 candidate, over the restored registry.

Two candidate arrays are audited in one pass over every hash-pinned registry
raster that is physically present in the workspace:

* the **surface before placement** (the fitted intensity, dense by construction);
* the **final dots** (the binary support actually written to the GeoTIFF).

Thresholds and the support definition are the inherited literal ones
(``:mod:`gems59.uniqueness``): signed full-footprint Spearman > 0.90, or more
than 70 % of candidate support within Euclidean 3 px of a prior's support.
Positive finite prediction > 0 is the support rule; no density or
reverse-overlap exemption is applied.

The script also profiles the registry itself, because the inherited protocol is
unsatisfiable for *any* non-empty candidate whenever a prior raster has positive
support on every allowed cell.  That fact is measured and reported here instead
of being argued from memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid                                              # noqa: E402
from gems59.grid import TRANSFORM                                          # noqa: E402
from gems59.uniqueness import (JACCARD_LIMIT, OVERLAP_LIMIT, RHO_LIMIT,     # noqa: E402
                               _dots, _jaccard, _near, _sha,
                               summarize_registry_rows)

EVID = ROOT / "evidence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(EVID / "registry_refreshed.json"))
    ap.add_argument("--dots", default=str(ROOT / ".cache/h58_dots.npy"))
    ap.add_argument("--surface", default=str(ROOT / ".cache/h58_prob.npy"))
    ap.add_argument("--tif", default="", help="written TIFF, for byte-hash identity checks")
    ap.add_argument("--out-prefix", default="h58")
    a = ap.parse_args()
    t0 = time.time()
    grid = load_grid()
    fp = grid.footprint
    index = json.loads(Path(a.index).read_text())
    records = index["rasters"]
    dots = np.load(a.dots).astype(bool)
    surface = np.load(a.surface).astype(np.float32)
    surface = np.where(fp, surface, 0.0)
    allowed = fp & ~grid.catalogue
    file_sha = _sha(Path(a.tif)) if a.tif else None

    cands = {}
    for name, arr in (("final_dots", dots.astype(np.float32)), ("surface_before_placement", surface)):
        support = _dots(arr) & fp
        values = arr[fp]
        ranks = rankdata(values).astype(np.float64)
        ranks -= ranks.mean()
        cands[name] = dict(arr=arr, support=support, n=int(support.sum()), ranks=ranks,
                           norm=float(np.linalg.norm(ranks)), near=_near(support),
                           decoded=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest())
        print(f"[gates] candidate {name}: support={cands[name]['n']} rank_norm>0="
              f"{cands[name]['norm'] > 0}", flush=True)

    rows = {k: [] for k in cands}
    profile = dict(rasters_indexed=len(records), read=0, missing=0, sha_mismatch=0,
                   universal_overlap_blockers=0, blocker_examples=[], support_hist=[])
    for i, rec in enumerate(records, 1):
        path = ROOT / rec["cache_file"]
        if not path.is_file():
            for k in rows:
                rows[k].append(dict(repo=rec["repo_first"], submission=rec["sources"][0],
                                    sha256=rec["sha256"], error="missing registry file; "
                                    "cannot certify uniqueness"))
            profile["missing"] += 1
            continue
        digest = _sha(path)
        if digest != rec["sha256"]:
            for k in rows:
                rows[k].append(dict(repo=rec["repo_first"], submission=rec["sources"][0],
                                    sha256=rec["sha256"], error=f"sha256 mismatch {digest}"))
            profile["sha_mismatch"] += 1
            continue
        with rasterio.open(path) as src:
            if src.count != 1 or src.shape != grid.shape or src.crs.to_epsg() != 32611 \
                    or src.transform != TRANSFORM:
                for k in rows:
                    rows[k].append(dict(repo=rec["repo_first"], submission=rec["sources"][0],
                                        sha256=rec["sha256"], error="grid/CRS/transform mismatch"))
                continue
            theirs = src.read(1).astype(np.float32)
        theirs = np.where(np.isfinite(theirs), theirs, 0.0)
        theirs[~fp] = 0.0
        support = (theirs > 0) & fp
        near_theirs = _near(support)
        tv = theirs[fp]
        dense = bool(near_theirs[allowed].all())
        profile["read"] += 1
        profile["support_hist"].append(int(support.sum()))
        if dense:
            profile["universal_overlap_blockers"] += 1
            if len(profile["blocker_examples"]) < 12:
                profile["blocker_examples"].append(dict(
                    repo=rec["repo_first"], submission=rec["sources"][0], sha256=rec["sha256"],
                    positive_pixels=int(support.sum()), allowed_pixels=int(allowed.sum())))
        decoded = hashlib.sha256(np.ascontiguousarray(theirs).tobytes()).hexdigest()
        binary = np.all((tv == 0) | (tv == 1))
        for name, c in cands.items():
            if c["norm"] == 0 or np.ptp(tv) == 0:
                rho = None
            elif binary:
                nt, n = int(support.sum()), tv.size
                rho = float(c["ranks"][support[fp]].sum() / (c["norm"] * np.sqrt(max(nt * (n - nt), 1) / n)))
            else:
                tr = rankdata(tv).astype(np.float64)
                tr -= tr.mean()
                rho = float(np.dot(c["ranks"], tr) / (c["norm"] * np.linalg.norm(tr)))
            ov = float(near_theirs[c["support"]].mean()) if c["n"] else 0.0
            rows[name].append(dict(repo=rec["repo_first"], submission=rec["sources"][0],
                                   file=path.name, sha256=digest, decoded_sha256=decoded,
                                   their_dots=int(support.sum()),
                                   my_dots_within_3px_of_theirs=ov,
                                   spearman_full_footprint=rho,
                                   jaccard_dot_sets=_jaccard(c["support"], support),
                                   identical_bytes=bool(file_sha and digest == file_sha),
                                   identical_decoded_predictions=bool(decoded == c["decoded"]),
                                   duplicate_by_rho=bool(rho is not None and rho > RHO_LIMIT),
                                   duplicate_by_overlap=bool(ov > OVERLAP_LIMIT),
                                   duplicate_by_jaccard=bool(_jaccard(c["support"], support) > JACCARD_LIMIT)))
        if i % 50 == 0:
            print(f"[gates] {i}/{len(records)} in {time.time()-t0:.0f}s", flush=True)

    summaries = {}
    for name, c in cands.items():
        summaries[name] = summarize_registry_rows(
            rows[name], registry_rasters_expected=len(records),
            complete_accessible_scan=profile["missing"] == 0 and profile["sha_mismatch"] == 0,
            source_errors=[], candidate_meta=dict(my_file=a.tif or None, my_dots=c["n"],
            candidate_decoded_sha256=c["decoded"], candidate_file_sha256=file_sha,
            candidate_rank_variation=bool(c["norm"] > 0)),
            scope=("Every hash-pinned grid raster of the indexed public owner-repository "
                   "inventory that is present in this workspace; not a complete organiser "
                   "registry. External, private, unlinked or later-written rasters may be absent."))
        save = EVID / f"{a.out_prefix}_uniqueness_{name}.json"
        save.write_text(json.dumps(summaries[name], indent=2, allow_nan=False) + "\n")
        print(f"[gates] {name}: unique={summaries[name]['unique']} "
              f"worst_rho={summaries[name]['worst_spearman_full_footprint']} "
              f"worst_overlap={summaries[name]['worst_dot_overlap']} "
              f"worst_jaccard={summaries[name]['worst_jaccard_dot_sets']} -> {save.name}", flush=True)
    profile.update(evidence_class="REGISTRY-MEASUREMENT", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   support_definition="finite prediction > 0 inside the sample footprint",
                   allowed_pixels=int(allowed.sum()),
                   universal_block_rule=("a prior whose 3 px dilation covers every allowed cell "
                                         "makes the 0.70 overlap gate unsatisfiable for any "
                                         "non-empty candidate"),
                   seconds=round(time.time() - t0, 1))
    (EVID / f"{a.out_prefix}_registry_profile.json").write_text(
        json.dumps(profile, indent=2, allow_nan=False) + "\n")
    print(f"[gates] registry: {profile['read']} read, {profile['universal_overlap_blockers']} "
          f"universal blockers, {profile['missing']} missing, {profile['sha_mismatch']} mismatched "
          f"in {profile['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
