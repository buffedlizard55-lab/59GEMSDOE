#!/usr/bin/env python3
"""Session-6 full-registry verification and top-score mechanism check.

Read-only with respect to the competition: no model fitting, no dot placement,
no GeoTIFF build, no download from DrivenData, no submission.

What it does
------------
1. Re-verifies every raster in ``REGISTRY_INDEX`` (the current 696-entry index, 2026-10-10T20:01Z)
   into ``.cache/registry/`` (gitignored, never committed) using the shared
   ``scripts/refresh_registry.fetch_blob``: immutable git-blob SHA1, pinned
   SHA256, single-band / EPSG:32611 / pinned-transform checks. A fetch that
   fails is reported, never silently counted as present.
2. Re-checks the git blob SHA1 of every cached file (the shared helper only
   checks it on download).
3. Profiles each raster against the competition footprint: positive-cell
   count (the literal gate's "dot" definition, finite > 0), footprint coverage,
   positive cells outside the footprint, value range, dtype.
4. Runs the shared ``gems59.uniqueness.saturation_certificate`` on the dense
   17GEMSDOE witness raster.
5. Checks the owner-reported 0.2708 -> 0.2778 pair at the file level:
   containment, removed dot count, and catalogue distance of removed dots.

Scores quoted here are OWNER-REPORTED labels from the brief; nothing in this
script produces or verifies a competition score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid  # noqa: E402
from gems59.uniqueness import saturation_certificate  # noqa: E402

DENSE_FRACTION = 0.5
WITNESS_BLOB = "374c88b1da2b803e2ed160de467fd71ae114010d"  # 17GEMSDOE E-proba-multiscale
OWNER_REPORTED_PAIR = {
    "lower": dict(label="gems28-h27-4-r1-solo-d2-8 (OWNER-REPORTED 0.2708)",
                  blob="12b0a4ddf1cf76f2552621b8a8b1a1b88d783786"),
    "higher": dict(label="gemsdoe32-h33-h33-2-b2 zeros (OWNER-REPORTED 0.2778)",
                   blob="17a76895f68174cc93f3cb1597d25686f6d6bc67"),
}


REGISTRY_INDEX = "evidence/registry_refreshed_20261010T2001.json"  # current index (696 unique grid rasters)

def _load_refresh_module():
    spec = importlib.util.spec_from_file_location("refresh_registry", ROOT / "scripts" / "refresh_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_array(a: np.ndarray, footprint: np.ndarray) -> dict:
    """Literal gate semantics: a 'dot' is a finite value > 0."""
    a = np.asarray(a)
    finite = np.isfinite(a)
    pos = finite & (a > 0)
    pos_in_fp = int((pos & footprint).sum())
    fp_n = int(footprint.sum())
    values = a[finite]
    return dict(
        finite_cells=int(finite.sum()),
        positive_cells=int(pos.sum()),
        positive_in_footprint=pos_in_fp,
        footprint_fraction_positive=pos_in_fp / fp_n if fp_n else 0.0,
        positive_outside_footprint=int((pos & ~footprint).sum()),
        all_finite_values_in_0_1=bool(values.size == 0 or ((values >= 0) & (values <= 1)).all()),
        dense=bool(fp_n and pos_in_fp / fp_n >= DENSE_FRACTION),
    )


def containment(base: np.ndarray, target: np.ndarray, catalogue: np.ndarray) -> dict:
    """Is ``target`` a pure subset of ``base``; what was removed; where."""
    base = np.asarray(base, bool)
    target = np.asarray(target, bool)
    dist = distance_transform_edt(~np.asarray(catalogue, bool))  # px to nearest mapped catalogue cell
    removed = base & ~target
    added = target & ~base
    out = dict(
        base_positive=int(base.sum()),
        target_positive=int(target.sum()),
        target_subset_of_base=bool(not added.any()),
        added_vs_base=int(added.sum()),
        removed_vs_base=int(removed.sum()),
        removed_on_catalogue=int((removed & np.asarray(catalogue, bool)).sum()),
        removed_distance_px_min=float(dist[removed].min()) if removed.any() else None,
        removed_distance_px_max=float(dist[removed].max()) if removed.any() else None,
        target_distance_px_min=float(dist[target].min()) if target.any() else None,
    )
    return out


def run(out_profile: Path, out_mechanism: Path, workers: int = 8) -> dict:
    index = json.loads((ROOT / REGISTRY_INDEX).read_text())
    records = index["rasters"]
    refresh = _load_refresh_module()
    cache = ROOT / ".cache" / "registry"
    cache.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    fp = grid.footprint

    def work(rec):
        try:
            _, reason = refresh.fetch_blob(rec, cache)  # fetch_blob raises on hash/grid problems
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            return dict(blob=rec["blob"], error=f"{type(exc).__name__}: {exc}")
        path = cache / f"{rec['blob']}.tif"
        if reason is not None:
            return dict(blob=rec["blob"], error=f"grid check: {reason}")
        row = dict(blob=rec["blob"], source=rec["sources"][0], n_sources=len(rec["sources"]),
                   pinned_sha256=rec["sha256"], sha256_matches_pin=sha256_file(path) == rec["sha256"],
                   git_blob_sha1_matches=git_blob_sha1(path) == rec["blob"], bytes=path.stat().st_size)
        with rasterio.open(path) as src:
            row.update(dtype=src.dtypes[0], bands=src.count, shape=list(src.shape),
                       crs_epsg=src.crs.to_epsg() if src.crs else None)
            a = src.read(1).astype(np.float32)
        row.update(profile_array(a, fp))
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, records))

    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    dense = [r for r in ok if r["dense"]]
    summary = dict(
        indexed=len(records), fetched_and_verified=len(ok), errors=len(errors),
        sha256_pin_mismatches=sum(not r["sha256_matches_pin"] for r in ok),
        git_blob_sha1_mismatches=sum(not r["git_blob_sha1_matches"] for r in ok),
        dense_rasters_ge_50pct_footprint=len(dense),
        rasters_with_positive_outside_footprint=sum(r["positive_outside_footprint"] > 0 for r in ok),
        rasters_with_values_outside_0_1=sum(not r["all_finite_values_in_0_1"] for r in ok),
        dtype_counts=dict(Counter(r["dtype"] for r in ok)),
        footprint_cells=int(fp.sum()),
    )
    profile = dict(
        evidence_class="REGISTRY-MEASUREMENT (verification of the indexed public owner-repository inventory; not a score)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        registry_index=REGISTRY_INDEX,
        registry_index_sha256=sha256_file(ROOT / REGISTRY_INDEX),
        definition="dot = finite value > 0 (inherited literal gate); footprint = finite cells of data/official/sample_submission.tif",
        summary=summary,
        errors=errors,
        rows=rows,
    )
    out_profile.write_text(json.dumps(profile, indent=1, allow_nan=False) + "\n")

    # Witness proof: a dense raster covering the footprint blocks every nonempty candidate.
    witness_path = cache / f"{WITNESS_BLOB}.tif"
    witness = saturation_certificate(witness_path, fp, catalogue=grid.catalogue)

    # Owner-reported pair mechanism (file-level, not causal).
    pair = {}
    for key, spec in OWNER_REPORTED_PAIR.items():
        with rasterio.open(cache / f"{spec['blob']}.tif") as src:
            a = src.read(1)
        pair[key] = dict(label=spec["label"], blob=spec["blob"],
                         positive=np.isfinite(a) & (a > 0))
    mech = containment(pair["lower"]["positive"], pair["higher"]["positive"], grid.catalogue)
    mech.update(lower_label=OWNER_REPORTED_PAIR["lower"]["label"],
                higher_label=OWNER_REPORTED_PAIR["higher"]["label"],
                note="Owner-reported labels only. File containment and distances are measured; the causal link to the score is NOT established (no organizer receipt).")
    witness_out = {k: v for k, v in witness.items()}
    out = dict(evidence_class="RASTER-MEASUREMENT + REGISTRY-MEASUREMENT (not scores)",
               generated_utc=profile["generated_utc"],
               witness_saturation=witness_out,
               owner_reported_pair_containment=mech)
    out_mechanism.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    return dict(summary=summary, witness=witness_out, mechanism=mech)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile-out", type=Path, default=ROOT / "evidence" / "registry_profile_session6.json")
    p.add_argument("--mechanism-out", type=Path, default=ROOT / "evidence" / "session6_mechanism_and_witness.json")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    result = run(args.profile_out, args.mechanism_out, workers=args.workers)
    print(json.dumps(result, indent=1, default=str))
    s = result["summary"]
    return 0 if s["errors"] == 0 and s["sha256_pin_mismatches"] == 0 and s["git_blob_sha1_mismatches"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
