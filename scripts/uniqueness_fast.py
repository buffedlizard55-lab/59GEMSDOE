#!/usr/bin/env python3
"""Run the shared, literal registry gate on a surface or final-dot GeoTIFF.

The old KD-tree report used stride-sampled rank statistics and then tried to
clear dense/support-asymmetric peers with an alternate representation rule.
That interpretation is not allowed by the standing protocol. This compatibility
entry point now delegates to ``gems59.gates.lane_uniqueness_report`` and the
shared exact inventory comparison. Dense positive support is not exempted.

A failed or incomplete gate exits non-zero. Passing this local indexed scan is
not organizer confirmation and does not itself authorize a weekly slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.gates import lane_uniqueness_report, write_report  # noqa: E402
from gems59.grid import load_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="candidate surface or final-dot GeoTIFF")
    parser.add_argument("--phase", choices=("surface", "dots"), default="dots")
    parser.add_argument("--registry", default="evidence/registry_refreshed_20261010T2001.json")
    parser.add_argument("--out", default=None, help="JSON receipt path")
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    out_path = Path(args.out) if args.out else ROOT / "evidence" / f"lane_gate_{args.phase}.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    grid = load_grid(ROOT / "data/official")
    with rasterio.open(candidate_path) as source:
        if source.count != 1:
            raise SystemExit("candidate must be single-band")
        values = source.read(1)
        if source.shape != grid.shape or source.crs != grid.crs or source.transform != grid.transform:
            raise SystemExit("candidate CRS/shape/transform differ from the pinned grid")
    index = json.loads(registry_path.read_text(encoding="utf-8"))
    report = lane_uniqueness_report(
        values,
        grid.footprint,
        sample=ROOT / "data/official/sample_submission.tif",
        registry_index=index,
        phase=args.phase,
    )
    report["candidate_file"] = str(candidate_path.relative_to(ROOT)) if candidate_path.is_relative_to(ROOT) else str(candidate_path)
    report["registry_index"] = str(registry_path.relative_to(ROOT)) if registry_path.is_relative_to(ROOT) else str(registry_path)
    write_report(out_path, report)
    print(json.dumps({
        "phase": args.phase,
        "ok": report["ok"],
        "duplicate": report["duplicate"],
        "registry_rasters_checked": report.get("registry_rasters_checked"),
        "complete_accessible_scan": report.get("complete_accessible_scan"),
        "worst_spearman_full_footprint": report.get("worst_spearman_full_footprint"),
        "worst_dot_overlap": report.get("worst_dot_overlap"),
        "stop_required": report.get("stop_required"),
        "report": str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
