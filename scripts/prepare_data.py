"""Inventory the competition rasters in data/ and write data/manifest.json (git-ignored).

Checks every raster against the official facts (DrivenData problem page, fetched 2026-10-11):
EPSG:32611, 100 m resolution. It also reports the grid size, which the reference notebook prints
as width 3292 x height 3730 for its features and labels rasters (github.com/drivendataorg/
gems-prize-reference-solution, stored notebook output). Mismatches are reported, not corrected.

Usage:  python scripts/prepare_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

EXPECTED = {"epsg": 32611, "res": 100.0, "width": 3292, "height": 3730}
ALIASES = {
    "training_features": ["training_features.tif", "numeric_features.tif", "gems-geodawn-numerical-features.tif"],
    "labels": ["labels.tif", "existing_faults.tif"],
    "template": ["sample_submission.tif", "example_submission.tif"],
}


def describe(path: Path) -> dict:
    with rasterio.open(path) as src:
        info = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": list(src.dtypes),
            "nodata": src.nodata,
            "crs": src.crs.to_string() if src.crs else None,
            "epsg": src.crs.to_epsg() if src.crs else None,
            "res": list(src.res),
            "transform": list(src.transform)[:6],
            "band_descriptions": list(src.descriptions),
        }
    info["matches_expected_grid"] = (
        info["epsg"] == EXPECTED["epsg"]
        and abs(info["res"][0]) == EXPECTED["res"]
        and info["width"] == EXPECTED["width"]
        and info["height"] == EXPECTED["height"]
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    data = Path(args.data_dir)
    report = {"expected": EXPECTED, "rasters": {}, "roles": {}, "missing_roles": []}
    for f in sorted(data.glob("*.tif")):
        try:
            report["rasters"][f.name] = describe(f)
        except Exception as exc:  # report and continue
            report["rasters"][f.name] = {"error": str(exc)}
    for role, names in ALIASES.items():
        found = next((n for n in names if (data / n).exists()), None)
        report["roles"][role] = found
        if found is None:
            report["missing_roles"].append(role)
    data.mkdir(parents=True, exist_ok=True)
    (data / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["missing_roles"]:
        raise SystemExit(f"missing: {report['missing_roles']} (see docs/index.html#data)")


if __name__ == "__main__":
    main()
