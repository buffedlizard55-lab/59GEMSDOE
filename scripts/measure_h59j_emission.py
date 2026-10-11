#!/usr/bin/env python3
"""Measure the SHIPPED H59-J raster against the proxy truth, in receipt form.

``scripts/design_emission_h59j.py`` scored *synthetic* arms drawn from the
out-of-fold field.  The build then shipped one arm at a smaller budget (30,000
dots) than the design's preferred 44,090.  This script measures that shipped
file itself with the same truth definition, the same kernel and the same
criterion, so the number in the release receipts is reproducible from the
GeoTIFF instead of quoted from a session log.  No fitting, no relaxation: it
only re-reads ``downloads/<name>.tif`` and the pinned SGMC raster.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid  # noqa: E402

NAME = "gems59-h59j-offcat-propen-30k-20261011"


def _design_module():
    spec = importlib.util.spec_from_file_location(
        "design_emission_h59j", ROOT / "scripts/design_emission_h59j.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", default=str(ROOT / "downloads" / f"{NAME}.tif"))
    ap.add_argument("--out", default=str(ROOT / "evidence/h59j_emission_measurement.json"))
    args = ap.parse_args()
    t0 = time.time()

    design = _design_module()
    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    with rasterio.open(ROOT / "data/external/sgmc_faults_100m.tif") as src:
        sgmc = src.read(1) == 1
    proxy = sgmc & foot & ~cat
    d_cat = ndi.distance_transform_edt(~cat)
    truth_far = np.argwhere(proxy & (d_cat > design.COLLAR_PX))
    truth_all = np.argwhere(proxy)

    with rasterio.open(args.tif) as src:
        raster = src.read(1)
    dots = np.argwhere(np.isfinite(raster) & (raster > 0)).astype(np.int64)
    on_catalogue = int(cat[dots[:, 0], dots[:, 1]].sum())
    outside = int(((raster > 0) & ~foot).sum())

    stats_far = design.stats(dots, truth_far)
    stats_all = design.stats(dots, truth_all)
    criterion = float(stats_far["T"] / (0.2 * stats_far["S"] + 0.8 * design.G_CALIBRATED))
    out = dict(
        evidence_class="PROXY-MEASUREMENT (SGMC off-catalogue proxy truth; no organizer score)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/measure_h59j_emission.py",
        file=str(Path(args.tif).relative_to(ROOT)) if Path(args.tif).is_absolute()
        else str(args.tif),
        truth_definitions=dict(
            far="SGMC fault pixels inside the footprint, off the mapped catalogue, "
                 f"more than {design.COLLAR_PX} px from it",
            all="SGMC fault pixels inside the footprint, off the mapped catalogue",
            far_pixels=int(truth_far.shape[0]), all_pixels=int(truth_all.shape[0])),
        criterion_rule=("T_proxy_far / (0.2 * S_proxy_far + 0.8 * G_calibrated), "
                        f"G_calibrated = {design.G_CALIBRATED:.0f}"),
        far=dict(criterion=criterion, **{k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                         for k, v in stats_far.items()}),
        all=dict(**{k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                    for k, v in stats_all.items()}),
        dots=int(dots.shape[0]), on_catalogue_dots=on_catalogue, positive_outside_footprint=outside,
        elapsed_s=round(time.time() - t0, 2),
    )
    Path(args.out).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps(dict(dots=out["dots"], criterion=round(criterion, 4), T=round(stats_far["T"], 1),
                          M=round(stats_far["M"], 1), S=round(stats_far["S"], 1),
                          credit_per_dot=round(stats_far["credit_per_dot"], 4)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
