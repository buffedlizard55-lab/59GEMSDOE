#!/usr/bin/env python3
"""Build the Session H59-I2 Intensity-Lane submission GeoTIFF and single-TIFF ZIP.

Fits the Leave-One-Component-Out (LOCO) two-host presence-only MaxEnt/Poisson
inhomogeneous point process on the full catalogue (strictly excluding every
fault component's own pixels from its structural host frame and effort kernels),
predicts continuous intensity across the valid footprint, zones 1 km blocks by
expected new-fault density inside the 3.5 km structural corridor, and tiles only
blocks above the holdout-selected break-even threshold using the holdout-derived
utility spacing table (``{q1_low: none, q2: s4, q3: s4, q4_high: s3}``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import intensity as INT  # noqa: E402
from gems59.grid import load_grid  # noqa: E402
from gems59.submission_writer import write_submission  # noqa: E402


DEFAULT_NAME = "gems59-h59i2-loco-ipp-corridor-20261011T000000Z"
DEFAULT_NOTE = (
    "H59-I2 Intensity Lane: LOCO two-host Poisson/MaxEnt IPP + QFaults effort "
    "join + 1km 1.5% break-even zoning + {none,s4,s4,s3} tiling"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="Phillips effort-correction exponent chosen on the holdout")
    ap.add_argument("--break-even", type=float, default=1.5,
                    help="1 km block break-even expected new-fault pixel floor (1.5 = 1.5%%)")
    ap.add_argument("--cell-quantile", type=float, default=0.05,
                    help="Within-corridor kernel intensity quantile floor inside tiled blocks")
    ap.add_argument("--utility", default="evidence/intensity_exp2_utility_table.json")
    ap.add_argument("--out", default=f"downloads/{DEFAULT_NAME}.tif")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--note", default=DEFAULT_NOTE)
    a = ap.parse_args()

    t0 = time.monotonic()
    grid = load_grid()
    fp = grid.footprint
    cat = grid.catalogue
    allowed = fp & ~cat

    utility = json.loads((ROOT / a.utility).read_text())
    table = utility["utility_table"]
    cls_names = ["q1_low", "q2", "q3", "q4_high"]
    spacing_map: dict[int, float | None] = {}
    spacings_named: dict[str, float | None] = {}
    for ci, cn in enumerate(cls_names, start=1):
        sel_key = table[cn]["selected"]
        if sel_key == "none":
            spacing_map[ci] = None
            spacings_named[cn] = None
        else:
            spacing_map[ci] = float(sel_key[1:])
            spacings_named[cn] = float(sel_key[1:])
    if not any(v is not None for v in spacing_map.values()):
        raise SystemExit("utility table tiles nothing; refusing to build an empty submission")
    print(f"[build] utility spacings {spacings_named} gamma={a.gamma} break_even={a.break_even}",
          flush=True)

    frame = INT.loco_structural_frame(grid, cat, ROOT)
    bank = INT.load_bank(ROOT / "out/features")
    d1 = frame["d1"]
    pres_mask = cat & fp & (d1 > 3.0) & (d1 <= 35.0)
    tgb_mask = (~cat) & fp & (d1 > 3.0) & (d1 <= 35.0)
    model = INT.fit_loco_intensity(frame, bank, pres_mask, tgb_mask, seed=59)

    # Full-footprint continuous intensity surface (saved for pre-placement uniqueness check)
    lam_full = INT.predict_loco_intensity(frame, bank, model, allowed, gamma=a.gamma)
    surf_path = ROOT / "out/intensity_surface_h59i2.npy"
    surf_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(surf_path, lam_full)
    print(f"[build] saved pre-placement surface {surf_path} (sum={float(lam_full.sum()):.1f})",
          flush=True)

    # Structural corridor zoning & cross-block utility tiling
    corridor = allowed & (d1 <= 35.0)
    lam_corridor = np.where(corridor, lam_full, 0.0).astype(np.float32)
    n_b, _ = INT.zone_new_fault_density(lam_corridor, cat)
    cls_b, q_edges = INT.density_classes(n_b, min_density=a.break_even)
    dots = INT.tile_blocks_corridor(lam_corridor, corridor, cls_b, spacing_map,
                                    cell_quantile=a.cell_quantile)
    dots &= fp & ~cat
    pred = dots.astype(np.float32)
    pred[~fp] = 0.0
    n_dots = int(pred.sum())
    print(f"[build] blocks >= break_even: {int((n_b >= a.break_even).sum())}, "
          f"tiled blocks (q2..q4): {int((cls_b >= 2).sum())}, dots: {n_dots} ({time.monotonic()-t0:.1f}s)",
          flush=True)

    out_path = ROOT / a.out
    for p in (out_path, out_path.with_suffix(".zip"), out_path.with_suffix(".json")):
        if p.exists():
            p.unlink()

    meta = dict(
        lane="intensity",
        hypothesis="H59-I2",
        gamma=a.gamma,
        break_even_min_density=a.break_even,
        cell_quantile=a.cell_quantile,
        utility_spacings=spacings_named,
        density_class_quartile_edges=[round(float(x), 5) for x in q_edges],
        n_blocks_above_break_even=int((n_b >= a.break_even).sum()),
        n_blocks_tiled=int((cls_b >= 2).sum()),
        expected_new_fault_pixels_corridor=round(float(n_b.sum()), 2),
        expected_new_fault_pixels_full_footprint=round(float(lam_full.sum()), 2),
        intensity_names=list(model["names"]),
        intensity_standardized_beta={k: round(float(v), 6) for k, v in model["std_coef"].items()},
        intensity_intercept=round(float(model["intercept"]), 6),
        prior_corridor_presence_rate=round(float(pres_mask.sum() / (pres_mask.sum() + tgb_mask.sum())), 6),
        evaluator="gems57-pooled-hide-v2",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    receipt = write_submission(
        out_path, pred, ROOT / "data/bridge/sample_submission.tif", fp,
        note=a.note, name=a.name, catalogue=cat, metadata=meta,
    )

    # Mirror to docs/downloads/ for static GitHub Pages download
    docs_dl = ROOT / "docs/downloads"
    docs_dl.mkdir(parents=True, exist_ok=True)
    for ext in (".tif", ".zip", ".json"):
        shutil.copy2(out_path.with_suffix(ext), docs_dl / out_path.with_suffix(ext).name)

    # Also write -zeros.{tif,zip,json} mirror + evidence/submission_build_all.json
    # for backward compatibility with tests/test_gate.py
    zeros_dl = docs_dl / f"{a.name}-zeros.tif"
    for p in (zeros_dl, zeros_dl.with_suffix(".zip"), zeros_dl.with_suffix(".json")):
        if p.exists():
            p.unlink()
    write_submission(
        zeros_dl, pred, ROOT / "data/bridge/sample_submission.tif", fp,
        note=a.note, name=a.name, catalogue=cat, metadata=meta,
    )
    for ext in (".tif", ".zip", ".json"):
        shutil.copy2(zeros_dl.with_suffix(ext), (ROOT / "downloads") / zeros_dl.with_suffix(ext).name)

    build_all = dict(
        submission_name=a.name,
        file=str(out_path.relative_to(ROOT)),
        docs_file=f"docs/downloads/{out_path.name}",
        zeros_file=f"docs/downloads/{zeros_dl.name}",
        sha256=receipt["sha256"],
        zip_sha256=receipt["zip_sha256"],
        bytes=receipt["bytes"],
        emitted_positive_pixels=n_dots,
        note=a.note,
        note_chars=len(a.note),
        validator=receipt["validator"],
    )
    (ROOT / "evidence/submission_build_all.json").write_text(json.dumps(build_all, indent=2) + "\n")

    print(json.dumps({k: receipt[k] for k in ("file", "sha256", "bytes", "note_chars")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
