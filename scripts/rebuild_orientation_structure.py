#!/usr/bin/env python3
"""Rebuild the descriptive hide-and-recover geometry receipt only.

This repairs a serialization collision in the orientation report: the scalar
withheld-positive count overwrote the relative-strike histogram's ``n_withheld``
array. It reuses the exact seed-20 buffered whole-component draw and the shared
visible-only geometry functions. It does not fit a model, calculate DTI, place
dots, change the experiment plan, or spend a competition slot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid
from gems59.anatomy import package_holdout_structure, relative_strike_distribution
from gems59.holdout import buffered_component_draw


def rebuild(root: Path = ROOT, seed: int = 20) -> dict:
    grid = load_grid()
    folds, _ = buffered_component_draw(grid, seed=seed)
    visible = folds[0]["visible"]
    hidden = folds[0]["hidden_all"]
    domain = np.logical_or.reduce([fold["region"] for fold in folds]) & ~folds[0]["masked_known"]
    active = domain & ~visible
    withheld = hidden[active]
    if not withheld.any():
        raise ValueError("fixed holdout draw has no scored withheld positives")

    # Match fold_geometry's visible-fault Euclidean distance column exactly.
    # fold_geometry stores the distance feature as float32; preserve that dtype
    # before quantiles so the repaired receipt reproduces the existing values.
    distances = ndi.distance_transform_edt(~visible)[active].astype(np.float32)
    if not np.isfinite(distances).all():
        raise ValueError("distance field contains non-finite values")
    rel = relative_strike_distribution(grid, visible, hidden)
    receipt = package_holdout_structure(
        rel,
        withheld_positive_pixels=int(withheld.sum()),
        distance_positive_quantiles_px=np.quantile(
            distances[withheld], [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
        distance_domain_quantiles_px=np.quantile(
            distances, [0.1, 0.5, 0.9, 0.95, 0.99]
        ).tolist(),
    )
    receipt.update(
        split_version="buffered-whole-components-loqo-v2",
        seed=seed,
        fold="pooled-hidden-components / visible-context draw",
        feature_buffer_px=3,
        model_fit_performed=False,
        dti_evaluated=False,
        production_dots_generated=False,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        correction="Restored the relative-strike histogram previously overwritten by a scalar count; no model/result values changed.",
    )
    path = root / "evidence" / "orientation_structure.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> int:
    receipt = rebuild()
    angles = receipt["relative_strike"]
    print(json.dumps({
        "evidence_class": receipt["evidence_class"],
        "withheld_positive_pixels": receipt["withheld_positive_pixels"],
        "relative_strike_valid_pixels": angles["n_withheld_total"],
        "relative_strike_histogram_sum": int(sum(angles["n_withheld"])),
        "relative_strike_edges_degrees": angles["edges"],
        "distance_positive_quantiles_px": receipt["distance_positive_quantiles_px"],
        "model_fit_performed": receipt["model_fit_performed"],
        "dti_evaluated": receipt["dti_evaluated"],
        "production_dots_generated": receipt["production_dots_generated"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
