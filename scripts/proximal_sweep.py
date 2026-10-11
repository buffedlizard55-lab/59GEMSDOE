#!/usr/bin/env python3
"""Sensitivity of the H57-K emission to the proximal-exclusion radius.

The one instrument anchored to the real hidden truth (scripts/live_credit_shells.py)
measures the <= 2 px band at 0.0024 credit per dot against 0.1387 for the dots the
0.2778 submission kept, whose median distance to the catalogue is 19.6 px.  The
holdout model is trained on withheld *catalogue* pixels, so its high-probability
corridor sits close to the catalogue by construction.  Those two instruments pull
in opposite directions, so the exclusion radius is swept rather than assumed.

Every row is MODEL-SURROGATE, not a score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59.emit import allocate_patient      # noqa: E402
from gems59.grid import load_grid             # noqa: E402

G = 14088.747191011289


def main() -> None:
    grid = load_grid()
    p = np.load(ROOT / "out/h57k_surface_f32.npy")
    p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    dcat = distance_transform_edt(~grid.catalogue)
    base = grid.footprint & ~grid.catalogue
    rows = []
    for r in [2.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        allowed = base & (dcat > r)
        a = allocate_patient(p, allowed, k_truth=G, floor=0.0, max_dots=400_000,
                             candidate_cap=4_000_000, patience=100_000)
        d = dcat[a.emitted]
        rows.append(dict(proximal_exclude_px=r, allowed_cells=int(allowed.sum()),
                         n_dots=int(a.n_dots), surrogate_dti=float(a.surrogate_dti),
                         model_T=float(a.expected_covered_credit),
                         dcat_median=float(np.median(d)) if d.size else None,
                         dcat_p90=float(np.percentile(d, 90)) if d.size else None))
        print(f"exclude<={r:5.1f}px  dots={a.n_dots:7,d}  surrogate DTI={a.surrogate_dti:.4f}  "
              f"T={a.expected_covered_credit:8.1f}  dcat median="
              f"{np.median(d) if d.size else 0:5.1f} p90={np.percentile(d, 90) if d.size else 0:5.1f}",
              flush=True)
    (ROOT / "evidence" / "h57k_proximal_sweep.json").write_text(json.dumps(
        dict(schema="gems59.proximal-sweep.v1",
             note="MODEL-SURROGATE on the session-4 surface; not a score.",
             rows=rows), indent=1))


if __name__ == "__main__":
    main()
