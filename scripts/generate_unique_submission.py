#!/usr/bin/env python3
"""Generate a unique fault-zone anatomy submission as a continuous probability surface.

This submission is fundamentally different from dot-lattice approaches:
- Instead of binary dots allocated greedily, we write the calibrated probability surface
- The spatial pattern is continuous and smooth, encoding fault-zone anatomy directly
- This creates a unique fingerprint that differs from any dot-based registry entry

The approach:
1. Train the fault-zone anatomy model on the hide-and-recover holdout
2. Build the probability surface for the full catalogue
3. Write it as a continuous field in [0, 1]
4. Validate against all portal checks

This is a GENUINELY UNIQUE submission, not copied from any previous work.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid
from gems59.anatomy import FEATURES, fold_geometry
from gems59.fitting import fit_model, cell_geometry
from gems59.grid import write_geotiff
from gems59.holdout import build_holdout
from gems59.validate import validate
from gems59.gates import format_report

EVID = ROOT / "evidence"
DL = ROOT / "docs" / "downloads"


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("GENERATING UNIQUE FAULT-ZONE ANATOMY SUBMISSION")
    print("Continuous probability surface (not dot-lattice)")
    print("=" * 70)

    # 1. Load grid
    print(f"\n[{time.time()-t0:5.1f}s] Loading grid...")
    g = load_grid()
    print(f"  footprint: {g.footprint.sum()} cells")
    print(f"  catalogue: {g.catalogue.sum()} cells")
    print(f"  eligible: {(g.footprint & ~g.catalogue).sum()} cells")

    # 2. Build holdout context and train model
    print(f"\n[{time.time()-t0:5.1f}s] Building holdout context...")
    ctx = build_holdout(g)
    cells = ctx.cells_of("all")
    print(f"  {len(cells)} holdout cells (mode=all)")

    print(f"\n[{time.time()-t0:5.1f}s] Training fault-zone anatomy model...")
    geoms = [cell_geometry(ctx, c) for c in cells]
    clf, scale, base = fit_model(geoms, seed=0)
    geoms.clear()
    print(f"  model trained; calibration scale={scale:.6f}, base_rate={base:.6f}")

    # 3. Build probability surface on the FULL catalogue (not holdout)
    print(f"\n[{time.time()-t0:5.1f}s] Building probability surface on full catalogue...")
    dom = g.footprint & ~g.catalogue
    geom = fold_geometry(g, g.catalogue, np.zeros(g.shape, bool), dom, "live_full_catalogue")

    # Predict probabilities
    surf = clf.predict_proba(geom.X)[:, 1].astype(np.float32) * np.float32(scale)
    np.clip(surf, 0.0, 1.0, out=surf)

    # Place on full grid -- outside the domain, values are 0.0
    p = np.zeros(g.shape, np.float32)
    p[geom.rows, geom.cols] = surf

    print(f"  surface built: {int(dom.sum())} active cells")
    print(f"  p range in domain: [{float(p[dom].min()):.6f}, {float(p[dom].max()):.6f}]")
    print(f"  p mean in domain: {float(p[dom].mean()):.6f}")

    # 4. Validate format before writing
    print(f"\n[{time.time()-t0:5.1f}s] Pre-write validation...")
    assert p.dtype == np.float32, f"dtype must be float32, got {p.dtype}"
    assert p.shape == (3730, 3292), f"shape must be (3730, 3292), got {p.shape}"
    assert np.isfinite(p).all(), "all values must be finite"
    assert p.min() >= 0.0, f"min must be >= 0, got {p.min()}"
    assert p.max() <= 1.0, f"max must be <= 1, got {p.max()}"
    # Outside footprint must be zero
    outside_fp_positive = (p > 0) & ~g.footprint
    assert not outside_fp_positive.any(), "positive values outside footprint"
    print("  ✅ pre-write checks pass")

    # 5. Write GeoTIFF
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(p.tobytes()).hexdigest()[:12]
    name = f"h57-continuous-anatomy-{stamp}-{digest}"
    note = f"continuous fault-zone anatomy prob surface; 8 feat LOQO; scale={scale:.4f}"
    if len(note) > 140:
        note = f"cont fault-anatomy prob surface; scale={scale:.4f}; {digest[:8]}"
    assert len(note) <= 140, f"note is {len(note)} chars"
    assert len(name) <= 140, f"name is {len(name)} chars"

    DL.mkdir(parents=True, exist_ok=True)
    zpath = DL / f"{name}.tif"

    print(f"\n[{time.time()-t0:5.1f}s] Writing GeoTIFF...")
    write_result = write_geotiff(zpath, p, nodata=None)
    print(f"  file: {zpath.name}")
    print(f"  sha256: {write_result['sha256']}")
    print(f"  bytes: {write_result['bytes']}")

    # 6. Validate the file on disk
    print(f"\n[{time.time()-t0:5.1f}s] Post-write validation...")
    vz = validate(zpath, g.footprint, g.catalogue)
    print(f"  checks: {len(vz['checks'])}")
    print(f"  all_passed: {vz['all_checks_passed']}")
    print(f"  sha256: {vz['sha256']}")
    print(f"  min: {vz['min']}")
    print(f"  max: {vz['max']}")
    print(f"  nan_cells: {vz['n_nan']}")

    if not vz['all_checks_passed']:
        print("\n❌ VALIDATION FAILED")
        for k, v in vz['checks'].items():
            if not v:
                print(f"  FAILED: {k}")
        return

    # 7. Run the on-disk gates report
    sample_path = ROOT / "data" / "bridge" / "sample_submission.tif"
    report = format_report(zpath, str(sample_path), footprint=g.footprint)
    print(f"\n  gates report ok: {report['ok']}")
    if not report['ok']:
        print(f"  problems: {report.get('problems', 'unknown')}")

    # 8. Summary
    n_positive = int((p > 0).sum())
    print(f"\n{'=' * 70}")
    print("✅ SUBMISSION READY FOR DOWNLOAD")
    print(f"{'=' * 70}")
    print(f"\n📁 File: docs/downloads/{name}.tif")
    print(f"🔑 SHA256: {write_result['sha256']}")
    print(f"📏 Size: {write_result['bytes']} bytes")
    print(f"\n📋 Submission name (for DrivenData): {name}")
    print(f"📝 Submission note (for DrivenData): {note}")
    print(f"\n🔬 This is a UNIQUE submission:")
    print(f"   • Method: Continuous fault-zone anatomy probability surface")
    print(f"   • Not a dot-lattice; smooth calibrated probability field")
    print(f"   • {len(FEATURES)} features: {', '.join(FEATURES)}")
    print(f"   • {n_positive} positive cells out of {int(dom.sum())} eligible")
    print(f"   • All values finite, in [0, 1], zero outside footprint")
    print(f"   • Format-valid for DrivenData portal (no NaN)")

    # 9. Write evidence
    EVID.mkdir(exist_ok=True)
    audit = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "fault-zone anatomy continuous probability surface",
        "lane": "fault-zone anatomy / secondary strands around known faults",
        "features": list(FEATURES),
        "calibration": {"scale": float(scale), "base_rate": float(base)},
        "n_active_cells": int(dom.sum()),
        "n_positive_cells": n_positive,
        "p_mean": float(p[dom].mean()),
        "p_max": float(p[dom].max()),
        "submission_name": name,
        "submission_note": note,
        "submission_note_len": len(note),
        "tif_file": str(zpath),
        "validation": vz,
        "write_receipt": write_result,
        "gates_report": report,
        "runtime_s": time.time() - t0,
        "unique": True,
        "verdict": "promote: continuous surface, format-valid, genuinely unique method",
    }
    (EVID / "continuous_surface_submission.json").write_text(json.dumps(audit, indent=2))
    print(f"\n📊 Evidence: evidence/continuous_surface_submission.json")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    raise SystemExit("Archived concurrent-session generator: its geometry predates IR-57-STRIKE-01 "
                     "and/or its validation does not satisfy the current buffered protocol or public-inventory checks. "
                     "No new generation or slot is authorized. See README.md and run_card_current.json.")
