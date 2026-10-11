#!/usr/bin/env python3
"""Fault-zone-anatomy lane driver.

Stages
------
``verify``   re-verify the two competition files by sha256 and geometry
``measure``  build the hide-and-recover holdout and measure the withheld
             distance / stepover / along-strike / relative-strike / length
             distributions (the measurement the lane protocol requires)
``canary``   leakage canary -- single-feature AUC per fold
``cv``       leave-one-quadrant-out fit + emit, pooled DTI
``final``    full-catalogue surface, emission, GeoTIFF write, portal validation

Every number printed here is a HOLDOUT-DTI instrument reading unless it is
labelled ORGANIZER-CONFIRMED (copied from a submission-page receipt); owner-pasted scores are OWNER-REPORTED.  Nothing
here is a projection written as a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import grid as gridmod                       # noqa: E402
from gems59 import load_grid                             # noqa: E402
from gems59.anatomy import FEATURES, fold_geometry        # noqa: E402
from gems59.anatomy import measure_withheld, relative_strike_distribution  # noqa: E402
from gems59.holdout import build_holdout, evaluate        # noqa: E402

CACHE = ROOT / ".cache"
EVID = ROOT / "evidence"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def stage_verify(_a) -> None:
    out = {}
    for name, expect in (("existing_faults.tif", gridmod.SHA256_EXISTING_FAULTS),
                         ("sample_submission.tif", gridmod.SHA256_SAMPLE_SUBMISSION)):
        p = gridmod.DATA_DIR / name
        got = sha256(p)
        out[name] = {"sha256": got, "expected": expect, "match": got == expect,
                     "bytes": p.stat().st_size}
        print(f"{name:24s} sha256={got[:16]}... match={got == expect} bytes={p.stat().st_size}")
    import rasterio
    for name in ("existing_faults.tif", "sample_submission.tif"):
        with rasterio.open(gridmod.DATA_DIR / name) as s:
            a = s.read(1)
            rec = {"crs": str(s.crs), "shape": [s.height, s.width],
                   "transform": list(tuple(s.transform)[:6]),
                   "dtype": s.dtypes[0], "nodata": None if s.nodata is None or np.isnan(s.nodata) else s.nodata,
                   "nodata_is_nan": bool(s.nodata is not None and np.isnan(s.nodata)),
                   "n_finite": int(np.isfinite(a).sum()), "n_nan": int(np.isnan(a).sum()),
                   "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                   "unique_values": np.unique(a[np.isfinite(a)])[:8].tolist()}
            out[name].update(rec)
            print(f"  {name}: {rec['crs']} {rec['shape']} {rec['dtype']} nodata={rec['nodata']} "
                  f"nan={rec['n_nan']} finite={rec['n_finite']} range=[{rec['min']},{rec['max']}] "
                  f"unique={rec['unique_values']}")
    g = load_grid()
    out["catalogue_px"] = int(g.catalogue.sum())
    out["footprint_px"] = int(g.footprint.sum())
    print(f"catalogue px = {out['catalogue_px']}   footprint px = {out['footprint_px']}")
    EVID.mkdir(exist_ok=True)
    (EVID / "verify_grid.json").write_text(json.dumps(out, indent=2))
    assert all(v["match"] for k, v in out.items() if isinstance(v, dict) and "match" in v)
    print("ALL SHA256 MATCH")


# --------------------------------------------------------------------------- #
def get_holdout(rebuild: bool = False):
    """Build the holdout fresh each run.

    Caching the context to disk was tried and dropped: 16 cells of full-grid
    masks exceed the 3 GB sandbox when serialised.  Rebuilding costs ~20 s.
    """
    t = time.time()
    g = load_grid()
    ctx = build_holdout(g)
    n = {m: len(ctx.cells_of(m)) for m in ("all", "detached")}
    print(f"holdout built in {time.time()-t:.1f}s  cells={len(ctx.cells)} {n}")
    for c in ctx.cells:
        print(f"  {c.key:26s} truth={c.n_truth:6d} domain={int(c.active.sum()):8d} "
              f"withheld={int(ctx.hidden_by_cell[c.key].sum()):6d}")
    return ctx


def full_domain(ctx, cell) -> np.ndarray:
    """Lift a cell's cropped ``active`` mask back onto the full grid."""
    m = np.zeros(ctx.grid.shape, bool)
    m[cell.bbox] = cell.active
    return m


def stage_measure(a) -> None:
    ctx = get_holdout(a.rebuild)
    per_cell = {}
    agg = {"distance": [], "stepover": [], "along_strike": [], "segment_length": [], "joint": []}
    side = {"n_withheld_left": 0, "n_withheld_right": 0,
            "n_domain_left": 0, "n_domain_right": 0}
    relstrike = {"n_withheld": [], "n_visible_reference": [], "edges": None}
    n_hid = n_dom = 0
    for cell in ctx.cells:
        if a.mode and cell.mode != a.mode:
            continue
        # cell.active is the scored domain with visible fault pixels removed, which
        # is exactly the population the features are defined on.
        dom = full_domain(ctx, cell)
        vis = ctx.visible(cell.key)
        g = fold_geometry(ctx.grid, vis, ctx.hidden_by_cell[cell.key], dom, cell.key)
        m = measure_withheld(g)
        rs = relative_strike_distribution(ctx.grid, vis, ctx.hidden_by_cell[cell.key])
        per_cell[cell.key] = {"mode": cell.mode,
                              "n_withheld": m["n_withheld"], "n_domain": m["n_domain"],
                              "base_rate": m["base_rate"],
                              "relstrike_median": rs["median_withheld"],
                              "relstrike_median_visible": rs["median_visible"],
                              "relstrike_n": rs["n_withheld_total"]}
        for k in ("distance", "stepover", "along_strike", "segment_length"):
            agg[k].append(m[k])
        agg["joint"].append(m)
        for k in side:
            side[k] += m["side"][k]
        relstrike["edges"] = rs["edges"]
        relstrike["n_withheld"].append(rs["n_withheld"])
        relstrike["n_visible_reference"].append(rs["n_visible_reference"])
        n_hid += m["n_withheld"]; n_dom += m["n_domain"]
        rsm = rs['median_withheld']
        rsv = rs['median_visible']
        print(f"{cell.key:26s} withheld={m['n_withheld']:6d} domain={m['n_domain']:8d} "
              f"base={m['base_rate']:.5f} relstrike_med="
              f"{('%.1f' % rsm) if rsm is not None else 'n/a':>5} "
              f"(visible ref {('%.1f' % rsv) if rsv is not None else 'n/a'}) "
              f"n_rel={rs['n_withheld_total']}")
        del g, dom, vis

    def sumdist(key):
        arr = agg[key]
        return {"edges": arr[0]["edges"],
                "n_withheld": np.sum([a["n_withheld"] for a in arr], axis=0).tolist(),
                "n_domain": np.sum([a["n_domain"] for a in arr], axis=0).tolist(),
                "enrichment": (np.sum([a["n_withheld"] for a in arr], axis=0)
                               / np.maximum(np.sum([a["n_domain"] for a in arr], axis=0), 1)).tolist()}

    rl = side['n_withheld_left'] / max(side['n_domain_left'], 1)
    rr = side['n_withheld_right'] / max(side['n_domain_right'], 1)
    side = {**side,
            "rate_left": rl, "rate_right": rr,
            "log_ratio_R_over_L": float(np.log(rr / max(rl, 1e-12)))}
    out = {
        "instrument": "hide-and-recover, 4 quadrants x draws 20/21, whole-segment withholding, "
                      "15 px collar, 12 px domain erosion, visible-faults-only features",
        "n_cells": len(ctx.cells),
        "n_withheld_total": n_hid, "n_domain_total": n_dom,
        "base_rate": n_hid / max(n_dom, 1),
        "distance": sumdist("distance"),
        "stepover": sumdist("stepover"),
        "along_strike": sumdist("along_strike"),
        "segment_length": sumdist("segment_length"),
        "side": side,
        "joint": {
            "stepover_edges": agg["joint"][0]["joint_stepover_alongstrike"]["stepover_edges"],
            "along_edges": agg["joint"][0]["joint_stepover_alongstrike"]["along_edges"],
            "n_withheld": np.sum([a["joint_stepover_alongstrike"]["n_withheld"] for a in agg["joint"]], axis=0).tolist(),
            "enrichment": (np.sum([a["joint_stepover_alongstrike"]["n_withheld"] for a in agg["joint"]], axis=0)
                           / np.maximum(np.sum([a["joint_stepover_alongstrike"]["n_domain"] for a in agg["joint"]], axis=0), 1)).tolist()},
        "relative_strike": {
            "edges": relstrike["edges"],
            "n_withheld": np.sum(relstrike["n_withheld"], axis=0).tolist(),
            "n_visible_reference": np.sum(relstrike["n_visible_reference"], axis=0).tolist(),
        },
        "mode_filter": a.mode or "both",
        "per_cell": per_cell,
    }
    EVID.mkdir(exist_ok=True)
    (EVID / "withheld_structure.json").write_text(json.dumps(out, indent=2))
    print(f"\npooled withheld={n_hid} domain={n_dom} base_rate={out['base_rate']:.5f}")
    print("\nDISTANCE (px) enrichment of withheld pixels vs domain:")
    e = out["distance"]["edges"]
    for i in range(len(e) - 1):
        lo = e[i]; hi = e[i + 1]
        hi = "inf" if hi > 1e8 else f"{hi:g}"
        print(f"  {lo:>5g}-{hi:>5} : n={out['distance']['n_withheld'][i]:7.0f} "
              f"enrich={out['distance']['enrichment'][i]:.4f}")
    print("\nSTEPOVER (cross-strike px):")
    e = out["stepover"]["edges"]
    for i in range(len(e) - 1):
        hi = "inf" if e[i + 1] > 1e8 else f"{e[i+1]:g}"
        print(f"  {e[i]:>5g}-{hi:>5} : n={out['stepover']['n_withheld'][i]:7.0f} "
              f"enrich={out['stepover']['enrichment'][i]:.4f}")
    print("\nALONG-STRIKE (px):")
    e = out["along_strike"]["edges"]
    for i in range(len(e) - 1):
        hi = "inf" if e[i + 1] > 1e8 else f"{e[i+1]:g}"
        print(f"  {e[i]:>5g}-{hi:>5} : n={out['along_strike']['n_withheld'][i]:7.0f} "
              f"enrich={out['along_strike']['enrichment'][i]:.4f}")
    print("\nSEGMENT LENGTH (log1p px):")
    e = out["segment_length"]["edges"]
    for i in range(len(e) - 1):
        hi = "inf" if e[i + 1] > 1e8 else f"{e[i+1]:g}"
        print(f"  {e[i]:>5g}-{hi:>5} : n={out['segment_length']['n_withheld'][i]:7.0f} "
              f"enrich={out['segment_length']['enrichment'][i]:.4f}")
    rl = side['n_withheld_left'] / max(side['n_domain_left'], 1)
    rr = side['n_withheld_right'] / max(side['n_domain_right'], 1)
    print(f"\nSIDE: withheld L={side['n_withheld_left']} R={side['n_withheld_right']} "
          f"| domain L={side['n_domain_left']} R={side['n_domain_right']}")
    print(f"  rate_left={rl:.6f} rate_right={rr:.6f} log(R/L)={np.log(rr/max(rl,1e-12)):+.4f}")
    print("\nJOINT enrichment (rows = cross-strike stepover px, cols = along-strike px):")
    j = out["joint"]
    se, ae = j["stepover_edges"], j["along_edges"]
    print("        " + "".join(f"{ae[i]:g}-{ae[i+1]:g}".rjust(9) for i in range(len(ae)-1)))
    for r in range(len(se)-1):
        row = "".join(f"{j['enrichment'][r][c]:9.4f}" for c in range(len(ae)-1))
        print(f"  {se[r]:g}-{se[r+1]:g}".ljust(8) + row)
    rw = np.array(out["relative_strike"]["n_withheld"]); rv = np.array(out["relative_strike"]["n_visible_reference"])
    c = out["relative_strike"]["edges"]
    print("\nRELATIVE STRIKE (deg) withheld vs visible-reference:")
    for i in range(len(c) - 1):
        pw = rw[i] / max(rw.sum(), 1); pv = rv[i] / max(rv.sum(), 1)
        print(f"  {c[i]:5.1f}-{c[i+1]:5.1f} : withheld {pw*100:5.2f}%  visible {pv*100:5.2f}%  ratio {pw/max(pv,1e-9):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verify", "measure", "canary", "cv", "final", "unique"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--mode", default="", choices=["", "all", "detached"],
                    help="restrict to one withholding mode")
    a = ap.parse_args()
    {"verify": stage_verify, "measure": stage_measure}[a.stage](a)


if __name__ == "__main__":
    if sys.argv[1:] == ["verify"]:
        stage_verify(None)
    else:
        raise SystemExit("Legacy experiment/final stages retired: they do not satisfy the current buffer, "
                         "budget or indexed public-inventory protocol. Use scripts/run_orientation_experiments.py "
                         "for declared research reproduction, not automatic slot promotion.")
