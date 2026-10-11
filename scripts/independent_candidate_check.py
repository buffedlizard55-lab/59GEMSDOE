#!/usr/bin/env python3
"""Session-5 independent re-check of candidate GeoTIFFs (no new placement).

Re-derives the literal gates from src/gems59/uniqueness.py against the LOCAL
cache slice only (registry/rasters, 19 files). The separately indexed 679-row
public owner-repository inventory is not materialized here as rasters; neither
source establishes complete organizer coverage. Private, unlinked, external,
and otherwise inaccessible rasters may be absent. Self-matches (identical
sha256) are excluded and reported separately. Writes
evidence/independent_candidate_check.json.
"""
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np, rasterio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems59.uniqueness import dot_overlap, surface_rho, _dots  # shared functions, not a fork
from gems59.metric import RADIUS_PX

CANDIDATES = [
    "docs/downloads/gems59-h57r2-shipped8-all-flank0-20261009T180433Z-90e532947353-zeros.tif",
    "downloads/gems59-relative-strand-surface-20261009T195324Z-88e3bcb35a95.tif",
]
REF = ROOT / "data/official/sample_submission.tif"
OVERLAP_LIMIT, RHO_LIMIT, JACCARD_LIMIT = 0.70, 0.90, 0.50

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out = {"evidence_class": "RASTER-MEASUREMENT (not a score)", "radius_px": RADIUS_PX,
           "limits": dict(overlap=OVERLAP_LIMIT, rho_full_footprint=RHO_LIMIT, jaccard=JACCARD_LIMIT),
           "registry_scope": "19-file local cache slice; 679-row indexed public owner-repository inventory is not materialized here as rasters and is not a complete organizer registry; private, unlinked, external, and otherwise inaccessible rasters may be absent",
           "candidates": []}
    with rasterio.open(REF) as r:
        ref_shape, ref_tr, ref_crs = r.shape, tuple(r.transform)[:6], r.crs.to_string()
    registry = sorted(glob.glob(str(ROOT / "registry/rasters/*.tif")))
    for cp in CANDIDATES:
        cpath = ROOT / cp
        with rasterio.open(cpath) as d:
            m = d.read(1)
            fmt = dict(crs=d.crs.to_string(), shape=list(d.shape), dtype=d.dtypes[0], count=d.count,
                       transform_ok=tuple(d.transform)[:6] == ref_tr,
                       crs_ok=d.crs.to_string() == ref_crs, shape_ok=d.shape == ref_shape,
                       nan_count=int((~np.isfinite(m)).sum()),
                       min=float(np.nanmin(m)), max=float(np.nanmax(m)))
        valid = np.isfinite(m)
        c = dict(file=cp, sha256=sha(cpath), format=fmt, dots=int(_dots(m).sum()), comparisons=[])
        for rp in registry:
            t_sha = sha(rp)
            if t_sha == c["sha256"]:
                c["comparisons"].append(dict(registry=Path(rp).name, self_match=True))
                continue
            with rasterio.open(rp) as r:
                t = r.read(1)
            if t.shape != m.shape:
                c["comparisons"].append(dict(registry=Path(rp).name, error="shape mismatch")); continue
            s = surface_rho(m, t, valid)
            c["comparisons"].append(dict(registry=Path(rp).name, self_match=False,
                                         forward_overlap=dot_overlap(m, t), reverse_overlap=dot_overlap(t, m),
                                         rho_full_footprint=s["spearman_full_footprint"],
                                         jaccard=s["jaccard_dot_sets"]))
        non_self = [x for x in c["comparisons"] if not x.get("self_match") and "error" not in x]
        c["worst_forward_overlap_non_self"] = max(x["forward_overlap"] for x in non_self)
        c["worst_rho_non_self"] = max(x["rho_full_footprint"] for x in non_self)
        c["worst_jaccard_non_self"] = max(x["jaccard"] for x in non_self)
        c["triggered_non_self"] = [x["registry"] for x in non_self if x["forward_overlap"] > OVERLAP_LIMIT
                                   or x["rho_full_footprint"] > RHO_LIMIT or x["jaccard"] > JACCARD_LIMIT]
        c["literal_gate_pass_local_registry"] = len(c["triggered_non_self"]) == 0
        out["candidates"].append(c)
    p = ROOT / "evidence/independent_candidate_check.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    for c in out["candidates"]:
        print(c["file"].split("/")[-1][:60], "fwd_worst", round(c["worst_forward_overlap_non_self"], 4),
              "rho_worst", round(c["worst_rho_non_self"], 4), "pass", c["literal_gate_pass_local_registry"],
              "triggered", c["triggered_non_self"])
    print("wrote", p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
