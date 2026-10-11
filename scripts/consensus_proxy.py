#!/usr/bin/env python3
"""CONSENSUS-PROXY: transfer plausibility against the family's live-calibrated rasters.

Label discipline: this is NOT a score and never a HOLDOUT-DTI.  The registry's
top rasters (organizer-confirmed live scores 0.2632-0.2778) are independent
measurements of where the hidden live truth sits: each captured roughly 28-40%
of it with ~37-42k dots.  Where several of them *agree*, the live-truth density
is higher than any one of them alone.  A candidate whose dots land in the
consensus more often than budget-matched random dots has transfer plausibility;
one that does not is suspect regardless of its holdout numbers.

Statistic: for each candidate dot, does it lie within 3 px of >= K registry
top-scorer dots (K=2 default)?  Baseline: same statistic for uniform-random dot
sets of the same size inside the candidate's own support.

Output: evidence/consensus_proxy_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIVE_MIN = 0.26          # registry rasters at or above this are "top scorers"
K_MIN = 2                # agreement level (>= K top scorers within 3 px)
RADIUS_PX = 3.0
N_RANDOM = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path, help="GeoTIFF to test (binary dots)")
    ap.add_argument("--tag", default="candidate")
    ap.add_argument("--k", type=int, default=K_MIN)
    a = ap.parse_args()

    idx = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    tops = [r for r in idx if (r.get("owner_reported_score") or 0) >= LIVE_MIN]
    with rasterio.open(ROOT / "data" / "official" / "sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))
    with rasterio.open(a.candidate) as ds:
        cand = ds.read(1)
    cand = np.where(np.isfinite(cand), cand, 0.0)
    my_dots = cand > 0
    if not my_dots.any():
        raise SystemExit("candidate has no dots")

    # count of top scorers covering each pixel (within 3 px)
    agree_count = np.zeros(cand.shape, np.int16)
    top_dots = []
    for r in tops:
        p = ROOT / r["file"]
        if not p.exists():
            continue
        with rasterio.open(p) as ds:
            t = ds.read(1)
        t = np.where(np.isfinite(t), t, 0.0) > 0
        top_dots.append(t)
        d = distance_transform_edt(~t)
        agree_count += (d <= RADIUS_PX)
    consensus = agree_count >= a.k

    frac = float((my_dots & consensus).sum() / my_dots.sum())
    rng = np.random.default_rng(20261009)
    ys, xs = np.nonzero(valid)
    base = []
    n = int(my_dots.sum())
    for _ in range(N_RANDOM):
        pick = rng.choice(len(ys), size=n, replace=False)
        m = np.zeros(cand.shape, bool)
        m[ys[pick], xs[pick]] = True
        base.append(float((m & consensus).sum() / n))
    base = np.array(base)
    out = {
        "evidence_class": "CONSENSUS-PROXY (not a score; transfer plausibility only)",
        "candidate": str(a.candidate),
        "candidate_dots": n,
        "top_registry_rasters": [
            {"submission": r["submission"], "live_score": r["owner_reported_score"],
             "dots": int((t > 0).sum())}
            for r, t in zip([r for r in tops if (ROOT / r["file"]).exists()], top_dots)],
        "definition": f"fraction of candidate dots within {RADIUS_PX} px of >= {a.k} "
                      "top-registry (live >= 0.26) raster dots",
        "candidate_frac_in_consensus": frac,
        "random_baseline_mean": float(base.mean()),
        "random_baseline_std": float(base.std()),
        "random_baseline_max": float(base.max()),
        "enrichment_over_random": float(frac / max(base.mean(), 1e-9)),
        "beats_all_random_draws": bool(frac > base.max()),
        "verdict": "consensus-enriched (transfer plausible)" if frac > base.max()
                   else "not enriched over random (transfer doubtful on this proxy)",
    }
    EVID = ROOT / "evidence"
    EVID.mkdir(exist_ok=True)
    fp = EVID / f"consensus_proxy_{a.tag}.json"
    fp.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(out, indent=2))
    print(f"wrote {fp}")


if __name__ == "__main__":
    raise SystemExit("Historical runner retained for learning, not current buffered/inventory validation. "
                     "No new experiment or slot authorized; read the current README and run card.")
