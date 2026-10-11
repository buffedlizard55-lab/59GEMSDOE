#!/usr/bin/env python3
"""Session H59-J step 1b -- which official layer do the *high-scoring* dot fields sit on?

Motivation (and why this is not the usual AUC table)
----------------------------------------------------
Every previous session measured which layers localise the *catalogue* (an AUC
against `existing_faults.tif`).  But the organiser masks the catalogue out of
scoring, and the owner-reported live scores vary 26-fold at a *constant*
37,654-dot budget -- so placement, not budget, is what separates the winners.
This script asks the complementary question: at the locations a given artifact
actually emitted, what is the empirical percentile of each official band?

Method
------
For each of the 19 officially provided layers, the footprint-wide distribution
is estimated from a fixed random sample of 400,000 footprint cells.  For each
artifact, the mean rank-percentile of that layer at the artifact's dots is
measured with `np.searchsorted` against the sampled distribution.  No fitting,
no truth raster, no hidden labels: this is a description of where the published
files put their mass.

Reading the output
------------------
A band is a candidate placement signal when the artifact-mean percentile rises
with the owner-reported score across the 60+ artifacts (Spearman).  Because the
scores are owner-reported claims and the artifacts are not independent (many
are edits of one another), the correlation is a screen, never a proof.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid   # noqa: E402

BAND_NAMES = [
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf",
    "ieq_n100a15", "cond_surf", "iso_grav_anom_hg", "det_elev_slope",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(ROOT / "evidence/live_ledger_merged.json"))
    ap.add_argument("--registry", default=str(ROOT / "evidence/registry_refreshed.json"))
    ap.add_argument("--features", default=str(ROOT / ".cache/features"))
    ap.add_argument("--out", default=str(ROOT / "evidence/dot_band_profile_h59j.json"))
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=59)
    args = ap.parse_args()

    grid = load_grid()
    foot = grid.footprint
    rng = np.random.default_rng(args.seed)
    fy, fx = np.nonzero(foot)
    pick = rng.choice(fy.size, size=min(args.sample, fy.size), replace=False)
    sy, sx = fy[pick], fx[pick]

    registry = json.loads(Path(args.registry).read_text())
    by_sha = {r["sha256"]: r for r in registry["rasters"]}
    ledger = json.loads(Path(args.ledger).read_text())["rows"]

    artifacts = []
    for row in ledger:
        rec = by_sha.get(row["sha256"])
        if rec is None:
            continue
        path = ROOT / rec["cache_file"]
        if not path.is_file():
            continue
        with rasterio.open(path) as src:
            a = src.read(1)
        dots = np.isfinite(a) & (a > 0)
        n = int(dots.sum())
        if n == 0:
            continue
        dy, dx = np.nonzero(dots)
        if dy.size > 250_000:                      # thin for speed, deterministic
            step = dy.size // 250_000 + 1
            dy, dx = dy[::step], dx[::step]
        artifacts.append(dict(name=row["name"], score=float(row["score"]), n_dots=n,
                              y=dy, x=dx))
    print(f"[profile] {len(artifacts)} artifacts with restored bytes")

    feats = Path(args.features)
    per_band = {}
    for b in range(1, 20):
        path = feats / f"band-{b:02d}.npy"
        if not path.is_file():
            print(f"[profile] band {b} cache missing -> skipped")
            continue
        band = np.load(path, mmap_mode="r")
        ref = np.asarray(band[sy, sx], dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        ref.sort()
        if ref.size < 1000:
            continue
        rows = []
        for art in artifacts:
            v = np.asarray(band[art["y"], art["x"]], dtype=np.float64)
            ok = np.isfinite(v)
            if ok.sum() < 100:
                continue
            pct = np.searchsorted(ref, v[ok], side="left") / ref.size
            rows.append(dict(name=art["name"], score=art["score"], n_dots=art["n_dots"],
                             mean_percentile=float(pct.mean()),
                             frac_above_p50=float((pct > 0.5).mean())))
        scores = np.array([r["score"] for r in rows])
        pcts = np.array([r["mean_percentile"] for r in rows])
        rho, p = (spearmanr(scores, pcts) if np.std(pcts) > 0 else (float("nan"), float("nan")))
        per_band[BAND_NAMES[b - 1]] = dict(band=b, n_artifacts=len(rows),
                                           spearman_score_vs_mean_percentile=float(rho),
                                           p_value=float(p), rows=rows)
        print(f"[profile] band {b:02d} {BAND_NAMES[b-1]:<20s} "
              f"rho={rho:+.3f} p={p:.4f}  (footprint median percentile at dots: "
              f"{np.median(pcts):.3f})")

    ranked = sorted(per_band.items(), key=lambda kv: -abs(kv[1]["spearman_score_vs_mean_percentile"]))
    result = dict(
        evidence_class="OWNER-REPORTED-LABEL + BAND-MEASUREMENT (not an organiser receipt)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/dot_band_profile.py",
        method=(f"footprint distribution of each official band sampled at {args.sample} cells "
                "(seed 59); mean rank-percentile measured at each artifact's emitted dots; "
                "Spearman across artifacts between that percentile and the owner-reported score"),
        caveat=("artifacts are not independent (many are edits of one another) and the scores are "
                "owner-reported claims, so these correlations are a screen for placement signals, "
                "not a fitted model and not a forecast"),
        n_artifacts=len(artifacts),
        ranked_bands=[dict(name=k, **{kk: vv for kk, vv in v.items() if kk != "rows"})
                      for k, v in ranked],
        per_band=per_band,
    )
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[profile] wrote {out}")
    for k, v in ranked[:8]:
        print(f"  |rho|={abs(v['spearman_score_vs_mean_percentile']):.3f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
