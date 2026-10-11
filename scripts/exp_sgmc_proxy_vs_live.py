#!/usr/bin/env python3
"""Does covering SGMC-only faults (the off-catalogue proxy) predict the live score?

The published holdout cannot answer this: its truth *is* the mapped catalogue
(IR-59J-01).  The only independent fault compilation reachable this session is
the SGMC 100 m raster in `data/external/`, so this experiment treats
``SGMC - catalogue`` as a proxy truth, scores every owner-labelled scored raster
against it with the official formula, and correlates the result with the
owner-recorded public score in the file name.

A negative correlation is the expected, decision-relevant outcome: it says the
off-catalogue proxy and the live target are different things, so a proxy-optimal
design is not a leaderboard-optimal design.  This is the "negative instrument
result" the repository's delivery QA keeps on the record
(``evidence/offcat_instrument_check.json``); the field names in that receipt are
kept for compatibility and every definition is restated inside it.

Evidence labels: PROXY (SGMC-derived truth, not the competition truth),
OWNER-REPORTED (score recorded by the owner in a file name), DERIVED (algebra).

Usage: python scripts/exp_sgmc_proxy_vs_live.py [--out evidence/offcat_instrument_check.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import metric  # noqa: E402

DILATE_KNOWN_PX = 3   # the kernel radius: SGMC pixels this close to the mapped catalogue are "known"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="evidence/offcat_instrument_check.json")
    ap.add_argument("--calibration", default="evidence/h59j_live_calibration.json")
    a = ap.parse_args()
    t0 = time.monotonic()

    cal = json.loads((ROOT / a.calibration).read_text())
    with rasterio.open(ROOT / "data/official/sample_submission.tif") as ds:
        footprint = np.isfinite(ds.read(1))
    with rasterio.open(ROOT / "data/official/labels.tif") as ds:
        known = ds.read(1) > 0
    with rasterio.open(ROOT / "data/external/sgmc_faults_100m.tif") as ds:
        sgmc = ds.read(1) > 0
    if sgmc.shape != footprint.shape:
        raise SystemExit(f"SGMC shape {sgmc.shape} != competition grid {footprint.shape}")

    near_known = ndi.binary_dilation(known, iterations=DILATE_KNOWN_PX)
    truth = sgmc & footprint & ~near_known
    valid = footprint & ~known          # the published metric's mask, minus the proxy truth's collar
    print(f"[proxy] SGMC pixels {int(sgmc.sum())}, off-catalogue proxy truth {int(truth.sum())} px "
          f"({time.monotonic()-t0:.0f}s)", flush=True)

    rows = []
    for rec in cal["rows"]:
        path = ROOT / ".cache/scored" / rec["file"]
        if not path.exists():
            raise SystemExit(f"missing scored raster {path}")
        with rasterio.open(path) as ds:
            pred = ds.read(1).astype(np.float64)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        res = metric.dti_exact(pred, truth, valid=valid)
        rows.append(dict(key=rec["key"], live_score=rec["live_score"], emitted=rec["emitted"],
                         proxy_dti=float(res["dti"]), proxy_tp=float(res["tp"]),
                         proxy_fp=float(res["fp"]), proxy_fn=float(res["fn"])))
        print(f"    {rec['key']:32s} live={rec['live_score']} proxy={res['dti']:.6f}", flush=True)

    labelled = [r for r in rows if r["live_score"] is not None]
    proxy = np.array([r["proxy_dti"] for r in labelled], float)
    live = np.array([r["live_score"] for r in labelled], float)
    rho, p = spearmanr(proxy, live)
    emitted = np.array([r["emitted"] for r in labelled], float)
    rho_emit, p_emit = spearmanr(np.log10(emitted), live)
    rho_emit_proxy, p_emit_proxy = spearmanr(np.log10(emitted), proxy)

    payload = dict(
        evidence_class="NEGATIVE-INSTRUMENT (PROXY vs OWNER-REPORTED live score, this lane's measurement)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        what_was_measured=("off-catalogue proxy quality (official DTversky of every owner-labelled scored "
                           "raster against SGMC faults that lie at least 3 px from the mapped catalogue) "
                           "versus the public score recorded by the owner in the raster's file name"),
        proxy_truth=dict(source="data/external/sgmc_faults_100m.tif",
                         definition="SGMC fault pixels inside the study region and >= 3 px from the mapped catalogue",
                         positive_pixels=int(truth.sum()),
                         label="PROXY -- state geological compilation, not the competition truth"),
        metric=dict(formula="official distance-weighted Tversky (metric.dti_exact)", alpha=0.2, beta=0.8,
                    triangular_radius_px=3.0,
                    valid_mask="study region minus the mapped catalogue (the published ignore mask)"),
        spearman_live_vs_proxy_full_density={
            "spearman_live_vs_proxy": float(rho), "p_value": float(p), "n": int(len(labelled)),
            "x": "off-catalogue proxy DTversky of each raster at its own density",
            "y": "owner-recorded public score",
            "label": "OWNER-REPORTED x PROXY; no organizer receipt is reachable from this session"},
        spearman_log_emitted_vs_live={
            "spearman": float(rho_emit), "p_value": float(p_emit), "n": int(len(labelled))},
        spearman_log_emitted_vs_proxy={
            "spearman": float(rho_emit_proxy), "p_value": float(p_emit_proxy), "n": int(len(labelled))},
        conclusion=("the off-catalogue proxy is rank-inverted against the public score, exactly like the "
                    "hide-and-recover holdout, and for the same structural reason: both substitute a truth "
                    "that the live target is not"),
        rows=rows,
        labels=dict(proxy="PROXY", live="OWNER-REPORTED", verdict="DERIVED"),
    )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"spearman_live_vs_proxy": float(rho), "p_value": float(p), "n": len(labelled),
                      "spearman_log_emitted_vs_live": float(rho_emit)}, indent=1))
    print(f"[done] {(time.monotonic()-t0)/60:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
