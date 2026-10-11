#!/usr/bin/env python3
"""Session H59-J experiment 2 -- emission design under a pre-declared criterion.

Pre-declared selection rule (fixed before the sweep was read)
-------------------------------------------------------------
The official metric collapses to ``DTI = T / (alpha*S + beta*G)`` (see
``docs/research/utility-derivation.md`` for the algebra).  With a proxy truth
the only quantity that is *calibrated* offline is the size of the scored
population, ``G``.  The off-catalogue SGMC proxy used here carries 61,664
pixels where the calibrated hidden population is ~12,700, so the raw proxy DTI
is dominated by coverage (a budget proxy) rather than by placement.

The rule applied here therefore replaces the proxy's own prevalence with the
calibrated one and keeps the proxy's *credit*:

    criterion(arm) = T_proxy / (alpha * S_proxy + beta * G_calibrated)

with ``alpha = 0.2``, ``beta = 0.8``, ``G_calibrated = 12,700`` (H28 / GEMSDOE39
inversions of the owner-reported anchors: 12,691 and 14,143), and
``truth = SGMC fault pixels > 300 m from the catalogue``.

This rule reproduces the *observed* within-family live ladder of the owner's own
thinnings (0.2778 > 0.2600 > 0.2477 > 0.1922), which is the only regime in which
any local instrument has been shown to track the live board.  It is a proxy
screen, never a score.

Two other readings are reported for every arm and never used for selection:
``dti_proxy_raw`` (the proxy's own prevalence) and ``credit_per_dot``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid                     # noqa: E402
from gems59.metric import ALPHA, BETA, RADIUS_PX, kernel   # noqa: E402

G_CALIBRATED = 12700.0
COLLAR_PX = 3.0


def greedy_nms(scores, coords, n_keep, min_dist):
    cell = max(min_dist, 1.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    kept = []
    for value, (yy, xx) in zip(scores, coords):
        gy, gx = int(yy // cell), int(xx // cell)
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (py, px) in grid.get((gy + dy, gx + dx), ()):
                    if (py - yy) ** 2 + (px - xx) ** 2 < min_dist * min_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gy, gx), []).append((yy, xx))
            kept.append((yy, xx))
            if len(kept) >= n_keep:
                break
    return np.array(kept, np.int64).reshape(-1, 2)


def stats(dots_yx, truth_yx):
    if dots_yx.size == 0 or truth_yx.size == 0:
        return dict(T=0.0, S=float(dots_yx.shape[0]), DTI_raw=0.0, credit_per_dot=0.0)
    dd, dt = cKDTree(dots_yx), cKDTree(truth_yx)
    d_t, _ = dd.query(truth_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    T = float(kernel(np.where(np.isfinite(d_t), d_t, RADIUS_PX)).sum())
    d_x, _ = dt.query(dots_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    M = float(kernel(np.where(np.isfinite(d_x), d_x, RADIUS_PX)).sum())
    S = T + dots_yx.shape[0] - M
    n = dots_yx.shape[0]
    g = truth_yx.shape[0]
    return dict(T=T, M=M, S=float(S), n=int(n), G=g,
                DTI_raw=float(T / (ALPHA * S + BETA * g + 1e-12)),
                DTI_calibrated=float(T / (ALPHA * S + BETA * G_CALIBRATED + 1e-12)),
                credit_per_dot=float(T / n), coverage=float(T / g))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default=str(ROOT / ".cache/oof_propensity.npy"))
    ap.add_argument("--out", default=str(ROOT / "evidence/emission_design_h59j.json"))
    ap.add_argument("--budgets", default="8000,12000,20000,30000,44090,60000")
    ap.add_argument("--spacings", default="1.0,2.0,2.5,3.0,4.0")
    args = ap.parse_args()
    t0 = time.time()

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    H, W = grid.shape
    with rasterio.open(ROOT / "data/external/sgmc_faults_100m.tif") as src:
        sgmc = src.read(1) == 1
    proxy = sgmc & foot & ~cat
    d_cat = ndi.distance_transform_edt(~cat)
    truth_far = np.argwhere(proxy & (d_cat > COLLAR_PX))
    truth_all = np.argwhere(proxy)
    print(f"[design] proxy truth all {truth_all.shape[0]}, beyond the kernel collar {truth_far.shape[0]}")

    oof = np.load(Path(args.oof), mmap_mode="r")
    flat = np.asarray(oof).ravel()
    keep_idx = np.flatnonzero(foot.ravel() & np.isfinite(flat))
    order = keep_idx[np.argsort(-flat[keep_idx])]
    top_n = min(order.size, 400_000)
    top = order[:top_n]
    coords = np.stack([top // W, top % W], 1).astype(np.int64)
    vals = flat[top]

    arms = []
    picks = {}
    for spacing in [float(s) for s in args.spacings.split(",")]:
        picked = greedy_nms(vals, coords, n_keep=top_n, min_dist=spacing)
        picks[spacing] = picked
        print(f"[design] spacing {spacing:.1f} px -> {len(picked)} candidates after NMS")
        for budget in [int(b) for b in args.budgets.split(",")]:
            if picked.shape[0] < budget:
                continue
            dots = picked[:budget]
            row = dict(spacing_px=spacing, budget=budget,
                       on_catalogue=int(cat[dots[:, 0], dots[:, 1]].sum()),
                       distance_p50_px=float(np.median(d_cat[dots[:, 0], dots[:, 1]])))
            row.update(stats(dots, truth_far))
            arms.append(row)
    for row in sorted(arms, key=lambda r: -r["DTI_calibrated"]):
        print(f"[design] sp={row['spacing_px']:.1f} n={row['budget']:6d} "
              f"crit={row['DTI_calibrated']:.4f} raw={row['DTI_raw']:.4f} "
              f"cred/dot={row['credit_per_dot']:.4f} cov={row['coverage']:.3f}")

    # marginal-credit curve on the winning spacing
    best = max(arms, key=lambda r: r["DTI_calibrated"])
    picked = picks[best["spacing_px"]]
    curve = []
    for k in (2000, 5000, 8000, 12000, 20000, 30000, 44090, 60000, 80000):
        if picked.shape[0] < k:
            continue
        st = stats(picked[:k], truth_far)
        marginal = None
        if curve:
            dT = st["T"] - curve[-1]["T"]
            dn = k - curve[-1]["n"]
            marginal = float(dT / dn) if dn else None
            if marginal is not None and not np.isfinite(marginal):
                marginal = None
        curve.append(dict(n=k, T=st["T"], S=st["S"], credit_per_dot=st["credit_per_dot"],
                          marginal_credit=marginal, DTI_calibrated=st["DTI_calibrated"]))
    print("[design] marginal credit along the winning spacing:")
    for c in curve:
        mg = c['marginal_credit']
        print(f"   n={c['n']:6d} T={c['T']:9.1f} cred/dot={c['credit_per_dot']:.4f} "
              f"marginal={'n/a' if mg is None else format(mg, '.4f')} "
              f"crit={c['DTI_calibrated']:.4f}")

    # reference arms on the identical instrument
    registry = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    by_sha = {r["sha256"]: r for r in registry["rasters"]}
    ledger = json.loads((ROOT / "evidence/live_ledger_merged.json").read_text())["rows"]
    reference = []
    for row in ledger:
        if not row["name"].startswith(("h33-h33-2-b2", "dotted-h19-5-d2-8",
                                       "h25-1-dotted-h19-5-d1-5",
                                       "h19-5-powerlaw-budget-multiline",
                                       "sgmc-off-catalogue-44k",
                                       "h18-4-usgs-geologic-map-faults-gap")):
            continue
        rec = by_sha[row["sha256"]]
        with rasterio.open(ROOT / rec["cache_file"]) as src:
            a = src.read(1)
        dots = np.argwhere(np.isfinite(a) & (a > 0))
        entry = dict(name=row["name"][:64], owner_reported_score=row["score"])
        entry.update(stats(dots, truth_far))
        reference.append(entry)
        print(f"[design] ref {entry['name'][:44]:<46s} n={entry['n']:7d} "
              f"crit={entry['DTI_calibrated']:.4f} raw={entry['DTI_raw']:.4f} "
              f"cred/dot={entry['credit_per_dot']:.4f} live={row['score']:.4f}")

    result = dict(
        evidence_class="PROXY-MEASUREMENT (owner-reported anchors; no organiser receipt)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/design_emission_h59j.py",
        selection_rule=("criterion = T_proxy / (0.2*S_proxy + 0.8*G_calibrated), "
                        "G_calibrated = 12700, truth = SGMC off-catalogue faults > 300 m from the "
                        "catalogue; pre-declared. Reproduces the observed within-family live ladder "
                        "(b2 > d2.8 > d1.5 > h19-5) which is the only regime where a local "
                        "instrument has been shown to track the live board."),
        G_calibrated=G_CALIBRATED,
        arms=arms, marginal_curve=curve, reference_arms=reference,
        selected=dict(spacing_px=best["spacing_px"], budget=best["budget"],
                      criterion=best["DTI_calibrated"], on_catalogue=best["on_catalogue"]),
        elapsed_s=float(time.time() - t0),
    )
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[design] selected spacing={best['spacing_px']} budget={best['budget']} "
          f"crit={best['DTI_calibrated']:.4f} -> {args.out} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
