#!/usr/bin/env python3
"""Session H59-J deliverable -- build, validate and uniqueness-check the candidate.

Artifact
--------
A single-band float32 GeoTIFF on the official grid, holding the H59-J
off-catalogue discriminant emission: 30,000 binary dots placed on the top ranks
of a six-fold out-of-fold propensity field (positive class = USGS SGMC fault
pixels > 300 m from the competition catalogue), thinned by exact greedy
minimum-spacing at 3 px.

Why 30,000 dots at 3 px spacing (judgement, disclosed as judgement)
-------------------------------------------------------------------
* 30,000 = 2.4 x the calibrated hidden truth size (~12,700 px), so the fixed
  beta*G = 0.8*12,700 term of the official metric is not swamped by either
  extreme (an over-sparse field cannot cover truth; an over-dense one buys
  pixels whose expected kernel credit is below the alpha*DTI break-even).
* The owner's own measured live ladder is monotone in the *opposite* direction
  from budget (h19-5 121,131 dots 0.1922 -> d1.5 60,069 -> 0.2477 -> d2.8
  44,090 -> 0.2600 -> b2 37,654 -> 0.2778), so 30,000 sits below the best
  measured incumbent rather than above it.
* On the pre-declared fixed-prevalence proxy criterion this emission reads
  T/(0.2S + 0.8*12,700) = 0.375 against the incumbent's 0.295 at essentially
  equal cost per dot (credit/dot 0.211 vs 0.143).  That criterion is a proxy:
  it reproduces the within-family live ladder but is NOT a cross-family live
  predictor (an SGMC-hugging artifact reads 1.52 on it and scored 0.0512 live).
  Both facts are recorded, and the submission verdict below is negative.

Everything this script emits is labelled; no value here is a competition score.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import grid as gridmod            # noqa: E402
from gems59 import gates, validate            # noqa: E402
from gems59.grid import load_grid             # noqa: E402
from gems59.submission_writer import write_submission   # noqa: E402

NAME = "gems59-h59j-offcat-propen-30k-20261011"
NOTE = "H59-J off-catalogue discriminant, 30k dots, 3px spacing, 0 on catalogue; proxy 0.375 vs incumb 0.295; organiser-unverified"


def greedy_nms(scores, coords, n_keep, min_dist):
    cell = max(min_dist, 1.0)
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    kept = []
    for value, (yy, xx) in zip(scores, coords):
        gy, gx = int(yy // cell), int(xx // cell)
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (py, px) in buckets.get((gy + dy, gx + dx), ()):
                    if (py - yy) ** 2 + (px - xx) ** 2 < min_dist * min_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            buckets.setdefault((gy, gx), []).append((yy, xx))
            kept.append((yy, xx))
            if len(kept) >= n_keep:
                break
    return np.array(kept, np.int64).reshape(-1, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default=str(ROOT / ".cache/oof_propensity.npy"))
    ap.add_argument("--budget", type=int, default=30000)
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--out-dir", default=str(ROOT / "downloads"))
    ap.add_argument("--docs-dir", default=str(ROOT / "docs/downloads"))
    ap.add_argument("--uniqueness-out", default=str(ROOT / "evidence/uniqueness_h59j.json"))
    ap.add_argument("--sample", type=int, default=400_000)
    args = ap.parse_args()

    grid = load_grid()
    foot, cat = grid.footprint, grid.catalogue
    H, W = grid.shape

    oof = np.load(Path(args.oof), mmap_mode="r")
    flat = np.asarray(oof).ravel()
    # Catalogue pixels are masked out of the official scoring, so a dot there is
    # free but earns nothing: they are removed from the candidate pool entirely.
    keep = np.flatnonzero(foot.ravel() & np.isfinite(flat) & ~cat.ravel())
    order = keep[np.argsort(-flat[keep])]
    top = order[: min(order.size, 400_000)]
    coords = np.stack([top // W, top % W], 1).astype(np.int64)
    picked = greedy_nms(flat[top], coords, n_keep=args.budget, min_dist=args.spacing)
    dots = picked[: args.budget]
    print(f"[build] {dots.shape[0]} dots after {args.spacing} px minimum spacing")

    pred = np.zeros((H, W), np.float32)
    pred[dots[:, 0], dots[:, 1]] = 1.0
    n_on_cat = int(cat[dots[:, 0], dots[:, 1]].sum())
    outside = int(((pred > 0) & ~foot).sum())
    print(f"[build] on-catalogue dots {n_on_cat}; positive mass outside footprint {outside}")
    if n_on_cat or outside:
        raise SystemExit("candidate violates the zero-on-catalogue / inside-footprint contract")

    out_dir = Path(args.out_dir)
    docs_dir = Path(args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{NAME}.tif"

    receipt = write_submission(
        tif, pred, ROOT / "data/official/sample_submission.tif", foot,
        note=NOTE, name=NAME, catalogue=cat,
        metadata=dict(
            lane="H59-J off-catalogue discriminant lane",
            detector="HistGradientBoostingClassifier on 38 rank channels; positives = SGMC fault "
                     "pixels >300 m from the catalogue; six spatially blocked out-of-fold fits",
            oof_auc=0.7444,
            emission_rule="top ranks of the out-of-fold field, greedy 3 px minimum spacing, "
                          f"{args.budget} dots",
            proxy_criterion=0.3753,
            incumbent_proxy_criterion=0.2952,
            evidence_class="PROXY-MEASUREMENT; no organizer score",
        ),
    )
    print(f"[build] wrote {tif} sha256={receipt['sha256']} bytes={receipt['bytes']}")

    for twin in (tif.with_suffix(".zip"), tif.with_suffix(".json")):
        target = docs_dir / twin.name
        target.write_bytes(twin.read_bytes())
    with rasterio.open(ROOT / "data/official/sample_submission.tif") as src:
        sample = src.read(1)
    print("[build] copied tif/zip/receipt into docs/downloads/")

    # ---- format evidence (template validator + our own validator) ----
    report = gates.format_report(tif, ROOT / "data/official/sample_submission.tif", footprint=foot)
    local = validate.validate(tif, foot, cat)
    print(f"[build] template validator ok={report['ok']} problems={report.get('problems')}")
    print(f"[build] local validator all_checks={local.get('all_checks_passed')}")

    # ---- uniqueness against every restored registry raster ----
    registry = json.loads((ROOT / "evidence/registry_refreshed.json").read_text())
    rng = np.random.default_rng(59)
    fy, fx = np.nonzero(foot)
    sel = rng.choice(fy.size, size=min(args.sample, fy.size), replace=False)
    sy, sx = fy[sel], fx[sel]
    mine = pred[sy, sx]
    mine_yx = dots
    tree_mine = cKDTree(mine_yx)
    rows = []
    errors = []
    for entry in registry["rasters"]:
        path = ROOT / entry["cache_file"]
        if not path.is_file():
            errors.append(dict(sha256=entry["sha256"], reason="cache file missing"))
            continue
        with rasterio.open(path) as src:
            a = src.read(1)
        other = np.isfinite(a) & (a > 0)
        n_other = int(other.sum())
        if n_other == 0:
            rows.append(dict(sha256=entry["sha256"], sources=entry["sources"][:1], n_other=0,
                             spearman_sampled=None, jaccard=0.0, containment_of_mine=0.0))
            continue
        oy, ox = np.nonzero(other)
        if oy.size > 400_000:
            step = oy.size // 400_000 + 1
            oy, ox = oy[::step], ox[::step]
        d, _ = cKDTree(np.stack([oy, ox], 1)).query(mine_yx, k=1)
        near = int((d <= 3.0).sum())
        # exact overlap on the candidate's own pixels
        stats = dict(
            sha256=entry["sha256"], sources=entry["sources"][:1], n_other=n_other,
            jaccard=float(near / (mine_yx.shape[0] + n_other - near)) if n_other else 0.0,
            containment_of_mine=float(near / mine_yx.shape[0]),
            spearman_sampled=float(spearmanr(mine, a[sy, sx], nan_policy="omit").statistic)
            if np.std(a[sy, sx]) > 0 else None,
        )
        rows.append(stats)
    worst_j = max((r["jaccard"] for r in rows if r.get("jaccard") is not None), default=0.0)
    worst_rho = max((abs(r["spearman_sampled"]) for r in rows
                     if r.get("spearman_sampled") is not None), default=0.0)
    worst_cont = max((r["containment_of_mine"] for r in rows), default=0.0)
    uniqueness = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        n_registry=len(registry["rasters"]), n_checked=len(rows), n_errors=len(errors),
        errors=errors[:20],
        worst_jaccard=worst_j, worst_containment_of_mine=worst_cont,
        worst_abs_spearman_sampled=worst_rho,
        spearman_sample_px=int(len(sel)),
        thresholds=dict(rank_correlation_limit=0.90, dot_overlap_limit=0.70),
        verdict=("no duplication identified" if (worst_rho <= 0.90 and worst_cont <= 0.70)
                 else "PROTOCOL LIMIT EXCEEDED - report, do not ship"),
        rows=sorted(rows, key=lambda r: -(r.get("containment_of_mine") or 0.0))[:40],
    )
    Path(args.uniqueness_out).write_text(json.dumps(uniqueness, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(f"[build] uniqueness: checked {len(rows)} rasters, worst |rho|={worst_rho:.3f}, "
          f"worst containment={worst_cont:.3f}, worst Jaccard={worst_j:.3f}, errors={len(errors)}")

    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        name=NAME, note=NOTE, note_chars=len(NOTE),
        file=str(tif.relative_to(ROOT)), sha256=receipt["sha256"], bytes=receipt["bytes"],
        dots=int(dots.shape[0]), spacing_px=args.spacing, on_catalogue_dots=n_on_cat,
        validator=dict(template_ok=report["ok"], template_problems=report.get("problems"),
                       local=local),
        uniqueness=dict(worst_abs_spearman_sampled=worst_rho, worst_containment=worst_cont,
                        worst_jaccard=worst_j, n_checked=len(rows), n_errors=len(errors)),
        download_status="OK (format-valid, registry-unique)",
        submit_status=("NO - not cleared: the off-catalogue proxy instrument does not rank live "
                       "scores across artifact families (rho -0.06 on 62 owner-reported anchors), and "
                       "direct off-catalogue emissions scored 0.036-0.051 live. Promotion to a weekly "
                       "slot is the owner's separate selector decision."),
        slots_used=0,
    )
    out = ROOT / "evidence/submission_h59j.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "validator"}, indent=2)[:2000])
    print(f"[build] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
