#!/usr/bin/env python3
"""Session H59-J step 1 -- what actually moves the live score?

Question this answers, with measurements only
---------------------------------------------
The project already has two contradictory accounts of why the owner-reported
best file scores 0.2778: a catalogue-hide-and-recover account ("hug the visible
traces") and a live-evidence account ("stay away from the catalogue").  Both
were argued from small samples and neither was fitted to the whole score
ledger.  This script measures, for every artifact in the ledger that is
readable from the restored registry cache, the quantities the official metric
is actually made of, then tests how well those quantities rank the
owner-reported scores.

Exact identities used (verified algebra, see `src/gems59/metric.py`)
-------------------------------------------------------------------
For a binary dot field with ``n`` dots, ``G`` truth pixels inside the scored
domain and the official 300 m triangular kernel::

    T = TP_w = sum over truth pixels of the best kernel credit from any dot
    M       = sum over dots of the best kernel credit to any truth pixel
    S = TP_w + FP_w = T + n - M          (alpha + beta = 1)
    DTI = T / (alpha * S + beta * G)     alpha = 0.2, beta = 0.8

``T``, ``M`` and ``S`` are computed here with a KD-tree in exact kernel algebra,
not estimated.

Instruments
-----------
* ``catalogue`` -- the published USGS/INGENIOUS catalogue itself.  This is the
  truth the organiser masks out, so it is a *negative* control: it cannot be
  the live truth.
* ``proxy``  -- ``data/external/derived_sgmc_faults_100m.tif``, the in-repo
  off-catalogue SGMC-minus-catalogue raster.  It is a *proxy* for the hidden
  "new fault" population, never an organiser label.

Evidence classes
----------------
Owner-reported scores are copied from the project brief / sibling ledgers; they
are NOT organiser receipts.  Every number this script writes is labelled
OWNER-REPORTED-LABEL or PROXY-MEASUREMENT.  No value here is a competition
score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from gems59.grid import load_grid          # noqa: E402
from gems59.metric import ALPHA, BETA, RADIUS_PX   # noqa: E402

ZERO_TOL = 0.0


def kernel_from_dist(d: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - np.asarray(d, dtype=np.float64) / RADIUS_PX, 0.0)


def credit_algebra(dots_yx: np.ndarray, truth_yx: np.ndarray) -> dict:
    """Exact T, M, S, DTI for a binary dot field against one truth set.

    ``dots_yx``/``truth_yx`` are (N, 2) integer pixel coordinates.
    """
    n = int(dots_yx.shape[0])
    g = int(truth_yx.shape[0])
    if n == 0 or g == 0:
        return dict(n=n, G=g, T=0.0, M=0.0, S=float(n), DTI=0.0, coverage=0.0)
    dt = cKDTree(truth_yx)
    dd = cKDTree(dots_yx)
    # T: per truth pixel, best credit from any dot within the kernel
    dist_t, _ = dd.query(truth_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    kt = np.where(np.isfinite(dist_t), kernel_from_dist(dist_t), 0.0)
    T = float(kt.sum())
    # M: per dot, best credit to any truth pixel
    dist_x, _ = dt.query(dots_yx, k=1, distance_upper_bound=RADIUS_PX - 1e-9)
    kx = np.where(np.isfinite(dist_x), kernel_from_dist(dist_x), 0.0)
    M = float(kx.sum())
    S = T + n - M
    dti = T / (ALPHA * S + BETA * g + 1e-12)
    return dict(n=n, G=g, T=T, M=M, S=float(S), DTI=float(dti),
                coverage=float(T / g))


def load_ledger(path: Path) -> list[dict]:
    """Accept either the sibling ledger (``artifacts``) or the merged ledger (``rows``)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
        return [dict(id=r["name"], score=float(r["score"]), sha256=r["sha256"],
                     name=r["name"], provenance=r.get("provenance", "merged-ledger"))
                for r in rows if r.get("score") is not None]
    rows = data["artifacts"] if isinstance(data, dict) else data
    out = []
    for row in rows:
        if row.get("score") is None:
            continue
        out.append(dict(id=row["id"], score=float(row["score"]),
                        sha256=row["sha256"], name=Path(row["file"]).name,
                        provenance="sibling ledger"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(ROOT / "evidence/artifact_ledger_gemsdoe25.json"))
    ap.add_argument("--registry", default=str(ROOT / "evidence/registry_refreshed.json"))
    ap.add_argument("--out", default=str(ROOT / "evidence/ledger_forensics_h59j.json"))
    ap.add_argument("--max-raster", type=int, default=0,
                    help="0 = all ledger artifacts with a restored cache file")
    args = ap.parse_args()

    grid = load_grid()
    cat = grid.catalogue
    foot = grid.footprint
    H, W = grid.shape
    print(f"[forensics] footprint {int(foot.sum())} px, catalogue {int(cat.sum())} px")

    d_cat, (iy_cat, ix_cat) = ndi.distance_transform_edt(~cat, return_indices=True)

    proxy_path = ROOT / "data/external/derived_sgmc_faults_100m.tif"
    with rasterio.open(proxy_path) as src:
        proxy_all = src.read(1) > 0
    proxy = proxy_all & foot & ~cat
    proxy_far = proxy & (d_cat > RADIUS_PX)
    print(f"[forensics] proxy truth {int(proxy.sum())} px "
          f"({int(proxy_far.sum())} px > {RADIUS_PX:.0f} px from catalogue)")

    cat_yx = np.argwhere(cat)
    proxy_yx = np.argwhere(proxy)
    proxy_far_yx = np.argwhere(proxy_far)
    foot_yx = np.argwhere(foot)

    registry = json.loads(Path(args.registry).read_text())
    by_sha = {r["sha256"]: r for r in registry["rasters"]}

    ledger = load_ledger(Path(args.ledger))
    if args.max_raster:
        ledger = ledger[: args.max_raster]
    print(f"[forensics] ledger has {len(ledger)} scored artifacts")

    rows = []
    missing = []
    for entry in ledger:
        rec = by_sha.get(entry["sha256"])
        if rec is None:
            missing.append(dict(id=entry["id"], reason="sha256 not in registry index"))
            continue
        path = ROOT / rec["cache_file"]
        if not path.is_file():
            missing.append(dict(id=entry["id"], reason="cache file not restored",
                                cache_file=rec["cache_file"]))
            continue
        with rasterio.open(path) as src:
            a = src.read(1)
        dots = np.isfinite(a) & (a > ZERO_TOL)
        n = int(dots.sum())
        if n == 0:
            missing.append(dict(id=entry["id"], reason="no positive cells"))
            continue
        dots_yx = np.argwhere(dots)
        off_cat = dots & ~cat
        d_dots = d_cat[dots_yx[:, 0], dots_yx[:, 1]]
        # pure background control: truth = every footprint pixel is impossible
        # (12 M truth cells); the control used instead is the catalogue itself.
        res_cat = credit_algebra(dots_yx, cat_yx)
        res_proxy = credit_algebra(dots_yx, proxy_yx)
        res_proxy_far = credit_algebra(dots_yx, proxy_far_yx)
        row = dict(
            id=entry["id"], score_owner_reported=entry["score"], n_dots=n,
            cache_file=rec["cache_file"], sha256=entry["sha256"],
            on_catalogue_dots=int((dots & cat).sum()),
            frac_within_1px_cat=float((d_dots <= 1).mean()),
            frac_within_2px_cat=float((d_dots <= 2).mean()),
            frac_within_3px_cat=float((d_dots <= 3).mean()),
            distance_p10_px=float(np.percentile(d_dots, 10)),
            distance_p50_px=float(np.percentile(d_dots, 50)),
            distance_p90_px=float(np.percentile(d_dots, 90)),
            dti_catalogue=res_cat["DTI"], T_catalogue=res_cat["T"],
            dti_proxy=res_proxy["DTI"], T_proxy=res_proxy["T"],
            credit_per_dot_proxy=res_proxy["T"] / n,
            dti_proxy_far=res_proxy_far["DTI"], T_proxy_far=res_proxy_far["T"],
            credit_per_dot_proxy_far=res_proxy_far["T"] / n,
            S_proxy=res_proxy["S"], S_over_n_proxy=res_proxy["S"] / n,
        )
        rows.append(row)
        print(f"[forensics] {entry['id']:<12s} score={entry['score']:.4f} n={n:7d} "
              f"d50={row['distance_p50_px']:6.1f} dti_proxy={row['dti_proxy']:.4f} "
              f"cred/dot={row['credit_per_dot_proxy']:.4f}")

    if not rows:
        print("[forensics] nothing to fit")
        return 1

    scores = np.array([r["score_owner_reported"] for r in rows])
    n_dots = np.array([r["n_dots"] for r in rows], float)
    t_proxy = np.array([r["T_proxy"] for r in rows], float)
    t_proxy_far = np.array([r["T_proxy_far"] for r in rows], float)

    def rho(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return dict(spearman=None, p_value=None, n=int(len(a)))
        r, p = spearmanr(a, b)
        return dict(spearman=float(r), p_value=float(p), n=int(len(a)))

    correlations = dict(
        score_vs_n_dots=rho(scores, n_dots),
        score_vs_log_n_dots=rho(scores, np.log(n_dots)),
        score_vs_T_proxy=rho(scores, t_proxy),
        score_vs_T_proxy_far=rho(scores, t_proxy_far),
        score_vs_credit_per_dot_proxy=rho(scores, t_proxy / n_dots),
        score_vs_credit_per_dot_proxy_far=rho(scores, t_proxy_far / n_dots),
        score_vs_dti_proxy=rho(scores, np.array([r["dti_proxy"] for r in rows])),
        score_vs_frac_within_2px_cat=rho(scores, np.array([r["frac_within_2px_cat"] for r in rows])),
        score_vs_distance_p50=rho(scores, np.array([r["distance_p50_px"] for r in rows])),
    )

    # ---- two-parameter structural fit:  DTI ~= kappa * T_proxy / (alpha*S_proxy + beta*G_hat)
    # Fitted by grid search over G_hat (interpretable: the hidden truth count)
    # with kappa solved in closed form by least squares, and assessed by LOO.
    def fit_g(truth_credit: np.ndarray):
        best = None
        for g_hat in np.linspace(2000.0, 60000.0, 581):
            denom = ALPHA * (truth_credit + n_dots * 0.0) + BETA * g_hat
            # structural form: DTI = kappa * T / (alpha S + beta G); S is measured
            s_proxy = np.array([r["S_proxy"] for r in rows], float)
            denom = ALPHA * s_proxy + BETA * g_hat
            x = truth_credit / denom
            kappa = float(np.dot(x, scores) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
            pred = kappa * x
            sse = float(((scores - pred) ** 2).sum())
            if best is None or sse < best["sse"]:
                best = dict(G_hat=float(g_hat), kappa=kappa, sse=sse, pred=pred)
        # leave-one-out
        loo = []
        for i in range(len(rows)):
            keep = np.arange(len(rows)) != i
            g_hat, kappa = best["G_hat"], None
            s_proxy = np.array([r["S_proxy"] for r in rows], float)
            denom = ALPHA * s_proxy[keep] + BETA * g_hat
            x = truth_credit[keep] / denom
            kappa = float(np.dot(x, scores[keep]) / np.dot(x, x))
            loo.append(kappa * truth_credit[i] / (ALPHA * s_proxy[i] + BETA * g_hat))
        loo = np.array(loo)
        return dict(G_hat=best["G_hat"], kappa=best["kappa"], sse=best["sse"],
                    loo_mae=float(np.abs(loo - scores).mean()),
                    loo_pred=[round(float(v), 4) for v in loo],
                    pred=[round(float(v), 4) for v in best["pred"]],
                    spearman_pred_vs_score=float(spearmanr(best["pred"], scores)[0]))

    structural_fit = dict(
        note=("score ~= kappa * T_proxy / (alpha*S_proxy + beta*G_hat); T_proxy and S_proxy are "
              "MEASURED on the off-catalogue proxy truth, kappa and G_hat are the only fitted "
              "quantities and kappa is closed-form given G_hat. A 2-parameter fit on n artifacts "
              "is a screen, never a score."),
        proxy=fit_g(t_proxy),
        proxy_far=fit_g(t_proxy_far),
    )
    del iy_cat, ix_cat, foot_yx

    result = dict(
        evidence_class="OWNER-REPORTED-LABEL + PROXY-MEASUREMENT (no organiser receipt)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/ledger_forensics.py",
        ledger=dict(source=str(args.ledger), n_scored=len(ledger), n_measured=len(rows),
                    missing=missing),
        proxy_truth=dict(file=str(proxy_path.relative_to(ROOT)),
                         px=int(proxy.sum()), px_far=int(proxy_far.sum()),
                         far_definition=f">{RADIUS_PX:.0f} px (300 m) from the catalogue",
                         sha256=hashlib.sha256(proxy_path.read_bytes()).hexdigest()),
        grid=dict(shape=[H, W], footprint_px=int(foot.sum()),
                  catalogue_px=int(cat.sum())),
        correlations=correlations,
        structural_fit=structural_fit,
        rows=rows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[forensics] wrote {out}")
    for key, val in correlations.items():
        print(f"  {key:34s} rho={val['spearman']:+.3f} p={val['p_value']:.4f} n={val['n']}")
    for key in ("proxy", "proxy_far"):
        f = structural_fit[key]
        print(f"  fit[{key:9s}] G_hat={f['G_hat']:.0f} kappa={f['kappa']:.3f} "
              f"LOO MAE={f['loo_mae']:.4f} rho_pred={f['spearman_pred_vs_score']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
