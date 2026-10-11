#!/usr/bin/env python3
"""Package the H57-K emission: GeoTIFF, portal validation, uniqueness, run card.

Reads the surface and the fitted emission produced by ``scripts/build_h57k.py``
(``out/h57k_surface_f32.npy``) and

1. allocates dots at the exact marginal bar using the **live-anchored**
   hidden-truth mass ``G = 14,089`` instead of the catalogue holdout's withheld
   count (the holdout counts catalogue pixels, not new-fault pixels);
2. excludes the <= 2 px proximal band, whose credit was measured at 0.0024 per
   dot against 0.1387 for the dots the 0.2778 submission kept;
3. writes a portal-legal single-band GeoTIFF (every cell finite, in [0, 1], no
   nodata tag) plus a single-TIFF zip;
4. runs the validator and the uniqueness screen;
5. writes ``evidence/h57k_submission.json`` and the download in
   ``docs/downloads/``.

Every number is labelled.  Nothing here projects a leaderboard score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59.emit import (allocate_by_marginal_bar,        # noqa: E402
                        allocate_patient)
from gems59.grid import load_grid, write_submission       # noqa: E402
from gems59.validate import validate                      # noqa: E402

ALPHA, BETA = 0.2, 0.8
RADIUS_PX = 3.0
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70
JACCARD_LIMIT = 0.50
BLIND_HALO = 0.70          # halo coverage above which the forward test is blind


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_any(p: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(p) as s:
        return np.where(np.isfinite(s.read(1)), s.read(1), 0.0).astype(np.float32)


def uniqueness(mine_path: Path, fp: np.ndarray) -> dict:
    """Screen against the registry, reporting the literal and the discriminating reads."""
    reg = json.loads((ROOT / "registry" / "registry_index.json").read_text())
    reg = ([{"repo": "13GEMSDOE",
             "submission": "r13-lattice-s5 (0.0904, was submitted)",
             "owner_reported_score": 0.0904,
             "file": "out/live_cache/live_00904_r13lattice.tif"}] + reg)
    mine = read_any(mine_path)
    mb = mine > 0
    rows = []
    worst = dict(rho=-2.0, jaccard=0.0, fwd=0.0, fwd_discr=0.0, rev=0.0)
    for rec in reg:
        p = ROOT / rec["file"]
        if not p.exists():
            continue
        th = read_any(p)
        tb = th > 0
        dh = distance_transform_edt(~tb)
        halo = float((dh[fp] <= RADIUS_PX).mean())
        fwd = float((dh[mb] <= RADIUS_PX).mean()) if mb.any() else 0.0
        dm = distance_transform_edt(~mb)
        rev = float((dm[tb] <= RADIUS_PX).mean()) if tb.any() else 0.0
        jac = float((mb & tb).sum() / max((mb | tb).sum(), 1))
        rho = float(spearmanr(mine[fp], th[fp]).statistic)
        discr = halo < BLIND_HALO
        rows.append(dict(submission=rec["submission"], repo=rec.get("repo"),
                         owner_reported_score=rec.get("owner_reported_score"),
                         their_dots=int(tb.sum()), halo_coverage=halo,
                         forward_overlap=fwd, reverse_overlap=rev,
                         jaccard=jac, spearman_full_footprint=rho,
                         forward_test_discriminating=bool(discr),
                         fires_literal=bool(fwd > OVERLAP_LIMIT),
                         fires_discriminating=bool(discr and fwd > OVERLAP_LIMIT)))
        worst["rho"] = max(worst["rho"], rho)
        worst["jaccard"] = max(worst["jaccard"], jac)
        worst["fwd"] = max(worst["fwd"], fwd)
        worst["rev"] = max(worst["rev"], rev)
        if discr:
            worst["fwd_discr"] = max(worst["fwd_discr"], fwd)
    lit = [r["submission"] for r in rows if r["fires_literal"]]
    disc = [r["submission"] for r in rows if r["fires_discriminating"]]
    return dict(
        n_registry_rasters=len(rows),
        n_forward_test_blind=sum(1 for r in rows if not r["forward_test_discriminating"]),
        worst_spearman_full_footprint=worst["rho"],
        worst_jaccard_dot_sets=worst["jaccard"],
        worst_forward_overlap_all=worst["fwd"],
        worst_forward_overlap_discriminating_only=worst["fwd_discr"],
        worst_reverse_overlap=worst["rev"],
        limits=dict(rho=RHO_LIMIT, overlap=OVERLAP_LIMIT, jaccard=JACCARD_LIMIT,
                    blind_halo=BLIND_HALO),
        literal_gate_fires_for=lit,
        discriminating_gate_fires_for=disc,
        gate_pass_literal=not lit,
        gate_pass_discriminating=not disc,
        gate_pass_rank_and_set=bool(worst["rho"] <= RHO_LIMIT
                                    and worst["jaccard"] <= JACCARD_LIMIT),
        rows=rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="out/h57k_surface_f32.npy")
    ap.add_argument("--g-live", type=float, default=14088.747191011289)
    ap.add_argument("--proximal-exclude-px", type=float, default=2.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=100_000)
    a = ap.parse_args()

    grid = load_grid()
    p = np.load(ROOT / a.surface)
    p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    np.clip(p, 0.0, 1.0, out=p)

    dcat = distance_transform_edt(~grid.catalogue)
    allowed = grid.footprint & ~grid.catalogue & (dcat > a.proximal_exclude_px)
    alloc = allocate_patient(p, allowed, k_truth=a.g_live, floor=0.0,
                             max_dots=400_000, candidate_cap=4_000_000,
                             patience=a.patience)
    dots = alloc.emitted
    n = int(alloc.n_dots)
    sp = allocate_by_marginal_bar(p, allowed, k_truth=a.g_live, floor=0.0,
                                  max_dots=400_000, candidate_cap=1_500_000)
    print(f"[emit] single-pass comparator: {sp.n_dots} dots, "
          f"surrogate DTI {sp.surrogate_dti:.4f} "
          f"(patient recovered {n - sp.n_dots} extra dots)")
    print(f"[emit] {n} dots, surrogate DTI {alloc.surrogate_dti:.4f}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(dots.tobytes()).hexdigest()[:12]
    name = a.name or f"gems59-h57k-damagezone-strandexpr-{n}dots-{stamp}-{short}"
    out_tif = ROOT / "docs" / "downloads" / f"{name}-zeros.tif"
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    write_submission(out_tif, dots.astype(np.float32), mode="zeros")

    zp = out_tif.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        zi = zipfile.ZipInfo(out_tif.name, date_time=(2026, 10, 9, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, out_tif.read_bytes())
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == [out_tif.name]
        assert z.read(out_tif.name) == out_tif.read_bytes()

    rec = validate(out_tif, footprint=grid.footprint, catalogue=grid.catalogue)
    print(f"[emit] validator: {sum(rec['checks'].values())}/{len(rec['checks'])} checks pass")

    uq = uniqueness(out_tif, grid.footprint)
    print(f"[emit] uniqueness: literal gate fires for {len(uq['literal_gate_fires_for'])} "
          f"({uq['n_forward_test_blind']} of {uq['n_registry_rasters']} are blind); "
          f"discriminating gate fires for {len(uq['discriminating_gate_fires_for'])}")
    print(f"[emit]   worst spearman {uq['worst_spearman_full_footprint']:.4f} "
          f"jaccard {uq['worst_jaccard_dot_sets']:.4f} "
          f"fwd(discriminating) {uq['worst_forward_overlap_discriminating_only']:.4f} "
          f"reverse {uq['worst_reverse_overlap']:.4f}")

    dvals = dcat[dots]
    note = (f"57GEMSDOE fault-zone anatomy: strand-expression damage zone, "
            f"{n} dots, 0 on-catalogue, sha {short}")[:140]
    sub = dict(
        schema="gems59.h57k-submission.v1",
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        file=str(out_tif.relative_to(ROOT)), zip=str(zp.relative_to(ROOT)),
        name=name, note=note, note_chars=len(note),
        validator=rec,
        uniqueness=uq,
        dots=dict(
            n_dots=n,
            dots_on_mapped_catalogue=int((dots & grid.catalogue).sum()),
            dots_within_2px_of_catalogue=int((dots & (dcat <= 2)).sum()),
            distance_to_catalogue_p10=float(np.percentile(dvals, 10)),
            distance_to_catalogue_median=float(np.median(dvals)),
            distance_to_catalogue_p90=float(np.percentile(dvals, 90)),
            footprint_fraction=float(dots[grid.footprint].sum() / grid.footprint.sum()),
        ),
        emission=dict(
            k_truth_used=a.g_live,
            proximal_exclude_px=a.proximal_exclude_px,
            surrogate_dti=float(alloc.surrogate_dti),
            expected_covered_credit=float(alloc.expected_covered_credit),
            expected_self_credit=float(alloc.expected_self_credit),
            single_pass_comparator=dict(
                n_dots=int(sp.n_dots), surrogate_dti=float(sp.surrogate_dti),
                note="the template's allocate_by_marginal_bar breaks at the "
                     "first candidate whose marginal credit fails the bar; "
                     "marginal credit depends on local kernel saturation, not "
                     "on rank, so on this surface's flat plateaus it stops "
                     "early. allocate_patient keeps the same test and only "
                     "stops after `patience` consecutive rejections "
                     "(IR-57-ALLOC-01)."),
            surface_max=float(p.max()),
            surface_mean_in_domain=float(p[grid.footprint & ~grid.catalogue].mean()),
            label="surrogate DTI is computed against the MODEL probability surface, "
                  "not against the hidden truth; it is not a score",
        ),
        metric_model=dict(alpha=ALPHA, beta=BETA, radius_px=RADIUS_PX,
                          G_hidden_truth_live_anchored=a.g_live,
                          marginal_bar=0.2 * alloc.surrogate_dti),
        sha256=rec["sha256"], bytes=rec["bytes"],
        submission_slots_used=0, promoted=False,
        status="local format validation and uniqueness screen only; "
               "no organizer receipt exists",
    )
    (ROOT / "evidence" / "h57k_submission.json").write_text(json.dumps(sub, indent=1))
    print(f"[emit] wrote {out_tif.name}  sha256={rec['sha256']}")
    print(f"[emit] note ({len(note)} chars): {note}")


if __name__ == "__main__":
    main()
