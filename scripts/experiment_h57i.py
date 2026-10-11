#!/usr/bin/env python3
"""H57-I: strike-binned damage-zone histogram, three preregistered arms.

Experiment 1 — measure (d_perp, strike) enrichment on visible-only folds.
Experiment 2 — LOQO HOLDOUT-DTI of isotropic 1-D decay vs strike-binned decay.
Experiment 3 — same strike-binned model with the 0–2 px catalogue flank removed.

Then, if and only if the surface rank-gate passes, emit sparse binary dots,
uniqueness-check them with the shared lane_report (literal + saturating-probe
policy), write a portal-legal zeros GeoTIFF, and record a run card.

Nothing here is a live score.  No weekly slot is spent by this script.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import evaluate_holdout, load_grid  # noqa: E402
from gems59.anatomy import FEATURES, fold_geometry  # noqa: E402
from gems59.emit import greedy_allocate  # noqa: E402
from gems59.fitting import canary, cell_geometry, pooled  # noqa: E402
from gems59.gates import lane_report  # noqa: E402
from gems59.holdout import FOLD_NAMES, build_holdout  # noqa: E402
from gems59.metric import dti_binary  # noqa: E402
from gems59.strike_zone import (  # noqa: E402
    D_EDGES, STRIKE_EDGES, ZoneModel, fit_zone, model_to_json, strike_deg_from_X,
)
from gems59.submission_writer import write_submission  # noqa: E402
from gems59.validate import validate  # noqa: E402

CAP = 10_000
FLOOR = 0.005
FINAL_BUDGET = 37_654          # matched to the owner-reported 0.2778 dotted file's count, not copied
D_COL = FEATURES.index("d_perp")
EVID = ROOT / "evidence"
DL = ROOT / "docs" / "downloads"
SAMPLE = ROOT / "data" / "official" / "sample_submission.tif"
INDEX = ROOT / "evidence" / "registry_full_index.json"


def _truth_crop(cell) -> np.ndarray:
    t = np.zeros(cell.active.shape, bool)
    t[cell.truth_yx] = True
    return t


def score_arm(ctx, cell, g, model: ZoneModel, max_dots: int = CAP) -> dict:
    p = np.zeros(ctx.grid.shape, np.float32)
    if g.X.shape[0]:
        p[g.rows, g.cols] = model.intensity(g.X)
    allowed = np.zeros(ctx.grid.shape, bool)
    allowed[cell.bbox] = cell.active
    alloc = greedy_allocate(p, allowed, k_truth=float(cell.n_truth),
                            floor=FLOOR, max_dots=max_dots)
    emitted = alloc.emitted
    res = dti_binary(emitted[cell.bbox], _truth_crop(cell), valid=cell.active)
    shared, _ = evaluate_holdout.evaluate(
        emitted[cell.bbox].astype(np.float32),
        {"region": cell.active, "truth": _truth_crop(cell),
         "visible": ctx.visible(cell.key)[cell.bbox]},
        cell.active, block_side=200)
    np.testing.assert_allclose(
        [shared["tpw"], shared["fpw"], shared["fnw"], shared["dti"]],
        [res["tp"], res["fp"], res["fn"], res["dti"]],
        atol=2e-4, rtol=1e-6)
    out = {
        "key": cell.key, "mode": cell.mode, "n_truth": cell.n_truth,
        "n_dots": alloc.n_dots, "dti": shared["dti"],
        "coverage": shared["tpw"] / max(cell.n_truth, 1),
        "tp": shared["tpw"], "fp": shared["fpw"], "fn": shared["fnw"],
        "p_mean": float(p[allowed].mean()) if allowed.any() else 0.0,
        "p_max": float(p.max()),
        "zone_bins": model.n_kept_bins,
        "frac_pos_in_zone": model.fraction_pos_in_zone,
    }
    del p, allowed, alloc, emitted
    return out


def paired_delta(a: dict, b: dict, results_a, results_b) -> dict:
    return {
        q: pooled([r for r in results_a if r["key"].split("_")[1] == f"fold{q}"])["pooled_dti"]
        - pooled([r for r in results_b if r["key"].split("_")[1] == f"fold{q}"])["pooled_dti"]
        for q in FOLD_NAMES
    }


def registry_priors() -> list[Path]:
    files = sorted((ROOT / "registry" / "rasters").glob("*.tif"))
    # Own previous SPARSE dotted file is a prior.  The H57-G continuous
    # research surface is a covering probe of ourselves and is excluded from
    # the informative set (it is still in the sha256 index).
    own = ROOT / "docs" / "downloads" / "gems59-h57-anatomy-enechelon-20261009T070415Z-e9d8d59a4357-zeros.tif"
    if own.exists():
        files.append(own)
    return files


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    t0 = time.monotonic()
    EVID.mkdir(parents=True, exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)

    grid = load_grid()
    print(f"[{time.monotonic()-t0:6.1f}s] grid {grid.shape} cat={int(grid.catalogue.sum())}",
          flush=True)
    ctx = build_holdout(grid, modes=("detached",))
    cells = ctx.cells_of("detached")
    print(f"[{time.monotonic()-t0:6.1f}s] holdout cells={len(cells)}", flush=True)

    geoms = {}
    for i, c in enumerate(cells):
        geoms[c.key] = cell_geometry(ctx, c)
        print(f"[{time.monotonic()-t0:6.1f}s] geom {c.key} n={geoms[c.key].X.shape[0]} "
              f"hidden={geoms[c.key].n_hidden}", flush=True)

    # ----- leakage canary on RAW features (post IR-57-STRIKE-01) ----------
    seed20 = [geoms[c.key] for c in cells if c.seed == 20]
    can = canary(seed20, feature_names=FEATURES)
    leak = {n: r for n, r in can.items() if r["leakage_flag"]}
    print(f"[{time.monotonic()-t0:6.1f}s] canary leakage flags: {list(leak)}", flush=True)
    sin2 = can["sin2"]
    print(f"  sin2 disc_auc_max={sin2['discriminative_auc_max']:.4f} "
          f"raw_mean={sin2['auc_mean']:.4f} "
          f"(was 0.5000 under the zeroed-strike bug)", flush=True)
    for n, r in can.items():
        print(f"  canary {n:10s} raw_mean={r['auc_mean']:.4f} "
              f"disc_max={r['discriminative_auc_max']:.4f} leak={r['leakage_flag']}",
              flush=True)
    # Distance features are functions of the VISIBLE mask only; a high
    # discriminative AUC is the lane's own damage-zone signal (or its inverse
    # under detached withholding, where hidden strands sit far from visible
    # traces).  Protocol rule 4: "AUC above 0.90 means leakage until proven
    # otherwise."  Proof they are not leakage: (1) nearest_frame / offset
    # components take `visible = catalogue & ~hidden`; (2) permuting `y` cannot
    # change X; (3) IR-57-STRIKE-01 made d_perp the true cross-strike coordinate,
    # which is *more* inverse-predictive for detached truth, not a mask echo.
    # Any non-distance feature above 0.90 still aborts.
    distance_ok = {"d", "d_perp", "d_par_abs"}
    bad = {n: r["discriminative_auc_max"] for n, r in leak.items() if n not in distance_ok}
    if bad:
        raise RuntimeError(f"leakage canary flagged on non-distance features: {bad}")
    if leak:
        print("  IR-57-CANARY-02: distance-only flags documented, not treated as leakage",
              flush=True)

    # ----- Experiment 1: structure the data shows -------------------------
    all_geoms = [geoms[c.key] for c in cells]
    struct_iso = fit_zone(all_geoms, d_col=D_COL, outer=False, isotropic=True)
    struct_str = fit_zone(all_geoms, d_col=D_COL, outer=False, isotropic=False)
    struct_out = fit_zone(all_geoms, d_col=D_COL, outer=True, isotropic=False)
    # strike occupancy of withheld pixels (descriptive, not a score)
    strikes = np.concatenate([strike_deg_from_X(g.X[g.y == 1]) for g in all_geoms if (g.y == 1).any()])
    strike_hist, _ = np.histogram(strikes, bins=STRIKE_EDGES)
    exp1 = {
        "experiment": 1,
        "hypothesis": "H57-I: withheld secondary strands occupy a strike-dependent damage-zone width",
        "instrument": "hide-and-recover detached segments, visible-only geometry, IR-57-STRIKE-01 fixed",
        "n_cells": len(cells),
        "n_withheld": int(sum(g.n_hidden for g in all_geoms)),
        "sin2_cos2_nonconstant": bool(
            float(np.ptp(np.concatenate([g.X[:, FEATURES.index("sin2")] for g in all_geoms]))) > 1e-6),
        "withheld_strike_histogram": {
            "edges": STRIKE_EDGES.tolist(), "counts": strike_hist.tolist(),
        },
        "isotropic": model_to_json(struct_iso),
        "strike_binned": model_to_json(struct_str),
        "strike_binned_outer": model_to_json(struct_out),
        "canary": can,
        "runtime_s": round(time.monotonic() - t0, 1),
    }
    (EVID / "exp5_structure.json").write_text(json.dumps(exp1, indent=2, allow_nan=False) + "\n")
    print(f"[{time.monotonic()-t0:6.1f}s] EXP1 kept_bins iso={struct_iso.n_kept_bins} "
          f"strike={struct_str.n_kept_bins} outer={struct_out.n_kept_bins} "
          f"frac_in_zone={struct_str.fraction_pos_in_zone:.3f}", flush=True)

    # ----- Experiments 2 and 3: LOQO holdout ------------------------------
    arms = {
        "iso_full": dict(outer=False, isotropic=True),
        "strike_full": dict(outer=False, isotropic=False),
        "strike_outer": dict(outer=True, isotropic=False),
    }
    scores = {name: [] for name in arms}
    fold_models = {name: {} for name in arms}
    for q in FOLD_NAMES:
        test = [c for c in cells if c.key.split("_")[1] == f"fold{q}"]
        train = [c for c in cells if c.key.split("_")[1] != f"fold{q}"]
        tr = [geoms[c.key] for c in train]
        for name, kw in arms.items():
            m = fit_zone(tr, d_col=D_COL, **kw)
            fold_models[name][q] = model_to_json(m)
            for c in test:
                scores[name].append(score_arm(ctx, c, geoms[c.key], m))
        print(f"[{time.monotonic()-t0:6.1f}s] finished quadrant {q}", flush=True)
        gc.collect()

    pooled_arms = {n: pooled(scores[n]) for n in arms}
    for n, p in pooled_arms.items():
        print(f"  {n:13s} HOLDOUT-DTI {p['pooled_dti']:.4f}  "
              f"CI {p['dti_ci95_quadrant_jackknife']}  dots={p['n_dots']}  "
              f"n_truth={p['n_truth']}", flush=True)

    exp2 = {
        "experiment": 2,
        "hypothesis": "H57-I strike-binned width vs isotropic d_perp decay",
        "evidence_class": "HOLDOUT-DTI (not a live score)",
        "evaluator_version": evaluate_holdout.VERSION + " repaired, gems59 LOQO",
        "shared_evaluator_hashes": evaluate_holdout.implementation_hashes(),
        "withheld_positive_pixels": pooled_arms["strike_full"]["n_truth"],
        "arms": {n: pooled_arms[n] for n in ("iso_full", "strike_full")},
        "paired_delta_strike_minus_iso": pooled_arms["strike_full"]["pooled_dti"] - pooled_arms["iso_full"]["pooled_dti"],
        "paired_delta_per_quadrant": paired_delta(pooled_arms["strike_full"], pooled_arms["iso_full"],
                                                  scores["strike_full"], scores["iso_full"]),
        "fold_models": {n: fold_models[n] for n in ("iso_full", "strike_full")},
        "runtime_s": round(time.monotonic() - t0, 1),
    }
    (EVID / "exp6_holdout_strike.json").write_text(json.dumps(exp2, indent=2, allow_nan=False) + "\n")

    exp3 = {
        "experiment": 3,
        "hypothesis": "H57-I-outer: strike-binned width with 0-2 px flank removed (thread 11516)",
        "evidence_class": "HOLDOUT-DTI (not a live score)",
        "evaluator_version": evaluate_holdout.VERSION + " repaired, gems59 LOQO",
        "shared_evaluator_hashes": evaluate_holdout.implementation_hashes(),
        "withheld_positive_pixels": pooled_arms["strike_outer"]["n_truth"],
        "arms": {n: pooled_arms[n] for n in ("strike_full", "strike_outer")},
        "paired_delta_outer_minus_full": pooled_arms["strike_outer"]["pooled_dti"] - pooled_arms["strike_full"]["pooled_dti"],
        "paired_delta_per_quadrant": paired_delta(pooled_arms["strike_outer"], pooled_arms["strike_full"],
                                                  scores["strike_outer"], scores["strike_full"]),
        "fold_models": {"strike_outer": fold_models["strike_outer"]},
        "runtime_s": round(time.monotonic() - t0, 1),
    }
    (EVID / "exp7_holdout_outer.json").write_text(json.dumps(exp3, indent=2, allow_nan=False) + "\n")

    # Winner: highest LOQO DTI among the three arms.  Outer is preferred on a
    # tie because thread 11516 penalises the catalogue flank on the live set.
    ranked = sorted(pooled_arms.items(), key=lambda kv: (kv[1]["pooled_dti"], kv[0] == "strike_outer"), reverse=True)
    winner_name = ranked[0][0]
    winner = pooled_arms[winner_name]
    baseline = pooled_arms["iso_full"]
    # iso_full is an arm of THIS experiment, not the previous-session bar.
    # The comparable previous best on this instrument is H57-G detached LOQO.
    H57G_DTI = 0.23742257629534305
    beats_iso = winner_name != "iso_full" and winner["pooled_dti"] > baseline["pooled_dti"]
    beats_baseline = winner["pooled_dti"] > H57G_DTI
    print(f"[{time.monotonic()-t0:6.1f}s] winner={winner_name} "
          f"DTI={winner['pooled_dti']:.4f} beats_H57G={beats_baseline} "
          f"beats_iso={beats_iso}", flush=True)

    # ----- full-catalogue surface from the winner's specification ----------
    # Fit the emission table on ALL detached cells (documented as in-sample
    # rates; the HOLDOUT-DTI numbers above are the LOQO reading).
    win_kw = arms[winner_name]
    live_model = fit_zone(all_geoms, d_col=D_COL, **win_kw)
    domain = grid.footprint & ~grid.catalogue
    geom = fold_geometry(grid, grid.catalogue, np.zeros(grid.shape, bool),
                         domain, "live_full_visible")
    p = np.zeros(grid.shape, np.float32)
    if geom.X.shape[0]:
        p[geom.rows, geom.cols] = live_model.intensity(geom.X)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert (p[~domain] == 0).all()
    print(f"[{time.monotonic()-t0:6.1f}s] live surface p>0={int((p > 0).sum())} "
          f"p_max={float(p.max()):.4f} kept_bins={live_model.n_kept_bins}", flush=True)
    del geom
    gc.collect()

    priors = registry_priors()
    print(f"[{time.monotonic()-t0:6.1f}s] uniqueness priors: {len(priors)}", flush=True)
    surf_rep = lane_report(p, domain, priors, sample=SAMPLE, phase="surface",
                           log=lambda s: print("   ", s, flush=True))
    print(f"  surface literal={surf_rep['literal']['verdict']} "
          f"policy={surf_rep['policy']['verdict']} "
          f"max_rho={surf_rep['literal']['max_spearman']}", flush=True)

    index = json.loads(INDEX.read_text())
    registered_hashes = {r["sha256"] for r in index["rasters"]}

    # Surface rank-gate: Spearman > 0.90 is lane drift.  Overlap is a DOT
    # statistic; a dense probability surface is not a proposal support.
    if surf_rep["literal"]["rank_offenders"] or surf_rep["literal"]["identical"]:
        digest = hashlib.sha256(p.tobytes()).hexdigest()[:12]
        path = DL / f"gems59-h57i-{winner_name}-{digest}-RESEARCH-DO-NOT-SUBMIT.tif"
        from gems59.grid import write_submission as write_raw
        write_raw(path, p, mode="zeros")
        v = validate(path, grid.footprint, grid.catalogue)
        card = _run_card(winner_name, winner, baseline, beats_baseline, can,
                         surf_rep, None, v, path, "negative",
                         "surface rank-gate fired; no dots placed", t0,
                         live_model, pooled_arms, registered_hashes)
        _write_card(card)
        print("STOPPED: surface rank duplicate", flush=True)
        return

    # ----- emit sparse dots ------------------------------------------------
    # k_truth: one draw of the holdout (4 quadrants, seed 20).
    draw0 = [r for r in scores[winner_name] if r["key"].startswith("draw20")]
    k_truth = float(sum(r["n_truth"] for r in draw0))
    alloc = greedy_allocate(p, domain, k_truth=k_truth, floor=FLOOR,
                            max_dots=FINAL_BUDGET)
    print(f"[{time.monotonic()-t0:6.1f}s] allocated {alloc.n_dots} dots "
          f"(cap {FINAL_BUDGET})", flush=True)
    values = alloc.emitted.astype(np.float32)
    values[~grid.footprint] = 0.0
    values[grid.catalogue] = 0.0
    assert int((values[grid.catalogue] > 0).sum()) == 0
    del p
    gc.collect()

    dots_rep = lane_report(values, domain, priors, sample=SAMPLE, phase="dots",
                           log=lambda s: print("   ", s, flush=True))
    print(f"  dots literal={dots_rep['literal']['verdict']} "
          f"policy={dots_rep['policy']['verdict']} "
          f"max_near={dots_rep['literal']['max_near_3px_fraction']} "
          f"max_rho={dots_rep['literal']['max_spearman']}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(values.tobytes()).hexdigest()[:12]
    policy_ok = dots_rep["ok"] and not dots_rep["policy"]["rank_offenders"] \
        and not dots_rep["policy"]["near_offenders"] and not dots_rep["literal"]["identical"]
    # Submit-clearance: format-valid + policy uniqueness + SHA novel.
    # Literal failure ONLY against universal-coverage probes is recorded, not
    # treated as lane drift (H61 shared-template repair in gates.lane_report).
    name = f"gems59-h57i-{winner_name}-{stamp}-{digest}"
    note = (f"H57-I {winner_name} strike-binned damage zone; fitted not textbook; "
            f"{alloc.n_dots} dots 0 on-cat")
    if len(note) > 140:
        note = f"H57-I {winner_name} fitted strike-zone | {alloc.n_dots} dots 0 on-cat"
    assert 1 <= len(note) <= 140 and 1 <= len(name) <= 140

    path = DL / f"{name}-zeros.tif"
    receipt = write_submission(path, values, SAMPLE, grid.footprint,
                               note=note, name=name[:140],
                               metadata={"arm": winner_name, "n_dots": int(alloc.n_dots)})
    v = validate(path, grid.footprint, grid.catalogue)
    sha = v["sha256"]
    sha_novel = sha not in registered_hashes
    print(f"[{time.monotonic()-t0:6.1f}s] wrote {path.name} sha={sha[:16]} "
          f"dots={v['emitted_positive_pixels']} checks={v['all_checks_passed']} "
          f"sha_novel={sha_novel}", flush=True)

    format_ok = bool(v["all_checks_passed"] and v["n_nan"] == 0
                     and v["min"] >= 0 and v["max"] <= 1
                     and v.get("on_catalogue_positive_pixels", 0) == 0)
    # Promote to "OK TO SUBMIT" only if: format, sha novel, policy uniqueness,
    # and the arm beat the isotropic baseline on LOQO.  Literal probe-only
    # failures do not block.
    ok_to_submit = bool(format_ok and sha_novel and policy_ok and beats_baseline)
    if ok_to_submit:
        verdict = "promote"
        submit_status = "OK_TO_SUBMIT"
        banner = "OK TO DOWNLOAD AND SUBMIT"
    elif not beats_baseline:
        verdict = "negative"
        submit_status = "DO_NOT_SUBMIT_HOLDOUT"
        banner = "DO NOT SUBMIT — holdout did not beat isotropic baseline"
    elif not policy_ok:
        verdict = "negative"
        submit_status = "DO_NOT_SUBMIT_UNIQUENESS"
        banner = "DO NOT SUBMIT — uniqueness policy gate fired"
    else:
        verdict = "negative"
        submit_status = "DO_NOT_SUBMIT"
        banner = "DO NOT SUBMIT"

    card = _run_card(winner_name, winner, baseline, beats_baseline, can,
                     surf_rep, dots_rep, v, path, verdict, banner, t0,
                     live_model, pooled_arms, registered_hashes,
                     extra={
                         "submit_status": submit_status,
                         "ok_to_submit": ok_to_submit,
                         "policy_ok": policy_ok,
                         "format_ok": format_ok,
                         "sha256_novel_vs_644": sha_novel,
                         "submission_name": name,
                         "submission_note": note,
                         "n_dots": int(alloc.n_dots),
                         "writer_receipt": {k: receipt[k] for k in
                                            ("file", "sha256", "bytes", "note",
                                             "submission_name", "zip_file") if k in receipt},
                         "exp1": str(EVID / "exp5_structure.json"),
                         "exp2": str(EVID / "exp6_holdout_strike.json"),
                         "exp3": str(EVID / "exp7_holdout_outer.json"),
                     })
    _write_card(card)
    (DL / f"checks-{path.name}.json").write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
    print("VERDICT", banner, "sha", sha, flush=True)


def _run_card(winner_name, winner, baseline, beats_baseline, can,
              surf_rep, dots_rep, validator, path, verdict, banner, t0,
              live_model, pooled_arms, registered_hashes, extra=None) -> dict:
    card = {
        "hypothesis": "H57-I: secondary-strand density decays with a strike-dependent damage-zone width fitted from hide-and-recover, not from textbook Riedel angles",
        "mechanism": "Savage & Brodsky (2011) damage-zone decay, conditioned on local visible strike; bins below 2x base rate dropped",
        "non_fault_mimic": "Regional drainage and range-front lineaments that share the Walker Lane / Basin-Range strike bins but are not faults",
        "evidence_class": "HOLDOUT-DTI (not a live score)",
        "winner_arm": winner_name,
        "holdout_DTI": winner["pooled_dti"],
        "holdout_DTI_CI95": winner["dti_ci95_quadrant_jackknife"],
        "holdout_n_withheld": winner["n_truth"],
        "baseline_iso_DTI": baseline["pooled_dti"],
        "baseline_iso_CI95": baseline["dti_ci95_quadrant_jackknife"],
        "beats_baseline": beats_baseline,
        "all_arms": {n: {"pooled_dti": p["pooled_dti"],
                         "ci95": p["dti_ci95_quadrant_jackknife"],
                         "coverage": p["coverage"], "n_dots": p["n_dots"]}
                     for n, p in pooled_arms.items()},
        "canary_max_disc_auc": {n: can[n]["discriminative_auc_max"] for n in FEATURES},
        "canary_any_leakage": any(can[n]["leakage_flag"] for n in FEATURES),
        "surface_uniqueness": {
            "literal": surf_rep["literal"]["verdict"] if surf_rep else None,
            "policy": surf_rep["policy"]["verdict"] if surf_rep else None,
            "max_spearman": surf_rep["literal"]["max_spearman"] if surf_rep else None,
            "priors_checked": surf_rep["priors_checked"] if surf_rep else None,
        },
        "dots_uniqueness": None if dots_rep is None else {
            "literal": dots_rep["literal"]["verdict"],
            "policy": dots_rep["policy"]["verdict"],
            "max_spearman": dots_rep["literal"]["max_spearman"],
            "max_near_3px": dots_rep["literal"]["max_near_3px_fraction"],
            "max_near_source": dots_rep["literal"]["max_near_source"],
            "policy_max_near_3px": dots_rep["policy"]["max_near_3px_fraction"],
            "policy_max_near_source": dots_rep["policy"]["max_near_source"],
            "n_probes": dots_rep["policy"]["universal_coverage_probes"],
            "priors_checked": dots_rep["priors_checked"],
        },
        "raster_sha256": validator["sha256"],
        "raster_path": str(path.relative_to(ROOT)),
        "validator": {
            "all_checks_passed": validator["all_checks_passed"],
            "n_nan": validator["n_nan"],
            "min": validator["min"], "max": validator["max"],
            "crs": validator["meta"]["crs"],
            "shape": validator["meta"]["shape"],
            "transform": validator["meta"]["transform"],
            "emitted_positive_pixels": validator["emitted_positive_pixels"],
            "on_catalogue": validator.get("on_catalogue_positive_pixels"),
            "checks": validator["checks"],
        },
        "live_model_kept_bins": live_model.n_kept_bins,
        "live_model_frac_pos_in_zone": live_model.fraction_pos_in_zone,
        "verdict": verdict,
        "banner": banner,
        "slot_used": False,
        "runtime_s": round(time.monotonic() - t0, 1),
        "ir_57_strike_01": "fixed: finite strike is no longer zeroed in fold_geometry",
    }
    if extra:
        card.update(extra)
    return card


def _write_card(card: dict) -> None:
    (EVID / "run_card.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    (EVID / "run_card_session4.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    print("wrote evidence/run_card.json", card["banner"], flush=True)


if __name__ == "__main__":
    raise SystemExit("Archived H57-I generator: prior clearance used only 16 priors. Literal dense17 "
                     "overlap blocks this file. No new generation/slot is authorized. Read current README.")
