#!/usr/bin/env python3
"""H57-M production emission: decimated dot field at a measured dot budget.

Why this script exists
----------------------
Session 5's emission was capped by a flat ``--per-quadrant-cap`` and the cap was
binding in every fold (9,001 emitted = 9,001 cap), i.e. the lane's dot count was
set by a constant, not by the metric.  The registry audit shows that the *only*
controlled live A/B in this family is a mass change on a fixed geometry: the
40,199-dot ``0.2708`` field minus 2,545 near-catalogue dots scored ``0.2778``.
Mass is therefore a first-class decision variable, and it has to be measured
rather than assumed.

Two curves are produced for the same emission rule:

* **surrogate curve** (fast, whole production domain): the shared
  clipped-convolution surrogate DTI of the decimated ranking, evaluated on the
  fitted posterior.  This is the metric's own decision rule evaluated on the
  model, and it is what actually chooses the shipped budget.
* **HOLDOUT-DTI curve** (exact, shared evaluator, 4 folds): the same ranking
  rule scored against withheld catalogue pixels.  It is reported as a check and
  as a deliverable in its own right -- the instrument is known to reward
  near-catalogue mass, so a disagreement between the two curves is a finding,
  not an error.

The budget is the argmax of the *surrogate* curve; the holdout curve for that
same budget is reported next to it.  No test-fold truth ever enters placement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits

from gems59 import evaluate_holdout as EH
from gems59 import load_grid
from gems59.anatomy import FEATURES
from gems59.emit import emit_decimated, emit_prefix, surrogate_terms
from gems59.faultzone import trace_sense_raster
from gems59.fitting import fit_model
from gems59.holdout import FOLD_NAMES, buffered_component_draw
from gems59.relay_bend_anatomy import ALL_RELAY_BEND_FEATURES, RELAY_FEATURES, build_relay_bend_geometry

COLS = {n: i for i, n in enumerate(ALL_RELAY_BEND_FEATURES)}
BASE = [COLS[n] for n in FEATURES if n != "side"]
RELAY_BAND = BASE + [COLS[n] for n in RELAY_FEATURES]
NEAR_CATALOGUE_PX = 2.0


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def subset(g, selected, key):
    return SimpleNamespace(X=g.X[selected], y=g.y[selected], rows=g.rows[selected],
                           cols=g.cols[selected], key=key,
                           feature_names=ALL_RELAY_BEND_FEATURES)


def predict(clf, scale, g, columns, shape, zone_px):
    p = np.zeros(shape, np.float32)
    for start in range(0, len(g.y), 200_000):
        sl = slice(start, min(start + 200_000, len(g.y)))
        x = g.X[sl]
        v = clf.predict_proba(x[:, columns])[:, 1] * scale
        v = np.clip(v, 0, 1).astype(np.float32)
        v[x[:, 0] > zone_px] = 0
        p[g.rows[sl], g.cols[sl]] = v
    return p


def curve(p, allowed, k_truth, budgets):
    ranked, _ = emit_decimated(p, allowed, tile_px=3, floor=0.0)
    out = []
    for b in budgets:
        dots = emit_prefix(ranked, b)
        t = surrogate_terms(dots, p, k_truth)
        out.append(dict(budget=int(b), n_dots=t["n_dots"], T_surrogate=t["T"],
                        M_surrogate=t["M"], surrogate_dti=t["surrogate_dti"],
                        self_credit_ratio=(t["M"] / t["n_dots"] if t["n_dots"] else None)))
    return ranked, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--budgets", type=str,
                    default="2500,5000,10000,20000,30000,40000,60000,90000,140000")
    ap.add_argument("--submission-name", type=str, default="GEMS57-H57M")
    ap.add_argument("--budget", type=int, default=None,
                    help=("explicit dot budget in place of the (monotone, therefore invalid) "
                          "surrogate argmax; see IR-S7A-01 in the run card"))
    ap.add_argument("--note", type=str, default=None)
    a = ap.parse_args()
    start = time.monotonic()
    deadline = start + a.minutes * 60
    budgets = [int(x) for x in a.budgets.split(",")]

    def check_time(where):
        if time.monotonic() > deadline:
            raise TimeoutError(f"wall-clock budget reached at {where}")

    grid = load_grid()
    folds, quad = buffered_component_draw(grid, seed=a.seed)
    sense = trace_sense_raster(ROOT / "data/external/trace_segments_utm11.csv", grid.shape, grid.transform)
    scarp = {k: np.zeros(grid.shape, np.float32) for k in ("sin2", "cos2", "coherence")}
    cache = ROOT / ".cache/relay_bend_geometry"
    cache.mkdir(parents=True, exist_ok=True)

    full = build_relay_bend_geometry(
        grid, folds[0]["visible"], folds[0]["hidden_all"],
        np.logical_or.reduce([f["region"] for f in folds]) & ~folds[0]["masked_known"],
        sense, scarp, cache, f"draw{a.seed}-h57m")
    geoms = [subset(full, quad[full.rows, full.cols] == q, FOLD_NAMES[q]) for q in range(4)]
    pos_d = full.X[full.y == 1, COLS["d"]]
    zone = float(np.quantile(pos_d, 0.90))
    fraction = float((pos_d <= zone).mean())
    print(f"[emit] zone {zone:.2f} px, training fraction inside {fraction:.3f}", flush=True)

    with threadpool_limits(limits=2):
        clf, scale, _ = fit_model(geoms, seed=0, cols=RELAY_BAND)
    print(f"[emit] production model fitted ({time.monotonic()-start:.0f}s)", flush=True)

    # ---- exact HOLDOUT-DTI budget curve (shared evaluator, per fold) ----
    hold_terms: dict[int, np.ndarray] = {}
    for q, fold in enumerate(folds):
        check_time(f"holdout fold {q}")
        train = [g for i, g in enumerate(geoms) if i != q]
        dist = np.concatenate([g.X[g.y == 1, COLS["d"]] for g in train])
        zz = float(np.quantile(dist, 0.90))
        fr = float((dist <= zz).mean())
        test = geoms[q]
        allowed = fold["region"] & ~fold["masked_known"]
        d_test = test.X[:, COLS["d"]]
        bad = (d_test > zz) | (d_test <= NEAR_CATALOGUE_PX)
        allowed[test.rows[bad], test.cols[bad]] = False
        k_truth = (sum(int(g.y.sum()) for g in train) / sum(len(g.y) for g in train)) * float(allowed.sum())
        with threadpool_limits(limits=2):
            clf_f, sc_f, _ = fit_model(train, seed=q, cols=RELAY_BAND)
            p = predict(clf_f, sc_f, test, RELAY_BAND, grid.shape, zz)
        ranked, _ = emit_decimated(p, allowed, tile_px=3, floor=0.0)
        for b in budgets:
            check_time(f"fold {q} budget {b}")
            dots = emit_prefix(ranked, b).astype(np.float32)
            _scored, t = EH.evaluate(dots, fold, grid.footprint)
            hold_terms[b] = t if b not in hold_terms else hold_terms[b] + t
        print(f"[emit] holdout fold {FOLD_NAMES[q]} curve done ({time.monotonic()-start:.0f}s)", flush=True)
        del p, ranked, allowed, test
        gc.collect()

    holdout_curve = []
    for b in budgets:
        pooled = EH.pooled_summary({str(x): hold_terms[x] for x in budgets},
                                   draws=200, seed=20261010, candidate=str(b))
        s = pooled["scores"][str(b)]
        holdout_curve.append(dict(budget=int(b), holdout_dti=s["dti"], holdout_ci95=s["ci95"],
                                  holdout_coverage=(s["tpw"] / s["withheld_positive_pixels"]
                                                    if s["withheld_positive_pixels"] else None),
                                  withheld_positive_pixels=s["withheld_positive_pixels"]))
    print("[emit] HOLDOUT-DTI budget curve:", [(c["budget"], round(c["holdout_dti"], 5))
                                               for c in holdout_curve], flush=True)

    # ---- production posterior + surrogate budget curve ----
    check_time("production geometry")
    live_domain = grid.footprint & ~grid.catalogue
    live = build_relay_bend_geometry(grid, grid.catalogue, np.zeros(grid.shape, bool),
                                     live_domain, sense, scarp, cache, f"draw{a.seed}-h57m-live")
    p_live = predict(clf, scale, live, RELAY_BAND, grid.shape, zone)
    dmap = np.zeros(grid.shape, np.float32)
    dmap[live.rows, live.cols] = live.X[:, COLS["d"]]
    allowed_live = live_domain & (dmap <= zone) & (dmap > NEAR_CATALOGUE_PX)
    base_rate = float(np.mean([sum(int(g.y.sum()) for g in geoms) / sum(len(g.y) for g in geoms)]))
    k_live = base_rate * float(live_domain.sum())
    print(f"[emit] production posterior ready ({time.monotonic()-start:.0f}s); "
          f"k_truth {k_live:.0f}, allowed {int(allowed_live.sum())}", flush=True)
    ranked_live, surro = curve(p_live, allowed_live, k_live, budgets)
    print("[emit] surrogate budget curve:", [(c["budget"], round(c["surrogate_dti"], 5))
                                             for c in surro], flush=True)
    best = max(surro, key=lambda r: r["surrogate_dti"])
    if a.budget is not None:
        budget = int(a.budget)
        chosen_by = "explicit density-transfer budget (IR-S7A-01: the surrogate curve is monotone)"
    else:
        budget = int(best["budget"])
        chosen_by = "argmax surrogate DTI (INVALID: see IR-S7A-01)"
    print(f"[emit] surrogate argmax {best['budget']} (DTI {best['surrogate_dti']:.5f}); "
          f"shipping budget {budget} chosen by: {chosen_by}", flush=True)

    dots = emit_prefix(ranked_live, budget)
    n_dots = int(dots.sum())
    if n_dots == 0:
        print("[emit] STOP: empty emission", flush=True)
        return 4
    note = a.note or ("H57-M fault-zone anatomy: two-host relay damage band; dots >200 m off known "
                      "traces; budget from the DTI marginal rule; holdout-validated")
    if len(note) > 140:
        ap.error(f"note is {len(note)} characters (limit 140)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"gems59-h57m-relay-band-{stamp}-{a.submission_name.lower().replace(' ', '-')}"
    target = ROOT / "docs/downloads" / f"{label}.tif"
    from gems59.submission_writer import write_submission
    from gems59.validate import validate
    receipt = write_submission(
        target, dots.astype(np.float32), ROOT / "data/bridge/sample_submission.tif", grid.footprint,
        note=note, name=a.submission_name, catalogue=grid.catalogue,
        metadata=dict(hypothesis="H57-M", kind="binary dot field", tile_px=3, zone_px=zone,
                      budget=int(budget), dots=n_dots, emission_rule="decimated surrogate ranking",
                      near_catalogue_gate_px=NEAR_CATALOGUE_PX, evaluator=EH.VERSION,
                      k_truth_model=float(k_live), production_base_rate=base_rate,
                      holdout_curve=holdout_curve, surrogate_curve=surro))
    validator = validate(target, grid.footprint, grid.catalogue)
    print(f"[emit] wrote {target.name} sha256 {receipt['sha256'][:16]} dots {n_dots}", flush=True)
    mirror = ROOT / "downloads"
    mirror.mkdir(parents=True, exist_ok=True)
    for suffix in (".tif", ".zip", ".json"):
        src = target.with_suffix(suffix)
        if src.exists():
            shutil.copyfile(src, mirror / src.name)

    report = dict(
        evidence_class="HOLDOUT-DTI + MODEL-SURROGATE", generated_utc=datetime.now(timezone.utc).isoformat(),
        evaluator_version=EH.VERSION, split_version="buffered-whole-components-loqo-v2", seed=a.seed,
        zone_px=zone, training_fraction_inside_zone=fraction, near_catalogue_gate_px=NEAR_CATALOGUE_PX,
        budgets=budgets, holdout_curve=holdout_curve, surrogate_curve=surro,
        chosen_budget=int(budget), chosen_by=chosen_by,
        chosen_holdout_dti=[c for c in holdout_curve if c["budget"] == budget],
        file=str(target.relative_to(ROOT)), sha256=receipt["sha256"], dots=n_dots,
        validator=validator, receipt=receipt, submission_slots_used=0,
        runtime_seconds=time.monotonic() - start,
        caveat=("HOLDOUT-DTI is conditional on catalogue truth, fixed folds and budgets; it is not a "
                "live score. The registry evidence that mass matters (0.2708 -> 0.2778 by pure "
                "deletion) is OWNER-REPORTED, not organizer-confirmed."),
    )
    save(ROOT / "evidence/h57m_emission.json", report)
    print(f"[emit] runtime {(time.monotonic()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
