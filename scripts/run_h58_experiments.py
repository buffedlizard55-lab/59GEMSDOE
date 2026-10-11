#!/usr/bin/env python3
"""H58 fault-zone-anatomy session: three predeclared experiments, then a build.

Predeclared in ``evidence/h58_experiment_plan.json`` before any number was seen
(rule 7: three experiments, no weekly slot spent):

``E1 structure``
    Measure what the lane asks for -- the distance and signed relative-strike
    distributions of withheld segments against their nearest visible fault --
    and fit the displacement-scaling law ``W(L) = w0 (L/L0)**gamma`` for the
    damage-zone half-width. Only structure the data shows is kept.
``E2 holdout``
    Leave-one-quadrant-out comparison of nested feature arms with the shared
    evaluator (``gems59-pooled-hide-v2``) at a budget set by the fitted
    inside-zone fraction, under two catalogue-adjacency policies: ``flank0``
    keeps every off-trace pixel, ``flank2`` additionally drops pixels within
    2 px of a visible trace (the geometry organiser thread 11516 penalises).
``E3 build``
    Fit on the whole catalogue, predict the intensity, allocate dots, validate
    and write the GeoTIFF/ZIP/receipt. The literal parallel-run gates run in
    ``scripts/h58_gates.py`` on the surface before placement and on the final
    dots, because they need the full restored registry cache.

Every score is labelled HOLDOUT-DTI (evaluator version, withheld positives, 95%
CI). Nothing here projects a live score. Promotion to a real slot is a separate
selector step and is not performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                    # noqa: E402

from gems59 import evaluate_holdout as EH                             # noqa: E402
from gems59 import load_grid                                          # noqa: E402
from gems59.anatomy import FEATURES, SENSE_FEATURES, fold_geometry     # noqa: E402
from gems59.anatomy import (measure_withheld, package_holdout_structure,    # noqa: E402
                            relative_strike_distribution)
from gems59.damagezone_envelope import (FLANK_PX_DEFAULT, OBLIQUITY_EDGES,  # noqa: E402
                                        fit_width_law, signed_obliquity)
from gems59.emit import greedy_allocate                               # noqa: E402
from gems59.faultzone import trace_sense_raster                       # noqa: E402
from gems59.fitting import canary, fit_model                           # noqa: E402
from gems59.holdout import FOLD_NAMES, buffered_component_draw         # noqa: E402
from gems59.strike_zone import fit_zone, model_to_json                 # noqa: E402
from gems59.submission_writer import write_submission                   # noqa: E402
from gems59.validate import validate                                   # noqa: E402

EVID = ROOT / "evidence"
CACHE = ROOT / ".cache"
SEED = 2026                 # a fresh whole-component draw, not a draw earlier sessions selected on
PER_QUADRANT_CAP = 10_000   # 40,000 total ceiling before the fitted shrink
FLANK_PX = FLANK_PX_DEFAULT
ZONE_Q = 0.90
D_EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 33, 41, 51, 1e9])
BASE = [n for n in FEATURES if n != "side"]
CANDIDATE, REFERENCE = "anatomy10_sense_obliq", "anatomy10"
CHUNK = 150_000


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def view(X, y, names):
    """Minimal object the shared fitting/canary/zone helpers accept."""
    return type("View", (), dict(X=X, y=y, feature_names=tuple(names)))()


class Fold:
    """One quadrant of the shared draw: visible-only features, labels and pixels."""

    def __init__(self, name, X, y, rows, cols, names, length):
        self.name, self.X, self.y = name, X, y
        self.rows, self.cols, self.names = rows, cols, list(names)
        self.d = X[:, self.names.index("d")].astype(np.float64)
        self.length = length                                # mapped-component length, px
        self.pos = y == 1

    def __len__(self):
        return len(self.y)

    def rel(self):
        return self.X[:, self.names.index("obliq_signed_base")].astype(np.float64)

    def sense_sgn(self):
        return self.X[:, self.names.index("sense_sgn")].astype(np.float64)

    def design(self, wanted, law):
        """Arm design matrix, resolving the three virtual columns."""
        idx = {n: i for i, n in enumerate(self.names)}
        cols = []
        for n in wanted:
            if n == "obliq_signed":
                cols.append(self.rel())
            elif n == "obliq_signed_sense":
                cols.append(self.rel() * self.sense_sgn())
            elif n == "d_over_w":
                cols.append(self.d / law.half_width(self.length))
            elif n in idx:
                cols.append(self.X[:, idx[n]].astype(np.float64))
            else:
                raise KeyError(f"unknown feature {n}")
        return np.stack(cols, axis=1).astype(np.float32)


def predict_in_chunks(clf, scale, D, rows, cols, shape):
    p = np.zeros(shape, np.float32)
    for start in range(0, len(D), CHUNK):
        sl = slice(start, min(start + CHUNK, len(D)))
        v = clf.predict_proba(D[sl])[:, 1] * scale
        p[rows[sl], cols[sl]] = np.clip(v, 0, 1).astype(np.float32)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["all", "structure", "holdout", "build"])
    ap.add_argument("--out-tif-dir", default=str(ROOT / "docs/downloads"))
    a = ap.parse_args()
    t0 = os.times()[4]
    CACHE.mkdir(parents=True, exist_ok=True)
    plan = dict(
        generated_utc=now(), evidence_class="EXPERIMENT PLAN (predeclared, no numbers yet)",
        lane="fault-zone anatomy (secondary strands around mapped faults)",
        split="buffered-whole-components-loqo-v2", draw_seed=SEED, evaluator=EH.VERSION,
        alpha=0.2, beta=0.8, triangular_radius_m=300.0,
        per_quadrant_cap=PER_QUADRANT_CAP, flank_px=FLANK_PX, zone_quantile=ZONE_Q,
        arms=None, candidate=CANDIDATE, reference=REFERENCE,
        retention_rule=("candidate retained only if its paired 95% CI lower bound against the "
                        "reference arm under the same flank policy is strictly positive; "
                        "otherwise the simpler arm ships"),
        budget_rule=("per-quadrant cap times the TRAINING fraction of withheld positives inside "
                     "the fitted zone, clipped to [0.25, 1.0]; k_truth comes from the TRAINING "
                     "base rate times allowed cells, never from test truth"),
        width_law=("W(L)=w0*(L/L0)**gamma fitted on training folds only by bin-wise quantile "
                   "regression of withheld distance on mapped-component length, bin bootstrap CI"),
        no_textbook_angle=True, submission_slots_used=0, maximum_experiments=3,
    )

    grid = load_grid()
    folds, quad = buffered_component_draw(grid, seed=SEED)
    visible, hidden = folds[0]["visible"], folds[0]["hidden_all"]
    domain = np.logical_or.reduce([f["region"] for f in folds]) & ~folds[0]["masked_known"]
    sense = trace_sense_raster(ROOT / "data/external/trace_segments_utm11.csv",
                              grid.shape, grid.transform).astype(np.int8)
    sense_vis = np.where(visible, sense, 0).astype(np.int8)

    g = fold_geometry(grid, visible, hidden, domain, "draw-buffered", sense_src=sense_vis)
    ys, xs = np.nonzero(np.asarray(domain, bool) & ~np.asarray(visible, bool))
    assert np.array_equal(ys, g.rows) and np.array_equal(xs, g.cols), \
        "signed-obliquity pixel order must match fold_geometry"
    rel, n_censored = signed_obliquity(visible, ys, xs)
    names = list(g.feature_names) + ["obliq_signed_base"]
    X = np.concatenate([g.X, rel[:, None]], axis=1)
    y = g.y
    comp_len = np.expm1(X[:, names.index("log_len")].astype(np.float64))
    qrow = quad[ys, xs]
    FOLDS = [Fold(FOLD_NAMES[q], X[qrow == q], y[qrow == q], ys[qrow == q], xs[qrow == q],
                  names, comp_len[qrow == q]) for q in range(4)]
    del g, rel, domain, qrow
    gc.collect()
    print(f"[h58] rows={len(y)} positives={int(y.sum())} cols={X.shape[1]} "
          f"censored_strikes={n_censored} in {os.times()[4]-t0:.0f}s", flush=True)

    ARMS = {
        "distance_only": ["d"],
        "anatomy10": BASE,
        "anatomy10_sense": BASE + list(SENSE_FEATURES),
        "anatomy10_sense_obliq": BASE + list(SENSE_FEATURES)
            + ["obliq_signed", "obliq_signed_sense", "d_over_w"],
    }
    plan["arms"] = {k: list(v) for k, v in ARMS.items()}
    save(EVID / "h58_experiment_plan.json", plan)
    print("[h58] plan saved: three declared experiments, no weekly slot will be used", flush=True)
    pooled = view(X, y, names)

    # ------------------------------------------------------------------ E1
    pos = y == 1
    law_all = fit_width_law(comp_len[pos], X[:, names.index("d")][pos],
                            np.ones(int(pos.sum()), bool), quantile=ZONE_Q)
    meas = measure_withheld(pooled, d_edges=D_EDGES)
    rel_hist = relative_strike_distribution(grid, visible, hidden)
    structure = package_holdout_structure(
        rel_hist, withheld_positive_pixels=int(pos.sum()),
        distance_positive_quantiles_px=np.quantile(X[:, names.index("d")][pos], [.1, .5, .9, .95, .99]),
        distance_domain_quantiles_px=np.quantile(X[:, names.index("d")], [.1, .5, .9, .95, .99]))
    ob = pooled.X[:, names.index("obliq_signed_base")].astype(np.float64)
    ho = np.histogram(ob[pos], bins=OBLIQUITY_EDGES)[0].astype(float)
    ha = np.histogram(ob, bins=OBLIQUITY_EDGES)[0].astype(float)
    sgn = pooled.X[:, names.index("sense_sgn")].astype(np.float64)   # all domain rows
    ob_pos, sgn_pos = ob[pos], sgn[pos]                    # withheld positives only
    asym = {}
    for label, sel_rows, sel_pos in (
            ("recorded_strike_slip_host", sgn != 0, sgn_pos != 0),
            ("no_recorded_sense", sgn == 0, sgn_pos == 0)):
        left = int(((ob_pos < -12) & sel_pos).sum()); right = int(((ob_pos > 12) & sel_pos).sum())
        nl = int(((ob < -12) & sel_rows).sum()); nr = int(((ob > 12) & sel_rows).sum())
        rl, rr = left / max(nl, 1), right / max(nr, 1)
        asym[label] = dict(positives_left=left, positives_right=right, domain_left=nl,
                          domain_right=nr, rate_left=rl, rate_right=rr,
                          log_ratio_R_over_L=float(np.log(max(rr, 1e-12) / max(rl, 1e-12))))
    zone_models = {}
    for iso in (True, False):
        z = fit_zone([pooled], d_col=names.index("d"), outer=True, isotropic=iso, min_ratio=2.0)
        zone_models[z.name] = model_to_json(z)
    frac_in_zone = float(np.mean([z["fraction_pos_in_zone"] for z in zone_models.values()]))
    can = canary([pooled], feature_names=tuple(names))
    flagged = sorted(n for n, r in can.items() if r["leakage_flag"])
    save(EVID / "h58_canary.json", dict(
        evidence_class="LEAKAGE-CANARY (AUC, not DTI)", generated_utc=now(),
        split_version=plan["split"], draw_seed=SEED, features=can, flagged=flagged,
        withheld_positive_pixels=int(pos.sum()),
        note=("Every column is a deterministic function of the visible mask (plus the "
              "visible-restricted INGENIOUS sense raster), so no channel runs from the withheld "
              "set into a feature. Near-field distance AUC is the lane's signal, not a leak.")))
    save(EVID / "h58_structure.json", dict(
        evidence_class="HOLDOUT-STRUCTURE (descriptive, not a score)", generated_utc=now(),
        split_version=plan["split"], draw_seed=SEED,
        withheld_positive_pixels=int(pos.sum()), domain_pixels=int(len(y)),
        base_rate=float(pos.mean()), relative_strike=rel_hist, structure=structure,
        withheld_distance_enrichment=meas["distance"],
        withheld_stepover_enrichment=meas["stepover"],
        withheld_along_strike_enrichment=meas["along_strike"],
        withheld_joint_stepover_alongstrike=meas["joint_stepover_alongstrike"],
        segment_length_enrichment=meas["segment_length"], unsigned_side=meas["side"],
        censored_anchor_strikes=int(n_censored),
        obliquity_histogram=dict(edges=OBLIQUITY_EDGES.tolist(), n_withheld=ho.tolist(),
                                 n_domain=ha.tolist(), enrichment=(ho / np.maximum(ha, 1)).tolist()),
        side_asymmetry_by_sense_record=asym, width_law_all_folds=law_all.to_json(),
        strike_zone_fit=zone_models, fraction_pos_in_kept_zone=frac_in_zone,
        canary_flags=flagged,
        note=("Withheld positives are mapped catalogue pixels, so this instrument measures "
              "recovery of catalogue geometry, not discovery of unpublished expert faults. "
              "Obliquity is pixel-weighted against the nearest visible trace; no significance "
              "test is implied.")))
    print("[h58] E1 width law " + json.dumps({k: law_all.to_json()[k] for k in
          ("gamma", "gamma_ci95", "w0_px", "sublinear", "n_bins", "n_positives")}), flush=True)
    print(f"[h58] E1 obliquity enrichment={(ho / np.maximum(ha, 1)).round(3).tolist()} "
          f"asym={ {k: round(v['log_ratio_R_over_L'], 3) for k, v in asym.items()} } "
          f"frac_in_kept_zone={frac_in_zone:.3f} canary_flags={flagged}", flush=True)
    if flagged:
        save(EVID / "h58_holdout.json", dict(evidence_class="HOLDOUT-DTI",
             status="negative: unresolved leakage canary", flagged=flagged, holdout_dti=None))
        print("[h58] STOP: leakage canary fired", flush=True)
        return 3
    if a.stage == "structure":
        return 0

    # ------------------------------------------------------------------ E2
    terms, surf_terms, details = {}, {}, []
    for q, fold in enumerate(FOLDS):
        tr = [FOLDS[i] for i in range(4) if i != q]
        law = fit_width_law(np.concatenate([f.length[f.pos] for f in tr]),
                            np.concatenate([f.d[f.pos] for f in tr]),
                            np.ones(sum(int(f.pos.sum()) for f in tr), bool), quantile=ZONE_Q)
        tr_pos = sum(int(f.pos.sum()) for f in tr)
        base_rate = tr_pos / sum(len(f) for f in tr)
        ofold = folds[q]
        allowed = ofold["region"] & ~ofold["masked_known"]
        k_est = base_rate * float(allowed.sum())
        inside = np.concatenate([f.d[f.pos] for f in tr]) <= law.half_width(
            np.concatenate([f.length[f.pos] for f in tr]))
        frac_in = float(inside.mean())
        cap = max(1, int(PER_QUADRANT_CAP * min(1.0, max(frac_in, 0.25))))
        flank_ok = fold.d > FLANK_PX
        print(f"[h58] E2 {FOLD_NAMES[q]:3s} gamma={law.gamma:+.3f} w0={np.exp(law.log_w0):.2f} "
              f"frac_in_zone={frac_in:.3f} cap={cap} k_est={k_est:.0f}", flush=True)
        for arm, wanted in ARMS.items():
            Dtr = [f.design(wanted, law) for f in tr]
            Dte = fold.design(wanted, law)
            clf, scale, _ = fit_model([view(D, f.y, wanted) for D, f in zip(Dtr, tr)], seed=q)
            p = predict_in_chunks(clf, scale, Dte, fold.rows, fold.cols, grid.shape)
            for flank, keep in (("flank0", np.ones(len(fold), bool)), ("flank2", flank_ok)):
                al = np.zeros(grid.shape, bool)
                al[fold.rows[keep], fold.cols[keep]] = True
                al &= allowed
                dots = greedy_allocate(p, al, k_truth=k_est, floor=0.015, max_dots=cap)
                scored, t = EH.evaluate(dots.emitted.astype(np.float32), ofold, grid.footprint)
                raw, st = EH.evaluate(p, ofold, grid.footprint)
                key = f"{arm}__{flank}"
                terms[key] = t if key not in terms else terms[key] + t
                surf_terms[key] = st if key not in surf_terms else surf_terms[key] + st
                details.append(dict(arm=arm, flank=flank, fold=FOLD_NAMES[q],
                                    evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
                                    withheld_positive_pixels=scored["n_truth"], dti=scored["dti"],
                                    coverage=scored["coverage"], soft_surface_dti=raw["dti"],
                                    dots=int(dots.n_dots), dot_cap=cap,
                                    training_fraction_inside_zone=frac_in,
                                    training_gamma=law.gamma, training_gamma_ci=law.gamma_ci95,
                                    k_estimate_from_training=k_est,
                                    test_truth_used_for_placement=False))
                print(f"[h58] E2 {FOLD_NAMES[q]:3s} {key:36s} HOLDOUT-DTI {scored['dti']:.5f} "
                      f"soft {raw['dti']:.5f} dots {dots.n_dots}", flush=True)
                del al, dots, scored, raw
            del p, clf, Dtr, Dte
            gc.collect()
    cand, ref = f"{CANDIDATE}__flank2", f"{REFERENCE}__flank2"
    pooled_scores = EH.pooled_summary(terms, draws=1000, seed=20261010, candidate=cand)
    surfaces = EH.pooled_summary(surf_terms, draws=1000, seed=20261010, candidate=cand)
    flank_cost = {}
    for arm in ARMS:
        a2 = float(EH.from_terms(terms[f"{arm}__flank2"].sum(axis=0)))
        a0 = float(EH.from_terms(terms[f"{arm}__flank0"].sum(axis=0)))
        flank_cost[arm] = dict(evidence_class="HOLDOUT-DTI", evaluator_version=EH.VERSION,
                              dti_flank2=a2, dti_flank0=a0, delta=a2 - a0,
                              interpretation=("cost/benefit of dropping pixels within "
                                              f"{FLANK_PX:g} px of a visible trace"))
    gain = pooled_scores["paired_differences"].get(ref, {})
    retained = bool(gain.get("ci95") and gain["ci95"][0] > 0)
    best_arm = cand if retained else ref
    save(EVID / "h58_holdout.json", dict(
        **pooled_scores, split_version=plan["split"], draw_seed=SEED, per_fold=details,
        soft_surface_holdout=surfaces, candidate=cand, reference=ref,
        candidate_gain_over_reference=gain, candidate_retained=retained,
        flank_policy_effect=flank_cost, production_arm=best_arm,
        experiments_used=2, submission_slots_used=0,
        runtime_seconds=round(os.times()[4] - t0, 1),
        note=("HOLDOUT-DTI only. The hide-and-recover target is mapped catalogue geometry, so a "
              "cost for the catalogue-adjacency suppression is expected here and says nothing "
              "about live new-fault scoring, where thread 11516 makes the same pixels pure false "
              "positives. No live score is projected or implied.")))
    print("[h58] E2 pooled " + json.dumps({k: round(v["dti"], 5)
          for k, v in pooled_scores["scores"].items()}), flush=True)
    print(f"[h58] E2 gain vs reference delta={gain.get('delta')} CI={gain.get('ci95')} "
          f"retained={retained} -> production arm {best_arm}", flush=True)
    if a.stage == "holdout":
        return 0

    # ------------------------------------------------------------------ E3
    arm_name, flank_policy = best_arm.split("__")
    wanted = ARMS[arm_name]
    all_views = [view(f.design(wanted, law_all), f.y, wanted) for f in FOLDS]
    clf, scale, base = fit_model(all_views, seed=0)
    live_domain = grid.footprint & ~grid.catalogue
    sense_full = np.where(grid.catalogue, sense, 0).astype(np.int8)
    del all_views, X, y, comp_len
    gc.collect()
    live = fold_geometry(grid, grid.catalogue, np.zeros(grid.shape, bool), live_domain, "live",
                         sense_src=sense_full)
    lrows, lcols = live.rows, live.cols
    lrel, lcen = signed_obliquity(grid.catalogue, lrows, lcols)
    lnames = list(live.feature_names) + ["obliq_signed_base"]
    LX = np.concatenate([live.X, lrel[:, None]], axis=1)
    live_fold = Fold("live", LX, live.y, lrows, lcols, lnames,
                     np.expm1(LX[:, lnames.index("log_len")].astype(np.float64)))
    Dlive = live_fold.design(wanted, law_all)
    prob = predict_in_chunks(clf, scale, Dlive, lrows, lcols, grid.shape)
    in_zone = live_fold.d <= law_all.half_width(live_fold.length)
    flank = live_fold.d > (FLANK_PX if flank_policy == "flank2" else 0.0)
    allowed_prod = np.zeros(grid.shape, bool)
    allowed_prod[lrows[flank & in_zone], lcols[flank & in_zone]] = True
    cap = max(1, int(4 * PER_QUADRANT_CAP * min(1.0, max(frac_in_zone, 0.25))))
    dots = greedy_allocate(prob, allowed_prod, k_truth=base * float(allowed_prod.sum()),
                           floor=0.015, max_dots=cap)
    pred = dots.emitted.astype(np.float32)
    print(f"[h58] E3 arm={best_arm} cap={cap} dots={int(pred.sum())} "
          f"frac_in_zone={frac_in_zone:.3f} live_rows={len(lrows)} at {os.times()[4]-t0:.0f}s",
          flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dec = hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()[:12]
    label = f"gems59-h58-damagezone-envelope-{int(pred.sum())}dots-{stamp}-{dec}-zeros"[:140]
    note = ("H58 damage-zone envelope: W(L)=w0*L^gamma fitted; handed obliquity tested and "
            "rejected; dots kept >2 px off the mapped catalogue.")
    if len(note) > 140:
        raise SystemExit("H58 note must fit the 140-character portal field")
    target = Path(a.out_tif_dir) / f"{label}.tif"
    receipt = write_submission(target, pred, ROOT / "data/official/sample_submission.tif",
                              grid.footprint, note=note, name=label, catalogue=grid.catalogue,
                              metadata=dict(arm=best_arm, candidate_retained=retained,
                                            width_law=law_all.to_json(), flank_px=FLANK_PX,
                                            dot_cap=cap, evaluator=EH.VERSION, draw_seed=SEED,
                                            holdout=pooled_scores["scores"].get(best_arm),
                                            censored_anchor_strikes=int(lcen),
                                            no_prior_raster_used_to_build_predictions=True))
    vres = validate(target, grid.footprint, grid.catalogue)
    np.save(CACHE / "h58_dots.npy", dots.emitted)
    np.save(CACHE / "h58_prob.npy", prob)
    save(EVID / "h58_build.json", dict(
        evidence_class="BUILD-MEASUREMENT", generated_utc=now(),
        file=str(target.relative_to(ROOT)), raster_sha256=receipt["sha256"],
        zip_sha256=receipt["zip_sha256"], receipt_file=str(target.with_suffix('.json').relative_to(ROOT)),
        validator=vres, arm=best_arm, retained=retained, dots=int(pred.sum()), dot_cap=cap,
        fraction_pos_in_kept_zone=frac_in_zone, width_law=law_all.to_json(),
        live_censored_anchor_strikes=int(lcen), submission_name=label, submission_note=note,
        submission_note_chars=len(note), holdout=pooled_scores["scores"].get(best_arm),
        paired_gain=pooled_scores["paired_differences"].get(ref),
        runtime_seconds=round(os.times()[4] - t0, 1)))
    print(f"[h58] E3 wrote {target.name} sha256={receipt['sha256'][:16]}... "
          f"validator_all_checks_passed={vres['all_checks_passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
