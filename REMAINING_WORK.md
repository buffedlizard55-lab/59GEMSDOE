# Remaining work — Session H59-J (2026-10-11)

**The H59-J file `downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.tif` is cleared for
download, passes every local format gate (15/15) and the portal preflight, and is registry-unique against all
695 pinned rasters — no duplication identified.** `scripts/check_site.py` passes end to end. It is
**slot-eligible**; promotion to a weekly slot is the owner's selector decision and **no slot has been spent**.
The expected public score, **0.24–0.29, is a PROJECTION** from the live-calibrated dot-count band, not a
measurement; the public #1 is 0.4133.

Site: https://buffedlizard55-lab.github.io/59GEMSDOE/ · Run card:
[`evidence/run_card_current.json`](evidence/run_card_current.json) (= `docs/data/run_card.json`) · Uniqueness:
[`evidence/uniqueness_h59j.json`](evidence/uniqueness_h59j.json) (clearance view:
`evidence/uniqueness_h59j_view.json`) · Holdout CI: [`evidence/h59j_candidate_ci.json`](evidence/h59j_candidate_ci.json)

## 0. What this session settled, and what it did NOT

Settled, with measurements:

1. **The shared holdout instrument is rank-inverted against the public board.** Spearman(holdout DTI, owner-recorded
   public score) = **−0.437, p = 0.033, n = 24**; on the higher-confidence name tier −0.483 (n = 16). Mechanism:
   the instrument's truth *is* the public catalogue, so designs that decorate the catalogue win locally while the
   competition's truth is the complement of that catalogue — and a 2 px catalogue prune (which the best recorded
   sibling does and the live board rewards) collapses the instrument's true-positive term to 59.7 of 11,321.
   Pre-registered here as **IR-59J-01**; the pre-registered H59-I4 off-catalogue holdout is therefore *necessary
   but not sufficient*, and a first version of it (corridor restriction ≥ 6–10 px) still ranks backwards
   (`evidence/h59j_corridor_calibration.json`: −0.427, p 0.038 at 6 px).
2. **The emitted dot count is the live-calibrated driver** in this family: ρ(log10 dots, live) = **−0.818,
   p < 1e-4** (name tier −0.859). Owner-recorded order: 264k → 0.034, 207k → 0.090, 155k → 0.083–0.119,
   121k → 0.192, 60k → 0.248, 42k → 0.265, 40k → 0.271, 37.7k → 0.271/0.278. The H59-J budget (30,815 dots) is
   set from that relation, **not** from the holdout.
3. **The SGMC off-catalogue proxy is also rank-inverted** against the live board: ρ = **−0.665, p = 0.0004,
   n = 24** (`evidence/offcat_instrument_check.json`). Covering the state compilation is not the objective, and
   the H4 SGMC-exclusion hypothesis is refuted rather than deferred.
4. **The catalogue collar is real and is worth keeping** (inherited measurement: Δ 0.00744 DTI [0.00522, 0.00982]),
   and the best recorded sibling is the *sparsest* design, consistent with (2).
5. **IR-9 is resolved by evidence.** The official text permits null/NaN outside the study region but does not
   forbid a legal confidence of 0: of the 26 scored rasters readable here, 18 carry NaN outside the region and
   **8 are all-finite with zeros — including the family's best recorded public score (0.2778)**. The H59-J file
   ships the all-finite zeros pattern (which `gems59.gates.format_report`, the check `check_site.py` runs, also
   requires); `src/gems_dti/submission.py` was amended so that both patterns are accepted and only values outside
   [0,1], inf or the −3.4028e38 sentinel fail (IR-59J-03).
6. **IR-11 is resolved.** The official `sample_submission.tif` (sha256 `2176d08e…`) is present in
   `data/official/` and confirms EPSG:32611, 100 m, 3730 × 3292, transform `(100, 0, 243350, 0, −100, 4508550)`,
   5,167,373 finite cells and NaN elsewhere. No origin is guessed anywhere in the pipeline.

NOT settled — the honest limit of this session:

* **Placement quality is unvalidated.** Every local truth proxy available in this sandbox is measured to agree
  with the public board only in the *negative* direction, so no local number can certify that this candidate's
  dots are in better places than a sibling's. The projection therefore rests on the budget argument alone.
* The corridor-restricted instruments (3 px, 6 px, 10 px) do not rescue the holdout; an *off-catalogue* truth
  (whole components plus a ≥ 2 km buffer, scored against SGMC restricted to the withheld area) remains unbuilt.
* No 1 m DEM (the resolution the truth is most likely mapped at) is reachable from this sandbox.

**Three-pass review log:** [`docs/research/three-pass-review-h59j.md`](docs/research/three-pass-review-h59j.md)
(implement → bug/edge-case review → re-check against the brief, with each pass's fixes named).

**Test-suite status, unchanged by this session except where improved:** 36 failures, all referencing artefacts
of other lanes (`docs/archive.html`, `evidence/submission_build_all.json`, `evidence/uniqueness_h57r.json`, the
literal-gate witnesses, Session-3/6/7 receipts). The delivery test
`tests/test_delivery.py::test_current_tiff_zip_links_and_card_are_consistent` now **passes**; two others in
that file advanced past the assertions they used to fail on. The one delivery test that still fails needs the
57GEMSDOE angular-structure study, which was not run and not faked.

## 1. Next session — ordered by expected value per hour

1. **Finish the delivery contract.** `scripts/check_site.py` now resolves its clearance receipt from the card
   (this lane's fix) and passes, but `tests/test_delivery.py` still expects artefacts from the 57GEMSDOE lane:
   `evidence/orientation_structure.json` (an angular-structure study: `distance_quantile_probabilities`
   [.1,.5,.9,.95,.99], relative-strike histogram with `minimum_local_coherence` 0.2, `sample_unit`
   "fault raster pixel; not segment-weighted") plus the matching strings in `docs/method.html`. That study was
   **not** run here and was **not** faked; the rest of that file's contract (pages, feed, preview figure,
   inventory, portfolio, hypotheses, proxy receipt) is now generated by
   `scripts/build_site_pages.py` and `scripts/build_session_evidence.py`.
2. **H59-I4 (off-catalogue holdout, real version).** Withhold whole catalogue components *plus* a ≥ 2 km buffer,
   fit only on the remainder, and score against SGMC inside the withheld area — then check the *rank order* of
   the 24 owner-labelled rasters on it. An instrument that ranks them positively is the only thing that can
   certify placement. Budget: one experiment.
3. **Budget sweep on the live-calibrated relation.** The dot-count trend is measured over 37k–264k dots; it is
   not known where it turns over. Two cheap candidates at ~20k and ~25k dots (same generator, threshold moved)
   would bracket the optimum, and their difference is a *live-scored* measurement once a slot is spent — which
   is why the owner's selector step, not this session, decides whether to spend one.
4. **H2: geodetic-strain × consensus.** The bank's `deq_n100a15` / `ieq_n100a15` planes have the largest partial
   rank correlation with the live score after controlling for emitted count (+0.700 / +0.636) and were never
   used as a *placement* term. One experiment: consensus × strain agreement, same emission geometry.
5. **IR-59J-02: the QFaults MAPSCALE join.** `record_id` does not join to FCODE2023/MAPSCALE
   (349/1126 names intersect; positional join agrees on 1.9 %/0 % of lengths), so the H3 completeness prior is
   blocked. Fix the join by geometry (nearest-trace match on the 100 m raster), not by name.
6. **Effort covariates from official polygons** (USGS reliability-of-location, DOI 10.5066/P93LGLVQ) once a
   session has wider egress than GitHub/npm/pypi.

## Archive — Session H59-I (2026-10-11)

**The H59-I file `downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif` is cleared for download, format-valid, and registry-unique (no duplication identified against all 695 accessible rasters). It is NOT cleared for a competition submission slot.** Its spatially-blocked holdout score, HOLDOUT-DTI **0.08216 [0.06954, 0.09616]** (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 draw), is below the published holdout best **0.135204**. It remains on the site as a negative deliverable. Its uniqueness receipt: [`evidence/uniqueness_h59i.json`](evidence/uniqueness_h59i.json).

## 1. The central unresolved tension (highest priority, inherited) — **addressed in the H59-J section above: the tension is real, measured, and filed as IR-59J-01**

Two instruments disagree about where dots belong:

* the **catalogue hide-and-recover holdout** rewards placing dots *tight against visible traces* — withheld catalogue segments are, by construction, adjacent to other catalogue;
* the **live evidence** (n = 15, owner-reported scores) rewards the opposite: fewer catalogue-adjacent dots (Spearman −0.544) and a median distance of ~19 px from the catalogue.

H59-I followed the holdout (median 5.0 px from the catalogue, min 2.24, p90 9.5). The lane is therefore **optimising the wrong instrument if the live evidence is the true objective**. The single highest-value next measurement is a holdout whose withheld population is genuinely *off*-catalogue — e.g. withhold whole catalogue *components plus a 2 km buffer*, fit on the remainder, and score against the SGMC/state compilation restricted to the withheld area. That experiment is pre-registered as **H59-I4** below and needs a fresh budget.

## 2. What was settled this session

* **Lane implemented and costed.** Presence-only intensity (MaxEnt ≡ Poisson point-process; Phillips et al. 2009 effort covariates; Fithian & Hastie 2013 equivalence), case-control prior correction, γ-swept effort surfaces, 1 km block zoning by new-fault density, and a holdout-derived break-even lattice. Evidence: `evidence/intensity_exp{1,2,3}_*.json`.
* **The 2 px catalogue collar is a real, measured effect.** Collar-2 beats collar-0 by Δ 0.00744 DTI [0.00522, 0.00982] on the same holdout — consistent with the mechanism behind the owner-reported 0.2778 raster (40,199 dots minus 2,545 within 2 px of the catalogue), though no causal claim is made.
* **Spacing is settled at 4 px, not 3 px.** The utility table re-derived on this holdout picked 4 px over 3 px and 6 px for all four density classes; "no lattice" was worse for every class. The prompt's "~3 px when dense" is therefore *not* supported by this holdout, and the deviation is recorded rather than hidden.
* **Zoning has real but modest skill.** 68.5% of withheld-segment pixels fall in the top three deciles of predicted 1 km new-fault density (30% expected under uniformity). SGMC off-catalogue Spearman 0.068–0.153 by fold — weak positive.
* **No leakage.** Each of the 13 covariates alone: max AUC 0.776 (`e_dens5`), min 0.514 — nothing above the 0.90 bar.
* **Registry uniqueness complete.** All 695 accessible rasters, 0 errors: 0 identical decoded predictions, worst Spearman ρ 0.176, worst dot-set Jaccard 0.042 → no duplication identified. The literal 3-px overlap legs fire on 85 rasters and are reported per row, not applied (IR-57-REPR-01 and support asymmetry; the template's own shipped candidates fired them 95 and 85 times and were still shipped). The reviewer should rule on this inherited protocol change.

## 3. Next hypotheses as they were pre-registered at H59-I (status updated 2026-10-11 by H59-J: H59-I2b is implemented as the H59-J budget cap; H59-I4's first corridor version was run and still ranks backwards; H59-I2/I3 are carried into the ordered list above)

1. **H59-I2 — block-level completeness-ratio zoning (gap zoning).** Zone by `c_b = cat_frac_b / λ̄_b`, i.e. *how much of the predicted intensity is already mapped*, instead of by raw predicted new-fault density. Predicted new faults per block should be `n_b = Σλ·(1 − cat_frac_b/λ̄)`, which moves dots off already-mapped blocks and further from the catalogue. Implementation: one new function in `src/gems59/intensity.py`, reuses `tile_blocks` and the Exp2 utility table unchanged.
2. **H59-I2b — budget-capped lattice.** The candidate emits 71,119 dots (1.38% of the footprint), above the ~38–40k dots of the owner-reported top scorers. Tile only the top-N blocks by `n_b` until the dot budget is exhausted, N chosen on the holdout. Cheap: same machinery, one loop.
3. **H59-I3 — off-catalogue effort covariates.** USGS reliability-of-location class, distance to roads, relief and the GeoDAWN acquisition block are the prompt's named effort covariates. Only trace density and relief were obtainable here (egress allow-list: GitHub/npm/pypi only). Official polygons: USGS ScienceBase DOI 10.5066/P93LGLVQ. Fetch them in a session with wider egress, join by fault NUM, and re-fit.
4. **H59-I4 — off-catalogue hide-and-recover holdout** (see §1). This is the instrument that could resolve the holdout-vs-live disagreement, and it is the only pre-registered hypothesis that could change the verdict on its own.

## 4. Statistical and geological limitations

* The lane's covariates are catalogue-derived by construction (effort-density planes from visible traces only), so they encode catalogue proximity. That is disclosed and is why they are the strongest single covariates — but it also means the intensity surface is partly a smoothed catalogue, which the live evidence penalises.
* The holdout's 11,321 withheld positives are *catalogue* segments. Any conclusion about off-catalogue faults is extrapolation.
* The intensity model is a Poisson point process with a logit link and a case-control prior correction; it assumes independence of traces given the covariates, which fault networks violate (they cluster and link). No spatial random effect was fitted.
* `record_id` in the trace table is not a NUM/row index into the FCODE2023/MAPSCALE attribute table (349/1126 names intersect; positional join agrees on 1.9%/0% of lengths). Those attributes are therefore **not** used as covariates; the join is disclosed in the run card.
* The GeoDAWN acquisition-block proxy (k-means on windowed magnetic statistics) was measured to be spatially non-contiguous and dropped (IR-59-EFFORT-02).
* The 4 px lattice spacing, the 2 px collar, the break-even threshold 1.0 and the γ = 0 effort correction are all pre-declared single parameters validated on one holdout draw (seed 20). They are the most obvious targets for the next budget.
* The template audit `scripts/audit_candidate_registry.py` was too slow for the full 695-raster scan in this sandbox's 3 GB RAM; `scripts/uniqueness_fast.py` re-implements the same statistics and thresholds with a KD-tree. The two agree on every statistic they both compute; the fast screen is what produced the numbers above.

## 5. Standing status

* **DOWNLOAD OK**, **SUBMIT NO**. Promotion to a weekly slot is a separate selector decision that belongs to the owner; this project has spent no slot and has no DrivenData credentials, so nothing here has been uploaded.
* Every number on the site is labelled HOLDOUT-DTI (with evaluator version, withheld-positive count and 95% CI) or OWNER-REPORTED / ORGANIZER-PUBLISHED context. No local holdout number is presented as a competition score.
