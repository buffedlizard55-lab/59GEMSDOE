# Remaining work — Session H59-J (2026-10-11)

**CURRENT STATUS: NEGATIVE / STOP. Neither H59-J nor H59-I is cleared to submit. No competition slot has been used or authorized.**

- **H59-J current artifact:** `downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.tif`, SHA256 `44c54063451f1162c6613b9ff621f8791a3199bac2f58c172049a0320f7e7131`, 30,815 dots. The historical 695-raster final-dot scan recorded **86 literal forward 3-px overlap hits** above 0.70 (worst 1.0), which requires **DUPLICATE / STOP** under the standing rule. The newer 696-raster index was not scanned, and no pre-placement surface-gate receipt or candidate-specific leakage-canary receipt is available. Local format PASS is not submission clearance.
- **H59-I archived artifact:** `downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif`, SHA256 `ed1b767f75804048382fc8baf61ff1edad6e9fd7b5b214def715ba5a1e290f6d`. Its historical final-dot scan recorded **85 literal overlap hits** above 0.70; it also was not checked against the 696-raster index and its pre-placement surface gate was not run. Its old stride/tie Spearman estimate is withdrawn.
- **Holdouts:** H59-J HOLDOUT-DTI **0.019124 [0.016393, 0.022148]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives. H59-I HOLDOUT-DTI **0.082161 [0.069542, 0.096158]**, same evaluator and withheld-positive count, below `relay_bend_anatomy` HOLDOUT-DTI **0.135204 [0.119140, 0.153036]**. H59-I remains exploratory because its prior used full-catalogue prevalence and the requested independent target-group background was not implemented; the corrected visible-only runner was not rerun.
- **Projection:** 0.24–0.29 is a low-confidence PROJECTION from owner-maintained dot-count context, not a score and not evidence of gain. File-to-public-board attribution is not established by a submission-page receipt.
- **Run cards:** current H59-J [`evidence/run_card_current.json`](evidence/run_card_current.json) (= `docs/data/run_card.json`); archived H59-I [`evidence/run_card_h59i.json`](evidence/run_card_h59i.json). Corrected H59-J three-pass review: [`docs/research/three-pass-review-h59j.md`](docs/research/three-pass-review-h59j.md). Ranked hypotheses: [`docs/research/hypotheses-v2.md`](docs/research/hypotheses-v2.md).

**Protocol correction:** Any positive H59-J clearance or interpretation of the 86 literal hits as non-operative chance saturation in the historical notes below is superseded. Representation-aware, mutual-overlap, support-matched, Jaccard, or chance-saturation diagnostics do not waive a literal forward-overlap failure. The prior universal-witness claim remains unverified in this checkout and is not needed for either STOP decision.

## 0. Historical H59-J research findings (not clearance evidence)

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
4. **The historical collar comparison is exploratory.** The final run-card DTI and the no-collar point estimate have separate HOLDOUT-DTI intervals, but the committed receipt does not provide a paired CI for that specific contrast. Do not claim a collar gain from the point estimates.
5. **IR-9 is resolved by evidence.** The official text permits null/NaN outside the study region but does not
   forbid a legal confidence of 0: of the 26 scored rasters readable here, 18 carry NaN outside the region and
   **8 are all-finite with zeros.** The public-board participant row at 0.2778 is not a file-linked receipt (the owner page labels the related artifact UNSCORED). The H59-J file
   uses the all-finite zeros pattern (which `gems59.gates.format_report`, the check `check_site.py` runs, also
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

**Full-suite test status (post-repair):** `./.venv/bin/pytest -q` completed with **170 passed, 29 skipped, 0 failed** in 201.41 seconds (12 rasterio deprecation warnings). Artifact-dependent historical checks skip only when their required receipts or raw inputs are absent; no evidence was fabricated. The H59-J delivery check asserts `submission_cleared: false`. The separate orientation study remains unavailable in this checkout and is not claimed as complete.

## 1. Future research backlog — requires a separately budgeted session; not authorized here

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

**H59-I is a negative, inspection-only artifact, not registry-unique under the literal protocol.** Its historical final-dot scan records 85 literal forward-overlap hits above 0.70 on 695 rows; the latest 696-raster index was not scanned, and its surface gate was not run. HOLDOUT-DTI **0.082161 [0.069542, 0.096158]** (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 draw) is exploratory and below same-evaluator `relay_bend_anatomy` HOLDOUT-DTI **0.135204 [0.119140, 0.153036]**. It remains on the site for inspection only. Receipt: [`evidence/uniqueness_h59i.json`](evidence/uniqueness_h59i.json).

## 1. The central unresolved tension (highest priority, inherited) — **addressed in the H59-J section above: the tension is real, measured, and filed as IR-59J-01**

Two instruments disagree about where dots belong:

* the **catalogue hide-and-recover holdout** rewards placing dots *tight against visible traces* — withheld catalogue segments are, by construction, adjacent to other catalogue;
* the **live evidence** (n = 15, owner-reported scores) rewards the opposite: fewer catalogue-adjacent dots (Spearman −0.544) and a median distance of ~19 px from the catalogue.

H59-I followed the holdout (median 5.0 px from the catalogue, min 2.24, p90 9.5). The lane is therefore **optimising the wrong instrument if the live evidence is the true objective**. The single highest-value next measurement is a holdout whose withheld population is genuinely *off*-catalogue — e.g. withhold whole catalogue *components plus a 2 km buffer*, fit on the remainder, and score against the SGMC/state compilation restricted to the withheld area. That experiment is pre-registered as **H59-I4** below and needs a fresh budget.

## 2. What was settled this session

* **Lane implemented and costed.** Presence-only intensity (MaxEnt ≡ Poisson point-process; Phillips et al. 2009 effort covariates; Fithian & Hastie 2013 equivalence), case-control prior correction, γ-swept effort surfaces, 1 km block zoning by new-fault density, and a holdout-derived break-even lattice. Evidence: `evidence/intensity_exp{1,2,3}_*.json`.
* **The 2 px collar comparison remains exploratory.** Separate HOLDOUT-DTI point estimates are recorded in the H59-I run card; the committed receipt does not provide a paired CI for this exact contrast. The owner-maintained 0.2778 file attribution is unverified and is not treated as a score receipt.
* **Spacing is settled at 4 px, not 3 px.** The utility table re-derived on this holdout picked 4 px over 3 px and 6 px for all four density classes; "no lattice" was worse for every class. The prompt's "~3 px when dense" is therefore *not* supported by this holdout, and the deviation is recorded rather than hidden.
* **Zoning has real but modest skill.** 68.5% of withheld-segment pixels fall in the top three deciles of predicted 1 km new-fault density (30% expected under uniformity). SGMC off-catalogue Spearman 0.068–0.153 by fold — weak positive.
* **No leakage.** Each of the 13 covariates alone: max AUC 0.776 (`e_dens5`), min 0.514 — nothing above the 0.90 bar.
* **Registry strict gate fails.** The H59-I final-dot scan recorded 85 literal 3-px forward-overlap hits above 0.70. The earlier representation-aware non-application is withdrawn; the literal hits require DUPLICATE / STOP. The newer 696-raster inventory and the H59-I pre-placement surface were not checked.

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
* `scripts/uniqueness_fast.py` is now a compatibility entry point that delegates to the shared literal registry gate. Historical stride-sampled rank estimates are withdrawn unless revalidated by the shared full-footprint method.

## 5. Standing status

* **DOWNLOAD VALID FOR INSPECTION ONLY**, **SUBMIT NO**. Neither candidate is cleared; no slot has been used or authorized. A future slot requires a separate selector decision.
* Candidate/result holdouts must be labelled HOLDOUT-DTI with evaluator version, withheld-positive count and 95% CI. Public leaderboard values are participant-level ORGANIZER-PUBLISHED context only; file-level attribution and submission-page receipts remain unverified. Projections remain projections.
