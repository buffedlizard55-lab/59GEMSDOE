# Remaining work — Session H59-I (2026-10-11)

**The H59-I file `downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif` is cleared for download, format-valid, and registry-unique (no duplication identified against all 695 accessible rasters). It is NOT cleared for a competition submission slot.** Its spatially-blocked holdout score, HOLDOUT-DTI **0.08216 [0.06954, 0.09616]** (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 draw), is below the published holdout best **0.135204**. The budget (3 experiments / 2 h) is exhausted. **No competition slot was spent by this project.**

Site: https://buffedlizard55-lab.github.io/59GEMSDOE/ · Run card: [`evidence/run_card.json`](evidence/run_card.json) · Uniqueness: [`evidence/uniqueness_h59i.json`](evidence/uniqueness_h59i.json)

## 1. The central unresolved tension (highest priority, inherited, still open)

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

## 3. Next hypotheses, pre-registered, none run

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
