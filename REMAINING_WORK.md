# Remaining work — H59-I (status reviewed 2026-10-11 UTC)

## Current decision: download-valid for inspection; **not cleared to submit**

The existing artifact `downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif` is a valid, locally validated GeoTIFF and may be downloaded for inspection. It is **not** cleared as a competition submission. Do not upload it; do not spend a weekly slot; no slot was used or authorized.

Why the decision is negative:

1. The historical final-dot audit found **85 literal >70% forward-overlap firings** among the 695 rasters in its dated scan. Under the standing rule, that is a duplicate/STOP. The previous representation-aware override was incorrect and is now removed from the operative decision.
2. The latest saved index contains 696 rasters, but it was not scanned against this artifact. Historical exact full-footprint rank correlation was not verified: the old screen used stride-3 samples and incorrect tie ranking, so its reported `rho≈0.176` is withdrawn.
3. No pre-placement surface gate was run for the historical artifact. New experiment and build scripts now call the shared surface gate before placement and a final-dot gate before export.
4. Historical **HOLDOUT-DTI 0.082161 [0.069542, 0.096158]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed 20, is below the same-evaluator `relay_bend_anatomy` reference **0.135204 [0.119140, 0.153036]**. This local result is exploratory, not a competition score. The old run used a full-catalogue prevalence in its case-control prior; the runner now uses visible-only prevalence, but it has not been rerun. The requested target-group background was not implemented.
5. The historical experiment budget is used: **3 experiments**. No new holdout experiment or candidate TIFF was produced in this review.

The stored image remains a research artifact for inspection; its byte-level identity and local validator PASS do not override its registry failure. Run card: [`evidence/run_card.json`](evidence/run_card.json). Site: <https://buffedlizard55-lab.github.io/59GEMSDOE/>. Research ranking: [`docs/research/hypotheses.md`](docs/research/hypotheses.md). Three review passes: [`docs/review_passes.md`](docs/review_passes.md).

## Highest-priority blockers before future research

### A. Registry and literal-gate issue

- Correctly apply the exact shared thresholds to every hash-pinned public-owner raster: full-footprint Spearman `rho > 0.90` OR more than `0.70` of candidate positive cells within a 3-pixel Euclidean neighborhood of one prior's positive support means duplicate/STOP. No dense-raster, representation, Jaccard, or directional-support exemption may override that rule.
- `evidence/uniqueness_h59i.json` preserves the historical 695-row final-dot measurements. The literal forward-overlap rule fired 85 times, with worst overlap 1.0. The exact `rho` statistic was not rerun after the old stride/tie-ranking flaw was found.
- The later `evidence/registry_refreshed_20261010T2001.json` contains 696 rasters and has not been compared to this artifact. Current registry completeness for this artifact is therefore **false**.
- A prior session asserted that an indexed positive-support witness covers every allowed footprint cell within 3 px, but the pinned witness TIFF and its verification receipts (`evidence/literal_gate_witness_verification_20261010.json`, `evidence/uniqueness_saturation_certificate.json`, and `evidence/session7_registry_precheck.json`) are absent from this checkout. Treat that universal-blocker claim as unverified here. The independently retained 85 historical literal overlap firings are sufficient for the current STOP; do not rely on the missing witness or a representation-aware subset as clearance.

### B. Presence-only method is incomplete

- The user-requested independent **target-group background** is not implemented. No pinned, independent, spatially matched same-effort non-target mapping inventory has been established. Do not substitute the target fault catalogue itself as target-group background; that would be circular.
- The old H59-I fit used uniform background plus trace-density and relief proxies. These are not equivalent to target-group background.
- The old run's case-control prior used full-catalogue prevalence. The runner was corrected to estimate it from visible presences only, but the historical scores must remain labelled exploratory until a properly gated rerun occurs.
- Official GeoDAWN acquisition blocks, USGS reliability-of-location classes and distance-to-road effort layers are not pinned. ScienceBase and other required sources were not reachable under the sandbox host allow-list.

### C. Holdout and experiments

- The old HOLDOUT-DTI used a spatially blocked, whole-component buffered split and shared evaluator, but its prior leakage makes it unqualified for promotion.
- Do not add experiments in this run: the 3-experiment / 2-hour limit is exhausted. A new authorization/budget is needed after blockers A and B have a defensible path.
- Any future holdout must rebuild catalogue-derived covariates and prevalence from visible faults only, mask visible faults pixel-exactly, and score pooled DTI α=0.2, β=0.8, 300 m triangular kernel. Use the shared evaluator and submission writer; do not fork them.
- Run the leakage canary per feature before trusting results. AUC >0.90 is leakage until explained.

## Geological research priorities (no candidate is validated)

See [`docs/research/hypotheses.md`](docs/research/hypotheses.md), which ranks five mechanisms with layers, physical signatures, non-fault mimics, registry overlap, data availability, expected research value and cost. The highest-information candidate is an independent geothermal-spring / discharge inventory, but its data download, coverage and effort are unverified. The other candidates overlap strongly with existing sibling submissions or have source/join limitations. None has a holdout result or slot recommendation.

## 0.2778 / 0.3195 score provenance

- **0.3195:** ORGANIZER-PUBLISHED public leaderboard value for participant DARD at rank 8 in the 2026-10-11 snapshot. It is not the overall highest score and has no captured file-level receipt.
- **0.2778:** ORGANIZER-PUBLISHED public leaderboard value for participant extradr19 at rank 22. The mapping to GEMSDOE32's `h33-h33-2-b2` TIFF is **unverified**. At owner commit `3ae286af599fdf86dbb0f333591f8273efcee806`, the GEMSDOE32 page calls H33-2-B2 **UNSCORED** and says no organizer score exists for artifacts in that repository. The owner audit's `predicted_leaderboard_dti` is a **PROJECTION**, not a score.
- H59-I's local DTI is HOLDOUT-DTI, never an organizer score. No submission-page receipt exists for this file.

## Three review passes (this turn)

1. **Evidence provenance:** separated public-board participant scores, owner file reports, local file measurements, local HOLDOUT-DTI and projections; corrected the H33-2-B2 attribution language.
2. **Scientific / validation:** recorded target-group background as unimplemented, exposed the historical prior leakage, preserved the old HOLDOUT-DTI as exploratory, and did not run another experiment.
3. **Registry / delivery:** made the literal overlap rule operative, marked the 696-raster comparison incomplete, retained the local GeoTIFF only for inspection, and kept submission/slot status NO.

## Reproducibility

The pinned input and feature-bank setup currently succeeds with:

```bash
./.venv/bin/python scripts/prepare_data.py --fetch --cache-bands
```

That restores hash-verified bridge bytes; it is not an organizer provenance receipt. Do **not** run `scripts/run_intensity_lane.py` or `scripts/build_submission_intensity.py` to create a candidate until the explicit protocol blockers are resolved. The current scripts fail closed on the registry/holdout gates and never use a competition slot.
