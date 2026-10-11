# Three-pass review record — H59-I (2026-10-11 UTC)

This is one agent's three distinct review passes, not three independent human reviews. It records the review scope, evidence checked, decision and unresolved items. No model experiment, candidate construction, or competition-slot action was performed in these passes.

## Pass 1 — Evidence provenance and attribution

**Checked:** `TASK_PROMPT.md`, `README.md`, `evidence/leaderboard_snapshot_2026-10-11.json`, `docs/research/leaderboard-analysis.md`, and the saved owner-repository H33 evidence references.

**Findings:**

- The public leaderboard snapshot lists 0.3195 for DARD at rank 8 and 0.2778 for extradr19 at rank 22. These are participant-level public-board values, not file-linked submission-page receipts.
- The 0.2778 ↔ GEMSDOE32 H33-2-B2 TIFF link is unverified. The owner-maintained GEMSDOE32 page at commit `3ae286af599fdf86dbb0f333591f8273efcee806` labels H33-2-B2 UNSCORED and says no organizer score exists for any artifact in that repo.
- The owner's audit field `predicted_leaderboard_dti` is a model projection, not an organizer score.
- H59-I values are local HOLDOUT-DTI only. No organizer receipt for the H59-I TIFF exists.

**Action:** updated [`leaderboard-analysis.md`](research/leaderboard-analysis.md), [`irregularities.md`](research/irregularities.md), README and the run card. No projection is called a score.

## Pass 2 — Scientific and holdout validity

**Checked:** `src/gems59/intensity.py`, `scripts/run_intensity_lane.py`, the three historical intensity evidence receipts, feature-cache receipts, and the 19-band feature mapping.

**Findings:**

- The historical H59-I model used uniform background and catalogue-derived density/relief proxies. It did not implement the requested independent target-group background.
- The historical case-control prior used full-catalogue prevalence. This exposes withheld label mass; the runner is now changed to visible-only prevalence, but no rerun was made.
- The selected historical local result is **HOLDOUT-DTI 0.082161 [0.069542, 0.096158]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives. It is below the comparable local reference and remains exploratory because of the prior issue and missing target-group background.
- The old feature-alone AUC canary is a leakage screen; it does not validate the full method or repair the prior issue.
- The prior three-experiment budget is exhausted. No further experiment was run.

**Action:** corrected method docstrings, run-card caveats and the visible-only prevalence implementation; left historical results intact but explicitly unqualified. Added a five-hypothesis qualitative research ranking without DTI projections.

## Pass 3 — Registry, validator, and delivery governance

**Checked:** `evidence/uniqueness_h59i.json`, the latest saved 696-raster index, `src/gems59/uniqueness.py`, `scripts/uniqueness_fast.py`, `scripts/finalize_uniqueness.py`, the candidate writer receipt, and build/holdout entry points.

**Findings:**

- The historical final-dot screen measured 85 literal forward-overlap firings above 0.70 on its 695-raster snapshot; worst overlap was 1.0. This alone requires duplicate/STOP under the literal rule.
- The latest saved index contains 696 rasters and was not scanned against this artifact. The prior exact rank-correlation claim was based on stride-3 sampling with incorrect tie handling; exact rho is not reported.
- No pre-placement surface gate was run for the historical artifact. The shared exact registry gate is now wired into future holdout placement and submission construction before any output, followed by a final-dot gate before export. The builder also has a holdout gate and refuses the historical below-reference result.
- The saved TIFF passes its local format/value validator, but validator PASS does not equal registry uniqueness or submission clearance.

**Action:** removed representation-aware overrides, made literal overlap failures operative, marked the registry comparison incomplete for the latest saved inventory, kept the file available only for inspection, and set slot/submit status to NO.

## Final decision

**NEGATIVE / STOP.** The existing TIFF is download-valid for inspection but not cleared to submit. Do not use a weekly slot. No score gain is claimed.
