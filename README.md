# 59GEMSDOE — GEMS research status (H59-J current, H59-I archived)

**Status reviewed 2026-10-11: NEGATIVE / STOP.** The current H59-J TIFF and archived H59-I TIFF pass their local format checks and are available for inspection, but **neither is cleared to submit**. The strict literal registry rule applies to every indexed raster without a density/representation exemption.

- **H59-J current artifact:** [`downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.tif`](downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.tif), SHA256 `44c54063451f1162c6613b9ff621f8791a3199bac2f58c172049a0320f7e7131`, 30,815 dots. The saved 695-raster final-dot screen recorded **86 literal forward-overlap hits** above 0.70 (worst 1.0). Its former representation-aware `promote-candidate` verdict is superseded; these hits require **DUPLICATE / STOP**. The latest saved 696-raster index was not scanned against this TIFF, and no pre-placement surface-gate receipt is available; the final-dot STOP already rules out submission.
- **H59-I archived artifact:** [`downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif`](downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif), SHA256 `ed1b767f75804048382fc8baf61ff1edad6e9fd7b5b214def715ba5a1e290f6d`. Its historical 695-raster audit recorded **85 literal overlap hits**; the latest 696-raster index was not scanned against it and the pre-placement surface gate was not run. Its old stride/tie Spearman estimate is withdrawn.
- **Download is not submit clearance:** both files are local format/range-valid only. Do not upload either. No slot was used or authorized; no slot recommendation is made.
- **H59-J holdout:** HOLDOUT-DTI **0.01912 [0.01639, 0.02215]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives. This is local instrument evidence, not a leaderboard score. The 0.24–0.29 expectation is a **PROJECTION**, not a score; no gain is claimed.
- **H59-I holdout:** HOLDOUT-DTI **0.082161 [0.069542, 0.096158]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives. This is below the same-evaluator `relay_bend_anatomy` HOLDOUT-DTI reference **0.135204 [0.119140, 0.153036]** (same 11,321 withheld positives); it is exploratory, not an organizer score. Its old case-control prior used full-catalogue prevalence and its requested Phillips et al. (2009) target-group background was not implemented; the corrected visible-only runner was not rerun.
- **Run cards:** current H59-J [`evidence/run_card_current.json`](evidence/run_card_current.json); archived H59-I [`evidence/run_card_h59i.json`](evidence/run_card_h59i.json). H59-I's review record is [`docs/review_passes.md`](docs/review_passes.md); ranked future hypotheses are [`docs/research/hypotheses.md`](docs/research/hypotheses.md); remaining work is [`REMAINING_WORK.md`](REMAINING_WORK.md).
- **Score provenance:** 0.3195 and 0.2778 are participant-level public-board rows, not file-linked receipts. H33-2-B2 attribution to 0.2778 remains unverified; the owner page calls it UNSCORED, and its projected value is not a score. See [`docs/research/leaderboard-analysis.md`](docs/research/leaderboard-analysis.md) and [`docs/research/why-02778.md`](docs/research/why-02778.md).
- **Prior witness limitation:** a prior session asserted a universal-support witness, but its TIFF and verification receipts are absent from this checkout. That assertion is explicitly unverified and not used for either STOP decision; the measured H59-I/H59-J literal overlaps suffice.

The H59-J portal note and submission name remain in the card for provenance only; they are **not authorized for use**. The historical H59-I three-experiment budget is exhausted. No new experiment, candidate TIFF, latest-index scan, candidate-specific leakage canary, or competition submission was made in this review. The sibling band canary's maximum AUC of 0.776 does not certify H59-J's designed field.

## Reproducible, non-submitting data setup

The current pinned bridge inputs and ignored feature caches can be restored with:

```bash
./.venv/bin/python scripts/prepare_data.py --fetch --cache-bands
```

This verifies hashes against the public bridge pin, not an independent organizer provenance receipt. Feature caches under ignored `.cache/` and `out/` are split-sensitive: catalogue-derived features must be rebuilt from visible faults inside each holdout fold. Do not run candidate construction or promotion from the superseded H59-J clearance; the shared literal gate is operative.

## Standing request (re-read at the start of every session)

Review the repo.

THE FOLLOWING IS THE HIGHEST URGENCY AND MUST BE FOLLOWED!

MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION.  DO NOT COPY A PREVIOUS SUBMISSION UNLESS IT'S FOR
LEARNING AND EDUCATION.  BUT WE MUST GENERATE A UNIQUE TIF SUBMISSION.  IT MUST BE OBVIOUS WHETHER IT IS OK
TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.

There should be an easy to download submission tif file as described by the prompt.  Read the entire prompt.

Intensity lane: zone by new-fault density, then tile only what clears break-even. Because a lattice pays only
above roughly 1–2% density, the target is how many new-fault pixels sit in each 1 km block, not a per-pixel
probability. Fit a presence-only point-process intensity from the catalogue, correcting mapping-effort bias
with target-group background (Phillips et al., Ecological Applications, 2009). MaxEnt and Poisson
point-process models are equivalent in form (Fithian and Hastie, Annals of Applied Statistics, 2013). Use
effort covariates (USGS reliability-of-location class, distance to roads, relief, GeoDAWN acquisition block)
alongside the geophysical layers. The 1 km block deliberately exceeds the 400 m registration error Hermant
et al. measured. Check predicted versus withheld-segment counts by decile on the holdout. Then tile only
blocks above break-even, with spacing from a utility table you re-derive on your own holdout (about 3 px
when dense and wider as density falls). Output the standard validated GeoTIFF, uniqueness-checked against
every earlier raster.

PARALLEL-RUN PROTOCOL — read first. This session is one of several running from this same prompt.

1. LANE. Your lane is the single method paragraph below. Stay inside it. If your raster's rank-correlation
   with any registry raster exceeds [0.90], or more than [70%] of your dots fall within 3 px of one registry
   raster's dots, you have drifted into another lane: log it as a duplicate and stop. Check this on the
   surface before placement AND on the final dots.
2. REUSE, DON'T REBUILD. Use the template's cached feature stack, evaluate_holdout.py and
   submission_writer.py. Holdout = hide-and-recover: withhold whole fault segments with a buffer, derive
   every catalogue-based feature only from the visible faults, mask visible faults pixel-exactly, score
   pooled DTI (alpha 0.2, beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it once in the
   template and report it; never keep a private fork.
3. LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, number of withheld positives, 95% CI) or
   ORGANIZER-CONFIRMED (copied from a submission-page receipt). A projection is never written as a score.
4. LEAKAGE CANARY. Test each feature alone on the holdout before trusting any result. AUC above [0.90] means
   leakage until proven otherwise.
5. RUN CARD. End with one JSON card: hypothesis; mechanism; the named non-fault process that could mimic it;
   holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator output (no NaN inside the
   footprint, values in [0,1], CRS/shape/transform match); submission name + note of at most 140 characters;
   verdict promote / negative. Negative results are deliverables.
6. BUDGET. Stop after [3] experiments or [2] hours. Do not pick submissions: promotion to a real slot is a
   separate selector step, within the weekly cap shown on the submission page.

(The full historical brief — sibling-site results, leaderboard context, site-creation instructions, core
values — is preserved verbatim in `TASK_PROMPT.md`; current status at the top of this file takes precedence
over stale claims there.)

## Core values in practice

- **Maximize P(Win):** reject unsupported improvements; fit zones and budgets from training evidence;
  distinguish representations; preserve real weekly slots (none spent).
- **Own the Outcome:** restore data by hash, repair shared tools once and report, publish complete negative
  evidence, keep download/submission status unmistakable. No silent protocol relaxation.

---

## Preserved parallel-session record

A concurrent 59GEMSDOE run (branch `arena/94956c66`, merged to `main` as PR #1/#2) built the
organizer-metric tooling and a summary site but could not place the competition data from this sandbox
(no DrivenData auth, Dropbox off-allow-list) and therefore shipped no validated TIF. Its record is
preserved verbatim at `evidence/run_card_arena94956c66.json`, and its pages remain in `docs/`. This
session restored the hash-pinned 19-band feature stack through the public GitHub mirror bridge
(`scripts/download_features.sh`, `scripts/prepare_data.py`) and therefore supersedes it on data
availability. Re-read `TASK_PROMPT.md` (verbatim standing brief) alongside this file.

---

## Preserved parallel-session record — lane `arena/1ac83927` (PR #4)

A second concurrent 59GEMSDOE session ran the same intensity brief independently, on the
10 km zoning variant with strictly off-catalogue deployment (`scripts/replication_run_lane.py`,
`src/gems59/intensity_replication.py`, receipts `*_lane1ac8.json`, its own site at
[`docs/replication/`](docs/replication/index.html)). It reached the same family of conclusions by a
different route, and its numbers are worse, so it is preserved as the weaker of the two
replications — **do not treat it as a submission**: its run card
([`evidence/run_card_lane_v1.json`](evidence/run_card_lane_v1.json)) is verdict **negative**, download OK /
submit NO, 0 slots used.

What it adds beyond the primary lane:

* **A decile calibration of the lane's own ranking** (`evidence/decile_calibration_lane1ac8.json`).
  The densest decile of 1 km blocks holds **0.0495** of the withheld catalogue positives for
  **0.1014** of the footprint area — a lift of **0.49, below chance** — while ranking blocks by
  catalogue distance or catalogue density reaches lifts of **8.46** and **8.26** (random-ranking
  calibration 0.92, so the method is sound). Discipline: the off-catalogue field is *anti-aligned*
  with the trunk corridors on a frame whose truth sits on them. Recorded as **IR-59-FRAME-01**.
* **A second release-overlap measurement** (`evidence/sibling_release_overlap_lane1ac8.json`): the
  two intensity-lane releases of this family are independent realisations, not copies —
  0.1235 of the sibling's dots within 3 px of this one, rank correlation 0.0435, against a drift
  limit of 0.70 / 0.90.
* The same uniqueness receipt methodology (`evidence/uniqueness_lane_v1.json`, 710 rasters) with the
  saturation artefact of the literal leg called out rather than hidden (IR-59-REPR-01).

Merged additively: the replication's paths are namespaced (`replication_*`, `*_lane1ac8.json`,
`docs/replication/`) so both sessions' records survive in one tree.
