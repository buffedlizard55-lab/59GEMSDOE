# 59GEMSDOE — intensity lane (session H59-I)

**Status (2026-10-11):** DOWNLOAD OK · **SUBMIT NO** — registry-unique, format-valid candidate; holdout DTI
**0.08216 [0.06954, 0.09616]** (evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 draw) is
below the published holdout best 0.135204, so it is **not cleared for a weekly slot** (owner's separate
selector step; no slot spent). Negative results are deliverables.

- **Site:** https://buffedlizard55-lab.github.io/59GEMSDOE/ · [Executive summary / how to submit](docs/executive-summary.html)
- **Download:** [`docs/downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif`](docs/downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif) (SHA256 `ed1b767f…90f6d`, 71,119 binary dots, all-finite, [0,1], EPSG:32611, 3730×3292) + single-TIFF ZIP + writer receipt
- **Run card:** [`evidence/run_card.json`](evidence/run_card.json) (verdict `negative`)
- **Registry uniqueness:** [`evidence/uniqueness_h59i.json`](evidence/uniqueness_h59i.json) — all 695 accessible
  registry rasters, 0 errors: 0 identical decoded predictions, worst Spearman ρ 0.176 (limit 0.90), worst dot-set
  Jaccard 0.042 (limit 0.50) → no duplication identified. The literal 3-px overlap legs fire on
  continuous-surface and support-asymmetric peers (IR-57-REPR-01) and are reported per row, not applied.
- **Method:** presence-only MaxEnt/Poisson point-process intensity (Phillips et al. 2009 effort covariates;
  Fithian & Hastie 2013 equivalence), 1 km new-fault-density zoning, holdout-derived break-even lattice
  (4 px, 2 px catalogue collar)

Reproduce: `python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt`
→ `.venv/bin/python scripts/prepare_data.py --fetch --cache-bands`
→ `.venv/bin/python scripts/run_intensity_lane.py --seed 20 --break-even 1.0`
→ `.venv/bin/python scripts/build_submission_intensity.py --gamma 0.0 --collar 2 --break-even 1.0`

---

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
