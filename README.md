# 59GEMSDOE — Intensity Lane (Session H59-I2)

**Status (2026-10-11):** **DOWNLOAD: OK** · **SUBMIT TO COMPETITION: OK** (Verdict: **`promote`** · 0 weekly slots spent automatically)

- **Promoted Submission GeoTIFF:** [`docs/downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif`](docs/downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif) (`192,643` bytes, SHA256 `088069aec07a9150d9473f94f259bd1f7e4c5212f4ff55c1866bf41b3d1fc8e6`, `77,727` binary dots, `float32`, `[0, 1]`, `0` NaN, `0` outside footprint, `0` on mapped catalogue, `EPSG:32611`, `3730×3292`)
- **Single-TIFF Submission ZIP:** [`docs/downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.zip`](docs/downloads/gems59-h59i2-loco-ipp-corridor-20261011T000000Z.zip) (`134,884` bytes, SHA256 `4fa7074e…` / verified round-trip)
- **Submission Name:** `gems59-h59i2-loco-ipp-corridor-20261011T000000Z`
- **Submission Note (131 / 140 chars):** `H59-I2 Intensity Lane: LOCO two-host Poisson/MaxEnt IPP + QFaults effort join + 1km 1.5% break-even zoning + {none,s4,s4,s3} tiling`
- **4-Fold Pooled Holdout Score (`HOLDOUT-DTI`, `gems57-pooled-hide-v2`, 11,321 withheld positives, seed 20):** **`0.166789 [0.153242, 0.179658]`** (`NW=0.15710, NE=0.15742, SW=0.17929, SE=0.16428`) — **+103.00% (`+0.084628`)** above Session H59-I (`0.082161 [0.069542, 0.096158]`) and **+23.36% (`+0.031585`)** above the published all-lane holdout best (`relay_bend_anatomy`, `0.135204 [0.119140, 0.153036]`), with the lower 95% CI bound (`0.153242`) exceeding the published best's upper 95% CI bound (`0.153036`).
- **1 km Block Decile Calibration (`52,309` footprint blocks):** Strictly monotone across all 10 deciles (`59 -> 112 -> 175 -> 252 -> 424 -> 527 -> 913 -> 1,448 -> 2,649 -> 4,762` withheld pixels); **`78.25%` (`8,859 / 11,321`)** of withheld pixels sit in the top 3 deciles (`D8–D10`, vs `68.46%` in H59-I).
- **Leakage Canary (`31` covariates):** `0` flags (`max holdout AUC = 0.7715 < 0.90`, `max train LOCO AUC = 0.7439 < 0.90`).
- **695-Raster + Session H59-I Uniqueness Audit (`evidence/uniqueness_h59i2.json`):** `0` identical decoded predictions; pre-placement continuous surface Spearman `|ρ| = 0.8568 <= 0.90` (`0.1252` vs H59-I); final binary dot Spearman `|ρ| = 0.1655 <= 0.90` (`0.0279` vs H59-I); Jaccard `0.0585 <= 0.50` (`0.0226` vs H59-I); forward 3 px overlap vs all `487` peers of `<= 77,727` dots is `0.6473 <= 0.70` (`0.5722 <= 0.70` vs Session H59-I).
- **Run Card & Site:** [`evidence/run_card.json`](evidence/run_card.json) · [`docs/index.html`](docs/index.html) · [`docs/executive-summary.html`](docs/executive-summary.html) · [`docs/method.html`](docs/method.html) · [`REMAINING_WORK.md`](REMAINING_WORK.md)

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
