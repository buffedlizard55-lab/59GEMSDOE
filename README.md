# 59GEMSDOE — DOE GEMS Prize (DrivenData #306): geothermal fault-discovery submission

**Live summary page (GitHub Pages):** `docs/index.html` (publish: *Settings → Pages → Deploy from branch `main`, folder `/docs`*).

> **Status, read this first (updated 2026-10-11):** there is **no validated submission GeoTIFF yet**.
> The organizer's training rasters (`training_features.tif`, `labels.tif`, `sample_submission.tif`)
> are behind a DrivenData login, and the public Dropbox mirrors are unreachable from this sandbox.
> Nothing here has been scored on a holdout or by the organizer. The tooling that turns a probability
> raster into a portal-legal file is built and tested. The first real submission waits on data placement
> (section 5).

---

**Publishing the summary page.** GitHub: *Settings → Pages → Deploy from a branch → `main` → folder `/docs`*.
`docs/.nojekyll` makes Pages serve the folder unchanged.

---

## 0. Standing brief (read at the start of every session)

This section is the working brief we keep in the repo so every session starts from the same aim.
It is condensed from the project prompts; the verbatim requirements remain in the session history.

**Aim.** Produce a GeoTIFF that scores at the top of the DrivenData GEMS leaderboard and that
anyone can download and upload in one step. The score is a distance-weighted Tversky index
(alpha 0.2, beta 0.8, 300 m triangular kernel). The public leaderboard top is 0.3774 (rank #1).

**Core values (Arena).** *Maximize P(Win)*: weigh tradeoffs and choose the path that maximizes the
probability of winning. *Own the Outcome*: act without waiting for permission, and treat failure
and success as signals.

**Evidence rules (non-negotiable).**
1. Every number is labelled as one of:
   - `ORGANIZER-CONFIRMED`: copied from the DrivenData leaderboard or a submission receipt, with the date fetched.
   - `HOLDOUT-DTI`: our own holdout, with the evaluator version, the number of withheld positives, and a 95% CI.
   - `SYNTHETIC`: a check on toy data, never a competition claim.
   - `PROJECTION`: a model estimate. Never written as a score.
2. Every claim links to an official, verifiable source (see `docs/research/sources.md`).
3. Nothing is hallucinated. If something cannot be verified from the sandbox, it is marked
   *unverified* and listed under limitations, not asserted.
4. Irregularities are flagged for human review in `docs/research/irregularities.md`.
5. No manual input is assumed beyond the one data-placement step in section 5.

**Parallel-run protocol (how this lane stays distinct).** Before placement and again on the final dots,
compare a candidate raster against the registry. A rank correlation above 0.90, or more than 70% of
dots within 3 px of one registry raster's dots, means the candidate has drifted into another lane:
log it as a duplicate and stop (`scripts/check_uniqueness.py`).

**Experiment budget.** Three experiments or two hours per lane. Submissions are not chosen here;
promotion to a real slot is a separate selector step, within the weekly cap on the submission page.

**Negative results are deliverables.** Record them in `evidence/` and in the run card.

---

## 1. What the official rules require (verified)

Sources: problem page <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/> and
official rules PDF <https://docs.nlr.gov/docs/fy26osti/96647.pdf> (both fetched 2026-10-11).

| Requirement | Value | Where it is stated |
| --- | --- | --- |
| CRS | EPSG:32611 (UTM 11N) | problem page, "Submission format" |
| Resolution | 100 m | same |
| Bounds | same as training data; outside = null or NaN | same |
| Layers | single band | same |
| Dtype | float32 | same |
| Values | in [0, 1] (probability or confidence) | same |
| Metric | distance-weighted Tversky index, alpha 0.2, beta 0.8, R = 300 m | problem page, "Performance metric" |
| Known faults | masked from evaluation (staff, 2026-09-16) | <https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516> |
| Prize | Phase 1: top 5 on private set, $50k; Phase 2: $250k on expanded labels | rules PDF section 1.1 |
| One submission | single GeoTIFF per team, chosen before results | problem page, "Competition structure" |

## 2. Repo layout

```
src/gems_dti/metric.py          official distance-weighted Tversky index (hand-verified tests)
src/gems_dti/submission.py      writer + validator: CRS, shape, transform, float32, [0,1], NaN outside the template footprint, no sentinel
scripts/download_competition_data.sh   fetch the public Dropbox files; reports what is still missing
scripts/prepare_data.py         inventory of data/ rasters against the official grid
scripts/make_submission.py      probability raster -> validated, uniquely named GeoTIFF + receipt
scripts/validate_submission.py  independent check of any GeoTIFF
scripts/check_uniqueness.py     registry duplicate check (rank correlation + 3 px dot overlap)
scripts/utility_table.py        synthetic re-derivation of the emission (break-even) rule
tests/                          20 unit tests (metric 10, submission writer/validator 10)
docs/index.html                 GitHub Pages summary, submission steps, status
docs/research/                  sources, hypotheses, leaderboard analysis, irregularities, derivation
evidence/                       machine-readable run card and synthetic check outputs
submissions/                    where validated GeoTIFFs and receipts are written (kept small)
data/                           competition inputs (git-ignored, see section 5)
```

## 3. How to run what exists

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                           # 20 tests, all passing at commit time
python scripts/utility_table.py                    # synthetic emission-rule check
bash scripts/download_competition_data.sh           # public files only; DrivenData files need a browser
python scripts/prepare_data.py                      # inventory and grid check
```

## 4. Current results (what is and is not established)

| Item | Status | Label |
| --- | --- | --- |
| Metric implementation matches the official formula | 10 hand-computed tests pass, including the official worked-example arithmetic (TP 3.00, FP 1.89, FN 2.00 -> 0.60). The official example's geometry is an image we could not read, so the geometry itself is not reproduced. | verified against formula |
| Writer and validator enforce the official format | 10 tests pass: range, NaN, sentinel, dtype, CRS, shape, template footprint, note length, name pattern | verified by test |
| Exact emission rule (break-even) | sign identity holds in 500 of 500 random trials | SYNTHETIC |
| Spacing "about 3 px when dense" | **not reproduced** by the synthetic toy (denser is monotonically better there); needs a real holdout | SYNTHETIC, not confirmed |
| Our submission score | none; no candidate raster exists | — |
| Public leaderboard | #1 0.3774; #8 DARD 0.3195; #22 participant `extradr19` 0.2778 | ORGANIZER-CONFIRMED (fetched 2026-10-11) |
| 0.2778 belongs to GEMSDOE32's file | **not confirmed**; the leaderboard shows the participant, not the file | flagged |
| Independent check of the GEMSDOE32 0.2778 candidate file | format passes; sha256 `c55bafc4…` matches its own record; 37,654 positive cells, values only 0 and 1 | verified by our reader |

Full analysis: `docs/research/leaderboard-analysis.md`. Candidate hypotheses: `docs/research/hypotheses.md`.
Flagged irregularities: `docs/research/irregularities.md`. Run card: `evidence/run_card.json`.

## 5. The single blocker, and the exact steps to clear it

The competition inputs are not in the sandbox. Clear it with these steps (no coding required):

1. Log in to <https://www.drivendata.org/competitions/306/competition-doe-gems/data/> and download
   `training_features.tif`, `labels.tif`, `sample_submission.tif` and `1m_DEM_links.csv`.
2. Put them in `data/` (on your machine or in the repo checkout). Git ignores this folder.
3. Also place the Dropbox files if you can reach them (`bash scripts/download_competition_data.sh`).
4. Run `python scripts/prepare_data.py`. It must report `missing_roles: []`.
5. Build a probability raster on the official grid (saved as `.npy` or single-band GeoTIFF with values in [0, 1]).
6. Run `python scripts/make_submission.py --prob <file> --template data/sample_submission.tif --name <unique-name> --note "<≤140 chars>"`. This writes `submissions/<name>.tif` and publishes it to `docs/downloads/` so the summary page offers it for download.
7. Run `python scripts/check_uniqueness.py submissions/<name>.tif <registry_dir>` and confirm no duplicate flag.
8. Upload `submissions/<name>.tif` on the submission page. You can only do this step yourself, because it needs your login.

Nothing in steps 5–6 depends on any particular method. The method is the next piece of work.

## 6. Limitations (what stops a successful project right now)

- **No training data in the sandbox.** Every model claim is blocked until section 5 is done.
- **No DrivenData access** means no organizer score for any candidate, and no check of the weekly submission cap.
- **Sandbox network** allows only GitHub and PyPI for direct downloads. The fetch tool reached the
  official pages, but binary rasters cannot be pulled through it.
- **Competitor repos contain copies of competition files.** `GEMSDOE/data/bridge/` holds
  `existing_faults.tif` and GeoDAWN feature tiles from a third party. This project does not use them.
  Their provenance and redistribution status need human review (see `irregularities.md`).
- **Geotransform not yet confirmed from an official file.** The GEMSDOE32 manifest reports origin
  (243350, 4508550) at 100 m. The writer copies the transform from the official template instead of
  using that number.

## 7. Next work (in order)

1. Place the data (section 5). This unblocks everything below.
2. Build the spatially blocked holdout (hide-and-recover of whole fault segments, with a buffer; derive all catalogue-based features from visible faults only).
3. Run the leakage canary: score each feature alone on the holdout. AUC above 0.90 means leakage until proven otherwise.
4. Validate the top hypothesis from `docs/research/hypotheses.md` against the current holdout best before any slot is spent.
5. Write the run card for each experiment, and promote only what beats the holdout best.
