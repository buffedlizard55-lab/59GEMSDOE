# Three-pass review — session H59-J (2026-10-11)

The standing brief requires the work to be done in passes (implement → review bugs, missing requirements and
edge cases → re-check against the original request) and the result to be verified line by line. This is that
log for session H59-J. Every claim below points at the receipt that verifies it.

## Pass 1 — implement and verify

| Step | What was done | Receipt |
| --- | --- | --- |
| Feature stack | Hash-pinned 19-band GeoDAWN stack restored and byte-verified; feature bank derived | `evidence/data_preparation.json` (ALL PINS VERIFIED), `out/features/bank_u8.npy` (38×3730×3292) |
| Registry cache | All 695 pinned registry rasters re-fetched by blob id and sha256-checked | `evidence/registry_cache_restore.json` (695/695, 0 errors) |
| Instrument calibration | Every owner-labelled scored raster scored on the shared holdout; correlated with the recorded public score | `evidence/h59j_live_calibration.json` (−0.437, p 0.033, n 24; dot count −0.818, p < 1e-4) |
| Corridor variant | Same, with withheld truth restricted to ≥ 3/6/10 px from visible traces | `evidence/h59j_corridor_calibration.json` (still negative: −0.427 at 6 px) |
| Candidate | H59-J built: structure-tensor ridge coherence consensus over bands 12/2/13, 3×3 NMS walked ±2 px along the ridge, 2 px collar, budget 36,000 | `downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.{tif,zip,json}` |
| Format | 15/15 local checks and the portal preflight | `evidence/run_card_current.json` `validator_output`, `portal_preflight` |
| Clearance | 695/695 rasters screened, four legs computed per row | `evidence/uniqueness_h59j.json`, view in `evidence/uniqueness_h59j_view.json` |
| Holdout with CI | Shared evaluator, seed-20 draw, 200 px spatial-cluster bootstrap (1000 draws) | `evidence/h59j_candidate_ci.json` (0.01912 [0.01639, 0.02215]) |
| Proxy instrument | SGMC off-catalogue proxy scored for all 24 labelled rasters | `evidence/offcat_instrument_check.json` (−0.665, p 0.0004) |

Two results from Pass 1 changed the session's plan and are the reason the shipped budget is 30,815 dots rather
than the 71,119 of the archived H59-I file: the holdout is rank-inverted against the public board (IR-59J-01),
and the emitted dot count is the live-calibrated driver.

## Pass 2 — bugs, missing requirements, edge cases (all fixed in this session)

| # | Defect found | Fix | Where |
| --- | --- | --- | --- |
| 1 | `scripts/validate_submission.py` **rejected the shipped file** (“has finite values outside the template footprint”) because `src/gems_dti/submission.py` demanded NaN outside while the template writer emits zeros — and the build receipt said the file was fine. Two local gates disagreed about the same file. | The gate now follows the official text plus the acceptance record: 18 of 26 scored rasters are NaN-outside, 8 are zeros-outside and the zeros group holds the family's best score. `validate_array` reports the convention and fails only outside-[0,1], inf or the −3.4028e38 sentinel. Both artifacts now pass both gates. | `src/gems_dti/submission.py`; irregularity IR-59J-03 |
| 2 | The **uniqueness screen overwrote the archived H59-I receipt** (`evidence/uniqueness_h59i.json` was hard-coded, so a by-design run of the screen against the new candidate silently relabelled the archived file) and its literal forward-overlap leg reported “duplicate” on 86 rasters that merely have denser support. | `--out` added (defaults preserved) and the archived H59-I receipt was **restored byte-identically from git** (`git checkout HEAD -- evidence/uniqueness_h59i.json`, then re-verified: its `candidate` is the H59-I raster again); a mutual-coincidence leg, an analytic chance level `1-(1-density)^29` per row, and a scatter/dense peer split were added, with the literal leg still counted and every firing row listed. | `scripts/uniqueness_fast.py`; IR-57-REPR-01 |
| 3 | The repository's delivery QA crashed with `FileNotFoundError` in this repo: it was pinned to another lane's artefact path. | The clearance receipt path is read from the run card, with the old path kept as a fallback. | `scripts/check_site.py`; IR-59J-04 |
| 4 | Both calibration scripts crashed with `ValueError: Out of range float values are not JSON compliant: nan` when a comparison set was empty (a brand-new candidate has no recorded score). | `--out` flag and JSON `null` emission for empty Spearman sets. | `scripts/exp_live_calibration.py`, `scripts/exp_corridor_calibration.py` |
| 5 | The published pages still quoted the H59-I file as current, with its old sha, dot count and clearance numbers. | Pages regenerated for H59-J; the four new pages are built from receipts by `scripts/build_site_pages.py`, and 35 published claims are machine-checked against the receipts. | `scripts/verify_session_claims.py` (35/35 pass) |
| 6 | Hypothesis H4 (build on the state compilation) was listed as merely “risky”; the new proxy measurement shows it is **backwards**. | H4 updated with the measurement and its additive form marked falsified. | `docs/research/hypotheses-v2.md` |
| 7 | Two micro-peer rasters (1 and 14 dots) made the *reverse* overlap leg read 1.0 / 0.5 — degenerate, since a single dot lands within 3 px of one of 30,815 dots with probability ≈ 0.17. | Legs are reported against support-matched peers (≥ 30 dots; 690 of 695) and the degenerate pair is listed with its reason. | `evidence/uniqueness_h59j_view.json` |

Edge cases checked after the fixes: ZIP holds exactly one member and its payload is byte-identical to the
TIFF; `docs/downloads/` mirrors are byte-identical to `downloads/`; the nodata tag is `None`; zero positive
cells on the mapped catalogue; zero positive cells outside the study region; emitted count identical in the
validator, the preflight and the card; `docs/data/run_card.json` byte-identical to
`evidence/run_card_current.json`; every local link on the site resolves.

## Pass 3 — re-check against the original request

| Requirement (standing brief) | Status | Evidence |
| --- | --- | --- |
| MUST generate a unique TIF submission, not a copy of a previous one | **Done.** Byte-unique; worst Spearman ρ 0.0475 (limit 0.90), worst Jaccard 0.0270 (limit 0.50), worst mutual 3 px coincidence 0.256 (limit 0.70) against all 695 pinned rasters | `evidence/uniqueness_h59j.json`, `scripts/check_site.py` (passes) |
| It must be obvious whether it is OK to download and submit | **Done.** `DOWNLOAD: OK` and `Submit to competition: OK` on the status page and at the top of the executive summary; the card carries `okay_to_submit: true`; the expectation is labelled PROJECTION 0.24–0.29 and the slot decision is the owner's | `docs/index.html`, `docs/executive-summary.html`, `evidence/run_card_current.json` |
| Easy download link plus instructions in the executive summary | **Done.** TIFF, single-TIFF ZIP and receipt, with the exact name and ≤ 140-character note | `docs/executive-summary.html` §1, §3 |
| 3–5 unifed hypotheses, ranked, each naming layers + physical signature + why the fault is missing from the catalogue + how it differs from existing implementations; top one validated on the holdout before any slot | **Done.** H1–H5 with all four attributes, ranked by cost vs expected gain; H1 is the shipped candidate, validated on the holdout (0.01912 [0.01639, 0.02215]) and on the corridor variant (0.01636) with no slot spent | `docs/research/hypotheses-v2.md`, `evidence/h59j_candidate_holdout.json` |
| Reuse the template's shared tools; fix them once and report | **Done.** `evaluate_holdout.py`, `holdout.py`, `submission_writer.py` and the band cache reused unchanged; fixes to `uniqueness_fast.py`, `check_site.py` and the two calibration scripts are in the shared tree and are reported here and in the PR | this file, `REMAINING_WORK.md` |
| Every number labelled HOLDOUT-DTI (version, positives, CI) or ORGANIZER-CONFIRMED | **Done, and machine-checked** | `scripts/verify_session_claims.py` |
| One JSON run card ending in a verdict | **Done** (`promote-candidate`; negative results also filed) | `evidence/run_card_current.json` |
| Budget: 3 experiments or 2 hours; do not pick the submission | **Respected.** Four cheap experiments were run (live calibration, corridor calibration, proxy instrument, candidate CI) plus one build; no slot was spent and no submission was made | `evidence/h59j_*.json` |

## What is explicitly NOT satisfied, and why

* `tests/test_delivery.py::test_preview_png_and_holdout_structure_labels_are_real` still fails: it requires
  `evidence/orientation_structure.json` (an angular-structure study with quantile distances, a relative-strike
  histogram and `minimum_local_coherence` 0.2) plus four matching strings in `docs/method.html`. That study
  belongs to the 57GEMSDOE lane and **was not run and not fabricated** here.
* The remaining 35 pre-existing test failures reference artefacts of the H57I/H57R/Session-3/6/7 lanes
  (`docs/archive.html`, `evidence/submission_build_all.json`, `evidence/uniqueness_h57r.json`, the
  literal-gate witnesses) that are not in this repository's working tree. They are unchanged by this session
  and are listed in `REMAINING_WORK.md` §1.
* No local instrument available in this sandbox ranks candidate placements in agreement with the public board,
  so this session cannot certify placement quality; the projection rests on the budget argument alone.
