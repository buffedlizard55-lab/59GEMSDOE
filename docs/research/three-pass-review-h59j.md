# Three-pass review — session H59-J (2026-10-11)

The standing brief requires the work to be done in passes (implement → review bugs, missing requirements and
edge cases → re-check against the original request) and the result to be verified line by line. This is that
log for session H59-J. Every claim below points at the receipt that verifies it.

## Protocol correction (after the original H59-J passes)

The original H59-J review treated 86 forward-overlap firings as non-operative chance saturation and issued a
positive submit verdict. That interpretation conflicts with the standing literal rule. The strict re-review
supersedes those clearance conclusions: **86 literal forward 3-px overlap hits above 0.70 on 695 rasters mean
DUPLICATE / STOP**. The newer 696-raster index was not scanned against H59-J, and no pre-placement surface-gate receipt is available. The saved H59-J artifact is therefore download-valid for inspection only, not cleared to submit; the older H59-I artifact separately has 85 literal hits and is also STOP. No competition slot was used or authorized.

## Pass 1 — implement and verify

| Step | What was done | Receipt |
| --- | --- | --- |
| Feature stack | Hash-pinned 19-band GeoDAWN stack restored and byte-verified; feature bank derived | `evidence/data_preparation.json` (ALL PINS VERIFIED), `out/features/bank_u8.npy` (38×3730×3292) |
| Registry cache | All 695 pinned registry rasters re-fetched by blob id and sha256-checked | `evidence/registry_cache_restore.json` (695/695, 0 errors) |
| Instrument calibration | Every owner-labelled scored raster scored on the shared holdout; correlated with the recorded public score | `evidence/h59j_live_calibration.json` (−0.437, p 0.033, n 24; dot count −0.818, p < 1e-4) |
| Corridor variant | Same, with withheld truth restricted to ≥ 3/6/10 px from visible traces | `evidence/h59j_corridor_calibration.json` (still negative: −0.427 at 6 px) |
| Candidate | H59-J built: structure-tensor ridge coherence consensus over bands 12/2/13, 3×3 NMS walked ±2 px along the ridge, 2 px collar, budget 36,000 | `downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.{tif,zip,json}` |
| Format | 15/15 local checks and the portal preflight | `evidence/run_card_current.json` `validator_output`, `portal_preflight` |
| Historical registry scan | 695/695 rasters in the 2026-10-10 pinned index screened; 86 literal forward-overlap hits; newer 696-raster index not scanned | `evidence/uniqueness_h59j.json` plus `strict_literal_gate` |

| Holdout with CI | Shared evaluator, seed-20 draw, 200 px spatial-cluster bootstrap (1000 draws) | `evidence/h59j_candidate_ci.json` (0.01912 [0.01639, 0.02215]) |
| Proxy instrument | SGMC off-catalogue proxy scored for all 24 labelled rasters | `evidence/offcat_instrument_check.json` (−0.665, p 0.0004) |

Two results from Pass 1 changed the session's plan and are the reason the shipped budget is 30,815 dots rather
than the 71,119 of the archived H59-I file: the holdout is rank-inverted against the public board (IR-59J-01),
and the emitted dot count is the live-calibrated driver.

## Pass 2 — bugs, missing requirements, edge cases (including a subsequent protocol correction)

| # | Defect found | Fix | Where |
| --- | --- | --- | --- |
| 1 | `scripts/validate_submission.py` **rejected the shipped file** (“has finite values outside the template footprint”) because `src/gems_dti/submission.py` demanded NaN outside while the template writer emits zeros — and the build receipt said the file was fine. Two local gates disagreed about the same file. | The gate now follows the official text plus the acceptance record: 18 of 26 scored rasters are NaN-outside, 8 are zeros-outside and the zeros group holds the family's best score. `validate_array` reports the convention and fails only outside-[0,1], inf or the −3.4028e38 sentinel. Both local format validators now accept the legal NaN-outside and zeros-outside conventions; that format result is not submission clearance. | `src/gems_dti/submission.py`; irregularity IR-59J-03 |
| 2 | The original review treated the literal forward-overlap leg's 86 H59-J firings as non-operative because of density / representation. That was inconsistent with the standing literal rule. | **Correction:** the exemption is withdrawn. Keep all 86 firings operative; H59-J is DUPLICATE / STOP. The compatibility CLI delegates to the shared literal gate. | `scripts/uniqueness_fast.py`; `evidence/uniqueness_h59j.json`; IR-59J-05 |
| 3 | The repository's delivery QA crashed with `FileNotFoundError` in this repo: it was pinned to another lane's artefact path. | The clearance receipt path is read from the run card, with the old path kept as a fallback. | `scripts/check_site.py`; IR-59J-04 |
| 4 | Both calibration scripts crashed with `ValueError: Out of range float values are not JSON compliant: nan` when a comparison set was empty (a brand-new candidate has no recorded score). | `--out` flag and JSON `null` emission for empty Spearman sets. | `scripts/exp_live_calibration.py`, `scripts/exp_corridor_calibration.py` |
| 5 | The published pages still quoted the H59-I file as current, with its old sha, dot count and clearance numbers. | Pages regenerated for H59-J; the four new pages are built from receipts by `scripts/build_site_pages.py`, and 35 published claims are machine-checked against the receipts. | `scripts/verify_session_claims.py` (39/39 pass) |
| 6 | Hypothesis H4 (build on the state compilation) was listed as merely “risky”; the new proxy measurement shows it is **backwards**. | H4 updated with the measurement and its additive form marked falsified. | `docs/research/hypotheses-v2.md` |
| 7 | Two micro-peer rasters (1 and 14 dots) made the *reverse* overlap leg read 1.0 / 0.5 — degenerate, since a single dot lands within 3 px of one of 30,815 dots with probability ≈ 0.17. | Legs are reported against support-matched peers (≥ 30 dots; 690 of 695) and the degenerate pair is listed with its reason. | `evidence/uniqueness_h59j_view.json` |

Edge cases checked after the fixes: ZIP holds exactly one member and its payload is byte-identical to the
TIFF; `docs/downloads/` mirrors are byte-identical to `downloads/`; the nodata tag is `None`; zero positive
cells on the mapped catalogue; zero positive cells outside the study region; emitted count identical in the
validator, the preflight and the card; `docs/data/run_card.json` byte-identical to
`evidence/run_card_current.json`; every local link on the site resolves.

## Pass 3 — re-check against the original request (strict correction)

| Requirement (standing brief) | Current status | Evidence |
| --- | --- | --- |
| Unique TIF / literal registry rule | **NOT CLEARED.** The saved H59-J scan recorded 86 forward 3-px overlap hits >0.70; the literal rule requires STOP. The previous representation-aware override is withdrawn. | `evidence/uniqueness_h59j.json`, `evidence/run_card_current.json` |
| Surface-before-placement registry check | **NOT VERIFIED.** No pre-placement surface-gate receipt is present in this checkout. This is an additional protocol gap; the final-dot literal failure independently requires STOP. | `evidence/run_card_current.json` `registry_gate_phases` |
| Candidate-specific leakage canary | **NOT VERIFIED.** H59-J's designed field was not rerun through a per-feature canary; the sibling-band maximum AUC 0.776 is not a substitute. | `evidence/run_card_current.json` `leakage_canary` |
| Download and submit status must be obvious | **Download-valid for inspection; not cleared to submit.** The old submit-OK page/card verdict is superseded. The later 696-raster index was not scanned against H59-J. | `docs/index.html`, `docs/executive-summary.html`, current H59-J run card |
| Geological hypotheses and holdout | H59-J HOLDOUT-DTI 0.019124 [0.016393, 0.022148], evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives. It does not override the registry STOP or represent a leaderboard score. H59-I's prior was unqualified and its budget is exhausted. | `evidence/h59j_candidate_ci.json`, `evidence/run_card_h59i.json` |
| Reuse shared tools | The H59-I experiment/build entry points fail closed on missing target-group background and require literal surface/final-dot checks before output. The compatibility uniqueness CLI now delegates to the shared literal gate. | `scripts/run_intensity_lane.py`, `scripts/build_submission_intensity.py`, `scripts/uniqueness_fast.py` |
| Label scores and preserve provenance | Public-board rows, local HOLDOUT-DTI, and projections remain distinct; no H59-J or H59-I organizer receipt is claimed here. | `docs/research/leaderboard-analysis.md`, run cards |
| One JSON run card | **Done, negative verdict.** H59-J and H59-I have separate cards; neither authorizes a slot. | `evidence/run_card_current.json`, `evidence/run_card_h59i.json` |
| Budget and slot | No weekly submission slot was used or authorized. The historical H59-I experiment budget remains exhausted. | Both run cards; `REMAINING_WORK.md` |

## What is explicitly NOT satisfied, and why

* **The H59-J submission gates remain STOP.** Its saved 695-raster final-dot scan has 86 literal forward-overlap hits; the 696-raster index was not scanned, and no pre-placement surface-gate or candidate-specific leakage-canary receipt is present. H59-I remains negative too; neither artifact is cleared, and no slot has been used or authorized.
* `tests/test_delivery.py::test_preview_png_and_holdout_structure_labels_are_real` is skipped because it requires
  `evidence/orientation_structure.json` (an angular-structure study with quantile distances, a relative-strike
  histogram and `minimum_local_coherence` 0.2) plus four matching strings in `docs/method.html`. That study
  belongs to the 57GEMSDOE lane and **was not run and not fabricated** here.
* The full suite was rerun after test repairs: **170 passed, 29 skipped, 0 failed** in 201.41 seconds
  (12 rasterio deprecation warnings). Historical receipt-dependent tests skip only when required sibling-lane
  artifacts are absent; no receipt was synthesized. See `REMAINING_WORK.md` for the measured status.
* No local instrument available in this sandbox ranks candidate placements in agreement with the public board,
  so this session cannot certify placement quality; the projection rests on the budget argument alone.
