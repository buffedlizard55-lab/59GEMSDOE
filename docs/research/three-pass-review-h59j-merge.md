# Three-pass review — 2026-10-11 merge of the H59-J lanes

Scope: the merge of `origin/main` (parallel lanes `arena/1ee0553d` + `arena/1ac83927`, already merged
upstream) into this session's branch `arena/fba37db3`, and the release bookkeeping that follows from it.
Written by this session; every number below is reproducible from a receipt in this repository, and the one
claim that could not be verified was removed rather than shipped (Pass-2 finding 4).

## Pass 1 — implement and verify

| Step | Command | Result |
| --- | --- | --- |
| Merge upstream lanes | `git merge origin/main` | 22 conflicts, all resolved additively; both lanes' candidates, receipts and pages kept |
| Refactor check on the shared screen | `scripts/uniqueness_fast.py <h59i candidate> --out /tmp/uq_h59i_recheck.json` | **reproduces the stored H59-I receipt exactly**: 71,119 dots, worst ρ 0.175818, worst exact Jaccard 0.042365, 85 literal rows, 54 dense + 641 dot peers, exit 0 |
| Re-scan this lane's candidate | `scripts/uniqueness_fast.py downloads/gems59-h59j-offcat-propen-30k-20261011.tif --out evidence/uniqueness_h59j_offcat.json` | **reproduces the stored H59-J receipt exactly**: 30,000 dots, worst \|ρ\| 0.063579, worst exact Jaccard 0.020913, 80 literal firings, exit 0 (695 rasters, 0 errors, 424 s) |
| Reconcile receipts | `scripts/finalize_h59j_release.py` | one representation-aware verdict; 6-bullet promotion decision; run card carries both lanes |
| Rebuild the manifest | `scripts/build_download_manifest.py` | 4 candidates, hashes recomputed from the bytes in `docs/downloads/` |
| Delivery QA | `scripts/check_site.py` | **PASS** — 13 pages, 4 candidates, current candidate 695 rasters, all badges receipt-driven |

The refactor check matters: the CLI change to `uniqueness_fast.py` (positional candidate + `--out`) is
behaviour-preserving *by measurement*, not by assumption — the old receipts and the new runs agree on every
statistic the two versions share.

## Pass 2 — bug review (findings and fixes)

1. **`finalize_h59j_release.py` was not idempotent.** It captured the "H59-I history" block from whatever the
   card held at run time, so a *second* run would have written the H59-J raster into `lane_h59i`. Fixed with a
   one-time guard (`if "lane_h59i" not in card`). Re-running the script now leaves the history block stable.
2. **My first fix to the representation-aware rule produced a false duplicate.** The rule classified a peer as
   "informative" for the forward leg using the random-placement chance level `1-(1-d)**29`; for the regular
   206,895-dot lattice (`13GEMSDOE r13-lattice-s5_v2`, density 3.0 %) that gives 0.586 (< 0.70) and the leg
   then "fired" at 0.9991, contradicting the IR-57-REPR-01 disclosure. Fixed by (a) using the expected 3 px
   disc coverage `min(1, 29·density)` — 0.87 for that lattice, correctly non-operative — and (b) making the
   **mutual** coincidence (`min(forward, reverse)`) the operative directional test. Result on the shipped file:
   worst mutual 0.4086 (limit 0.70), **0 mutual firings**, 80 literal firings all in the two documented
   uninformative directions.
3. **`check_site.py` assumed one receipt and one schema.** After the merge there are four published candidates
   and three generations of receipt (this lane's `representation_aware` block; the parallel lane's
   mutual-overlap summary; the replication lane's nested `summary` with `candidate_dots` /
   `registry_rasters_compared`). Rewritten to read each candidate's `uniqueness_receipt` from the manifest,
   accept all three schemas, and keep the delivery test's scalar readings
   (`registry_rasters_checked`, `worst_spearman`, `worst_jaccard`, `worst_reverse_overlap`,
   `worst_support_matched_overlap`) pinned to the current candidate.
4. **An unverifiable number was about to ship.** A draft of `REMAINING_WORK.md` and an irregularities row
   quoted "0.2377" as the build-time Jaccard. Verification: the build script computes a *near-set* Jaccard
   (`inter(near)/(mine+peer−inter(near))`, `scripts/build_submission_h59j.py` ~line 187), but its value was
   overwritten by the canonical rescan before the release commit and survives in **no committed receipt**
   (`grep -r 0.2377` → only the draft text). The number was deleted from every file; the entry is now
   IR-H59J-12, which records the *definition* mismatch and the exact statistic that is quotable (0.0209).
   The receipt carries `uniqueness.build_inline_screen.worst_jaccard_near_set = null` plus that note.
5. **Manifest builder path bug.** The replication lane's writer receipt lives under `docs/downloads/`, not
   `downloads/`; the new four-candidate builder would have raised `FileNotFoundError`. Fixed, and each row now
   declares its `uniqueness_receipt`, `run_card` and `lane`.
6. **Metric algebra cross-check (positive finding).** The lane's `credit_stats()` uses
   `DTI = T / (0.2·S + 0.8·K)` with `S = T + n − M`. The repository's shared metric
   (`src/gems59/metric.py → dti_from_components`) computes `T + α(n−M) + β(K−T)`, which is algebraically the
   same expression at α = 0.2, β = 0.8. No drift between the lane's screen and the shared metric.
7. **Two conflicting `check_site.py` revisions.** The parallel lane had kept a Session-7-era script that
   asserted on `evidence/run_card_current.json` (not produced by any H59-era session, so it crashed). Took the
   fail-closed, manifest-driven revision and kept the `inspect_site()` compatibility helper that
   `tests/test_site.py` imports.
8. **Site consistency.** Unified navigation across the four primary pages; added the "Other candidates on this
   site" panel so every published file has an explicit DOWNLOAD / SUBMIT badge read from its receipt;
   corrected the leaderboard context to revision 2 (`#1 0.4133`, `#23 0.2778` extradr19, `#50 0.2448` this team)
   and kept revision 1 on the record with its timestamp.
9. **Disclosed, not hidden: IR-H59J-10.** The parallel lane's front page says "Submit to competition: OK" on the
   strength of a PROJECTION, while that lane's own writer receipt records `approved_for_weekly_slot=False` and
   `promoted=False`. The merged site prints both readings and sets the badge from the receipts (NO). This is a
   convention disagreement for the owner to settle, not something the merge is allowed to paper over.

## Pass 3 — re-check against the standing brief, line by line

| Brief requirement | Where it is satisfied | Check |
| --- | --- | --- |
| Generate a **unique** TIF (never copy a prior submission) | `downloads/gems59-h59j-offcat-propen-30k-20261011.tif`, sha256 `6435ecec1391a0eea3648ea1d802af4ecb22715d5dfbf566724f90d3da4839c6`; 0 identical decoded predictions in 695 rasters; worst \|ρ\| 0.0636; worst exact Jaccard 0.0209 | receipt `evidence/uniqueness_h59j_offcat.json`, re-scanned in Pass 1 |
| Make it **obvious whether it is OK to download and submit** | every candidate carries `download_status` / `submit_status` in `docs/downloads/manifest.json`; the front page shows badges + the 6-bullet promotion decision and `what_would_clear_it` | `check_site.py` PASS; badges asserted against receipts |
| **Easy-to-download** submission TIF | three buttons (`GeoTIFF`, single-TIFF `ZIP`, writer receipt) at the top of `docs/index.html` and in the executive summary; ZIP contains exactly the one GeoTIFF | `check_site.py` (ZIP round-trip, byte counts, hashes) |
| Executive summary explains **exactly how to submit** (incl. the "[0, 1]" form error) | `docs/executive-summary.html` — step-by-step, the paste-ready note (122/140 chars), the name suggestion, and the explanation of the portal's range error | page present, `check_site.py` link pass |
| Unique **name and short comment** | name `gems59-h59j-offcat-propen-30k-20261011`, note 122 chars, both in the receipt and on the site | `evidence/submission_h59j.json` |
| Parallel-run protocol rules 1–6 | 1 lane discipline (rank correlation + both overlap directions, per row); 2 reuse of the template's writer/validator/holdout modules; 3 every number labelled HOLDOUT-DTI / PROXY-MEASUREMENT / OWNER-REPORTED / PROJECTION / VERIFIED-FILE; 4 leakage canary (max single channel 0.7347 < 0.90); 5 **one** run card (`evidence/run_card.json` = `docs/data/run_card.json`, covering both lanes); 6 three experiments / budget respected, **no slot consumed** | receipts + card |
| Deep research stored in the repo | `docs/research/` (leaderboard analysis, hypotheses, irregularities, sources, utility derivation, why-02778, plus both lanes' reviews) | files present, linked from the site |
| 3–5 candidate hypotheses, ranked, before implementing | `docs/research/hypotheses.md`; this lane's off-catalogue instrument was the pre-registered H59-I4 follow-up | hypothesis section in the run card |
| No hallucinations; flag irregularities | 13 H59-J-era entries + 4 H59-I-era entries in `docs/research/irregularities.md`; Pass-2 finding 4 removed an unverifiable number | this document |
| Create a PR and merge it to `main` | PR opened from `arena/fba37db3-59gemsdoe` after this review, merged, then the Pages deploy verified | PR body + Pages run |

## Residual risks (unchanged by this merge)

* No organizer score exists for any published file: the off-catalogue proxy is a within-family screen only
  (ρ −0.057, p 0.66 across 62 owner-reported anchors), so neither candidate is promoted.
* The registry scope is the 2026-10-10T20:01Z snapshot (695 rasters); later publications are a disclosed gap.
* Two lanes disagree about the submit badge convention (IR-H59J-10); the merged site follows the receipts.
* Geometry (chain vs scatter) remains an unvalidated hypothesis about the owner-reported spread
  (IR-H59J-08) and is pre-registered as H59-K, not used to steer this release.

## Second merge (same day): `arena/1ba7bc3e` (H59-A intensity lane)

While this branch was being finished, `main` received a third lane (PR #11, commit `410a4d0`, candidate
`gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.tif`, 38,593 dots, own holdout 0.08811 [0.08715, 0.08907]).
It was merged here the same way: every file kept, no lane's evidence overwritten.

* Added to `docs/downloads/manifest.json` as the fifth candidate; its row carries `lane`,
  `uniqueness_receipt`, `run_card = docs/h59a.html` and `shared_template_gate_passed: false`.
* `scripts/check_site.py` gained a fourth receipt schema: H59-A declares its registry leg inside the writer
  receipt (`uniqueness_vs_696_registry` = max_spearman 0.6718 / max_lift 1.999 / verdict NO DRIFT, plus
  `uniqueness_vs_main_releases` NO DRIFT). The gate now checks the declared template-gate claim against reality
  for **every** candidate and requires any flag to be disclosed on the front page.
* Finding: that candidate's own validator passes 12/12, but the shared template gate rejects its `nodata` tag
  (IR-H59J-15). Disclosed on the front page and in the manifest — not silently passed, not silently failed.
* The front page now lists five candidates with per-file badges, and explains that download-safety and slot
  promotion are different questions.
