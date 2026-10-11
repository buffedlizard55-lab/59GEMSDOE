# Remaining work — Session H59-J / arena/fba37db3 (2026-10-11)

**Both candidate files are cleared for download and format-valid; neither is cleared for a competition
submission slot.** The current candidate is **H59-J** `downloads/gems59-h59j-offcat-propen-30k-20261011.tif`
(30,000 dots, 0 on the masked catalogue, SHA256 `6435ecec…39c6`); **H59-I**
`downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif` stays on the site as a superseded
(same verdict class) predecessor. **No competition slot was spent by this project.**

Site: https://buffedlizard55-lab.github.io/59GEMSDOE/ · Run card: [`evidence/run_card.json`](evidence/run_card.json) ·
H59-J receipts: [`evidence/submission_h59j.json`](evidence/submission_h59j.json),
[`evidence/uniqueness_h59j_offcat.json`](evidence/uniqueness_h59j_offcat.json),
[`evidence/offcat_lane_h59j.json`](evidence/offcat_lane_h59j.json),
[`evidence/emission_design_h59j.json`](evidence/emission_design_h59j.json),
[`evidence/h59j_emission_measurement.json`](evidence/h59j_emission_measurement.json)

## 1. What H59-J answered

* **The instrument H59-I4 pre-registered now exists.** A six-fold spatially-blocked out-of-fold discriminant
  trained on *off-catalogue* map faults (positives = SGMC fault pixels > 300 m from the catalogue; negatives =
  catalogue-free background) separates off-catalogue faults from background out of fold: **AUC 0.7444**
  (folds 0.638–0.774). Leakage canary: strongest single feature AUC 0.7347 < 0.90 → no flag. Evidence:
  `evidence/offcat_lane_h59j.json`; field: `.cache/oof_propensity.npy` (rebuilt by
  `scripts/run_offcatalogue_lane.py`, ~80 s).
* **It does not resolve the holdout-vs-live tension in the promotable direction.** The proxy criterion
  (`T_proxy_far / (0.2·S_proxy_far + 0.8·G_calibrated)`, `G_calibrated = 12,700`) reproduces the *within-family*
  live ladder of the owner's own thinnings (0.2778 > 0.2600 > 0.2477 > 0.1922), but **does not rank live scores
  across families**: on the 62-row owner-reported ledger the score-vs-proxy-DTI correlation is ρ −0.057
  (p = 0.66), the only significant term is `credit_per_dot_proxy_far` ρ +0.252 (p = 0.048), and the two
  cross-family hybrids that clear the same criterion (sgmc-off-cat-44k criterion ≈ 1.52, h18-4 ≈ 2.53) scored
  **0.0512 and 0.0360 live** against the 0.2778 incumbent. Evidence: `evidence/ledger_forensics_h59j.json`,
  `evidence/emission_design_h59j.json` (reference arms).
* **Emission design, pre-declared.** The screen preferred 3 px spacing at 44,090 dots (criterion 0.4606); the
  build used 3 px at 30,000 dots (measured criterion 0.3759, T 6,350.0 / S 33,668.5 / credit-per-dot 0.2117 in
  `evidence/h59j_emission_measurement.json`) because the owner-reported live ladder is monotone **down** in dot
  budget, so the conservative arm was shipped. The design's own preferred arm remains unbuilt.
* **Registry uniqueness: no duplication identified** — 695 pinned rasters, 0 errors, 0 identical decoded
  predictions, worst |Spearman ρ| 0.0636 (peer <code>GEMSDOE4 proxy_only46</code>), worst **exact** dot-set
  Jaccard 0.0209 (<code>5GEMSDOE pindrop-v4-ridge</code>). Two different statistics were called “Jaccard” in
  this lane's tooling (IR-H59J-12): the canonical screen intersects exact dot pixels (0.0209), while the
  build's inline screen intersected 3 px neighbourhoods. The build-time value is not preserved in any committed
  receipt and is therefore <strong>not quoted anywhere</strong>; only the exact figure is used as the duplication
  statistic. The literal 3-px overlap leg fires
  against dense lattice/continuous peers and is reported per row, not applied (IR-57-REPR-01). The build's
  literal-protocol verdict string vs the submission receipt's "OK" was reconciled by
  `scripts/finalize_h59j_release.py`; both receipts now carry one representation-aware verdict.
* **Promotion decision written down**: *negative — not cleared for a weekly slot*, with the basis and the
  conditions that would clear it, in `evidence/submission_h59j.json → promotion_decision`.

## 2. What remains open, ranked by expected value

1. **A cross-family instrument (the single blocker for the whole project).** Every local instrument validated so
   far ranks artifacts *within* one family only. Concrete next steps: (a) de-duplicate the 62-row ledger into
   independent edit families and re-fit the two-parameter model (`Ĝ`, `κ`) on family means with a widened `Ĝ`
   grid — the current fit pins `Ĝ` at the grid floor 2,000 (κ 0.107, LOO MAE 0.096) and cannot yet be quoted as
   calibrated; (b) for each family in the ledger compute the *same* proxy criterion at that family's own budget
   and test rank agreement on family means with a permutation test (n ≈ 15 families — report as exploratory,
   never as a score); (c) if the owner can spend one probe slot, the cheapest decisive experiment is an
   organizer-verified submission of one off-catalogue emission, which calibrates the whole family.
2. **Ledger hygiene before any further inference** — three artifacts are near-identical thinnings of
   `h33-2-b2` (0.2778 / 0.2750 / 0.2710 and the 0.2708/0.2707 pair), so n = 62 overstates independent evidence.
   De-dup is a prerequisite for quoting any p-value.
3. **H59-K — chain-geometry emission (new, found this session).** At equal dot budget the owner-reported
   winners are unpacked only by *geometry*, not by features: the top group (12 artifacts ≥ 0.24, e.g.
   h33-2-b2 0.2778, anderson-pinn 0.2750, h36-1 0.2710) has nearest-neighbour median **3.08 px** and a local
   chain elongation median **0.199** (p90 0.219 of the smaller/larger eigenvalue ratio inside 6 px), while the
   bottom group (28 artifacts < 0.10) sits at **0.289** with the same spacing. The same-budget failures
   (r12-scarp 0.0843, h60-lidarscarp 0.0430, h34-scatter 0.0778, h51-faultzone 0.0106) all sit at 0.23–0.33.
   The shipped H59-J emission is at **0.343**, i.e. in the losers' geometry class — a disclosed negative signal
   about this session's own candidate. Testable next step: emit by tracing the propensity field's ridges
   (chains at ~3 px along the ridge, stop when the field drops) instead of greedy space-filling NMS, then test
   the geometry rule out of sample (it is post-hoc on 62 owner-reported rows and confounded with family
   identity; one clear counterexample at 335,879 dots exists). Receipt and caveats:
   `evidence/emission_geometry_h59j.json`.
4. **Own-holdout zone/placement rule (the prompt's lane requirement, still unrun).** H59-I2 (block-level
   completeness-ratio zoning `n_b = Σλ·(1 − cat_frac_b/λ̄)`) and H59-I2b (budget-capped lattice) remain
   pre-registered and unimplemented; the H59-J emission sidesteps the zoning question rather than answering it.
5. **Three-pass review of the H59-J code.** Pass 1 (implement + verify) is complete: build receipts, validators
   and the registry scan all reproduce. A formal Pass 2 (bug review of `run_offcatalogue_lane.py`,
   `design_emission_h59j.py`, `build_submission_h59j.py`, `finalize_h59j_release.py`) and Pass 3 (re-check
   against the standing brief line by line) are the next session's first cheap task.
6. **Organizer-side unknowns that no local work can close** (see `docs/research/irregularities.md`): whether the
   catalogue mask applies to the private/Final-Round label set (IR-1/IR-14), the accepted out-of-footprint
   convention (IR-9), the metric epsilon (IR-10) and the weekly cap (IR-12).

## 3. H59-I history (kept for audit, superseded)

H59-I built the intensity lane end to end: presence-only MaxEnt/Poisson intensity (Phillips et al. 2009 effort
covariates; Fithian & Hastie 2013 equivalence), 1 km block zoning by predicted new-fault density, and a
holdout-derived break-even lattice. HOLDOUT-DTI **0.08216 [0.06954, 0.09616]** (evaluator
`gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 draw) is below the published holdout best
**0.135204** → negative. Settled inside it: the 2 px catalogue collar is worth Δ 0.00744 [0.00522, 0.00982];
the own-holdout utility table picked **4 px**, not the prompt's 3 px; zoning ranks withheld segments (68.5 % in
the top three deciles vs 30 % under uniformity); no leakage (max single-covariate AUC 0.776). Evidence:
`evidence/intensity_exp{1,2,3}_*.json`, `evidence/uniqueness_h59i.json`.

## 4. Statistical and geological limitations

* Every proxy truth used here is a *map compilation* (SGMC, QFault) — a different, ~6× more prevalent
  population than the hidden expert-verified new faults, and it mixes non-fault lineaments (unit contacts,
  thrust sheets, mining-district structures). Proxy numbers are never competition scores.
* The off-catalogue discriminant is fit on the same 38 rank channels used elsewhere in the repo; its OOF AUC is
  modest (0.744) and the emission step reduces a continuous field to binary dots, which throws away the
  ranking's uncertainty.
* Fault networks violate the independence assumptions of the discriminant (they cluster, link and relay); no
  spatial random effect is fitted.
* The two registries disagree about off-catalogue prevalence by ~1 % (78,433 vs 79,615 px), and the calibrated
  population `G` (12,700) is an inversion of owner-reported anchors with an acknowledged ±10 % band; the
  criterion's ordering is robust to that band, its absolute value is not.
* The registry scope is the 2026-10-10T20:01Z snapshot of 695 rasters; publications after that snapshot are a
  disclosed gap in the uniqueness claim.

## 4b. Merge bookkeeping (2026-10-11, PR #9) — three lanes, one tree

* **Four candidates are published, none promoted**: this lane's off-catalogue discriminant (30,000 dots,
  current), the H59-I intensity lattice (71,119), the parallel lane's ridge-coherence consensus (30,815,
  `arena/1ee0553d`) and the replication lane's intensity run (50,651, `arena/1ac83927`). Every row of
  `docs/downloads/manifest.json` carries `download_status` / `submit_status` / `uniqueness_receipt` / `lane`.
* **Name collisions were resolved by renaming this lane's receipts** (`*_offcat`): three sessions published
  different files called “H59-J” and two of them wrote `evidence/uniqueness_h59j.json` for different candidates
  (IR-H59J-11). No lane's evidence was overwritten.
* **One run card covers both lanes** (`evidence/run_card.json` = `docs/data/run_card.json`): H59-I history under
  `lane_h59i`, this lane's release fields, and a `parallel_lane` block that quotes the other lane's numbers,
  its `approved_for_weekly_slot=False` receipt status, and the status disagreement (IR-H59J-10).
* **Status disagreement is disclosed, not smoothed**: the parallel lane's page presents its candidate as
  submit-OK on a PROJECTION; the merged site prints both readings and the badges follow the receipts (SUBMIT NO
  for every file). The owner's selector step remains the only promoter.
* **Delivery QA now covers all four**: `scripts/check_site.py` re-checks each candidate's uniqueness receipt
  (three schemas), hashes, formats, ZIP twins and the badge consistency — PASS (13 pages, 4 candidates).
* **Review artefact**: [`docs/research/three-pass-review-h59j-merge.md`](docs/research/three-pass-review-h59j-merge.md)
  — Pass 1 (merge + reproduction checks), Pass 2 (nine findings incl. an idempotency bug, a false-duplicate rule
  and one unverifiable number removed before shipping), Pass 3 (brief re-check line by line).

## 5. Standing status

* **DOWNLOAD OK** for both candidates, **SUBMIT NO** for both. Promotion to a weekly slot is a separate selector
  decision that belongs to the owner; this project holds no DrivenData credentials and has spent no slot.
* Every number is labelled: HOLDOUT-DTI (evaluator version, withheld positives, 95 % CI), PROXY-MEASUREMENT
  (SGMC-derived truth) or OWNER-REPORTED / ORGANIZER-PUBLISHED context. No local number is presented as a
  competition score.
