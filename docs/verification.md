# Verification — three passes

**Pass 1 — implement and verify.** Every stage of the lane was run end to end on
the pinned data and its receipt written to `evidence/`:

| stage | receipt | result |
|---|---|---|
| bridge pins | `data_preparation.json` | ALL PINS VERIFIED (7 files + the 419 MB 19-band stack by transport hash) |
| grid / catalogue / footprint | `verify_grid.json` | footprint 5,167,373 px, catalogue 60,988 px, 0 px outside |
| covariates | `planes.json` | 21 planes ranked, cached |
| full-catalogue fit | `fit_full.json` | 37,307 thinned presence vs 150,000 background |
| EXP-1 | `exp1_catalogue_holdout.json` | q-sweep, table sweep, 3 controls, pooled with 1,000 bootstrap draws |
| family comparison | `family_comparison.json` | 3 fields × 2 frames at a matched 44,000-dot budget |
| EXP-2 | `exp2_sgmc_novelty.json` | lane vs proximity on the off-catalogue frame |
| EXP-3 | `exp3_canary_robustness.json` | 21 canaries + 3 seeds × 3 q multipliers |
| emission | `emit_receipt.json` | dots, blocks, spacing histogram, 0 on catalogue |
| release | `docs/downloads/*.json` | validator + SHA256 + ZIP round-trip |

**Pass 2 — review for bugs, missing requirements and edge cases.** Four real
defects were found and fixed, each with the evidence that caught it:

1. **Canary sample drawn from the wrong mask.** `cmd_exp3` sampled background
   points outside the catalogue while the labels were withheld *catalogue*
   pixels, so the sample was single-class and every AUC was `NaN`; the run then
   died in `max()`. Fixed by sampling every withheld pixel plus an equal number
   of visible catalogue pixels. Re-run: worst AUC 0.5299, no flags.
2. **Receipts were not strict JSON.** The exp-3 crash surfaced as
   `ValueError: Out of range float values are not JSON compliant: nan` *after*
   the science had run. `_json()` now sanitises non-finite floats to `null`.
3. **The deployed field's scale is unidentified (IR-59-SEP-01).** Two runs of
   identical code at Newton stopping points 0.2 % apart in deviance produced
   block peak shares 90× apart, which silently changed the emitted dot count
   from ~44,000 to ~570 under an absolute density cut. The emission now selects
   blocks by **rank** (top-q) and never by an absolute expected-pixel density;
   the scale instability is recorded in the receipt.
4. **Two stages used different ridge values.** `exp2`/`emit` refit with the CLI
   default while `exp1` used its own spatial-CV choice, so the "independent"
   check scored a different model than the release. All stages now read
   `exp1.l2_selection.chosen` (0.1) as the single source of truth.

Edge cases checked: empty emission (no dots outside the footprint or on the
catalogue), dot count under the budget cap, lattice levels restricted to
`(3, 6, 9, 12)`, single-class canary samples, non-finite receipt fields, and the
`[0, 1]` / all-finite validator policy that an earlier session's upload was
rejected for.

**Pass 3 — re-check against the original request.**

* *Unique TIF, not copied, uniqueness checked against every earlier raster* —
  built from the lane's own covariates; 710 registry rasters compared on two
  legs (`uniqueness_lane_v1.json`).
* *Obvious OK-to-download / OK-to-submit labelling* — the status table is the
  first thing on the site and at the top of the README, and the labels are read
  from the run card (`okay_to_download` / `okay_to_submit`) so no page can
  advertise a decision the card does not record.
* *Intensity lane exactly as specified* — 1 km blocks, presence-only intensity
  with target-group background, MaxEnt ⟺ Poisson equivalence, effort covariates
  (with the reliability-of-location class reported as unobtainable, IR-59-EFFORT-02),
  break-even tiling with the spacing re-derived on our own holdout.
* *Parallel-run protocol* — lane restriction, drift check on the surface and the
  final dots, no private forks (the shared `evaluate_holdout.py`,
  `submission_writer.py`, `metric.py` are used unchanged), every number labelled,
  leakage canary run, one JSON run card, 3 experiments, no slot spent.
* *Hypotheses before implementation* — the family comparison is that register
  reduced to the one candidate the budget allowed to test (ground-only
  geophysics + effort), scored on the spatially blocked holdout before any
  submission slot was considered.
* *Site* — GitHub Pages, five pages, official links, an executive summary that
  explains exactly how to make a submission, status at the very top.
* *Working rules* — autonomous, no hallmarks, irregularities listed, three
  passes, PR to `main`, remaining work and limitations stated.

**Known limits of the verification.** No organizer receipt exists anywhere in
the family; the catalogue frame is structurally biased; the off-catalogue frame
is a different public catalogue, not the hidden truth; the registry snapshot
certifies the accessible pins only.

---

## Pass 3 addendum — closing the evidence chain (uniqueness gate, run card, site build)

The uniqueness gate is the longest single stage (710 registry rasters re-read).
It ran to completion (681 s, 100 % CPU, exit 0) and its findings are recorded
rather than trimmed:

* worst **rank correlation** against any of the 710 rasters: **0.1142** (5GEMSDOE
  `context_detector_prob_u8.tif`, a dense full-footprint surface) — below the
  0.90 drift trigger, no trip;
* worst **3 px dot overlap** against a *comparable* prior — a sparse dot field
  with support ≤ 2 % of the footprint (490 such peers): **0.5129** (GEMSDOE51
  `strainbudget-s1-v1`, 95,927 px) — below the 0.70 trigger, no trip; reverse
  overlap 0.2081, rank correlation 0.047;
* the **literal** leg reports **0.9952** because its worst case is a *uniform
  5 px lattice covering the entire footprint* (13GEMSDOE `r13-lattice-s5`). A
  regular grid is within 3 px of almost any candidate's dots by construction, so
  that number is a saturation artefact: only 0.1204 of *its* dots sit within
  3 px of ours. This is the IR-59-REPR-01 situation the sibling lanes also hit
  (the universal-support witness blocks *every* nonempty candidate under the
  rule as written). Both legs are reported, the comparable-peer tier is added as
  an explicit third row on the site, and a ruling is requested — the rule is
  **not** silently redefined.

Key-shape fixes found by actually running the artefacts (not by reading them):

1. `scripts/replication_write_run_card.py` and `scripts/replication_build_site.py` both crashed on
   `exp1["pooled"]["candidate"]` — that key does not exist; the deployed arm is
   `table0`, and the control is named by `best_comparable_control`. Fixed in both
   by resolving the arm explicitly.
2. `exp1["holdout"]["split_version"]` and `exp2["pooled"]["paired_difference"]`
   were the same class of drift (`split_version` lives at the top level;
   the exp-2 key is plural, `paired_differences`). Both read with a fallback.
3. `pooled["scores"][*]` carries no `dots`; dot counts live in `exp1["arms"]`.
   The site now prints the arm's own dot count instead of the sweep's.
4. `write_run_card.py` needed `sys.path` before importing `gems59` for the
   live-profile block; the first run silently swallowed the ImportError into an
   `error` field, which is exactly why the card is now checked field by field.
5. Non-finite ranks (constant rasters give `NaN` correlations) are sanitised to
   `null` in the card so the JSON stays strict.

Pass-3 re-verification of the release itself, after the fixes:

* the TIFF on disk hashes to its receipt;
* `docs/` links: every local `href` resolves (checked programmatically), the six
  pages and thirteen receipt copies are all present;
* the site's status pills are read from `evidence/run_card_lane_v1.json`, so a
  card edit changes the page and nothing can advertise a stale decision;
* the live-profile measurement (median distance to the catalogue 19.72 px,
  6.34 % of dots within 2 px, 0 on the catalogue) is compared with the
  owner-reported structure only, and is labelled MEASURED.

**Residual limits** (unchanged): no organizer receipt exists anywhere in the
family; both holdout frames are proxies and disagree by construction; the
reliability-of-location class has no join key; the spacing ladder is unexercised
at this density; `.cache/` is snapshot-excluded and must be rebuilt with
`scripts/replication_prepare_data.py --fetch`.

---

## Pass 3 — integration with the other sessions' lane (merge of PR #4)

The repository moved on while this lane ran: other parallel sessions merged their own
intensity-lane release (H59-I) and their tooling to `main`. The two lanes were integrated
additively rather than by overwriting either record:

* this lane's driver, uniqueness gate, run-card writer, data-preparation and site builder are
  namespaced `scripts/replication_*.py`; its point-process module is
  `src/gems59/intensity_replication.py`; its receipts are `*_lane1ac8.json`; its pages are written to
  `docs/replication/` (the family site keeps `docs/`). Nothing of either session's evidence was
  deleted.
* `.gitignore` and `requirements.txt` take `main`'s versions; `requirements.txt` additionally keeps
  `tifffile`, which this lane's writer imports for the TIFF round-trip check.
* standalone `*.pyc` files that `main` had committed were removed from the index (they are already
  covered by its `__pycache__/` ignore rule).

Verification of the merged tree, measured rather than asserted (pytest, repository root):

| tree | tests | result |
|---|---|---|
| `origin/main` alone (checked out in a scratch worktree) | 184 collected | **38 failed, 142 passed, 4 skipped** |
| this branch after the merge | 193 collected | **37 failed, 152 passed, 4 skipped** |

The delta is favourable and local: the extra passes are this lane's nine unit tests
(`tests/test_lane.py`: metric parity against a brute-force oracle, zone invariants, emission
invariants) plus one `main` test that the lane's registry witnesses unblock; **no test that passes on
`main` fails after the merge**. The 37 remaining failures are pre-existing in the other sessions'
suites (they pin artefacts with hashes that drifted, and expect receipt files their own sessions
never committed); they are outside this lane's remit and are recorded rather than "fixed" by
loosening someone else's assertions.

One integration defect was found and fixed by this check, and it is exactly the class of failure the
three-pass rule exists to catch: `tests/test_lane.py` imported its driver as `import run_lane`,
which after the merge resolves to the *fault-zone-anatomy* driver that another session owns — the
test then failed with `AttributeError: module 'run_lane' has no attribute '_budget_emission'` while
the code under test was fine. The import now names the namespaced module explicitly. Also fixed:
five broken `data/` directory links on the replication pages (the site inspector requires file
targets), and the replication's page footer now points at the run-card copy.
