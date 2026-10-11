# 59GEMSDOE — intensity lane

**DrivenData “DOE GEMS” competition #306.** One parallel session of a family working the
same standing brief: *zone new-fault density in 1 km blocks, fit a presence-only
point-process intensity with mapping-effort (target-group) background, and emit the
blocks above break-even as a sparse dot raster* — with every number labelled by its
evidence class and nothing submitted.

---

## STATUS — read this first

| | |
|---|---|
| **Download** | **OK.** One click: [`docs/downloads/`](docs/downloads/) — TIFF, one-TIFF ZIP, JSON receipt. |
| **Submit** | **HOLD.** No competition slot was used by this repository. Reason: the release is a *bet on the off-catalogue reading*, and its own paired test on that frame is not significant (see below). Promotion to a slot is the owner's separate selector step. |
| **File** | `59gemsdoe-intensity-lane-50651dots-s3-t33333-20261011T005447Z-b25dd620-zeros.tif` |
| **SHA256** | `d8e83e1eeae9f8ea82fba4447131e876d3cc601b283df84aaa88303bd7a4f15c` · 79,553 bytes |
| **Form** | 50,651 binary dots · float32 · **all 12,277,160 grid pixels finite, min 0.0, max 1.0, 0 NaN, 0 Inf** · EPSG:32611 · 3730×3292 · 100 m · transform matches the sample submission · **0 positive pixels on the mapped catalogue** · 0 outside the footprint · all 15 local validator checks pass |
| **Why HOLD** | Off-catalogue frame (the arbiter): **0.04399 [0.03231, 0.05601]** vs proximity control 0.03914 [0.03301, 0.04594] — paired difference **+0.00485 [−0.00814, +0.01893]**, i.e. *not* significant. Catalogue frame: 0.03339 vs proximity 0.23490 — the release *loses* there, because that frame mechanically rewards dots beside the visible catalogue and the live evidence penalises exactly that. A matched-budget family comparison did favour a ground-only field on the arbiter frame (0.05708 vs 0.03548, +61 %) but at a different emission configuration, so it is not a claim about this raster. |
| **What would flip it** | Regenerate at the matched-budget configuration and re-score the arbiter frame with a lower bound above zero; or the owner's selector step choosing this file against the family's retained candidates. |

The Pages site (`docs/index.html`) shows the same table at the very top. The
permission labels there are read from `evidence/run_card_lane_v1.json`
(`okay_to_download` / `okay_to_submit`), so no page can advertise a decision the
card does not record.

---

## The mechanism, in one paragraph

Zone by **new-fault density per 1 km block**; fit a **presence-only point-process
intensity** to the mapped catalogue with the observer process absorbed by
**target-group background** (Phillips et al. 2009), which is exact because
**MaxEnt ⟺ inhomogeneous Poisson process** (Fithian & Hastie 2013). Presence
points are thinned to 2 px separation, 150,000 background points are drawn from
the study-area footprint, and a ridge logistic (spatial-CV λ = 0.1) is fitted on
21 covariates: potential-field texture (analytic signal at 1.5/3/6 km, tilt,
theta, total horizontal derivative of gravity at 3/8 km), independent layers
(geodetic dilatation/second invariant, earthquake density), conductivity,
depth-to-basement, detrended elevation derivatives, local relief, slip-rate
maximum, acquisition block and distance-to-catalogue/tips/slip-rated traces.
**Deployment drops every covariate that is a function of the catalogue's own
geometry** (IR-30-039: those re-predict the mapped traces, which the competition
masks out and the live evidence penalises) and zeroes the intercept, so the
released field is ground evidence + observer effort only. The field is converted
to block shares, blocks are ranked by intensity, the top *q* are tiled with a
3 px lattice (spacings are exact subsets, phase 1), and the mapped catalogue and
everything outside the footprint are removed. Emission never uses an absolute
expected-pixel cut: **IR-59-SEP-01** measured that the deployed field's *scale* is
not identified (two Newton stopping points 0.2 % apart in deviance give block
peak shares 90× apart), so only the block **rank** is trustworthy and the design
rule is stated in rank terms.

## The standing brief (preserved, and read every session)

> **Goal.** Build this project into a competition-ready artifact for DrivenData
> “DOE GEMS” #306, aiming past the 0.3195 leaderboard leader (best owned file
> 0.2778, `h33-h33-2-b2-…-zeros`; the 0.3774 quoted on a GEMSDOE site is a site
> record, not the board leader).
>
> **Must.** Generate a **unique TIF submission**; never copy a previous
> submission except to learn; check uniqueness against **every** earlier raster;
> make it **obvious** whether the file is OK to download and to submit.
>
> **Lane (the intensity paragraph).** Zone by new-fault density per 1 km block
> (not per-pixel probability); fit presence-only point-process intensity from the
> catalogue with mapping-effort bias corrected by target-group background
> (Phillips et al., *Ecological Applications* 2009), noting MaxEnt ⟺ Poisson
> process (Fithian & Hastie, *Annals of Applied Statistics* 2013); effort
> covariates = USGS reliability-of-location class, distance to roads, relief,
> GeoDAWN acquisition block; 1 km block size exceeds the 400 m registration
> error (Hermant et al.); check predicted vs withheld-segment counts by decile on
> the holdout; tile only blocks above break-even with spacing re-derived on our
> own holdout (~3 px dense, wider as density falls).
>
> **Parallel-run protocol.** (1) Stay in lane; drift trigger = rank-correlation
> > 0.90 with any registry raster, or > 70 % of dots within 3 px of one registry
> raster → log duplicate and stop; check on the surface before placement **and**
> on the final dots. (2) Reuse the template's cached feature stack,
> `evaluate_holdout.py`, `submission_writer.py`; holdout = whole fault segments
> withheld with a buffer; catalogue features from visible faults only; visible
> faults masked pixel-exactly; pooled DTI α = 0.2, β = 0.8, 300 m triangular
> kernel; fix shared tools once, never a private fork. (3) Label every number
> HOLDOUT-DTI (evaluator version, # withheld positives, 95 % CI) or
> ORGANIZER-CONFIRMED; a projection is never a score. (4) Leakage canary: each
> feature alone on the holdout; AUC > 0.90 = leakage until proven otherwise.
> (5) End with one JSON run card: hypothesis; mechanism; named non-fault process
> that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster
> sha256; validator output; submission name + note ≤ 140 chars; verdict
> promote/negative. (6) **Budget: stop after 3 experiments or 2 hours**;
> promotion to a real slot is a separate selector step.
>
> **Before implementing.** Generate 3–5 candidate geological hypotheses not yet
> tried, each naming specific layers, the physical signature (edge/curvature
> transform), why it catches a fault missing from the USGS/INGENIOUS catalogue
> rather than one already in it, and how it differs from anything in the repo;
> rank by expected DTI gain and implementation cost; validate the top candidate
> on the spatially blocked holdout before touching a weekly submission slot; if a
> candidate needs external data, name the specific free official source and
> verify obtainability.
>
> **Study.** Explain at PhD level why
> `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` scored 0.2778 and whether it can
> be beaten; use that answer to generate the unique TIF.
>
> **Site.** GitHub Pages: clean, simple, all relevant info, official verified
> source links, an executive-summary subpage explaining exactly how to make a
> submission, and the download/status obvious at the very top.
>
> **Working rules.** Autonomously, no manual input; no hallucinations; verify
> line by line; flag irregularities; three passes (implement+verify, review for
> bugs/missing requirements/edge cases, re-check against the original request);
> then a PR merged to `main`; state remaining work and limitations. Keep Arena's
> core values — maximise P(win), own the outcome — as the focal point.

**IR-59-PROMPT-01.** The original chat text is not machine-recoverable inside
this sandbox, so the brief above preserves the *requirements* exactly and does
not claim word-for-word fidelity. The sibling repository's preserved prompt (the
same protocol skeleton in the user's own words) is copied to
[`docs/task-prompt-sibling.md`](docs/task-prompt-sibling.md).

## Why 0.2778, and what could beat it

All of the following is **OWNER-REPORTED / model-based**, never an organizer
receipt:

* The metric is `DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w)` with a 300 m
  triangular kernel and the mapped catalogue removed from every term
  (`src/gems59/metric.py`, verified against a brute-force oracle in
  `tests/test_lane.py`). Because β = 0.8 weights misses four times as heavily as
  false positives, **coverage** dominates — but only while a dot's realised
  credit clears `0.2·s/(1−0.2·s)` (0.0683 at s = 0.3195, ≈280 m).
* The best-owned file (0.2600) is a *thinned* subset of a denser parent (44,090
  of 121,131 px), and every owner-reported high scorer sits in the 38,854–44,090
  dot band with a large median distance to the catalogue. Thinning bought score
  by removing dots below break-even; the remaining gap to 0.3195 is realised
  credit per pixel, i.e. **placement**, not format.
* The catalogue hide-and-recover holdout and the live board **disagree on
  ordering** (IR-30-039), which this session measured directly at a matched
  budget: proximity 0.23426 / catalogue-family 0.23234 / ground-family 0.03165 on
  the catalogue frame, but 0.03548 / 0.03556 / **0.05708** on the off-catalogue
  frame. The scored target is faults *absent* from the catalogue, so the
  off-catalogue frame is the arbiter and the release follows it — at the cost of
  a catalogue-frame number that looks bad, reported rather than hidden.

## Evidence (all HOLDOUT-DTI unless labelled otherwise)

| frame | arm | dots | DTI | 95 % CI |
|---|---|---|---|---|
| catalogue hide-and-recover (`gems57-pooled-hide-v2`, 11,354 withheld px, 4 folds, 1,000 paired bootstrap draws) | **release design** | 50,422 | **0.03339** | [0.02282, 0.04447] |
| | proximity-only control | 48,916 | 0.23490 | [0.22011, 0.24839] |
| | uniform same count | 50,422 | 0.03468 | [0.03030, 0.03875] |
| | random blocks | 22,145 | 0.01678 | [0.01204, 0.02171] |
| off-catalogue (SGMC novelty > 300 m from the catalogue; 61,664 px) | **release** | 50,651 | **0.04399** | [0.03231, 0.05601] |
| | proximity-only | 48,916 | 0.03914 | [0.03301, 0.04594] |
| matched-budget family comparison (44,000 dots, both frames) | ground family | 44,002 | 0.03165 (cat) | 0.05708 (off-cat) |
| | catalogue family | 44,001 | 0.23234 (cat) | 0.03556 (off-cat) |
| | proximity | 44,004 | 0.23426 (cat) | 0.03548 (off-cat) |

**Unique against the registry (measured, not asserted).** All **710** earlier
rasters were re-read and compared with this release (`evidence/uniqueness_lane_v1.json`):

* worst **rank correlation** against any registry raster: **0.1142** (5GEMSDOE
  `context_detector_prob_u8.tif`, a dense full-footprint surface) against the
  0.90 drift trigger — no trip;
* worst **3 px dot overlap** against a *comparable* prior (a sparse dot field,
  support ≤ 2 % of the footprint; 490 such peers): **0.5129** (GEMSDOE51
  `strainbudget-s1-v1`, 95,927 px) against the 0.70 trigger — no trip; the
  reverse overlap is 0.2081 and the rank correlation 0.047;
* the *literal* leg reports 0.9952 because the worst case is a **uniform 5 px
  lattice covering the whole footprint** (13GEMSDOE `r13-lattice-s5`): a regular
  grid sits within 3 px of almost any candidate dot, so that number is a
  saturation artefact, not shared structure (only 0.1204 of *its* dots are within
  3 px of ours). Recorded as **IR-59-REPR-01** with a ruling requested; the
  comparable-peer tier is the meaningful test and it passes.

**Live-profile sanity check** (MEASURED, for comparison with the owner-reported
structure — not a score): the release carries 50,651 dots, **median distance to
the catalogue 19.72 px**, 6.34 % of dots within 2 px of the catalogue, 0 dots on
it. The owner's high scorers sit at ≈19 px median distance from the catalogue,
and the live evidence penalises dots within 2 px (−0.544) and near sub-parallel
placement (−0.810) — this release sits in that band by construction, because the
deployed family drops every catalogue-geometry term and masks the catalogue out.

Leakage canary: worst single-feature AUC **0.5299** (`neg_deq`) against a 0.90
limit, no flags. Robustness across seeds 59/101/202 × q × 0.6/1.0/1.6: catalogue
frame 0.0116–0.0391 (the release sits mid-band; the q multipliers move the dot
count, which that frame rewards monotonically).

Per-stage receipts: `evidence/exp1_catalogue_holdout.json`,
`evidence/exp2_sgmc_novelty.json`, `evidence/exp3_canary_robustness.json`,
`evidence/family_comparison.json`, `evidence/emit_receipt.json`,
`evidence/uniqueness_lane_v1.json`, `evidence/run_card_lane_v1.json`.
The three review passes are written up in [`docs/verification.md`](docs/verification.md).

## Reproduce

```bash
python3 scripts/replication_prepare_data.py --fetch   # hash-verify bridge pins + rebuild the 419 MB stack
python3 scripts/replication_run_lane.py verify        # grid / catalogue / footprint receipt
python3 scripts/replication_run_lane.py planes        # 21 covariates + ranks (cached)
python3 scripts/replication_run_lane.py fit           # full-catalogue fit
python3 scripts/replication_run_lane.py exp1          # EXP-1: q-sweep, table sweep, controls
python3 scripts/replication_run_lane.py family        # family comparison on both frames
python3 scripts/replication_run_lane.py exp2          # off-catalogue frame
python3 scripts/replication_run_lane.py exp3          # leakage canary + robustness
python3 scripts/replication_run_lane.py emit          # the dot field
python3 scripts/replication_run_lane.py ship          # validated TIFF + ZIP + receipt
python3 scripts/replication_check_uniqueness.py       # all registry rasters, both legs
python3 scripts/replication_write_run_card.py         # the one run card
python3 scripts/replication_build_site.py             # docs/ Pages site
python3 -m unittest discover -s tests     # metric parity + emission invariants
```

## Limitations and remaining work

1. **No organizer receipt exists anywhere in this family.** Everything is
   owner-reported or a holdout proxy. That is why the submit label is HOLD.
2. **The release's own advantage is not established.** On the arbiter frame the
   paired difference is +0.00485 [−0.00814, +0.01893]. The matched-budget
   configuration that *did* separate (ground family +61 % over proximity) should
   be the next release candidate, re-scored on the arbiter frame.
3. **Both frames are proxies and they contradict each other** (measured above).
   The choice of arbiter is a judgement, stated in `family_comparison.json`.
4. **Effort covariates are incomplete.** The USGS reliability-of-location class
   cannot be joined to the grid: the attribute table keys on `NUM`, the trace
   segments on `record_id`, and no join key exists (IR-59-EFFORT-02). Roads were
   not available in any clone. The acquisition block is >99.9 % covered inside
   the footprint and therefore weak there.
5. **The spacing ladder is not exercised at this density.** Every selected block
   lands in the finest stratum, so the release is a flat 3 px lattice; the
   "wider as density falls" arm exists and is measurably worse on both frames at
   this dot budget (reported, not hidden).
6. **The literal uniqueness rule is unsatisfiable** for any nonempty candidate
   while dense-support priors are in the registry (IR-59-REPR-01); both legs are
   measured and the ruling is requested.
7. **`.cache/` is not snapshotted** (419 MB stack, covariate/rank planes);
   `scripts/replication_prepare_data.py --fetch` rebuilds it from the pinned bridge parts.

## Official sources

* Competition: DrivenData #306 “DOE GEMS” — <https://www.drivendata.org/competitions/306/> (login-walled from the sandbox; the problem statement and metric facts here were transcribed from page 967 and cross-checked against sibling receipts).
* Official rules (NLR FY26 OSTI 96647): <https://docs.nlr.gov/docs/fy26osti/96647.pdf> (unreachable from the sandbox; quoted facts are labelled OWNER-REPORTED).
* USGS QFFD / INGENIOUS — the label source named by the problem statement. California QFaults v2.0 (cross-check only): <https://www.conservation.ca.gov/cgs/qfaults>. SGMC faults (off-catalogue proxy frame): <https://ngmdb.usgs.gov/>.
* Methods: Phillips et al. 2009 <https://doi.org/10.1890/08-2025.1>; Fithian & Hastie 2013 <https://doi.org/10.1214/13-AOAS667>; Hermant et al. 2025 (SGW); Mattéo et al. 2021 (JGR).

## Irregularities

`IR-59-PROMPT-01` (prompt fidelity) · `IR-59-SEP-01` (deployed-field scale not
identified → rank-based emission) · `IR-59-EFFORT-02` (no QFaults join key) ·
`IR-59-SGMC-01` (the “derived” SGMC layer is byte-identical to the base layer) ·
`IR-59-REPR-01` (literal overlap rule unsatisfiable against dense priors) ·
`IR-30-039` (catalogue proxies rank opposite to the live board). Full text on the
site's audit page and in `docs/verification.md`.

**No competition slot was used by this repository, and none of its numbers is an
organizer-confirmed score.**
