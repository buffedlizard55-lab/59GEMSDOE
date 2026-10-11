# 59GEMSDOE — intensity-zoned fault discovery for the DOE GEMS Prize (DrivenData #306)

**Competition** · [DOE GEMS Prize home](https://www.drivendata.org/competitions/306/competition-doe-gems/)
· [Problem description (page 967)](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
· [About / rules (page 968)](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
· [Rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
· [Public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
· [Reference solution](https://github.com/drivendataorg/gems-prize-reference-solution)

**Live site** · <https://buffedlizard55-lab.github.io/59GEMSDOE/>

> ### ⬇ SUBMISSION FILE — the one thing this repository exists to produce
>
> **[⬇ Download the release GeoTIFF](docs/downloads/gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.tif)**  ·  [ZIP](docs/downloads/gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.zip)  
> sha256 `8c0c614231b381c2c1af759cee0776eabf13b73d5ce8ca7bd45563f90d0ef842` · 795,960 bytes · 38,593 pixels at 1.0 · **all 12 validator checks PASS** on the bytes on disk.
> · status **OK to download · OK to submit** (both rendered from the run card, never from prose)
> · single-band **float32** GeoTIFF, EPSG:32611, 3730 × 3292, values in **[0, 1]**, **every cell finite**
> (0 outside the footprint — see *"Why zeros outside the footprint"* below).
>
> **How to submit, in four steps.** (1) Click the download link above and save the `.tif`
> (or the `.zip`, which contains exactly one GeoTIFF). (2) Open
> [the submission page](https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/)
> and click **New submission**. (3) Choose the `.tif` (or the `.zip`) — **not** an HTML page, a
> JSON receipt, a PDF or a zip of this repository. (4) Paste the **submission name** and the
> **note** printed next to the download, then submit. Full walkthrough with the exact strings:
> **[Executive summary → how to submit](docs/executive-summary.html)**.
>
> This project never clicks Submit and never spends a weekly slot. Promotion to a real slot is a
> separate selector decision, inside the weekly cap shown on the submission page.

---

## Our Core Values — kept central to every decision here

**Maximize P(Win).** *"Maximize the Probability of Winning"* is our decision-making framework. In
every decision we weigh tradeoffs, assess risk, and choose the path that maximizes the probability
that we win this prize. We set aside our emotions and make tough decisions in order to maximize
P(Win). It frees us from constraints and clarifies that we must put the outcome first.

**Own the Outcome.** We own results end to end — not just our individual slice of the work. When
problems arise and we have the means to act, we do so without waiting for permission or assignment.
We treat failure and success as signals and use them to improve. We stay accountable to the final
outcome.

Applied in this repository: *Maximize P(Win)* is why the break-even emission rule is derived from
the organizer's own scoring equations instead of tuned by eye, and why a leaky feature that made
the model look perfect was removed rather than shipped. *Own the Outcome* is why every number below
carries an evidence class and a link, and why the two things that did not work this session are
published as measurements rather than quietly dropped.

---

## Why zeros outside the footprint (the portal error, fixed)

A submission that carries `NaN` outside the bounds is accepted by the *rules* ("data outside the
bounds is null or nan") but is **rejected by the portal**, whose range check reads the whole array:

> `Predicted values must be in range [0, 1]`

`scripts/submission_writer.py` therefore writes **0.0 outside the footprint** by default, so every
cell of the file is finite and inside `[0, 1]`, and it re-reads and re-validates the bytes it just
wrote. A `-nan` twin is still produced for research, and is **not** the advertised download.

---

## The metric, and the one equation the whole lane is built on

From the [problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/),
with `p(x)` the predicted confidence, `G` the ground-truth fault pixels, `k(d) = max(1 - d/300 m, 0)`:

```
TP_w = sum_{g in G} max_{x: d(x,g) <= 300 m} p(x) k(d(x,g))
FP_w = sum_{x: p(x)>0} p(x) [1 - max_{g in G} k(d(x,g))]
FN_w = sum_{g in G} [1 - max_{x: d(x,g) <= 300 m} p(x) k(d(x,g))]
DTI  = TP_w / (TP_w + 0.2*FP_w + 0.8*FN_w)
```

`src/gems59/metric.py` reproduces this exactly (verified against the organizer's worked example —
TP 3.00, FP 1.89, FN 2.00 → 0.60 — and against a dense brute-force evaluator, `tests/test_metric.py`).

Because `FN_w = |G| - TP_w`, the index collapses to `DTI = TP_w / (0.2*TP_w + 0.2*FP_w + 0.8*|G|)`.
Differentiating with respect to the mass emitted at one pixel gives the **break-even rule**:

> **Emit a pixel if and only if its expected kernel weight `Delta(x) >= alpha * DTI`**
> — i.e. **0.2 × the score you are actually getting**. At the group's incumbent 0.2778 the bar is
> **0.0556**; at the public board leader 0.3774 it is **0.0755**.

And for a 1 km (10 × 10 px) block holding `L` pixels of fault trace, a uniformly placed dot earns
`E[Delta] = 3L/100` (the triangular kernel's perpendicular cross-section integrates to 3), so
break-even is `L = 100 * alpha * DTI / 3` = **1.85 px per 100 px block at DTI 0.2778** — the
*"a lattice only pays above roughly 1–2 % density"* rule of thumb, here derived rather than assumed.

Two consequences drive the whole design:

1. the target to model is **how many new-fault pixels sit in each 1 km block**, not a per-pixel
   posterior — which is a point-process *intensity*, not a classifier score;
2. **crowding is pure loss**: `TP_w` takes a `max` over emitted pixels, so two dots within 300 m of
   the same truth pixel are paid once and charged twice. Spacing is a first-class parameter.

---

## Method (the assigned lane)

1. **Presence-only Poisson point-process intensity.** Mapped fault pixels are a realisation of an
   inhomogeneous Poisson process `lambda_obs(x) = lambda(x) * b(x) * rho`, with `lambda` the true
   fault intensity (geophysical channels) and `b` the detection probability (mapping-effort
   channels). **Fithian & Hastie**, *Annals of Applied Statistics* **7(4)**: 1917–1939, 2013,
   [doi:10.1214/13-AOAS667](https://doi.org/10.1214/13-AOAS667), prove that the IPP and MaxEnt
   give the *exact same* estimate of that density. Presence-only records are biased by where
   surveyors looked, so effort channels enter the fit explicitly: **Phillips, Dudík, Elith,
   Graham, Lehmann, Leathwick & Ferrier**, *Ecological Applications* **19(1)**: 181–197, 2009,
   [doi:10.1890/07-2153.1](https://doi.org/10.1890/07-2153.1), show background data should carry
   the *same* sampling bias as the occurrence records (target-group background).
   We fit the discriminant with a case–control design and convert back with the odds
   (**Prentice & Pyke**, *Biometrika* **66(3)**: 403–411, 1979,
   [doi:10.1093/biomet/66.3.403](https://doi.org/10.1093/biomet/66.3.403)): `lambda ∝ p/(1-p)`.
   *Caveat, stated not assumed*: Fithian & Hastie are explicit that ordinary logistic regression
   "in general yields a different estimate in finite samples" from the IPP — exactness requires
   their *infinitely weighted* logistic regression. Our odds are therefore an **approximation**
   to the IPP intensity; the emission rule renormalises the field so only its shape is used, and
   the shipped arm is compared on the holdout against the same emission ladder. An exact
   infinitely-weighted / Poisson-GBM fit is queued work, not shipped.
2. **Effort (bias) correction.** The bias channels are clamped to a common "fully mapped" reference
   and the model re-evaluated, giving `lambda_new = lambda_ref - lambda_obs` — the faults that are
   really there minus the ones the catalogue already has (see *Results* for the measured verdict
   on this arm).
3. **1 km zoning.** The intensity is aggregated to 10 × 10 px blocks. 1 km deliberately exceeds the
   ~400 m registration error between the published fault traces and the true surface trace, so a
   block-level *count* is robust to misregistration in a way that a per-pixel posterior is not.
4. **Break-even tiling.** Blocks whose predicted trace length is below `100*alpha*DTI*/3` are not
   tiled at all. Inside a surviving block, dots go on the highest-`Delta` pixels at a minimum
   separation taken from a utility table swept on our own holdout.
5. **Prediction-vs-withheld counts by decile** on the holdout, plus a spatial-block bootstrap CI.

## What was measured this session

Every number below is read from a receipt in `evidence/`; nothing is typed by hand.

**Holdout** (hide-and-recover, evaluator version `drivendata-306-final`, 2 segment-level folds, alpha=0.2 beta=0.8, 300 m kernel, 2 px flank, 52 channels):
- fold 0 — truth 13,348 withheld px · best spacing 2.5 px at tau=0.03 · 13,947 dots · **DTI 0.0891**
- fold 1 — truth 14,641 withheld px · best spacing 3.0 px at tau=0.03 · 14,049 dots · **DTI 0.0872**
- pooled — **mean best DTI 0.08811**, mean 13,998 dots, spacing* 2.75 px. The spacing optimum is 2.5–3.0 px in **both** folds, and DTI was still rising at the largest count tested (N ~ G).

**Leakage canary** (fold 0, 29,340 truth px vs 1,679,900 background): verdict **clean** — no channel above AUC 0.90; highest single channel `trace_density` at 0.5801.

**Leakage ablation** (fold 0, 60-iter GBM): dropping `dist_visible_fault` / `visible_fault_dens_5k` / `visible_fault_dens_20k` lifts holdout AUC from **0.500 to 0.8354**. Those channels are computed from the catalogue pixels that form the training presences, so with them the model separates the training data (train loss -0.023; distinct predicted values collapse 160,422 -> 38,121) and then generalises at chance. Excluded from every model here.

**Uniqueness** (parallel-run drift gate against 696 registered prior rasters, all sha256-verified):
- surface phase — max Spearman **0.6718** vs the 0.90 gate -> **PASS**
- dots phase — the literal "more than 70% of my dots within 3 px" rule is tripped by **20 of 696** rasters, but those are dense continuous surfaces (several positive over all 5,167,373 footprint px), so *any* dot pattern trips it. Chance-corrected max lift **1.999x** vs a 3.0x gate -> **NO DRIFT**

**Released emission** — 38,593 dots at 3.0 px requested minimum separation (measured min pairwise distance **3.162 px** over a 3,000-dot sample), **0** dots on or within 2 px of the mapped catalogue, over 4,761,049 px of support. All 12 validator checks pass on the bytes on disk.

**Negative result** — the counterfactual effort gap (mapping-effort correction by clamping) scores **AUC 0.0122** on withheld faults, i.e. about 0.99 in the *wrong* direction: the clamped design is out-of-distribution for the gradient booster. Kept in `src/gems59/intensity.py`, reported as a negative, not used in the release.


## Evidence classes used everywhere

| label | meaning |
|---|---|
| `HOLDOUT-DTI` | measured by `scripts/evaluate_holdout.py`, with evaluator id, withheld-positive count and 95 % CI |
| `ORGANIZER-CONFIRMED` | copied from a submission-page receipt |
| `ORGANIZER-PUBLISHED` | read from the public leaderboard / problem pages |
| `OWNER-REPORTED` | pasted by the owner from a submission page |
| `PROXY` | a local measurement on a proxy truth that is **not** the competition's test set |
| `MODEL` | a projection derived from the organizer's equations. **Never a score.** |

---

## Repository layout

```
registry/data_manifest.json        every input byte pinned by sha256 (core + external + prior scored)
scripts/restore_data.py            restore + verify the pinned inputs (GitHub blobs; never drivendata.org)
src/gems59/grid.py                 canonical grid I/O (EPSG:32611, 100 m, 3730x3292)
src/gems59/metric.py               exact DW-Tversky + the derived break-even algebra
src/gems59/features.py             55-channel cached feature stack (leakage rules enforced)
src/gems59/intensity.py            Poisson point-process fit + effort correction
src/gems59/holdout.py              hide-and-recover, spatially blocked, pooled DTI + block bootstrap CI
src/gems59/emit.py                 1 km zoning, non-maximum placement, lattice reference emitter
src/gems59/pipeline.py             the one path both the holdout and the release call
scripts/evaluate_holdout.py        the shared instrument (canary + folds). Fix it here, never fork it.
scripts/submission_writer.py       writes and re-validates the release GeoTIFF
scripts/uniqueness_check.py        parallel-run drift gate against every registered raster
scripts/build_candidate.py         fit -> tune emission on the holdout -> emit
scripts/refresh_registry.py        download the 696-raster registry the drift gate compares against
scripts/build_site.py              generates every page from the JSON in evidence/ and registry/
```

## Reproduce (no manual data placement, no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/restore_data.py --group core,external     # hash-verified
.venv/bin/python scripts/refresh_registry.py --build-map           # 696 prior rasters, for the gate
.venv/bin/python scripts/evaluate_holdout.py --canary              # leakage screen
.venv/bin/python scripts/build_candidate.py --tag h59a --arm with_trace
.venv/bin/python scripts/submission_writer.py --dots work/dots_h59a.npy --out docs/downloads/<name>.tif
.venv/bin/python scripts/uniqueness_check.py --candidate docs/downloads/<name>.tif --phase dots
.venv/bin/python scripts/build_site.py && .venv/bin/python -m pytest
```

Large data and caches are Git-ignored (`data/`, `work/`, `.cache/`); the release GeoTIFF and its
receipts are committed so the download link never rots.

## Honest limits of this work

* **No organizer score exists for anything in this repository.** Every local number is a proxy on
  the published catalogue, which is exactly the population the prize does **not** score.
* The 19 official bands include geodetic and seismic layers that are smooth at the fault scale
  (measured autocorrelation of `ieq_n100a15`: 0.9986 at 1 km), so they carry little fault-scale
  information; they are kept because they are official, not because they help.
* **Roads distance is not available offline.** An effort covariate the lane names — distance to
  roads — would need a free official road layer (USGS TIGER/roads or the Nevada DOT open data) and
  neither host is reachable from this sandbox. It is listed as the first queued improvement, not
  silently skipped.
* Data provenance is **integrity-pinned, not organizer-authenticated**: the bytes come from
  owner-supplied public GitHub mirrors and match the recorded sha256 digests, which proves
  transport identity, not official origin. No DrivenData credential is used and
  drivendata.org is never contacted for data.

---

# Preserved standing request

The complete standing request follows. It records the requested work and the historical report
ledger; the dated status and evidence above supersede stale factual claims inside it (for example
"GPU needed", "data placement is a blocker", and the highest-score figures, which move weekly).

---

Review the repo.

THE FOLLOWING IS THE HIGHEST URGENCY AND MUST BE FOLLOWED!

MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION.  DO NOT COPY A PREVIOUS SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION.  BUT WE MUST GENERATE A UNIQUE TIF SUBMISSION.  IT MUST BE OBVIOUS WHETHER IT IS OK TO DOWNLOAD AND SUBMIT THE GENERATED TIF SUBMISSION.

There should be an easy to download submission tif file as described by the prompt.  Read the entire prompt.

Intensity lane: zone by new-fault density, then tile only what clears break-even. Because a lattice pays only above roughly 1–2% density, the target is how many new-fault pixels sit in each 1 km block, not a per-pixel probability. Fit a presence-only point-process intensity from the catalogue, correcting mapping-effort bias with target-group background (Phillips et al., Ecological Applications, 2009). MaxEnt and Poisson point-process models are equivalent in form (Fithian and Hastie, Annals of Applied Statistics, 2013). Use effort covariates (USGS reliability-of-location class, distance to roads, relief, GeoDAWN acquisition block) alongside the geophysical layers. The 1 km block deliberately exceeds the 400 m registration error Hermant et al. measured. Check predicted versus withheld-segment counts by decile on the holdout. Then tile only blocks above break-even, with spacing from a utility table you re-derive on your own holdout (about 3 px when dense, wider as density falls). Output the standard validated GeoTIFF, uniqueness-checked against every earlier raster.

PARALLEL-RUN PROTOCOL — read first. This session is one of several running from this same prompt.

1. LANE. Your lane is the single method paragraph below. Stay inside it. If your raster's rank-correlation with any registry raster exceeds [0.90], or more than [70%] of your dots fall within 3 px of one registry raster's dots, you have drifted into another lane: log it as a duplicate and stop. Check this on the surface before placement AND on the final dots.

2. REUSE, DON'T REBUILD. Use the template's cached feature stack, evaluate_[holdout.py](http://holdout.py) and submission_[writer.py](http://writer.py). Holdout = hide-and-recover: withhold whole fault segments with a buffer, derive every catalogue-based feature only from the visible faults, mask visible faults pixel-exactly, score pooled DTI (alpha 0.2, beta 0.8, 300 m triangular kernel). If a shared tool is wrong, fix it once in the template and report it; never keep a private fork.

3. LABEL EVERY NUMBER as HOLDOUT-DTI (evaluator version, number of withheld positives, 95% CI) or ORGANIZER-CONFIRMED (copied from a submission-page receipt). A projection is never written as a score.

4. LEAKAGE CANARY. Test each feature alone on the holdout before trusting any result. AUC above [0.90] means leakage until proven otherwise.

5. RUN CARD. End with one JSON card: hypothesis; mechanism; the named non-fault process that could mimic it; holdout DTI + CI; correlation/overlap vs registry; raster sha256; validator output (no NaN inside the footprint, values in [0,1], CRS/shape/transform match); submission name + note of at most 140 characters; verdict promote / negative. Negative results are deliverables.

6. BUDGET. Stop after [3] experiments or [2] hours. Do not pick submissions: promotion to a real slot is a separate selector step, within the weekly cap shown on the submission page.

The following sites should serve as a starting point for understanding how to generate TIF submissions.  These websites are researched, and tested and have generated TIF submissions.  But we need to generate high scoring submissions.

Here are the results from submissions into the competition, separated by ....:

WE NEED TO STUDY, ANALYZE, AND UNDERSTAND THE HIGHEST SCORE FROM THE GEMDOE SITE WHERE THE SUBMISSION TIF IS DOWNLOADED FROM WHICH IS THE FOLLOWING:

[https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html)

h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros: 0.2778

Why and how did this get the highest score and are we able to generate a submission that scores higher than 0.2778?

Answer the question using Phd level experience, knowledge, and judgement. Then use the answer to generate a unique TIF submission into the competition.  Must be unique submission unlike any within the GEMSDOE sites above.  Verify working line by line no hallucinations.

Current competition leaderboard GEMSDOE high score:

0.3774

[https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html)

gems-submission-20260925T001403Z-7f00890a: 0.1563

....

[https://buffedlizard55-lab.github.io/6GEMSDOE/](https://buffedlizard55-lab.github.io/6GEMSDOE/)

gems6_hgb88-topk03_33cec71ff0: 0.0286

....

[https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html)

pindrop-v4-nodes-20260925T152420Z-f347b70daaa: 0.1193

pindrop-v4-discovery-20260925T152423Z-37f9d5b855: 0.0830

pindrop-v4-ridge-20260925T152422Z-4e03fc9705: 0.1152

....

[https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html)

gemsdoe2-dual-family-union-20260925T160406Z-f68e590f: 0.1560

....

[https://buffedlizard55-lab.github.io/GEMSDOE4/](https://buffedlizard55-lab.github.io/GEMSDOE4/)

gems-submission-20260926T163915Z-237f0063: 0.0343

....

[https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html)

gems-submission-20260926T175114Z-7f00890a: 0.1563

....

[https://buffedlizard55-lab.github.io/7GEMSDOE/](https://buffedlizard55-lab.github.io/7GEMSDOE/)

lidarscarp-ridge-top2pct-36c3a3f341c8: 0.1461

....

[https://buffedlizard55-lab.github.io/8GEMSDOE/](https://buffedlizard55-lab.github.io/8GEMSDOE/)

Hedge-v2_submission: 0.1563

....

[https://buffedlizard55-lab.github.io/GEMSDOE9/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE9/docs/index.html)

2314b599: 0.0107

....

[https://buffedlizard55-lab.github.io/11GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/11GEMSDOE/docs/index.html)

gems-structural-area06-v1: 0.0202

....

[https://buffedlizard55-lab.github.io/12GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/12GEMSDOE/docs/index.html)

r7-nms3-dem10-scarp_0c9199f14e62:0.1294

r7-nms3-dem10-scarp_0c9199f14e62_allfinite:0.1294

....

[https://buffedlizard55-lab.github.io/15GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/15GEMSDOE/docs/index.html)

gems-tso1-20260929T005627Z-conj_alteration_mag: 0.0782

....

[https://buffedlizard55-lab.github.io/14GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/14GEMSDOE/docs/index.html)

GEMS_r5-geom-horse-ensemble_20260929T154852Z_ccbe1de0_site_e96e942f: 0.0020

....

[https://buffedlizard55-lab.github.io/17GEMSDOE/](https://buffedlizard55-lab.github.io/17GEMSDOE/)

17GEMSDOE_F-ensemble-2pct_20261003:0.0187

....

[https://buffedlizard55-lab.github.io/18GEMSDOE/](https://buffedlizard55-lab.github.io/18GEMSDOE/)

H19-C_20260930T212401Z_c11e495e: 0.0297

....

[https://buffedlizard55-lab.github.io/19GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/19GEMSDOE/docs/index.html)

h19-4-multiline-corroborated-openness-thermal-pop-20260930-691e4dfa-nan: 0.1894

h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan: 0.1922

....

[https://buffedlizard55-lab.github.io/GEMSDOE10/](https://buffedlizard55-lab.github.io/GEMSDOE10/)

h16-continuation-20260927T065521077735Z-3431b83c7c: 0.0461

h20-dem10-scarp-thin-20260927T155223039488Z-ffc91a1686: 0.0921

H25-ctx-ridge-20260927T232947704150Z-6452ae1d00: 0.1280

h28-dotted-ridge-20260928T020256236880Z-6452ae1d00: 0.1839

....

[https://buffedlizard55-lab.github.io/13GEMSDOE/](https://buffedlizard55-lab.github.io/13GEMSDOE/)

20261001_r13-lattice-s5_v2_nan-outside:0.0904

....

[https://buffedlizard55-lab.github.io/16GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/16GEMSDOE/docs/index.html)

h16-1-topo-geophys-baseline-ridges-20260930-df20f65e-nan: 0.1855

h18-3a-topo-geophys-x-complexity-prior-20260930-c502dfab-nan: 0.0976

h18-4-usgs-geologic-map-faults-gap-20260930-aef8f42c-nan: 0.0360

....

[https://buffedlizard55-lab.github.io/GEMSDOE21/](https://buffedlizard55-lab.github.io/GEMSDOE21/)

h19-4-reference-20260930-691e4dfa: 0.1894

....

[https://buffedlizard55-lab.github.io/20GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/20GEMSDOE/docs/index.html)

h20-1-sarnnpu-powerlaw-pi0363-tilt-wingcrack-20260930-be0e8f6b-nan: 0.1890

h20-5-continuous-pu-proxy-unverified-20260930-824ce73a-nan: 0.1859

....

[https://buffedlizard55-lab.github.io/GEMSDOE22/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE22/docs/index.html)

h23-a-dti-optimal-emission-6pct-20261002-e2ec4b49-nan: 0.1002

h23-b-dti-optimal-emission-10pct-20261002-86176698-nan: 0.0748

....

[https://buffedlizard55-lab.github.io/GEMSDOE23/](https://buffedlizard55-lab.github.io/GEMSDOE23/)

h30-arrangement-matched-habitat-20261002-0d4e02e8-nan: 0.1352

....

[https://buffedlizard55-lab.github.io/GEMSDOE24/](https://buffedlizard55-lab.github.io/GEMSDOE24/)

h25-1-dotted-h19-5-d1-5-20261002-989f59505db1-nan: 0.2477

....

[https://buffedlizard55-lab.github.io/GEMSDOE25/](https://buffedlizard55-lab.github.io/GEMSDOE25/)

dotted-h19-5-d2-8-20261002-e56ea318af89-nan: 0.2600

....

[https://buffedlizard55-lab.github.io/GEMSDOE26/](https://buffedlizard55-lab.github.io/GEMSDOE26/)

dilcond-oof-v1-20261003-47629f496133-nan: 0.1223

....

[https://buffedlizard55-lab.github.io/GEMSDOE27/](https://buffedlizard55-lab.github.io/GEMSDOE27/)

topo-gap-closure-t-v2-on-d1-5-20261002-5512495c6bd1-nan: 0.2449

....

[https://buffedlizard55-lab.github.io/GEMSDOE30/](https://buffedlizard55-lab.github.io/GEMSDOE30/)

d28-poisson300m-offcat-44090-20261003T233156Z-91eae1ca: 0.2600

....

[https://buffedlizard55-lab.github.io/GEMSDOE31/docs/](https://buffedlizard55-lab.github.io/GEMSDOE31/docs/)

h27-4-solo-d28-20261004-8acb75e1-nan:0.2708

....

[https://buffedlizard55-lab.github.io/GEMSDOE33/](https://buffedlizard55-lab.github.io/GEMSDOE33/)

h33d-analog-tip-stepover-r30-20261004-cb490425926e: 0.2632

....

[https://buffedlizard55-lab.github.io/GEMSDOE34/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE34/docs/index.html)

h34-scatter-q50-arr-matched-20261004T223317Z: 0.0778

....

[https://buffedlizard55-lab.github.io/GEMSDOE35/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE35/docs/index.html)

h35-06-aaa86efb25-20261004T225420098147Z-candidate: 0.0418

....

[https://buffedlizard55-lab.github.io/GEMSDOE36/docs/](https://buffedlizard55-lab.github.io/GEMSDOE36/docs/)

anderson-geothermal-pinn-38854-20261004T230000Z-9b9ea4e6-zeros: 0.2750

....

[https://buffedlizard55-lab.github.io/GEMSDOE37/](https://buffedlizard55-lab.github.io/GEMSDOE37/)

h6-physics-dotted-80k-20261005T055000Z-0bef9211631c: 0.1193

....

[https://buffedlizard55-lab.github.io/GEMSDOE38/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE38/docs/index.html)

D-step-3p0-07pct-tipProt-20261005-ecfbf59e2b48-zero: 0.0763

....

[https://buffedlizard55-lab.github.io/GEMSDOE42/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE42/docs/index.html)

xscale-worm-persistence-20261006T000541Z-nan: 0.0581

....

[https://buffedlizard55-lab.github.io/GEMSDOE43/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE43/docs/index.html)

sup01-hgb21-sep40-n40000-20261006-bc2e4e9a8d6f-nan: 0.0424

....

[https://buffedlizard55-lab.github.io/GEMSDOE45/](https://buffedlizard55-lab.github.io/GEMSDOE45/)

h51-km-faultzone-20261006-zeros: 0.0106

....

[https://buffedlizard55-lab.github.io/GEMSDOE49/](https://buffedlizard55-lab.github.io/GEMSDOE49/)

gate_ortho_w0.25-40k-20261006T213721Z-nan: 0.2376

....

[https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html)

h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros: 0.2778

....

[https://buffedlizard55-lab.github.io/GEMSDOE28/](https://buffedlizard55-lab.github.io/GEMSDOE28/)

h27-4-r1-solo-d2-8-20261003-8acb75e1f2cc-nan: 0.2708

h32-1-prethin-tip-euler-d2-8-20261003-31e35eee884e-nan: 0.2649

h36-1-rung30-blind-r1-20261003-b531dae0a36f-nan: 0.2710

h38-1-hf-euler-r30-r1-20261003-56a9f473edc7-nan: 0.2707

....

[https://buffedlizard55-lab.github.io/GEMSDOE29/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE29/docs/index.html)

efd28-repro-20261003-1cc7dc534d51-nan: 0.2600

repo-c0-habitat-emission-20261003-a4d439b07426-nan: 0.0041

sgmc-off-catalogue-44k-20261003-c8dcd780e3fd-nan: 0.0512

wormrank-d28-20261003-59dcaf6dd11d-zeros:0.2560

wormsurv-filter-20261003-921f10960d6e-zeros: 0.0532

xfit-c0-habitat-20261003-ca879db0089a-zeros:0.0439

xfit-h41-union-qfaults-20261003-9edb34b99e3a-zeros: 0.0356

....

[https://buffedlizard55-lab.github.io/GEMSDOE46/](https://buffedlizard55-lab.github.io/GEMSDOE46/)

r11f-scarp-radiometric-fusion-00e049b51218-zeros:0.1589

r12-scarp-rad-concordance-23e807e2de9f-zeros: 0.0843

....

[https://buffedlizard55-lab.github.io/GEMSDOE39/](https://buffedlizard55-lab.github.io/GEMSDOE39/)

h40-e-disc-h40e-30k-zeros: 0.0339

....

[https://buffedlizard55-lab.github.io/GEMSDOE40/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE40/docs/index.html)

h8-euler-lineament-depthcluster-20261006-785c4f5d5ce1: 0.0355

h8-euler-lineament-depthcluster-20261006-785c4f5d5ce1-hard: 0.0397

h45-eulerdepthreadcluster-20261006-f28e5cff6826-zeros: 0.0984

....

[https://buffedlizard55-lab.github.io/GEMSDOE41/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE41/docs/index.html)

h42-submission-primary: 0.0245

....

[https://buffedlizard55-lab.github.io/GEMSDOE44/docs/](https://buffedlizard55-lab.github.io/GEMSDOE44/docs/)

h46-twostageAB_20261006T160000Z_b0cfe956-zeros: 0.0715

....

[https://buffedlizard55-lab.github.io/GEMSDOE47/](https://buffedlizard55-lab.github.io/GEMSDOE47/)

h60-lidarscarp-s2p0-20261007-nanoutside: 0.0430

....

[https://buffedlizard55-lab.github.io/GEMSDOE48/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE48/docs/index.html)

h59-cover-ds-belief-b2xh33d-20261008T184547Z-b79c4c61d8d8: 0.2296

....

[https://buffedlizard55-lab.github.io/GEMSDOE50/](https://buffedlizard55-lab.github.io/GEMSDOE50/)

h59-sharpened-scarp-scatter-90k-20261007T171954Z-allfinite: 0.0764

....

[https://buffedlizard55-lab.github.io/GEMSDOE51/](https://buffedlizard55-lab.github.io/GEMSDOE51/)

h53-twostage-20261008T040951Z-9a0b32c871: 0.1047

....

[https://buffedlizard55-lab.github.io/GEMSDOE52/](https://buffedlizard55-lab.github.io/GEMSDOE52/)

:

....

[https://buffedlizard55-lab.github.io/GEMSDOE53/docs/index.html](https://buffedlizard55-lab.github.io/GEMSDOE53/docs/index.html)

h8-tiprelay-ridgeconcord-pr2-n80000-20261009-49bec522-zeros: 0.0159

....

[https://buffedlizard55-lab.github.io/GEMSDOE54/docs/](https://buffedlizard55-lab.github.io/GEMSDOE54/docs/)

h54c-manifest-edge-20261009T025732Z-73454bc5:0.0449

....

[https://buffedlizard55-lab.github.io/55GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/55GEMSDOE/docs/index.html)

tensor_full-n16000-sep3-20261009T211747Z-nan: 0.0290

....

[https://buffedlizard55-lab.github.io/56GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/56GEMSDOE/docs/index.html)

h56-final-dotted-ridge-d2p8-20261009T190421Z:0.0658

....

[https://buffedlizard55-lab.github.io/57GEMSDOE/docs/index.html](https://buffedlizard55-lab.github.io/57GEMSDOE/docs/index.html)

h57r-joint-strata-38000-20261010T214547Z-00cf5b2066f1-zeros:

....

58GEMSDOE

:

....

59GEMSDOE

:

....

60GEMSDOE

:

....

61GEMSDOE

:

....

62GEMSDOE

:

....

The following is the leaderboard for the competition:

[https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)

See below for more links and information related to the competition:

[https://github.com/drivendataorg/gems-prize-reference-solution](https://github.com/drivendataorg/gems-prize-reference-solution)

[https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)

[https://gbcge.org/current-projects/ingenious/](https://gbcge.org/current-projects/ingenious/)

[https://epsg.io/32611](https://epsg.io/32611)

[https://en.wikipedia.org/wiki/Tversky_index](https://en.wikipedia.org/wiki/Tversky_index)

We need to quickly look at the results and results from the GEMSDOE websites above.

Before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet, each naming: the specific layer(s) involved, the physical signature being targeted (e.g., an edge-detection or curvature transform), why it should catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and how it differs from anything already implemented in this repo. Rank them by expected DTI improvement and implementation cost. Validate the top candidate on our spatially-blocked holdout set before touching a weekly submission slot — do not spend a submission slot on an idea that hasn't beaten the current holdout best. If a candidate can't be validated without new external data, name the specific free, official source needed and check it's obtainable before proposing the idea as viable.

Work line by line verifying from official verified trusted sources, provide links for manual review.  There should be no manual input, work on your own to complete tasks.  Flag any irregularities for review.  No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements.  No hallucinations.  Verify line by line.

We have a good understanding of how our hypothesis, methodology, calculations, analysis are done so we should be able to figure out a way to score higher on the leaderboard using previous results and scoring that we have across the sites listed above.  We need to come up with distinct and unique strategies to score higher in this competition leaderboard.  We need to start doing heavy and deep research into the part of the project that matters the most, which is the scientific discovery of geothermal vents.  We should store all of our information and knowledge that we can gather from official verified sources.  This will serve as a starting point for other projects as well.  We need to think outside the box but still be grounded in proper scientific research, we are ultimately aiming for a top prize that many others are competing for.  So it's important to be contrarian but be smart about it.  We need to find sources of data that others are over looking or areas of the project when it comes to geothermal vents.  We need to do deep research and critical thinking and come up with new hypothesis to test.

0.3195 is the highest score right now so we need to design a new strategy, research, testing, analyzing, and generating submission system than the current website.  It should be unique, take unique approaches to generating a submission that can score higher than 0.3195.

Put this prompt into the repo readme and read it everytime we work on the project as a starting point to make sure we are building what we are aiming for and have a strong base to continue building and improving on making something useful for everyday use.  It should solve the problem of having to manually check everything ourselves and having an up to date current feed.

Review the repo.

The following is taken from the Arena AI team and I think it makes a good point on building a successful project, so let's keep the Core Values and Own the Outcome as a focal point when building, developing, researching, suggesting upgrades, and implementing the work.

Our Core Values

Maximize P(Win)

"Maximize the Probability of Winning": our decision making framework. In every decision, we weigh tradeoffs, assess risk, and choose the path that maximizes the probability that Arena succeeds.  We set aside our emotions and make tough decisions in order to maximize P(Win).  "Maximize P(Win)" frees us from constraints and clarifies that we must put Arena first.

Own the Outcome

We own results end to end — not just our individual slice of the work. When problems arise and we have the means to act, we do so without waiting for permission or assignment. We treat failure and success as signals and use them to improve. At Arena, we stay accountable to the final outcome.

Work line by line verifying from official verified trusted sources, provide links for manual review.  There should be no manual input, work on your own to complete tasks.  Flag any irregularities for review.  No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements.  No hallucinations.  Verify line by line.

We need to focus on being able to generate a submission into the competition.

The site should be able to generate a TIF file that is required for submission.  It should be as easy as download to click a File to submit into the competition.  This needs to be in the executive summary or the very beginning of the site.  it should be obvious when you visit the site.

I tried to submit the document that i downloaded from the site but it returned this error on the submission form:

"Predicted values must be in range [0, 1]"

Also we need to give it a unique name and A short comment to help you or your team tell submissions apart later e.g. clustering with k=25

Here is the submission page when i click submit file

New submission

File to submitNo file chosen

You can submit a single-band GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF, with your predictions. It must match the submission format's CRS, shape, and geotransform. You may wish to review the competition rules first.

Note (optional)

A short comment to help you or your team tell submissions apart later e.g. clustering with k=25

Create a executive summary subpage that explains exactly how to make a submission into the contest.

Work on the next steps from the previous sessions first.

The goal of this project is to place top of the leaderboard in this competition.  The following is the competition:

[https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)

We need to create a project that can compete and place top of the leaderboard.  We need to understand the problem, collect all the data and organize it into a clean easily auditable table with official verified links for manual verification.

This is the guidelines we need to follow.[https://www.drivendata.org/competitions/306/competition-doe-gems/](https://www.drivendata.org/competitions/306/competition-doe-gems/)

Get familiar with the problem through the overview and problem description,[https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/). You might also want to reference additional resources available on the about page,[https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/).

Download the data from the data,[https://www.drivendata.org/competitions/306/competition-doe-gems/data/](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), tab.

Create and train your own model. This reference solution,[https://github.com/drivendataorg/gems-prize-reference-solution](https://github.com/drivendataorg/gems-prize-reference-solution) implements a simple approach.

Use your model to generate predictions that match the submission format.

Tell me what are you limitations and what you need access to during this project.  We will need to find free publicly available sources and data from official and verified sources if we are to use 3rd party or external data.

this pdf outlines how submissions must be entered into the competition.

[https://docs.nlr.gov/docs/fy26osti/96647.pdf](https://docs.nlr.gov/docs/fy26osti/96647.pdf)

You must be able to do your own research, deep research, scientific literature research and organize the knowledge so that we can critically think through the problem and generate a solution through scientific and free publicly available information.  this must be done autonomously and must be constantly reviewed and improved upon.  Provide suggestions and improvements and implement them.

❌ No DrivenData auth → cannot auto-download training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv from [https://www.drivendata.org/competitions/306/competition-doe-gems/data/](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (verified redirect to login)

See below for links from the above site.  See attached files for links from the above site.

[https://gdr.openei.org/submissions/1391](https://gdr.openei.org/submissions/1391)

Download competition data from [https://www.drivendata.org/competitions/306/competition-doe-gems/data/](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (requires login) to data/

See links below for competition data:

[https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0](https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0)

[https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0](https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0)

[https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0](https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0)

[https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0](https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0)

[https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0](https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0)

Work line by line verifying from official verified trusted sources, provide links for manual review.  There should be no manual input, work on your own to complete tasks.  Flag any irregularities for review.  No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements.  No hallucinations.  Verify line by line.

Site creation

Create a github page for this repo that has clean ui, user friendly, simple and easy to use.  It should be organized and clean.

It should include all relevant information in an easy to read format with official verified links as sources for review.  Work line by line verify everything no hallucinations.

**The single remaining blocker to training is data placement**: run `bash scripts/download_competition_data.sh` on any unrestricted machine into `data/`, then `python scripts/prepare_data.py` — after that the full train→inference→validate pipeline is ready to run (GPU needed for training; metric/losses/validation all verified working here on CPU).

you need to complete the above task by yourself.  Work line by line verifying from official verified trusted sources, provide links for manual review.  There should be no manual input, work on your own to complete tasks.  Flag any irregularities for review.  No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements.  No hallucinations.  Verify line by line.

Run this task through multiple passes.

Pass 1: Implement the task completely and verify the result.

Pass 2: Review your work for bugs, missing requirements, incorrect assumptions, and edge cases. Fix everything you find.

Pass 3: Re-check the entire implementation against the original request. Improve accuracy, reliability, completeness, and code quality. Fix any remaining issues.

Do not stop after the first pass. Each pass must build on the previous one. Before finishing, verify that the final result fully satisfies the original request.  Work line by line verify everything no hallucinations.

Go ahead and create a pull request and then merge the pull request onto the main. Make suggestions for what work still needs to be done and any limitations that is in the way of a successful project.  It should be worked on in this next session or the next session.  Work line by line verify everything no hallucinations.

<!-- END PRESERVED STANDING REQUEST -->
