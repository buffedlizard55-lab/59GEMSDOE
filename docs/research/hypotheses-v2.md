# Candidate hypotheses, session H59-J — five ranked, with overlap and obtainability checks

**Status labels.** `BUILT / HELD` (a raster exists but fails or has an incomplete strict gate; inspection only),
`MEASURED` (a number was produced here, with its instrument named), `PROJECTION` (judgement, never a score),
`BLOCKED` (cannot be tested here, with the reason and the source that is missing).

Nothing in this page is a competition score. Evidence classes are the ones defined in
`docs/research/why-02778.md`. The participant-level public board snapshot is #1 0.4133 / #8 0.3195 /
#23 extradr19 0.2778 (ORGANIZER-CONFIRMED, fetched 2026-10-11); no row is a file-linked receipt for H59-J or H59-I.

---

## What the measurement in `why-02778.md` rules out

The session's owner-maintained calibration record shows that emitted dot count and the shared
hide-and-recover holdout are strongly associated with participant-level public-board values, but the
file-to-score attribution is not authenticated by a submission-page receipt. The holdout/public-board
rank relation is negative (IR-59J-01), so a local holdout is not used as a promotion criterion. A new
hypothesis therefore has to clear three bars at once:

1. it must produce a **precision** point set (dots within ≈ 2.83 px of an unmapped fault), because at
   DTI ≈ 0.28 a dot beyond that radius is a net loss (`docs/research/utility-derivation.md`);
2. it must not be a catalogue-echo (the truth is by construction *not* in the catalogue, and dots on
   the catalogue are masked and worthless);
3. it must be rank-distinguishable from the 695 registry rasters, or it is a duplicate by protocol
   rule 1.

---

## H1 — Multi-physics structure-tensor coherence consensus, dotted at a live-calibrated budget (BUILT / STOP)

* **Layers:** band 12 `det_elev` (topographic scarps), band 2 `rtp` (basement/contact lineaments),
  band 13 `iso_grav_anom` (density-contrast lineaments) — all from the official 19-band GeoDAWN stack
  (sha256 `4371c82e…23bc5`), all catalogue-independent.
* **Physical signature:** the *structure tensor* of each field at two scales (σ = 2, 5 px), reduced to
  gradient energy × coherence `((Jxx−Jyy)² + 4Jxy²)^0.5 / (Jxx+Jyy)`. A fault trace is an anisotropic
  gradient feature: high energy **and** high coherence. Isotropic texture (alluvial roughness,
  radiometric speckle) has high energy but low coherence.
* **Why it can catch a fault the catalogue lacks:** the detector never sees the catalogue; the
  consensus requires agreement between topography, magnetics and gravity, so a single-physics mimic
  (a drainage line, a flight-line levelling step, a gravity datum seam) cannot fire on its own. The
  emission is additionally pulled ≥ 2 px off every mapped trace.
* **How it differs from what already exists:** the registry's own "multi-physics lineament consensus"
  (GEMSDOE32 H33-A, rank 1 of its set) is documented only by name; the previous lanes here used
  isotropic 9×9 local-contrast ranks (`gems59.intensity`) and block-tiled lattices. This is the first
  emission in the record gated on *coherence* (anisotropy), emitted as 2–3-dot fragments along the
  ridge direction, and budgeted from the live-calibrated dot-count relation rather than from the
  (inverted) holdout.
* **Named non-fault process that could mimic it:** long-wavelength sediment fabric and playa margins in
  the gravity/topography pair; magnetic dyke swarms and survey-line levelling residuals in `rtp`.
  Control: consensus across three independent physics + the collar + a budget cap.
* **Measured here:** 19,187 eligible ridge sites → 12,000 strongest sites × ≤3 dots = **30,815 dots**
  (0.596 % of the footprint); consensus mean 0.4221, at mapped catalogue 0.4841; 0 dots on the
  catalogue; 0 cells outside the footprint. HOLDOUT-DTI **0.019124 [0.016393, 0.022148]** — evaluator
  `gems57-pooled-hide-v2`, seed 20, 11,321 withheld positives. A separate corridor-6 diagnostic used
  8,062 withheld positives but has no 95% CI in its receipt, so no corridor score is reported as a
  qualified result. The qualified holdout instrument is anti-correlated with the live score (IR-59J-01)
  and does not clear this candidate.
* **Cost:** 20 s of CPU (3 band reads + 12 Gaussian filters + ranking). No new data.
* **Verdict:** Artifact built and held for inspection only; the strict literal registry gate records 86
  forward-overlap hits above 0.70 (DUPLICATE / STOP), and the newer 696-raster index was not scanned.
  Placement quality remains a `PROJECTION`, not a validated claim or competition score.

## H2 — Instrumental-seismicity lineaments × geodetic strain co-location (not run)

* **Layers:** band 10 `deq_n100a15` and band 16 `ieq_n100a15` (earthquake distance/density),
  band 4 `geod_2ndinv`, band 7 `geod_shearrate` (strain-rate tensor invariants).
* **Physical signature:** lineaments formed by *declustered* hypocenters, weighted by moment or count,
  intersected with the second invariant of the geodetic strain-rate tensor. Active faults concentrate
  both.
* **Why it catches a fault the catalogue lacks:** the catalogue is a **surface-geology** compilation —
  it records what a mapper could see. A seismogenic structure under basin fill can be missing from it
  while still producing earthquakes and measurable strain.
* **Why it is not just what already exists:** the previous lane used `deq`/`ieq` as *intensity
  covariates*; the registry's GEMSDOE32 H62-3 used strain "raw". An explicit seismicity-lineament
  transform (declustering → linear-cluster extraction → strain gate) does not appear in the record.
* **Supporting (weak) evidence, n = 24 rasters, partial Spearman controlling for dot count:**
  `geod_2ndinv` rank under the dots **+0.700 (p < 0.001)**, `deq_n100a15` **+0.636 (p = 0.001)** —
  i.e. the designs whose dots sit in high-strain, earthquake-dense ground are, conditional on size,
  the ones that scored higher. `n = 24` non-independent rasters; hypothesis-generating only.
* **Obtainability:** all layers are already in the official stack; a declustered catalogue would ideally
  come from USGS ComCat (<https://earthquake.usgs.gov/fdsnws/event/1/>), which this sandbox cannot
  reach (egress allow-list). The provided `deq`/`ieq` grids are a usable stand-in.
* **Cost:** medium (cluster extraction + a strain co-location rule + a budget). **Expected gain:**
  PROJECTION, unknown.

## H3 — Official mapping-scale completeness prior from the QFaults attribute table (not run)

* **Layer/data:** `data/external/qfault_attributes.csv` — 22,956 records of the USGS/INGENIOUS QFaults
  compilation; measured here: `MAPSCALE` has 12 values, dominated by 10 (11,334 records), 24 (3,025),
  100 (1,856) and **250 (2,229)**.
* **Physical signature:** none — this is a *completeness* prior, not a geophysical one. It
  parameterises where the existing map is allowed to be wrong: a 1:250,000 compilation cannot record
  scarps shorter than a few hundred metres, while a 1:24,000 quadrangle map can.
* **Why it catches a fault the catalogue lacks:** it predicts where a *missing* fault can be at all,
  independently of any geophysical response. This is the one candidate whose information is about the
  mapping process rather than the geology.
* **Difference from existing work:** the previous lane's effort covariates were *derived* (1 km / 5 km
  trace density, relief). The official scale attribute has not been used, precisely because the join
  from `trace_segments_utm11.csv` (84,331 segments, field `record_id`) to the attribute table
  (field `NUM`, e.g. `829a`, `2351g`) is unresolved — a prior session measured only 349/1,126
  intersecting names and 1.9 %/0 % agreement on a positional join.
* **Irregularity for review (IR-59J-02):** the join is a data-engineering problem, not a scientific one,
  and it must be solved before this hypothesis can be tested. Until then it is NOT viable.
* **Cost:** medium; **expected gain:** PROJECTION, low-to-medium.

## H4 — Doubly-unknown structures: lineaments that neither the catalogue nor the state compilation records (falsification-first, not run)

* **Data:** `data/external/sgmc_faults_100m.tif` (82,151 positive cells, measured) versus the catalogue
  (60,988 cells) — the set difference is 78,433 cells (measured).
* **Physical signature:** ridge coherence high **and** distance to the nearest SGMC fault large.
* **Why it is contrarian:** the recorded live scores say the opposite of "chase SGMC" — coverage of the
  SGMC-only population is the *most negative* correlate of the recorded score
  (`cov_sgmc` ρ = −0.699, p = 0.011, n = 24; the SGMC-designed raster `sgmc44k_00512.tif` scored
  0.0512, OWNER-RECORDED). So the only defensible use of SGMC is subtractive: use it to *remove*
  lineaments from the emission, on the theory that a structure already captured by a state compilation
  is likely to be old/inactive and is not what the expert panel added.
* **Difference from existing work:** the registry's `sgmc-gap`, `sgmc44k`, `sgmc-off-catalogue-44k` and
  GEMSDOE41's `statemap-lattice` all *emit on* SGMC-derived structure; none uses SGMC as an exclusion.
* **Measured this session (the falsification already ran):** scoring all 24 owner-labelled rasters against
  an off-catalogue proxy truth (SGMC pixels ≥ 3 px from the mapped catalogue) with the official metric gives
  Spearman(live score, proxy DTversky) = **−0.665 (p = 0.0004, n = 24)**; the raster that covers the proxy
  best (`sgmc44k`, proxy 0.56) has one of the worst live scores (0.0512), and the family's best live raster
  covers the proxy worst. So "chase the state compilation" is not merely risky, it is **measured backwards**;
  the only surviving form of H4 is the *subtractive* one (use SGMC distance as an exclusion term), which is
  still untested. Receipt: [`../../evidence/offcat_instrument_check.json`](../../evidence/offcat_instrument_check.json).
* **Cost:** low. **Expected gain:** PROJECTION, unknown. This is the cheapest falsifiable test left, and its
  additive form is now falsified.

## H5 — 1 m DEM scarp curvature (BLOCKED — not proposed as viable)

* **What it would add:** the expert panel almost certainly worked with sub-metre topography; a 1 m DEM
  resolves scarplets that the 100 m `det_elev` band smears out.
* **Named source:** `1m_DEM_links.csv` on the DrivenData data tab
  (<https://www.drivendata.org/competitions/306/competition-doe-gems/data/> — verified to redirect to
  login), whose entries point at USGS 3DEP / OpenTopography hosts that this sandbox's egress allow-list
  cannot reach. **Obtainability check: FAILED here.** No claim is made that it would help.
* **Cost:** high (tens of GB of tiles, registration to the 100 m grid, scarp morphology).

---

## Ranking summary

| Rank | Hypothesis | Overlap with the 695-raster registry | Data obtainable here | Cost | Expected gain |
| ---: | --- | --- | --- | --- | --- |
| 1 | **H1** coherence consensus, dotted, budget-capped | **FAIL / STOP:** 86 literal forward 3-px overlaps above 0.70 on the historical 695-raster scan; the later 696-raster index was not checked, and no surface-stage receipt is available | yes (official stack) | 20 s (historical) | BUILT / HELD for inspection only; 0.24–0.29 remains a low-confidence PROJECTION, not a score; no slot authorized |
| 2 | **H2** seismicity × strain lineaments | not built | yes, but ComCat is blocked | medium | PROJECTION |
| 3 | **H4** doubly-unknown (SGMC-exclusion; additive form **falsified** this session, ρ −0.665) | not built | yes, local | low | PROJECTION |
| 4 | **H3** official map-scale completeness prior | not built | attribute table local, **join unresolved** (IR-59J-02) | medium | PROJECTION |
| 5 | **H5** 1 m DEM scarps | not built | **no (blocked)** | high | unknown |

No weekly slot was spent. Validation on the shared holdout is reported for every arm but, per the
calibration in `why-02778.md`, it is no longer used as a promotion gate.
