# Ranked geological hypotheses for a future, gated H59 run

**Status: research plan only; none of these new candidates was run or promoted in this session.** The H59-I holdout budget is exhausted (3 experiments / 2 hours), the historical holdout is not protocol-qualified, the latest known registry has a literal universal-overlap blocker, and no submission slot is authorized. Ranking is qualitative **research priority**, not a DTI forecast. No numerical gain is projected.

The active modeling lane remains a presence-only point-process intensity → 1 km expected-new-fault density zoning → holdout-derived break-even lattice. Every geological hypothesis below must be tested inside that lane, not by importing a sibling placement map. The requested independent target-group background is a prerequisite, not optional decoration: H59-I did not implement it.

## Prerequisite: establish a defensible target-group background

Phillips et al. (2009) motivate correcting presence-only sample-selection bias. Here, a target-group background must be built from **independent, spatially matched, same-effort non-target mapping records**—not by using the response fault catalogue both as presences and as its own background. The current pins contain the target fault traces, QFault attributes, a state fault compilation, and the 19-band GeoDAWN stack; they do not contain a suitable independent mapper-effort inventory. The official GeoDAWN acquisition blocks and USGS location-reliability/road covariates are also not present in the pinned bundle. ScienceBase is outside this sandbox's host allow-list, so availability has not been verified beyond the listed DOI. Until that source is obtained, hashed, grid-checked, and made split-safe, no result may be called target-group-background corrected.

## Ranking

| Rank | Hypothesis | Expected research value (qualitative, not DTI) | Cost | Registry overlap / stop risk |
| ---: | --- | --- | --- | --- |
| 1 | Independent geothermal-spring / discharge conduits | Potentially high information because it is an external observation channel; very uncertain until coverage and effort are audited. | Medium–high; data acquisition and point provenance are the bottlenecks. | No spring-specific map was identified in the saved 696-raster public-owner inventory, but that inventory is not organizer-complete. Not a uniqueness clearance. |
| 2 | Persistent basin-margin density boundaries confirmed by independent gravity and magnetic edges | Medium potential under alluvial cover; geology is directly represented in the supplied grid. | Low–medium; transforms and holdout rebuild are inexpensive. | High overlap risk: multi-physics lineament consensus and gravity/magnetic edge lanes are already present in sibling repositories (e.g. H33-A). |
| 3 | Fault-network relay / secondary-strand density conditioned on observed relative strike and length | Medium potential if the training segments show a reproducible distribution; must not hard-code a textbook angle. | High; segment attributes need a validated key join and fold-specific geometry. | High overlap risk with analog/stepover submissions (e.g. GEMSDOE33 H33D) and catalogue-distance lanes. |
| 4 | Geodetic strain localization interacting with mapped basin-edge structure | Low–medium; potentially useful for active structures but the supplied strain fields are coarse and confounded. | Medium. | Medium–high overlap with existing geodetic and structural-prior submissions; exact registry scan still required. |
| 5 | Relocated earthquake lineaments as a point-process covariate | Potentially high for active faults, but event completeness and induced/aftershock controls dominate uncertainty. | High; an independently sourced, spatially complete catalogue and provenance are required. | Some earthquake-lineament work appears in the sibling history (H33-E); current local inventory does not establish novelty. |

### Rank 1 — Independent geothermal springs / discharge corridors

- **Layers:** independent hot/warm-spring locations with source, date, temperature/chemistry and positional uncertainty; supplied `cond_surf` (band 17), `depth_to_base_surf` (band 15), isostatic gravity / gravity horizontal gradient (bands 13/18), magnetic total field / horizontal gradient (bands 14/3), and detrended elevation / slope (bands 12/19) as conditioning or confounder layers. The official stack is at 100 m; the point inventory must not be upsampled and mistaken for 100 m ground truth.
- **Physical signature:** spring clusters or aligned discharge points may mark permeable fault conduits, including structures concealed by basin fill. Support should persist after controlling for terrain, basin depth, mapped hydrothermal areas and observation effort.
- **Rationale:** unlike a fault-density transform, a genuinely independent spring inventory can add information where the fault catalogue has no trace. A relationship with active faults is a hypothesis, not a guaranteed consequence.
- **Named non-fault mimic:** valley-floor groundwater discharge, playa margins, volcanic heat and basin-margin lithologic contacts can align springs without a new surface fault.
- **Overlap:** the saved 696-raster owner-repository inventory did not reveal a spring-point predictor, but it is neither organizer-complete nor evidence of novelty; the strict surface and final-dot literal gate remains mandatory.
- **Data / obtainability:** a USGS Science Data Catalog listing for regional hot-spring chemistry is noted in `docs/research/sources.md`; download, license, coverage and coordinate quality are **not verified**. The sandbox can reach GitHub, npm and PyPI hosts only, not ScienceBase. Do not synthesize a dataset from an unverified map.
- **Expected value / cost:** highest priority for information independence, but expected DTI gain is unknown. Medium–high cost: acquire the official table, audit completeness, deduplicate locations, quantify effort/uncertainty, then build a split-safe feature.
- **Validation:** pre-register source and covariates; hide whole fault components with a spatial buffer; build every catalogue-derived term from visible faults; fit on a training-visible estimate of prevalence; run a per-feature leakage canary (AUC > 0.90 is leakage until explained); compare pooled HOLDOUT-DTI and 95% CI against a same-budget control. No slot decision before that and both literal registry gates.

### Rank 2 — Basin-margin density boundaries with cross-physics confirmation

- **Layers:** isostatic gravity anomaly and horizontal gradient (bands 13 and 18), total magnetic intensity and horizontal gradient (bands 14 and 3), tilt derivative (band 6), basement-depth surface (band 15), and detrended elevation / slope (bands 12 and 19).
- **Physical signature:** a persistent, spatially coherent density/susceptibility boundary or basin-margin step that aligns across independent physics and remains stable across modest scale changes. Use the edge as an intensity covariate, not as a copied line/placement template.
- **Rationale:** buried basin-bounding structures may have weak topographic expression but a geophysical contrast.
- **Named non-fault mimic:** lithologic contacts and survey levelling / flight-line artefacts can produce the same aligned gradients.
- **Overlap:** high. Multi-physics lineament consensus and gravity/magnetic edge maps are already represented in the accessible sibling-repository history. A high rank or visual resemblance is not novelty; check candidate surface and final dots against the literal gate first.
- **Data / obtainability:** all named raster layers are present in the pinned 19-band feature stack. Organizer origin is not independently authenticated from the public bridge. GeoDAWN survey-block metadata are not pinned.
- **Expected value / cost:** medium geological plausibility, but lower priority than an independent spring channel because of duplication risk. Low–medium cost.
- **Validation:** fold-specific visible-fault features; holdout-derived 1 km expected-new-fault density and utility table; holdout comparison to an effort-only and uniform control; no hard-coded edge thresholds chosen after seeing withheld labels.

### Rank 3 — Relay / secondary-strand intensity from observed fault geometry

- **Layers:** `trace_segments_utm11.csv` plus validated QFault attributes, geodetic shear-rate and dilatation bands (7/8), geodetic second-invariant band (4), and geophysical edge planes (3/6/18/19). Use recorded slip sense only where its join is verified.
- **Physical signature:** a within-segment distribution of secondary-strand distances, relative strikes, and density that is learned from withheld segments. Fault length may be evaluated as a displacement proxy, but must be treated as uncertain and calibrated rather than assumed. No Riedel angle is hard-coded.
- **Rationale:** damage-zone/relay architecture could concentrate additional mapped strands near major systems and at transfer zones.
- **Named non-fault mimic:** the mapmaker's habit of tracing parallel lithologic contacts, roads, or survey artefacts; near-catalogue proximity can also look predictive because the holdout is built from the same catalogue.
- **Overlap:** high with existing analog/stepover and fault-tip work (including GEMSDOE33 H33D) and other catalogue-proximity fields. Do not treat a different implementation as a new lane.
- **Data / obtainability:** segment and attribute CSVs are hash-pinned, but the historical `record_id` join to FCODE/MAPSCALE was not valid; positional join evidence is documented in `REMAINING_WORK.md`. Do not use those attributes until a stable key is independently verified.
- **Expected value / cost:** medium, conditional on a measured holdout distribution; high implementation cost because of joins, topology, and fold-specific rebuilding.
- **Validation:** fit distance/relative-strike distributions on visible training segments; measure withheld nearest-visible distributions independently; shrink the 1 km budget if few withheld positives lie in the learned zone. Compare against a same-budget random/effort-only control.

### Rank 4 — Geodetic strain localization plus basin structure

- **Layers:** geodetic shear-rate (7), dilatation-rate (8), second invariant (4), plus gravity/magnetic edges and basement depth (13/18/14/3/15).
- **Physical signature:** localized strain gradients co-located with an independently supported structural boundary; a broad regional strain maximum by itself is not a fault trace.
- **Rationale:** active deformation could distinguish mechanically active structures from old contacts, particularly where combined with a buried basin edge.
- **Named non-fault mimic:** geodetic interpolation seams, block-adjustment boundaries, compaction/subsidence and geothermal-fluid deformation.
- **Overlap:** medium–high because prior repository work already uses geodetic strain as a prior and the current intensity regime includes it.
- **Data / obtainability:** supplied geodetic bands are available in the pinned raster. The underlying product, error covariance and effective resolution need provenance checks before a stronger interpretation.
- **Expected value / cost:** low–medium, with high confounding risk; medium cost.
- **Validation:** feature-alone canary, ablation against geophysics and mapping effort, spatial-block holdout; keep only interactions that improve pooled DTI beyond uncertainty and pass the gates.

### Rank 5 — Relocated earthquake lineaments

- **Layers:** an independent, time-stamped relocated seismicity catalogue (not the fault response labels), event uncertainty, and supplied geodetic / magnetic / gravity covariates for confounder controls.
- **Physical signature:** persistent linear event clusters along active structures after removing aftershock sequences and accounting for magnitude completeness.
- **Rationale:** seismicity is an independent physical observation of active faulting and might identify structures with no mapped surface trace.
- **Named non-fault mimic:** aftershock clouds, induced seismicity, geothermal operations, volcanic events and spatially varying station/catalog completeness.
- **Overlap:** medium. Sibling notes indicate H33-E explored seismicity but lacked data; no novelty conclusion follows from that absence.
- **Data / obtainability:** USGS ComCat / relocated event service access was not available in this sandbox and no event table is pinned. Coverage, licensing and completeness remain unverified.
- **Expected value / cost:** potentially useful but uncertain; high cost due to completeness and sequence controls.
- **Validation:** only after source audit; train on visible faults, withhold whole components and buffered spatial blocks, assess canary and event-catalogue completeness; no external score projection.

## Mandatory decision gates for any future candidate

1. Re-establish a complete dated public-owner inventory and preserve organizer-completeness limits. Run literal Spearman and forward 3-px overlap against **every** accessible raster on the candidate surface before placement and the final dots after placement. A single `rho > 0.90` or forward overlap `> 0.70` is a duplicate/STOP; do not use representation-based exemptions. A prior session asserted a whole-footprint witness, but its TIFF and verification receipts are absent from this checkout, so that universal-blocker claim is unverified here; the 85 retained historical overlap firings still require STOP.
2. Reuse shared `evaluate_holdout.py`, `submission_writer.py`, and feature caches. Use whole components with a buffer; rebuild every catalogue-derived feature and training prevalence from visible faults only; mask visible fault pixels exactly; score pooled DTI α=0.2, β=0.8, 300 m triangular kernel.
3. Implement target-group background only from an independently documented same-effort source, not from the target catalogue. Run each feature-alone leakage canary; AUC > 0.90 means leakage until explained.
4. Label every metric HOLDOUT-DTI with evaluator, withheld-positive count and 95% CI, or ORGANIZER-CONFIRMED only when a matching organizer receipt exists. Projections remain projections.
5. The three-experiment / two-hour budget for the historical H59-I run is exhausted. A separate selector decision is required for any future slot; no weekly slot is used or recommended here.
