# Candidate hypotheses — ranked, with overlap checks and obtainability

**Status: none validated.** No holdout exists yet (the data is blocked; see README section 5).
Nothing here may be submitted, and no slot may be spent, until a candidate beats the holdout best.
Expected-gain labels are qualitative judgement (`PROJECTION`), not measurements.

## Overlap check (what the registry already tried)

We read four public repos: GEMSDOE32, GEMSDOE36, GEMSDOE46 and the root GEMSDOE. The other repos
were not read, so a full registry scan is still owed.

| Idea | Tried by | Result as reported |
| --- | --- | --- |
| Tilt-angle zero-crossing lineaments | GEMSDOE32 (H62-1) | falsified; no gain on 4 blocks |
| Detrended-elevation curvature | GEMSDOE32 (H62-4) | `curv_s2` best channel |
| Geodetic strain as prior | GEMSDOE32 (H62-3) | used raw |
| Basement step × gravity | GEMSDOE32 (H62-2) | +3.5 % transform |
| Drainage-valley chains | GEMSDOE32 (H33-B) | rejected |
| 1 m LiDAR scarplets | GEMSDOE32 (H33-C), GEMSDOE46 | tried |
| Seismicity lineaments | GEMSDOE32 (H33-E) | data-blocked there |
| GeoDAWN K/Th/U radiometrics | GEMSDOE46 (H46-R11F-1) | part of shipped R11 |
| Multi-physics lineament consensus | GEMSDOE32 (H33-A) | rank 1 of their H33 set |
| Euler deconvolution lineaments | GEMSDOE40 (h8) | tried |
| Physics-informed networks | GEMSDOE36 (H5) | tried |

## Candidates

### H-1 (rank 1) — Thermal-spring alignment prior from an independent point inventory

- **Layers:** none of the 19 official bands. New input: hot-spring and warm-spring locations with temperature or chemistry.
- **Physical signature:** springs rise along permeable conduits, and fault zones are the main ones. Clusters and alignments of springs should follow structures that are unmapped in the catalogue, especially under alluvium where there is no surface trace.
- **Why it catches a fault the catalogue lacks:** the spring record is independent of the USGS fault database, so it can locate a structure the catalogue has missed. This is the one candidate here whose information source is not the fault catalogue.
- **How it differs from the registry:** none of the registry ideas uses spring locations. The closest is the drainage-valley chain (H33-B), which is also a valley-bottom signal. That is exactly the mimic, so it must be controlled.
- **Named non-fault mimic:** valley-floor groundwater discharge, playa margins and drainage lines produce spring alignments with no fault beneath them. Control by regressing on distance to drainage and on basin depth (band 15).
- **Evidence for the physics (verified):** a USGS publication reports that high-temperature systems follow young NE-trending faults and that extensional systems track strain rates. Source: <https://www.usgs.gov/publications/geothermal-gis-coverage-great-basin-usa-defining-regional-controls-and-favorable>. The same page documents a weights-of-evidence analysis.
- **Data obtainability:** a USGS Science Data Catalog entry lists a 1974 hot-spring chemistry compilation (Mariner et al., OFR 74-1066) and a regional geophysical compilation. Listing verified: <https://data.usgs.gov/datacatalog/search?otherKeyword=%5B%22geothermal+resources%22%5D>. **Download not verified** from this sandbox; the next step is a manual download of the official file and a sha256 record.
- **Implementation cost:** medium (point cleaning, deduplication, a distance-to-spring surface and a controlled regression).
- **Expected DTI gain:** unknown; could be high if springs mark unmapped faults, nil if they mostly mark valleys. *PROJECTION: unmeasured.*
- **Validation plan:** holdout on withheld segments only, with springs recomputed from visible data alone. The leakage canary applies: AUC above 0.90 on the spring feature alone means leakage until proven otherwise.

### H-2 (rank 2, demoted) — Gravity horizontal-gradient maxima (Blakely & Simpson 1986), with a magnetic coincidence rule

- **Layers:** isostatic gravity anomaly (band 13); total magnetic intensity (band 14) for coincidence.
- **Physical signature:** density contrasts at basin-bounding faults show up as ridges of the horizontal-gradient magnitude, and the ridge loci are the body edges. Blakely & Simpson (1986) formalised locating those maxima: *Geophysics* 51:1494–1498, DOI 10.1190/1.1442197. Citation chain verified through Mayhew (1987) and a Sciencedirect paper on the method.
- **Why it could catch a missing fault:** basin edges under alluvium often have no surface trace.
- **Overlap:** high with H33-A (multi-physics oriented-lineament consensus), which already combines physics. **This is a near-duplicate of a registry lane.** Running it separately risks the drift rule (rho above 0.90 or more than 70 % dot overlap). Treat as a component of H33-A, not a new lane.
- **Expected DTI gain:** low-to-medium, since it probably reproduces H33-A.
- **Cost:** low (a few hundred lines on a grid already provided).

### H-3 (rank 3) — A regional gravity compilation as an upgrade to the supplied gravity layers

- **Layers:** the supplied isostatic gravity (bands 5, 11, 13, 18) is the baseline. The candidate is an independent regional gravity grid for the Great Basin, which the USGS Science Data Catalog lists as "regional geophysical maps of the Great Basin". Listing verified: <https://data.usgs.gov/datacatalog/search?otherKeyword=%5B%22geothermal+resources%22%5D>.
- **Physical signature:** sharper density boundaries from better long-wavelength gravity, which feed the same edge tests as H-2.
- **Why it could catch a missing fault:** the same as H-2, but with a different and possibly better gravity source. It is data-driven, so it is less likely to just reproduce a registry transform.
- **Obtainability:** **not verified** (listing only). Datum and regridding to EPSG:32611 have not been checked.
- **Expected DTI gain:** unknown; probably small unless the new grid is clearly better over the study area.
- **Cost:** medium to high (datum, resampling, edge tests).

## Ranking summary

| Rank | Candidate | Overlap with registry | Data obtainable (verified here) | Cost | Expected gain (PROJECTION) |
| ---: | --- | --- | --- | --- | --- |
| 1 | H-1 thermal springs | low | listing only; download not verified | medium | unknown |
| 2 | H-2 gravity maxima | **high** (H33-A) | yes (supplied bands) | low | low–medium |
| 3 | H-3 regional gravity grid | low | listing only; not verified | medium–high | unknown, likely small |

## What happens next

1. Place the organizer data (README section 5). No candidate can be tested without it.
2. Verify the H-1 download and record its sha256, or drop H-1 and move to H-3.
3. Run the binary baseline on the holdout first, so a candidate's gain has something to be measured against.
4. Gate: a candidate is promoted only if its holdout DTI beats the holdout best, with its 95 % CI, and passes the leakage canary and the uniqueness check.
