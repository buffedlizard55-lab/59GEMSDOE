# Sources — verification log

Every row says how it was checked. "fetched" means the page text was read through the fetch tool
on 2026-10-11. "search" means it appeared in a search result with the quoted text. "blocked" means
the sandbox could not reach it, so it is **not** verified and must not be relied on.

## Competition (official)

| Source | URL | Check | Used for |
| --- | --- | --- | --- |
| DrivenData competition page | <https://www.drivendata.org/competitions/306/competition-doe-gems/> | fetched | overview, prize structure |
| Problem description (metric, format) | <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/> | fetched | the DTI formula, kernel R = 300 m, CRS/format requirements |
| Official rules PDF (NLR, FY26 OSTI 96647) | <https://docs.nlr.gov/docs/fy26osti/96647.pdf> | fetched | prize phases, eligibility, submission process |
| Leaderboard | <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/> | snapshot saved 2026-10-11 in `evidence/leaderboard_snapshot_2026-10-11.json` | participant-level public values #1 0.3774, #8 DARD 0.3195, #22 extradr19 0.2778; no file-linked submission-page receipt captured |
| Data download page | <https://www.drivendata.org/competitions/306/competition-doe-gems/data/> | fetched; redirects to login | confirms login is required |
| Staff clarification on masking of known faults (2026-09-16) | <https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516> | fetched | "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation" |
| Reference solution (GitHub) | <https://github.com/drivendataorg/gems-prize-reference-solution> | cloned; notebook read | grid facts: EPSG:32611, 100 m, width 3292, height 3730; output written with labels.tif CRS and transform; sigmoid outputs in [0,1] |
| Dropbox data links | listed in the project brief | **blocked** from sandbox (TLS failure) | official-supplied mirrors; not verified here |

## Data source (USGS, official)

| Source | URL | Check | Used for |
| --- | --- | --- | --- |
| GeoDAWN airborne magnetic and radiometric surveys | <https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and> | fetched | DOI 10.5066/P93LGLVQ; CC0; four acquisition blocks (Winnemucca, Fallon, Hawthorne, Tonopah); radiometric and magnetic products; Area 1 rank-1 and Area 2 lower-resolution flight specs |
| USGS Geothermal Resource Investigations Project | <https://www.usgs.gov/centers/gmeg/science/geothermal-resource-investigations-project> | search | context: GeoDAWN is part of this program |
| Geothermal GIS coverage of the Great Basin (USGS publication) | <https://www.usgs.gov/publications/geothermal-gis-coverage-great-basin-usa-defining-regional-controls-and-favorable> | search | high-temperature systems prefer young NE-trending faults (weights-of-evidence); extensional systems track high strain rate |
| USGS Science Data Catalog, geothermal search | <https://data.usgs.gov/datacatalog/search?otherKeyword=%5B%22geothermal+resources%22%5D> | search | lists a 1974 hot-spring chemistry compilation (Mariner et al., OFR 74-1066) and a regional geophysical compilation for the Great Basin. **Download not verified.** |

## Method literature (checked)

| Citation | Link | Check | Used for |
| --- | --- | --- | --- |
| Phillips, Dudík, Elith, Graham, Lehmann, Leathwick, Ferrier (2009). Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. *Ecological Applications* 19(1):181–197. DOI 10.1890/07-2153.1 | <https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/07-2153.1> | search (publisher page) | target-group background to correct sampling-effort bias |
| Fithian & Hastie (2013). Finite-sample equivalence in statistical models for presence-only data. *Annals of Applied Statistics* 7(4):1917–1939. DOI 10.1214/13-AOAS667 | <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-4/Finite-sample-equivalence-in-statistical-models-for-presence-only-data/10.1214/13-AOAS667.full> | search (publisher page) | MaxEnt and inhomogeneous Poisson point-process equivalence |
| Blakely & Simpson (1986). Approximating edges of source bodies from magnetic or gravity anomalies. *Geophysics* 51:1494–1498. DOI 10.1190/1.1442197 | cited by Mayhew (1987, *Rev. Geophys.*) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/RG025i005p00971>; method discussed in <https://www.sciencedirect.com/science/article/abs/pii/S0926985117307176> | search | horizontal-gradient maxima for body edges |
| Hermant et al. (TLS Geothermics), 50th Workshop on Geothermal Reservoir Engineering, Stanford, 2025 | <https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Hermant.pdf> | search | states that the distance "between USGS Quaternary faults and TLS fault label can be up to 400m". **This is a discrepancy between two maps, not a measured registration error.** See irregularities. |

## Competitor repositories (public GitHub; read for learning only, not copied)

| Repo | URL | Check | Notes |
| --- | --- | --- | --- |
| GEMSDOE32 | <https://github.com/buffedlizard55-lab/GEMSDOE32> | owner site/audit read at commit `3ae286af599fdf86dbb0f333591f8273efcee806` | The owner page labels H33-2-B2 UNSCORED and says no organizer score exists for any artifact in that repo; its audit's `predicted_leaderboard_dti` is a projection. The 0.2778 leaderboard participant-to-file mapping is unverified. See `leaderboard-analysis.md`. |
| GEMSDOE36 | <https://github.com/buffedlizard55-lab/GEMSDOE36> | cloned, hypotheses read | H1–H6 registry |
| GEMSDOE46 | <https://github.com/buffedlizard55-lab/GEMSDOE46> | cloned, hypotheses read | used GeoDAWN K/Th/U radiometric grids (not novel for us) |
| GEMSDOE | <https://github.com/buffedlizard55-lab/GEMSDOE> | listed | contains `data/bridge/` copies of competition files (see irregularities) |

## Blocked from this sandbox (not verified)

- DrivenData data downloads (login)
- Dropbox mirrors listed in the brief
- OpenEI submission 1391, ScienceBase, USGS ComCat earthquake service, GitHub Pages sites of the competitors
  (the competitor **repositories** are reachable on github.com; their Pages sites are not)
