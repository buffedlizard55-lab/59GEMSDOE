# Official & Verified Sources — Session H59-I2

Every competition rule, data artifact, and scientific reference used in Session H59-I2 is verified below with direct links and SHA256 checksums.

## 1. Competition Rules, Problem Definition & Reference Code
- **Problem Description & Metric (`alpha = 0.2, beta = 0.8, 300 m` triangular kernel):**
  - DrivenData Competition 306 Page 967: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
- **Competition Rules & Submission Format (`EPSG:32611`, `3730x3292`, `[0, 1]`, 3/week limit):**
  - DrivenData Competition 306 Page 968: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>
  - Official DOE / NREL Rules PDF (FY26 OSTI 96647): <https://docs.nlr.gov/docs/fy26osti/96647.pdf>
- **Official Reference Solution Repository:**
  - <https://github.com/drivendataorg/gems-prize-reference-solution>
- **Coordinate Reference System (`EPSG:32611` — WGS 84 / UTM zone 11N):**
  - <https://epsg.io/32611>

## 2. Geological & Geophysical Data Compilations
- **USGS GeoDAWN Airborne Magnetic & Radiometric Surveys (NW Great Basin, NV/OR):**
  - USGS Data Page: <https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and>
  - ScienceBase Data Release: <https://doi.org/10.5066/P93LGLVQ>
- **INGENIOUS Great Basin Structural & QFaults Compilation (`FCODE2023`, `MAPSCALE`):**
  - Geothermal Data Repository (GDR) Submission 1391: <https://gdr.openei.org/submissions/1391>
  - Committed field definitions in repository: `data/external/qfaults_8_README_fielddefinitions_qfaults_ingenious_nad83conus117_2023-06-27.txt`
  - Committed attribute table (`22,956` rows, SHA256 `3b8745f0…9507`): `data/external/qfault_attributes.csv`
  - Committed UTM 11N trace segments (`84,331` segments across `1,126` `record_id`s, SHA256 `c5edf9df…8513`): `data/external/trace_segments_utm11.csv`

## 3. Peer-Reviewed Point-Process & Fault-Mapping Literature
- **Sample-Selection Bias & Target-Group Effort Covariates in Presence-Only Models:**
  - S. J. Phillips, M. Dudík, J. Elith, C. H. Graham, A. Lehmann, J. Leathwick, & S. Ferrier (2009), "Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data," *Ecological Applications*, 19(1):181–197. <https://doi.org/10.1890/07-2153.1>
- **Equivalence of MaxEnt and Inhomogeneous Poisson Point Processes (IPP):**
  - W. Fithian & T. Hastie (2013), "Finite-sample equivalence in statistical models for presence-only data," *Annals of Applied Statistics*, 7(4):1917–1939. <https://doi.org/10.1214/13-AOAS610>
- **Fault Damage-Zone Scaling & Catalogue Positional Discrepancy (~400 m):**
  - H. M. Savage & E. E. Brodsky (2011), "Collateral damage: Evolution with displacement of fracture distribution and secondary fault strands in fault damage zones," *Journal of Geophysical Research: Solid Earth*, 116:B03405. <https://doi.org/10.1029/2010JB007665>
  - Hermant et al. (2025), positional discrepancy (~400 m) of regional fault compilations vs high-resolution geophysical/lidar traces (referenced in competition task prompt, motivating the $1\text{ km} \times 1\text{ km}$ block zoning resolution).
