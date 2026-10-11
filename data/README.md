# Data manifest — current availability, bridge pins, and sources

**Session 7 (2026-10-10) update:** `training_features.tif` was restored with `scripts/prepare_data.py --fetch`. The SHA256 matches the pin (`4371c82e…23bc5`, 19 bands), and the band cache was regenerated with `--cache-bands`. The bridge hash proves byte identity with the bridge only; it is not an organizer receipt. The earlier "absent" statement is superseded. The rest of this file describes the Session-6 audit.

**Earlier audit (Session 6):** no new data were downloaded, restored, or prepared in that audit. The competition
`training_features.tif` was absent from that checkout. Earlier receipts report a 19-band
bridge file, but that historical receipt does not establish present availability or
independent official-origin authentication.

The independent [preflight](../evidence/preflight_anatomy.json) confirms the literal overlap STOP; restored data do not authorize a new candidate.

## Competition-grid bridge files

The listed byte counts and hashes describe checked-in bridge copies or an expected historical
bridge file. Matching a bridge hash establishes byte identity with that bridge, not an
independent receipt from DrivenData.

| File | Bytes | SHA256 | Current interpretation |
|---|---:|---|---|
| `data/bridge/labels.tif` / `data/official/labels.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` | Bridge copies are byte-identical; not independently authenticated against a fresh official download. |
| `data/bridge/existing_faults.tif` / `data/official/existing_faults.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` | Same bridged bytes as labels; the known-fault mask is pixel-exact, not dilated for scoring. |
| `data/bridge/sample_submission.tif` / `data/official/sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` | Bridged grid header used for local metadata checks; not an organizer-upload receipt. |
| `data/bridge/training_features.tif` (expected) | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` | Historical bridge receipt reports 19 bands; the file is not present now and the hash is not a current local verification. |

The DrivenData data tab was observed behind login. No authenticated data download or
organizer data/submission receipt is available in this audit. The bridge's
`example_submission.tif` contains both 0 and 1 while the public task description frames the
sample as predicting fault absence; do not claim the bridge example's values are verified
identical to an authenticated official sample. The bridged header and footprint are used for
local grid compatibility checks only. EPSG:32611 and 100 m resolution match the public
specification.

## Historical external files and interpretation

Files under `data/external/` are local historical inputs with recorded hashes. Their presence
does not make them organizer labels or authorize another experiment.

| File | Bytes | SHA256 | Scope / limitation |
|---|---:|---|---|
| `sgmc_faults_100m.tif` | 213,034 | `26d142c4c93282cd94f6950ab96f22aeff59fbbea523d43d662e76fa1b161b5c` | State Geologic Map Compilation raster; historical auxiliary layer only. |
| `derived_sgmc_faults_100m.tif` | 198,602 | `643cbe992ef4ba37588fb469163ed8291e3ceb23d6c1f78a3cfaa462430c2da0` | Historical off-catalogue proxy derived from SGMC minus the mapped union; not hidden truth. |
| `trace_segments_utm11.csv` | 5,283,577 | `c5edf9df3e0e1413c52784884fc38e5b9ca54627b27c6f8baf7713e908f48513` | INGENIOUS/QFault trace export; projection and rasterization provenance remain limited. |
| `qfault_attributes.csv` | 5,786,456 | `3b8745f0086a57b8255f6679afa1bf1168c10b5573491666c75cf53217029507` | INGENIOUS/QFault attributes; field availability and missingness require care. |

Prior SGMC use is historical and is not current-code validation, an organizer score, or
clearance to build or submit. This audit did not use SGMC to train, fit, allocate, or score a
candidate. Historical feature descriptions are metadata, not trusted semantics; disputed
geophysical bands are not evidence of a verified fault feature in this audit.

## Free official sources checked for future anatomy research

These catalogue records ground possible future data availability only. They do not establish
coverage of the contest AOI, grid alignment, usable raster content, or an expected score gain.
No GeoDAWN data were downloaded, clipped, registered to the contest grid, or used.

- [USGS GeoDAWN airborne magnetic and radiometric surveys](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and), linked to [ScienceBase DOI 10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ): public catalogue/CC0 metadata checked; spatial coverage and usable-file compatibility remain unchecked.
- [Glen & Earney (2024), GDR submission 1591](https://gdr.openei.org/submissions/1591): page identifies public access and CC BY 4.0 and links to the USGS ScienceBase record; overlap with the contest grid remains unchecked.
- [Grauch (2002), USGS Open-File Report 2002-384](https://pubs.usgs.gov/publication/ofr02384): official publication record for high-resolution aeromagnetic data in Dixie Valley; no assertion is made that its raster covers or aligns with this competition AOI.
- [INGENIOUS GDR compilation, DOI 10.15121/1881483](https://gdr.openei.org/submissions/1391): public record reviewed; licensing and attribute-specific suitability must be checked before any future use.
- [Competition rules](https://docs.nlr.gov/docs/fy26osti/96647.pdf): entrant eligibility, third-party licensing, AI-use disclosure, and reproducibility requirements still need review before a future submission.

The free-source review did not require a binary download. External downloads from official hosts are
outside this audit's actions. If a future ranked hypothesis depends on new data, first re-check
official source accessibility, license, spatial coverage, CRS, and contest-grid overlap under an
authorized budget.
