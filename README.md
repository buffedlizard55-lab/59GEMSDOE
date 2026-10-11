# 59GEMSDOE — H59-I intensity lane

**Status (2026-10-11 UTC): NEGATIVE / STOP.** The existing GeoTIFF is locally format-valid and **download-valid for inspection only**. It is **not cleared to submit**. The final-dot registry audit recorded 85 literal >70% forward-overlap firings on its 695-raster snapshot, so the standing rule says duplicate/STOP; the latest saved 696-raster inventory was not compared against this candidate. The historical pre-placement surface gate was not run. No slot was used or authorized.

- **Existing research artifact:** [`downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif`](downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif), SHA256 `ed1b767f75804048382fc8baf61ff1edad6e9fd7b5b214def715ba5a1e290f6d`. It passes local GeoTIFF/range checks, but registry failure overrides that validator PASS. Do not upload it.
- **Historical local holdout (not a competition score):** **HOLDOUT-DTI 0.082161 [0.069542, 0.096158]**, evaluator `gems57-pooled-hide-v2`, seed-20 buffered whole-component draw, 11,321 withheld positives, α=0.2, β=0.8, 300 m triangular kernel. It is below the comparable local holdout reference 0.135204 [0.119140, 0.153036]. The old case-control prior used full-catalogue prevalence; the runner is corrected to visible-only prevalence but was not rerun.
- **Requested method gap:** the Phillips et al. (2009) independent **target-group background is not implemented**. The historical model used uniform background plus catalogue-derived density/relief proxies; no pinned, independent same-effort non-target mapping inventory is available. Do not describe the historical result as target-group corrected.
- **Strict registry:** [`evidence/uniqueness_h59i.json`](evidence/uniqueness_h59i.json). Literal overlap hits are operative; representation-aware diagnostics cannot clear them. Historical rho `0.176` is withdrawn because the old screen used stride-3 sampling with incorrect tie ranking. Exact rho was not rerun; no uniqueness claim is made.
- **Run card:** [`evidence/run_card.json`](evidence/run_card.json). It records negative verdict, both `download-valid` and `cleared-to-submit` status, all limitations, 0 slots used, and the 140-character submission note (for reference only; not authorized for use).
- **Three review passes:** [`docs/review_passes.md`](docs/review_passes.md). **Research ranking:** [`docs/research/hypotheses.md`](docs/research/hypotheses.md). **Remaining work:** [`REMAINING_WORK.md`](REMAINING_WORK.md).

## Score provenance: 0.2778 and 0.3195

The saved 2026-10-11 public-board snapshot lists 0.3195 for DARD at rank 8 and 0.2778 for extradr19 at rank 22. These are participant-level public leaderboard values, not file-linked submission-page receipts. The 0.2778-to-GEMSDOE32 H33-2-B2 TIFF attribution remains unverified; the owner-maintained GEMSDOE32 page calls that file **UNSCORED**, and its audit's `predicted_leaderboard_dti` is a **projection, not a score**. See [`docs/research/leaderboard-analysis.md`](docs/research/leaderboard-analysis.md).

No score gain is claimed. H59-I is a negative, exploratory local result; the historical experiment budget (3 experiments) is exhausted. No weekly competition slot is used or recommended. Promotion to a real slot is a separate selector decision, and currently there is no candidate clearance to present to that selector.

## Reproducible, non-submitting data setup

The current pinned bridge inputs and ignored feature caches can be restored with:

```bash
./.venv/bin/python scripts/prepare_data.py --fetch --cache-bands
```

This verifies hashes against the public bridge pin; it is not an independent organizer provenance receipt. Cache files live under ignored `.cache/` and `out/`. The training feature stack is 19 bands, 3730×3292, EPSG:32611, 100 m pixels; its SHA256 is `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5`.

**Do not run the holdout or build commands to produce a submission until the registry and method blockers are resolved.** `scripts/run_intensity_lane.py` now gates every candidate surface and dot field using the shared literal registry checker; `scripts/build_submission_intensity.py` also fails closed on the holdout result, surface gate, and final-dot gate. Neither command spends a competition slot.

## Project principles

- **Maximize P(Win):** separate organizer receipts from public-board context, owner reports, projections and local HOLDOUT-DTI; do not claim unsupported improvement.
- **Own the Outcome:** record the exact failures; do not silently relax a gate; keep the download status distinct from submit clearance; preserve negative results and slots.
- Read [`TASK_PROMPT.md`](TASK_PROMPT.md) and this README before continuing. The complete standing brief and historical score ledger remain in `TASK_PROMPT.md`; current status at the top of this README takes precedence over stale historical statements.
