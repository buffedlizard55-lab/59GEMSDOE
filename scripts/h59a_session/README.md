# H59-A session scripts (provenance only)

These are the exact scripts that produced
`docs/downloads/gems59-h59a-intensity-zone-breakeven-8c0c6142-zeros.tif`.

They are stored here rather than overwriting `scripts/build_candidate.py`,
`scripts/submission_writer.py`, `scripts/uniqueness_check.py` and
`scripts/finalize_emission.py`, because `main` already carried different,
newer versions of those filenames from the H59-I / H59-J sessions.

**These are provenance, not the live path.** The live, maintained versions of the
shared tools live in `scripts/` and `src/gems59/`. If you rerun anything here,
expect it to import the archived `*_h59a.py` modules, not `src/gems59/`.

| file | role |
|---|---|
| `build_candidate.py` | 2-fold segment-level holdout + (spacing × break-even bar) sweep |
| `finalize_emission.py` | budget-targeted τ bisection + separation audit |
| `submission_writer.py` | float32 GeoTIFF writer that re-reads and validates its own output |
| `uniqueness_check.py` | parallel-run drift gate (Spearman + chance-corrected 3 px lift) |
| `emit_h59a.py`, `intensity_h59a.py`, `holdout_h59a.py` | the modules those scripts imported |

Receipts: `evidence/build_h59a_with_trace.json`,
`evidence/emission_final_h59a.json`, `evidence/uniqueness_dots.json`,
`evidence/uniqueness_surface.json`, `evidence/uniqueness_h59a_vs_siblings.json`,
`evidence/leakage_channel_ablation.json`.
