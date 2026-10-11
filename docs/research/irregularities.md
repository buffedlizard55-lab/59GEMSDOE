# Irregularities Log — Session H59-I2

Every irregularity encountered in the template, data, or previous session is documented below with its diagnosis and verification receipt.

## Resolved in Session H59-I2

### `IR-59-LOCO-01` — Presence Self-Distance Degeneracy (`edt(~visible) == 0`)
- **Symptom:** In Session H59-I, training presences were `visible` catalogue pixels where `edt(~visible) == 0` identically, preventing any structural host-distance or stepover covariate from entering the presence-only MaxEnt/Poisson GLM and limiting holdout DTI to `0.082161`.
- **Resolution:** Implemented `loco_structural_frame(grid, visible, root)` in `src/gems59/intensity.py` using `two_host_frame(visible)` + `cKDTree` over `visible` pixels. Every pixel on a visible fault component $C_1(x)$ is assigned its Leave-One-Component-Out (LOCO) distance $d_1(x) \leftarrow d(x, C_2(x))$ (`10th/50th/90th pct = 3.16 / 9.00 / 29.97 px`) and secondary host $d_2(x) \leftarrow d(x, C_3(x))$ (`7.00 / 16.64 / 45.28 px`), matching withheld fault distances (`4.00 / 9.06 / 25.55 px`) strictly from `visible` faults.

### `IR-59-LOCO-EFFORT` — Self-Component Effort Density Contamination
- **Symptom:** Evaluating 5 km (`51x51`) and 10 km (`101x101`) fault-density boxcar windows on a training presence pixel $x \in C_1(x)$ includes $C_1(x)$'s own pixels (`~51 px` in a 5 km box), inflating the training LOCO AUC of `e_dens_5km` to `0.9032 > 0.90`.
- **Resolution:** Subtracted $\min(|C_1(x)|, L_{\text{win}}) / L_{\text{win}}^2$ on `visible` pixels (`e_dens_5km_loco`, `e_dens_10km_loco`), reducing training LOCO AUC to `0.6408` and `0.6009`. All 31 covariates pass the `< 0.90` leakage canary on both holdout (`max AUC = 0.7715`) and training LOCO (`max AUC = 0.7439`).

### `IR-59-LATTICE-MOD10` — Modulo-10 Block Lattice Wraparound Collision Bug
- **Symptom:** Session H59-I's `tile_blocks` computed `ry = (ay + np.arange(0, 10, s)) % 10` inside each $10 \times 10$ (`1 km`) block. For $s = 3$ and anchor $a_y = 8$, `(8 + [0, 3, 6, 9]) % 10 = [8, 1, 4, 7]`, placing dots at row `7` and row `8` (`1 px` apart — `100%` redundant 300 m kernel overlap!) and colliding across adjacent 1 km block boundaries.
- **Resolution:** Fixed `tile_blocks` to generate non-wrapping offsets `ay + k*s in [0, block)` and implemented `tile_blocks_corridor` to enforce exact Euclidean spacing $s_c$ across 1 km block boundaries via greedy 300 m triangular-kernel expected-reward maximization (`E[w_300m * lambda]`).

### `IR-59-EFFORT-02` — USGS/INGENIOUS QFaults Effort Attribute Join
- **Symptom:** Session H59-I left `data/external/qfault_attributes.csv` (`22,956` rows) and `data/external/trace_segments_utm11.csv` (`84,331` segments) unjoined.
- **Resolution:** Verified that `record_id` (`0..22,922`, `1,126` distinct values in `trace_segments_utm11.csv`) is the exact 0-based row index of `qfault_attributes.csv`, matching `60,988 / 60,988` (`100.0%`) of mapped catalogue pixels in `existing_faults.tif`. Joined `FCODE2023` (`1=accurate: 714`, `2=approximate: 331`, `3=inferred/concealed: 81`) and `MAPSCALE` (`1:250,000: 846`, `1:100,000: 280`) into `qfaults_effort_rasters`. Note: external USGS ScienceBase acquisition-block polygons (`DOI 10.5066/P93LGLVQ`) and Census TIGER roads remain outside the sandbox egress allowlist.

### `IR-59-DIAG-02` — Padded-Zero Block Decile Tie Collapse
- **Symptom:** Evaluating `decile_calibration` across all $373 \times 329 = 122,717$ padded bounding-box blocks (of which `70,408` lie outside the rotated polygon footprint with `n_b == 0`) collapsed Deciles 1–5 in Session H59-I.
- **Resolution:** Implemented `decile_calibration_blocks` over the `52,309` valid footprint blocks (`~5,231` blocks per decile), yielding strictly monotone withheld counts across all 10 deciles (`59 -> 112 -> 175 -> 252 -> 424 -> 527 -> 913 -> 1,448 -> 2,649 -> 4,762` withheld pixels; `78.25%` in D8–D10).

## Inherited & Disclosed Scope Conditions

### `IR-57-REPR-01` & `IR-59-UNIQ-01` — Footprint-Saturating Peers on Unrestricted 3 px Overlap
- **Diagnosis:** Continuous probability rasters (`5,167,373` positive pixels, `100%` of footprint) and footprint-wide 5–6 px lattices (`13GEMSDOE:r13-lattice-s5_v2`, `206,895` dots whose 3 px Euclidean halo covers `99.87%` of the footprint; `13GEMSDOE:13gems-composite-plus`, `380,365` dots covering `83.49%` of the footprint) force any non-empty candidate to have unrestricted forward 3 px overlap `> 0.70`. Against all `487` peers of `<= 77,727` dots (plus Session H59-I), our promoted raster `gems59-h59i2-loco-ipp-corridor-20261011T000000Z.tif` achieves max forward 3 px overlap `0.6473 <= 0.70` (`0.5722 <= 0.70` vs Session H59-I), pre-placement surface Spearman `0.8568 <= 0.90`, final dot Spearman `0.1655 <= 0.90`, and Jaccard `0.0585 <= 0.50`.
