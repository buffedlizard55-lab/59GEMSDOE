# Study of `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros` (GEMSDOE32, 0.2778) & Leaderboard Analysis

## 1. Provenance & Evidence Labels
- **Public Leaderboard Snapshot (`ORGANIZER-CONFIRMED`, `evidence/leaderboard_snapshot_2026-10-11.json`, captured 2026-10-10T20:40Z):**
  - Rank 1: `xiaofanhu` — **`0.3774`**
  - Rank 8: `DARD` — **`0.3195`**
  - Rank 22: `extradr19` — **`0.2778`**
- **Portfolio Attribution (`OWNER-REPORTED`, not organizer-receipted per file):**
  - `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif` (`GEMSDOE32`, SHA256 `4fa7074e8be917f184aa80dd75073d60c53429d9c09d30916d30e8a642dc8d74`) -> `0.2778` (`OWNER-REPORTED`).
  - Unpruned parent `h33-h33-1-20261004T220000Z-e5eb6e7e-zeros.tif` (`GEMSDOE32`, SHA256 `9da4616629431bc563a3fd179f323338e89e4603ff64137a1060813537080bc3`) -> `0.2708` (`OWNER-REPORTED`).

## 2. Exact Pixel-Level Anatomy of Why `h33-h33-2-b2` Scored `0.2778` (vs `0.2708`)
Using `scripts/session7_verify.py` (`check_0_2778_pair()`), we verified byte-for-byte from the cached registry rasters:
1. **Exact subset relationship:** `h33-h33-2-b2` (`37,654` binary dots, `0.7287%` of footprint) is a strict pixel subset of `h33-h33-1` (`40,199` binary dots, `0.7779%` of footprint).
2. **Exact pruning rule (`b2` = 2 px catalogue buffer):** The `2,545` removed dots (`40,199 - 37,654 = 2,545`) are **100% of the dots in `h33-h33-1` that sat within $d \le 2.0\text{ px}$ ($\le 200\text{ m}$) of the mapped catalogue (`existing_faults.tif`)**. Every single kept dot in `h33-h33-2-b2` has Euclidean distance $d \ge \sqrt{5} \approx 2.236\text{ px}$ from the mapped catalogue (`median = 19.03 px`, `p90 = 42.05 px`).
3. **Mathematical mechanism under Asymmetric DTI ($\alpha = 0.2, \beta = 0.8, R = 300\text{ m}$):**
   - Known catalogue pixels (`existing_faults.tif`) are masked from scoring (`masked_known`). A dot placed at $d = 1\text{ px}$ from a long mapped fault strand has up to $13/29$ (`44.9%`) of its 29-pixel triangular kernel disk overlapping masked known-fault pixels or the immediate $100\text{ m}$ flank, where any true splay tip is already covered by a dot at $d = 2.24\text{–}3.0\text{ px}$.
   - Deleting those `2,545` redundant near-catalogue dots removes $0.2 \times 2,545 = 509$ units of false-positive denominator penalty while losing negligible true-positive coverage, raising the score from `0.2708` to `0.2778` (`+0.0070`).

## 3. Why Session H59-I Scored `0.082161` on Holdout and How Session H59-I2 Doubles It (`0.166789`)
In Session H59-I, the Intensity Lane baseline scored `HOLDOUT-DTI = 0.082161 [0.069542, 0.096158]` because:
1. It fit its point-process intensity without structural distance or two-host stepover covariates (`IR-59-LOCO-01`).
2. Its `tile_blocks` function wrapped lattice coordinates modulo 10 (`(ay + np.arange(0, 10, s)) % 10`), placing dots 1 px apart at offsets `8, 9` for $s = 3$ (`IR-59-LATTICE-MOD10`).
3. It did not join `qfault_attributes.csv` (`FCODE2023`, `MAPSCALE`, `IR-59-EFFORT-02`).

In **Session H59-I2**, fixing all three (`src/gems59/intensity.py`) raises 4-fold pooled `HOLDOUT-DTI` to **`0.166789 [0.153242, 0.179658]`**:
- **+103.00% (`+0.084628` DTI)** over Session H59-I (`0.082161 [0.069542, 0.096158]`).
- **+23.36% (`+0.031585` DTI)** over the published all-lane holdout best (`relay_bend_anatomy`, `0.135204 [0.119140, 0.153036]`).

## 4. Can We Beat `0.2778`, `0.3195`, and `0.3774`?
- **Holdout vs Live Test Caveat:** Local `HOLDOUT-DTI` (`gems57-pooled-hide-v2`) withholds whole components from the USGS/INGENIOUS catalogue (`median distance to visible host = 9.06 px`), whereas the private competition test set evaluates genuinely unmapped faults compiled after the public release (`median distance of GEMSDOE32 dots = 19.03 px`). Consequently, no local holdout number is a projection of a live leaderboard score.
- **Why H59-I2 is positioned to compete with `0.2778+` in the Intensity Lane:**
  1. Unlike pure distance halos (`GEMSDOE32`), H59-I2 weights every candidate pixel by its 31-covariate LOCO Poisson/MaxEnt intensity $\lambda(x)$ combining **two-host relay facing/ratio (`rfacing * rratio`)**, **host-bend curvature (`turn_vis`)**, **strike-aligned 3DEP topographic scarp coherence (`g_scarp_rel`)**, **magnetic/gravity horizontal gradients (`g_mag_hgrad`, `g_grav_hgrad`)**, and **joined USGS/INGENIOUS mapping-scale covariates (`FCODE2023`, `MAPSCALE`)**.
  2. H59-I2's 1 km block zoning concentrates `78.25%` of withheld fault pixels into the top 3 deciles (`D8–D10`) and tiles only blocks clearing the break-even floor with cross-block spacing `{none, 4 px, 4 px, 3 px}`, achieving the highest `HOLDOUT-DTI` (`0.166789`) recorded in this repository series.
  3. To close the remaining gap toward `0.3195` (`DARD`) and `0.3774` (`xiaofanhu`), `REMAINING_WORK.md` pre-registers `H60-I1` (Quaternary basin-fill burial interaction so geophysical gravity/magnetic basement edges carry 3x weight where surface scarps are buried under alluvium) and `H60-I2` (anisotropic strike-aligned $2\text{ km} \times 0.5\text{ km}$ corridor blocks).
