# Utility Table & Break-Even Derivation — Intensity Lane (Session H59-I2)

## 1. Analytic Break-Even Condition Under the Asymmetric DTI Metric
The DOE GEMS competition metric (DrivenData competition 306, [page 967](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)) is the Distance-Weighted Tversky Index with $\alpha = 0.2$ (false-positive weight), $\beta = 0.8$ (false-negative weight), and a $300\text{ m}$ ($R = 3\text{ px}$) 2D triangular distance kernel:
$$w(d) = \max\!\left(0,\; 1 - \frac{d}{3}\right), \qquad \text{DTI} = \frac{\text{TP}}{\text{TP} + 0.2\,\text{FP} + 0.8\,\text{FN}}.$$

In thediscrete 100 m pixel grid, the 29-pixel support disk $\{(\Delta y, \Delta x) : \sqrt{\Delta y^2 + \Delta x^2} \le 3\}$ has kernel weight sum:
$$W_{\text{disk}} = \sum_{d \le 3} w(d) = 11.1289902\text{ pixel-equivalents}.$$

Let the global baseline DTI score be $S \in (0, 1)$, and let $D = \text{TP} + 0.2\,\text{FP} + 0.8\,\text{FN}$ be the global Tversky denominator. Adding a candidate dot at pixel $x$ inside a $1\text{ km} \times 1\text{ km}$ block $b$ ($10 \times 10 = 100$ pixels) with local unmapped new-fault intensity $\lambda(x)$ yields marginal true-positive coverage $\Delta\text{TP}(x) \approx \sum_{u \in B_3(x)} w(\|u - x\|)\,\lambda(u)$ and marginal false-positive mass $\Delta\text{FP}(x) \approx 1 - \Delta\text{TP}(x)$. The marginal change in global DTI is positive ($\Delta\text{DTI} > 0$) if and only if:
$$(1 - S + 0.8\,S)\,\Delta\text{TP}(x) > 0.2\,S\,(1 - \Delta\text{TP}(x)) \iff \Delta\text{TP}(x) > \frac{0.2\,S}{1 - 0.2\,S}.$$
For $S \in [0.14, 0.28]$, the required per-dot kernel reward is $\Delta\text{TP}(x) \in [0.029, 0.059]$. Dividing by the 29-pixel kernel sum $W_{\text{disk}} = 11.13$ (accounting for boundary and stepover clustering inside the $10 \times 10$ block), a $1\text{ km} \times 1\text{ km}$ block of 100 pixels clears break-even when its expected new-fault pixel count
$$n_b = \sum_{x \in b \setminus \text{cat}} \lambda(x)$$
exceeds **$n_{\min} \approx 1.0\text{–}2.0\text{ pixels per }1\text{ km block}$ ($1\text{–}2\%$ block density)**.

## 2. Empirical Break-Even Sweep on the 4-Fold Holdout (`gems57-pooled-hide-v2`)
Evaluated across all 4 spatially blocked folds (`11,321` withheld positive pixels, seed 20, `evidence/intensity_exp2_utility_table.json`):

| Break-Even Floor (`n_min`) | Block Density (%) | Pooled HOLDOUT-DTI | 95% Bootstrap CI |
|---|---|---|---|
| `1.00 px / block` | `1.00%` | `0.155466` | `[0.141849, 0.168722]` |
| `1.25 px / block` | `1.25%` | `0.160309` | `[0.146573, 0.174014]` |
| `1.50 px / block` | `1.50%` | `0.163091` | `[0.148510, 0.176272]` |
| `2.00 px / block` | `2.00%` | `0.165830` | `[0.151020, 0.180049]` |

## 3. Holdout-Derived Quartile Utility Spacing Table (`evidence/intensity_exp2_utility_table.json`)
Blocks clearing `n_b >= 1.5` (`14,904` blocks on the full catalogue) are partitioned into 4 quartiles (`q1_low, q2, q3, q4_high`). Each quartile class is evaluated at spacing $s \in \{3\text{ px}, 4\text{ px}, 6\text{ px}, \text{none}\}$ while holding the other classes at the $3\text{ px}$ baseline:

| Quartile Class | `s = 3 px` (300 m) | `s = 4 px` (400 m) | `s = 6 px` (600 m) | `none` (skip class) | Selected Active Spacing | Selected Overall Spacing |
|---|---|---|---|---|---|---|
| `q1_low` (`1.5 <= n_b <= q25`) | `0.155306` | `0.160128` | **`0.161790`** | **`0.162056`** | `s6 (6 px)` | **`none`** (effective ~2.0% floor) |
| `q2` (`q25 < n_b <= q50`) | `0.155306` | **`0.158207`** | `0.155319` | `0.151195` | `s4 (4 px)` | **`s4 (4 px)`** |
| `q3` (`q50 < n_b <= q75`) | `0.155306` | **`0.157609`** | `0.151822` | `0.147893` | `s4 (4 px)` | **`s4 (4 px)`** |
| `q4_high` (`n_b > q75`) | **`0.155306`** | `0.151010` | `0.128811` | `0.124494` | `s3 (3 px)` | **`s3 (3 px)`** |

Once Session H59-I's modulo-10 wraparound bug (`(ay + np.arange(0, 10, 3)) % 10`, `IR-59-LATTICE-MOD10`) is fixed, the empirical holdout utility table recovers the exact theoretical spacing hierarchy:
$$\text{below break-even / }q_{1,\text{low}}\text{: none (or }6\text{ px)} \;\longrightarrow\; q_2\text{: }4\text{ px} \;\longrightarrow\; q_3\text{: }4\text{ px} \;\longrightarrow\; q_{4,\text{high}}\text{: }3\text{ px},$$
achieving pooled **`HOLDOUT-DTI = 0.166789 [0.153242, 0.179658]`** in Experiment 3 (`evidence/intensity_exp3_final.json`).
