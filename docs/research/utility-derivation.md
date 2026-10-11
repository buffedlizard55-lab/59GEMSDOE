# Utility derivation — when does adding a dot raise the score?

Status: **algebra verified on synthetic trials** (500/500, `scripts/utility_table.py`, `evidence/utility_table_synthetic.json`). The numeric rows below are scenario inputs to the identity, not measured candidate scores. A HOLDOUT-DTI is only one carrying its evaluator, withheld-positive count and 95% CI; public-board values are participant context and do not establish file attribution.

## 1. Exact identity

Let `G` be the number of ground-truth pixels (after the known-fault mask), `TP`, `FP`, `FN` the
distance-weighted quantities from the official problem page, and `S = TP + FP`.

- `TP + FN = G` exactly, because `TP = Σ_g M(g)` and `FN = Σ_g (1 − M(g))`, where `M(g)` is the best kernel-weighted prediction within 300 m of truth pixel g.
- Therefore the denominator `TP + α·FP + β·FN = α·S + β·G` (using `α + β = 1`).

So **DTI = TP / (α·S + β·G)**, with α = 0.2, β = 0.8 (official).

## 2. Marginal rule (exact)

Write `D = DTI`. Adding mass changes `TP` by `ΔT` and `S` by `ΔS`. Then

`sign(ΔD) = sign(ΔT − α·D·ΔS)`.

Proof: `ΔD = [ΔT·(αS+βG) − T·α·ΔS] / (αS+βG)²` and `T = D·(αS+βG)`, so the numerator is `(αS+βG)·(ΔT − αD·ΔS)`.

**Consequence:** a dot pays iff its gain in TP exceeds `α·D` times its gain in `(TP+FP)`.

## 3. The single-pixel special case

For an isolated dot whose only effect is on one truth pixel at kernel credit `k = 1 − d/R`:
`ΔT = k`, `ΔFP = 1 − k`, so `ΔS = 1`, and the dot pays iff `k > α·D`.

| Assumed D input for this algebra scenario (not a candidate score) | Derived break-even credit k* = 0.2·D | Derived distance d* = 3·(1 − k*) | Derived metres |
| ---: | ---: | ---: | ---: |
| 0.10 (illustrative) | 0.0200 | 2.94 px | 294 m |
| 0.2778 (illustrative; public-board participant row only; H33 file mapping unverified) | 0.0556 | 2.83 px | 283 m |
| 0.3195 (illustrative; DARD public-board row, rank 8, no TIFF link) | 0.0639 | 2.81 px | 281 m |
| 0.3774 (illustrative; public-board rank 1, not a GEMSDOE file receipt) | 0.0755 | 2.77 px | 277 m |
| 0.50 (illustrative) | 0.1000 | 2.70 px | 270 m |

Reading: at any plausible score level the single-dot break-even sits just inside the 300 m kernel.
A dot further than about 280–295 m from a true fault is worth nothing in this simple case.

## 4. Why this is not "a density of 1–2 %"

The 1–2 % figure in the brief is a *density* heuristic. The rule above is about **credit per dot**,
which depends on distance to truth and on overlap. Two consequences:

- A dot can pay even at low density if it sits within ~280 m of a true fault.
- A dense lattice can lose if most of its dots lie further than ~280 m from truth, because each one adds `ΔS ≈ 1` and almost no `ΔT`.

Illustration only (`PROJECTION`, not measured): if a pixel lies within 300 m of an unmapped fault with
probability π, and the average kernel credit given that is c, then expected credit is π·c. With c = 0.5,
the break-even is π ≈ 0.11 at D = 0.28. The value of c is unknown until a holdout is run.

## 5. Spacing along a fault (SYNTHETIC toy)

A 60-px straight truth line; dots at p = 1 every s px along it.

| perpendicular offset | s = 1 | s = 2 | s = 3 | s = 4 | s = 6 | s = 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 px (on the line) | 1.000 | 0.862 | 0.809 | 0.704 | 0.539 | 0.437 |
| 1 px | 0.667 | 0.627 | 0.609 | 0.532 | 0.407 | 0.329 |
| 2 px | 0.333 | 0.318 | 0.308 | 0.252 | 0.181 | 0.145 |

**Not reproduced:** the brief's "about 3 px when dense, wider as density falls" would predict an
interior optimum. This toy has no false-positive mass away from the fault, so denser is monotonically
better and no optimum appears. The toy cannot test the claim. A holdout with real off-fault
background is required, and until one exists the spacing rule is **unconfirmed**.

## 6. What is still needed

1. A protocol-qualified spatial holdout after the independent target-group source is established and the case-control prior is computed from visible training labels only. A historical H59-I holdout exists, but its prior used full-catalogue prevalence and it did not use target-group background; it is exploratory and must not set a promotion threshold.
2. Estimate the credit distribution `E[k | x]` for candidate pixels from a defensible training design, then compare complete emission sets on the blocked holdout with confidence intervals.
3. Apply the strict surface registry check before placement and final-dot check after; a literal `rho > 0.90` or forward 3 px overlap `> 0.70` is STOP. A prior session's universal-support-witness claim is unverified in this checkout because its TIFF and verification receipts are absent; the 85 retained historical final-dot overlap firings independently trigger STOP.
4. Only after those gates, choose emission by thresholding `E[k | x]` at `α·D*`, where `D*` is a protocol-qualified HOLDOUT-DTI of the chosen set. This algebra is not a score projection and does not itself authorize a weekly slot.
