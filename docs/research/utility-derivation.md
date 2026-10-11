# Utility derivation — when does adding a dot raise the score?

Status: **algebra verified** (500/500 random trials, `scripts/utility_table.py`, `evidence/utility_table_synthetic.json`).
Labels: all numbers here are `SYNTHETIC` or `PROJECTION`. None is a holdout or organizer score.

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

| DTI (score level) | break-even credit k* = 0.2·DTI | break-even distance d* = 3·(1 − k*) | in metres |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.0200 | 2.94 px | 294 m |
| 0.2778 (the 0.2778 figure) | 0.0556 | 2.83 px | 283 m |
| 0.3774 (public #1 at revision 1) | 0.0755 | 2.77 px | 277 m |
| 0.4133 (public #1 at revision 2, 2026-10-11) | 0.0827 | 2.75 px | 275 m |
| 0.50 | 0.1000 | 2.70 px | 270 m |

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

1. A holdout on which `TP`, `FP` and `FN` can be measured for real emission sets (blocked until the data is placed).
2. The estimated credit distribution `E[k | x]` for each candidate pixel, which the model must supply.
3. Only then, choose emission by thresholding `E[k | x]` at `α·D*`, where `D*` is the holdout DTI of the chosen set (fixed point).
