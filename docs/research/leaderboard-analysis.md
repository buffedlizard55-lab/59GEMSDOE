# Leaderboard analysis — why 0.2778, and can we beat it?

Evidence labels: `ORGANIZER-CONFIRMED` (copied from the DrivenData leaderboard page, fetched 2026-10-11),
`VERIFIED-FILE` (checked by our own reader on a public file), `DERIVED` (algebra from the official formula),
`UNVERIFIED` (claim we could not check), `PROJECTION` (estimate, not a score).

## 1. The numbers, with their status

| Item | Value | Label | Source |
| --- | ---: | --- | --- |
| Public leaderboard #1 (xiaofanhu, 15 submissions) | 0.3774 | ORGANIZER-CONFIRMED | <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/> |
| #2 JerryDataWorks | 0.3418 | ORGANIZER-CONFIRMED | same |
| #8 DARD | 0.3195 | ORGANIZER-CONFIRMED | same. This is the "0.3195 highest right now" figure in the brief. |
| #22 extradr19 | 0.2778 | ORGANIZER-CONFIRMED (participant name only) | same |
| #50 MSU-Laskowski-GEMS | 0.2448 | ORGANIZER-CONFIRMED | same |
| File `h33-h33-2-b2-…-zeros.tif` (GEMSDOE32) | scored 0.2778 per the brief | **UNVERIFIED mapping** | the leaderboard shows a participant, not a file |
| Gap, #1 minus #22 | 0.0996 | DERIVED | 0.3774 − 0.2778 |

**Irregularity:** the brief calls 0.3774 the "GEMSDOE high score". It is the overall #1 (a different
team), not a GEMSDOE entry. Nothing on the public board ties 0.2778 to GEMSDOE32's file, so the
mapping is an assumption until the owner confirms it.

## 2. What the 0.2778 file is (VERIFIED-FILE)

Read directly from `GEMSDOE32/docs/downloads/…-zeros.tif` with our own reader (`gems_dti.submission.validate_file`, self-template):

- single band, float32, EPSG:32611, 3730 × 3292, 100 m, no nodata tag
- values exactly {0, 1}; **37,654 positive cells**
- sha256 `c55bafc470054e8271dcb89347a17e07fefe50de6af6e6ba6c4b169ef7ab6fa9` (matches the repo's own record)
- the NaN-outside twin has **5,167,373 finite cells** (the footprint); the zeros twin has all 12,279,160 cells finite. 37,654 ÷ 5,167,373 = **0.729 %** of the footprint.
- transform (100, 0, 243350, 0, −100, 4508550), as reported by the repo. **UNVERIFIED against the official template**; our writer copies the transform from the official file instead.

## 3. Why would a binary file at ~0.7 % of the footprint score this well? (DERIVED + judgement)

From the exact rule (see `utility-derivation.md`), the score is `DTI = TP / (α·S + β·G)`. Four consequences:

1. **Recall dominates.** β = 0.8 against α = 0.2 means a missed fault costs four times a wrong dot. A sparse, confident set can still score if its hits are well placed.
2. **Break-even is about credit per dot, not density.** At DTI ≈ 0.28 a dot pays only if its kernel credit exceeds ≈ 0.056, i.e. it sits within ≈ 283 m of a true fault. Dots beyond that are pure cost.
3. **Calibration is not the lever.** Under the exact rule, the best per-pixel choice is a threshold on expected credit, which is why binary 0/1 output is natural. This is a heuristic argument, not a proof: the `max` over the kernel makes the objective non-linear. The 0.2778 file uses exactly that form.
4. **The mass budget is set by where the truth probably is.** 0.73 % of the footprint is a deliberate, small set, placed where the model thinks faults are. It is not a random lattice.

So the most defensible account of *why it scored* is: a sparse binary set, chosen by a thresholded
field, with recall weighted above precision by the official metric. That is judgement, not a
measured attribution. We have one public score and no organizer decomposition.

## 4. What the 0.2778 file's own story gets wrong (IRREGULARITY, review)

- The GEMSDOE32 README says the scorer "masks out" catalogue pixels **and** that dots on the
  catalogue "cost α = 0.2 each and can earn nothing". Those two statements contradict each other.
- DrivenData staff (2026-09-16) answered that pixels on known faults are "masked / excluded from
  evaluation, so they do not count towards penalty terms." **A dot on a known fault costs nothing.**
- So GEMSDOE32's pruning of catalogue dots ("0 dots within 200 m of the catalogue") does not change the
  official score. Its own instrument (`IR-32-INSTR-01`, a 47 % discontinuity) appears to come from charging
  catalogue dots. That instrument is therefore miscalibrated against the official rule.
- Consequence for us: any proxy that penalises known-fault dots will mis-rank candidates. Our holdout
  must apply the same mask the organizers use.

## 5. Can we beat 0.2778? (honest answer)

**Not demonstrated, and not testable in this sandbox.** To beat it we need, in order:

1. The organizer data (blocked, section 5 of the README).
2. A holdout that reproduces the official mask, with withheld whole segments and a buffer.
3. A candidate whose holdout DTI exceeds the holdout DTI of a 0.7 %-footprint binary baseline built the same way. The baseline matters: without it, a number like 0.2778 cannot be interpreted.

Any statement that we "can exceed 0.2778" before step 3 would be a `PROJECTION`, and we have not made one.

## 6. Recommendations for the next session (ranked by P(Win) per hour)

1. **Place the data and build the holdout with the official mask** (highest value; blocks everything).
2. **Build the binary baseline first** (same mass as 0.7 % of footprint, placed by a plain field). It calibrates the holdout before any idea is judged.
3. **Test threshold choice on the holdout** against the marginal rule in `utility-derivation.md` (emit where expected credit > 0.2·DTI).
4. **Confirm the 0.2778 participant-to-file mapping** with the owner before citing it as a GEMSDOE result.
5. **Ask DrivenData (community forum) whether the public leaderboard uses the same mask as the private set.** The staff answer covers "known faults"; it does not say whether public and private sets are scored the same way.
