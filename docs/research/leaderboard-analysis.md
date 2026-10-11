# Leaderboard figures, file attribution, and the 0.2778 / 0.3195 claims

**Review date: 2026-10-11 UTC.** This review separates organizer-published public-board entries, per-file attribution, owner-authored reports, local HOLDOUT-DTI, and model projections. A leaderboard row is not automatically a receipt tying a score to a particular TIFF.

## 1. What the public leaderboard says

The saved snapshot is [`evidence/leaderboard_snapshot_2026-10-11.json`](../../evidence/leaderboard_snapshot_2026-10-11.json), fetched from the [DrivenData public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/). Its values are **ORGANIZER-PUBLISHED public leaderboard context** (participant-level; not submission-page receipts and not private-set scores):

| Public rank | Participant shown | Public DTI | Attribution boundary |
| ---: | --- | ---: | --- |
| 1 | xiaofanhu | 0.3774 | Not a GEMSDOE repository submission based on this snapshot. |
| 2 | JerryDataWorks | 0.3418 | Participant-level only. |
| 8 | DARD | 0.3195 | This is the brief's “0.3195 highest right now” value; it was rank 8, not the overall high score. No TIFF link was captured. |
| 22 | extradr19 | 0.2778 | The board has this participant-level value; it does not link it to the GEMSDOE32 file. |
| 50 | MSU-Laskowski-GEMS | 0.2448 | Participant-level only. |

The competition rules distinguish public and private phases. These public-board values do not establish a private-set score, and none is an organizer receipt for the specific H33 TIFF discussed below.

## 2. The GEMSDOE32 `h33-h33-2-b2` file is not confirmed to have scored 0.2778

The preserved brief attaches `0.2778` to `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif`. We independently verified the public file's bytes and format, but that is only a **file measurement**. The leaderboard's 0.2778 belongs to the displayed participant `extradr19`; the page does not show a filename or repository link.

More importantly, at the owner-maintained GEMSDOE32 commit [`3ae286af599fdf86dbb0f333591f8273efcee806`](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/3ae286af599fdf86dbb0f333591f8273efcee806/docs/index.html), the site calls H33-2-B2 **UNSCORED** and says that no organizer score exists for any artifact in that repository. The paired [owner audit JSON](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/3ae286af599fdf86dbb0f333591f8273efcee806/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-audit.json) describes the TIFF and local calculations; its `predicted_leaderboard_dti` is explicitly a **model projection, not a score**. The file itself is available [here](https://github.com/buffedlizard55-lab/GEMSDOE32/blob/3ae286af599fdf86dbb0f333591f8273efcee806/docs/downloads/gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif).

**Conclusion:** the public value 0.2778 is organizer-published for a participant, but the file-to-participant mapping is **UNVERIFIED** and conflicts with the owner's contemporaneous `UNSCORED` statement. We do not report the H33 TIFF as having an organizer-confirmed score, and we do not use its projected leaderboard value as a target.

## 3. What H59-I measured—and did not measure

The historical H59-I final arm has **HOLDOUT-DTI 0.082161 [0.069542, 0.096158]**, evaluator `gems57-pooled-hide-v2`, 11,321 withheld positives, seed-20 buffered whole-component draw, α=0.2, β=0.8, 300 m triangular kernel. This is a local, exploratory hide-and-recover result—not a live/public-board score or an organizer receipt. The historical run used the full-catalogue prevalence in its case-control prior; the runner was corrected to use visible-only prevalence, but no rerun was performed. The requested target-group background is not implemented. The interval and point estimate therefore cannot clear the candidate for promotion.

The historical HOLDOUT-DTI is below the same-evaluator Session-5 reference `relay_bend_anatomy` **0.135204 [0.119140, 0.153036]**. That reference is also a local holdout value, not a leaderboard score. The H59-I file also fails the literal registry overlap gate, so there is no slot recommendation regardless of its score comparison.

## 4. No projection is a score

Any local metric estimated from a file, label proxy, assumed prevalence, utility model, or leaderboard back-calculation must be labelled **PROJECTION** or **DERIVED**. It is never promoted to an organizer score by matching a nearby leaderboard value. No file-to-score attribution for H33-2-B2 was established, and no organizer submission-page receipt for the H59-I artifact is present.

## 5. Decision

The responsible answer to “can we score above 0.2778 or 0.3195?” is **not established**. The H59-I result is a negative, protocol-incomplete local experiment; its download-valid TIFF is retained for inspection only. No competition slot was used, no weekly slot is recommended, and no claim of winning or score gain is made.
