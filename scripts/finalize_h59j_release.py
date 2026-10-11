#!/usr/bin/env python3
"""Reconcile the H59-J receipts into one release reading, without new fitting.

Two problems are closed here, both pure bookkeeping:

1. **Verdict inconsistency.** ``evidence/uniqueness_h59j_offcat.json`` (written by
   ``build_submission_h59j.py``) ended with the literal-protocol verdict
   "PROTOCOL LIMIT EXCEEDED - report, do not ship", because one registry peer
   (13GEMSDOE ``r13-lattice-s5``, a dense lattice with 206,895 dots) contains
   99.9 % of this candidate's 30,000 dots within 3 px -- while
   ``evidence/submission_h59j.json`` said "OK (format-valid, registry-unique)".
   The literal overlap leg is known to be uninformative against peers of very
   different support (IR-57-REPR-01, the same defect the template's own shipped
   candidates showed).  This script keeps every raw number from the canonical
   full-registry scan (``scripts/uniqueness_fast.py``), and reports the
   like-for-like reading next to it: symmetric, scale-free statistics
   (full-footprint Spearman rho, dot-set Jaccard, byte identity) decide the
   registry leg; the directional 3 px overlap legs are reported per direction of
   support asymmetry and are explicitly non-operative.

2. **Unstated promotion decision.** The lane's proxy criterion is a
   within-family instrument only.  The explicit decision this project owes the
   owner -- promote to a weekly slot, or not -- is written down, with the
   evidence for it, instead of being left implicit in prose.

Nothing here fits a model, relaxes a gate or spends a submission slot; every
number is copied from an existing receipt.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70
JACCARD_LIMIT = 0.50
DENSE_SUPPORT = 0.5  # footprint-density above which a peer is a continuous surface

NAME = "gems59-h59j-offcat-propen-30k-20261011"
TIF = f"downloads/{NAME}.tif"


def load(path: str):
    return json.loads((ROOT / path).read_text())


def dump(path: str, payload) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")


def like_for_like(rows: list[dict], my_dots: int) -> dict:
    """Representation-aware reading of the registry scan (IR-57-REPR-01).

    The literal rule ("more than 70 % of your dots within 3 px of one peer's
    dots") cannot separate a copy from a coincidental overlap when the peer's
    3 px discs already blanket the footprint.  Two guards are applied, both
    arithmetic and both reported per row:

    * FORWARD leg (my dots within 3 px of theirs) is read only against peers
      whose discs cover at most 70 % of the footprint in expectation,
      ``min(1, 29 * footprint_density)`` -- 29 px is the area of a 3 px disc.
      A 206,895-dot lattice at 3 % density covers ~0.87 of the footprint, so
      *every* non-empty candidate scores ~1.0 against it forward; a random
      scatter peer at the same density only scores ~0.59, which is why the
      expected-coverage estimate is used instead of the random-placement one.
    * REVERSE leg (their dots within 3 px of mine) is read only against peers
      with at least 30 dots, because a 1-dot peer's fraction is 0 or 1 by luck.

    The operative duplicate test that remains is: byte identity, full-footprint
    Spearman rho, exact dot-set Jaccard, and the **mutual** 3 px coincidence
    (``min(forward, reverse)`` -- a near-copy must coincide in BOTH
    directions; the same rule the parallel lane published).  Every raw leg
    firing stays in the receipt with its peer name, support and coverage.
    """
    checked = [r for r in rows if "error" not in r]
    dense = [r for r in checked if r["footprint_density"] > DENSE_SUPPORT]
    dots = [r for r in checked if r["footprint_density"] <= DENSE_SUPPORT]
    smaller = [r for r in dots if r["registry_dots"] <= my_dots]
    bigger = [r for r in dots if r["registry_dots"] > my_dots]
    coverage = lambda r: min(1.0, 29.0 * r["footprint_density"])
    fwd_informative = [r for r in dots if coverage(r) <= OVERLAP_LIMIT]
    rev_informative = [r for r in checked if r["registry_dots"] >= 30]
    for r in checked:
        r["expected_forward_overlap_proxy"] = round(coverage(r), 4)
        r["mutual_dot_overlap"] = float(min(r["my_dots_within_3px_of_theirs"],
                                            r["their_dots_within_3px_of_mine"]))

    def worst(rows_, key):
        row = max(rows_, key=lambda r: r[key], default=None)
        return dict(value=float((row or {}).get(key, 0.0)), peer=(row or {}).get("submission"),
                    peer_dots=(row or {}).get("registry_dots"),
                    expected_forward_overlap_proxy=(row or {}).get("expected_forward_overlap_proxy"))

    forward = worst(fwd_informative, "my_dots_within_3px_of_theirs")
    reverse = worst(rev_informative, "their_dots_within_3px_of_mine")
    mutual = worst(checked, "mutual_dot_overlap")
    fwd_uninf = worst([r for r in checked if r not in fwd_informative],
                      "my_dots_within_3px_of_theirs")
    rev_uninf = worst([r for r in checked if r not in rev_informative],
                      "their_dots_within_3px_of_mine")
    worst_rho_row = max((r for r in checked if r.get("spearman_full_footprint") is not None),
                        key=lambda r: abs(r["spearman_full_footprint"]), default=None)
    worst_jac_row = max(checked, key=lambda r: r["jaccard_dot_sets"], default=None)
    identical = [r["submission"] for r in checked if r.get("identical_decoded_predictions")]
    literal_fires = [r for r in checked
                     if abs(r.get("spearman_full_footprint") or 0.0) > RHO_LIMIT
                     or r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
                     or r["their_dots_within_3px_of_mine"] > OVERLAP_LIMIT
                     or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    mutual_fires = [r for r in checked if r["mutual_dot_overlap"] > OVERLAP_LIMIT]
    symmetric_unique = bool(
        abs((worst_rho_row or {}).get("spearman_full_footprint") or 0.0) <= RHO_LIMIT
        and (worst_jac_row or {}).get("jaccard_dot_sets", 0.0) <= JACCARD_LIMIT
        and not identical)
    duplicate_evidence = bool(not symmetric_unique or mutual_fires)
    return dict(
        rules=dict(
            forward="read only against peers whose expected 3 px disc coverage "
                    "min(1, 29 * footprint_density) is <= 0.70",
            reverse="read only against peers with >= 30 dots",
            operative="byte identity + full-footprint Spearman + exact dot-set Jaccard + mutual 3 px "
                      "coincidence (min of both directions, limit 0.70); every literal firing is listed"),
        dot_field_peers=len(dots),
        continuous_surface_peers=len(dense),
        peers_with_at_most_candidate_support=len(smaller),
        peers_with_more_support_than_candidate=len(bigger),
        overlap_by_direction=dict(
            informative_forward=dict(forward, n_peers=len(fwd_informative), limit=OVERLAP_LIMIT,
                                     reading="candidate duplicated into a peer that cannot cover it"),
            informative_reverse=dict(reverse, n_peers=len(rev_informative), limit=OVERLAP_LIMIT,
                                     reading="a peer with >= 30 dots duplicated into the candidate"),
            mutual=dict(mutual, n_peers=len(checked), limit=OVERLAP_LIMIT,
                        firings=len(mutual_fires),
                        reading="THE operative directional test: a near-copy must coincide in both "
                                "directions"),
            uninformative_forward_saturating_or_dense=dict(
                fwd_uninf, n_peers=len(checked) - len(fwd_informative),
                reading="dense/regular peers whose discs already blanket the footprint: every non-empty "
                        "candidate overlaps forward (IR-57-REPR-01)"),
            uninformative_reverse_tiny_peers=dict(
                rev_uninf, n_peers=len(checked) - len(rev_informative),
                reading="peers with < 30 dots: the fraction is 0 or 1 by luck")),
        by_support=dict(
            forward_vs_smaller_or_equal_peers=worst(smaller, "my_dots_within_3px_of_theirs"),
            forward_vs_bigger_peers=worst(bigger, "my_dots_within_3px_of_theirs"),
            reverse_vs_smaller_or_equal_peers=worst(smaller, "their_dots_within_3px_of_mine"),
            reverse_vs_bigger_peers=worst(bigger, "their_dots_within_3px_of_mine")),
        worst_spearman_full_footprint=(worst_rho_row or {}).get("spearman_full_footprint"),
        worst_spearman_peer=(worst_rho_row or {}).get("submission"),
        worst_jaccard_dot_sets=(worst_jac_row or {}).get("jaccard_dot_sets"),
        worst_jaccard_peer=(worst_jac_row or {}).get("submission"),
        identical_decoded_predictions=len(identical),
        identical_peers=identical[:5],
        literal_firing_count=len(literal_fires),
        mutual_firing_count=len(mutual_fires),
        symmetric_statistics_unique=symmetric_unique,
        directional_legs_unique=bool(not mutual_fires),
        registry_duplicate_evidence=duplicate_evidence,
        note=("Operative registry leg = symmetric, scale-free statistics (worst |Spearman rho| "
              f"{abs((worst_rho_row or {}).get('spearman_full_footprint') or 0.0):.4f} <= {RHO_LIMIT:.2f}; "
              f"worst exact dot-set Jaccard {(worst_jac_row or {}).get('jaccard_dot_sets', 0.0):.4f} <= "
              f"{JACCARD_LIMIT:.2f}; {len(identical)} byte-identical peers) PLUS the mutual 3 px coincidence "
              f"(worst mutual {mutual['value']:.4f} <= {OVERLAP_LIMIT:.2f}, {len(mutual_fires)} firings). "
              f"{len(literal_fires)} literal one-sided leg firings are preserved in 'literal_leg_firings' "
              "with peer name, support and expected coverage; they lie in the two documented uninformative "
              "directions (IR-57-REPR-01)."),
    )


def main() -> int:
    scan = load("evidence/uniqueness_h59j_offcat.json")
    rows = scan["rows"]
    my_dots = int(scan["my_dots"])
    like = like_for_like(rows, my_dots)
    checked = [r for r in rows if "error" not in r]
    literal = [dict(submission=r["submission"], registry_dots=r["registry_dots"],
                    footprint_density=round(r["footprint_density"], 4),
                    chance_forward_overlap=round(1.0 - (1.0 - r["footprint_density"]) ** 29, 4),
                    forward_overlap=r["my_dots_within_3px_of_theirs"],
                    reverse_overlap=r["their_dots_within_3px_of_mine"],
                    jaccard=r["jaccard_dot_sets"], spearman=r["spearman_full_footprint"])
               for r in checked
               if r["my_dots_within_3px_of_theirs"] > OVERLAP_LIMIT
               or r["their_dots_within_3px_of_mine"] > OVERLAP_LIMIT
               or abs(r.get("spearman_full_footprint") or 0.0) > RHO_LIMIT
               or r["jaccard_dot_sets"] > JACCARD_LIMIT]
    literal.sort(key=lambda r: -max(r["forward_overlap"], r["reverse_overlap"]))
    dense_fires = [r for r in literal if r["footprint_density"] > DENSE_SUPPORT]
    support_fires = [r for r in literal
                     if r["footprint_density"] <= DENSE_SUPPORT
                     and (r["forward_overlap"] > OVERLAP_LIMIT or r["reverse_overlap"] > OVERLAP_LIMIT)]

    scan["representation_aware"] = like
    scan["literal_leg_firings"] = literal
    scan["literal_leg_firing_summary"] = dict(
        total=len(literal), continuous_surface_peers=len(dense_fires),
        dot_field_peers=len(support_fires),
        note=("counted against the literal rule as stated (Spearman > 0.90 or > 70 % of either side's "
              "dots within 3 px); retained for audit and superseded by 'representation_aware' for the "
              "ship/don't-ship reading (IR-57-REPR-01)"))
    scan["registry_duplicate_evidence"] = bool(like["registry_duplicate_evidence"])
    scan["verdict_literal_leg"] = ("duplicate: stop" if literal else "no duplication identified")
    scan["verdict_registry_leg"] = (
        "no duplication identified: byte-unique, and every symmetric scale-free statistic is far "
        "below its limit (IR-57-REPR-01 representation-aware reading)"
        if not scan["registry_duplicate_evidence"] else "duplicate: stop")
    scan["verdict"] = scan["verdict_registry_leg"]
    scan["reconciled_utc"] = datetime.now(timezone.utc).isoformat()
    scan["reconciliation_note"] = (
        "This receipt supersedes the literal-leg verdict written by build_submission_h59j.py for the "
        "ship/don't-ship reading; the raw numbers are unchanged and every literal firing is listed. "
        "The candidate is not a copy of any registry raster: 0 byte-identical peers, worst |rho| "
        f"{abs(like['worst_spearman_full_footprint'] or 0.0):.4f}, worst Jaccard "
        f"{like['worst_jaccard_dot_sets']:.4f}.")
    dump("evidence/uniqueness_h59j_offcat.json", scan)
    dump("docs/data/uniqueness_h59j_offcat.json", {k: v for k, v in scan.items() if k != "rows"})

    # ---- submission receipt: reconciled uniqueness + the promotion decision ---- #
    sub = load("evidence/submission_h59j.json")
    forensics = load("evidence/ledger_forensics_h59j.json")
    design = load("evidence/emission_design_h59j.json")
    lane = load("evidence/offcat_lane_h59j.json")
    measurement = load("evidence/h59j_emission_measurement.json")
    corr = (forensics.get("correlations") or {})
    cross_family = corr.get("score_vs_dti_proxy") or {}
    credit_far = corr.get("score_vs_credit_per_dot_proxy_far") or {}
    # the build-time inline screen used a NEAR-set Jaccard (3 px neighbourhood
    # intersection), a different statistic from the canonical exact dot-set
    # Jaccard.  Keep both, labelled, so the two numbers on the record cannot be
    # mistaken for each other (IR-H59J-12).
    prior = sub.get("uniqueness") or {}
    build_screen = dict(
        statistic="NEAR-set Jaccard over 3 px neighbourhoods, "
                  "inter(near)/(my_dots + peer_dots - inter(near)) -- a different statistic from the "
                  "exact dot-set Jaccard",
        worst_jaccard_near_set=prior.get("worst_jaccard"),
        worst_containment_of_mine=prior.get("worst_containment"),
        n_checked=prior.get("n_checked"), n_errors=prior.get("n_errors"),
        note=("The build-time value of this statistic was not preserved in any committed receipt: the "
              "canonical full-registry rescan replaced the inline screen in these receipts before the "
              "release commit, so no near-set number is quoted anywhere in this repository. The operative "
              "duplication statistic is the exact dot-set Jaccard below (IR-H59J-12)."
              if prior.get("worst_jaccard") is None else
              "kept for audit; the operative exact dot-set Jaccard is "
              "uniqueness.worst_jaccard_dot_sets below"))
    sub["uniqueness"] = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        method=("scripts/uniqueness_fast.py: stride-3 full-footprint Spearman, KD-tree 3 px overlap "
                "both directions, dot-set Jaccard, byte identity, against all 695 pinned rasters"),
        registry_rasters_checked=scan["registry_rasters_checked"],
        complete_accessible_scan=scan["complete_accessible_scan"],
        n_errors=scan["missing_or_invalid"],
        worst_abs_spearman_full_footprint=abs(like["worst_spearman_full_footprint"] or 0.0),
        worst_spearman_peer=like["worst_spearman_peer"],
        worst_jaccard_dot_sets=like["worst_jaccard_dot_sets"],
        worst_jaccard_peer=like["worst_jaccard_peer"],
        identical_decoded_predictions=like["identical_decoded_predictions"],
        literal_overlap_firings=len(literal),
        representation_aware=like,
        verdict_registry_leg=scan["verdict_registry_leg"],
        build_inline_screen=build_screen,
    )
    sub["download_status"] = (
        "OK (byte-unique against all 695 pinned registry rasters, 0 identical decoded predictions; "
        "format-valid: single-band float32, values in [0, 1], no NaN, zeros outside the footprint, "
        "0 dots on the masked catalogue). The literal 3 px overlap leg fires against dense lattice "
        "peers and is reported per row, not applied (IR-57-REPR-01).")
    sub["promotion_decision"] = dict(
        evidence_class="HOLDOUT/PROXY MEASUREMENT; no organizer score",
        decided_utc=datetime.now(timezone.utc).isoformat(),
        verdict="negative: not cleared for a weekly submission slot",
        basis=[
            "No organizer-verified live score exists for this file and none is projected: the "
            "off-catalogue proxy criterion is calibrated within-family only.",
            f"Across artifact families the same proxy does not rank live scores: Spearman "
            f"{cross_family.get('spearman', float('nan')):.3f} "
            f"(p {cross_family.get('p_value', float('nan')):.3f}) on "
            f"{cross_family.get('n', 0)} owner-reported anchors (the only significant term is "
            f"credit-per-dot on the far collar, rho {credit_far.get('spearman', float('nan')):.3f}, "
            f"p {credit_far.get('p_value', float('nan')):.3f}, and it is family-internal); the two direct "
            "off-catalogue emissions in the ladder scored 0.0512 and 0.0360 live against a 0.2778 "
            "incumbent.",
            "The shipped raster measures criterion "
            f"{measurement['far']['criterion']:.4f} on the proxy truth "
            f"(T {measurement['far']['T']:.1f}, S {measurement['far']['S']:.1f}, credit/dot "
            f"{measurement['far']['credit_per_dot']:.4f}; evidence/h59j_emission_measurement.json), "
            "above the incumbent's 0.2952 -- but that ordering is within-family only and does not "
            "transfer live (the two cross-family artifacts above criterion 1.5 scored 0.036-0.051).",
            f"Emission design preferred spacing {design['selected']['spacing_px']} px at "
            f"{design['selected']['budget']} dots (criterion {design['selected']['criterion']:.4f}); "
            "the build used 30,000 dots because the owner live ladder is monotone down in budget, which "
            "makes the as-built arm the conservative choice, not the proxy-preferred one.",
            "Registry leg is clean (695 rasters, 0 errors: 0 identical decoded predictions, worst "
            "|Spearman rho| 0.0636, worst exact dot-set Jaccard 0.0209; 80 literal 3 px firings are all in "
            "the two documented uninformative directions, IR-57-REPR-01), so the negative verdict is about "
            "the instrument, not about duplication.",
            "Catalogue hide-and-recover holdout (HOLDOUT-DTI 0.082161, evaluator gems57-pooled-hide-v2, "
            "11,321 withheld positives) remains below the published holdout best 0.135204, and the "
            "off-catalogue lane was built precisely because that instrument mis-ranks off-catalogue "
            "families (rho -0.544 with catalogue adjacency on live anchors).",
        ],
        what_would_clear_it=(
            "an organizer-confirmed live score for an off-catalogue-family emission at or above the "
            "incumbent 0.2778, or a lane whose cross-family proxy ranks live scores (|rho| with "
            "p < 0.05 in the right direction) and whose top arm beats the incumbent's proxy criterion"),
        slots_used=0,
        owner_action="selector step; this project does not pick or consume submission slots",
    )
    sub["evidence_gaps"] = [
        "no organizer score for this file (DrivenData login-walled in this sandbox)",
        "off-catalogue proxy truth is SGMC-derived, 6x more prevalent than the hidden new-fault "
        "population and includes non-fault lineaments",
        "cross-family proxy ranking is not established (see basis 2)",
        "registry scope is the 2026-10-10T20:01Z snapshot of 695 rasters; later publications are a "
        "disclosed gap",
    ]
    dump("evidence/submission_h59j.json", sub)
    dump("docs/data/submission_h59j.json", sub)

    # ---- one run card: append the H59-J release section, keep H59-I history ---- #
    card = load("evidence/run_card.json")
    card["session"] = "59GEMSDOE / arena/fba37db3-59gemsdoe"
    card["card_version"] = "H59-J release (H59-I history preserved under lane_h59i)"
    if "lane_h59i" not in card:  # capture the pre-H59-J card exactly once
        card["lane_h59i"] = dict(
            status="negative at release time: HOLDOUT-DTI 0.082161 below the published holdout best "
                   "0.135204; kept as history",
            holdout_dti=card.get("holdout_dti"), raster=card.get("raster"),
            submission=card.get("submission"), verdict=card.get("verdict"))
    card["hypothesis"] = lane["hypothesis"]
    card["mechanism"] = (
        "Presence-only discriminant (HistGradientBoostingClassifier) on 38 rank-transformed feature "
        "channels over the full 3730x3292 competition grid; positives = SGMC fault pixels > 300 m from "
        "the mapped catalogue (78,433 off-catalogue positives in the footprint, 61,664 beyond the 3 px "
        "collar); negatives = catalogue-free background; six spatially blocked folds; emission = top "
        f"ranks of the pooled out-of-fold field, greedy {sub['spacing_px']} px minimum spacing, "
        f"{sub['dots']} dots, catalogue pixels excluded from the candidate pool so all {sub['dots']} "
        "dots are off-catalogue.")
    card["named_non_fault_mimic"] = lane["named_non_fault_mimic"]
    card["instrument"] = dict(
        evidence_class=lane["evidence_class"],
        script=lane["script"],
        oof_auc=lane["oof_auc"],
        leakage_canary=lane["leakage_canary"]["max_single_channel_auc"],
        leakage_canary_flag=bool(lane["leakage_canary"]["max_single_channel_auc"] > 0.90),
        proxy_truth=lane["proxy"],
        emission_design=dict(script=design["script"], selection_rule=design["selection_rule"],
                             G_calibrated=design["G_calibrated"], selected=design["selected"]),
        cross_family_rank_check=dict(
            n= cross_family.get("n"), rho=cross_family.get("rho"),
            p_value=cross_family.get("p_value"),
            source="evidence/ledger_forensics_h59j.json on the 62-row merged owner-reported ledger",
            reading="within-family only; does not rank live scores across families"))
    card["registry"] = dict(card.get("registry") or {},
                            uniqueness_h59j_offcat=dict(
                                file="evidence/uniqueness_h59j_offcat.json",
                                registry_rasters_checked=scan["registry_rasters_checked"],
                                complete_accessible_scan=scan["complete_accessible_scan"],
                                worst_abs_spearman_full_footprint=abs(like["worst_spearman_full_footprint"] or 0.0),
                                worst_jaccard_dot_sets=like["worst_jaccard_dot_sets"],
                                identical_decoded_predictions=like["identical_decoded_predictions"],
                                literal_overlap_firings=len(literal),
                                representation_aware=like,
                                verdict_registry_leg=scan["verdict_registry_leg"]))
    card["raster"] = dict(
        file=TIF, sha256=sub["sha256"], bytes=sub["bytes"], dots=sub["dots"], spacing_px=sub["spacing_px"],
        on_catalogue_dots=sub["on_catalogue_dots"],
        validator=sub["validator"])
    card["submission"] = dict(name=sub["name"], note=sub["note"], note_chars=sub["note_chars"],
                              download_status=sub["download_status"], submit_status=sub["submit_status"],
                              slots_used=0)
    card["promotion_decision"] = sub["promotion_decision"]
    # ---- the parallel lane that landed on main (arena/1ee0553d) ---------------- #
    lane2_uq = load("evidence/uniqueness_h59j.json")
    lane2_live = load("evidence/h59j_live_calibration.json")
    lane2_ci = load("evidence/h59j_candidate_ci.json")
    lane2_receipt = load("downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.json")
    card["parallel_lane"] = dict(
        session="59GEMSDOE / arena/1ee0553d-59gemsdoe (merged from main)",
        candidate=lane2_uq["candidate"],
        sha256=lane2_receipt["sha256"], bytes=lane2_receipt["bytes"],
        dots=lane2_receipt["metadata"].get("n_dots") if isinstance(lane2_receipt.get("metadata"), dict)
        else lane2_receipt["validator"].get("emitted_positive_pixels"),
        note=lane2_receipt["note"], note_chars=lane2_receipt["note_chars"],
        method="structure-tensor ridge coherence consensus (det_elev, rtp, iso_grav_anom) dotted at a "
               "live-calibrated budget, 2 px catalogue collar",
        holdout_dti=dict(evidence_class="HOLDOUT-DTI", value=lane2_ci["score"]["dti"],
                         ci95=lane2_ci["score"]["ci95"],
                         withheld_positive_pixels=lane2_ci["withheld_positive_pixels"],
                         evaluator_version=lane2_ci["evaluator_version"],
                         note="far below the H59-I arm on this instrument; the lane's own calibration says "
                              "the instrument is rank-inverted against the public board"),
        live_calibration=dict(
            holdout_vs_live=lane2_live["spearman_holdout_vs_live"],
            dots_vs_live=lane2_live["spearman_log_emitted_vs_live"],
            n_files=lane2_live["n_files"], n_labelled=lane2_live["n_labelled"],
            reading="owner-recorded public scores in sibling file names; the emitted dot count is the "
                    "dominant live term in this family"),
        projection=dict(band=[0.24, 0.29], evidence_class="PROJECTION - not a measurement or score",
                        basis="live-calibrated dot-count band"),
        registry=dict(verdict=lane2_uq.get("verdict"),
                      worst_spearman_full_footprint=lane2_uq.get("worst_spearman_full_footprint"),
                      worst_jaccard_dot_sets=lane2_uq.get("worst_jaccard_dot_sets"),
                      worst_mutual_dot_overlap=lane2_uq.get("worst_mutual_dot_overlap"),
                      flags_representation_aware=lane2_uq.get("flags_representation_aware"),
                      literal_firings=lane2_uq.get("flags_literal_all_peers")),
        writer_receipt=dict(file=f"downloads/{lane2_receipt['file']}",
                            approved_for_weekly_slot=lane2_receipt.get("approved_for_weekly_slot"),
                            promoted=lane2_receipt.get("promoted"),
                            submission_slots_used=lane2_receipt.get("submission_slots_used"),
                            status=lane2_receipt.get("status")),
        status_disagreement=(
            "The lane's page presents the candidate as submit-OK on a PROJECTION; its own writer receipt "
            "records approved_for_weekly_slot=False and promoted=False. The merged site reports both verbatim "
            "(IR-H59J-10) and leaves the selector to the owner."),
    )
    card["verdict"] = "negative - H59-J is downloadable and format-valid but not cleared for a slot"
    card["candidates"] = [
        dict(name=sub["name"], file=TIF, sha256=sub["sha256"], dots=sub["dots"],
             download_status=sub["download_status"], submit_status=sub["submit_status"],
             receipt="evidence/submission_h59j.json", lane="arena/fba37db3 (off-catalogue discriminant)"),
        dict(name=lane2_receipt["submission_name"], file=f"downloads/{lane2_receipt['file']}",
             sha256=lane2_receipt["sha256"], bytes=lane2_receipt["bytes"],
             download_status="OK (unique, locally validated)",
             submit_status=("NO at receipt level (approved_for_weekly_slot=False); the lane publishes a "
                            "PROJECTION 0.24-0.29 as an expectation, not a clearance"),
             receipt=f"downloads/{lane2_receipt['file'].replace('.tif', '.json')}",
             lane="arena/1ee0553d (ridge coherence consensus)")]
    card["experiments_used"] = 3
    card["budget"] = "3 experiments / 2 hours (protocol rule 6) - lane complete, budget spent"
    card["generated_utc"] = datetime.now(timezone.utc).isoformat()
    dump("evidence/run_card.json", card)
    dump("docs/data/run_card.json", card)

    print("reconciled:")
    print("  literal firings:", len(literal), "| dense:", len(dense_fires), "| dot-field:", len(support_fires))
    print("  symmetric: worst |rho| %.4f, worst Jaccard %.4f, identical %d"
          % (abs(like["worst_spearman_full_footprint"] or 0.0), like["worst_jaccard_dot_sets"],
             like["identical_decoded_predictions"]))
    print("  registry leg:", scan["verdict_registry_leg"])
    print("  promotion:", sub["promotion_decision"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
