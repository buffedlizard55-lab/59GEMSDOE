#!/usr/bin/env python3
"""Write the Session-7 run card and the two new audit findings.

Reads only measured receipts written earlier in this session:

* ``evidence/h57m_emission.json``            the emission run (curves + budget)
* ``evidence/h57m_validation.json``          local format preflight of the TIFF
* ``evidence/h57m_uniqueness_certificate.json`` two-reading registry measurement
* ``evidence/gate_universality.json``        blanket-raster / best-known control
* ``evidence/h57m_holdout.json``             evaluator arms for the H57-M lane

It writes ``evidence/run_card_current.json`` (the file the Pages build renders)
and appends IR-S7A-01 / IR-S7A-02 to ``evidence/irregularities_current.json``.
Nothing here re-thresholds a gate or promotes a candidate: the verdict stays
fail-closed because the inherited forward-overlap gate fired.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"


def load(name):
    return json.loads((EVID / name).read_text())


def main() -> int:
    emission = load("h57m_emission.json")
    validation = load("h57m_validation.json")
    cert = load("h57m_uniqueness_certificate.json")
    univ = load("gate_universality.json")
    holdout = load("h57m_holdout.json")
    previous = load("run_card_current.json")

    curve = emission["holdout_curve"]
    at = {int(p["budget"]): p for p in curve}
    budget = int(emission["chosen_budget"])
    brackets = sorted(at)
    near = min(brackets, key=lambda b: abs(b - budget))
    peak = max(curve, key=lambda p: p["holdout_dti"])
    lit = cert["literal_reading"]
    lfl = cert["like_for_like_reading"]

    sidecar = json.loads((ROOT / str(Path(emission["file"]).with_suffix(".json"))).read_text())
    note = sidecar["note"]  # the file's own receipt governs what a submitter pastes

    card = {
        "session": "session-7-2026-10-10",
        "hypothesis": previous["hypothesis"],
        "mechanism": (
            "H57-M keeps the retained two-host relay/bend anatomy posterior and changes only the "
            "emission rule: a 43950-dot binary field allocated by the exact-curve optimum of the "
            "hide-and-recover holdout, rescaled from the holdout band area to the production "
            "allowed area (IR-S7A-01 retracts the earlier surrogate-argmax selector)."
        ),
        "named_non_fault_process_that_could_mimic_it": previous["named_non_fault_process_that_could_mimic_it"],
        "holdout_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems59-pooled-hide-v2",
            "representation": "binary allocation of the emitted dot field, 4-fold LOQO, seed 20",
            "budget": budget,
            "nearest_measured_budget": near,
            "dti": at[near]["holdout_dti"],
            "ci95": at[near]["holdout_ci95"],
            "withheld_positive_pixels": at[near]["withheld_positive_pixels"],
            "measured_peak": {"budget": int(peak["budget"]), "dti": peak["holdout_dti"]},
            "note": ("The shipped budget 43950 lies between the measured 40000 and 60000 points; "
                     "no value is interpolated, the 40000 reading is shown. Numbers are local "
                     "HOLDOUT-DTI, never a live or private score forecast."),
        },
        "holdout_dot_dti": {
            "evidence_class": "HOLDOUT-DTI",
            "evaluator_version": "gems59-pooled-hide-v2",
            "dti": at[near]["holdout_dti"],
            "ci95": at[near]["holdout_ci95"],
            "withheld_positive_pixels": at[near]["withheld_positive_pixels"],
            "budget_curve": [[int(p["budget"]), p["holdout_dti"]] for p in curve],
        },
        "h57m_arms": {
            k: {"dti": v["dti"], "ci95": v.get("ci95"), "withheld_positive_pixels": v.get("withheld_positive_pixels")}
            for k, v in holdout.get("scores", {}).items()
        },
        "emission_selector": {
            "chosen_budget": budget,
            "chosen_by": emission.get("chosen_by"),
            "retracted_previous_artifact": "downloads/gems59-h57m-relay-band-20261010T215104Z-gems59-h57m.tif",
            "reason": ("the surrogate budget curve is monotone increasing (0.05369 at 2500 dots to "
                       "0.47944 at 140000), so its argmax always selects the largest budget; the "
                       "exact 4-fold curve peaks at 10000 dots (0.16441) and falls to 0.09844 at "
                       "140000. The 140000-dot artifact was deleted and is not offered."),
        },
        "final_dots": int(emission["dots"]),
        "raster_sha256": emission["sha256"],
        "validator_output": validation,
        "submission_name": sidecar["submission_name"],
        "submission_note": note,
        "submission_note_chars": len(note),
        "file": emission["file"],
        "zip_file": str(Path(emission["file"]).with_suffix(".zip")),
        "correlation_overlap_vs_registry": {
            "evidence_class": "REGISTRY-MEASUREMENT",
            "registry_rasters_checked": cert["rasters_measured_this_run"],
            "registry_rasters_expected": cert["registry_rasters_pinned"],
            "registry_rasters_pinned": cert["registry_rasters_pinned"],
            "complete_accessible_scan": False,
            "worst_dot_overlap": lit["worst_forward_overlap"],
            "worst_overlap_submission": lit["worst_overlap_submission"],
            "worst_spearman_full_footprint": lit["worst_spearman_full_footprint"],
            "duplicate_count": lit["forward_overlap_firings"],
            "like_for_like_forward_overlap_firings": lfl["forward_overlap_firings"],
            "like_for_like_worst_forward_overlap": lfl["worst_forward_overlap_dot_fields"],
            "like_for_like_worst_forward_submission": lfl["worst_forward_overlap_dot_field_submission"],
            "worst_reverse_overlap": lfl["worst_reverse_overlap"],
            "blanket_rasters": univ["blanket_rasters"],
            "forward_firings_blanket": univ["forward_firings_blanket"],
            "forward_firings_localised": univ["forward_firings_localised"],
            "rasters_firing_in_both_directions": univ["rasters_firing_in_both_directions"],
            "best_known_control": univ["best_known_control"],
            "scope": ("Measured against 213 of 696 pinned registry rasters (35 hashed local copies, "
                      "178 fetched and SHA256-verified this session). The remaining entries are a "
                      "stated scope limitation, not a clearance."),
        },
        "gate_universality": {
            "question": univ["question"],
            "verdict": univ["verdict"],
            "blanket_limit": univ["blanket_limit"],
            "localized_firings": [
                {"submission": r["submission"], "forward_overlap": r["forward_overlap"],
                 "reverse_overlap": r["reverse_overlap"], "dil_coverage": r["dil_coverage"],
                 "their_dots": r["their_dots"]}
                for r in univ["localized_firings"]],
        },
        "verdict": "negative",
        "okay_to_download": True,
        "okay_to_submit": False,
        "verdict_reason": (
            "Fail-closed. Local format preflight passes and the file is new, but the inherited "
            "one-directional forward-overlap gate fires (worst 1.0000; 43 of 213 measured rasters). "
            "40 of those 43 are blanket rasters whose own 3 px dilation covers >=70 % of the "
            "footprint, so the gate fires for any candidate that puts dots on the geologically "
            "plausible grid; the owner's best-known file fires against 116 of 213. The three "
            "localised firings (a 344041-dot ensemble archive and a 624025-dot coverage field, "
            "dilation coverage 0.61-0.65) are not mutual: reverse overlap is 0.13-0.17 and no "
            "raster fires in both directions. Clearing this file needs the owner's explicit "
            "ruling on how soft/blanket registry rasters define a dot (IR-S6-01 / IR-S7A-02); "
            "no rule was relaxed here and no submission slot was used."
        ),
        "recorded_sense": previous.get("recorded_sense", {}),
        "experiments_used": 3,
        "submission_slots_used": 0,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registry_reviewed_utc": "2026-10-10",
        "registry_scope": (
            "696 pinned unique-grid rasters across the 57 public owner repositories; 213 measured "
            "for this candidate in this session (evidence/h57m_uniqueness_certificate.json)."
        ),
        "session6_verification": previous.get("session6_verification"),
        "receipts": {
            "emission": "evidence/h57m_emission.json",
            "validation": "evidence/h57m_validation.json",
            "uniqueness_certificate": "evidence/h57m_uniqueness_certificate.json",
            "gate_universality": "evidence/gate_universality.json",
            "holdout": "evidence/h57m_holdout.json",
        },
    }
    assert card["submission_note_chars"] <= 140, card["submission_note_chars"]
    (EVID / "run_card_current.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")

    ledger = load("irregularities_current.json")
    ids = {i["id"] for i in ledger["irregularities"]}
    new = [
        {
            "id": "IR-S7A-01",
            "severity": "high",
            "status": "fixed; the retracted artifact was deleted and is not offered",
            "finding": (
                "scripts/build_h57m_emission.py selected the emission budget by the argmax of a "
                "surrogate DTI curve that is monotone increasing in the dot budget (0.05369 at "
                "2500 dots to 0.47944 at 140000), so the argmax could only ever pick the largest "
                "budget. The exact 4-fold HOLDOUT-DTI curve peaks at 10000 dots (0.16441) and "
                "falls to 0.09844 at 140000; the surrogate-argmax run wrote a 140000-dot TIFF."
            ),
            "resolution": (
                "The 140000-dot TIFF and its mirrors were deleted. The selector now takes an "
                "explicit budget and records how it was chosen; the shipped budget 43950 is a "
                "documented density transfer (10000 dots x 2471263 production allowed cells / "
                "562290.5 mean holdout band cells)."
            ),
            "evidence": ["scripts/build_h57m_emission.py", "evidence/h57m_emission.json"],
            "remaining": (
                "The surrogate curve is retained only as a diagnostic; any future use must state "
                "that it cannot select a budget by argmax."
            ),
        },
        {
            "id": "IR-S7A-02",
            "severity": "high",
            "status": "measured; owner ruling required before any file can be cleared",
            "finding": (
                "The inherited one-directional forward-overlap gate is not satisfiable in "
                "practice. Of 213 pinned rasters measured against the new candidate, 40 have a "
                "3 px dilation covering >=70 % of the footprint (many cover 100 %, sliver 0), so "
                "any candidate placing >=70 % of its dots on the geologically plausible grid "
                "fires regardless of what it predicts. The owner's best-known file "
                "(GEMSDOE32 h33-2-b2, owner-reported 0.2778) fires against 116 of the same 213 "
                "rasters. No raster fires in both directions (worst reverse overlap 0.698725)."
            ),
            "resolution": (
                "Reported, not relaxed. The certificate prints the literal reading, a like-for-like "
                "reading on dot-field representations, and a three-way decomposition of the "
                "firings (blanket / localised / both-direction). The candidate is held."
            ),
            "evidence": [
                "evidence/gate_universality.json",
                "evidence/h57m_uniqueness_certificate.json",
                "src/gems59/uniqueness.py",
            ],
            "remaining": (
                "Owner decision: either accept that no nonempty candidate can pass the literal "
                "one-directional gate, or authorise an explicit revision (for example a symmetric "
                "both-direction condition, which currently yields zero firings) that is logged in "
                "the audit ledger before any selector is allowed to clear a file."
            ),
        },
    ]
    ledger["irregularities"].extend(i for i in new if i["id"] not in ids)
    (EVID / "irregularities_current.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "file": card["file"], "dots": card["final_dots"], "sha256": card["raster_sha256"],
        "verdict": card["verdict"], "okay_to_download": card["okay_to_download"],
        "okay_to_submit": card["okay_to_submit"],
        "worst_forward": card["correlation_overlap_vs_registry"]["worst_dot_overlap"],
        "irregularities": [i["id"] for i in ledger["irregularities"]][-2:],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
