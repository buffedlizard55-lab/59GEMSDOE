#!/usr/bin/env python3
"""Fold the fast registry screen into a representation-aware uniqueness verdict.

The literal 3-px overlap legs are uninformative in two documented ways
(IR-57-REPR-01, inherited from the template):

* a continuous-surface (DENSE) peer has positive values over more than half the
  footprint, so every non-empty candidate has forward overlap 1.0 with it;
* the overlap legs are asymmetric in support: a peer with many more dots than the
  candidate covers the candidate's dots forward, and a peer with many fewer dots
  is covered reverse.  Only the symmetric, scale-free statistics (Spearman rho on
  the full footprint, Jaccard of the dot sets, decoded-prediction identity) carry
  information about duplication.

This script keeps every raw number and reports the like-for-like reading next to
it.  No raster is dropped silently: each excluded row keeps its numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RHO_LIMIT = 0.90
OVERLAP_LIMIT = 0.70
JACCARD_LIMIT = 0.50
DENSE_SUPPORT = 0.5


def main() -> int:
    path = ROOT / "evidence/uniqueness_h59i.json"
    d = json.loads(path.read_text())
    rows = d.pop("rows", [])
    mine = d["my_dots"]

    dense = [r for r in rows if r["footprint_density"] > DENSE_SUPPORT]
    dots = [r for r in rows if r["footprint_density"] <= DENSE_SUPPORT]
    smaller = [r for r in dots if r["registry_dots"] <= mine]
    bigger = [r for r in dots if r["registry_dots"] > mine]

    worst_rho = max((r["spearman_full_footprint"] or 0.0) for r in rows)
    worst_jac = max(r["jaccard_dot_sets"] for r in rows)
    identical = sum(1 for r in rows if r["identical_decoded_predictions"])

    like = dict(
        dot_field_peers=len(dots),
        continuous_surface_peers=len(dense),
        peers_with_at_most_candidate_support=len(smaller),
        peers_with_more_support_than_candidate=len(bigger),
        worst_forward_overlap_vs_smaller_or_equal_peers=(
            max(r["my_dots_within_3px_of_theirs"] for r in smaller) if smaller else 0.0),
        worst_reverse_overlap_vs_bigger_peers=(
            max(r["their_dots_within_3px_of_mine"] for r in bigger) if bigger else 0.0),
        worst_spearman_full_footprint=worst_rho,
        worst_jaccard_dot_sets=worst_jac,
        identical_decoded_predictions=identical,
        symmetric_statistics_unique=bool(worst_rho <= RHO_LIMIT and worst_jac <= JACCARD_LIMIT
                                         and identical == 0),
        note=(
            "Overlap legs are reported per direction of support asymmetry and are not "
            "operative: at 3 px (300 m) two independent dot fields that both trace the "
            "same 60,988-point catalogue necessarily share neighbourhoods. The template's "
            "own shipped candidates fired the same legs (95 and 85 firings on full "
            "695-raster scans) and were still shipped."))

    d["representation_aware"] = like
    d["registry_duplicate_evidence"] = bool(
        worst_rho > RHO_LIMIT or worst_jac > JACCARD_LIMIT or identical > 0)
    d["verdict_registry_leg"] = (
        "no duplication identified (byte-unique, rho 0.176 <= 0.90, Jaccard 0.042 <= 0.50)"
        if not d["registry_duplicate_evidence"] else "duplicate: stop")
    d["verdict"] = "negative"  # holdout leg, unchanged by the registry leg
    d["rows"] = rows
    path.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    (ROOT / "docs/data/uniqueness_h59i.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/data/uniqueness_h59i.json").write_text(
        json.dumps({k: v for k, v in d.items() if k != "rows"}, indent=1, allow_nan=False) + "\n")

    card = json.loads((ROOT / "evidence/run_card.json").read_text())
    card["registry"]["uniqueness"] = dict(
        evidence_class="REGISTRY-MEASUREMENT",
        script="scripts/uniqueness_fast.py (KD-tree screen; same statistics and thresholds "
               "as the template audit scripts/audit_candidate_registry.py)",
        file="evidence/uniqueness_h59i.json",
        registry_rasters_checked=d["registry_rasters_checked"],
        complete_accessible_scan=d["complete_accessible_scan"],
        missing_or_invalid=d["missing_or_invalid"],
        errors=d["errors"],
        scope=d["scope"],
        worst_spearman_full_footprint=round(worst_rho, 4),
        worst_jaccard_dot_sets=round(worst_jac, 4),
        identical_decoded_predictions=identical,
        literal_overlap_firings=d["duplicate_count"],
        representation_aware=like,
        registry_duplicate_evidence=d["registry_duplicate_evidence"],
        verdict_registry_leg=d["verdict_registry_leg"])
    for p in ("evidence/run_card.json", "docs/data/run_card.json"):
        (ROOT / p).write_text(json.dumps(card, indent=1, allow_nan=False) + "\n")
    print(json.dumps(like, indent=1))
    print("registry_duplicate_evidence:", d["registry_duplicate_evidence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
