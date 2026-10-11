#!/usr/bin/env python3
"""Fold the completed registry restore and uniqueness screen into the run card."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
card = json.loads((ROOT / "evidence/run_card.json").read_text())
restore = json.loads((ROOT / "evidence/registry_cache_restore.json").read_text())
card["registry"] = dict(
    evidence_class="REGISTRY-MEASUREMENT",
    pinned_index="evidence/registry_refreshed.json (695 rasters, snapshot 2026-10-10T20:01Z)",
    cache_restore_complete=bool(restore["complete"]),
    restored=restore["restored"], already_cached=restore["already_cached"], errors=restore["n_errors"],
    uniqueness="evidence/uniqueness_h59i.json (scripts/uniqueness_fast.py)")
uq_path = ROOT / "evidence/uniqueness_h59i.json"
if uq_path.exists():
    uq = json.loads(uq_path.read_text())
    card["registry"]["uniqueness"] = dict(
        registry_rasters_checked=uq["registry_rasters_checked"],
        complete_accessible_scan=uq["complete_accessible_scan"],
        worst_spearman_full_footprint=uq["worst_spearman_full_footprint"],
        worst_rho_submission=uq["worst_rho_submission"],
        worst_dot_overlap=uq["worst_dot_overlap"],
        worst_overlap_submission=uq["worst_overlap_submission"],
        worst_jaccard_dot_sets=uq["worst_jaccard_dot_sets"],
        duplicate_count=uq["duplicate_count"], unique=uq["unique"], scope=uq["scope"],
        duplicate_rows=uq["duplicate_rows"])
    if not uq["unique"]:
        card["verdict"] = "negative"
for p in ("evidence/run_card.json", "docs/data/run_card.json"):
    (ROOT / p).write_text(json.dumps(card, indent=1, allow_nan=False) + "\n")
print("run card updated; uniqueness present:", uq_path.exists())
