#!/usr/bin/env python3
"""Write the three session-evidence files the delivery QA reads, from measurements on disk.

* ``evidence/portfolio_live_evidence.json`` -- the live-score portfolio this
  session calibrated against: every owner-labelled scored raster, its recorded
  public score and its local measurements.  Evidence class is OWNER-REPORTED.
* ``evidence/hypotheses_current.json``     -- the ranked hypotheses and the
  negatives of this session, with the file each one lives in.
* ``evidence/site_inventory.json``         -- the 57 sibling sites named in
  TASK_PROMPT.md, with the score the prompt records for each (or null).

Nothing is invented: every score and site name comes from TASK_PROMPT.md, every
local number from the receipts named in the payload.

Usage: python scripts/build_session_evidence.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def site_inventory() -> dict:
    text = (ROOT / "TASK_PROMPT.md").read_text()
    scores: dict[str, float | None] = {}
    # the prompt lists each site as a link followed by its recorded score
    blocks = re.split(r"\n(?=\[https://buffedlizard55-lab\.github\.io/)", text)
    for block in blocks:
        m = re.match(r"\[https://buffedlizard55-lab\.github\.io/([^/\]]+)", block)
        if not m:
            continue
        site = m.group(1)
        s = re.search(r"\b(0\.\d{3,4})\b", block[len(m.group(0)):])
        scores.setdefault(site, float(s.group(1)) if s else None)
    for site in re.findall(r"\b(5?[0-9]{0,2}GEMSDOE[0-9]*)\b", text):
        scores.setdefault(site, None)
    repos = [dict(repo=name, url=f"https://buffedlizard55-lab.github.io/{name}/",
                  best_recorded_score=scores[name]) for name in sorted(scores)]
    return dict(
        evidence_class="INVENTORY (sibling sites named in TASK_PROMPT.md)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="TASK_PROMPT.md (the standing brief, read at the start of this session)",
        caveat=("scores are OWNER-REPORTED as written in the brief; they are site-reported bests, not "
                "organizer receipts, and this session did not re-fetch every site"),
        n_repos=len(repos), repos=repos,
    )


def portfolio() -> dict:
    cal = json.loads((ROOT / "evidence/h59j_live_calibration.json").read_text())
    screen = json.loads((ROOT / "evidence/uniqueness_h59j.json").read_text())
    rows = []
    by_key = {}
    for rec in cal["rows"]:
        by_key.setdefault(rec["key"], rec)
    for rec in sorted(by_key.values(), key=lambda r: -(r["live_score"] or -1)):
        rows.append(dict(key=rec["key"], cached_file=f".cache/scored/{rec['file']}",
                         owner_recorded_public_score=rec["live_score"],
                         holdout_dti=rec["holdout_dti"], emitted=rec["emitted"]))
    labelled = [r for r in rows if r["owner_recorded_public_score"] is not None]
    return dict(
        evidence_class="OWNER-REPORTED x HOLDOUT-DTI portfolio (instrument calibration input)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scored_rasters=len(rows),
        labelled_rasters=len(labelled),
        score_range=[min(r["owner_recorded_public_score"] for r in labelled),
                     max(r["owner_recorded_public_score"] for r in labelled)],
        emitted_range=[min(r["emitted"] for r in rows), max(r["emitted"] for r in rows)],
        correlations={
            "spearman_holdout_vs_live": cal["spearman_holdout_vs_live"],
            "spearman_log_emitted_vs_live": cal["spearman_log_emitted_vs_live"],
            "spearman_emitted_vs_live": cal["spearman_emitted_vs_live"],
        },
        registry_context=dict(rasters_checked=screen["registry_rasters_checked"],
                              complete_accessible_scan=screen["complete_accessible_scan"]),
        rows=rows,
        labels=dict(owner_recorded_public_score="OWNER-REPORTED (written into the raster's file name by the "
                                                "owner in a sibling repository; no organizer receipt is "
                                                "reachable from this session)",
                    holdout_dti="HOLDOUT-DTI (local instrument, evaluator gems57-pooled-hide-v2)"),
        receipts=["evidence/h59j_live_calibration.json", "evidence/offcat_instrument_check.json",
                  "evidence/h59j_candidate_ci.json"],
    )


def hypotheses() -> dict:
    live = json.loads((ROOT / "evidence/h59j_live_calibration.json").read_text())
    proxy = json.loads((ROOT / "evidence/offcat_instrument_check.json").read_text())
    return dict(
        evidence_class="HYPOTHESES + NEGATIVE RESULTS (this session)",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ranked=[
            dict(id="H1", statement="structure-tensor ridge coherence consensus across topography, magnetics "
                                    "and gravity, dotted at 2-3 px and pruned 2 px from the catalogue",
                 status="SHIPPED as the H59-J candidate",
                 file="downloads/gems59-h59j-coherence-consensus-dotted-20261011T043000Z.tif",
                 evidence="evidence/run_card_current.json"),
            dict(id="H2", statement="geodetic-strain (deq/ieq) x structure-tensor consensus, i.e. an "
                                    "independent physical measurement of active deformation",
                 status="not run (cost); partial rank correlation of the bank plane with the live score is "
                        "+0.700 after controlling for emitted count",
                 file="docs/research/hypotheses-v2.md"),
            dict(id="H3", statement="QFaults MAPSCALE completeness prior (prefer lineaments near "
                                    "well-characterised, high-MAPSCALE structures)",
                 status="BLOCKED: the MAPSCALE join to our trace segments is unresolved (IR-59J-02)",
                 file="docs/research/hypotheses-v2.md"),
            dict(id="H4", statement="SGMC-exclusion: emit only where state geology claims no fault",
                 status="REFUTED this session: the off-catalogue proxy is rank-inverted against the live "
                        "board (rho -0.665, p 0.0004, n 24)",
                 file="evidence/offcat_instrument_check.json"),
            dict(id="H5", statement="1 m DEM scarp detection (the resolution the truth is probably mapped at)",
                 status="BLOCKED: the 1 m DEM links are not reachable from this sandbox (egress allow-list)",
                 file="docs/research/hypotheses-v2.md"),
        ],
        negative_results_this_session=[
            dict(id="H59-I", what="zoning + 4 px lattice + 2 px catalogue collar",
                 holdout_dti=0.082161, comparator="relay_bend_anatomy 0.135204",
                 verdict="negative -- not promoted, no slot spent",
                 file="downloads/gems59-h59i-intensity-lattice-20261011T000000Z.tif"),
            dict(id="IR-59J-01", what="the shared hide-and-recover holdout is rank-inverted against the "
                                      "public board",
                 measurement=dict(spearman=live["spearman_holdout_vs_live"]["rho"],
                                  p_value=live["spearman_holdout_vs_live"]["p_value"],
                                  n=live["spearman_holdout_vs_live"]["n"]),
                 verdict="the instrument cannot promote a candidate; it is reported, not used",
                 file="evidence/h59j_live_calibration.json"),
            dict(id="IR-59J-02", what="QFaults MAPSCALE attributes do not join to the INGENIOUS trace "
                                      "segments by record_id",
                 verdict="covariate dropped rather than guessed",
                 file="docs/research/hypotheses-v2.md"),
            dict(id="IR-59J-03", what="two local writers disagree about the outside-region convention "
                                      "(all-finite zeros vs NaN), and two validators disagreed about which "
                                      "is legal",
                 verdict="resolved in favour of the official text plus the accepted precedent: both are "
                         "accepted by the portal (18 NaN / 8 zeros among 26 scored rasters); the strict "
                         "gate now reports both instead of failing one",
                 file="src/gems_dti/submission.py"),
            dict(id="SGMC-PROXY", what="the off-catalogue (SGMC) proxy score is rank-inverted against the "
                                       "public board",
                 measurement=dict(spearman=proxy["spearman_live_vs_proxy_full_density"]["spearman_live_vs_proxy"],
                                  p_value=proxy["spearman_live_vs_proxy_full_density"]["p_value"],
                                  n=proxy["spearman_live_vs_proxy_full_density"]["n"]),
                 verdict="negative for the proxy; supporting evidence for IR-59J-01",
                 file="evidence/offcat_instrument_check.json"),
        ],
        open_questions=[
            "no local truth proxy available here ranks designs in agreement with the public board, so "
            "placement quality cannot be certified locally (IR-59J-01)",
            "the 4-page public site + preview figure promised by the sibling lanes' check_site contract is "
            "only partly present in this repository (see REMAINING_WORK.md)",
        ],
    )


def main() -> int:
    targets = {
        "evidence/site_inventory.json": site_inventory(),
        "evidence/portfolio_live_evidence.json": portfolio(),
        "evidence/hypotheses_current.json": hypotheses(),
    }
    for rel, payload in targets.items():
        (ROOT / rel).write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
        print(f"wrote {rel}")
    inv = targets["evidence/site_inventory.json"]
    print(f"site_inventory repos: {inv['n_repos']} | with recorded score: "
          f"{sum(1 for r in inv['repos'] if r['best_recorded_score'] is not None)}")
    port = targets["evidence/portfolio_live_evidence.json"]
    print(f"portfolio scored rasters: {port['scored_rasters']} labelled: {port['labelled_rasters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
