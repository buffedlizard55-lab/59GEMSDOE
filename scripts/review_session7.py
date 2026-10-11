#!/usr/bin/env python3
"""Three review passes over the Session-7 release, recorded as evidence.

Pass 1 — numbers: every headline number in README.md and the run card is
re-derived from the receipts and compared.
Pass 2 — claims: the rendered site and README are scanned for forbidden or
stale claims; both download mirror trees must be byte-identical; the retracted
140,000-dot artifact must be absent everywhere and unreferenced.
Pass 3 — behaviour: the shared uniqueness reading is exercised, the fail-closed
gates are asserted, and the full pytest suite + site QA are run.

Writes ``evidence/review_passes_h57m.json``. Exits non-zero on any failed check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
sys.path.insert(0, str(ROOT / "src"))

RETRACTED = "gems59-h57m-relay-band-20261010T215104Z-gems59-h57m.tif"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pass_numbers(README: str, card: dict) -> dict:
    cert = json.loads((EVID / "h57m_uniqueness_certificate.json").read_text())
    univ = json.loads((EVID / "gate_universality.json").read_text())
    emission = json.loads((EVID / "h57m_emission.json").read_text())
    validation = json.loads((EVID / "h57m_validation.json").read_text())
    tif = ROOT / card["file"]
    digest = sha256(tif)

    by_dots = {r["submission"]: r for r in univ["rows"]}
    lattice = by_dots["13GEMSDOE:docs/downloads/13gems_20261001_r13-lattice-s5_v2_zerofill.tif"]

    checks = {
        "tif_sha_matches_card_receipts": digest == card["raster_sha256"] == validation["sha256"],
        "card_dots_is_43950": card["final_dots"] == emission["dots"] == 43950,
        "validator_all_pass": card["validator_output"]["all_checks_passed"] is True,
        "certificate_is_same_bytes": cert["candidate_sha256"] == digest,
        "literal_firings_43_of_213": cert["literal_reading"]["forward_overlap_firings"] == 43
                                    and cert["rasters_measured_this_run"] == 213,
        "literal_worst_forward_1": cert["literal_reading"]["worst_forward_overlap"] == 1.0,
        "worst_spearman_0_12678": abs(cert["literal_reading"]["worst_spearman_full_footprint"] - 0.12677896067271274) < 1e-12,
        "like_for_like_20_firings_worst_0_9992": cert["like_for_like_reading"]["forward_overlap_firings"] == 20
                                                 and abs(cert["like_for_like_reading"]["worst_forward_overlap_dot_fields"] - 0.9992036405005689) < 1e-12,
        "blanket_40_total_43": univ["blanket_rasters"] == 40 and univ["forward_firings_blanket"] == 40
                               and univ["forward_firings_localised"] == 3 and univ["forward_firings_total"] == 43,
        "zero_two_direction_firings": univ["rasters_firing_in_both_directions"] == 0,
        "worst_reverse_0_698725": abs(univ["worst_reverse"] - 0.698725) < 1e-6,
        "best_known_control_116_of_213": univ["best_known_control"]["forward_overlap_firings"] == 116
                                         and univ["best_known_control"]["rasters_measured"] == 213,
        "lattice_dilated_coverage_0_9987": abs(lattice["dil_coverage"] - 0.9987132726822701) < 1e-12
                                           and lattice["their_dots"] == 206895,
        "certificate_pins_696": cert["registry_rasters_pinned"] == 696,
        "curve_peak_10k_0_16441": abs(max(p["holdout_dti"] for p in emission["holdout_curve"]) - 0.1644069668693676) < 1e-12,
        "card_nearest_budget_40000": card["holdout_dti"]["nearest_measured_budget"] == 40000
                                     and card["holdout_dti"]["dti"]
                                     == dict((int(p["budget"]), p["holdout_dti"])
                                             for p in emission["holdout_curve"])[40000],
        "readme_sha_present": card["raster_sha256"] in README,
        "readme_dots_present": "43,950 dots" in README,
        "readme_213_of_696_present": "213" in README and "696" in README,
        "readme_states_zero_slots": "zero competition submission slots" in README,
    }
    for name, ok in checks.items():
        check(f"pass1:{name}", ok)
    return {"checks": checks}


def pass_claims(README: str, card: dict) -> dict:
    pages = list((ROOT / "docs").glob("*.html"))
    text = "\n".join(p.read_text() for p in pages)
    forbidden = ["OK TO DOWNLOAD AND SUBMIT", "CLEARED FOR SUBMISSION", "submit this file now"]
    offenders = [f for f in forbidden if f in text]
    check("pass2:no_forbidden_clearance_claims_on_site", not offenders, str(offenders))
    check("pass2:status_strings_present",
          "Download for research: OK" in (ROOT / "docs/index.html").read_text()
          and "Submit to competition: NO" in (ROOT / "docs/index.html").read_text())
    check("pass2:retracted_artifact_absent_everywhere",
          not any(p.name == RETRACTED
                  or (p.is_file() and RETRACTED in p.read_text(errors="ignore"))
                  for p in list((ROOT / "docs/downloads").glob("*"))
                  + list((ROOT / "downloads").glob("*"))
                  + [ROOT / "README.md", ROOT / "REMAINING_WORK.md"]))
    mirrors = []
    for name in ("file", "zip_file"):
        source = ROOT / card[name]
        for mirror in (ROOT / "downloads" / source.name, ROOT / "docs/downloads" / source.name):
            mirrors.append(mirror.is_file() and mirror.read_bytes() == source.read_bytes())
    check("pass2:both_mirror_trees_identical", all(mirrors))
    urls = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", README)
    check("pass2:readme_has_links", len(urls) > 5, f"{len(urls)} links")
    stale = [u for u in urls if "47ccc38b6bec" in u]
    check("pass2:readme_top_card_links_current_file", card["file"].split("/")[-1] in README)
    check("pass2:ledger_has_new_irregularities",
          {"IR-S7A-01", "IR-S7A-02"}.issubset(
              {i["id"] for i in json.loads((EVID / "irregularities_current.json").read_text())["irregularities"]}))
    check("pass2:session7_hypotheses_ranked",
          len(json.loads((EVID / "session7_hypotheses_h57m.json").read_text())["candidates"]) == 5)
    return {"forbidden_claims_found": offenders, "stale_top_card_links": stale}


def pass_behaviour(card: dict) -> dict:
    from gems59.uniqueness import representation_class, two_phase_summary
    rc = representation_class(206895, 5167373)
    check("pass3:lattice_classified_as_dot_field", rc["kind"] == "dot_field", str(rc))
    rows = [dict(repo="x", submission="a", their_dots=5106385,
                 my_dots_within_3px_of_theirs=1.0, their_dots_within_3px_of_mine=0.09,
                 spearman_full_footprint=0.01),
            dict(repo="y", submission="b", their_dots=1000,
                 my_dots_within_3px_of_theirs=0.1, their_dots_within_3px_of_mine=0.1,
                 spearman_full_footprint=0.02)]
    summary = two_phase_summary(rows, 5167373)
    check("pass3:two_phase_summary_literal_fires", summary["literal_reading"]["duplicate"] is True)
    check("pass3:two_phase_summary_likeforlike_clean",
          summary["like_for_like_reading"]["duplicate"] is False)
    check("pass3:fail_closed_card", card["okay_to_submit"] is False and card["verdict"] == "negative"
          and card["submission_slots_used"] == 0)
    pytest = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
                            cwd=ROOT, capture_output=True, text=True, timeout=1800)
    tail = pytest.stdout.strip().splitlines()[-1] if pytest.stdout.strip() else pytest.stderr[-200:]
    check("pass3:pytest_green", pytest.returncode == 0, tail)
    site = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/check_site.py"],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    check("pass3:check_site_green", site.returncode == 0 and '"submission_cleared": false' in site.stdout,
          site.stderr[-300:])
    return {"pytest_tail": tail, "check_site_tail": "submission_cleared=false" if site.returncode == 0 else site.stderr[-300:]}


def main() -> int:
    README = (ROOT / "README.md").read_text()
    card = json.loads((EVID / "run_card_current.json").read_text())
    report = {
        "status": "three local review passes (Session 7 — 2026-10-10)",
        "candidate": card["file"],
        "candidate_sha256": card["raster_sha256"],
        "pass1_numbers": pass_numbers(README, card),
        "pass2_claims": pass_claims(README, card),
        "pass3_behaviour": pass_behaviour(card),
        "failures": FAILURES,
        "slots_used": card["submission_slots_used"],
    }
    (EVID / "review_passes_h57m.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"failures": FAILURES}, indent=1))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
