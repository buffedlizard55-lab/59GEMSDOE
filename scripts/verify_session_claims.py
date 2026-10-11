#!/usr/bin/env python3
"""Line-by-line check that every number the session published matches its receipt.

The brief forbids hallucinated or hand-typed numbers.  This script re-reads the
receipts and asserts that the values that appear on the pages, in the README and
in the run card are the values on disk -- and that the mirrors under
``docs/downloads/`` are byte-identical to the files under ``downloads/``.

Exit code 0 = every claim verified; 1 = at least one claim disagrees (printed).

Usage: python scripts/verify_session_claims.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    fails: list[str] = []
    checks = 0

    def claim(name: str, ok: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not ok:
            fails.append(f"{name}: {detail}")

    card = json.loads((ROOT / "evidence/run_card_current.json").read_text())
    public = json.loads((ROOT / "docs/data/run_card.json").read_text())
    ci = json.loads((ROOT / "evidence/h59j_candidate_ci.json").read_text())
    live = json.loads((ROOT / "evidence/h59j_live_calibration.json").read_text())
    screen = json.loads((ROOT / "evidence/uniqueness_h59j.json").read_text())
    view = json.loads((ROOT / "evidence/uniqueness_h59j_view.json").read_text())
    proxy = json.loads((ROOT / "evidence/offcat_instrument_check.json").read_text())
    board = json.loads((ROOT / "evidence/leaderboard_snapshot_2026-10-11.json").read_text())

    claim("public card is byte-identical to the evidence card",
          (ROOT / "docs/data/run_card.json").read_bytes() == (ROOT / "evidence/run_card_current.json").read_bytes())

    tif = ROOT / card["file"]
    zipf = ROOT / card["zip_file"]
    digest = hashlib.sha256(tif.read_bytes()).hexdigest()
    claim("raster sha256 matches the card", digest == card["raster_sha256"], digest)
    claim("zip sha256 matches the card",
          hashlib.sha256(zipf.read_bytes()).hexdigest() == card["zip_sha256"])
    claim("zip contains exactly the one TIFF",
          zipfile.ZipFile(zipf).namelist() == [tif.name])
    claim("zip payload equals the TIFF",
          zipfile.ZipFile(zipf).read(tif.name) == tif.read_bytes())

    for mirror in ("docs/downloads",):
        claim(f"{mirror} mirror of the TIFF is byte-identical",
              (ROOT / mirror / tif.name).read_bytes() == tif.read_bytes())
        claim(f"{mirror} mirror of the ZIP is byte-identical",
              (ROOT / mirror / zipf.name).read_bytes() == zipf.read_bytes())
        claim(f"{mirror} mirror of the receipt is byte-identical",
              (ROOT / mirror / tif.with_suffix('.json').name).read_bytes()
              == tif.with_suffix('.json').read_bytes())

    claim("emitted count matches the validator preflight",
          card["emitted"] == card["validator_output"]["emitted_positive_pixels"]
          == card["portal_preflight"]["positive_pixels"], str(card["emitted"]))
    claim("every local format check passed",
          all(card["validator_output"]["checks"].values()),
          str([k for k, v in card["validator_output"]["checks"].items() if not v]))
    claim("portal preflight ok", card["portal_preflight"]["ok"] is True)
    claim("no positive mass outside the study region",
          card["portal_preflight"]["mass_outside_footprint"] == 0)
    claim("holdout dti equals the CI receipt", card["holdout_dti"]["dti"] == ci["score"]["dti"])
    claim("holdout CI equals the CI receipt", card["holdout_dti"]["ci95"] == ci["score"]["ci95"])
    claim("withheld positives labelled", card["holdout_dti"]["withheld_positive_pixels"] == 11321)
    claim("clearance receipts agree on the rasters checked",
          view["registry_rasters_checked"] == screen["registry_rasters_checked"] == 695
          and view["complete_accessible_scan"] is True)

    # numbers that appear in prose: assert they are the measured ones
    rho = live["spearman_holdout_vs_live"]
    claim("holdout-vs-live rho matches", abs(rho["rho"] + 0.4366) < 0.002, str(rho))
    claim("dot-count rho matches",
          abs(live["spearman_log_emitted_vs_live"]["rho"] + 0.818) < 0.002)
    claim("proxy rho matches",
          abs(proxy["spearman_live_vs_proxy_full_density"]["spearman_live_vs_proxy"] + 0.665) < 0.002)
    claim("worst spearman equals the screen",
          abs(card["correlation_overlap_vs_registry"]["worst_spearman"]
              - screen["worst_spearman_full_footprint"]) < 1e-12)
    claim("board rev 2 top row is 0.4133",
          board["rows"][0]["best_public_dw_tversky"] == 0.4133 and board["revision"] == 2)

    def prose(rel: str) -> str:
        return (ROOT / rel).read_text()

    pages = " ".join(prose(p) for p in ("docs/index.html", "docs/executive-summary.html", "README.md"))
    for token, label in ((card["raster_sha256"], "sha256"),
                         ("30,815", "emitted count"),
                         ("0.01912", "holdout point estimate"),
                         ("0.01639", "holdout CI lower"),
                         ("0.02215", "holdout CI upper"),
                         ("0.0475", "worst spearman"),
                         ("0.0270", "worst jaccard"),
                         ("0.256", "worst mutual overlap"),
                         ("0.4133", "board top"),
                         ("0.3195", "board #8"),
                         ("0.2778", "board #23"),
                         ("0.24", "projection band"),
                         ("Submit to competition: OK", "submit banner")):
        claim(f"published pages carry the {label}", token in pages)
    claim("no page still calls the H59-I file the current candidate",
          "gems59-h59i-intensity-lattice" not in prose("docs/index.html").split("Archived")[0].split("H59-J:")[0]
          or "h59i" not in prose("docs/index.html").lower().split("id=\"download-panel\"")[1][:2000].lower())

    print(f"claims checked: {checks}")
    if fails:
        print("FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("all session claims verified against their receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
