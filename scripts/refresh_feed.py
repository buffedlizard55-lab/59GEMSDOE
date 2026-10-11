#!/usr/bin/env python3
"""Refresh a public organizer leaderboard snapshot, never a submission receipt.

Runs on the scheduled GitHub Actions runner. A failed fetch retains the last
successful snapshot and records a separate failed-attempt status; the site
builder publishes both. No authentication or submission endpoint is used, and
this script never reads or spends a competition submission slot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/"
MAX_HTML_BYTES = 4 * 1024 * 1024


class TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
            self.cell = None


def parse_leaderboard(html):
    """Parse rows present in returned HTML; never infer hidden/private scores."""
    parser = TableRows()
    parser.feed(html)
    results = []
    for cells in parser.rows:
        if not cells or not re.fullmatch(r"#?\d+", cells[0]):
            continue
        scores = [c for c in cells[1:] if re.fullmatch(r"(?:0|1)\.\d{4,}", c)]
        if len(scores) != 1:
            continue
        score = float(scores[0])
        if not 0 <= score <= 1:
            raise ValueError("organizer page score outside [0,1]")
        rank = int(cells[0].lstrip("#"))
        participant = cells[2] if len(cells) >= 4 else cells[1]
        results.append(dict(rank=rank, participant_display=participant, public_dti=score))
    if not results or results[0]["rank"] != 1:
        raise ValueError("no valid public leaderboard table; refusing to invent a feed")
    if len({row["rank"] for row in results}) != len(results):
        raise ValueError("duplicate rank in public leaderboard HTML")
    if any(a["public_dti"] < b["public_dti"] for a, b in zip(results, results[1:])):
        raise ValueError("leaderboard not sorted; parser or page has changed")
    return results


def refresh(root=ROOT):
    root = Path(root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    status = dict(
        attempted_utc=now,
        url=URL,
        method="scripts/refresh_feed.py using urllib; unauthenticated public HTML only",
        ok=False,
    )
    snapshot_path = evidence / "leaderboard_snapshot.json"
    try:
        request = Request(URL, headers={"User-Agent": "57GEMSDOE-public-source-audit/1.0"})
        with urlopen(request, timeout=30) as response:
            final_url = urlsplit(response.url)
            if final_url.scheme != "https" or final_url.hostname not in {
                "drivendata.org", "www.drivendata.org"
            }:
                raise ValueError("public board redirected outside the trusted HTTPS host")
            if "/login" in final_url.path.lower():
                raise ValueError("public board redirected to login")
            raw_html = response.read(MAX_HTML_BYTES + 1)
        if len(raw_html) > MAX_HTML_BYTES:
            raise ValueError(f"public HTML exceeds {MAX_HTML_BYTES} byte safety limit")
        html = raw_html.decode("utf-8")
        rows = parse_leaderboard(html)
        snapshot = dict(
            evidence_class=(
                "ORGANIZER-PUBLISHED public leaderboard; NOT a submission-page receipt "
                "or exact-file score attribution"
            ),
            source_url=URL,
            retrieved_utc=now,
            retrieval_method="scripts/refresh_feed.py; unauthenticated public HTML retrieval",
            rendered_html_sha256=sha256(raw_html).hexdigest(),
            rendered_html_bytes=len(raw_html),
            raw_html_retained=False,
            rows=rows,
            rendered_row_count=len(rows),
            top_public_dti=rows[0]["public_dti"],
            organizer_confirmed_submission_score=None,
            private_score=None,
            weekly_slots_remaining=None,
            receipt_attribution_available=False,
            scope=(
                "Rows parsed from the public HTML response only; not private scores, an exact-file "
                "receipt list, or an organizer-complete inventory of submissions."
            ),
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=evidence,
                prefix=".leaderboard_snapshot.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(
                    json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
                )
            temporary_path.replace(snapshot_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        status.update(
            ok=True,
            rows=len(rows),
            rendered_html_sha256=snapshot["rendered_html_sha256"],
            rendered_html_bytes=len(raw_html),
            raw_html_retained=False,
            retained_previous_snapshot=False,
        )
    except Exception as error:
        status.update(
            ok=False,
            error=f"{type(error).__name__}: {error}",
            retained_previous_snapshot=snapshot_path.is_file(),
        )
    (evidence / "feed_refresh_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return status


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    status = refresh()
    print(json.dumps(status, indent=2))
    # A failed source refresh is visible and preserves the prior snapshot; it
    # does not discard the evidence-led site or disguise the failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
