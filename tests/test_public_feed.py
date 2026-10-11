import hashlib
import json

import pytest

from scripts import build_site, refresh_feed


HTML = """
<table>
  <tr><th>Rank</th><th>Team members</th><th>Participant</th><th>Best public DTI</th><th>Shared work</th></tr>
  <tr><td>#1</td><td>team one</td><td>first</td><td>0.3774</td><td></td></tr>
  <tr><td>#2</td><td>team two</td><td>second</td><td>0.3418</td><td></td></tr>
</table>
"""


def test_parse_leaderboard_captures_only_public_rows():
    rows = refresh_feed.parse_leaderboard(HTML)
    assert rows == [
        {"rank": 1, "participant_display": "first", "public_dti": 0.3774},
        {"rank": 2, "participant_display": "second", "public_dti": 0.3418},
    ]


def test_parse_leaderboard_fails_closed_on_unsorted_or_empty_table():
    with pytest.raises(ValueError, match="not sorted"):
        refresh_feed.parse_leaderboard(HTML.replace("0.3774", "0.3000"))
    with pytest.raises(ValueError, match="no valid public leaderboard"):
        refresh_feed.parse_leaderboard("<html><body>no table</body></html>")


def test_refresh_records_html_hash_and_never_claims_receipt(tmp_path, monkeypatch):
    payload = HTML.encode("utf-8")

    class Response:
        url = refresh_feed.URL

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit == refresh_feed.MAX_HTML_BYTES + 1
            return payload

    monkeypatch.setattr(refresh_feed, "urlopen", lambda *args, **kwargs: Response())
    status = refresh_feed.refresh(tmp_path)
    snapshot = json.loads((tmp_path / "evidence" / "leaderboard_snapshot.json").read_text())

    assert status["ok"] is True
    assert status["rows"] == 2
    assert status["retained_previous_snapshot"] is False
    assert snapshot["rendered_html_sha256"] == hashlib.sha256(payload).hexdigest()
    assert snapshot["raw_html_retained"] is False
    assert snapshot["organizer_confirmed_submission_score"] is None
    assert snapshot["receipt_attribution_available"] is False
    assert "not a submission-page receipt" in snapshot["evidence_class"].lower()
    assert not list((tmp_path / "evidence").glob("*.html"))


def test_failed_refresh_retains_last_snapshot_and_discloses_failure(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    snapshot_path = evidence / "leaderboard_snapshot.json"
    old_snapshot = '{"top_public_dti":0.25}\n'
    snapshot_path.write_text(old_snapshot)

    def fail(*args, **kwargs):
        raise OSError("fixture network failure")

    monkeypatch.setattr(refresh_feed, "urlopen", fail)
    status = refresh_feed.refresh(tmp_path)
    assert status["ok"] is False
    assert status["retained_previous_snapshot"] is True
    assert "fixture network failure" in status["error"]
    assert snapshot_path.read_text() == old_snapshot


@pytest.mark.parametrize(
    "redirect_url",
    [
        "http://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/",
        "https://example.com/leaderboard/",
        "https://www.drivendata.org/login/",
    ],
)
def test_refresh_rejects_untrusted_or_login_redirects(tmp_path, monkeypatch, redirect_url):
    class Response:
        url = redirect_url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            pytest.fail("must reject redirect before reading response body")

    monkeypatch.setattr(refresh_feed, "urlopen", lambda *args, **kwargs: Response())
    status = refresh_feed.refresh(tmp_path)
    assert status["ok"] is False
    assert status["retained_previous_snapshot"] is False
    assert not (tmp_path / "evidence" / "leaderboard_snapshot.json").exists()


def test_public_page_flags_stale_snapshot_after_failed_refresh():
    snapshot = {
        "source_url": refresh_feed.URL,
        "retrieved_utc": "2020-01-01T00:00:00Z",
        "scope": "Selected public rows only.",
        "rows": [{"rank": 1, "participant_display": "first", "public_dti": 0.25}],
        "top_public_dti": 0.25,
    }
    status = {
        "ok": False,
        "attempted_utc": "2020-01-02T00:00:00Z",
        "method": "scripts/refresh_feed.py using urllib; unauthenticated public HTML only",
        "retained_previous_snapshot": True,
        "error": "timeout",
    }

    page = build_site.build_public_board(snapshot, status)
    assert "Latest refresh failed" in page
    assert "STALE / CLOCK-SKEWED SNAPSHOT" in page
    assert "not a submission receipt" in page.lower()
    assert "Do not infer attribution or causality" in page
