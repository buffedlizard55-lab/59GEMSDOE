"""Retired publishers fail closed; these tests create no candidate outputs."""
from pathlib import Path
import pytest

from scripts import build_submission, build_r2_submission

PUBLISHERS = (build_submission, build_r2_submission)


@pytest.mark.parametrize("publisher", PUBLISHERS, ids=("current", "session-2-legacy"))
def test_retired_publisher_fails_without_writing_anything(publisher, capsys, tmp_path):
    before = set(tmp_path.iterdir())
    assert publisher.main() == 2
    captured = capsys.readouterr()
    notice = captured.err + captured.out
    assert "HOLD — NOT OK TO DOWNLOAD OR SUBMIT" in notice
    assert "not organizer-complete" in " ".join(notice.split())
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("publisher", PUBLISHERS, ids=("current", "session-2-legacy"))
def test_retired_module_exposes_no_model_or_clearance_runner(publisher):
    assert "creates no" in publisher.__doc__ or "no-output stub" in publisher.__doc__
    assert not hasattr(publisher, "load_clearance")
    assert not hasattr(publisher, "load_complete_registry")
    assert "renewed authorization" in publisher.__doc__ or "organizer-complete" in publisher.__doc__
