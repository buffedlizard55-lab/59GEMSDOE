"""The checked-in executive site must stay internally sound and HOLD-first."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_site import inspect_site  # noqa: E402


def test_published_site_is_fail_closed_and_has_no_broken_links():
    assert inspect_site(ROOT / "docs") == []
