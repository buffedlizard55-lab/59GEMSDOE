"""Thin re-export so scripts can import the validator without a path hack."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
try:
    import submission_writer as _sw  # noqa: F401
except ImportError:  # pragma: no cover
    _sw = None
