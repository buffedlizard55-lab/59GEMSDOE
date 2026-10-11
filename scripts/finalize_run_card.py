#!/usr/bin/env python3
"""Compatibility wrapper; the single strict interpreter owns the registry verdict."""
from __future__ import annotations

from finalize_uniqueness import main

if __name__ == "__main__":
    raise SystemExit(main())
