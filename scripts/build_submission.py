#!/usr/bin/env python3
"""Retired legacy submission publisher; it creates no files and does no model work.

This script's historical status text and registry counts are not the current H59-I
record. Consult ``evidence/run_card.json`` and ``README.md``: the existing TIFF is
download-valid for inspection only, not cleared to submit; 85 literal historical
overlap firings require STOP, and the latest 696-row index was not scanned against
that candidate. The prior universal-witness verification artifact is absent from
this checkout and must not be treated as confirmed. No slot was used or authorized.
"""
import sys


def main():
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
