#!/usr/bin/env python3
"""Retired session-2 publisher; kept as a no-output stub.

The original is archived at `evidence/history/build_r2_submission_legacy_2026-10-10.py`.
It wrote a TIFF before its uniqueness screen and changed the parallel-run rule by
excluding this repository's same-lane rasters (and treating Jaccard as a stop).
The current protocol compares every indexed raster and stops only on its literal
rank-correlation / forward 3-pixel rules; the public inventory is not
organizer-complete. Current status: HOLD — NOT OK TO DOWNLOAD OR SUBMIT. This
entry point performs no fit, download, placement, or file write.
"""
import sys


def main() -> int:
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
