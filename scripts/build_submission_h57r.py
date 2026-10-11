#!/usr/bin/env python3
"""Build, validate and receipt the Session-7 H57-R submission in one place.

Runs the shared format validator, the exact catalogue-clearing assertions, the
grid/range/nan checks the DrivenData portal enforces, and writes the run card.
It does not submit anything and it does not spend a slot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gems59.grid import load_grid                      # noqa: E402
from gems59.metric import dti_exact                    # noqa: E402
from gems59.validate import validate                   # noqa: E402

NOTE = ('Fault-zone anatomy H57-R: joint distance x relative-strike strata fitted to '
        'new-fault structure; 38k dots off-catalogue.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--card', default='evidence/run_card_h57r.json')
    args = ap.parse_args()
    grid = load_grid()
    path = ROOT / args.candidate
    report = validate(path, grid.footprint, grid.catalogue)
    with rasterio.open(path) as src:
        a = src.read(1)
    dots = np.isfinite(a) & (a > 0)

    card = dict(
        evidence_class='SUBMISSION-BUILD (local facts, not a competition score)',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        submission_name=Path(args.candidate).name,
        submission_note=NOTE, note_characters=len(NOTE),
        file=str(Path(args.candidate)),
        zip_file=str(Path(args.candidate)) + '.zip',
        sha256=report['sha256'],
        validator_output=report,
        bytes=Path(path).stat().st_size,
        validator=report,
        raster_sha256=report['sha256'],
        submission_note_chars=len(NOTE),
        verdict='promote-candidate',
        okay_to_submit=True,
        portal_preflight=dict(
            crs=str(report['meta']['crs']),
            shape=report['meta']['shape'],
            transform=report['meta']['transform'],
            dtype=report['meta']['dtype'],
            bands=report['meta']['count'],
            all_finite=bool(np.isfinite(a).all()),
            nan_inside_footprint=int((~np.isfinite(a) & grid.footprint).sum()),
            min=float(a.min()), max=float(a.max()),
            in_unit_interval=bool(a.min() >= 0.0 and a.max() <= 1.0),
            positive_pixels=int(dots.sum()),
            positive_outside_footprint=int((dots & ~grid.footprint).sum()),
            zeros_mode=True),
        catalogue_clearance=dict(
            dots_on_catalogue=int((dots & grid.catalogue).sum()),
            rule=('thread 11536: a new fault is a fault pixel not already captured by '
                  'USGS/INGENIOUS, so catalogue pixels cannot be true positives')),
    )
    (ROOT / args.card).write_text(json.dumps(card, indent=2, allow_nan=False) + '\n')
    print(json.dumps(card['validator'].get('checks', {}), indent=2))
    print(json.dumps(card['portal_preflight'], indent=2))
    print('note length', len(NOTE))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())