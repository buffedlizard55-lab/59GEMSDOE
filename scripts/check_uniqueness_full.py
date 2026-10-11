#!/usr/bin/env python3
"""One literal audit against the indexed public owner-repository inventory.

This scoped inventory is not a complete organizer registry; private, unlinked,
external, and otherwise inaccessible rasters may be absent. No partial-cache
or reverse-overlap exception is applied.

Default candidate is the current held research TIFF. Running this check does
not fit a model, place dots, promote a file or spend a competition slot.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gems59.grid import load_grid
from gems59.uniqueness import compare_to_registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--registry',type=Path,default=ROOT/'evidence/registry_refreshed.json')
    parser.add_argument('--out',type=Path,default=ROOT/'evidence/uniqueness_current_review.json')
    parser.add_argument('--phase',choices=['surface','dots'],default='surface')
    args=parser.parse_args()
    candidate=args.candidate
    if candidate is None:
        card=json.loads((ROOT/'evidence/run_card_current.json').read_text())
        candidate=ROOT/card['file']
    def progress(i,n,row):
        if i%50==0:print(f'[registry] {args.phase} {i}/{n}',flush=True)
    report=compare_to_registry(candidate,args.registry,load_grid().footprint,progress=progress)
    report.update(phase=args.phase,reviewed_utc=datetime.now(timezone.utc).isoformat(),submission_slots_used=0)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('registry_rasters_expected','registry_rasters_checked','complete_accessible_scan','worst_spearman_full_footprint','worst_dot_overlap','duplicate_count','unique','stop_required')},indent=2),flush=True)
    # Negative is a completed deliverable, not an exception or a hidden failed scan.
    return 0 if report['complete_accessible_scan'] else 2


if __name__=='__main__':raise SystemExit(main())
