"""Validate a candidate GeoTIFF against the official format and a template raster.

Usage:  PYTHONPATH=src python scripts/validate_submission.py <file.tif> <template.tif>
Exit code 0 = passes every check; 1 = fails (reasons printed).
"""

import json
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from gems_dti.submission import SubmissionError, validate_file


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        receipt = validate_file(Path(sys.argv[1]), Path(sys.argv[2]))
    except SubmissionError as exc:
        print(json.dumps({"passed": False, "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps({"passed": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
