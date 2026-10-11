"""Turn a probability raster into a validated competition GeoTIFF.

Input : a .npy float array OR a single-band GeoTIFF on the official grid, values intended in [0,1].
Output: submissions/<name>.tif  + submissions/<name>.receipt.json   (only if all checks pass)

Usage:
  PYTHONPATH=src python scripts/make_submission.py --prob path/to/prob.npy \
      --template data/sample_submission.tif --name my-unique-name --note "short comment"

Values are clipped only if they are within float rounding (1e-6) of [0,1]; anything else fails
loudly. Nothing is silently clipped, because clipping would hide a modelling bug.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from gems_dti.submission import SubmissionError, write_submission


def load_prob(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        with rasterio.open(path) as src:
            arr = src.read(1)
    arr = np.asarray(arr, dtype=np.float64)
    tol = 1e-6
    if np.all(np.isfinite(arr)) and arr.min() >= -tol and arr.max() <= 1 + tol:
        arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--out-dir", default="submissions", type=Path)
    ap.add_argument("--publish-dir", default="docs/downloads", type=Path,
                    help="copy the validated TIF + receipt here and update manifest.json (GitHub Pages)")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    out = args.out_dir / f"{args.name}.tif"
    try:
        receipt = write_submission(load_prob(args.prob), out, args.template, args.name, args.note)
    except SubmissionError as exc:
        raise SystemExit(f"REFUSED: {exc}")
    print(f"wrote {out}  sha256={receipt['sha256']}  bytes={receipt['bytes']}")
    if not args.no_publish:
        pub = args.publish_dir
        pub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, pub / out.name)
        shutil.copy2(out.with_suffix(".receipt.json"), pub / out.with_suffix(".receipt.json").name)
        man_path = pub / "manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {"files": []}
        man["files"] = [f for f in man["files"] if f["name"] != args.name]
        man["files"].append({
            "name": args.name,
            "tif": out.name,
            "receipt": out.with_suffix(".receipt.json").name,
            "sha256": receipt["sha256"],
            "bytes": receipt["bytes"],
            "note": args.note,
            "validated": True,
            "organizer_score": None,
        })
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        print(f"published to {pub}/ and updated manifest.json")


if __name__ == "__main__":
    main()
