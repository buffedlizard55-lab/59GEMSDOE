#!/usr/bin/env python3
"""Re-emit the H58 release with a note that fits the 140-character portal field.

The pixel array is *not* re-derived and *not* changed: this script loads the
dot array the experiment cached, proves byte-for-byte that the delivered raster
holds exactly that array, and rewrites only the filename stamp, the note and the
receipt through the shared ``submission_writer``.  The superseded trio is kept in
``evidence/superseded/`` and both hashes are recorded, so the release stays
auditable instead of silently edited.

Nothing here spends a submission slot or re-runs an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems59 import load_grid                                             # noqa: E402
from gems59.validate import validate                                          # noqa: E402
from gems59.submission_writer import write_submission                     # noqa: E402

NOTE = ("H58 damage-zone envelope: W(L)=w0*L^gamma fitted; handed obliquity tested and "
        "rejected; dots kept >2 px off the mapped catalogue.")
if len(NOTE) > 140 or not NOTE.endswith("."):    # a portal field, never a silent truncation
    raise SystemExit(f"note must fit 140 characters and end a sentence ({len(NOTE)})")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dots", default=str(ROOT / ".cache/h58_dots.npy"))
    ap.add_argument("--build", default=str(ROOT / "evidence/h58_build.json"))
    ap.add_argument("--out-tif-dir", default=str(ROOT / "docs/downloads"))
    a = ap.parse_args()
    build_path = Path(a.build)
    build = json.loads(build_path.read_text())
    grid = load_grid()
    pred = np.load(a.dots).astype(np.float32)
    old = ROOT / build["file"]
    with __import__("rasterio").open(old) as ds:
        delivered = ds.read(1)
    if not np.array_equal(delivered, pred):
        raise SystemExit("cached dot array is not the delivered raster's array; refusing to relabel")
    dec = hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"gems59-h58-damagezone-envelope-{int(pred.sum())}dots-{stamp}-{dec}-zeros"[:140]
    target = Path(a.out_tif_dir) / f"{label}.tif"
    receipt = write_submission(target, pred, ROOT / "data/official/sample_submission.tif",
                               grid.footprint, note=NOTE, name=label, catalogue=grid.catalogue,
                               metadata={**{k: build[k] for k in
                                           ("arm", "retained", "dot_cap", "fraction_pos_in_kept_zone",
                                            "width_law", "live_censored_anchor_strikes")},
                                         "evaluator": build["holdout"]["evaluator_version"],
                                         "draw_seed": 2026,
                                         "relabel_of": old.name,
                                         "predictions_recomputed": False,
                                         "no_prior_raster_used_to_build_predictions": True})
    vres = validate(target, grid.footprint, grid.catalogue)
    if not vres["all_checks_passed"] or vres["emitted_positive_pixels"] != int(pred.sum()):
        raise SystemExit("relabelled raster fails the preflight")
    if vres["meta"]["transform"] != build["validator"]["meta"]["transform"]:
        raise SystemExit("geotransform changed during relabel")
    superseded = ROOT / "evidence" / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    moved = []
    for suffix in (".tif", ".zip", ".json"):
        src = old.with_suffix(suffix)
        if src.is_file():
            shutil.move(str(src), str(superseded / src.name))
            moved.append(str((Path("evidence") / "superseded" / src.name)))
    record = dict(
        evidence_class="BUILD-MEASUREMENT (metadata relabel only)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        reason=("the emitted note was a 140-character truncation of a 149-character sentence and "
                "cut mid-phrase; the array is unchanged"),
        prediction_array_changed=False,
        aborted_first_attempt=("a first attempt raised an import error after writing the bytes; the "
                               "unreferenced trio was deleted and the array was never altered"),
        decoded_array_sha256=hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
        superseded=dict(file=build["file"], raster_sha256=build["raster_sha256"],
                        zip_sha256=build["zip_sha256"], note=build["submission_note"],
                        moved_to=moved),
        replacement=dict(file=str(target.relative_to(ROOT)), raster_sha256=receipt["sha256"],
                         zip_sha256=receipt["zip_sha256"], submission_name=label,
                         submission_note=NOTE, submission_note_chars=len(NOTE)))
    (ROOT / "evidence" / "h58_relabel.json").write_text(json.dumps(record, indent=2) + "\n")
    build.update(file=record["replacement"]["file"], raster_sha256=receipt["sha256"],
                 zip_sha256=receipt["zip_sha256"],
                 receipt_file=str(target.with_suffix(".json").relative_to(ROOT)),
                 validator=vres, submission_name=label, submission_note=NOTE,
                 submission_note_chars=len(NOTE), superseded_from=old.name,
                 relabel_record="evidence/h58_relabel.json")
    build_path.write_text(json.dumps(build, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(new_file=record["replacement"]["file"],
                          new_sha256=receipt["sha256"],
                          same_array=hashlib.sha256(
                              np.ascontiguousarray(pred).tobytes()).hexdigest()[:12],
                          note_chars=len(NOTE), validator_ok=vres["all_checks_passed"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
