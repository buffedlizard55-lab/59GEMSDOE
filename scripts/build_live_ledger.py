#!/usr/bin/env python3
"""Session H59-J step 1a -- build one live-score ledger out of three sources.

Sources
-------
1. ``TASK_PROMPT.md`` (the standing brief, verbatim) -- lines of the form
   ``<artifact name>: <score>``.  These are the owner's own reported numbers.
2. ``evidence/artifact_ledger_gemsdoe25.json`` -- a sibling ledger whose rows
   are pinned by full SHA-256 to owner-mirrored bytes (30 rows, 29 scored).
3. ``evidence/score_ledger_gemsdoe32.csv`` -- a sibling ledger with the same
   brief-derived rows plus official-leaderboard snapshots (not artifact rows).

Matching rule
-------------
A brief name is attached to a registry raster only when the normalised name is
a substring of one of the raster's published source paths and that match is
unique across the whole registry.  Ambiguous names are dropped and reported;
nothing is guessed.  Every output row carries its provenance string.

Evidence class: OWNER-REPORTED-NAME-MATCH.  These are NOT organiser receipts.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINE_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]{3,120}?)\s*:\s*([01]\.\d{4})\s*$")


def normalise(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"\.tif{1,2}$", "", n)
    return n


def parse_brief(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        name, score = m.group(1), float(m.group(2))
        if name.lower() in {"note", "url", "http", "https"}:
            continue
        rows.append(dict(name=name, score=score, source="TASK_PROMPT.md"))
    # de-duplicate identical (name, score) pairs, keep the first
    seen, out = set(), []
    for r in rows:
        key = (r["name"], r["score"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brief", default=str(ROOT / "TASK_PROMPT.md"))
    ap.add_argument("--sibling", default=str(ROOT / "evidence/artifact_ledger_gemsdoe25.json"))
    ap.add_argument("--registry", default=str(ROOT / "evidence/registry_refreshed.json"))
    ap.add_argument("--out", default=str(ROOT / "evidence/live_ledger_merged.json"))
    args = ap.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    rasters = registry["rasters"]
    # index: normalised source basename -> list of raster indices
    index: dict[str, list[int]] = {}
    for i, r in enumerate(rasters):
        for src in r["sources"]:
            base = normalise(Path(src.split(":", 1)[-1]).name)
            index.setdefault(base, []).append(i)
    # also index the full normalised path for substring matching
    path_index = []
    for i, r in enumerate(rasters):
        for src in r["sources"]:
            path_index.append((normalise(src), i))

    brief = parse_brief(Path(args.brief))
    matched, unmatched, ambiguous = [], [], []
    for row in brief:
        key = normalise(row["name"])
        hits = {i for path, i in path_index if key in path}
        if len(hits) == 1:
            i = hits.pop()
            matched.append(dict(name=row["name"], score=row["score"], raster=i,
                                sha256=rasters[i]["sha256"],
                                cache_file=rasters[i]["cache_file"],
                                source_paths=rasters[i]["sources"],
                                provenance="OWNER-REPORTED-NAME-MATCH (unique substring)",
                                match_rule="brief name is a unique substring of one source path"))
        elif len(hits) == 0:
            unmatched.append(dict(name=row["name"], score=row["score"],
                                  reason="no registry source path contains the name"))
        else:
            ambiguous.append(dict(name=row["name"], score=row["score"], n_hits=len(hits),
                                  candidates=[rasters[i]["sources"][0] for i in sorted(hits)][:6],
                                  reason="name matches several registry rasters"))

    sibling = json.loads(Path(args.sibling).read_text())
    sib_rows = [r for r in sibling["artifacts"] if r.get("score") is not None]
    by_sha = {r["sha256"]: i for i, r in enumerate(rasters)}
    sibling_matched = []
    for r in sib_rows:
        i = by_sha.get(r["sha256"])
        sibling_matched.append(dict(id=r["id"], score=float(r["score"]),
                                    sha256=r["sha256"], raster=i,
                                    provenance="SIBLING-LEDGER-SHA256-MATCH",
                                    sibling_status=r.get("status")))

    # consolidate: the sha256-keyed sibling rows win where they overlap a name match
    consolidated: dict[str, dict] = {}
    for row in matched:
        consolidated.setdefault(row["sha256"], row)
    for row in sibling_matched:
        if row["raster"] is not None:
            sha = rasters[row["raster"]]["sha256"]
            entry = consolidated.setdefault(sha, dict(
                name=row["id"], score=row["score"], raster=row["raster"], sha256=sha,
                cache_file=rasters[row["raster"]]["cache_file"],
                source_paths=rasters[row["raster"]]["sources"],
                provenance="SIBLING-LEDGER-SHA256-MATCH",
                match_rule="full SHA-256 equality with a sibling artifact pin"))
            entry.setdefault("sibling_id", row["id"])

    result = dict(
        evidence_class="OWNER-REPORTED-NAME-MATCH / SIBLING-LEDGER-SHA256-MATCH "
                       "(scores are not organiser receipts)",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/build_live_ledger.py",
        n_brief_rows=len(brief), n_matched=len(matched), n_unmatched=len(unmatched),
        n_ambiguous=len(ambiguous), n_consolidated=len(consolidated),
        brief_unmatched=unmatched, brief_ambiguous=ambiguous,
        rows=sorted(consolidated.values(), key=lambda r: -r["score"]),
    )
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[ledger] brief rows {len(brief)} matched {len(matched)} "
          f"ambiguous {len(ambiguous)} unmatched {len(unmatched)}")
    print(f"[ledger] consolidated (unique rasters) {len(consolidated)} -> {out}")
    for row in result["rows"][:45]:
        print(f"  {row['score']:.4f}  {row['name'][:52]:<52s} {row['provenance'][:34]}")
    if ambiguous:
        print("[ledger] ambiguous names (dropped):")
        for a in ambiguous:
            print(f"  {a['name'][:60]:<60s} hits={a['n_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
