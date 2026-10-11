#!/usr/bin/env python3
"""Compatibility CLI delegating to the shared literal registry gate.

The previous implementation used a private stride-sampled rho and a
representation-aware overlap exemption. Those measurements cannot clear a
candidate under the standing protocol. This wrapper accepts the legacy
``--candidate`` interface but runs ``gems59.gates.lane_uniqueness_report`` via
``scripts/uniqueness_fast.py`` (the name is retained for compatibility only).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uniqueness_fast import main as shared_gate_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--registry", default="evidence/registry_refreshed_20261010T2001.json")
    parser.add_argument("--out", default="evidence/lane_gate_dots.json")
    parser.add_argument("--phase", choices=("surface", "dots"), default="dots")
    args = parser.parse_args()
    sys.argv = [
        sys.argv[0], args.candidate,
        "--phase", args.phase,
        "--registry", args.registry,
        "--out", args.out,
    ]
    return shared_gate_main()


if __name__ == "__main__":
    raise SystemExit(main())
