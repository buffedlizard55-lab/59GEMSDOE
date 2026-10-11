#!/usr/bin/env python3
"""Retired: diagnostic support geometry cannot waive the literal registry gate.

A historical version of this script argued that a gate firing on a dense peer
was not a duplicate. That reasoning is incompatible with the standing rule:
any rho > 0.90 or >70% forward 3-px overlap is a STOP regardless of peer
representation or support asymmetry. This command intentionally performs no
analysis, emits no clearance, and writes no candidate evidence.
"""
from __future__ import annotations


def main() -> int:
    print(
        "RETIRED / FAIL-CLOSED: representation and support-size diagnostics cannot "
        "exempt a literal registry firing. Use scripts/uniqueness_fast.py with the "
        "shared exact gate; an incomplete index/cache is not a pass."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
