#!/usr/bin/env python3
"""Retired session-4 site generator; kept as a safe no-op entry point.

The historical implementation could rewrite current evidence and publish a
stale positive submission banner before reaching its retirement guard. Its
outputs remain archived in the repository, but this command intentionally has
no generation or file-writing behavior. Use ``scripts/build_site.py`` for the
current evidence-led, negative-status site.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Historical session-4 site generator is retired and has no side effects. "
        "Use scripts/build_site.py for the current negative release card."
    )


if __name__ == "__main__":
    main()
