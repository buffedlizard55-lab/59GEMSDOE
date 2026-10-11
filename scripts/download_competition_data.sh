#!/usr/bin/env bash
# Autonomous restore through GitHub's public, hash-pinned transport bridge.
# No DrivenData credentials are available; this is not official-origin authentication.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/prepare_data.py --fetch "$@"
