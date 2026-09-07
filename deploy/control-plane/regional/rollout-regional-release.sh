#!/usr/bin/env bash
# Thin launcher for the regional release orchestrator.
#
# The orchestrator is the `gpu_fault_release` package under `src/`. This file
# keeps the path the administrator CLI and the release driver exec
# (`deploy/control-plane/regional/rollout-regional-release.sh <command> ...`)
# stable while the Python moved; argv and the environment pass through
# untouched. `src` is put on PYTHONPATH so a plain checkout runs without an
# installed wheel, exactly as the old `exec python3 <dir>/rollout_regional_release.py`
# did.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m gpu_fault_release.rollout "$@"
