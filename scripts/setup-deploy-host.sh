#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3.12}"

exec "${PYTHON}" "${SCRIPT_DIR}/setup_deploy_host.py" \
  --repo-root "${REPO_ROOT}" \
  "$@"
