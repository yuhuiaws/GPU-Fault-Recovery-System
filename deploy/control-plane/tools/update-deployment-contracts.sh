#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODE=update
export PYTHONDONTWRITEBYTECODE=1

usage() {
    cat <<'EOF'
Usage:
  update-deployment-contracts.sh [--check]

Without arguments:
  1. Render the regional CPU role-split manifests.
  2. Generate cleanup-inventory.json from source Manifest annotations.
  3. Run deployment contract validation.

With --check:
  Do not modify files. Fail if generated manifests or the cleanup inventory
  are stale, invalid, or inconsistent with their source Manifests.
EOF
}

while (($# > 0)); do
    case "$1" in
        --check)
            MODE=check
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

cd "${ROOT}"

if [[ "${MODE}" == update ]]; then
    "${SCRIPT_DIR}/render-control-plane-role-split.sh"
    "${PYTHON_BIN}" scripts/generate-cleanup-inventory.py
fi

"${PYTHON_BIN}" scripts/generate-cleanup-inventory.py --check
PYTHONPATH=src "${PYTHON_BIN}" -m gpu_fault.config_cli validate \
    deploy/control-plane/regional/generated
"${PYTHON_BIN}" scripts/check-deploy-layout.py
"${PYTHON_BIN}" -m pytest -q \
    tests/regional/test_cleanup_inventory.py \
    tests/regional/test_installed_resource_registry.py \
    tests/regional/test_control_plane_role_split.py

printf 'deployment contracts %s passed\n' "${MODE}"
