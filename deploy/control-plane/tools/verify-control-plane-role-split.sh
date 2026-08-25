#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"

python3 "${SCRIPT_DIR}/verify_control_plane_role_split.py" "$@"

registry_args=(
    --plane cpu
    --namespace "${NAMESPACE}"
    --release-id "${GPU_FAULT_RELEASE_ID:-verified}"
)
if [[ -n "${KUBECONFIG:-}" ]]; then
    registry_args+=(--kubeconfig "${KUBECONFIG}")
fi
PYTHONDONTWRITEBYTECODE=1 python3 \
    "${SCRIPT_DIR}/sync_installed_resource_registry.py" \
    "${registry_args[@]}"
