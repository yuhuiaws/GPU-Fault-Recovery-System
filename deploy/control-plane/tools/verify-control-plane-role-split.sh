#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
SYNC_REGISTRY="${GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY:-true}"

[[ "${SYNC_REGISTRY}" == "true" || "${SYNC_REGISTRY}" == "false" ]] || {
    echo "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY must be true or false" >&2
    exit 2
}

python3 "${SCRIPT_DIR}/verify_control_plane_role_split.py" "$@"

registry_args=(
    --plane cpu
    --namespace "${NAMESPACE}"
    --release-id "${GPU_FAULT_RELEASE_ID:-verified}"
)
if [[ "${SYNC_REGISTRY}" == "true" ]]; then
    if [[ -n "${KUBECONFIG:-}" ]]; then
        registry_args+=(--kubeconfig "${KUBECONFIG}")
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${SCRIPT_DIR}/sync_installed_resource_registry.py" \
        "${registry_args[@]}"
fi
