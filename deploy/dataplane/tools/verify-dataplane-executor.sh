#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
KUBE_CONTEXT="${GPU_FAULT_KUBE_CONTEXT:-}"
SYNC_REGISTRY="${GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY:-true}"

[[ "${SYNC_REGISTRY}" == "true" || "${SYNC_REGISTRY}" == "false" ]] || {
    echo "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY must be true or false" >&2
    exit 2
}

python3 "${SCRIPT_DIR}/verify_dataplane_executor.py" "$@"

registry_args=(
    --plane gpu
    --namespace "${NAMESPACE}"
    --release-id "${GPU_FAULT_RELEASE_ID:-verified}"
)
if [[ "${SYNC_REGISTRY}" == "true" ]]; then
    if [[ -n "${KUBE_CONTEXT}" ]]; then
        registry_args+=(--context "${KUBE_CONTEXT}")
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
        "${registry_args[@]}"
fi
