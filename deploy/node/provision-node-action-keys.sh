#!/usr/bin/env bash
set -euo pipefail
set +x
umask 0077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KUBECTL_CONTEXT="${GPU_FAULT_KUBECTL_CONTEXT:-}"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
CLUSTER_ID="${GPU_FAULT_CLUSTER_ID:-}"
HYPERPOD_CLUSTER="${GPU_FAULT_HYPERPOD_CLUSTER:-${CLUSTER_ID}}"
MASTER_FILE="${GPU_FAULT_FLEET_MASTER_FILE:-}"
SECRET_NAME="$(
    printf '%s' \
        "${GPU_FAULT_NODE_ACTION_KEYS_SECRET:-gpu-fault-node-action-keys}"
)"
ROTATE_NODE="${GPU_FAULT_ROTATE_NODE_ACTION_KEY:-}"
CONTROL_PLANE_KUBECONFIG="${GPU_FAULT_CONTROL_PLANE_KUBECONFIG:-}"
CONTROL_PLANE_CONTEXT="${GPU_FAULT_CONTROL_PLANE_CONTEXT:-}"
CONTROL_PLANE_NAMESPACE="${GPU_FAULT_CONTROL_PLANE_NAMESPACE:-${NAMESPACE}}"

for command in kubectl python3; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done
[[ -n "${CLUSTER_ID}" ]] || {
    printf 'ERROR: GPU_FAULT_CLUSTER_ID is required\n' >&2
    exit 2
}
[[ -n "${HYPERPOD_CLUSTER}" ]] || {
    printf 'ERROR: GPU_FAULT_HYPERPOD_CLUSTER is required\n' >&2
    exit 2
}
[[ -f "${MASTER_FILE}" ]] || {
    printf 'ERROR: GPU_FAULT_FLEET_MASTER_FILE is required\n' >&2
    exit 2
}

# This provisioning entry runs on the trusted deploy host, including when
# selected from a Node bundle. No fleet master or kubeconfig is sent to a Pod.
PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec python3 "${SCRIPT_DIR}/provision_node_action_keys.py" \
        --gpu-context "${KUBECTL_CONTEXT}" \
        --namespace "${NAMESPACE}" \
        --cluster-id "${CLUSTER_ID}" \
        --hyperpod-cluster "${HYPERPOD_CLUSTER}" \
        --master-file "${MASTER_FILE}" \
        --secret-name "${SECRET_NAME}" \
        --rotate-node "${ROTATE_NODE}" \
        --cpu-kubeconfig "${CONTROL_PLANE_KUBECONFIG}" \
        --cpu-context "${CONTROL_PLANE_CONTEXT}" \
        --cpu-namespace "${CONTROL_PLANE_NAMESPACE}" \
        "$@"
