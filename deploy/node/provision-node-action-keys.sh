#!/usr/bin/env bash
set -euo pipefail

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

kubectl_context() {
    if [[ -n "${KUBECTL_CONTEXT}" ]]; then
        command kubectl --context "${KUBECTL_CONTEXT}" "$@"
    else
        command kubectl "$@"
    fi
}

mapfile -t nodes < <(
    kubectl_context get nodes \
        -l "sagemaker.amazonaws.com/cluster-name=${HYPERPOD_CLUSTER}" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
        sort
)
((${#nodes[@]} > 0)) || {
    printf 'ERROR: no nodes found for HyperPod cluster %s\n' \
        "${HYPERPOD_CLUSTER}" >&2
    exit 1
}

key_dir="$(mktemp -d)"
cleanup() {
    find "${key_dir}" -type f -exec shred -u {} + 2>/dev/null || true
    rmdir "${key_dir}" 2>/dev/null || true
}
trap cleanup EXIT
chmod 0700 "${key_dir}"

if kubectl_context -n "${NAMESPACE}" get secret \
    "${SECRET_NAME}" >/dev/null 2>&1; then
    existing_json="${key_dir}/existing.json"
    kubectl_context -n "${NAMESPACE}" get secret "${SECRET_NAME}" \
        -o json >"${existing_json}"
    chmod 0600 "${existing_json}"
    python3 - "${existing_json}" "${key_dir}" "${nodes[@]}" <<'PY'
import base64
import json
import os
import sys

source = sys.argv[1]
target = sys.argv[2]
nodes = set(sys.argv[3:])
data = json.load(open(source, encoding="utf-8")).get("data") or {}
for node in nodes & set(data):
    path = os.path.join(target, node)
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(data[node]))
    os.chmod(path, 0o600)
PY
    shred -u "${existing_json}"
fi

for node in "${nodes[@]}"; do
    [[ "${node}" =~ ^[A-Za-z0-9._-]+$ ]] || {
        printf 'ERROR: node name cannot be a Secret key: %s\n' \
            "${node}" >&2
        exit 1
    }
    if [[ "${node}" == "${ROTATE_NODE}" ]]; then
        python3 -c 'import secrets; print(secrets.token_hex(32), end="")' \
            >"${key_dir}/${node}"
    elif [[ ! -s "${key_dir}/${node}" ]]; then
        PYTHONPATH="${REPO_DIR}/src" python3 - \
            "${MASTER_FILE}" "${CLUSTER_ID}" "${node}" \
            >"${key_dir}/${node}" <<'PY'
import sys

from gpu_fault.fleet import derive_node_action_secret

master = open(sys.argv[1], encoding="utf-8").read().strip()
print(
    derive_node_action_secret(master, sys.argv[2], sys.argv[3]),
    end="",
)
PY
    fi
    chmod 0600 "${key_dir}/${node}"
done

kubectl_context -n "${NAMESPACE}" create secret generic \
    "${SECRET_NAME}" \
    --from-file="${key_dir}" \
    --dry-run=client -o yaml |
    kubectl_context apply -f - >/dev/null

mapfile -t stored < <(
    kubectl_context -n "${NAMESPACE}" get secret "${SECRET_NAME}" \
        -o json |
        python3 -c '
import json, sys
for key in sorted((json.load(sys.stdin).get("data") or {})):
    print(key)
' |
        sort
)
[[ "${stored[*]}" == "${nodes[*]}" ]] || {
    printf 'ERROR: node key Secret does not match the current nodes\n' >&2
    exit 1
}
printf 'provisioned %d node action key(s) in %s/%s\n' \
    "${#nodes[@]}" "${NAMESPACE}" "${SECRET_NAME}"

if [[ -n "${CONTROL_PLANE_KUBECONFIG}" ||
    -n "${CONTROL_PLANE_CONTEXT}" ]]; then
    control_args=()
    if [[ -n "${CONTROL_PLANE_KUBECONFIG}" ]]; then
        control_args+=(--kubeconfig "${CONTROL_PLANE_KUBECONFIG}")
    fi
    if [[ -n "${CONTROL_PLANE_CONTEXT}" ]]; then
        control_args+=(--context "${CONTROL_PLANE_CONTEXT}")
    fi
    command kubectl "${control_args[@]}" \
        -n "${CONTROL_PLANE_NAMESPACE}" create secret generic \
        "${SECRET_NAME}" \
        --from-file="${key_dir}" \
        --dry-run=client -o yaml |
        command kubectl "${control_args[@]}" apply -f - >/dev/null
    printf 'synchronized %s/%s on the control plane\n' \
        "${CONTROL_PLANE_NAMESPACE}" "${SECRET_NAME}"
fi
