#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KUBECTL_CONTEXT="${GPU_FAULT_KUBECTL_CONTEXT:-}"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
CLUSTER_ID="${GPU_FAULT_CLUSTER_ID:-}"
HYPERPOD_CLUSTER="${GPU_FAULT_HYPERPOD_CLUSTER:-${CLUSTER_ID}}"
VERSION="${GPU_FAULT_VERSION:-0.10.0}"
RUNTIME_PROFILE="${GPU_FAULT_RUNTIME_PROFILE:-hyperpod-v1}"
CONFIG_DIGEST="${GPU_FAULT_INSTALLER_CONFIG_DIGEST:-}"
ARTIFACT_SHA256="${GPU_FAULT_INSTALLER_ARTIFACT_SHA256:-}"
NODE_COMPATIBILITY_DIGEST="${GPU_FAULT_NODE_COMPATIBILITY_DIGEST:-}"
INSTALLER_CONFIG_MAP="$(
    printf '%s' \
        "${GPU_FAULT_INSTALLER_CONFIG_MAP:-gpu-fault-node-installer-0100}"
)"
WHEEL_CONFIG_MAP="${GPU_FAULT_WHEEL_CONFIG_MAP:-}"
EXECUTOR_WHEEL_FILENAME="${GPU_FAULT_EXECUTOR_WHEEL_FILENAME:-gpu_fault_cluster_executor-0.10.0-py3-none-any.whl}"
DCGM_METRICS_URL="${GPU_FAULT_DCGM_METRICS_URL:-http://127.0.0.1:9400/metrics}"
FLEET_MASTER_FILE="${GPU_FAULT_FLEET_MASTER_FILE:-}"
NODE_ACTION_KEYS_SECRET="$(
    printf '%s' \
        "${GPU_FAULT_NODE_ACTION_KEYS_SECRET:-gpu-fault-node-action-keys}"
)"
DEFAULT_RUNTIME_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"
RUNTIME_IMAGE="${GPU_FAULT_RUNTIME_IMAGE:-${DEFAULT_RUNTIME_IMAGE}}"

for command in kubectl sed; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done
for value in KUBECTL_CONTEXT CLUSTER_ID HYPERPOD_CLUSTER RUNTIME_PROFILE CONFIG_DIGEST ARTIFACT_SHA256 NODE_COMPATIBILITY_DIGEST WHEEL_CONFIG_MAP; do
    [[ -n "${!value}" ]] || {
        printf 'ERROR: %s is required\n' "${value}" >&2
        exit 2
    }
done
[[ "${RUNTIME_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_RUNTIME_PROFILE\n' >&2
    exit 2
}
[[ "${CONFIG_DIGEST}" =~ ^[a-zA-Z0-9._:-]{1,128}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_CONFIG_DIGEST\n' >&2
    exit 2
}
[[ "${ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_ARTIFACT_SHA256\n' >&2
    exit 2
}
[[ "${NODE_COMPATIBILITY_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_NODE_COMPATIBILITY_DIGEST\n' >&2
    exit 2
}
[[ -n "${RUNTIME_IMAGE}" &&
    "${RUNTIME_IMAGE}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_RUNTIME_IMAGE\n' >&2
    exit 2
}
[[ "${EXECUTOR_WHEEL_FILENAME}" =~ ^[A-Za-z0-9_.-]+\.whl$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_EXECUTOR_WHEEL_FILENAME\n' >&2
    exit 2
}

kubectl_context() {
    command kubectl --context "${KUBECTL_CONTEXT}" "$@"
}

kubectl_context -n "${NAMESPACE}" get configmap \
    "${INSTALLER_CONFIG_MAP}" >/dev/null
kubectl_context -n "${NAMESPACE}" get configmap \
    "${WHEEL_CONFIG_MAP}" >/dev/null
kubectl_context -n "${NAMESPACE}" get secret \
    gpu-fault-regional-connection >/dev/null
if [[ -n "${FLEET_MASTER_FILE}" ]]; then
    GPU_FAULT_KUBECTL_CONTEXT="${KUBECTL_CONTEXT}" \
    GPU_FAULT_NAMESPACE="${NAMESPACE}" \
    GPU_FAULT_CLUSTER_ID="${CLUSTER_ID}" \
    GPU_FAULT_HYPERPOD_CLUSTER="${HYPERPOD_CLUSTER}" \
    GPU_FAULT_FLEET_MASTER_FILE="${FLEET_MASTER_FILE}" \
    GPU_FAULT_NODE_ACTION_KEYS_SECRET="${NODE_ACTION_KEYS_SECRET}" \
        "${SCRIPT_DIR}/provision-node-action-keys.sh"
else
    kubectl_context -n "${NAMESPACE}" get secret \
        "${NODE_ACTION_KEYS_SECRET}" >/dev/null || {
        printf 'ERROR: %s is missing; provide GPU_FAULT_FLEET_MASTER_FILE\n' \
            "${NODE_ACTION_KEYS_SECRET}" >&2
        exit 2
    }
fi

NODE="$(
    kubectl_context get nodes \
        -l "sagemaker.amazonaws.com/cluster-name=${HYPERPOD_CLUSTER}" \
        -o jsonpath='{.items[0].metadata.name}'
)"
[[ -n "${NODE}" ]] || {
    printf 'ERROR: no Ready HyperPod node found for %s\n' \
        "${HYPERPOD_CLUSTER}" >&2
    exit 1
}
MANIFEST="$(mktemp)"
trap 'rm -f "${MANIFEST}"' EXIT

GPU_FAULT_KUBECTL_CONTEXT="${KUBECTL_CONTEXT}" \
GPU_FAULT_NAMESPACE="${NAMESPACE}" \
GPU_FAULT_CLUSTER_ID="${CLUSTER_ID}" \
GPU_FAULT_CONNECTION_MODE=regional \
GPU_FAULT_REGIONAL_CONNECTION_SECRET=gpu-fault-regional-connection \
GPU_FAULT_NODE_ACTION_KEYS_SECRET="${NODE_ACTION_KEYS_SECRET}" \
GPU_FAULT_RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
GPU_FAULT_VERSION="${VERSION}" \
GPU_FAULT_INSTALLER_CONFIG_MAP="${INSTALLER_CONFIG_MAP}" \
GPU_FAULT_INSTALLER_CONFIG_DIGEST="${CONFIG_DIGEST}" \
GPU_FAULT_INSTALLER_ARTIFACT_SHA256="${ARTIFACT_SHA256}" \
GPU_FAULT_NODE_COMPATIBILITY_DIGEST="${NODE_COMPATIBILITY_DIGEST}" \
GPU_FAULT_DCGM_METRICS_URL="${DCGM_METRICS_URL}" \
    "${SCRIPT_DIR}/run-hyperpod-installer-job.sh" \
    --node "${NODE}" --render-only >"${MANIFEST}"
TEMPLATE_SHA256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
TEMPLATE_CONFIG_MAP="gpu-fault-node-installer-template-${TEMPLATE_SHA256:0:12}"

kubectl_context -n "${NAMESPACE}" create configmap \
    "${TEMPLATE_CONFIG_MAP}" \
    --from-file="job.yaml=${MANIFEST}" \
    --dry-run=client -o yaml |
    kubectl_context apply -f -

sed \
    -e "s#REPLACE_WITH_CLUSTER_ID#${CLUSTER_ID}#g" \
    -e "s#REPLACE_WITH_HYPERPOD_CLUSTER#${HYPERPOD_CLUSTER}#g" \
    -e "s#REPLACE_WITH_INSTALLER_VERSION#${VERSION}#g" \
    -e "s#REPLACE_WITH_INSTALLER_CONFIG_DIGEST#${CONFIG_DIGEST}#g" \
    -e "s#REPLACE_WITH_INSTALLER_ARTIFACT_SHA256#${ARTIFACT_SHA256}#g" \
    -e "s#REPLACE_WITH_INSTALLER_TEMPLATE_CONFIG_MAP#${TEMPLATE_CONFIG_MAP}#g" \
    -e "s#REPLACE_WITH_DCGM_METRICS_URL#${DCGM_METRICS_URL}#g" \
    -e "s#gpu-fault-executor-wheel-0100#${WHEEL_CONFIG_MAP}#g" \
    -e "s#gpu_fault_cluster_executor-0.10.0-py3-none-any.whl#${EXECUTOR_WHEEL_FILENAME}#g" \
    -e "s#${DEFAULT_RUNTIME_IMAGE}#${RUNTIME_IMAGE}#g" \
    "${REPO_DIR}/deploy/dataplane/node-installer-reconciler.yaml" |
    kubectl_context apply -f -

kubectl_context -n "${NAMESPACE}" rollout status \
    deployment/gpu-fault-node-installer-reconciler --timeout=10m

python3 \
    "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
    --plane gpu \
    --context "${KUBECTL_CONTEXT}" \
    --namespace "${NAMESPACE}" \
    --release-id "${ARTIFACT_SHA256:0:12}"
