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
BUNDLE_SHA256="${GPU_FAULT_INSTALLER_BUNDLE_SHA256:-}"
TEMPLATE_SOURCE_SHA256="${GPU_FAULT_INSTALLER_TEMPLATE_SHA256:-}"
TEMPLATE_CONFIG_MAP_OVERRIDE="${GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP:-}"
NODE_COMPATIBILITY_DIGEST="${GPU_FAULT_NODE_COMPATIBILITY_DIGEST:-}"
MAX_UNAVAILABLE="${GPU_FAULT_INSTALLER_MAX_UNAVAILABLE:-1}"
ACTIVE_DEADLINE_SECONDS="$(
    printf '%s' "${GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS:-840}"
)"
ALLOWED_NODES="${GPU_FAULT_INSTALLER_ALLOWED_NODES-*}"
SYNC_REGISTRY="${GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY:-true}"
WAIT_FOR_ROLLOUT="${GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT:-true}"
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
NODE_INSTALLER_IMAGE="${GPU_FAULT_NODE_INSTALLER_IMAGE:-}"
PREFLIGHT_ONLY="${GPU_FAULT_RECONCILER_PREFLIGHT_ONLY:-false}"
REQUIRE_ROLLBACK_SLOT="${GPU_FAULT_REQUIRE_ROLLBACK_SLOT:-false}"
# Products an earlier run of the same release left behind, as advisory hints:
# the node set (by digest) whose action keys it provisioned, and the template
# ConfigMap it rendered. Both are honoured only after this run's own checks --
# the live node list must still hash to the recorded digest, and the ConfigMap
# must still carry the content its name promises -- so a stale hint costs one
# read and falls back to the full path. PRODUCTS_FILE is where this run reports
# its own products (names and digests only, never key material) for the next.
REUSE_NODE_SET_SHA256="${GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256:-}"
REUSE_TEMPLATE_CONFIG_MAP="${GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP:-}"
PRODUCTS_FILE="${GPU_FAULT_RECONCILER_PRODUCTS_FILE:-}"

for command in awk kubectl python3 sed sha256sum; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done
for value in KUBECTL_CONTEXT CLUSTER_ID HYPERPOD_CLUSTER RUNTIME_PROFILE CONFIG_DIGEST ARTIFACT_SHA256 BUNDLE_SHA256 TEMPLATE_SOURCE_SHA256 NODE_COMPATIBILITY_DIGEST WHEEL_CONFIG_MAP NODE_INSTALLER_IMAGE; do
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
[[ "${BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_BUNDLE_SHA256\n' >&2
    exit 2
}
[[ "${TEMPLATE_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_TEMPLATE_SHA256\n' >&2
    exit 2
}
[[ -z "${TEMPLATE_CONFIG_MAP_OVERRIDE}" ||
    "${TEMPLATE_CONFIG_MAP_OVERRIDE}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP\n' >&2
    exit 2
}
[[ "${MAX_UNAVAILABLE}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_MAX_UNAVAILABLE\n' >&2
    exit 2
}
{ [[ "${ACTIVE_DEADLINE_SECONDS}" =~ ^[1-9][0-9]*$ ]] &&
    ((ACTIVE_DEADLINE_SECONDS >= 60)); } || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS\n' >&2
    exit 2
}
[[ "${ALLOWED_NODES}" == "*" ||
    "${ALLOWED_NODES}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_ALLOWED_NODES\n' >&2
    exit 2
}
[[ "${SYNC_REGISTRY}" == "true" || "${SYNC_REGISTRY}" == "false" ]] || {
    printf 'ERROR: invalid GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY\n' >&2
    exit 2
}
[[ "${WAIT_FOR_ROLLOUT}" == "true" || "${WAIT_FOR_ROLLOUT}" == "false" ]] || {
    printf 'ERROR: GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT must be true or false\n' >&2
    exit 2
}
[[ "${PREFLIGHT_ONLY}" == "true" || "${PREFLIGHT_ONLY}" == "false" ]] || {
    printf 'ERROR: GPU_FAULT_RECONCILER_PREFLIGHT_ONLY must be true or false\n' >&2
    exit 2
}
[[ "${REQUIRE_ROLLBACK_SLOT}" == "true" ||
    "${REQUIRE_ROLLBACK_SLOT}" == "false" ]] || {
    printf 'ERROR: GPU_FAULT_REQUIRE_ROLLBACK_SLOT must be true or false\n' >&2
    exit 2
}
[[ "${NODE_COMPATIBILITY_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_NODE_COMPATIBILITY_DIGEST\n' >&2
    exit 2
}
[[ -z "${REUSE_NODE_SET_SHA256}" ||
    "${REUSE_NODE_SET_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256\n' >&2
    exit 2
}
[[ -z "${REUSE_TEMPLATE_CONFIG_MAP}" ||
    "${REUSE_TEMPLATE_CONFIG_MAP}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP\n' >&2
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
mapfile -t NODES < <(
    kubectl_context get nodes \
        -l "sagemaker.amazonaws.com/cluster-name=${HYPERPOD_CLUSTER}" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
)
(( ${#NODES[@]} > 0 )) || {
    printf 'ERROR: no HyperPod node found for %s\n' \
        "${HYPERPOD_CLUSTER}" >&2
    exit 1
}
# The identity of the fleet this run acts on. A hint from an earlier run is
# only honoured when the live fleet still hashes to what that run provisioned
# for; a node added or removed since produces a different digest and the full
# path runs again. The preflight never reuses: it is the run that has to prove
# the inputs from scratch.
NODE_SET_SHA256="$(printf '%s\n' "${NODES[@]}" | sort | sha256sum | awk '{print $1}')"
REUSE_PRODUCTS="false"
if [[ "${PREFLIGHT_ONLY}" != "true" && -n "${REUSE_NODE_SET_SHA256}" &&
    "${REUSE_NODE_SET_SHA256}" == "${NODE_SET_SHA256}" ]]; then
    REUSE_PRODUCTS="true"
fi

NODE_ACTION_KEYS_PROVISIONED="false"
if [[ -n "${FLEET_MASTER_FILE}" && "${PREFLIGHT_ONLY}" != "true" ]]; then
    if [[ "${REUSE_PRODUCTS}" == "true" ]]; then
        # Provisioning derives one key per node from the fleet master and
        # mirrors the Secret to the control plane; both are functions of the
        # node set alone, which has not changed since this release did them.
        # The node-scoped key verification below still runs on the live Secret.
        printf 'reusing node action keys in %s/%s provisioned earlier in this release\n' \
            "${NAMESPACE}" "${NODE_ACTION_KEYS_SECRET}"
    else
        GPU_FAULT_KUBECTL_CONTEXT="${KUBECTL_CONTEXT}" \
        GPU_FAULT_NAMESPACE="${NAMESPACE}" \
        GPU_FAULT_CLUSTER_ID="${CLUSTER_ID}" \
        GPU_FAULT_HYPERPOD_CLUSTER="${HYPERPOD_CLUSTER}" \
        GPU_FAULT_FLEET_MASTER_FILE="${FLEET_MASTER_FILE}" \
        GPU_FAULT_NODE_ACTION_KEYS_SECRET="${NODE_ACTION_KEYS_SECRET}" \
            "${SCRIPT_DIR}/provision-node-action-keys.sh"
        NODE_ACTION_KEYS_PROVISIONED="true"
    fi
else
    kubectl_context -n "${NAMESPACE}" get secret \
        "${NODE_ACTION_KEYS_SECRET}" >/dev/null || {
        printf 'ERROR: %s is missing; provide GPU_FAULT_FLEET_MASTER_FILE\n' \
            "${NODE_ACTION_KEYS_SECRET}" >&2
        exit 2
    }
fi

kubectl_context -n "${NAMESPACE}" get secret \
    "${NODE_ACTION_KEYS_SECRET}" -o json |
    python3 -c '
import base64
import json
import sys

document = json.load(sys.stdin)
data = document.get("data") or {}
keys = set(data.keys())
missing = sorted(set(sys.argv[1:]) - keys)
if missing:
    raise SystemExit(
        "node action key Secret is missing node-scoped keys: " + ", ".join(missing)
    )
invalid = []
for node in sys.argv[1:]:
    try:
        value = base64.b64decode(data[node], validate=True)
    except (ValueError, TypeError):
        invalid.append(node)
        continue
    if len(value) < 32:
        invalid.append(node)
if invalid:
    raise SystemExit(
        "node action key Secret has invalid node-scoped keys: "
        + ", ".join(sorted(invalid))
    )
' "${NODES[@]}"

kubectl_context -n "${NAMESPACE}" get configmap \
    "${INSTALLER_CONFIG_MAP}" -o json |
    python3 -c '
import json
import sys

document = json.load(sys.stdin)
keys = set((document.get("data") or {})) | set(document.get("binaryData") or {})
if sys.argv[1] not in keys:
    raise SystemExit("installer ConfigMap is missing the candidate bundle")
' "gpu-fault-node-installer-${VERSION}.tar.gz"
kubectl_context -n "${NAMESPACE}" get configmap \
    "${WHEEL_CONFIG_MAP}" -o json |
    python3 -c '
import json
import sys

document = json.load(sys.stdin)
keys = set((document.get("data") or {})) | set(document.get("binaryData") or {})
# The release engine stores wheels xz-compressed (<wheel>.xz) to stay under the
# 1 MiB ConfigMap ceiling; older ConfigMaps hold the raw wheel under its name.
if sys.argv[1] not in keys and sys.argv[1] + ".xz" not in keys:
    raise SystemExit("wheel ConfigMap is missing the candidate Executor wheel")
' "${EXECUTOR_WHEEL_FILENAME}"
kubectl_context -n "${NAMESPACE}" get secret \
    gpu-fault-regional-connection -o json |
    python3 -c '
import base64
import json
import sys

required = {"ca.crt", "cluster-id", "cluster-token", "control-plane-url"}
data = json.load(sys.stdin).get("data") or {}
keys = set(data.keys())
missing = sorted(required - keys)
if missing:
    raise SystemExit(
        "regional connection Secret is missing keys: " + ", ".join(missing)
    )
decoded = {
    key: base64.b64decode(data[key], validate=True).decode()
    for key in required
}
if decoded["cluster-id"] != sys.argv[1]:
    raise SystemExit("regional connection Secret cluster binding is invalid")
if not decoded["control-plane-url"].startswith("https://"):
    raise SystemExit("regional connection Secret endpoint must use HTTPS")
if not decoded["cluster-token"] or not decoded["ca.crt"]:
    raise SystemExit("regional connection Secret contains an empty credential")
' "${CLUSTER_ID}"

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    kubectl_context -n "${NAMESPACE}" get jobs -o json |
        python3 -c '
import json
import sys

active = []
for item in json.load(sys.stdin).get("items", []):
    name = str((item.get("metadata") or {}).get("name") or "")
    if not name.startswith("gpu-fault-install-"):
        continue
    status = item.get("status") or {}
    if int(status.get("active") or 0):
        active.append(name)
if active:
    raise SystemExit("active node installer Jobs exist: " + ", ".join(sorted(active)))
'
fi

NODE="${NODES[0]}"
MANIFEST="$(mktemp)"
CONFIG_MAP_MANIFEST="$(mktemp)"
RECONCILER_MANIFEST="$(mktemp)"
JOB_MANIFEST="$(mktemp)"
trap 'rm -f "${MANIFEST}" "${CONFIG_MAP_MANIFEST}" "${RECONCILER_MANIFEST}" "${JOB_MANIFEST}"' EXIT

render_installer_job() {
    local node="$1"
    shift
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
    GPU_FAULT_INSTALLER_BUNDLE_SHA256="${BUNDLE_SHA256}" \
    GPU_FAULT_INSTALLER_TEMPLATE_SHA256="${TEMPLATE_SOURCE_SHA256}" \
    GPU_FAULT_NODE_INSTALLER_IMAGE="${NODE_INSTALLER_IMAGE}" \
    GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS="${ACTIVE_DEADLINE_SECONDS}" \
    GPU_FAULT_NODE_COMPATIBILITY_DIGEST="${NODE_COMPATIBILITY_DIGEST}" \
    GPU_FAULT_DCGM_METRICS_URL="${DCGM_METRICS_URL}" \
    GPU_FAULT_REQUIRE_ROLLBACK_SLOT="${REQUIRE_ROLLBACK_SLOT}" \
        "${SCRIPT_DIR}/run-hyperpod-installer-job.sh" \
        --node "${node}" "$@"
}

# The digest of the job.yaml a template ConfigMap holds right now, computed
# from the object rather than assumed.
template_config_map_content_sha256() {
    kubectl_context -n "${NAMESPACE}" get configmap "$1" -o json |
        python3 -c '
import hashlib
import json
import sys

text = (json.load(sys.stdin).get("data") or {}).get("job.yaml") or ""
if not text.strip():
    raise SystemExit("template ConfigMap has no job.yaml")
print(hashlib.sha256(text.encode()).hexdigest())
'
}

# TEMPLATE_CONTENT_SHA256 pins the exact job.yaml bytes the reconciler will
# load, so a later edit to the template ConfigMap cannot become a privileged
# Pod on every node. In the render path it is the digest of the file we put in
# the ConfigMap; with an override it is the digest of what that ConfigMap
# holds right now, computed from the object rather than assumed.
TEMPLATE_RENDERED="false"
TEMPLATE_CONFIG_MAP=""
if [[ -n "${TEMPLATE_CONFIG_MAP_OVERRIDE}" ]]; then
    TEMPLATE_CONFIG_MAP="${TEMPLATE_CONFIG_MAP_OVERRIDE}"
    TEMPLATE_CONTENT_SHA256="$(
        template_config_map_content_sha256 "${TEMPLATE_CONFIG_MAP}"
    )"
elif [[ "${REUSE_PRODUCTS}" == "true" && -n "${REUSE_TEMPLATE_CONFIG_MAP}" ]]; then
    # A template rendered earlier in this release for the same inputs and the
    # same node set. The render path names the ConfigMap after its content, so
    # a name whose suffix no longer matches the live digest means the object
    # was edited or replaced: it is not reused, and the render below recreates
    # it from the inputs.
    if reused_sha256="$(
        template_config_map_content_sha256 "${REUSE_TEMPLATE_CONFIG_MAP}" \
            2>/dev/null
    )" && [[ "${REUSE_TEMPLATE_CONFIG_MAP}" == \
        "gpu-fault-node-installer-template-${reused_sha256:0:12}" ]]; then
        TEMPLATE_CONFIG_MAP="${REUSE_TEMPLATE_CONFIG_MAP}"
        TEMPLATE_CONTENT_SHA256="${reused_sha256}"
        printf 'reusing template ConfigMap %s rendered earlier in this release\n' \
            "${TEMPLATE_CONFIG_MAP}"
    fi
fi
if [[ -z "${TEMPLATE_CONFIG_MAP}" ]]; then
    TEMPLATE_RENDERED="true"
    render_installer_job "${NODE}" --render-only >"${MANIFEST}"
    TEMPLATE_SHA256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
    TEMPLATE_CONTENT_SHA256="${TEMPLATE_SHA256}"
    TEMPLATE_CONFIG_MAP="gpu-fault-node-installer-template-${TEMPLATE_SHA256:0:12}"

    kubectl_context -n "${NAMESPACE}" create configmap \
        "${TEMPLATE_CONFIG_MAP}" \
        --from-file="job.yaml=${MANIFEST}" \
        --dry-run=client -o yaml >"${CONFIG_MAP_MANIFEST}"
    if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
        kubectl_context apply --dry-run=server \
            -f "${CONFIG_MAP_MANIFEST}" >/dev/null
    else
        kubectl_context apply -f "${CONFIG_MAP_MANIFEST}"
    fi
fi

sed \
    -e "s#REPLACE_WITH_CLUSTER_ID#${CLUSTER_ID}#g" \
    -e "s#REPLACE_WITH_HYPERPOD_CLUSTER#${HYPERPOD_CLUSTER}#g" \
    -e "s#REPLACE_WITH_INSTALLER_VERSION#${VERSION}#g" \
    -e "s#REPLACE_WITH_INSTALLER_CONFIG_DIGEST#${CONFIG_DIGEST}#g" \
    -e "s#REPLACE_WITH_INSTALLER_ARTIFACT_SHA256#${ARTIFACT_SHA256}#g" \
    -e "s#REPLACE_WITH_INSTALLER_BUNDLE_SHA256#${BUNDLE_SHA256}#g" \
    -e "s#REPLACE_WITH_INSTALLER_TEMPLATE_SHA256#${TEMPLATE_SOURCE_SHA256}#g" \
    -e "s#REPLACE_WITH_INSTALLER_TEMPLATE_CONTENT_SHA256#${TEMPLATE_CONTENT_SHA256}#g" \
    -e "s#REPLACE_WITH_INSTALLER_MAX_UNAVAILABLE#${MAX_UNAVAILABLE}#g" \
    -e "s#REPLACE_WITH_INSTALLER_ACTIVE_DEADLINE_SECONDS#${ACTIVE_DEADLINE_SECONDS}#g" \
    -e "s#REPLACE_WITH_INSTALLER_ALLOWED_NODES#${ALLOWED_NODES}#g" \
    -e "s#REPLACE_WITH_INSTALLER_WAVE_GENERATION#${ARTIFACT_SHA256:0:12}-${MAX_UNAVAILABLE}#g" \
    -e "s#REPLACE_WITH_INSTALLER_TEMPLATE_CONFIG_MAP#${TEMPLATE_CONFIG_MAP}#g" \
    -e "s#REPLACE_WITH_DCGM_METRICS_URL#${DCGM_METRICS_URL}#g" \
    -e "s#gpu-fault-executor-wheel-0100#${WHEEL_CONFIG_MAP}#g" \
    -e "s#gpu_fault_cluster_executor-0.10.0-py3-none-any.whl#${EXECUTOR_WHEEL_FILENAME}#g" \
    -e "s#${DEFAULT_RUNTIME_IMAGE}#${RUNTIME_IMAGE}#g" \
    "${REPO_DIR}/deploy/dataplane/node-installer-reconciler.yaml" \
    >"${RECONCILER_MANIFEST}"

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    kubectl_context apply --dry-run=server \
        -f "${RECONCILER_MANIFEST}" >/dev/null
    for node in "${NODES[@]}"; do
        render_installer_job "${node}" --render-only >"${JOB_MANIFEST}"
        kubectl_context apply --dry-run=server \
            -f "${JOB_MANIFEST}" >/dev/null
        render_installer_job "${node}" --preflight-only --render-only \
            >"${JOB_MANIFEST}"
        kubectl_context apply --dry-run=server \
            -f "${JOB_MANIFEST}" >/dev/null
    done
    # One read-only preflight Job per node, all at once. The Jobs are
    # independent: each is pinned to its node and named after node + artifact,
    # and none of them mutates the host. Each one spends most of its ~35s on
    # Pod scheduling and its own 2s poll, so a serial loop cost the fleet size
    # times that for no ordering benefit (measured 148.8s for four nodes).
    # Output is captured per node so the verdict below stays the last line of
    # stdout, which the caller parses.
    PREFLIGHT_OUTPUT_DIR="$(mktemp -d)"
    trap 'rm -f "${MANIFEST}" "${CONFIG_MAP_MANIFEST}" "${RECONCILER_MANIFEST}" "${JOB_MANIFEST}"; rm -rf "${PREFLIGHT_OUTPUT_DIR}"' EXIT
    preflight_pids=()
    for node in "${NODES[@]}"; do
        render_installer_job "${node}" --preflight-only \
            >"${PREFLIGHT_OUTPUT_DIR}/${node}.log" 2>&1 &
        preflight_pids+=("$!")
    done
    preflight_failed=()
    for index in "${!NODES[@]}"; do
        if ! wait "${preflight_pids[${index}]}"; then
            preflight_failed+=("${NODES[${index}]}")
        fi
    done
    for node in "${NODES[@]}"; do
        printf -- '--- node preflight %s ---\n' "${node}"
        cat "${PREFLIGHT_OUTPUT_DIR}/${node}.log"
    done
    if ((${#preflight_failed[@]})); then
        printf 'ERROR: node preflight failed on: %s\n' \
            "${preflight_failed[*]}" >&2
        exit 1
    fi
    printf '{"node_count":%d,"status":"PASSED","template_config_map":"%s"}\n' \
        "${#NODES[@]}" "${TEMPLATE_CONFIG_MAP}"
    exit 0
fi

kubectl_context apply -f "${RECONCILER_MANIFEST}"

# What the next run of this release may reuse. Names and digests only: the
# node set by digest, the template ConfigMap by name and content digest, and
# whether this run actually provisioned keys or rendered -- never key material.
if [[ -n "${PRODUCTS_FILE}" ]]; then
    printf '{"node_set_sha256":"%s","node_action_keys_provisioned":%s,"template_config_map":"%s","template_content_sha256":"%s","template_rendered":%s}\n' \
        "${NODE_SET_SHA256}" "${NODE_ACTION_KEYS_PROVISIONED}" \
        "${TEMPLATE_CONFIG_MAP}" "${TEMPLATE_CONTENT_SHA256}" \
        "${TEMPLATE_RENDERED}" >"${PRODUCTS_FILE}"
fi

if [[ "${WAIT_FOR_ROLLOUT}" == "true" ]]; then
    kubectl_context -n "${NAMESPACE}" rollout status \
        deployment/gpu-fault-node-installer-reconciler --timeout=10m
fi

if [[ "${SYNC_REGISTRY}" == "true" ]]; then
    python3 \
        "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
        --plane gpu \
        --context "${KUBECTL_CONTEXT}" \
        --namespace "${NAMESPACE}" \
        --release-id "${ARTIFACT_SHA256:0:12}"
fi
