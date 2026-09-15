#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
CLUSTER_ID="${GPU_FAULT_CLUSTER_ID:?GPU_FAULT_CLUSTER_ID is required}"
KUBECTL_CONTEXT="${GPU_FAULT_KUBECTL_CONTEXT:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VERSION="${GPU_FAULT_VERSION:-$(
    python3 -c '
import pathlib
import tomllib

root = pathlib.Path(__import__("sys").argv[1])
print(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"])
' "${REPO_DIR}"
)}"
VERSION_TAG="${VERSION//./}"
INSTALLER_CONFIG_MAP="${GPU_FAULT_INSTALLER_CONFIG_MAP:-gpu-fault-node-installer-${VERSION_TAG}}"
INSTALLER_CONFIG_DIGEST="${GPU_FAULT_INSTALLER_CONFIG_DIGEST:-${VERSION}}"
INSTALLER_ARTIFACT_SHA256="${GPU_FAULT_INSTALLER_ARTIFACT_SHA256:-}"
INSTALLER_BUNDLE_SHA256="$(
    printf '%s' \
        "${GPU_FAULT_INSTALLER_BUNDLE_SHA256:-${INSTALLER_ARTIFACT_SHA256}}"
)"
INSTALLER_TEMPLATE_SHA256="$(
    printf '%s' "${GPU_FAULT_INSTALLER_TEMPLATE_SHA256:-$(
        sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}'
    )}"
)"
NODE_COMPATIBILITY_DIGEST="${GPU_FAULT_NODE_COMPATIBILITY_DIGEST:-${INSTALLER_ARTIFACT_SHA256}}"
SECRET_NAME="${GPU_FAULT_SECRET_NAME:-gpu-fault-control-plane-active}"
CONNECTION_MODE="${GPU_FAULT_CONNECTION_MODE:-local}"
REGIONAL_CONNECTION_SECRET="$(
    printf '%s' "${GPU_FAULT_REGIONAL_CONNECTION_SECRET:-gpu-fault-regional-connection}"
)"
NODE_ACTION_KEYS_SECRET="$(
    printf '%s' "${GPU_FAULT_NODE_ACTION_KEYS_SECRET:-gpu-fault-node-action-keys}"
)"
RUNTIME_PROFILE="${GPU_FAULT_RUNTIME_PROFILE:-hyperpod-v1}"
DEFAULT_NODE_INSTALLER_IMAGE="public.ecr.aws/amazonlinux/amazonlinux:2023"
NODE_INSTALLER_IMAGE="${GPU_FAULT_NODE_INSTALLER_IMAGE:-${DEFAULT_NODE_INSTALLER_IMAGE}}"
INSTALLER_ACTIVE_DEADLINE_SECONDS="$(
    printf '%s' "${GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS:-840}"
)"
INSTALLER_LOCK_TIMEOUT_SECONDS="$(
    printf '%s' "${GPU_FAULT_INSTALLER_LOCK_TIMEOUT_SECONDS:-30}"
)"
NODE_NAME=""
RENDER_ONLY="false"
PREFLIGHT_ONLY="false"
REQUIRE_ROLLBACK_SLOT="${GPU_FAULT_REQUIRE_ROLLBACK_SLOT:-false}"
DIAGNOSTIC_S3_URI="${GPU_FAULT_DIAGNOSTIC_S3_URI:-}"
ENABLE_FIELD_DIAGNOSTIC="${GPU_FAULT_ENABLE_NODE_FIELD_DIAGNOSTIC:-false}"
FIELD_DIAGNOSTIC_COMMAND="${GPU_FAULT_FIELD_DIAGNOSTIC_COMMAND:-}"
FIELD_DIAGNOSTIC_SHA256="${GPU_FAULT_FIELD_DIAGNOSTIC_SHA256:-}"
MEMORY_FIELD_DIAGNOSTIC_COMMAND="${GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_COMMAND:-}"
MEMORY_FIELD_DIAGNOSTIC_SHA256="${GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_SHA256:-}"
FIELD_DIAGNOSTIC_TIMEOUT_SECONDS="${GPU_FAULT_FIELD_DIAGNOSTIC_TIMEOUT_SECONDS:-1800}"
ENABLE_DRIVER_REMEDIATION="${GPU_FAULT_ENABLE_NODE_DRIVER_REMEDIATION:-false}"
ENABLE_EFA_DRIVER_REMEDIATION="${GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION:-true}"
DRIVER_REMEDIATION_COMMAND="${GPU_FAULT_DRIVER_REMEDIATION_COMMAND:-}"
DRIVER_REMEDIATION_SHA256="${GPU_FAULT_DRIVER_REMEDIATION_SHA256:-}"
TARGET_DRIVER_BRANCH="${GPU_FAULT_TARGET_DRIVER_BRANCH:-}"
ENABLE_FIRMWARE_UPDATE="${GPU_FAULT_ENABLE_NODE_FIRMWARE_UPDATE:-false}"
FIRMWARE_UPDATE_COMMAND="${GPU_FAULT_FIRMWARE_UPDATE_COMMAND:-}"
FIRMWARE_UPDATE_SHA256="${GPU_FAULT_FIRMWARE_UPDATE_SHA256:-}"
TARGET_FIRMWARE_VERSION="${GPU_FAULT_TARGET_FIRMWARE_VERSION:-}"
FIRMWARE_VERIFY_COMMAND="${GPU_FAULT_FIRMWARE_VERIFY_COMMAND:-}"
FIRMWARE_VERIFY_SHA256="${GPU_FAULT_FIRMWARE_VERIFY_SHA256:-}"
DCGM_EXPORTER_MODE="${GPU_FAULT_DCGM_EXPORTER_MODE:-existing}"
DCGM_METRICS_URL="${GPU_FAULT_DCGM_METRICS_URL:-http://127.0.0.1:9400/metrics}"
# The collect period of the exporter DaemonSet, in milliseconds. Filled per Job
# by the node installer reconciler from
# gpu_fault.dcgm_exporter_cadence.DCGM_EXPORTER_COLLECT_INTERVAL_MS; empty here
# because a shell template cannot read the constant, and an empty value makes
# the installer leave the collector's period unknown rather than guess it.
DCGM_EXPORTER_INTERVAL_MS=""
EXPECTED_GPU_COUNT="${GPU_FAULT_EXPECTED_GPU_COUNT:-}"
EXPECTED_EFA_DEVICE_COUNT="${GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT:-}"
INVENTORY_MISMATCH_SAMPLES="${GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES:-2}"
DCGM_EDGE_FILTER_ENABLED="${GPU_FAULT_DCGM_EDGE_FILTER_ENABLED:-true}"
DCGM_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS:-300}"
KERNEL_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS:-300}"
FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS:-300}"
NODE_LOG_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS:-300}"
DCGM_EDGE_CONFIRMATION_SAMPLES="${GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES:-3}"
DCGM_HISTORY_MAX_POINTS="${GPU_FAULT_DCGM_HISTORY_MAX_POINTS:-20}"
DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD="${GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD:-0.05}"
COLLECTOR_GZIP_MIN_BYTES="${GPU_FAULT_COLLECTOR_GZIP_MIN_BYTES:-4096}"
HOST_EDGE_FILTER_ENABLED="${GPU_FAULT_HOST_EDGE_FILTER_ENABLED:-true}"
HOST_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS:-300}"
HOST_HISTORY_MAX_POINTS="${GPU_FAULT_HOST_HISTORY_MAX_POINTS:-20}"
ENABLE_NODE_LOG_COLLECTOR="$(
    printf '%s' "${GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR:-false}" |
        tr '[:upper:]' '[:lower:]'
)"
ENABLE_NVIDIA_SMI_METRICS_COLLECTOR="$(
    printf '%s' "${GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR:-false}" |
        tr '[:upper:]' '[:lower:]'
)"

[[ -n "${NODE_INSTALLER_IMAGE}" &&
    "${NODE_INSTALLER_IMAGE}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_NODE_INSTALLER_IMAGE\n' >&2
    exit 2
}
[[ "${INSTALLER_BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_BUNDLE_SHA256\n' >&2
    exit 2
}
[[ "${INSTALLER_TEMPLATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_INSTALLER_TEMPLATE_SHA256\n' >&2
    exit 2
}
for value in INSTALLER_ACTIVE_DEADLINE_SECONDS INSTALLER_LOCK_TIMEOUT_SECONDS; do
    [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || {
        printf 'ERROR: invalid %s\n' "${value}" >&2
        exit 2
    }
done
((INSTALLER_ACTIVE_DEADLINE_SECONDS >= 60)) || {
    printf 'ERROR: installer active deadline must be at least 60 seconds\n' >&2
    exit 2
}

kubectl() {
    if [[ -n "${KUBECTL_CONTEXT}" ]]; then
        command kubectl --context "${KUBECTL_CONTEXT}" "$@"
    else
        command kubectl "$@"
    fi
}

encode() {
    printf '%s' "$1" | base64 | tr -d '\n'
}

DIAGNOSTIC_S3_URI_B64="$(encode "${DIAGNOSTIC_S3_URI}")"
FIELD_DIAGNOSTIC_COMMAND_B64="$(
    encode "${FIELD_DIAGNOSTIC_COMMAND}"
)"
MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64="$(
    encode "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}"
)"
DRIVER_REMEDIATION_COMMAND_B64="$(
    encode "${DRIVER_REMEDIATION_COMMAND}"
)"
FIRMWARE_UPDATE_COMMAND_B64="$(encode "${FIRMWARE_UPDATE_COMMAND}")"
FIRMWARE_VERIFY_COMMAND_B64="$(encode "${FIRMWARE_VERIFY_COMMAND}")"
DCGM_METRICS_URL_B64="$(encode "${DCGM_METRICS_URL}")"

usage() {
    printf '%s\n' \
        "Usage: $0 --node NODE_NAME [--render-only]" \
        "" \
        "Deploys the GPU fault collector and Agent through a privileged," \
        "node-bound Kubernetes Job. The Job never requests GPU resources." \
        "--render-only prints the resolved Job manifest without creating it." \
        "--preflight-only runs read-only host and candidate checks."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)
            [[ $# -ge 2 && -n "$2" ]] || {
                printf 'ERROR: --node requires a value\n' >&2
                exit 2
            }
            NODE_NAME="$2"
            shift 2
            ;;
        --render-only)
            RENDER_ONLY="true"
            shift
            ;;
        --preflight-only)
            PREFLIGHT_ONLY="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

[[ "${NODE_NAME}" =~ ^[a-zA-Z0-9.-]+$ ]] || {
    printf 'ERROR: invalid or missing node name\n' >&2
    exit 2
}
[[ "${INSTALLER_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: GPU_FAULT_INSTALLER_ARTIFACT_SHA256 must be a SHA-256\n' >&2
    exit 2
}
[[ "${NODE_COMPATIBILITY_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'ERROR: GPU_FAULT_NODE_COMPATIBILITY_DIGEST must be a SHA-256\n' >&2
    exit 2
}
[[ "${DCGM_EXPORTER_MODE}" =~ ^(existing|disabled)$ ]] || {
    printf 'ERROR: installer DCGM mode must be existing or disabled\n' >&2
    exit 2
}
[[ "${CONNECTION_MODE}" =~ ^(local|regional)$ ]] || {
    printf 'ERROR: GPU_FAULT_CONNECTION_MODE must be local or regional\n' >&2
    exit 2
}
[[ "${RUNTIME_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$ ]] || {
    printf 'ERROR: invalid GPU_FAULT_RUNTIME_PROFILE\n' >&2
    exit 2
}
[[ "${DCGM_METRICS_URL}" =~ ^https?://[^[:space:]]+$ ]] || {
    printf 'ERROR: invalid DCGM metrics URL\n' >&2
    exit 2
}
[[ "${ENABLE_NODE_LOG_COLLECTOR}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR must be true or false\n' >&2
    exit 2
}
[[ "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR must be true or false\n' >&2
    exit 2
}
[[ "${ENABLE_EFA_DRIVER_REMEDIATION}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION must be true or false\n' >&2
    exit 2
}
[[ "${REQUIRE_ROLLBACK_SLOT}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_REQUIRE_ROLLBACK_SLOT must be true or false\n' >&2
    exit 2
}
[[ "${DCGM_EDGE_FILTER_ENABLED}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_DCGM_EDGE_FILTER_ENABLED must be true or false\n' >&2
    exit 2
}
for dcgm_filter_count in \
    DCGM_HEALTH_SUMMARY_SECONDS \
    KERNEL_HEALTH_SUMMARY_SECONDS \
    FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS \
    NODE_LOG_HEALTH_SUMMARY_SECONDS \
    DCGM_EDGE_CONFIRMATION_SAMPLES \
    DCGM_HISTORY_MAX_POINTS; do
    [[ "${!dcgm_filter_count}" =~ ^[1-9][0-9]*$ ]] || {
        printf 'ERROR: %s must be a positive integer\n' \
            "${dcgm_filter_count}" >&2
        exit 2
    }
done
[[ "${HOST_EDGE_FILTER_ENABLED}" =~ ^(true|false)$ ]] || {
    printf 'ERROR: GPU_FAULT_HOST_EDGE_FILTER_ENABLED must be true or false\n' >&2
    exit 2
}
for host_filter_count in \
    HOST_HEALTH_SUMMARY_SECONDS \
    HOST_HISTORY_MAX_POINTS; do
    [[ "${!host_filter_count}" =~ ^[1-9][0-9]*$ ]] || {
        printf 'ERROR: %s must be a positive integer\n' \
            "${host_filter_count}" >&2
        exit 2
    }
done
if [[ "${DCGM_EXPORTER_MODE}" == "disabled" &&
    "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" != "true" ]]; then
    printf 'ERROR: DCGM cannot be disabled while NvidiaSmiMetricsCollector is disabled\n' >&2
    exit 2
fi

NODE_IP="$(
    kubectl get node "${NODE_NAME}" \
        -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}'
)"
NODE_UID="$(
    kubectl get node "${NODE_NAME}" -o jsonpath='{.metadata.uid}'
)"
NODE_INSTANCE_TYPE="$(
    kubectl get node "${NODE_NAME}" \
        -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}'
)"
# AWS EC2 DescribeInstanceTypes GPU count and MaximumEfaInterfaces,
# verified 2026-07-27. HyperPod labels may include the "ml." prefix.
case "${NODE_INSTANCE_TYPE}" in
    ml.p5.4xlarge|p5.4xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-1}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-1}"
        ;;
    ml.p5.48xlarge|p5.48xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-32}"
        ;;
    ml.p5e.48xlarge|p5e.48xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-32}"
        ;;
    ml.p5en.48xlarge|p5en.48xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-16}"
        ;;
    ml.p6-b200.48xlarge|p6-b200.48xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-8}"
        ;;
    ml.p6-b300.48xlarge|p6-b300.48xlarge)
        EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
        EXPECTED_EFA_DEVICE_COUNT="${EXPECTED_EFA_DEVICE_COUNT:-16}"
        ;;
esac
if [[ -z "${EXPECTED_GPU_COUNT}" ||
    -z "${EXPECTED_EFA_DEVICE_COUNT}" ]]; then
    printf \
      'ERROR: no GPU/EFA inventory catalog entry for instance type %s; configure both GPU_FAULT_EXPECTED_GPU_COUNT and GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT\n' \
      "${NODE_INSTANCE_TYPE:-UNKNOWN}" >&2
    exit 2
fi
[[ -z "${EXPECTED_GPU_COUNT}" ||
    "${EXPECTED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: invalid expected GPU count\n' >&2
    exit 2
}
[[ -z "${EXPECTED_EFA_DEVICE_COUNT}" ||
    "${EXPECTED_EFA_DEVICE_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: invalid expected EFA device count\n' >&2
    exit 2
}
[[ "${INVENTORY_MISMATCH_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: invalid inventory mismatch sample count\n' >&2
    exit 2
}
[[ "${NODE_IP}" =~ ^[0-9a-fA-F:.]+$ ]] || {
    printf 'ERROR: invalid node InternalIP: %s\n' "${NODE_IP}" >&2
    exit 1
}
if [[ "${CONNECTION_MODE}" == "regional" ]]; then
    kubectl -n "${NAMESPACE}" get secret \
        "${REGIONAL_CONNECTION_SECRET}" >/dev/null
    CONNECTION_SECRET="${REGIONAL_CONNECTION_SECRET}"
    NODE_ACTION_SECRET_NAME="${NODE_ACTION_KEYS_SECRET}"
    NODE_ACTION_SECRET_KEY="${NODE_NAME}"
    DERIVE_NODE_ACTION_SECRET="false"
    kubectl -n "${NAMESPACE}" get secret \
        "${NODE_ACTION_SECRET_NAME}" -o json |
        python3 -c '
import json, sys
node = sys.argv[1]
data = json.load(sys.stdin).get("data") or {}
if node not in data:
    raise SystemExit(
        f"node action key Secret has no key for {node}"
    )
' "${NODE_NAME}"
    CONTROL_PLANE_ENV="$(cat <<EOF
            - name: CONTROL_PLANE_URL
              valueFrom:
                secretKeyRef:
                  name: ${CONNECTION_SECRET}
                  key: control-plane-url
            - name: CLUSTER_ID
              valueFrom:
                secretKeyRef:
                  name: ${CONNECTION_SECRET}
                  key: cluster-id
EOF
)"
    # shellcheck disable=SC2016 # Spliced into the Job's remote script; ${INSTALL_RUN_ID} expands in the pod.
    CA_INSTALL_COMMAND='install -m 0644 /connection-secret/ca.crt /host/tmp/gpu-fault-control-plane-ca-${INSTALL_RUN_ID}.crt'
    # shellcheck disable=SC2016,SC1003 # Remote payload: ${INSTALL_RUN_ID} and the trailing line-continuation backslash are for the pod shell.
    CA_CHROOT_ENV='CONTROL_PLANE_CA_CERTIFICATE=/tmp/gpu-fault-control-plane-ca-${INSTALL_RUN_ID}.crt \'
    # The bearer token never enters an argv: not the installer's (--token)
    # and not /usr/bin/env's (VAR=value). The Job is hostPID, so any
    # /proc/*/cmdline reader on the node would otherwise see it. It travels
    # as a 0600 root-owned file from the mounted Secret, like the node
    # action secret, and the installer reads it with --token-file.
    # shellcheck disable=SC2016 # Remote payload; ${INSTALL_RUN_ID} expands in the pod.
    TOKEN_INSTALL_COMMAND='install -m 0600 /connection-secret/cluster-token /host/tmp/gpu-fault-control-plane-token-${INSTALL_RUN_ID}'
    # shellcheck disable=SC2016,SC1003 # Remote payload: ${INSTALL_RUN_ID} and the trailing line-continuation backslash are for the pod shell.
    TOKEN_CHROOT_ENV='CONTROL_PLANE_TOKEN_FILE=/tmp/gpu-fault-control-plane-token-${INSTALL_RUN_ID} \'
    # shellcheck disable=SC2016 # Remote payload; the path variables exist only inside the pod.
    TOKEN_INSTALLER_ARGS='--token-file "${CONTROL_PLANE_TOKEN_FILE}" --ca-certificate "${CONTROL_PLANE_CA_CERTIFICATE}"'
    CONNECTION_SECRET_MOUNT="$(cat <<'EOF'
            - name: connection-secret
              mountPath: /connection-secret
              readOnly: true
EOF
)"
    CONNECTION_SECRET_VOLUME="$(cat <<EOF
        - name: connection-secret
          secret:
            secretName: ${CONNECTION_SECRET}
            items:
              - key: ca.crt
                path: ca.crt
              - key: cluster-token
                path: cluster-token
                mode: 0400
EOF
)"
else
    CONTROL_PLANE_IP="$(
        kubectl -n "${NAMESPACE}" get service gpu-fault-api-canary \
            -o jsonpath='{.spec.clusterIP}'
    )"
    [[ "${CONTROL_PLANE_IP}" =~ ^[0-9a-fA-F:.]+$ ]] || {
        printf 'ERROR: invalid control-plane ClusterIP\n' >&2
        exit 1
    }
    CONNECTION_SECRET="${SECRET_NAME}"
    NODE_ACTION_SECRET_NAME="${SECRET_NAME}"
    NODE_ACTION_SECRET_KEY="node-action-secret"
    DERIVE_NODE_ACTION_SECRET="true"
    CONTROL_PLANE_ENV="$(cat <<EOF
            - name: CONTROL_PLANE_URL
              value: http://${CONTROL_PLANE_IP}:8080
            - name: CLUSTER_ID
              value: ${CLUSTER_ID}
EOF
)"
    CA_INSTALL_COMMAND=":"
    # These splice into a backslash-continued /usr/bin/env argument list;
    # an empty expansion would leave a blank line that ends the command
    # early, so local mode renders harmless empty assignments instead.
    # shellcheck disable=SC1003 # The trailing backslash is the pod shell's line continuation.
    CA_CHROOT_ENV='CONTROL_PLANE_CA_CERTIFICATE= \'
    TOKEN_INSTALL_COMMAND=":"
    # shellcheck disable=SC1003 # The trailing backslash is the pod shell's line continuation.
    TOKEN_CHROOT_ENV='CONTROL_PLANE_TOKEN_FILE= \'
    TOKEN_INSTALLER_ARGS=""
    CONNECTION_SECRET_MOUNT=""
    CONNECTION_SECRET_VOLUME=""
fi
if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    JOB_ID="$(
        printf '%s\0%s' "${NODE_NAME}" "${INSTALLER_ARTIFACT_SHA256}" |
            sha256sum |
            awk '{print substr($1, 1, 24)}'
    )"
    JOB_NAME="gpu-fault-preflight-${JOB_ID}"
    HOST_ROOT_READ_ONLY="true"
else
    JOB_NAME="gpu-fault-install-${NODE_NAME#hyperpod-}"
    HOST_ROOT_READ_ONLY="false"
fi
MANIFEST="$(mktemp)"
trap 'rm -f "${MANIFEST}"' EXIT

cat > "${MANIFEST}" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${INSTALLER_ACTIVE_DEADLINE_SECONDS}
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      nodeName: ${NODE_NAME}
      hostNetwork: true
      hostPID: true
      restartPolicy: Never
      tolerations:
        - operator: Exists
      containers:
        - name: installer
          image: ${NODE_INSTALLER_IMAGE}
          securityContext:
            privileged: true
          env:
            - name: TARGET_NODE_NAME
              value: ${NODE_NAME}
            - name: TARGET_NODE_IP
              value: ${NODE_IP}
            - name: TARGET_NODE_UID
              value: ${NODE_UID}
            - name: INSTALLER_ACTIVE_DEADLINE_SECONDS
              value: "${INSTALLER_ACTIVE_DEADLINE_SECONDS}"
            - name: INSTALL_RUN_ID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: INSTALLER_LOCK_TIMEOUT_SECONDS
              value: "${INSTALLER_LOCK_TIMEOUT_SECONDS}"
            - name: PREFLIGHT_ONLY
              value: "${PREFLIGHT_ONLY}"
            - name: REQUIRE_ROLLBACK_SLOT
              value: "${REQUIRE_ROLLBACK_SLOT}"
            - name: NODE_COMPATIBILITY_DIGEST
              value: "${NODE_COMPATIBILITY_DIGEST}"
            - name: DERIVE_NODE_ACTION_SECRET
              value: "${DERIVE_NODE_ACTION_SECRET}"
${CONTROL_PLANE_ENV}
            - name: RUNTIME_PROFILE
              value: ${RUNTIME_PROFILE}
            - name: DIAGNOSTIC_S3_URI_B64
              value: "${DIAGNOSTIC_S3_URI_B64}"
            - name: ENABLE_FIELD_DIAGNOSTIC
              value: "${ENABLE_FIELD_DIAGNOSTIC}"
            - name: FIELD_DIAGNOSTIC_COMMAND_B64
              value: "${FIELD_DIAGNOSTIC_COMMAND_B64}"
            - name: FIELD_DIAGNOSTIC_SHA256
              value: "${FIELD_DIAGNOSTIC_SHA256}"
            - name: MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64
              value: "${MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64}"
            - name: MEMORY_FIELD_DIAGNOSTIC_SHA256
              value: "${MEMORY_FIELD_DIAGNOSTIC_SHA256}"
            - name: FIELD_DIAGNOSTIC_TIMEOUT_SECONDS
              value: "${FIELD_DIAGNOSTIC_TIMEOUT_SECONDS}"
            - name: ENABLE_DRIVER_REMEDIATION
              value: "${ENABLE_DRIVER_REMEDIATION}"
            - name: ENABLE_EFA_DRIVER_REMEDIATION
              value: "${ENABLE_EFA_DRIVER_REMEDIATION}"
            - name: DRIVER_REMEDIATION_COMMAND_B64
              value: "${DRIVER_REMEDIATION_COMMAND_B64}"
            - name: DRIVER_REMEDIATION_SHA256
              value: "${DRIVER_REMEDIATION_SHA256}"
            - name: TARGET_DRIVER_BRANCH
              value: "${TARGET_DRIVER_BRANCH}"
            - name: ENABLE_FIRMWARE_UPDATE
              value: "${ENABLE_FIRMWARE_UPDATE}"
            - name: FIRMWARE_UPDATE_COMMAND_B64
              value: "${FIRMWARE_UPDATE_COMMAND_B64}"
            - name: FIRMWARE_UPDATE_SHA256
              value: "${FIRMWARE_UPDATE_SHA256}"
            - name: TARGET_FIRMWARE_VERSION
              value: "${TARGET_FIRMWARE_VERSION}"
            - name: FIRMWARE_VERIFY_COMMAND_B64
              value: "${FIRMWARE_VERIFY_COMMAND_B64}"
            - name: FIRMWARE_VERIFY_SHA256
              value: "${FIRMWARE_VERIFY_SHA256}"
            - name: DCGM_EXPORTER_MODE
              value: "${DCGM_EXPORTER_MODE}"
            - name: DCGM_METRICS_URL_B64
              value: "${DCGM_METRICS_URL_B64}"
            - name: DCGM_EXPORTER_INTERVAL_MS
              value: "${DCGM_EXPORTER_INTERVAL_MS}"
            - name: EXPECTED_GPU_COUNT
              value: "${EXPECTED_GPU_COUNT}"
            - name: NODE_INSTANCE_TYPE
              value: "${NODE_INSTANCE_TYPE}"
            - name: EXPECTED_EFA_DEVICE_COUNT
              value: "${EXPECTED_EFA_DEVICE_COUNT}"
            - name: INVENTORY_MISMATCH_SAMPLES
              value: "${INVENTORY_MISMATCH_SAMPLES}"
            - name: DCGM_EDGE_FILTER_ENABLED
              value: "${DCGM_EDGE_FILTER_ENABLED}"
            - name: DCGM_HEALTH_SUMMARY_SECONDS
              value: "${DCGM_HEALTH_SUMMARY_SECONDS}"
            - name: KERNEL_HEALTH_SUMMARY_SECONDS
              value: "${KERNEL_HEALTH_SUMMARY_SECONDS}"
            - name: FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS
              value: "${FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS}"
            - name: NODE_LOG_HEALTH_SUMMARY_SECONDS
              value: "${NODE_LOG_HEALTH_SUMMARY_SECONDS}"
            - name: DCGM_EDGE_CONFIRMATION_SAMPLES
              value: "${DCGM_EDGE_CONFIRMATION_SAMPLES}"
            - name: DCGM_HISTORY_MAX_POINTS
              value: "${DCGM_HISTORY_MAX_POINTS}"
            - name: DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD
              value: "${DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD}"
            - name: COLLECTOR_GZIP_MIN_BYTES
              value: "${COLLECTOR_GZIP_MIN_BYTES}"
            - name: HOST_EDGE_FILTER_ENABLED
              value: "${HOST_EDGE_FILTER_ENABLED}"
            - name: HOST_HEALTH_SUMMARY_SECONDS
              value: "${HOST_HEALTH_SUMMARY_SECONDS}"
            - name: HOST_HISTORY_MAX_POINTS
              value: "${HOST_HISTORY_MAX_POINTS}"
            - name: ENABLE_NODE_LOG_COLLECTOR
              value: "${ENABLE_NODE_LOG_COLLECTOR}"
            - name: ENABLE_NVIDIA_SMI_METRICS_COLLECTOR
              value: "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}"
          command: ["/bin/bash", "-ceuo", "pipefail"]
          args:
            - |
              # The installer bounds its node-agent drain by this Job's
              # remaining activeDeadlineSeconds; scheduling and image pull
              # before this line are covered by the installer's own margin.
              INSTALLER_STARTED_EPOCH="\$(date +%s)"
              if [[ "\${PREFLIGHT_ONLY}" == "true" ]]; then
                chroot /host /usr/bin/env \
                GPU_FAULT_PREFLIGHT_HOST_ROOT=/ \
                GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE=/run/gpu-fault-preflight-artifact/gpu-fault-node-installer-${VERSION}.tar.gz \
                GPU_FAULT_PREFLIGHT_BUNDLE_SHA256="${INSTALLER_BUNDLE_SHA256}" \
                GPU_FAULT_PREFLIGHT_ARTIFACT_SHA256="${INSTALLER_ARTIFACT_SHA256}" \
                GPU_FAULT_REQUIRE_ROLLBACK_SLOT="\${REQUIRE_ROLLBACK_SLOT}" \
                TARGET_NODE_NAME="\${TARGET_NODE_NAME}" \
                TARGET_NODE_UID="\${TARGET_NODE_UID}" \
                  /bin/bash -ceu -o pipefail '
                    expected_bundle_sha256="${INSTALLER_BUNDLE_SHA256}"
                    [[ "\${expected_bundle_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
                      echo "ERROR: installer bundle SHA-256 was not rendered; refusing to extract" >&2
                      exit 1
                    }
                    observed_bundle_sha256="\$(
                      sha256sum "\${GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE}"
                    )"
                    observed_bundle_sha256="\${observed_bundle_sha256%% *}"
                    [[ "\${observed_bundle_sha256}" == "\${expected_bundle_sha256}" ]] || {
                      echo "ERROR: installer bundle SHA-256 mismatch: expected \${expected_bundle_sha256} observed \${observed_bundle_sha256}" >&2
                      exit 1
                    }
                    tar -xOzf \
                      "\${GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE}" \
                      gpu-fault-node-installer-${VERSION}/deploy/node/preflight-gpu-fault-node.sh |
                      /bin/bash
                  '
                exit 0
              fi
              install -m 0600 \
                /artifact/gpu-fault-node-installer-${VERSION}.tar.gz \
                /host/tmp/gpu-fault-node-installer-${VERSION}.tar.gz
              install -m 0600 /node-secret/node-action-secret \
                "/host/tmp/gpu-fault-node-action-secret-\${INSTALL_RUN_ID}"
              ${CA_INSTALL_COMMAND}
              ${TOKEN_INSTALL_COMMAND}
              chroot /host /usr/bin/env \
                INSTALLER_ACTIVE_DEADLINE_SECONDS="\${INSTALLER_ACTIVE_DEADLINE_SECONDS}" \
                INSTALLER_STARTED_EPOCH="\${INSTALLER_STARTED_EPOCH}" \
                TARGET_NODE_NAME="\${TARGET_NODE_NAME}" \
                TARGET_NODE_IP="\${TARGET_NODE_IP}" \
                TARGET_NODE_UID="\${TARGET_NODE_UID}" \
                INSTALL_RUN_ID="\${INSTALL_RUN_ID}" \
                INSTALLER_LOCK_TIMEOUT_SECONDS="\${INSTALLER_LOCK_TIMEOUT_SECONDS}" \
                GPU_FAULT_INSTALLER_BUNDLE_SHA256="${INSTALLER_BUNDLE_SHA256}" \
                GPU_FAULT_INSTALLER_TEMPLATE_SHA256="${INSTALLER_TEMPLATE_SHA256}" \
                GPU_FAULT_NODE_COMPATIBILITY_DIGEST="\${NODE_COMPATIBILITY_DIGEST}" \
                DERIVE_NODE_ACTION_SECRET="\${DERIVE_NODE_ACTION_SECRET}" \
                CONTROL_PLANE_URL="\${CONTROL_PLANE_URL}" \
                ${CA_CHROOT_ENV}
                ${TOKEN_CHROOT_ENV}
                CLUSTER_ID="\${CLUSTER_ID}" \
                RUNTIME_PROFILE="\${RUNTIME_PROFILE}" \
                DIAGNOSTIC_S3_URI_B64="\${DIAGNOSTIC_S3_URI_B64}" \
                ENABLE_FIELD_DIAGNOSTIC="\${ENABLE_FIELD_DIAGNOSTIC}" \
                FIELD_DIAGNOSTIC_COMMAND_B64="\${FIELD_DIAGNOSTIC_COMMAND_B64}" \
                FIELD_DIAGNOSTIC_SHA256="\${FIELD_DIAGNOSTIC_SHA256}" \
                MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64="\${MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64}" \
                MEMORY_FIELD_DIAGNOSTIC_SHA256="\${MEMORY_FIELD_DIAGNOSTIC_SHA256}" \
                FIELD_DIAGNOSTIC_TIMEOUT_SECONDS="\${FIELD_DIAGNOSTIC_TIMEOUT_SECONDS}" \
                ENABLE_DRIVER_REMEDIATION="\${ENABLE_DRIVER_REMEDIATION}" \
                ENABLE_EFA_DRIVER_REMEDIATION="\${ENABLE_EFA_DRIVER_REMEDIATION}" \
                DRIVER_REMEDIATION_COMMAND_B64="\${DRIVER_REMEDIATION_COMMAND_B64}" \
                DRIVER_REMEDIATION_SHA256="\${DRIVER_REMEDIATION_SHA256}" \
                TARGET_DRIVER_BRANCH="\${TARGET_DRIVER_BRANCH}" \
                ENABLE_FIRMWARE_UPDATE="\${ENABLE_FIRMWARE_UPDATE}" \
                FIRMWARE_UPDATE_COMMAND_B64="\${FIRMWARE_UPDATE_COMMAND_B64}" \
                FIRMWARE_UPDATE_SHA256="\${FIRMWARE_UPDATE_SHA256}" \
                TARGET_FIRMWARE_VERSION="\${TARGET_FIRMWARE_VERSION}" \
                FIRMWARE_VERIFY_COMMAND_B64="\${FIRMWARE_VERIFY_COMMAND_B64}" \
                FIRMWARE_VERIFY_SHA256="\${FIRMWARE_VERIFY_SHA256}" \
                DCGM_EXPORTER_MODE="\${DCGM_EXPORTER_MODE}" \
                DCGM_METRICS_URL_B64="\${DCGM_METRICS_URL_B64}" \
                DCGM_EXPORTER_INTERVAL_MS="\${DCGM_EXPORTER_INTERVAL_MS}" \
                EXPECTED_GPU_COUNT="\${EXPECTED_GPU_COUNT}" \
                NODE_INSTANCE_TYPE="\${NODE_INSTANCE_TYPE}" \
                EXPECTED_EFA_DEVICE_COUNT="\${EXPECTED_EFA_DEVICE_COUNT}" \
                INVENTORY_MISMATCH_SAMPLES="\${INVENTORY_MISMATCH_SAMPLES}" \
                GPU_FAULT_DCGM_EDGE_FILTER_ENABLED="\${DCGM_EDGE_FILTER_ENABLED}" \
                GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS="\${DCGM_HEALTH_SUMMARY_SECONDS}" \
                GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS="\${KERNEL_HEALTH_SUMMARY_SECONDS}" \
                GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS="\${FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS}" \
                GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS="\${NODE_LOG_HEALTH_SUMMARY_SECONDS}" \
                GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES="\${DCGM_EDGE_CONFIRMATION_SAMPLES}" \
                GPU_FAULT_DCGM_HISTORY_MAX_POINTS="\${DCGM_HISTORY_MAX_POINTS}" \
                GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD="\${DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD}" \
                GPU_FAULT_COLLECTOR_GZIP_MIN_BYTES="\${COLLECTOR_GZIP_MIN_BYTES}" \
                GPU_FAULT_HOST_EDGE_FILTER_ENABLED="\${HOST_EDGE_FILTER_ENABLED}" \
                GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS="\${HOST_HEALTH_SUMMARY_SECONDS}" \
                GPU_FAULT_HOST_HISTORY_MAX_POINTS="\${HOST_HISTORY_MAX_POINTS}" \
                ENABLE_NODE_LOG_COLLECTOR="\${ENABLE_NODE_LOG_COLLECTOR}" \
                ENABLE_NVIDIA_SMI_METRICS_COLLECTOR="\${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" \
                /bin/bash -ceu -o pipefail '
              node_action_secret="/tmp/gpu-fault-node-action-secret-\${INSTALL_RUN_ID}"
              control_plane_ca="/tmp/gpu-fault-control-plane-ca-\${INSTALL_RUN_ID}.crt"
              control_plane_token="/tmp/gpu-fault-control-plane-token-\${INSTALL_RUN_ID}"
              trap "rm -f \${node_action_secret} \${control_plane_ca} \${control_plane_token}" EXIT
                  install -d -m 0755 /var/lock
                  exec 9>/var/lock/gpu-fault-installer.lock
                  flock -w "\${INSTALLER_LOCK_TIMEOUT_SECONDS}" 9 || {
                    echo "another GPU fault installer owns the node lock" >&2
                    exit 1
                  }
                  bundle="/tmp/gpu-fault-node-installer-${VERSION}.tar.gz"
                  expected_bundle_sha256="${INSTALLER_BUNDLE_SHA256}"
                  [[ "\${expected_bundle_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
                    echo "ERROR: installer bundle SHA-256 was not rendered; refusing to extract" >&2
                    exit 1
                  }
                  observed_bundle_sha256="\$(sha256sum "\${bundle}")"
                  observed_bundle_sha256="\${observed_bundle_sha256%% *}"
                  [[ "\${observed_bundle_sha256}" == "\${expected_bundle_sha256}" ]] || {
                    echo "ERROR: installer bundle SHA-256 mismatch: expected \${expected_bundle_sha256} observed \${observed_bundle_sha256}" >&2
                    exit 1
                  }
                  rm -rf /tmp/gpu-fault-node-installer-${VERSION}
                  tar -xzf "\${bundle}" -C /tmp
                  cd /tmp/gpu-fault-node-installer-${VERSION}
                  required_interfaces=""
                  for interface_path in \
                      /sys/class/infiniband/*/device/net/*; do
                    [[ -e "\${interface_path}" ]] || continue
                    interface="\${interface_path##*/}"
                    required_interfaces="\${required_interfaces:+\${required_interfaces},}\${interface}"
                  done
                  interface_args=()
                  if [[ -n "\${required_interfaces}" ]]; then
                    interface_args=(
                      --required-interfaces "\${required_interfaces}"
                    )
                  fi
                  decode() {
                    printf "%s" "\$1" | base64 -d
                  }
                  dcgm_metrics_url="\$(
                    decode "\${DCGM_METRICS_URL_B64}"
                  )"
                  exporter_args=()
                  if [[ -n "\${DCGM_EXPORTER_INTERVAL_MS}" ]]; then
                    exporter_args=(
                      --dcgm-exporter-interval-ms "\${DCGM_EXPORTER_INTERVAL_MS}"
                    )
                  fi
                  diagnostic_args=()
                  diagnostic_s3_uri="\$(decode "\${DIAGNOSTIC_S3_URI_B64}")"
                  if [[ -n "\${diagnostic_s3_uri}" ]]; then
                    diagnostic_args=(
                      --diagnostic-s3-uri "\${diagnostic_s3_uri}"
                    )
                  fi
                  inventory_args=(
                    --inventory-mismatch-samples
                    "\${INVENTORY_MISMATCH_SAMPLES}"
                  )
                  if [[ -n "\${EXPECTED_GPU_COUNT}" ]]; then
                    inventory_args+=(
                      --expected-gpu-count
                      "\${EXPECTED_GPU_COUNT}"
                    )
                  fi
                  if [[ -n "\${EXPECTED_EFA_DEVICE_COUNT}" ]]; then
                    inventory_args+=(
                      --expected-efa-device-count
                      "\${EXPECTED_EFA_DEVICE_COUNT}"
                    )
                  fi
                  field_diagnostic_args=()
                  if [[ "\${ENABLE_FIELD_DIAGNOSTIC}" == "true" ]]; then
                    field_diagnostic_args=(
                      --allow-field-diagnostic
                      --field-diagnostic-command
                      "\$(decode "\${FIELD_DIAGNOSTIC_COMMAND_B64}")"
                      --field-diagnostic-sha256
                      "\${FIELD_DIAGNOSTIC_SHA256}"
                      --field-diagnostic-timeout
                      "\${FIELD_DIAGNOSTIC_TIMEOUT_SECONDS}"
                    )
                    memory_field_command="\$(
                      decode "\${MEMORY_FIELD_DIAGNOSTIC_COMMAND_B64}"
                    )"
                    if [[ -n "\${memory_field_command}" ]]; then
                      field_diagnostic_args+=(
                        --memory-field-diagnostic-command
                        "\${memory_field_command}"
                        --memory-field-diagnostic-sha256
                        "\${MEMORY_FIELD_DIAGNOSTIC_SHA256}"
                      )
                    fi
                  fi
                  remediation_args=()
                  if [[ "\${ENABLE_DRIVER_REMEDIATION}" == "true" ]]; then
                    remediation_args+=(
                      --allow-driver-remediation
                      --driver-remediation-command
                      "\$(decode "\${DRIVER_REMEDIATION_COMMAND_B64}")"
                      --driver-remediation-sha256
                      "\${DRIVER_REMEDIATION_SHA256}"
                      --target-driver-branch "\${TARGET_DRIVER_BRANCH}"
                    )
                  fi
                  if [[ "\${ENABLE_EFA_DRIVER_REMEDIATION}" == "true" ]]; then
                    remediation_args+=(
                      --allow-efa-driver-remediation
                    )
                  else
                    remediation_args+=(
                      --disable-efa-driver-remediation
                    )
                  fi
                  if [[ "\${ENABLE_FIRMWARE_UPDATE}" == "true" ]]; then
                    remediation_args+=(
                      --allow-firmware-update
                      --firmware-update-command
                      "\$(decode "\${FIRMWARE_UPDATE_COMMAND_B64}")"
                      --firmware-update-sha256
                      "\${FIRMWARE_UPDATE_SHA256}"
                      --target-firmware-version
                      "\${TARGET_FIRMWARE_VERSION}"
                      --firmware-verify-command
                      "\$(decode "\${FIRMWARE_VERIFY_COMMAND_B64}")"
                      --firmware-verify-sha256
                      "\${FIRMWARE_VERIFY_SHA256}"
                    )
                  fi
                  node_log_args=()
                  if [[ "\${ENABLE_NODE_LOG_COLLECTOR}" == "true" ]]; then
                    node_log_args=(--enable-node-log-collector)
                  fi
                  nvidia_smi_args=()
                  if [[ "\${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" == "true" ]]; then
                    nvidia_smi_args=(
                      --enable-nvidia-smi-metrics-collector
                    )
                  fi
                  metrics_mode=dcgm
                  if [[ "\${DCGM_EXPORTER_MODE}" == "disabled" ]]; then
                    metrics_mode=nvidia-smi
                  fi
                  node_action_args=(
                    --node-action-secret-file "\${node_action_secret}"
                  )
                  if [[ "\${DERIVE_NODE_ACTION_SECRET}" == "true" ]]; then
                    node_action_args+=(--derive-node-action-secret)
                  else
                    node_action_args+=(--node-action-key-version 2)
                  fi
                  deploy/node/install-gpu-fault-collector.sh \
                    --control-plane-url "\${CONTROL_PLANE_URL}" \
                    ${TOKEN_INSTALLER_ARGS} \
                    --wheel-sha256 "${INSTALLER_ARTIFACT_SHA256}" \
                    --cluster-id "\${CLUSTER_ID}" \
                    --runtime-profile-version "\${RUNTIME_PROFILE}" \
                    --node-id "\${TARGET_NODE_NAME}" \
                    --node-instance-type "\${NODE_INSTANCE_TYPE}" \
                    --metrics-mode "\${metrics_mode}" \
                    --dcgm-exporter "\${DCGM_EXPORTER_MODE}" \
                    --dcgm-metrics-url "\${dcgm_metrics_url}" \
                    "\${exporter_args[@]}" \
                    "\${interface_args[@]}" \
                    "\${diagnostic_args[@]}" \
                    "\${inventory_args[@]}" \
                    "\${field_diagnostic_args[@]}" \
                    "\${remediation_args[@]}" \
                    "\${node_log_args[@]}" \
                    "\${nvidia_smi_args[@]}" \
                    --training-log-paths \
                      "/var/log/pods/*/*/*.log,/opt/ml/output/**/*.log" \
                    --python-command /usr/bin/python3.12 \
                    --enable-node-agent \
                    "\${node_action_args[@]}" \
                    --allow-gpu-reset \
                    --allow-fabric-reset \
                    --allow-service-quiesce \
                    --allow-fabric-manager-restart \
                    --node-agent-host "\${TARGET_NODE_IP}" \
                    --node-agent-advertise-url \
                      "https://\${TARGET_NODE_IP}:9099" \
                    --node-instance-id "\${TARGET_NODE_UID}"
                  /opt/gpu-fault/verify
                '
          volumeMounts:
            - name: host-root
              mountPath: /host
              readOnly: ${HOST_ROOT_READ_ONLY}
            - name: installer
              mountPath: /artifact
              readOnly: true
            - name: installer
              mountPath: /host/run/gpu-fault-preflight-artifact
              readOnly: true
            - name: node-secret
              mountPath: /node-secret
              readOnly: true
${CONNECTION_SECRET_MOUNT}
      volumes:
        - name: host-root
          hostPath:
            path: /
            type: Directory
        - name: installer
          configMap:
            name: ${INSTALLER_CONFIG_MAP}
        - name: node-secret
          secret:
            secretName: ${NODE_ACTION_SECRET_NAME}
            items:
              - key: ${NODE_ACTION_SECRET_KEY}
                path: node-action-secret
${CONNECTION_SECRET_VOLUME}
EOF

if [[ "${RENDER_ONLY}" == "true" ]]; then
    cat "${MANIFEST}"
    exit 0
fi

kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" \
    --ignore-not-found --wait=true
kubectl apply -f "${MANIFEST}"
deadline=$((SECONDS + INSTALLER_ACTIVE_DEADLINE_SECONDS + 60))
while true; do
    condition="$(
        kubectl -n "${NAMESPACE}" get "job/${JOB_NAME}" \
            -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}'
    )"
    [[ "${condition}" != *Complete* ]] || break
    if [[ "${condition}" == *Failed* || "${SECONDS}" -ge "${deadline}" ]]; then
        kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers
        if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
            kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" \
                --ignore-not-found --wait=true
        fi
        exit 1
    fi
    # The interval is spent inside a bounded `kubectl wait` instead of a timer,
    # so a Job that completes early in it is read back at once. A non-zero exit
    # (timeout, a Failed Job, a transient API error) only means "not yet": the
    # read at the top of the loop still decides, exactly as before.
    kubectl -n "${NAMESPACE}" wait "job/${JOB_NAME}" \
        --for=condition=complete --timeout=2s >/dev/null 2>&1 || true
done
kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers
if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" \
        --ignore-not-found --wait=true
    exit 0
fi
kubectl annotate node "${NODE_NAME}" --overwrite \
    "gpu-fault.io/installer-version=${VERSION}" \
    "gpu-fault.io/installer-config-digest=${INSTALLER_CONFIG_DIGEST}" \
    "gpu-fault.io/installer-artifact-sha256=${INSTALLER_ARTIFACT_SHA256}" \
    "gpu-fault.io/installer-bundle-sha256=${INSTALLER_BUNDLE_SHA256}" \
    "gpu-fault.io/installer-template-sha256=${INSTALLER_TEMPLATE_SHA256}" \
    "gpu-fault.io/installer-node-uid=${NODE_UID}" \
    "gpu-fault.io/installer-state=Succeeded"
