#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONTROL_PLANE_URL=""
CONTROL_PLANE_CA_CERTIFICATE=""
CERTIFICATE_MIN_VALIDITY_SECONDS="${GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS:-2592000}"
CLUSTER_ID=""
PROFILE_VERSION=""
TOKEN=""
NODE_ID="$(hostname -f 2>/dev/null || hostname)"
NODE_INSTANCE_TYPE=""
GPU_PRODUCT=""
GPU_PRODUCT_DISCOVERY="auto"
DRIVER_BRANCH=""
CUDA_VERSION=""
METRICS_MODE="auto"
METRICS_INTERVAL="15"
GPU_INVENTORY_INTERVAL_SECONDS="${GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS:-60}"
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
RANK_LIVENESS_ENABLED="${GPU_FAULT_RANK_LIVENESS_ENABLED:-true}"
RANK_PROGRESS_MIN_WRITE_BPS="${GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS:-1048576}"
RANK_PROGRESS_MIN_CPU_CORES="${GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES:-0.5}"
RANK_PROGRESS_GPU_IDLE_PERCENT="${GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT:-5}"
HOST_HEALTH_SUMMARY_SECONDS="${GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS:-300}"
HOST_HISTORY_MAX_POINTS="${GPU_FAULT_HOST_HISTORY_MAX_POINTS:-20}"
HOST_INTERVAL="15"
EXPECTED_GPU_COUNT=""
EXPECTED_EFA_DEVICE_COUNT=""
INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES="2"
LOG_INTERVAL="10"
FABRIC_MANAGER_LOG_INTERVAL="5"
FABRIC_MANAGER_JOURNAL="true"
FABRIC_MANAGER_IDENTIFIERS="nvidia-fabricmanager,nv-fabricmanager"
FABRIC_MANAGER_LOG_PATHS=""
FABRIC_MANAGER_LOG_PATHS_EXPLICIT="false"
FABRIC_MANAGER_CONFIG="/usr/share/nvidia/nvswitch/fabricmanager.cfg"
FILESYSTEMS="/,/var,/tmp"
REQUIRED_INTERFACES=""
NVSWITCH_TOPOLOGY_COMMAND=""
TRAINING_LOG_PATHS=""
NODE_LOG_MAX_ENTRIES_PER_BATCH="${GPU_FAULT_NODE_LOG_MAX_ENTRIES_PER_BATCH:-1000}"
NODE_LOG_MAX_BATCH_BYTES="${GPU_FAULT_NODE_LOG_MAX_BATCH_BYTES:-4194304}"
NODE_LOG_MAX_ENTRY_BYTES="${GPU_FAULT_NODE_LOG_MAX_ENTRY_BYTES:-65536}"
DCGM_EXPORTER_MODE="existing"
DCGM_METRICS_URL="http://127.0.0.1:9400/metrics"
DCGM_EXPORTER_IMAGE=""
WHEEL=""
WHEELHOUSE=""
PY_SPY_VERSION="0.4.1"
PY_SPY_BINARY_SHA256="e7c2de2dc54449ec88c086f1859555b4e34e63ccdcf3f8804496f9306cd44de6"
PYTHON_COMMAND="python3"
DISABLE_KERNEL="false"
ENABLE_NODE_LOG_COLLECTOR="false"
ENABLE_NVIDIA_SMI_METRICS_COLLECTOR="false"
NO_START="false"
ENABLE_NODE_AGENT="false"
NODE_ACTION_SECRET=""
NODE_ACTION_SECRET_FILE=""
DERIVE_NODE_ACTION_SECRET="false"
NODE_ACTION_KEY_VERSION_OVERRIDE=""
ALLOW_GPU_RESET="false"
SINGLE_GPU_RESET_SUPPORTED="${GPU_FAULT_NODE_SINGLE_GPU_RESET_SUPPORTED:-true}"
ALLOW_FABRIC_RESET="false"
ALLOW_SERVICE_QUIESCE="false"
ALLOW_FABRIC_MANAGER_RESTART="false"
DIAGNOSTIC_OUTPUT_DIR="/var/lib/gpu-fault/diagnostics"
DIAGNOSTIC_S3_URI=""
DIAGNOSTIC_RETENTION_SECONDS="604800"
DIAGNOSTIC_MAX_ARCHIVES="20"
ALLOW_FIELD_DIAGNOSTIC="false"
FIELD_DIAGNOSTIC_COMMAND=""
FIELD_DIAGNOSTIC_SHA256=""
MEMORY_FIELD_DIAGNOSTIC_COMMAND=""
MEMORY_FIELD_DIAGNOSTIC_SHA256=""
FIELD_DIAGNOSTIC_TIMEOUT_SECONDS="1800"
ALLOW_DRIVER_REMEDIATION="false"
ALLOW_EFA_DRIVER_REMEDIATION="true"
DRIVER_REMEDIATION_COMMAND=""
DRIVER_REMEDIATION_SHA256=""
TARGET_DRIVER_BRANCH=""
ALLOW_FIRMWARE_UPDATE="false"
FIRMWARE_UPDATE_COMMAND=""
FIRMWARE_UPDATE_SHA256=""
TARGET_FIRMWARE_VERSION=""
FIRMWARE_VERIFY_COMMAND=""
FIRMWARE_VERIFY_SHA256=""
QUIESCE_SERVICES="nvidia-fabricmanager,nvidia-dcgm,nvidia-persistenced,gpu-fault-gpu-persistence,gpu-fault-metrics-collector,gpu-fault-host-collector,kubelet"
QUIESCE_PROCESSES=""
QUIESCE_CONTAINERS="aws-hyperpod/health-monitoring-agent,gpu-fault-system/exporter,kube-system/nvidia-device-plugin-ctr"
CONTAINER_STOP_TIMEOUT_SECONDS="30"
CONTAINER_RESTORE_TIMEOUT_SECONDS="180"
# How long quiesce waits for a TERMed /dev/nvidiaN holder to let go
# before escalating to KILL. Zero means "signal and escalate at once".
DEVICE_SWEEP_TIMEOUT_SECONDS="20"
QUIESCE_FAILSAFE_SECONDS="420"
QUIESCE_RETRY_SECONDS="60"
QUIESCE_SETTLE_SECONDS="2"
RESTORE_SETTLE_SECONDS="30"
NODE_AGENT_PORT="9099"
NODE_AGENT_HOST=""
NODE_AGENT_ADVERTISE_URL=""
NODE_AGENT_ALLOW_PLAINTEXT="false"
NODE_AGENT_TLS_CERT=""
NODE_AGENT_TLS_KEY=""
NODE_AGENT_TLS_CLIENT_CA=""
NODE_HEARTBEAT_INTERVAL="30"
NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS="2100"
NODE_ACTION_RETENTION_SECONDS="604800"
NODE_ACTION_MAX_RESULTS="10000"
NODE_INSTANCE_ID=""

usage() {
    printf '%s\n' \
        "Install the read-only GPU fault collectors on one GPU instance." \
        "" \
        "Required:" \
        "  --control-plane-url URL" \
        "  --cluster-id ID" \
        "  --runtime-profile-version VERSION" \
        "" \
        "Options:" \
        "  --node-id ID                    Default: host FQDN" \
        "  --node-instance-type TYPE       EC2/HyperPod instance type" \
        "  --token TOKEN                   Optional bearer token" \
        "  --ca-certificate PATH           Private CA PEM for HTTPS control plane" \
        "  --certificate-min-validity-seconds N  Reject CA/leaf expiry inside N seconds" \
        "  --gpu-product PRODUCT" \
        "  --gpu-product-discovery auto|required|disabled" \
        "  --driver-branch BRANCH" \
        "  --cuda-version VERSION" \
        "  --metrics-mode auto|dcgm|nvidia-smi" \
        "  --metrics-interval SECONDS      Default: 15" \
        "  --host-interval SECONDS         Default: 15" \
        "  --expected-gpu-count COUNT      Expected visible GPUs; unset disables check" \
        "  --expected-efa-device-count COUNT Expected ACTIVE EFA devices; unset disables check" \
        "  --inventory-mismatch-samples N  Consecutive mismatches before action; default: 2" \
        "  --log-interval SECONDS          Default: 10" \
        "  --fabric-manager-log-interval SECONDS  Default: 5" \
        "  --disable-fabric-manager-journal Read configured files only" \
        "  --fabric-manager-identifiers NAMES  Journal identities; comma-separated" \
        "  --fabric-manager-log-paths GLOBS File globs; default: discover NVIDIA config" \
        "  --filesystems PATHS             Comma-separated; default: /,/var,/tmp" \
        "  --required-interfaces NAMES     Comma-separated critical interfaces" \
        "  --nvswitch-topology-command CMD Normalized JSON topology query" \
        "  --training-log-paths GLOBS      Comma-separated absolute globs" \
        "  --dcgm-exporter existing|docker|disabled" \
        "  --dcgm-metrics-url URL          Default: http://127.0.0.1:9400/metrics" \
        "  --dcgm-exporter-image IMAGE     Required with --dcgm-exporter docker" \
        "  --wheel PATH                    Default: newest wheel under dist/" \
        "  --wheelhouse DIR                Install dependencies without an index" \
        "  --python-command PATH           Python 3.12+ executable" \
        "  --disable-kernel                Do not install the /dev/kmsg collector" \
        "  --enable-node-log-collector     Enable system/training log collection; default: disabled" \
        "  --enable-nvidia-smi-metrics-collector  Allow nvidia-smi metrics fallback; default: disabled" \
        "  --enable-node-agent             Install signed node action agent" \
        "  --node-action-secret SECRET     Required with --enable-node-agent" \
        "  --node-action-secret-file PATH  Read the secret from a file" \
        "  --derive-node-action-secret     Derive a node-only key from the supplied fleet secret" \
        "  --node-action-key-version N     Declare a supplied key as version 1 or 2" \
        "  --node-agent-host ADDRESS       Bind address; default: node ID" \
        "  --allow-node-agent-plaintext    Explicitly permit HTTP without a server certificate" \
        "  --allow-gpu-reset               Allow fenced GPU reset commands" \
        "  --allow-fabric-reset            Allow full local GPU/NVSwitch reset" \
        "  --allow-service-quiesce         Allow fenced GPU service stop/start" \
        "  --allow-fabric-manager-restart  Allow fenced Fabric Manager restart" \
        "  --diagnostic-output-dir PATH    Local bundle directory" \
        "  --diagnostic-s3-uri S3_URI      Optional durable bundle destination" \
        "  --diagnostic-retention-seconds SEC Default: 604800" \
        "  --diagnostic-max-archives N     Default: 20" \
        "  --memory-field-diagnostic-command CMD  GPU memory Field Diagnostic" \
        "  --memory-field-diagnostic-sha256 HEX   Pinned memory diagnostic binary" \
        "  --allow-field-diagnostic        Allow pinned NVIDIA Field Diagnostic command" \
        "  --field-diagnostic-command CMD  Absolute command; {link_id}, {gpu_uuid}, {pci_bdf}" \
        "  --field-diagnostic-sha256 HEX   Pinned executable SHA-256" \
        "  --field-diagnostic-timeout SEC  60-7200; default: 1800" \
        "  --allow-driver-remediation      Allow explicitly mapped SXID driver repair" \
        "  --allow-efa-driver-remediation  Allow EFA PCI driver rebind after workload stop" \
        "  --driver-remediation-command CMD  Absolute, preinstalled command; {target} allowed" \
        "  --driver-remediation-sha256 HEX Pinned executable SHA-256" \
        "  --target-driver-branch INTEGER  Required driver branch after repair" \
        "  --allow-firmware-update         Allow explicitly mapped SXID firmware update" \
        "  --firmware-update-command CMD   Absolute, preinstalled command; {target} allowed" \
        "  --firmware-update-sha256 HEX    Pinned executable SHA-256" \
        "  --target-firmware-version VER   Required exact firmware version" \
        "  --firmware-verify-command CMD   Command whose stdout is the exact version" \
        "  --firmware-verify-sha256 HEX    Pinned verification executable SHA-256" \
        "  --quiesce-services NAMES        Ordered restore list; comma-separated" \
        "  --quiesce-processes NAMES       Exact process names sent TERM" \
        "  --quiesce-containers NAMES      Kubernetes namespace/container selectors" \
        "  --container-stop-timeout SEC    Graceful container stop timeout; default: 30" \
        "  --container-restore-timeout SEC Wait for container return; default: 180" \
        "  --quiesce-failsafe-seconds SEC  First automatic restore; default: 420" \
        "  --quiesce-retry-seconds SEC     Automatic restore retry; default: 60" \
        "  --quiesce-settle-seconds SEC    Wait after quiesce; default: 2" \
        "  --node-agent-port PORT          Default: 9099" \
        "  --node-agent-advertise-url URL  Control-plane reachable agent URL" \
        "  --node-agent-tls-cert PATH      Serve HTTPS with this certificate" \
        "  --node-agent-tls-key PATH       Private key for --node-agent-tls-cert" \
        "  --node-agent-tls-client-ca PATH Require a client certificate from this CA" \
        "  --node-heartbeat-interval SEC   Default: 30" \
        "  --node-inflight-wait-timeout SEC Duplicate command wait; default: 2100" \
        "  --node-action-retention-seconds SEC Ledger retention; default: 604800" \
        "  --node-action-max-results N     Ledger row cap; default: 10000" \
        "  --node-instance-id ID           Stable VM or Kubernetes Node UID" \
        "  --no-start                      Install and enable without starting" \
        "  -h, --help"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

validate_fabric_manager_log_paths() {
    local paths="$1"
    local path
    local forbidden

    [[ "${paths}" != *$'\n'* && "${paths}" != *$'\r'* ]] ||
        die "Fabric Manager log paths cannot contain newlines"
    IFS=',' read -r -a fabric_manager_paths <<< "${paths}"
    for path in "${fabric_manager_paths[@]}"; do
        [[ -n "${path}" && "${path}" == /* ]] ||
            die "Fabric Manager log paths must be non-empty absolute globs"
        for forbidden in $'\t' ' ' '`' '$' ';' '|' '&' '<' '>' \
            '"' "'" '\' '(' ')' '{' '}' '!'; do
            [[ "${path}" != *"${forbidden}"* ]] ||
                die "Fabric Manager log path contains an unsafe character"
        done
    done
}

discover_fabric_manager_log_path() {
    local line
    local key
    local value
    local log_file_name=""
    local log_use_syslog=""

    [[ "${FABRIC_MANAGER_LOG_PATHS_EXPLICIT}" == "false" ]] || return
    [[ -r "${FABRIC_MANAGER_CONFIG}" ]] || return

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "${line}" == *"="* ]] || continue
        key="$(trim_whitespace "${line%%=*}")"
        value="$(trim_whitespace "${line#*=}")"
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        case "${key}" in
            LOG_FILE_NAME) log_file_name="${value}" ;;
            LOG_USE_SYSLOG) log_use_syslog="${value}" ;;
        esac
    done < "${FABRIC_MANAGER_CONFIG}"

    [[ "${log_use_syslog}" == "0" && -n "${log_file_name}" ]] || return
    validate_fabric_manager_log_paths "${log_file_name}"
    if [[ -e "${log_file_name}" ]]; then
        FABRIC_MANAGER_LOG_PATHS="${log_file_name}"
        printf 'Discovered Fabric Manager log file: %s\n' \
            "${FABRIC_MANAGER_LOG_PATHS}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --control-plane-url) require_value "$@"; CONTROL_PLANE_URL="$2"; shift 2 ;;
        --cluster-id) require_value "$@"; CLUSTER_ID="$2"; shift 2 ;;
        --runtime-profile-version) require_value "$@"; PROFILE_VERSION="$2"; shift 2 ;;
        --token) require_value "$@"; TOKEN="$2"; shift 2 ;;
        --ca-certificate) require_value "$@"; CONTROL_PLANE_CA_CERTIFICATE="$2"; shift 2 ;;
        --certificate-min-validity-seconds) require_value "$@"; CERTIFICATE_MIN_VALIDITY_SECONDS="$2"; shift 2 ;;
        --node-id) require_value "$@"; NODE_ID="$2"; shift 2 ;;
        --node-instance-type) require_value "$@"; NODE_INSTANCE_TYPE="$2"; shift 2 ;;
        --gpu-product) require_value "$@"; GPU_PRODUCT="$2"; shift 2 ;;
        --gpu-product-discovery) require_value "$@"; GPU_PRODUCT_DISCOVERY="$2"; shift 2 ;;
        --driver-branch) require_value "$@"; DRIVER_BRANCH="$2"; shift 2 ;;
        --cuda-version) require_value "$@"; CUDA_VERSION="$2"; shift 2 ;;
        --metrics-mode) require_value "$@"; METRICS_MODE="$2"; shift 2 ;;
        --metrics-interval) require_value "$@"; METRICS_INTERVAL="$2"; shift 2 ;;
        --host-interval) require_value "$@"; HOST_INTERVAL="$2"; shift 2 ;;
        --expected-gpu-count) require_value "$@"; EXPECTED_GPU_COUNT="$2"; shift 2 ;;
        --expected-efa-device-count) require_value "$@"; EXPECTED_EFA_DEVICE_COUNT="$2"; shift 2 ;;
        --inventory-mismatch-samples) require_value "$@"; INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES="$2"; shift 2 ;;
        --log-interval) require_value "$@"; LOG_INTERVAL="$2"; shift 2 ;;
        --fabric-manager-log-interval) require_value "$@"; FABRIC_MANAGER_LOG_INTERVAL="$2"; shift 2 ;;
        --disable-fabric-manager-journal) FABRIC_MANAGER_JOURNAL="false"; shift ;;
        --fabric-manager-identifiers) require_value "$@"; FABRIC_MANAGER_IDENTIFIERS="$2"; shift 2 ;;
        --fabric-manager-log-paths)
            require_value "$@"
            FABRIC_MANAGER_LOG_PATHS="$2"
            FABRIC_MANAGER_LOG_PATHS_EXPLICIT="true"
            shift 2
            ;;
        --filesystems) require_value "$@"; FILESYSTEMS="$2"; shift 2 ;;
        --required-interfaces) require_value "$@"; REQUIRED_INTERFACES="$2"; shift 2 ;;
        --nvswitch-topology-command) require_value "$@"; NVSWITCH_TOPOLOGY_COMMAND="$2"; shift 2 ;;
        --training-log-paths) require_value "$@"; TRAINING_LOG_PATHS="$2"; shift 2 ;;
        --dcgm-exporter) require_value "$@"; DCGM_EXPORTER_MODE="$2"; shift 2 ;;
        --dcgm-metrics-url) require_value "$@"; DCGM_METRICS_URL="$2"; shift 2 ;;
        --dcgm-exporter-image) require_value "$@"; DCGM_EXPORTER_IMAGE="$2"; shift 2 ;;
        --wheel) require_value "$@"; WHEEL="$2"; shift 2 ;;
        --wheelhouse) require_value "$@"; WHEELHOUSE="$2"; shift 2 ;;
        --python-command) require_value "$@"; PYTHON_COMMAND="$2"; shift 2 ;;
        --disable-kernel) DISABLE_KERNEL="true"; shift ;;
        --enable-node-log-collector) ENABLE_NODE_LOG_COLLECTOR="true"; shift ;;
        --enable-nvidia-smi-metrics-collector) ENABLE_NVIDIA_SMI_METRICS_COLLECTOR="true"; shift ;;
        --enable-node-agent) ENABLE_NODE_AGENT="true"; shift ;;
        --node-action-secret) require_value "$@"; NODE_ACTION_SECRET="$2"; shift 2 ;;
        --node-action-secret-file) require_value "$@"; NODE_ACTION_SECRET_FILE="$2"; shift 2 ;;
        --derive-node-action-secret) DERIVE_NODE_ACTION_SECRET="true"; shift ;;
        --node-action-key-version)
            require_value "$@"
            NODE_ACTION_KEY_VERSION_OVERRIDE="$2"
            shift 2
            ;;
        --allow-gpu-reset) ALLOW_GPU_RESET="true"; shift ;;
        --allow-fabric-reset) ALLOW_FABRIC_RESET="true"; shift ;;
        --allow-service-quiesce) ALLOW_SERVICE_QUIESCE="true"; shift ;;
        --allow-fabric-manager-restart) ALLOW_FABRIC_MANAGER_RESTART="true"; shift ;;
        --diagnostic-output-dir) require_value "$@"; DIAGNOSTIC_OUTPUT_DIR="$2"; shift 2 ;;
        --diagnostic-s3-uri) require_value "$@"; DIAGNOSTIC_S3_URI="$2"; shift 2 ;;
        --diagnostic-retention-seconds) require_value "$@"; DIAGNOSTIC_RETENTION_SECONDS="$2"; shift 2 ;;
        --diagnostic-max-archives) require_value "$@"; DIAGNOSTIC_MAX_ARCHIVES="$2"; shift 2 ;;
        --allow-field-diagnostic) ALLOW_FIELD_DIAGNOSTIC="true"; shift ;;
        --field-diagnostic-command) require_value "$@"; FIELD_DIAGNOSTIC_COMMAND="$2"; shift 2 ;;
        --field-diagnostic-sha256) require_value "$@"; FIELD_DIAGNOSTIC_SHA256="$2"; shift 2 ;;
        --memory-field-diagnostic-command) require_value "$@"; MEMORY_FIELD_DIAGNOSTIC_COMMAND="$2"; shift 2 ;;
        --memory-field-diagnostic-sha256) require_value "$@"; MEMORY_FIELD_DIAGNOSTIC_SHA256="$2"; shift 2 ;;
        --field-diagnostic-timeout) require_value "$@"; FIELD_DIAGNOSTIC_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --allow-driver-remediation) ALLOW_DRIVER_REMEDIATION="true"; shift ;;
        --allow-efa-driver-remediation) ALLOW_EFA_DRIVER_REMEDIATION="true"; shift ;;
        --driver-remediation-command) require_value "$@"; DRIVER_REMEDIATION_COMMAND="$2"; shift 2 ;;
        --driver-remediation-sha256) require_value "$@"; DRIVER_REMEDIATION_SHA256="$2"; shift 2 ;;
        --target-driver-branch) require_value "$@"; TARGET_DRIVER_BRANCH="$2"; shift 2 ;;
        --allow-firmware-update) ALLOW_FIRMWARE_UPDATE="true"; shift ;;
        --firmware-update-command) require_value "$@"; FIRMWARE_UPDATE_COMMAND="$2"; shift 2 ;;
        --firmware-update-sha256) require_value "$@"; FIRMWARE_UPDATE_SHA256="$2"; shift 2 ;;
        --target-firmware-version) require_value "$@"; TARGET_FIRMWARE_VERSION="$2"; shift 2 ;;
        --firmware-verify-command) require_value "$@"; FIRMWARE_VERIFY_COMMAND="$2"; shift 2 ;;
        --firmware-verify-sha256) require_value "$@"; FIRMWARE_VERIFY_SHA256="$2"; shift 2 ;;
        --quiesce-services) require_value "$@"; QUIESCE_SERVICES="$2"; shift 2 ;;
        --quiesce-processes) require_value "$@"; QUIESCE_PROCESSES="$2"; shift 2 ;;
        --quiesce-containers) require_value "$@"; QUIESCE_CONTAINERS="$2"; shift 2 ;;
        --container-stop-timeout) require_value "$@"; CONTAINER_STOP_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --container-restore-timeout) require_value "$@"; CONTAINER_RESTORE_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --device-sweep-timeout) require_value "$@"; DEVICE_SWEEP_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --quiesce-failsafe-seconds) require_value "$@"; QUIESCE_FAILSAFE_SECONDS="$2"; shift 2 ;;
        --quiesce-retry-seconds) require_value "$@"; QUIESCE_RETRY_SECONDS="$2"; shift 2 ;;
        --quiesce-settle-seconds) require_value "$@"; QUIESCE_SETTLE_SECONDS="$2"; shift 2 ;;
        --node-agent-port) require_value "$@"; NODE_AGENT_PORT="$2"; shift 2 ;;
        --node-agent-host) require_value "$@"; NODE_AGENT_HOST="$2"; shift 2 ;;
        --node-agent-advertise-url) require_value "$@"; NODE_AGENT_ADVERTISE_URL="$2"; shift 2 ;;
        --allow-node-agent-plaintext) NODE_AGENT_ALLOW_PLAINTEXT="true"; shift ;;
        --node-agent-tls-cert) require_value "$@"; NODE_AGENT_TLS_CERT="$2"; shift 2 ;;
        --node-agent-tls-key) require_value "$@"; NODE_AGENT_TLS_KEY="$2"; shift 2 ;;
        --node-agent-tls-client-ca) require_value "$@"; NODE_AGENT_TLS_CLIENT_CA="$2"; shift 2 ;;
        --node-heartbeat-interval) require_value "$@"; NODE_HEARTBEAT_INTERVAL="$2"; shift 2 ;;
        --node-inflight-wait-timeout) require_value "$@"; NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS="$2"; shift 2 ;;
        --node-action-retention-seconds) require_value "$@"; NODE_ACTION_RETENTION_SECONDS="$2"; shift 2 ;;
        --node-action-max-results) require_value "$@"; NODE_ACTION_MAX_RESULTS="$2"; shift 2 ;;
        --node-instance-id) require_value "$@"; NODE_INSTANCE_ID="$2"; shift 2 ;;
        --no-start) NO_START="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

discover_fabric_manager_log_path
if [[ -n "${FABRIC_MANAGER_LOG_PATHS}" ]]; then
    validate_fabric_manager_log_paths "${FABRIC_MANAGER_LOG_PATHS}"
fi

[[ "${EUID}" -eq 0 ]] || die "run as root"
[[ -n "${CONTROL_PLANE_URL}" ]] || die "--control-plane-url is required"
[[ -n "${CLUSTER_ID}" ]] || die "--cluster-id is required"
[[ -n "${PROFILE_VERSION}" ]] || die "--runtime-profile-version is required"
[[ "${CERTIFICATE_MIN_VALIDITY_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    die "--certificate-min-validity-seconds must be positive"
if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
    [[ "${CONTROL_PLANE_CA_CERTIFICATE}" == /* ]] ||
        die "--ca-certificate must be an absolute path"
    [[ -r "${CONTROL_PLANE_CA_CERTIFICATE}" ]] ||
        die "--ca-certificate must be a readable file"
    "${SCRIPT_DIR}/verify-certificate-bundle.sh" \
        "${CONTROL_PLANE_CA_CERTIFICATE}" \
        "${CERTIFICATE_MIN_VALIDITY_SECONDS}" >/dev/null ||
        die "--ca-certificate is invalid or expires too soon"
fi
[[ "${GPU_PRODUCT_DISCOVERY}" =~ ^(auto|required|disabled)$ ]] || \
    die "--gpu-product-discovery must be auto, required, or disabled"
[[ "${METRICS_MODE}" =~ ^(auto|dcgm|nvidia-smi)$ ]] ||
    die "--metrics-mode must be auto, dcgm, or nvidia-smi"
[[ "${DCGM_EXPORTER_MODE}" =~ ^(existing|docker|disabled)$ ]] ||
    die "--dcgm-exporter must be existing, docker, or disabled"
[[ "${METRICS_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "--metrics-interval must be a positive number"
[[ "${GPU_INVENTORY_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    die "GPU inventory interval must be a positive integer"
[[ "${METRICS_INTERVAL}" != "0" ]] || die "--metrics-interval must be positive"
[[ "${DCGM_EDGE_FILTER_ENABLED}" =~ ^(true|false)$ ]] ||
    die "GPU_FAULT_DCGM_EDGE_FILTER_ENABLED must be true or false"
for dcgm_filter_count in \
    DCGM_HEALTH_SUMMARY_SECONDS \
    KERNEL_HEALTH_SUMMARY_SECONDS \
    FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS \
    NODE_LOG_HEALTH_SUMMARY_SECONDS \
    DCGM_EDGE_CONFIRMATION_SAMPLES \
    DCGM_HISTORY_MAX_POINTS; do
    [[ "${!dcgm_filter_count}" =~ ^[1-9][0-9]*$ ]] ||
        die "${dcgm_filter_count} must be a positive integer"
done
[[ "${DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
    awk -v value="${DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD}" \
        'BEGIN { exit !(value > 0 && value <= 1) }' ||
    die "GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD must be within (0, 1]"
[[ "${COLLECTOR_GZIP_MIN_BYTES}" =~ ^[0-9]+$ ]] ||
    die "GPU_FAULT_COLLECTOR_GZIP_MIN_BYTES must be a non-negative integer"
[[ "${HOST_EDGE_FILTER_ENABLED}" =~ ^(true|false)$ ]] ||
    die "GPU_FAULT_HOST_EDGE_FILTER_ENABLED must be true or false"
[[ "${RANK_LIVENESS_ENABLED}" =~ ^(true|false)$ ]] ||
    die "GPU_FAULT_RANK_LIVENESS_ENABLED must be true or false"
[[ "${RANK_PROGRESS_MIN_WRITE_BPS}" =~ ^[0-9]+$ ]] ||
    die "GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS must be a non-negative integer"
[[ "${RANK_PROGRESS_MIN_CPU_CORES}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
    awk -v value="${RANK_PROGRESS_MIN_CPU_CORES}" \
        'BEGIN { exit !(value > 0) }' ||
    die "GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES must be positive"
[[ "${RANK_PROGRESS_GPU_IDLE_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
    awk -v value="${RANK_PROGRESS_GPU_IDLE_PERCENT}" \
        'BEGIN { exit !(value >= 0 && value <= 100) }' ||
    die "GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT must be from 0 to 100"
for host_filter_count in \
    HOST_HEALTH_SUMMARY_SECONDS \
    HOST_HISTORY_MAX_POINTS; do
    [[ "${!host_filter_count}" =~ ^[1-9][0-9]*$ ]] ||
        die "${host_filter_count} must be a positive integer"
done
[[ "${HOST_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ &&
    "${HOST_INTERVAL}" != "0" ]] ||
    die "--host-interval must be a positive number"
if [[ -n "${EXPECTED_GPU_COUNT}" ]]; then
    [[ "${EXPECTED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] ||
        die "--expected-gpu-count must be a positive integer"
fi
if [[ -n "${EXPECTED_EFA_DEVICE_COUNT}" ]]; then
    [[ "${EXPECTED_EFA_DEVICE_COUNT}" =~ ^[1-9][0-9]*$ ]] ||
        die "--expected-efa-device-count must be a positive integer"
fi
[[ "${INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES}" =~ ^[1-9][0-9]*$ ]] ||
    die "--inventory-mismatch-samples must be a positive integer"
[[ "${LOG_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ &&
    "${LOG_INTERVAL}" != "0" ]] ||
    die "--log-interval must be a positive number"
for node_log_limit in \
    NODE_LOG_MAX_ENTRIES_PER_BATCH \
    NODE_LOG_MAX_BATCH_BYTES \
    NODE_LOG_MAX_ENTRY_BYTES; do
    value="${!node_log_limit}"
    [[ "${value}" =~ ^[0-9]+$ && "${value}" -gt 0 ]] ||
        die "${node_log_limit} must be a positive integer"
done
(( NODE_LOG_MAX_ENTRY_BYTES <= NODE_LOG_MAX_BATCH_BYTES )) ||
    die "node log entry limit must not exceed batch limit"
[[ "${FABRIC_MANAGER_LOG_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ &&
    "${FABRIC_MANAGER_LOG_INTERVAL}" != "0" ]] ||
    die "--fabric-manager-log-interval must be a positive number"
if [[ "${FABRIC_MANAGER_JOURNAL}" == "true" ]]; then
    [[ -n "${FABRIC_MANAGER_IDENTIFIERS}" ]] ||
        die "--fabric-manager-identifiers cannot be empty"
elif [[ -z "${FABRIC_MANAGER_LOG_PATHS}" ]]; then
    die "Fabric Manager collector requires journald or file paths"
fi
[[ "${NODE_AGENT_PORT}" =~ ^[0-9]+$ ]] ||
    die "--node-agent-port must be an integer"
(( NODE_AGENT_PORT >= 1 && NODE_AGENT_PORT <= 65535 )) ||
    die "--node-agent-port must be between 1 and 65535"
[[ "${NODE_HEARTBEAT_INTERVAL}" =~ ^[0-9]+$ ]] ||
    die "--node-heartbeat-interval must be an integer"
(( NODE_HEARTBEAT_INTERVAL >= 5 )) ||
    die "--node-heartbeat-interval must be at least 5 seconds"
[[ "${CONTAINER_RESTORE_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--container-restore-timeout must be an integer"
(( CONTAINER_RESTORE_TIMEOUT_SECONDS >= 30 )) ||
    die "--container-restore-timeout must be at least 30 seconds"
[[ "${NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--node-inflight-wait-timeout must be an integer"
(( NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS >= 10 )) ||
    die "--node-inflight-wait-timeout must be at least 10 seconds"
[[ "${NODE_ACTION_RETENTION_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--node-action-retention-seconds must be an integer"
(( NODE_ACTION_RETENTION_SECONDS >= 600 )) ||
    die "--node-action-retention-seconds must be at least 600"
[[ "${NODE_ACTION_MAX_RESULTS}" =~ ^[0-9]+$ ]] ||
    die "--node-action-max-results must be an integer"
(( NODE_ACTION_MAX_RESULTS >= 100 )) ||
    die "--node-action-max-results must be at least 100"
[[ "${DIAGNOSTIC_RETENTION_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--diagnostic-retention-seconds must be an integer"
(( DIAGNOSTIC_RETENTION_SECONDS >= 3600 )) ||
    die "--diagnostic-retention-seconds must be at least 3600"
[[ "${DIAGNOSTIC_MAX_ARCHIVES}" =~ ^[0-9]+$ ]] ||
    die "--diagnostic-max-archives must be an integer"
(( DIAGNOSTIC_MAX_ARCHIVES >= 1 )) ||
    die "--diagnostic-max-archives must be positive"
[[ "${QUIESCE_FAILSAFE_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--quiesce-failsafe-seconds must be an integer"
(( QUIESCE_FAILSAFE_SECONDS >= 30 &&
    QUIESCE_FAILSAFE_SECONDS <= 3600 )) ||
    die "--quiesce-failsafe-seconds must be between 30 and 3600"
[[ "${QUIESCE_RETRY_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--quiesce-retry-seconds must be an integer"
(( QUIESCE_RETRY_SECONDS >= 10 &&
    QUIESCE_RETRY_SECONDS <= 600 )) ||
    die "--quiesce-retry-seconds must be between 10 and 600"
[[ "${QUIESCE_SETTLE_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "--quiesce-settle-seconds must be a non-negative number"
[[ "${DEVICE_SWEEP_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] ||
    die "--device-sweep-timeout must be an integer"
(( DEVICE_SWEEP_TIMEOUT_SECONDS <= 120 )) ||
    die "--device-sweep-timeout must be between 0 and 120"
if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    if [[ -n "${NODE_ACTION_SECRET}" &&
        -n "${NODE_ACTION_SECRET_FILE}" ]]; then
        die "use only one node action secret input"
    fi
    if [[ -n "${NODE_ACTION_SECRET_FILE}" ]]; then
        [[ -r "${NODE_ACTION_SECRET_FILE}" ]] ||
            die "node action secret file is not readable"
        IFS= read -r NODE_ACTION_SECRET < "${NODE_ACTION_SECRET_FILE}" ||
            [[ -n "${NODE_ACTION_SECRET}" ]]
    fi
    [[ "${#NODE_ACTION_SECRET}" -ge 32 ]] ||
        die "node action secret must be at least 32 characters"
    NODE_ACTION_KEY_VERSION="${NODE_ACTION_KEY_VERSION_OVERRIDE:-1}"
    [[ "${NODE_ACTION_KEY_VERSION}" =~ ^(1|2)$ ]] ||
        die "--node-action-key-version must be 1 or 2"
    if [[ "${DERIVE_NODE_ACTION_SECRET}" == "true" ]]; then
        [[ -z "${NODE_ACTION_KEY_VERSION_OVERRIDE}" ||
            "${NODE_ACTION_KEY_VERSION_OVERRIDE}" == "2" ]] ||
            die "derived node action secrets require key version 2"
        NODE_ACTION_SECRET="$(
            GPU_FAULT_FLEET_SECRET="${NODE_ACTION_SECRET}" \
            GPU_FAULT_CLUSTER_ID="${CLUSTER_ID}" \
            GPU_FAULT_NODE_ID="${NODE_ID}" \
            "${PYTHON_COMMAND}" -c '
import hashlib
import hmac
import os

context = (
    "gpu-fault/node-action/v1\0"
    + os.environ["GPU_FAULT_CLUSTER_ID"]
    + "\0"
    + os.environ["GPU_FAULT_NODE_ID"]
).encode()
print(
    hmac.new(
        os.environ["GPU_FAULT_FLEET_SECRET"].encode(),
        context,
        hashlib.sha256,
    ).hexdigest()
)
'
        )"
        NODE_ACTION_KEY_VERSION="2"
    fi
    # The agent serves HTTPS only when it has a certificate, so the
    # advertised scheme has to follow the certificate, not the other way
    # round: a control plane calling https:// against a plain-HTTP
    # listener fails every node action with an unhelpful SSL error.
    if [[ -n "${NODE_AGENT_TLS_CERT}" || -n "${NODE_AGENT_TLS_KEY}" ]]; then
        [[ -n "${NODE_AGENT_TLS_CERT}" && -n "${NODE_AGENT_TLS_KEY}" ]] ||
            die "--node-agent-tls-cert and --node-agent-tls-key must be set together"
        [[ -r "${NODE_AGENT_TLS_CERT}" ]] ||
            die "--node-agent-tls-cert is not readable"
        [[ -r "${NODE_AGENT_TLS_KEY}" ]] ||
            die "--node-agent-tls-key is not readable"
    fi
    if [[ -n "${NODE_AGENT_TLS_CLIENT_CA}" ]]; then
        [[ -n "${NODE_AGENT_TLS_CERT}" ]] ||
            die "--node-agent-tls-client-ca requires --node-agent-tls-cert"
        [[ -r "${NODE_AGENT_TLS_CLIENT_CA}" ]] ||
            die "--node-agent-tls-client-ca is not readable"
    fi
    if [[ -z "${NODE_AGENT_ADVERTISE_URL}" ]]; then
        if [[ -n "${NODE_AGENT_TLS_CERT}" ]]; then
            NODE_AGENT_ADVERTISE_URL="https://${NODE_ID}:${NODE_AGENT_PORT}"
        else
            NODE_AGENT_ADVERTISE_URL="http://${NODE_ID}:${NODE_AGENT_PORT}"
        fi
    fi
    if [[ -z "${NODE_AGENT_HOST}" ]]; then
        NODE_AGENT_HOST="${NODE_ID}"
    fi
    [[ "${NODE_AGENT_ADVERTISE_URL}" =~ ^https?:// ]] ||
        die "--node-agent-advertise-url must use http or https"
    if [[ -n "${NODE_AGENT_TLS_CERT}" &&
        "${NODE_AGENT_ADVERTISE_URL}" != https://* ]]; then
        die "--node-agent-advertise-url must be https when TLS is configured"
    fi
    if [[ -z "${NODE_AGENT_TLS_CERT}" &&
        "${NODE_AGENT_ADVERTISE_URL}" == https://* ]]; then
        die "advertising https requires --node-agent-tls-cert and --node-agent-tls-key"
    fi
    if [[ -z "${NODE_AGENT_TLS_CERT}" &&
        "${NODE_AGENT_ALLOW_PLAINTEXT}" != "true" ]]; then
        die "plain HTTP node agent requires --allow-node-agent-plaintext"
    fi
    if [[ -z "${NODE_INSTANCE_ID}" &&
        -r /sys/devices/virtual/dmi/id/product_uuid ]]; then
        NODE_INSTANCE_ID="$(
            tr '[:upper:]' '[:lower:]' \
                < /sys/devices/virtual/dmi/id/product_uuid
        )"
    fi
    if [[ -z "${NODE_INSTANCE_ID}" && -r /etc/machine-id ]]; then
        IFS= read -r NODE_INSTANCE_ID < /etc/machine-id || true
    fi
    [[ -n "${NODE_INSTANCE_ID}" ]] ||
        die "cannot determine node instance ID; pass --node-instance-id"
elif [[ "${ALLOW_GPU_RESET}" == "true" ||
    "${ALLOW_FABRIC_RESET}" == "true" ||
    "${ALLOW_SERVICE_QUIESCE}" == "true" ||
    "${ALLOW_FABRIC_MANAGER_RESTART}" == "true" ||
    "${ALLOW_FIELD_DIAGNOSTIC}" == "true" ||
    "${ALLOW_DRIVER_REMEDIATION}" == "true" ||
    "${ALLOW_FIRMWARE_UPDATE}" == "true" ]]; then
    die "node mutation options require --enable-node-agent"
fi
if [[ "${ALLOW_SERVICE_QUIESCE}" == "true" &&
    "${ALLOW_GPU_RESET}" != "true" ]]; then
    die "--allow-service-quiesce requires --allow-gpu-reset"
fi
if [[ "${ALLOW_FABRIC_RESET}" == "true" &&
    "${ALLOW_SERVICE_QUIESCE}" != "true" ]]; then
    die "--allow-fabric-reset requires --allow-service-quiesce"
fi
[[ "${DIAGNOSTIC_OUTPUT_DIR}" == /* ]] ||
    die "--diagnostic-output-dir must be absolute"
if [[ -n "${DIAGNOSTIC_S3_URI}" ]]; then
    [[ "${DIAGNOSTIC_S3_URI}" == s3://* ]] ||
        die "--diagnostic-s3-uri must use s3://"
fi
if [[ "${ALLOW_FIELD_DIAGNOSTIC}" == "true" ]]; then
    [[ "${FIELD_DIAGNOSTIC_COMMAND}" == /* ]] ||
        die "--field-diagnostic-command must start with an absolute path"
    [[ "${FIELD_DIAGNOSTIC_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] ||
        die "--field-diagnostic-sha256 must contain 64 hex characters"
    [[ "${FIELD_DIAGNOSTIC_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] &&
        (( FIELD_DIAGNOSTIC_TIMEOUT_SECONDS >= 60 &&
           FIELD_DIAGNOSTIC_TIMEOUT_SECONDS <= 7200 )) ||
        die "--field-diagnostic-timeout must be from 60 to 7200"
    if [[ -n "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}" ]]; then
        [[ "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}" == /* ]] ||
            die "--memory-field-diagnostic-command must start with an absolute path"
        [[ "${MEMORY_FIELD_DIAGNOSTIC_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] ||
            die "--memory-field-diagnostic-sha256 must contain 64 hex characters"
    fi
elif [[ -n "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}" ||
    -n "${MEMORY_FIELD_DIAGNOSTIC_SHA256}" ]]; then
    die "memory Field Diagnostic requires --allow-field-diagnostic"
fi
if [[ "${ALLOW_DRIVER_REMEDIATION}" == "true" ]]; then
    [[ "${ALLOW_SERVICE_QUIESCE}" == "true" ]] ||
        die "--allow-driver-remediation requires --allow-service-quiesce"
    [[ "${DRIVER_REMEDIATION_COMMAND}" == /* ]] ||
        die "--driver-remediation-command must start with an absolute path"
    [[ "${DRIVER_REMEDIATION_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] ||
        die "--driver-remediation-sha256 must contain 64 hex characters"
    [[ "${TARGET_DRIVER_BRANCH}" =~ ^[0-9]+$ ]] ||
        die "--target-driver-branch must be an integer"
fi
if [[ "${ALLOW_FIRMWARE_UPDATE}" == "true" ]]; then
    [[ "${ALLOW_SERVICE_QUIESCE}" == "true" ]] ||
        die "--allow-firmware-update requires --allow-service-quiesce"
    [[ "${FIRMWARE_UPDATE_COMMAND}" == /* ]] ||
        die "--firmware-update-command must start with an absolute path"
    [[ "${FIRMWARE_UPDATE_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] ||
        die "--firmware-update-sha256 must contain 64 hex characters"
    [[ -n "${TARGET_FIRMWARE_VERSION}" ]] ||
        die "--target-firmware-version is required"
    [[ "${FIRMWARE_VERIFY_COMMAND}" == /* ]] ||
        die "--firmware-verify-command must start with an absolute path"
    [[ "${FIRMWARE_VERIFY_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] ||
        die "--firmware-verify-sha256 must contain 64 hex characters"
fi
if [[ "${METRICS_MODE}" == "dcgm" &&
    "${DCGM_EXPORTER_MODE}" == "disabled" ]]; then
    die "dcgm metrics mode cannot use a disabled exporter"
fi
if [[ "${METRICS_MODE}" == "nvidia-smi" &&
    "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" != "true" ]]; then
    die "nvidia-smi metrics collector is disabled; explicitly enable it only after validation"
fi
if [[ "${METRICS_MODE}" == "auto" &&
    "${DCGM_EXPORTER_MODE}" == "disabled" &&
    "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" != "true" ]]; then
    die "DCGM exporter cannot be disabled while nvidia-smi metrics collector is disabled"
fi
if [[ "${DISABLE_KERNEL}" == "false" ]]; then
    [[ -e /dev/kmsg ]] ||
        die "/dev/kmsg is unavailable; use --disable-kernel only when another XID source exists"
fi
for value in "${CONTROL_PLANE_URL}" "${CLUSTER_ID}" "${PROFILE_VERSION}" \
    "${TOKEN}" "${NODE_ID}" "${GPU_PRODUCT}" "${DRIVER_BRANCH}" \
    "${CUDA_VERSION}" "${DCGM_METRICS_URL}" "${DCGM_EXPORTER_IMAGE}" \
    "${FILESYSTEMS}" "${REQUIRED_INTERFACES}" \
    "${TRAINING_LOG_PATHS}"; do
    [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] ||
        die "configuration values cannot contain newlines"
done
for value in "${QUIESCE_SERVICES}" "${QUIESCE_PROCESSES}"; do
    [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] ||
        die "quiesce configuration cannot contain newlines"
done
[[ "${NODE_AGENT_ADVERTISE_URL}" != *$'\n'* &&
    "${NODE_AGENT_ADVERTISE_URL}" != *$'\r'* ]] ||
    die "node agent advertise URL cannot contain newlines"
[[ "${NODE_INSTANCE_ID}" != *$'\n'* &&
    "${NODE_INSTANCE_ID}" != *$'\r'* ]] ||
    die "node instance ID cannot contain newlines"

command -v "${PYTHON_COMMAND}" >/dev/null ||
    die "Python executable is unavailable: ${PYTHON_COMMAND}"
"${PYTHON_COMMAND}" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 12))' ||
    die "Python 3.12 or newer is required"
command -v systemctl >/dev/null || die "systemd is required"
systemctl stop gpu-fault-node-agent.service >/dev/null 2>&1 || true
shopt -s nullglob
existing_quiesce_states=(
    /var/lib/gpu-fault/quiesce/quiesce-*.json
)
if (( ${#existing_quiesce_states[@]} > 0 )); then
    existing_restore_command="/opt/gpu-fault/venv/bin/gpu-fault-restore-gpu-services"
    [[ -x "${existing_restore_command}" ]] ||
        die "quiesce state exists but restore command is unavailable"
    for state_file in "${existing_quiesce_states[@]}"; do
        "${existing_restore_command}" --state-file "${state_file}" ||
            die "GPU services could not be restored; upgrade aborted"
    done
fi
command -v curl >/dev/null || die "curl is required"
command -v nvidia-smi >/dev/null || die "NVIDIA driver/nvidia-smi is required"
if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    command -v timeout >/dev/null ||
        die "timeout is required for bounded hung diagnostics"
    command -v ps >/dev/null ||
        die "ps is required for hung diagnostics"
    command -v ss >/dev/null ||
        die "ss is required for hung network diagnostics"
    command -v strace >/dev/null ||
        die "strace is required for hung process diagnostics"
    if compgen -G "/sys/class/infiniband/*" >/dev/null; then
        command -v rdma >/dev/null ||
            die "rdma is required for hung EFA/RDMA diagnostics"
        command -v ethtool >/dev/null ||
            die "ethtool is required for hung EFA/RDMA diagnostics"
    fi
fi
nvidia-smi -L >/dev/null || die "nvidia-smi cannot enumerate GPUs"
NVIDIA_SMI="$(command -v nvidia-smi)"
sed "s|@NVIDIA_SMI@|${NVIDIA_SMI}|g" \
    "${REPO_DIR}/deploy/systemd/gpu-fault-gpu-persistence.service" \
    > /etc/systemd/system/gpu-fault-gpu-persistence.service
chmod 0644 /etc/systemd/system/gpu-fault-gpu-persistence.service
"${PYTHON_COMMAND}" -c \
    'import sys; value=float(sys.argv[1]); raise SystemExit(not 0 <= value <= 60)' \
    "${QUIESCE_SETTLE_SECONDS}" ||
    die "--quiesce-settle-seconds must be between 0 and 60"

dcgm_ready() {
    local output
    output="$(mktemp)"
    if curl --fail --silent --show-error --output "${output}" \
        "${DCGM_METRICS_URL}" &&
        grep -q 'DCGM_FI_DEV_' "${output}"; then
        rm -f "${output}"
        return 0
    fi
    rm -f "${output}"
    return 1
}

# wheel 有两种布局。节点安装包（deploy/node/build-node-installer-bundle.sh）
# 把 wheel 平铺在 dist/ 下，没有 current-release.json，也没有 scripts/，所以
# 这里不能改用 scripts/release-artifact-path.py。仓库 checkout 则是内容寻址的
# dist/<release_id>/，权威指针在 dist/current-release.json —— 原来只
# find -maxdepth 1 -type f，checkout 里一个都匹配不到，直接 die
# "collector wheel not found"。指针存在时用指针并核对 wheel_sha256，
# 避免在多份 release 里按 mtime 猜。
EXPECTED_WHEEL_SHA256=""
if [[ -z "${WHEEL}" ]]; then
    RELEASE_MANIFEST="${REPO_DIR}/dist/current-release.json"
    if [[ -f "${RELEASE_MANIFEST}" ]]; then
        MANIFEST_WHEEL="$("${PYTHON_COMMAND}" -c \
            'import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
wheel = pathlib.Path(str(manifest["wheel"]))
root = pathlib.Path(sys.argv[2])
print(wheel if wheel.is_absolute() else root / wheel)
print(manifest["wheel_sha256"])' \
            "${RELEASE_MANIFEST}" "${REPO_DIR}")" ||
            die "cannot read release manifest: ${RELEASE_MANIFEST}"
        WHEEL="${MANIFEST_WHEEL%%$'\n'*}"
        EXPECTED_WHEEL_SHA256="${MANIFEST_WHEEL##*$'\n'}"
    else
        WHEEL="$(find "${REPO_DIR}/dist" -maxdepth 1 -type f \
            -name 'gpu_fault_control_plane-*.whl' -printf '%T@ %p\n' 2>/dev/null |
            sort -nr | head -n1 | cut -d' ' -f2-)"
    fi
fi
[[ -n "${WHEEL}" && -f "${WHEEL}" ]] ||
    die "collector wheel not found; pass --wheel PATH"
WHEEL_SHA256="$(sha256sum "${WHEEL}" | cut -d' ' -f1)"
[[ -z "${EXPECTED_WHEEL_SHA256}" ||
    "${WHEEL_SHA256}" == "${EXPECTED_WHEEL_SHA256}" ]] ||
    die "wheel ${WHEEL} sha256 ${WHEEL_SHA256} does not match ${RELEASE_MANIFEST}"

if [[ "${DCGM_EXPORTER_MODE}" == "docker" ]]; then
    [[ "${METRICS_MODE}" != "nvidia-smi" ]] ||
        die "docker DCGM exporter is unused with nvidia-smi metrics mode"
    if command -v pgrep >/dev/null &&
        pgrep -x nv-hostengine >/dev/null; then
        die "an nv-hostengine is already running; use its exporter endpoint or nvidia-smi fallback"
    fi
    if dcgm_ready; then
        die "a DCGM exporter already answers at ${DCGM_METRICS_URL}; use --dcgm-exporter existing"
    fi
    [[ -n "${DCGM_EXPORTER_IMAGE}" ]] ||
        die "--dcgm-exporter-image is required for docker mode"
    command -v docker >/dev/null || die "docker is required for DCGM exporter mode"
    docker info >/dev/null || die "Docker daemon is unavailable"
fi

install -d -m 0755 /opt/gpu-fault /etc/gpu-fault \
    /var/lib/gpu-fault
"${PYTHON_COMMAND}" -m venv /opt/gpu-fault/venv
PIP_ARGS=(install --upgrade "${WHEEL}[collectors]")
if [[ -n "${WHEELHOUSE}" ]]; then
    [[ -d "${WHEELHOUSE}" ]] || die "wheelhouse does not exist: ${WHEELHOUSE}"
    PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
fi
/opt/gpu-fault/venv/bin/python -m pip "${PIP_ARGS[@]}"
/opt/gpu-fault/venv/bin/python -m pip install \
    --force-reinstall --no-deps "${WHEEL}"
if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    /opt/gpu-fault/venv/bin/python -m pip install \
        --force-reinstall --no-deps --only-binary=:all: \
        "py-spy==${PY_SPY_VERSION}"
    [[ "$(
        sha256sum /opt/gpu-fault/venv/bin/py-spy | cut -d' ' -f1
    )" == "${PY_SPY_BINARY_SHA256}" ]] ||
        die "py-spy binary SHA-256 mismatch"
    /opt/gpu-fault/venv/bin/py-spy --version >/dev/null ||
        die "py-spy installation verification failed"
fi

install -m 0644 "${REPO_DIR}/deploy/dataplane/dcgm-counters.csv" \
    /etc/gpu-fault/dcgm-counters.csv
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-kernel-collector.service" \
    /etc/systemd/system/gpu-fault-kernel-collector.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-metrics-collector.service" \
    /etc/systemd/system/gpu-fault-metrics-collector.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-host-collector.service" \
    /etc/systemd/system/gpu-fault-host-collector.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-log-collector.service" \
    /etc/systemd/system/gpu-fault-log-collector.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-fabric-manager-collector.service" \
    /etc/systemd/system/gpu-fault-fabric-manager-collector.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-certificate-check.service" \
    /etc/systemd/system/gpu-fault-certificate-check.service
install -m 0644 \
    "${REPO_DIR}/deploy/systemd/gpu-fault-certificate-check.timer" \
    /etc/systemd/system/gpu-fault-certificate-check.timer
if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    install -m 0644 \
        "${REPO_DIR}/deploy/systemd/gpu-fault-node-agent.service" \
        /etc/systemd/system/gpu-fault-node-agent.service
fi
install -m 0755 "${SCRIPT_DIR}/verify-gpu-fault-collector.sh" \
    /opt/gpu-fault/verify
install -m 0755 "${SCRIPT_DIR}/verify-certificate-bundle.sh" \
    /opt/gpu-fault/verify-certificate-bundle
install -m 0755 "${SCRIPT_DIR}/check-control-plane-certificate.sh" \
    /opt/gpu-fault/check-control-plane-certificate
install -m 0755 "${SCRIPT_DIR}/uninstall-gpu-fault-collector.sh" \
    /opt/gpu-fault/uninstall
if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
    install -m 0644 "${CONTROL_PLANE_CA_CERTIFICATE}" \
        /etc/gpu-fault/control-plane-ca.crt
fi

systemd_quote() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "${value}"
}

write_env() {
    local key="$1"
    local value="$2"
    printf '%s=%s\n' "${key}" "$(systemd_quote "${value}")"
}

{
    write_env GPU_FAULT_CONTROL_PLANE_URL "${CONTROL_PLANE_URL}"
    write_env GPU_FAULT_CLUSTER_ID "${CLUSTER_ID}"
    write_env GPU_FAULT_RUNTIME_PROFILE_VERSION "${PROFILE_VERSION}"
    write_env GPU_FAULT_CONTROL_PLANE_TOKEN "${TOKEN}"
    if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
        write_env SSL_CERT_FILE "/etc/gpu-fault/control-plane-ca.crt"
        write_env GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS \
            "${CERTIFICATE_MIN_VALIDITY_SECONDS}"
    fi
    write_env NODE_NAME "${NODE_ID}"
    write_env GPU_FAULT_NODE_INSTANCE_TYPE "${NODE_INSTANCE_TYPE}"
    write_env GPU_FAULT_GPU_PRODUCT "${GPU_PRODUCT}"
    write_env GPU_FAULT_GPU_PRODUCT_DISCOVERY "${GPU_PRODUCT_DISCOVERY}"
    write_env GPU_FAULT_DRIVER_BRANCH "${DRIVER_BRANCH}"
    write_env GPU_FAULT_CUDA_VERSION "${CUDA_VERSION}"
    write_env GPU_FAULT_METRICS_INTERVAL_SECONDS "${METRICS_INTERVAL}"
    write_env GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS \
        "${GPU_INVENTORY_INTERVAL_SECONDS}"
    write_env GPU_FAULT_NODE_INSTANCE_ID "${NODE_INSTANCE_ID}"
    write_env GPU_FAULT_DCGM_EDGE_FILTER_ENABLED \
        "${DCGM_EDGE_FILTER_ENABLED}"
    write_env GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS \
        "${DCGM_HEALTH_SUMMARY_SECONDS}"
    write_env GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS \
        "${KERNEL_HEALTH_SUMMARY_SECONDS}"
    write_env GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS \
        "${FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS}"
    write_env GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS \
        "${NODE_LOG_HEALTH_SUMMARY_SECONDS}"
    write_env GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES \
        "${DCGM_EDGE_CONFIRMATION_SAMPLES}"
    write_env GPU_FAULT_DCGM_HISTORY_MAX_POINTS \
        "${DCGM_HISTORY_MAX_POINTS}"
    write_env GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD \
        "${DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD}"
    write_env GPU_FAULT_COLLECTOR_GZIP_MIN_BYTES \
        "${COLLECTOR_GZIP_MIN_BYTES}"
    write_env GPU_FAULT_HOST_EDGE_FILTER_ENABLED \
        "${HOST_EDGE_FILTER_ENABLED}"
    write_env GPU_FAULT_RANK_LIVENESS_ENABLED \
        "${RANK_LIVENESS_ENABLED}"
    write_env GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS \
        "${RANK_PROGRESS_MIN_WRITE_BPS}"
    write_env GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES \
        "${RANK_PROGRESS_MIN_CPU_CORES}"
    write_env GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT \
        "${RANK_PROGRESS_GPU_IDLE_PERCENT}"
    write_env GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS \
        "${HOST_HEALTH_SUMMARY_SECONDS}"
    write_env GPU_FAULT_HOST_HISTORY_MAX_POINTS \
        "${HOST_HISTORY_MAX_POINTS}"
    write_env GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR \
        "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}"
    write_env GPU_FAULT_HOST_INTERVAL_SECONDS "${HOST_INTERVAL}"
    write_env GPU_FAULT_EXPECTED_GPU_COUNT "${EXPECTED_GPU_COUNT}"
    write_env GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT \
        "${EXPECTED_EFA_DEVICE_COUNT}"
    write_env GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES \
        "${INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES}"
    write_env GPU_FAULT_LOG_INTERVAL_SECONDS "${LOG_INTERVAL}"
    write_env GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR \
        "${ENABLE_NODE_LOG_COLLECTOR}"
    write_env GPU_FAULT_NODE_LOG_MAX_ENTRIES_PER_BATCH \
        "${NODE_LOG_MAX_ENTRIES_PER_BATCH}"
    write_env GPU_FAULT_NODE_LOG_MAX_BATCH_BYTES \
        "${NODE_LOG_MAX_BATCH_BYTES}"
    write_env GPU_FAULT_NODE_LOG_MAX_ENTRY_BYTES \
        "${NODE_LOG_MAX_ENTRY_BYTES}"
    write_env GPU_FAULT_FABRIC_MANAGER_LOG_INTERVAL_SECONDS \
        "${FABRIC_MANAGER_LOG_INTERVAL}"
    write_env GPU_FAULT_FABRIC_MANAGER_JOURNAL \
        "${FABRIC_MANAGER_JOURNAL}"
    write_env GPU_FAULT_FABRIC_MANAGER_IDENTIFIERS \
        "${FABRIC_MANAGER_IDENTIFIERS}"
    write_env GPU_FAULT_FABRIC_MANAGER_LOG_PATHS \
        "${FABRIC_MANAGER_LOG_PATHS}"
    write_env GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND \
        "${NVSWITCH_TOPOLOGY_COMMAND}"
    write_env GPU_FAULT_FABRIC_MANAGER_STATE_PATH \
        "/var/lib/gpu-fault/fabric-manager-collector-state.json"
    write_env GPU_FAULT_FILESYSTEMS "${FILESYSTEMS}"
    write_env GPU_FAULT_REQUIRED_INTERFACES "${REQUIRED_INTERFACES}"
    write_env GPU_FAULT_TRAINING_LOG_PATHS "${TRAINING_LOG_PATHS}"
    write_env GPU_FAULT_DCGM_METRICS_URL "${DCGM_METRICS_URL}"
} > /etc/gpu-fault/collector.env
chmod 0600 /etc/gpu-fault/collector.env

if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    NODE_ALLOWED_OPERATIONS="COLLECT_HUNG_TRIAGE,COLLECT_DIAGNOSTIC_BUNDLE,RUN_DCGM_DIAGNOSTIC,QUIESCE_GPU_SERVICES,VERIFY_NO_GPU_CLIENTS,TRIGGER_HEALTH_SNAPSHOT,RESET_GPU,RESET_ALL_GPUS_NVSWITCHES,RESTORE_GPU_SERVICES,RESTART_FABRIC_MANAGER"
    if [[ "${ALLOW_FIELD_DIAGNOSTIC}" == "true" ]]; then
        NODE_ALLOWED_OPERATIONS+=",RUN_NVLINK74_WORKFLOW"
        if [[ -n "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}" ]]; then
            NODE_ALLOWED_OPERATIONS+=",RUN_FIELD_DIAGNOSTIC"
        fi
    fi
    if [[ "${ALLOW_DRIVER_REMEDIATION}" == "true" ]]; then
        NODE_ALLOWED_OPERATIONS+=",REMEDIATE_DRIVER"
    fi
    if [[ "${ALLOW_EFA_DRIVER_REMEDIATION}" == "true" ]]; then
        NODE_ALLOWED_OPERATIONS+=",REMEDIATE_EFA_DRIVER"
    fi
    if [[ "${ALLOW_FIRMWARE_UPDATE}" == "true" ]]; then
        NODE_ALLOWED_OPERATIONS+=",UPDATE_SOFTWARE_FIRMWARE"
    fi
    {
        write_env GPU_FAULT_NODE_ACTION_SECRET "${NODE_ACTION_SECRET}"
        write_env GPU_FAULT_NODE_ACTION_KEY_VERSION \
            "${NODE_ACTION_KEY_VERSION}"
        write_env GPU_FAULT_NODE_ALLOWED_OPERATIONS \
            "${NODE_ALLOWED_OPERATIONS}"
        write_env GPU_FAULT_NODE_ALLOW_GPU_RESET "${ALLOW_GPU_RESET}"
        write_env GPU_FAULT_NODE_SINGLE_GPU_RESET_SUPPORTED \
            "${SINGLE_GPU_RESET_SUPPORTED}"
        write_env GPU_FAULT_NODE_ALLOW_FABRIC_RESET \
            "${ALLOW_FABRIC_RESET}"
        write_env GPU_FAULT_NODE_ALLOW_FABRIC_MANAGER_RESTART \
            "${ALLOW_FABRIC_MANAGER_RESTART}"
        write_env GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE \
            "${ALLOW_SERVICE_QUIESCE}"
        write_env GPU_FAULT_DIAGNOSTIC_OUTPUT_DIR \
            "${DIAGNOSTIC_OUTPUT_DIR}"
        write_env GPU_FAULT_DIAGNOSTIC_S3_URI \
            "${DIAGNOSTIC_S3_URI}"
        write_env GPU_FAULT_DIAGNOSTIC_RETENTION_SECONDS \
            "${DIAGNOSTIC_RETENTION_SECONDS}"
        write_env GPU_FAULT_DIAGNOSTIC_MAX_ARCHIVES \
            "${DIAGNOSTIC_MAX_ARCHIVES}"
        write_env GPU_FAULT_PYTHON_STACK_TOOL \
            "/opt/gpu-fault/venv/bin/py-spy"
        write_env GPU_FAULT_NODE_ALLOW_FIELD_DIAGNOSTIC \
            "${ALLOW_FIELD_DIAGNOSTIC}"
        write_env GPU_FAULT_FIELD_DIAGNOSTIC_COMMAND \
            "${FIELD_DIAGNOSTIC_COMMAND}"
        write_env GPU_FAULT_FIELD_DIAGNOSTIC_SHA256 \
            "${FIELD_DIAGNOSTIC_SHA256}"
        write_env GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_COMMAND \
            "${MEMORY_FIELD_DIAGNOSTIC_COMMAND}"
        write_env GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_SHA256 \
            "${MEMORY_FIELD_DIAGNOSTIC_SHA256}"
        write_env GPU_FAULT_FIELD_DIAGNOSTIC_TIMEOUT_SECONDS \
            "${FIELD_DIAGNOSTIC_TIMEOUT_SECONDS}"
        write_env GPU_FAULT_NODE_ALLOW_DRIVER_REMEDIATION \
            "${ALLOW_DRIVER_REMEDIATION}"
        write_env GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION \
            "${ALLOW_EFA_DRIVER_REMEDIATION}"
        write_env GPU_FAULT_DRIVER_REMEDIATION_COMMAND \
            "${DRIVER_REMEDIATION_COMMAND}"
        write_env GPU_FAULT_DRIVER_REMEDIATION_SHA256 \
            "${DRIVER_REMEDIATION_SHA256}"
        write_env GPU_FAULT_TARGET_DRIVER_BRANCH \
            "${TARGET_DRIVER_BRANCH}"
        write_env GPU_FAULT_NODE_ALLOW_FIRMWARE_UPDATE \
            "${ALLOW_FIRMWARE_UPDATE}"
        write_env GPU_FAULT_FIRMWARE_UPDATE_COMMAND \
            "${FIRMWARE_UPDATE_COMMAND}"
        write_env GPU_FAULT_FIRMWARE_UPDATE_SHA256 \
            "${FIRMWARE_UPDATE_SHA256}"
        write_env GPU_FAULT_TARGET_FIRMWARE_VERSION \
            "${TARGET_FIRMWARE_VERSION}"
        write_env GPU_FAULT_FIRMWARE_VERIFY_COMMAND \
            "${FIRMWARE_VERIFY_COMMAND}"
        write_env GPU_FAULT_FIRMWARE_VERIFY_SHA256 \
            "${FIRMWARE_VERIFY_SHA256}"
        write_env GPU_FAULT_QUIESCE_SERVICES "${QUIESCE_SERVICES}"
        write_env GPU_FAULT_QUIESCE_PROCESSES "${QUIESCE_PROCESSES}"
        write_env GPU_FAULT_QUIESCE_CONTAINERS "${QUIESCE_CONTAINERS}"
        write_env GPU_FAULT_CONTAINER_STOP_TIMEOUT_SECONDS \
            "${CONTAINER_STOP_TIMEOUT_SECONDS}"
        write_env GPU_FAULT_CONTAINER_RESTORE_TIMEOUT_SECONDS \
            "${CONTAINER_RESTORE_TIMEOUT_SECONDS}"
        write_env GPU_FAULT_DEVICE_SWEEP_TIMEOUT_SECONDS \
            "${DEVICE_SWEEP_TIMEOUT_SECONDS}"
        write_env GPU_FAULT_QUIESCE_FAILSAFE_SECONDS \
            "${QUIESCE_FAILSAFE_SECONDS}"
        write_env GPU_FAULT_QUIESCE_RETRY_SECONDS \
            "${QUIESCE_RETRY_SECONDS}"
        write_env GPU_FAULT_QUIESCE_SETTLE_SECONDS \
            "${QUIESCE_SETTLE_SECONDS}"
        write_env GPU_FAULT_RESTORE_SETTLE_SECONDS \
            "${RESTORE_SETTLE_SECONDS}"
        write_env GPU_FAULT_QUIESCE_STATE_DIR \
            "/var/lib/gpu-fault/quiesce"
        write_env GPU_FAULT_NODE_AGENT_PORT "${NODE_AGENT_PORT}"
        if [[ -n "${NODE_AGENT_TLS_CERT}" ]]; then
            write_env GPU_FAULT_NODE_AGENT_TLS_CERT \
                "${NODE_AGENT_TLS_CERT}"
            write_env GPU_FAULT_NODE_AGENT_TLS_KEY \
                "${NODE_AGENT_TLS_KEY}"
        fi
        if [[ -n "${NODE_AGENT_TLS_CLIENT_CA}" ]]; then
            write_env GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA \
                "${NODE_AGENT_TLS_CLIENT_CA}"
        fi
        write_env GPU_FAULT_NODE_CONTROL_PLANE_URL \
            "${CONTROL_PLANE_URL}"
        write_env GPU_FAULT_NODE_CONTROL_PLANE_TOKEN "${TOKEN}"
        if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
            write_env SSL_CERT_FILE "/etc/gpu-fault/control-plane-ca.crt"
            write_env GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS \
                "${CERTIFICATE_MIN_VALIDITY_SECONDS}"
        fi
        write_env GPU_FAULT_NODE_CLUSTER_ID "${CLUSTER_ID}"
        write_env GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION \
            "${PROFILE_VERSION}"
        write_env GPU_FAULT_NODE_ARTIFACT_SHA256 "${WHEEL_SHA256}"
        write_env GPU_FAULT_NODE_ADVERTISE_URL \
            "${NODE_AGENT_ADVERTISE_URL}"
        write_env GPU_FAULT_NODE_HEARTBEAT_INTERVAL_SECONDS \
            "${NODE_HEARTBEAT_INTERVAL}"
        write_env GPU_FAULT_NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS \
            "${NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS}"
        write_env GPU_FAULT_NODE_ACTION_RETENTION_SECONDS \
            "${NODE_ACTION_RETENTION_SECONDS}"
        write_env GPU_FAULT_NODE_ACTION_MAX_RESULTS \
            "${NODE_ACTION_MAX_RESULTS}"
        write_env GPU_FAULT_NODE_AGENT_HOST "${NODE_AGENT_HOST}"
        write_env GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT \
            "${NODE_AGENT_ALLOW_PLAINTEXT}"
        write_env GPU_FAULT_NODE_INSTANCE_ID "${NODE_INSTANCE_ID}"
        write_env GPU_FAULT_NODE_ACTION_DB \
            "/var/lib/gpu-fault/node-actions.db"
        write_env NODE_NAME "${NODE_ID}"
    } > /etc/gpu-fault/node-agent.env
    chmod 0600 /etc/gpu-fault/node-agent.env
fi

if [[ "${DCGM_EXPORTER_MODE}" == "docker" ]]; then
    docker pull "${DCGM_EXPORTER_IMAGE}"
    sed "s|@CONTAINER_RUNTIME@|$(command -v docker)|g" \
        "${REPO_DIR}/deploy/systemd/gpu-fault-dcgm-exporter.service" \
        > /etc/systemd/system/gpu-fault-dcgm-exporter.service
    chmod 0644 /etc/systemd/system/gpu-fault-dcgm-exporter.service
    {
        write_env GPU_FAULT_DCGM_EXPORTER_IMAGE "${DCGM_EXPORTER_IMAGE}"
    } > /etc/gpu-fault/dcgm-exporter.env
    chmod 0600 /etc/gpu-fault/dcgm-exporter.env
fi

find /etc/systemd/system -maxdepth 1 -type f \
    \( -name 'gpu-fault-*.service' -o -name 'gpu-fault-*.timer' \) \
    -printf '%f\n' |
    sort -u > /opt/gpu-fault/installed-units.txt
chmod 0644 /opt/gpu-fault/installed-units.txt
[[ -s /opt/gpu-fault/installed-units.txt ]] ||
    die "no installed gpu-fault systemd units were recorded"

systemctl daemon-reload
systemctl enable gpu-fault-gpu-persistence.service
if [[ "${NO_START}" == "false" ]]; then
    systemctl restart gpu-fault-gpu-persistence.service
    if nvidia-smi --query-gpu=persistence_mode \
        --format=csv,noheader |
        grep -Fvx "Enabled" >/dev/null; then
        die "not every GPU entered persistence mode"
    fi
fi
if [[ "${DCGM_EXPORTER_MODE}" == "docker" ]]; then
    systemctl enable gpu-fault-dcgm-exporter.service
    if [[ "${NO_START}" == "false" ]]; then
        systemctl restart gpu-fault-dcgm-exporter.service
        for _ in {1..30}; do
            dcgm_ready && break
            sleep 1
        done
        dcgm_ready ||
            die "DCGM exporter did not expose supported metrics"
    fi
fi

if [[ "${METRICS_MODE}" == "auto" ]]; then
    if [[ "${DCGM_EXPORTER_MODE}" != "disabled" &&
        "${NO_START}" == "false" ]] && dcgm_ready; then
        METRICS_MODE="dcgm"
    elif [[ "${DCGM_EXPORTER_MODE}" == "docker" ]]; then
        METRICS_MODE="dcgm"
    elif [[ "${ENABLE_NVIDIA_SMI_METRICS_COLLECTOR}" == "true" ]]; then
        METRICS_MODE="nvidia-smi"
    else
        die "DCGM metrics are unavailable and nvidia-smi metrics collector is disabled"
    fi
fi
printf 'GPU_FAULT_METRICS_MODE=%s\n' "$(systemd_quote "${METRICS_MODE}")" \
    >> /etc/gpu-fault/collector.env

systemctl enable gpu-fault-metrics-collector.service
systemctl enable gpu-fault-host-collector.service
systemctl enable gpu-fault-fabric-manager-collector.service
if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
    systemctl enable gpu-fault-certificate-check.timer
else
    systemctl disable --now gpu-fault-certificate-check.timer \
        >/dev/null 2>&1 || true
fi
if [[ "${ENABLE_NODE_LOG_COLLECTOR}" == "true" ]]; then
    systemctl enable gpu-fault-log-collector.service
else
    systemctl disable --now gpu-fault-log-collector.service \
        >/dev/null 2>&1 || true
fi
if [[ "${DISABLE_KERNEL}" == "false" ]]; then
    systemctl enable gpu-fault-kernel-collector.service
else
    systemctl disable gpu-fault-kernel-collector.service >/dev/null 2>&1 || true
fi
if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
    systemctl enable gpu-fault-node-agent.service
else
    systemctl disable --now gpu-fault-node-agent.service \
        >/dev/null 2>&1 || true
fi

if [[ "${NO_START}" == "false" ]]; then
    if [[ -n "${CONTROL_PLANE_CA_CERTIFICATE}" ]]; then
        systemctl restart gpu-fault-certificate-check.timer
        systemctl start gpu-fault-certificate-check.service
    fi
    # Installation verification is an explicit synchronous health snapshot,
    # not an ordinary service restart. Bypass the deterministic startup
    # phase so the post-install check does not race the 15-second spread.
    install -d -m 0755 /var/lib/gpu-fault/health-snapshot
    date -u +%FT%TZ \
        > /var/lib/gpu-fault/health-snapshot/gpu.request
    date -u +%FT%TZ \
        > /var/lib/gpu-fault/health-snapshot/host.request
    systemctl restart gpu-fault-metrics-collector.service
    systemctl restart gpu-fault-host-collector.service
    systemctl restart gpu-fault-fabric-manager-collector.service
    if [[ "${ENABLE_NODE_LOG_COLLECTOR}" == "true" ]]; then
        systemctl restart gpu-fault-log-collector.service
    fi
    if [[ "${DISABLE_KERNEL}" == "false" ]]; then
        systemctl restart gpu-fault-kernel-collector.service
    fi
    if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then
        systemctl restart gpu-fault-node-agent.service
        NODE_AGENT_HEALTH_URL="${NODE_AGENT_ADVERTISE_URL%/}/healthz"
        NODE_AGENT_HEALTH_ARGS=(--fail --silent)
        if [[ -n "${NODE_AGENT_TLS_CERT}" ]]; then
            NODE_AGENT_HEALTH_ARGS+=(
                --cacert "${NODE_AGENT_TLS_CERT}"
            )
        fi
        for _ in {1..30}; do
            curl "${NODE_AGENT_HEALTH_ARGS[@]}" \
                "${NODE_AGENT_HEALTH_URL}" \
                >/dev/null 2>&1 && break
            sleep 1
        done
        curl "${NODE_AGENT_HEALTH_ARGS[@]}" \
            "${NODE_AGENT_HEALTH_URL}" \
            >/dev/null ||
            die "signed node action agent did not become ready"
    fi
    /opt/gpu-fault/verify
fi

printf 'Installed GPU fault collectors: node=%s metrics=%s kernel=%s node_log=%s agent=%s reset=%s quiesce=%s\n' \
    "${NODE_ID}" "${METRICS_MODE}" \
    "$([[ "${DISABLE_KERNEL}" == "false" ]] && printf enabled || printf disabled)" \
    "${ENABLE_NODE_LOG_COLLECTOR}" "${ENABLE_NODE_AGENT}" "${ALLOW_GPU_RESET}" \
    "${ALLOW_SERVICE_QUIESCE}"
