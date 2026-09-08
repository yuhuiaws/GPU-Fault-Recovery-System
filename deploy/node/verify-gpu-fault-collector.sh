#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${GPU_FAULT_ENV_FILE:-/etc/gpu-fault/collector.env}"
NODE_AGENT_ENV_FILE="/etc/gpu-fault/node-agent.env"
VERIFY_STARTED_EPOCH="$(date +%s)"
[[ -r "${ENV_FILE}" ]] || {
    printf 'ERROR: cannot read %s\n' "${ENV_FILE}" >&2
    exit 1
}

mapfile -d '' -t config_values < <(
    python3 -c '
import shlex
import sys

wanted = {
    "GPU_FAULT_CONTROL_PLANE_URL": "",
    "GPU_FAULT_CONTROL_PLANE_TOKEN": "",
    "GPU_FAULT_CLUSTER_ID": "",
    "NODE_NAME": "",
    "GPU_FAULT_DCGM_METRICS_URL": "",
    "GPU_FAULT_METRICS_MODE": "",
    "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR": "false",
    "GPU_FAULT_NODE_AGENT_PORT": "",
    "GPU_FAULT_NODE_ADVERTISE_URL": "",
    "GPU_FAULT_NODE_AGENT_TLS_CERT": "",
    "SSL_CERT_FILE": "",
    "GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS": "2592000",
}
for path in sys.argv[1:]:
  if not __import__("os").path.exists(path):
    continue
  with open(path, encoding="utf-8") as stream:
    for raw in stream:
        parts = shlex.split(raw, comments=True, posix=True)
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        if key in wanted:
            wanted[key] = value
for key in wanted:
    sys.stdout.buffer.write(wanted[key].encode() + b"\0")
' "${ENV_FILE}" "${NODE_AGENT_ENV_FILE}"
)
GPU_FAULT_CONTROL_PLANE_URL="${config_values[0]:-}"
GPU_FAULT_CONTROL_PLANE_TOKEN="${config_values[1]:-}"
GPU_FAULT_CLUSTER_ID="${config_values[2]:-}"
NODE_NAME="${config_values[3]:-}"
GPU_FAULT_DCGM_METRICS_URL="${config_values[4]:-}"
GPU_FAULT_METRICS_MODE="${config_values[5]:-}"
GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR="${config_values[6]:-false}"
GPU_FAULT_NODE_AGENT_PORT="${config_values[7]:-}"
GPU_FAULT_NODE_ADVERTISE_URL="${config_values[8]:-}"
GPU_FAULT_NODE_AGENT_TLS_CERT="${config_values[9]:-}"
SSL_CERT_FILE="${config_values[10]:-}"
GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS="${config_values[11]:-2592000}"
[[ -n "${GPU_FAULT_CONTROL_PLANE_URL}" ]] || {
    printf 'ERROR: control plane URL is missing from %s\n' "${ENV_FILE}" >&2
    exit 1
}

failed=0
passes=0
warns=0
fails=0

# 责任分界（owner decision 6）：GPU 健康只上报，软件完整性才拦截。
# A node whose GPU fell off the bus is exactly the node the Node Agent has to
# reach, so a GPU-health gate only warns and the install stands, while every
# software-integrity gate (units, readiness, wheel digest, delivery) still
# fails the run and lets the installer roll back.
ok() {
    printf 'PASS  %s\n' "$1"
    passes=$(( passes + 1 ))
}
warn() {
    if [[ -n "${2:-}" ]]; then
        printf 'WARN  %s (%s)\n' "$1" "$2"
    else
        printf 'WARN  %s\n' "$1"
    fi
    warns=$(( warns + 1 ))
}
fail() {
    if [[ -n "${2:-}" ]]; then
        printf 'FAIL  %s (%s)\n' "$1" "$2"
    else
        printf 'FAIL  %s\n' "$1"
    fi
    fails=$(( fails + 1 ))
    failed=1
}
skip() {
    printf 'SKIP  %s (%s)\n' "$1" "$2"
}
check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        ok "${name}"
    else
        fail "${name}"
    fi
}

# `nvidia-smi -L` exits non-zero as soon as a single GPU is unreadable, even
# though it still lists the healthy ones. Zero enumerated GPUs is a host that
# cannot run the product at all; a partial list is the degraded host the
# collectors and the Agent must still cover.
gpu_enumeration_failed="false"
gpu_enumeration="$(nvidia-smi -L 2>/dev/null)" || gpu_enumeration_failed="true"
enumerated_gpus="$(
    printf '%s\n' "${gpu_enumeration}" | grep -c '^GPU [0-9]' || true
)"
if (( enumerated_gpus == 0 )); then
    fail "NVIDIA GPU enumeration" "no GPU was enumerated"
elif [[ "${gpu_enumeration_failed}" == "true" ]]; then
    warn "NVIDIA GPU enumeration" \
        "only ${enumerated_gpus} GPU(s) are enumerable"
else
    ok "NVIDIA GPU enumeration"
fi
CURL_TLS=()
if [[ -n "${SSL_CERT_FILE}" ]]; then
    [[ "${SSL_CERT_FILE}" == /* && -r "${SSL_CERT_FILE}" ]] || {
        printf 'ERROR: configured SSL_CERT_FILE is not readable\n' >&2
        exit 1
    }
    check "control plane CA certificate validity" \
        /opt/gpu-fault/verify-certificate-bundle \
        "${SSL_CERT_FILE}" \
        "${GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS}"
    check "control plane leaf certificate validity" \
        /opt/gpu-fault/check-control-plane-certificate \
        "${GPU_FAULT_CONTROL_PLANE_URL}" \
        "${SSL_CERT_FILE}" \
        "${GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS}"
    CURL_TLS=(--cacert "${SSL_CERT_FILE}")
fi

check "control plane health" curl --fail --silent --show-error \
    "${CURL_TLS[@]}" \
    "${GPU_FAULT_CONTROL_PLANE_URL%/}/healthz"
check "metrics collector service" systemctl is-active --quiet \
    gpu-fault-metrics-collector.service
check "host telemetry collector service" systemctl is-active --quiet \
    gpu-fault-host-collector.service
if [[ "${GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR}" == "true" ]]; then
    check "system/training log collector service" systemctl is-active --quiet \
        gpu-fault-log-collector.service
else
    if systemctl is-enabled --quiet gpu-fault-log-collector.service \
        >/dev/null 2>&1 ||
        systemctl is-active --quiet gpu-fault-log-collector.service \
        >/dev/null 2>&1; then
        fail "system/training log collector service" "must be disabled"
    else
        skip "system/training log collector service" \
            "disabled by deployment policy"
    fi
fi
check "Fabric Manager SXID collector service" systemctl is-active --quiet \
    gpu-fault-fabric-manager-collector.service
check "GPU persistence service" systemctl is-active --quiet \
    gpu-fault-gpu-persistence.service
if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader |
    grep -Fvx "Enabled" >/dev/null; then
    warn "GPU persistence mode" "not enabled on every GPU"
else
    ok "GPU persistence mode"
fi

if systemctl is-enabled --quiet gpu-fault-kernel-collector.service \
    >/dev/null 2>&1; then
    check "kernel XID/SXID collector service" systemctl is-active --quiet \
        gpu-fault-kernel-collector.service
else
    skip "kernel XID/SXID collector service" "disabled"
fi

if systemctl is-enabled --quiet gpu-fault-node-agent.service \
    >/dev/null 2>&1; then
    check "signed node action agent service" systemctl is-active --quiet \
        gpu-fault-node-agent.service
    node_agent_port="${GPU_FAULT_NODE_AGENT_PORT:-9099}"
    node_agent_url="${GPU_FAULT_NODE_ADVERTISE_URL:-http://127.0.0.1:${node_agent_port}}"
    node_agent_curl=(--fail --silent)
    if [[ -n "${GPU_FAULT_NODE_AGENT_TLS_CERT}" ]]; then
        node_agent_curl+=(
            --cacert "${GPU_FAULT_NODE_AGENT_TLS_CERT}"
        )
    fi
    check "signed node action agent health" \
        curl "${node_agent_curl[@]}" \
        "${node_agent_url%/}/healthz"
else
    skip "signed node action agent service" "disabled"
fi

if [[ "${GPU_FAULT_METRICS_MODE}" == "dcgm" ]]; then
    if [[ -z "${GPU_FAULT_DCGM_METRICS_URL}" ]]; then
        # A missing URL is a configuration defect, not GPU health.
        fail "DCGM metrics URL" "not configured"
    else
        dcgm_output="$(mktemp)"
        if curl --fail --silent --show-error --output "${dcgm_output}" \
            "${GPU_FAULT_DCGM_METRICS_URL}" &&
            grep -q "DCGM_FI_DEV_" "${dcgm_output}"; then
            ok "DCGM exporter supported metrics"
        else
            warn "DCGM exporter supported metrics" \
                "the exporter published no DCGM_FI_DEV_ metric"
        fi
        rm -f "${dcgm_output}"
    fi
else
    check "nvidia-smi metrics query" nvidia-smi \
        --query-gpu=index,uuid,temperature.gpu,power.draw \
        --format=csv,noheader,nounits
fi

encode_path() {
    python3 -c \
        'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' \
        "$1"
}

LATEST_URL="${GPU_FAULT_CONTROL_PLANE_URL%/}/v1/gpu-metrics/$(encode_path "${GPU_FAULT_CLUSTER_ID}")/$(encode_path "${NODE_NAME}")/latest"
CURL_AUTH=()
if [[ -n "${GPU_FAULT_CONTROL_PLANE_TOKEN}" ]]; then
    # The cluster-ID header is what tells the regional control plane to
    # authorize this read with the per-cluster token instead of the
    # operator execution token, which a node never holds. Sending it
    # unconditionally is harmless in single-cluster mode.
    CURL_AUTH=(
        -H "Authorization: Bearer ${GPU_FAULT_CONTROL_PLANE_TOKEN}"
        -H "X-GPU-Fault-Cluster-ID: ${GPU_FAULT_CLUSTER_ID}"
    )
fi
latest_output="$(mktemp)"
delivered="false"
# A freshly restarted collector can publish before the regional ingress
# replica serving this read has observed the new sample. Allow one full
# minute for the first post-install metric to become visible.
for _ in {1..60}; do
    if curl --fail --silent --show-error "${CURL_TLS[@]}" \
        "${CURL_AUTH[@]}" \
        --output "${latest_output}" "${LATEST_URL}" &&
        python3 -c \
            '
import datetime
import json
import sys

items = json.load(open(sys.argv[1]))
cutoff = datetime.datetime.fromtimestamp(
    float(sys.argv[2]) - 120,
    tz=datetime.timezone.utc,
)
recent = any(
    datetime.datetime.fromisoformat(
        item["observed_at"].replace("Z", "+00:00")
    ) >= cutoff
    for item in items
)
raise SystemExit(not recent)
' "${latest_output}" "${VERIFY_STARTED_EPOCH}"; then
        delivered="true"
        break
    fi
    sleep 1
done
rm -f "${latest_output}"
if [[ "${delivered}" == "true" ]]; then
    ok "metrics delivered to control plane"
else
    fail "metrics delivered to control plane"
fi

printf 'SUMMARY pass=%d warn=%d fail=%d\n' "${passes}" "${warns}" "${fails}"
exit "${failed}"
