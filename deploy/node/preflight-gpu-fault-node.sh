#!/usr/bin/env bash
set -euo pipefail

HOST_ROOT="${GPU_FAULT_PREFLIGHT_HOST_ROOT:-/host}"
CANDIDATE_BUNDLE="${GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE:-}"
EXPECTED_BUNDLE_SHA256="${GPU_FAULT_PREFLIGHT_BUNDLE_SHA256:-}"
EXPECTED_ARTIFACT_SHA256="${GPU_FAULT_PREFLIGHT_ARTIFACT_SHA256:-}"
TARGET_NODE_NAME="${TARGET_NODE_NAME:-}"
TARGET_NODE_UID="${TARGET_NODE_UID:-}"
REQUIRE_ROLLBACK_SLOT="${GPU_FAULT_REQUIRE_ROLLBACK_SLOT:-false}"
MIN_FREE_BYTES="${GPU_FAULT_INSTALLER_MIN_FREE_BYTES:-2147483648}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for command in awk chroot df grep readlink sha256sum tar; do
    command -v "${command}" >/dev/null ||
        die "${command} is required for node preflight"
done
[[ -d "${HOST_ROOT}" ]] || die "host root is unavailable"
[[ -r "${CANDIDATE_BUNDLE}" ]] || die "candidate installer bundle is unreadable"
[[ "${EXPECTED_BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
    die "candidate bundle SHA-256 is invalid"
[[ "${EXPECTED_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
    die "candidate node-runtime SHA-256 is invalid"
[[ "${TARGET_NODE_NAME}" =~ ^[A-Za-z0-9.-]+$ ]] ||
    die "target node name is invalid"
[[ -n "${TARGET_NODE_UID}" && "${TARGET_NODE_UID}" != *[[:space:]]* ]] ||
    die "target node UID is invalid"
[[ "${REQUIRE_ROLLBACK_SLOT}" =~ ^(true|false)$ ]] ||
    die "GPU_FAULT_REQUIRE_ROLLBACK_SLOT must be true or false"
[[ "${MIN_FREE_BYTES}" =~ ^[1-9][0-9]*$ ]] ||
    die "GPU_FAULT_INSTALLER_MIN_FREE_BYTES must be positive"

observed_bundle_sha256="$(sha256sum "${CANDIDATE_BUNDLE}" | awk '{print $1}')"
[[ "${observed_bundle_sha256}" == "${EXPECTED_BUNDLE_SHA256}" ]] ||
    die "candidate installer bundle SHA-256 mismatch"

mapfile -t bundle_members < <(tar -tzf "${CANDIDATE_BUNDLE}")
mapfile -t wheel_members < <(
    printf '%s\n' "${bundle_members[@]}" |
        awk '/\/dist\/gpu_fault_node_runtime-[^/]+\.whl$/'
)
(( ${#wheel_members[@]} == 1 )) ||
    die "candidate installer bundle must contain exactly one node-runtime wheel"
observed_artifact_sha256="$(
    tar -xOzf "${CANDIDATE_BUNDLE}" "${wheel_members[0]}" |
        sha256sum |
        awk '{print $1}'
)"
[[ "${observed_artifact_sha256}" == "${EXPECTED_ARTIFACT_SHA256}" ]] ||
    die "candidate node-runtime wheel SHA-256 mismatch"

for required_member in \
    /deploy/node/install-gpu-fault-collector.sh \
    /deploy/node/preflight-gpu-fault-node.sh \
    /deploy/systemd/gpu-fault-node-agent.service \
    /deploy/systemd/gpu-fault-host-collector.service \
    /deploy/systemd/gpu-fault-kernel-collector.service \
    /deploy/systemd/gpu-fault-metrics-collector.service \
    /requirements/node-runtime.lock; do
    printf '%s\n' "${bundle_members[@]}" |
        grep -q "${required_member}$" ||
        die "candidate installer bundle is missing ${required_member}"
done

available_kib="$(df -Pk "${HOST_ROOT}/opt" | awk 'NR == 2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] ||
    die "cannot determine host free space"
(( available_kib * 1024 >= MIN_FREE_BYTES )) ||
    die "host has less than ${MIN_FREE_BYTES} bytes free for node runtime"

host_shell() {
    chroot "${HOST_ROOT}" /bin/bash -ceu "$1"
}

for command in \
    /usr/bin/python3.12 \
    curl \
    nvidia-smi \
    ps \
    ss \
    strace \
    systemctl \
    timeout; do
    host_shell "command -v '${command}' >/dev/null" ||
        die "host prerequisite is unavailable: ${command}"
done
host_shell \
    "/usr/bin/python3.12 -c 'import sys, venv; raise SystemExit(sys.version_info < (3, 12))'" ||
    die "host Python 3.12 venv support is unavailable"
systemd_state="$(host_shell "systemctl is-system-running 2>/dev/null || true")"
[[ "${systemd_state}" == "running" || "${systemd_state}" == "degraded" ]] ||
    die "host systemd is not operational: ${systemd_state:-unknown}"
# `nvidia-smi -L` exits non-zero as soon as one GPU is unreadable while it
# still lists the healthy ones, and a node with one GPU off the bus is exactly
# the node that needs the Agent. Only an empty enumeration blocks the rollout.
# `host_shell` runs `bash -ceu`, so the fallback has to live inside the quoted
# command.
gpu_enumeration="$(
    host_shell "nvidia-smi -L 2>/dev/null || printf 'ENUMERATION_FAILED\n'"
)"
enumerated_gpus="$(
    printf '%s\n' "${gpu_enumeration}" | grep -c '^GPU [0-9]' || true
)"
(( enumerated_gpus > 0 )) || die "NVIDIA driver enumerated zero GPUs"
if [[ "${gpu_enumeration}" == *ENUMERATION_FAILED* ]]; then
    printf 'WARN  NVIDIA GPU enumeration (only %s GPU(s) are enumerable)\n' \
        "${enumerated_gpus}"
fi
[[ -e "${HOST_ROOT}/dev/kmsg" ]] || die "host /dev/kmsg is unavailable"

if compgen -G "${HOST_ROOT}/sys/class/infiniband/*" >/dev/null; then
    for command in ethtool rdma; do
        host_shell "command -v '${command}' >/dev/null" ||
            die "host EFA prerequisite is unavailable: ${command}"
    done
fi

if compgen -G \
    "${HOST_ROOT}/var/lib/gpu-fault/quiesce/quiesce-*.json" >/dev/null; then
    die "host has an unresolved GPU service quiesce state"
fi

if [[ "${REQUIRE_ROLLBACK_SLOT}" == "true" ]]; then
    rollback_slot=""
    current_link="${HOST_ROOT}/opt/gpu-fault/current"
    if [[ -L "${current_link}" ]]; then
        current_target="$(readlink "${current_link}")"
        if [[ "${current_target}" == /* ]]; then
            rollback_slot="${HOST_ROOT}${current_target}"
        else
            rollback_slot="$(
                readlink -f "$(dirname "${current_link}")/${current_target}"
            )"
        fi
    elif [[ -d "${HOST_ROOT}/opt/gpu-fault/venv" ]]; then
        rollback_slot="${HOST_ROOT}/opt/gpu-fault"
    fi
    [[ -n "${rollback_slot}" ]] ||
        die "host has no rollback runtime slot"
    [[ -x "${rollback_slot}/venv/bin/gpu-fault-node-agent" ]] ||
        die "rollback runtime slot has no Node Agent"
    [[ -x "${rollback_slot}/venv/bin/gpu-fault-restore-gpu-services" ]] ||
        die "rollback runtime slot has no GPU service restore command"
    [[ -s "${HOST_ROOT}/opt/gpu-fault/installed-units.txt" ]] ||
        die "rollback runtime unit inventory is unavailable"
fi

printf '{"artifact_sha256":"%s","bundle_sha256":"%s","node_id":"%s","node_uid":"%s","status":"PASSED"}\n' \
    "${EXPECTED_ARTIFACT_SHA256}" \
    "${EXPECTED_BUNDLE_SHA256}" \
    "${TARGET_NODE_NAME}" \
    "${TARGET_NODE_UID}"
