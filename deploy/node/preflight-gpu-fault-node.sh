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
NODE_WHEELHOUSE_SHA256="${GPU_FAULT_NODE_WHEELHOUSE_SHA256:-}"

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

# Matched in the shell, not with `printf ... | grep -q`: grep -q exits at its
# first match, and under pipefail the printf still writing the rest of the
# listing dies of SIGPIPE, so a member that is present was reported missing
# whenever grep won that race (always, once the listing outgrows a pipe buffer).
for required_member in \
    /deploy/node/install-gpu-fault-collector.sh \
    /deploy/node/runtime-slot.sh \
    /deploy/node/runtime_integrity.py \
    /deploy/node/preflight-gpu-fault-node.sh \
    /deploy/systemd/gpu-fault-node-agent.service \
    /deploy/systemd/gpu-fault-host-collector.service \
    /deploy/systemd/gpu-fault-kernel-collector.service \
    /deploy/systemd/gpu-fault-metrics-collector.service \
    /requirements/node-runtime.lock; do
    member_found="false"
    for bundle_member in "${bundle_members[@]}"; do
        if [[ "${bundle_member}" == *"${required_member}" ]]; then
            member_found="true"
            break
        fi
    done
    [[ "${member_found}" == "true" ]] ||
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

# Every missing host prerequisite is reported at once, as one verdict: the
# operator fixing a node image must not learn them one failed Job at a time
# (live 2026-10-04, plain EKS AL2023 NVIDIA AMI: no python3.12, no strace).
missing_prerequisites=()
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
        missing_prerequisites+=("${command}")
done
runtime_missing=${#missing_prerequisites[@]}

# The DCGM host engine of every GPU node, classified and reported as the
# "dcgm" member of the verdict. Host mode -- every HyperPod node (its AMI runs
# DCGM as a host service the exporter dials) and an eks-managed node group
# labelled gpu-fault.io/dcgm-hostengine=host -- needs a running nvidia-dcgm
# that sees the GPUs (dcgmi, nv-hostengine, systemctl is-active nvidia-dcgm,
# dcgmi discovery -l); eks host mode also refuses a CUDA flavor that does not
# match the driver and an nv-hostengine nvidia-dcgm.service does not own, which
# a HyperPod node only reports (WARN; the AMI is not ours to rebuild). The
# embedded mode needs no host engine at all, because the exporter starts its
# own and two engines on one node fight over the GPUs. A running engine's own
# version (dcgmi -v, "Hostengine build info") is reported as "version": NVIDIA
# requires the exporter's DCGM client library to be at least that version, and
# the release compares the two (gpu_fault.dcgm_versions).
dcgm_member=""
dcgm_remedy=""
dcgm_conflicts=()
dcgm_warnings=()
classify_dcgm_hostengine() {
    local provider="${GPU_FAULT_CLUSTER_PROVIDER:-hyperpod}"
    local mode="embedded" dcgmi="false" hostengine="false" active enabled
    local packages line major majors=() smi driver_major="" pids pid unit
    local processes="" unmanaged=() owned=() engine_runs="false" discovery="skipped"
    local status mismatch="false" state conflict="false" packages_json=""
    local installed="" version="" version_report
    local version_pattern='Hostengine build info:[[:space:]]*Version[[:space:]]*:[[:space:]]*([0-9]+[.][0-9]+[.][0-9]+)'
    if [[ "${provider}" != "eks-managed" ||
        "${GPU_FAULT_DCGM_HOSTENGINE:-}" == "host" ]]; then
        mode="host"
    fi
    if host_shell "command -v dcgmi >/dev/null"; then dcgmi="true"; fi
    if host_shell "command -v nv-hostengine >/dev/null"; then hostengine="true"; fi
    active="$(host_shell "systemctl is-active nvidia-dcgm 2>/dev/null || true")"
    active="${active%%$'\n'*}"
    active="${active//[^A-Za-z0-9._-]/}"
    enabled="$(host_shell "systemctl is-enabled nvidia-dcgm 2>/dev/null || true")"
    enabled="${enabled%%$'\n'*}"
    enabled="${enabled//[^A-Za-z0-9._-]/}"
    # The CUDA flavors of DCGM installed (datacenter-gpu-manager-4-cudaN).
    packages="$(host_shell "if command -v rpm >/dev/null; then rpm -qa --qf '%{NAME}\n' 'datacenter-gpu-manager-4-cuda*'; elif command -v dpkg-query >/dev/null; then dpkg-query -W -f='\${Package}\n' 'datacenter-gpu-manager-4-cuda*' 2>/dev/null; fi || true")"
    while IFS= read -r line; do
        if [[ "${line}" =~ ^datacenter-gpu-manager-4-cuda([0-9]+)$ ]]; then
            majors+=("${BASH_REMATCH[1]}")
        fi
    done <<<"${packages}"
    if (( ${#majors[@]} > 0 )); then
        mapfile -t majors < <(printf '%s\n' "${majors[@]}" | sort -nu)
    fi
    # The CUDA major the driver reports in the nvidia-smi header.
    smi="$(host_shell "nvidia-smi 2>/dev/null || true")"
    if [[ "${smi}" =~ CUDA\ Version:\ *([0-9]+) ]]; then
        driver_major="${BASH_REMATCH[1]}"
    fi
    # Every nv-hostengine, and whether nvidia-dcgm.service owns it.
    pids="$(host_shell "pgrep -x nv-hostengine 2>/dev/null || true")"
    while IFS= read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        unit=""
        if grep -qs 'nvidia-dcgm\.service' "${HOST_ROOT}/proc/${pid}/cgroup"; then
            unit="nvidia-dcgm.service"
            owned+=("${pid}")
        else
            unmanaged+=("${pid}")
        fi
        engine_runs="true"
        processes="${processes:+${processes},}{\"pid\":${pid},\"unit\":\"${unit}\"}"
    done <<<"${pids}"
    if [[ "${active}" == "active" ]]; then engine_runs="true"; fi
    if [[ "${dcgmi}" == "true" && "${engine_runs}" == "true" ]]; then
        status=0
        host_shell "timeout 30 dcgmi discovery -l >/dev/null 2>&1" || status=$?
        discovery="ok"
        if (( status != 0 )); then discovery="exit ${status}"; fi
        version_report="$(host_shell "timeout 10 dcgmi -v 2>/dev/null || true")"
        if [[ "${version_report}" =~ ${version_pattern} ]]; then
            version="${BASH_REMATCH[1]}"
        fi
    fi
    for major in "${majors[@]}"; do
        packages_json="${packages_json:+${packages_json},}\"${major}\""
        installed="${installed:+${installed},}cuda${major}"
    done
    if (( ${#majors[@]} > 0 )) && [[ -n "${driver_major}" &&
        " ${majors[*]} " != *" ${driver_major} "* ]]; then
        mismatch="true"
    fi
    if (( ${#unmanaged[@]} > 0 )); then
        state="unmanaged-hostengine"
    elif [[ "${enabled}" == "enabled" &&
        "${active}" =~ ^(activating|deactivating|failed)$ ]] ||
        [[ "${active}" == "active" && "${discovery}" != "ok" ]]; then
        state="host-failed"
    elif [[ "${mismatch}" == "true" ]]; then
        state="cuda-mismatch"
    elif [[ "${active}" == "active" ]]; then
        state="host-active"
    elif [[ "${dcgmi}" == "true" || "${hostengine}" == "true" ]]; then
        state="installed-inactive"
    else
        state="not-installed"
    fi
    if [[ "${mode}" == "host" ]]; then
        # Everything that keeps this node from host-active, with the missing
        # runtime tools in one verdict.
        if (( ${#unmanaged[@]} > 0 )); then conflict="true"; fi
        if [[ "${dcgmi}" != "true" ]]; then missing_prerequisites+=("dcgmi"); fi
        if [[ "${hostengine}" != "true" ]]; then
            missing_prerequisites+=("nv-hostengine")
        fi
        if [[ "${active}" != "active" ]]; then
            missing_prerequisites+=("nvidia-dcgm.service (${active:-unknown})")
        fi
        if [[ "${discovery}" == exit* ]]; then
            missing_prerequisites+=("dcgmi discovery -l (${discovery})")
        fi
        if [[ "${provider}" != "eks-managed" ]]; then
            dcgm_remedy="a HyperPod node runs DCGM as a host service the exporter dials at 127.0.0.1:5555: the instance group's AMI must ship datacenter-gpu-manager-4 and run nvidia-dcgm (systemctl enable --now nvidia-dcgm; dcgmi discovery -l lists the GPUs; see 管理员日常运维 / administrator-operations §5.0.1)"
            if [[ "${mismatch}" == "true" ]]; then
                dcgm_warnings+=("datacenter-gpu-manager-4-cuda${driver_major} (installed: ${installed})")
            fi
            for pid in "${unmanaged[@]}"; do
                dcgm_warnings+=("nv-hostengine pid ${pid} outside nvidia-dcgm.service")
            done
        else
            if [[ "${mismatch}" == "true" ]]; then
                missing_prerequisites+=("datacenter-gpu-manager-4-cuda${driver_major} (installed: ${installed})")
            fi
            for pid in "${unmanaged[@]}"; do
                missing_prerequisites+=("nv-hostengine pid ${pid} outside nvidia-dcgm.service")
            done
            dcgm_remedy="the managed node group declares gpu-fault.io/dcgm-hostengine=host: install the pinned datacenter-gpu-manager-4 packages and systemctl enable --now nvidia-dcgm in the launch template user data (see 管理员日常运维 / administrator-operations §5.0.1), or remove the label and deploy to run the exporter's embedded hostengine"
            if [[ "${mismatch}" == "true" ]]; then
                dcgm_remedy="${dcgm_remedy}; the DCGM CUDA flavor must match the driver's CUDA major: install datacenter-gpu-manager-4-cuda${driver_major} (nvidia-smi reports CUDA Version ${driver_major})"
            fi
            if (( ${#unmanaged[@]} > 0 )); then
                dcgm_remedy="${dcgm_remedy}; stop the hand-started nv-hostengine; nvidia-dcgm.service must own the host engine"
            fi
        fi
    else
        # A host engine next to the exporter's embedded one: running, about
        # to run, or starting at the next boot.
        if [[ "${active}" =~ ^(active|activating)$ || "${enabled}" == "enabled" ]]; then
            dcgm_conflicts+=("nvidia-dcgm.service (${active:-unknown}, ${enabled:-unknown})")
        fi
        for pid in "${owned[@]}"; do
            dcgm_conflicts+=("nv-hostengine pid ${pid}")
        done
        for pid in "${unmanaged[@]}"; do
            dcgm_conflicts+=("nv-hostengine pid ${pid} outside nvidia-dcgm.service")
        done
        if (( ${#dcgm_conflicts[@]} > 0 )); then conflict="true"; fi
    fi
    dcgm_member="$(printf '"dcgm":{"mode":"%s","state":"%s","service":{"active":"%s","enabled":"%s"},"dcgmi":%s,"nv_hostengine":%s,"cuda_packages":[%s],"driver_cuda_major":"%s","processes":[%s],"discovery":"%s","version":"%s","conflict":%s}' \
        "${mode}" "${state}" "${active}" "${enabled}" "${dcgmi}" "${hostengine}" \
        "${packages_json}" "${driver_major}" "${processes}" "${discovery}" \
        "${version}" "${conflict}")"
}
classify_dcgm_hostengine
for warning in "${dcgm_warnings[@]}"; do
    printf 'WARN  DCGM host engine: %s\n' "${warning}"
done
failure_reasons=()
failure_messages=()
if (( ${#missing_prerequisites[@]} > 0 )); then
    # Joined on a separator no item contains, so an item's own commas stay.
    missing_list="$(IFS=$'\x1f'; printf '%s' "${missing_prerequisites[*]}")"
    missing_list="${missing_list//$'\x1f'/, }"
    remedy=""
    if (( runtime_missing > 0 )); then
        remedy="install them on the node image before joining"
        if [[ "${GPU_FAULT_CLUSTER_PROVIDER:-hyperpod}" == "eks-managed" ]]; then
            remedy="the EKS AL2023 NVIDIA AMI does not ship them; install them in the managed node group launch template user data (dnf install -y python3.12 strace; see 管理员日常运维 / administrator-operations §5.0.1) and roll the node group, then rerun the join"
        fi
    fi
    if (( ${#missing_prerequisites[@]} > runtime_missing )); then
        remedy="${remedy:+${remedy}; }${dcgm_remedy}"
    fi
    failure_reasons+=("host prerequisites unavailable: ${missing_list}")
    failure_messages+=("host prerequisites unavailable: ${missing_list}; ${remedy}")
fi
if (( ${#dcgm_conflicts[@]} > 0 )); then
    conflict_list="$(IFS=$'\x1f'; printf '%s' "${dcgm_conflicts[*]}")"
    conflict_list="${conflict_list//$'\x1f'/, }"
    failure_reasons+=("DCGM host engine conflict: ${conflict_list}")
    failure_messages+=("DCGM host engine conflict: ${conflict_list}; a host nv-hostengine runs or is enabled next to the exporter's embedded hostengine: label the node group gpu-fault.io/dcgm-hostengine=host and deploy, or disable nvidia-dcgm (systemctl disable --now nvidia-dcgm; stop a hand-started nv-hostengine)")
fi
if (( ${#failure_reasons[@]} > 0 )); then
    reasons_json="$(printf '"%s",' "${failure_reasons[@]}")"
    printf '{"node_id":"%s","node_uid":"%s","status":"FAILED","reasons":[%s]%s}\n' \
        "${TARGET_NODE_NAME}" "${TARGET_NODE_UID}" "${reasons_json%,}" \
        "${dcgm_member:+,${dcgm_member}}"
    for ((index = 0; index < ${#failure_messages[@]} - 1; index++)); do
        printf 'ERROR: %s\n' "${failure_messages[index]}" >&2
    done
    die "${failure_messages[${#failure_messages[@]} - 1]}"
fi
host_shell \
    "/usr/bin/python3.12 -c 'import sys, venv; raise SystemExit(sys.version_info < (3, 12))'" ||
    die "host Python 3.12 venv support is unavailable"
if [[ -n "${NODE_WHEELHOUSE_SHA256}" ]]; then
    [[ "${NODE_WHEELHOUSE_SHA256}" =~ ^[0-9a-f]{64}$ ]] ||
        die "node wheelhouse inventory SHA-256 is invalid"
    integrity_member="${wheel_members[0]%/dist/*}/deploy/node/runtime_integrity.py"
    # The bundle has already been hash-verified; stream its validator without
    # extracting files onto the read-only host filesystem.
    tar -xOzf "${CANDIDATE_BUNDLE}" "${integrity_member}" |
        host_shell "/usr/bin/python3.12 -I -S -B - wheelhouse \
          --wheelhouse /run/gpu-fault-node-wheelhouse \
          --expected-inventory '${NODE_WHEELHOUSE_SHA256}' \
          --bundle /run/gpu-fault-preflight-artifact/$(basename "${CANDIDATE_BUNDLE}")" ||
        die "offline node wheelhouse integrity validation failed"
fi
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
    integrity_member="${wheel_members[0]%/dist/*}/deploy/node/runtime_integrity.py"
    tar -xOzf "${CANDIDATE_BUNDLE}" "${integrity_member}" |
        host_shell "/usr/bin/python3.12 -I -S -B - previous --root /opt/gpu-fault" ||
        die "rollback runtime integrity validation failed"
fi

printf '{"artifact_sha256":"%s","bundle_sha256":"%s",%s"node_id":"%s","node_uid":"%s","status":"PASSED"}\n' \
    "${EXPECTED_ARTIFACT_SHA256}" \
    "${EXPECTED_BUNDLE_SHA256}" \
    "${dcgm_member:+${dcgm_member},}" \
    "${TARGET_NODE_NAME}" \
    "${TARGET_NODE_UID}"
