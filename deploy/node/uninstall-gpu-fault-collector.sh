#!/usr/bin/env bash
set -euo pipefail

[[ "${EUID}" -eq 0 ]] || {
    printf 'ERROR: run as root\n' >&2
    exit 1
}

installed_units=()
if [[ -s /opt/gpu-fault/installed-units.txt ]]; then
    mapfile -t installed_units < /opt/gpu-fault/installed-units.txt
else
    shopt -s nullglob
    unit_paths=(
        /etc/systemd/system/gpu-fault-*.service
        /etc/systemd/system/gpu-fault-*.timer
    )
    for unit_path in "${unit_paths[@]}"; do
        installed_units+=("${unit_path##*/}")
    done
fi

for service in "${installed_units[@]}"; do
    [[ "${service}" == gpu-fault-*.service ||
        "${service}" == gpu-fault-*.timer ]] || {
        printf 'ERROR: invalid installed unit name: %s\n' "${service}" >&2
        exit 1
    }
    systemctl disable --now "${service}" >/dev/null 2>&1 || true
done

shopt -s nullglob
quiesce_states=(/var/lib/gpu-fault/quiesce/quiesce-*.json)
if (( ${#quiesce_states[@]} > 0 )); then
    restore_command=/opt/gpu-fault/venv/bin/gpu-fault-restore-gpu-services
    [[ -x "${restore_command}" ]] || {
        printf 'ERROR: quiesce state exists but restore command is unavailable\n' >&2
        exit 1
    }
    for state_file in "${quiesce_states[@]}"; do
        "${restore_command}" --state-file "${state_file}" || {
            printf 'ERROR: GPU services could not be restored; uninstall aborted\n' >&2
            exit 1
        }
    done
fi

for service in "${installed_units[@]}"; do
    [[ "${service}" == gpu-fault-*.service ||
        "${service}" == gpu-fault-*.timer ]] || {
        printf 'ERROR: invalid installed unit name: %s\n' "${service}" >&2
        exit 1
    }
    systemctl disable --now "${service}" >/dev/null 2>&1 || true
    rm -f "/etc/systemd/system/${service}"
done
systemctl daemon-reload
rm -rf /opt/gpu-fault
rm -rf /etc/gpu-fault
rm -rf /var/lib/gpu-fault
printf 'GPU fault node collectors removed\n'
