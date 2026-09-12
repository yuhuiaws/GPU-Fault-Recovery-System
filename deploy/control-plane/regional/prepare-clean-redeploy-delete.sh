#!/usr/bin/env bash
# Kubernetes deletion helpers sourced by prepare-clean-redeploy.sh. They use
# its cpu_kubectl/gpu_kubectl/die/log functions and its NAMESPACE and
# TIMEOUT_SECONDS settings, so they run only inside that script.
#
# Every delete is issued with --wait=false and the set is then polled once
# until it is gone: deleting one object after another with kubectl's own wait
# cost ~50 s for the application objects and ~49 s for the two namespaces on
# the live uninstall of 2026-09-12. Wave ordering is kept: a later wave starts
# only after the earlier one has disappeared.

# kubectl on one plane: an empty context is the CPU plane.
plane_kubectl() {
    local context=$1
    shift
    if [[ -z "${context}" ]]; then
        cpu_kubectl "$@"
    else
        gpu_kubectl "${context}" "$@"
    fi
}

# Whether kind/name is gone. An API error is fatal, as kubectl's own wait
# would have been: it must not read as "deleted".
kubernetes_object_absent() {
    local context=$1
    local kind=$2
    local name=$3
    local scope=$4
    local arguments=(get "${kind}" "${name}" --ignore-not-found -o name)
    local output
    if [[ "${scope}" != cluster ]]; then
        arguments=(-n "${NAMESPACE}" "${arguments[@]}")
    fi
    output="$(plane_kubectl "${context}" "${arguments[@]}")" ||
        die "could not verify the deletion of ${kind}/${name} (${context:-cpu})"
    [[ -z "${output}" ]]
}

# Poll until every record (kind, name, scope, context; tab-separated, the
# context last so the CPU plane's empty context survives IFS splitting) is
# gone, under one TIMEOUT_SECONDS deadline for the whole set.
wait_for_kubernetes_absence() {
    local description=$1
    shift
    (($# > 0)) || return 0
    local remaining=("$@")
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local entry
    local kind
    local name
    local scope
    local context
    local still
    while :; do
        still=()
        for entry in "${remaining[@]}"; do
            IFS=$'\t' read -r kind name scope context <<<"${entry}"
            if ! kubernetes_object_absent \
                "${context}" "${kind}" "${name}" "${scope}"; then
                still+=("${entry}")
            fi
        done
        if ((${#still[@]} == 0)); then
            return 0
        fi
        remaining=("${still[@]}")
        ((SECONDS < deadline)) ||
            die "${description}: ${#remaining[@]} object(s) remain after ${TIMEOUT_SECONDS}s"
        sleep 2
    done
}

# One wave on one plane: issue every delete, then wait for the wave once.
# Records are kind, name, scope (tab-separated).
delete_kubernetes_wave() {
    local context=$1
    shift
    (($# > 0)) || return 0
    local entry
    local kind
    local name
    local scope
    local pending=()
    for entry in "$@"; do
        IFS=$'\t' read -r kind name scope <<<"${entry}"
        if [[ "${scope}" == cluster ]]; then
            plane_kubectl "${context}" delete "${kind}" "${name}" \
                --ignore-not-found --wait=false
        else
            plane_kubectl "${context}" -n "${NAMESPACE}" \
                delete "${kind}" "${name}" --ignore-not-found --wait=false
        fi
        pending+=("${kind}"$'\t'"${name}"$'\t'"${scope}"$'\t'"${context}")
    done
    wait_for_kubernetes_absence \
        "application objects (${context:-cpu})" "${pending[@]}"
}

# Delete a plane's registered application objects wave by wave. The records
# (kind, name, scope, phase) come in inventory order, grouped by phase; each
# run of one phase is a wave, and the next wave starts once it is gone.
clean_plane_objects() {
    local context=$1
    local -n plane_records=$2
    ((${#plane_records[@]} > 0)) || return 0
    local entry
    local kind
    local name
    local scope
    local phase
    local current_phase=
    local wave=()
    for entry in "${plane_records[@]}"; do
        IFS=$'\t' read -r kind name scope phase <<<"${entry}"
        if [[ -n "${current_phase}" && "${phase}" != "${current_phase}" ]]; then
            delete_kubernetes_wave "${context}" "${wave[@]}"
            wave=()
        fi
        current_phase=${phase}
        wave+=("${kind}"$'\t'"${name}"$'\t'"${scope}")
    done
    delete_kubernetes_wave "${context}" "${wave[@]}"
}

# Issue the namespace delete on every GPU plane and the CPU plane first, then
# wait for all of them with one poll under one deadline.
delete_solution_namespaces() {
    local context
    local pending=()
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        gpu_kubectl "${context}" delete namespace "${NAMESPACE}" \
            --ignore-not-found --wait=false
        pending+=(namespace$'\t'"${NAMESPACE}"$'\t'cluster$'\t'"${context}")
    done
    cpu_kubectl delete namespace "${NAMESPACE}" \
        --ignore-not-found --wait=false
    pending+=(namespace$'\t'"${NAMESPACE}"$'\t'cluster$'\t')
    wait_for_kubernetes_absence "solution namespaces" "${pending[@]}"
}
