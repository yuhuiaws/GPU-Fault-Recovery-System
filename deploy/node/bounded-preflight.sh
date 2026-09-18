#!/usr/bin/env bash
# Sourced by deployment entry points. Each callback retains the caller's errexit.

run_bounded_preflight() {
    local phase="$1" limit="$2" output_dir="$3" callback="$4"
    shift 4
    if [[ ! "${limit}" =~ ^[1-8]$ || ! "${phase}" =~ ^(render|host)$ || $# == 0 ]]; then
        printf 'ERROR: invalid node preflight scheduling arguments\n' >&2
        return 2
    fi
    local -a nodes=("$@") pids=()
    local next=0 active=0 failed=0 completed status index completion_fd
    local fifo="${output_dir}/completion-${phase}"
    mkfifo -m 0600 "${fifo}"
    exec {completion_fd}<>"${fifo}"
    while (( next < ${#nodes[@]} || active > 0 )); do
        completed=""
        for index in "${!pids[@]}"; do
            if ! kill -0 "${pids[index]}" 2>/dev/null; then
                completed="${index}"
                break
            fi
        done
        if [[ -z "${completed}" ]]; then
            while (( failed == 0 && active < limit && next < ${#nodes[@]} )); do
                index="${next}"
                (
                    trap 'status=$?; printf "%s %s\n" "${index}" "${status}" >&"${completion_fd}"' EXIT
                    "${callback}" "${phase}" "${nodes[index]}"
                ) >"${output_dir}/${nodes[index]}.log" 2>&1 &
                pids[index]="$!"
                next=$((next + 1))
                active=$((active + 1))
            done
            (( active > 0 )) || break
            # SIGKILL cannot run the EXIT trap. Periodic liveness checks must
            # also reap such workers instead of waiting forever on the FIFO.
            read -r -t 1 -u "${completion_fd}" completed status || continue
        fi
        # A normally exited child can be reaped before its queued notice is read.
        [[ -n "${pids[completed]:-}" ]] || continue
        if wait "${pids[completed]}"; then
            status=0
        else
            status="$?"
            printf 'ERROR: node preflight %s failed on %s (exit %s)\n' \
                "${phase}" "${nodes[completed]}" "${status}" >&2
            failed=1
        fi
        unset 'pids[completed]'
        active=$((active - 1))
    done
    exec {completion_fd}>&-
    rm "${fifo}"
    # Keep diagnostics deterministic even when workers finish out of order.
    for (( index=0; index<next; index++ )); do
        cat "${output_dir}/${nodes[index]}.log"
    done
    return "${failed}"
}
