#!/usr/bin/env bash
# The regional release's schema stage: the same three one-shot Jobs, in the
# same order, as deploy/hyperpod/deploy.sh (F-J3 three-step index method).
#
#   1. gpu-fault-postgres-index-build     CREATE INDEX CONCURRENTLY, online,
#                                          for every declared index that is
#                                          missing or invalid
#   2. gpu-fault-postgres-schema-ensure    idempotent DDL + migration registry;
#                                          finds the indexes present and skips
#   3. gpu-fault-postgres-schema-preflight read-only gate: version, indexes,
#                                          in-flight safety workflows
#
# Running the ensure Job alone would build a missing index inside the DDL
# transaction, i.e. a plain CREATE INDEX holding a write lock on the hot
# tables. Any Job failing stops the release before a Deployment rolls.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
KUBECONFIG="${GPU_FAULT_CONTROL_PLANE_KUBECONFIG:?required}"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
WHEEL_CONFIGMAP="${GPU_FAULT_WHEEL_CONFIGMAP:?required}"
DEFAULT_RUNTIME_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"
RUNTIME_IMAGE="${GPU_FAULT_RUNTIME_IMAGE:?required}"

[[ "${RUNTIME_IMAGE}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_RUNTIME_IMAGE\n' >&2
    exit 2
}

timeout_seconds() {
    # "60m" / "90s" / "300" -> seconds.
    local value="$1"
    case "${value}" in
        *m) printf '%d' "$(( ${value%m} * 60 ))" ;;
        *s) printf '%d' "${value%s}" ;;
        *) printf '%d' "${value}" ;;
    esac
}

wait_for_postgres_job() {
    # `kubectl wait --for=condition=complete` sits out the whole timeout on a
    # Job that has already failed (live 2026-09-12: an hour on a two-pod
    # backoff). Poll both terminal conditions instead.
    local name="$1" timeout="$2" deadline conditions
    deadline=$(( SECONDS + $(timeout_seconds "${timeout}") ))
    while :; do
        conditions="$(kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
          get "job/${name}" \
          -o 'jsonpath={range .status.conditions[*]}{.type}={.status} {end}' \
          2>/dev/null || true)"
        case " ${conditions}" in
            *" Complete=True"*) return 0 ;;
            *" Failed=True"*)
                printf 'ERROR: postgres job %s failed\n' "${name}" >&2
                return 1 ;;
        esac
        if (( SECONDS >= deadline )); then
            printf 'ERROR: postgres job %s did not complete within %s\n' \
              "${name}" "${timeout}" >&2
            return 1
        fi
        sleep "${GPU_FAULT_JOB_POLL_SECONDS:-5}"
    done
}

run_postgres_job() {
    # Delete first: a completed Job's pod template is immutable and a new
    # release must run the new wheel. Wait, then print the log; a failure
    # prints the log and stops the release.
    local name="$1" manifest="$2" timeout="$3"
    kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
      delete "job/${name}" --ignore-not-found
    sed \
      -e "s/REPLACE_WITH_WHEEL_CONFIGMAP/${WHEEL_CONFIGMAP}/g" \
      -e "s/namespace: gpu-fault-system/namespace: ${NAMESPACE}/g" \
      -e "s#${DEFAULT_RUNTIME_IMAGE}#${RUNTIME_IMAGE}#g" \
      "${ROOT}/${manifest}" |
      kubectl --kubeconfig "${KUBECONFIG}" apply -f -
    if ! wait_for_postgres_job "${name}" "${timeout}"; then
        printf 'ERROR: postgres job %s did not complete; its log follows\n' "${name}" >&2
        kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
          logs "job/${name}" --all-containers=true >&2 || true
        return 1
    fi
    kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
      logs "job/${name}" --all-containers=true || true
}

run_postgres_job gpu-fault-postgres-index-build \
    deploy/migrations/postgres-index-build-job.yaml 60m
run_postgres_job gpu-fault-postgres-schema-ensure \
    deploy/migrations/postgres-schema-ensure-job.yaml 12m
run_postgres_job gpu-fault-postgres-schema-preflight \
    deploy/migrations/postgres-schema-preflight-job.yaml 5m
