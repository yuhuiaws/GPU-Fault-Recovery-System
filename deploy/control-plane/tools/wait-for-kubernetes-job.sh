#!/usr/bin/env bash
set -euo pipefail
umask 077

TIMEOUT_SECONDS="${1:?timeout seconds required}"
JOB="${2:?Job name required}"
shift 2
[[ "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ && "${JOB}" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || {
    printf 'ERROR: invalid Job wait arguments\n' >&2
    exit 2
}
(( $# > 0 )) || { printf 'ERROR: scoped kubectl command required\n' >&2; exit 2; }
kubectl_command=("$@")
deadline=$((SECONDS + TIMEOUT_SECONDS))
expected_uid=""
document=""
error_file="$(mktemp)"
trap 'rm -f "${error_file}"' EXIT

read_api_json() {
    local remaining budget output code error
    while (( SECONDS < deadline )); do
        remaining=$((deadline - SECONDS))
        budget=$((remaining < 15 ? remaining : 15))
        if output="$(timeout --foreground "${budget}s" "${kubectl_command[@]}" \
            "$@" -o json "--request-timeout=${budget}s" 2>"${error_file}")"; then
            printf '%s\n' "${output}"
            return 0
        else
            code="$?"
        fi
        error="$(<"${error_file}")"
        # Authentication, authorization, absence and TLS trust failures are not
        # network transients. Never reinterpret them as a pending Job.
        if [[ "${error}" =~ Forbidden|Unauthorized|NotFound|BadRequest|x509: ]]; then
            printf 'ERROR: cannot read Job %s: non-retryable API failure\n' "${JOB}" >&2
            return 1
        fi
        if [[ "${code}" != 124 &&
            ! "${error}" =~ TooManyRequests|ServiceUnavailable|InternalError|InternalServerError|Timeout|timeout|timed\ out|connection\ reset|connection\ refused|unexpected\ EOF|TLS\ handshake|HTTP\ (429|500|502|503|504) ]]; then
            printf 'ERROR: cannot read Job %s: unknown API failure\n' "${JOB}" >&2
            return 1
        fi
        (( SECONDS < deadline )) && sleep 1
    done
    return 124
}

while (( SECONDS < deadline )) || [[ -n "${document}" ]]; do
    if [[ -z "${document}" ]]; then
        if document="$(read_api_json get "job/${JOB}")"; then
            :
        else
            code="$?"
            [[ "${code}" == 124 ]] && break
            exit 1
        fi
    fi
    uid="$(jq -er '.metadata.uid | select(type == "string" and length > 0)' <<<"${document}")" || {
        printf 'ERROR: Job %s returned an invalid identity\n' "${JOB}" >&2
        exit 1
    }
    if [[ -n "${expected_uid}" && "${expected_uid}" != "${uid}" ]]; then
        printf 'ERROR: Job %s was replaced during its wait\n' "${JOB}" >&2
        exit 1
    fi
    expected_uid="${uid}"
    failed="$(jq -r '
        [.status.conditions[]? |
         select(.status == "True" and (.type == "Failed" or .type == "FailureTarget")) |
         (.reason // .type)] | join(",")
    ' <<<"${document}")"
    if [[ -n "${failed}" ]]; then
        printf 'ERROR: Job %s failed: %s\n' "${JOB}" "${failed}" >&2
        exit 1
    fi
    if jq -e 'any(.status.conditions[]?; .type == "Complete" and .status == "True")' \
        <<<"${document}" >/dev/null; then
        exit 0
    fi
    remaining=$((deadline - SECONDS))
    (( remaining > 0 )) || break
    if pods="$(read_api_json get pods -l "job-name=${JOB}")"; then
        jq -e '.items | type == "array"' <<<"${pods}" >/dev/null || {
            printf 'ERROR: Job %s returned an invalid Pod list\n' "${JOB}" >&2
            exit 1
        }
    else
        code="$?"
        [[ "${code}" == 124 ]] && break
        exit 1
    fi
    fatal="$(jq -r --arg uid "${uid}" '
        [.items[]? |
         select(any(.metadata.ownerReferences[]?; .uid == $uid)) |
         (.status.initContainerStatuses[]?, .status.containerStatuses[]?) |
         .state.waiting.reason // empty |
         select(. == "CreateContainerConfigError" or . == "InvalidImageName")] |
        unique | join(",")
    ' <<<"${pods}")"
    if [[ -n "${fatal}" ]]; then
        printf 'ERROR: Job %s cannot start: %s\n' "${JOB}" "${fatal}" >&2
        exit 1
    fi
    remaining=$((deadline - SECONDS))
    (( remaining > 0 )) || break
    interval=$((remaining < 5 ? remaining : 5))
    wait_started="${SECONDS}"
    watch_budget=$((interval + 10 < remaining ? interval + 10 : remaining))
    if document="$(timeout --foreground "${watch_budget}s" "${kubectl_command[@]}" wait "job/${JOB}" \
        --for=condition=complete "--timeout=${interval}s" --request-timeout=10s -o json \
        2>"${error_file}")"; then
        # The successful watch includes its object, so UID and terminal state
        # can be checked even at the deadline without a second API request.
        jq -e 'any(.status.conditions[]?; .type == "Complete" and .status == "True")' \
            <<<"${document}" >/dev/null || {
            printf 'ERROR: Job %s watch returned no completion proof\n' "${JOB}" >&2
            exit 1
        }
        continue
    fi
    document=""
    # A failed watch must not busy-loop against an unavailable API server.
    delay=$((interval - (SECONDS - wait_started)))
    if (( delay > 0 )); then
        sleep "${delay}"
    fi
done

printf 'ERROR: Job %s did not complete within %ss\n' "${JOB}" "${TIMEOUT_SECONDS}" >&2
exit 1
