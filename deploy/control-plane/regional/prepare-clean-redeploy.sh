#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RESOURCE_INVENTORY="${SCRIPT_DIR}/cleanup-inventory.json"
CLEANUP_STATE_TOOL="$(
    printf '%s' \
        "${REPO_ROOT}/deploy/control-plane/tools/cleanup_state.py"
)"
CLEAN_REDEPLOY_STORE_TOOL="$(
    printf '%s' \
        "${REPO_ROOT}/deploy/control-plane/tools/clean_redeploy_store.py"
)"
CLEAN_REDEPLOY_CONFIG_TOOL="$(
    printf '%s' \
        "${REPO_ROOT}/deploy/control-plane/tools/clean_redeploy_config.py"
)"
MODE=stop
SCOPE=all
NODE_MODE=stop
CONFIG=
STATE_FILE=
CONFIRM_RESET=
EXECUTE=false
OFFLINE_PLAN=false
TIMEOUT_SECONDS=600
DRAIN_POLL_SECONDS=5
NODE_CLEANUP_IMAGE="${GPU_FAULT_NODE_INSTALLER_IMAGE:-public.ecr.aws/amazonlinux/amazonlinux:2023}"
SELECTED_CLUSTER_IDS=()

usage() {
    cat <<'EOF'
Usage:
  prepare-clean-redeploy.sh --config FILE [options]

Safely stop an existing regional deployment before redeploying it. The
default is a dry run: no Kubernetes object or node service is changed.

Options:
  --config FILE             Regional release JSON (required).
  --scope all|gpu           Stop the whole region or selected GPU clusters.
                            Default: all.
  --cluster-id ID           GPU cluster to select; repeat as needed.
                            Required with --scope gpu.
  --mode stop|clean|reset   Stop runtimes; remove controllers/RBAC; or reset
                            the Kubernetes installation while preserving
                            CPU/GPU EKS. Default: stop.
  --node-mode stop|uninstall|skip
                            Stop node systemd units, run the installed node
                            uninstaller, or leave node units unchanged.
                            Default: stop.
  --state-file FILE         New mode-0600 JSON audit file. Required with
                            --execute.
  --confirm-reset VALUE     Required for --mode reset --execute. Value must be
                            RESET_GPU_FAULT_INSTALLATION.
  --timeout-seconds N       Drain and rollout timeout. Default: 600.
  --execute                 Perform the plan. Without it, only print the plan.
  --offline-plan            Do not query live installed-resource ConfigMaps.
                            Show the generated design inventory instead.
  -h, --help                Show this help.

Normal dry-run performs read-only Kubernetes queries so the plan reflects
resources actually installed by prior deployments. The reset mode deletes
the Kubernetes NLB Service and the solution namespaces.
It never deletes CPU EKS or any GPU EKS. The operations manual then deletes
the dedicated Aurora cluster and solution-owned NLB/IAM attachments.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[clean-redeploy] %s\n' "$*"
}

# shellcheck source-path=SCRIPTDIR
# shellcheck source=prepare-clean-redeploy-delete.sh
source "${SCRIPT_DIR}/prepare-clean-redeploy-delete.sh"

while (($# > 0)); do
    case "$1" in
        --config)
            (($# >= 2)) || die "--config requires a value"
            CONFIG=$2
            shift 2
            ;;
        --scope)
            (($# >= 2)) || die "--scope requires a value"
            SCOPE=$2
            shift 2
            ;;
        --cluster-id)
            (($# >= 2)) || die "--cluster-id requires a value"
            SELECTED_CLUSTER_IDS+=("$2")
            shift 2
            ;;
        --mode)
            (($# >= 2)) || die "--mode requires a value"
            MODE=$2
            shift 2
            ;;
        --node-mode)
            (($# >= 2)) || die "--node-mode requires a value"
            NODE_MODE=$2
            shift 2
            ;;
        --state-file)
            (($# >= 2)) || die "--state-file requires a value"
            STATE_FILE=$2
            shift 2
            ;;
        --confirm-reset)
            (($# >= 2)) || die "--confirm-reset requires a value"
            CONFIRM_RESET=$2
            shift 2
            ;;
        --timeout-seconds)
            (($# >= 2)) || die "--timeout-seconds requires a value"
            TIMEOUT_SECONDS=$2
            shift 2
            ;;
        --execute)
            EXECUTE=true
            shift
            ;;
        --offline-plan)
            OFFLINE_PLAN=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "${CONFIG}" ]] || die "--config is required"
[[ -f "${CONFIG}" ]] || die "config file does not exist: ${CONFIG}"
[[ -f "${RESOURCE_INVENTORY}" ]] ||
    die "cleanup resource inventory does not exist: ${RESOURCE_INVENTORY}"
[[ -f "${CLEAN_REDEPLOY_STORE_TOOL}" ]] ||
    die "store helper does not exist: ${CLEAN_REDEPLOY_STORE_TOOL}"
[[ -f "${CLEAN_REDEPLOY_CONFIG_TOOL}" ]] ||
    die "config helper does not exist: ${CLEAN_REDEPLOY_CONFIG_TOOL}"
[[ "${MODE}" == stop || "${MODE}" == clean || "${MODE}" == reset ]] ||
    die "--mode must be stop, clean, or reset"
[[ "${SCOPE}" == all || "${SCOPE}" == gpu ]] ||
    die "--scope must be all or gpu"
[[ "${NODE_MODE}" == stop || "${NODE_MODE}" == uninstall ||
    "${NODE_MODE}" == skip ]] ||
    die "--node-mode must be stop, uninstall, or skip"
[[ "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
    die "--timeout-seconds must be a positive integer"
[[ "${NODE_CLEANUP_IMAGE}" =~ ^[A-Za-z0-9._/@:-]+$ ]] ||
    die "GPU_FAULT_NODE_INSTALLER_IMAGE contains unsupported characters"
if [[ "${SCOPE}" == gpu && ${#SELECTED_CLUSTER_IDS[@]} -eq 0 ]]; then
    die "--scope gpu requires at least one --cluster-id"
fi
if [[ "${SCOPE}" == all && ${#SELECTED_CLUSTER_IDS[@]} -ne 0 ]]; then
    die "--cluster-id may only be used with --scope gpu"
fi
if [[ "${MODE}" == reset ]]; then
    [[ "${SCOPE}" == all ]] ||
        die "--mode reset requires --scope all"
    [[ "${NODE_MODE}" == uninstall ]] ||
        die "--mode reset requires --node-mode uninstall"
fi
if [[ "${EXECUTE}" == true && "${OFFLINE_PLAN}" == true ]]; then
    die "--offline-plan cannot be combined with --execute"
fi
if [[ "${EXECUTE}" == true ]]; then
    [[ -n "${STATE_FILE}" ]] ||
        die "--state-file is required with --execute"
    [[ ! -e "${STATE_FILE}" ]] ||
        die "state file already exists: ${STATE_FILE}"
    if [[ "${MODE}" == reset &&
        "${CONFIRM_RESET}" != RESET_GPU_FAULT_INSTALLATION ]]; then
        die "--mode reset --execute requires --confirm-reset RESET_GPU_FAULT_INSTALLATION"
    fi
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required"

EFFECTIVE_INVENTORY="${RESOURCE_INVENTORY}"
RUNTIME_INVENTORY_FILE=
FLEET_INVENTORY_FILE=
STATE_INITIALIZED=false
STATE_COMPLETE=false
CURRENT_PHASE=PREFLIGHT
cleanup_runtime_inventory() {
    if [[ -n "${RUNTIME_INVENTORY_FILE}" ]]; then
        rm -f "${RUNTIME_INVENTORY_FILE}"
    fi
    if [[ -n "${FLEET_INVENTORY_FILE}" ]]; then
        rm -f "${FLEET_INVENTORY_FILE}"
    fi
}
cleanup_on_exit() {
    local status=$?
    trap - EXIT
    if [[ "${status}" -ne 0 &&
        "${STATE_INITIALIZED}" == true &&
        "${STATE_COMPLETE}" != true ]]; then
        PYTHONDONTWRITEBYTECODE=1 python3 \
            "${CLEANUP_STATE_TOOL}" transition \
            --path "${STATE_FILE}" \
            --phase "${CURRENT_PHASE}" \
            --status FAILED \
            --message "cleanup command exited ${status}" \
            >/dev/null 2>&1 || true
    fi
    cleanup_runtime_inventory
    exit "${status}"
}
trap cleanup_on_exit EXIT
if [[ "${OFFLINE_PLAN}" != true ]]; then
    command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
    RUNTIME_INVENTORY_FILE="$(mktemp)"
    registry_arguments=(
        --config "${CONFIG}"
        --inventory "${RESOURCE_INVENTORY}"
    )
    if [[ "${EXECUTE}" == true ]]; then
        registry_arguments+=(--apply)
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${REPO_ROOT}/deploy/control-plane/tools/collect_installed_resource_registry.py" \
        "${registry_arguments[@]}" >"${RUNTIME_INVENTORY_FILE}"
    EFFECTIVE_INVENTORY="${RUNTIME_INVENTORY_FILE}"
fi

CONFIG_OUTPUT="$(
    PYTHONDONTWRITEBYTECODE=1 python3 "${CLEAN_REDEPLOY_CONFIG_TOOL}" \
        "${CONFIG}" "${EFFECTIVE_INVENTORY}" "${SELECTED_CLUSTER_IDS[@]}"
)" || exit $?

NAMESPACE=
CPU_KUBECONFIG=
CLUSTER_IDS=()
CLUSTER_CONTEXTS=()
CPU_RESOURCES=()
CPU_DEPLOYMENTS=()
CPU_INGRESS_DEPLOYMENTS=()
CPU_CONSUMER_DEPLOYMENTS=()
CPU_AUX_DEPLOYMENTS=()
CPU_DAEMONSETS=()
CPU_CRONJOBS=()
CPU_DELETE_RESOURCES=()
CPU_NLB_SERVICES=()
CPU_DATABASE_POD_PREFERENCE=()
GPU_RESOURCES=()
GPU_DEPLOYMENTS=()
GPU_PRODUCER_DEPLOYMENTS=()
GPU_EXECUTOR_DEPLOYMENTS=()
GPU_DAEMONSETS=()
GPU_DELETE_RESOURCES=()
GPU_NODE_ANNOTATIONS=()
GPU_NODE_LABELS=()
while IFS=$'\t' read -r record first second third fourth fifth sixth; do
    case "${record}" in
        META)
            NAMESPACE=${first}
            CPU_KUBECONFIG=${second}
            ;;
        CLUSTER)
            CLUSTER_IDS+=("${first}")
            CLUSTER_CONTEXTS+=("${second}")
            ;;
        RESOURCE)
            resource_record="${second}"$'\t'"${third}"$'\t'"${fourth}"
            if [[ "${first}" == cpu ]]; then
                CPU_RESOURCES+=("${resource_record}")
                if [[ "${second}" == deployment ]]; then
                    CPU_DEPLOYMENTS+=("${third}")
                    case "${fifth}" in
                        ingress)
                            CPU_INGRESS_DEPLOYMENTS+=("${third}")
                            ;;
                        consumer)
                            CPU_CONSUMER_DEPLOYMENTS+=("${third}")
                            ;;
                        auxiliary)
                            CPU_AUX_DEPLOYMENTS+=("${third}")
                            ;;
                    esac
                elif [[ "${second}" == daemonset ]]; then
                    CPU_DAEMONSETS+=("${third}")
                elif [[ "${second}" == cronjob ]]; then
                    CPU_CRONJOBS+=("${third}")
                fi
                if [[ "${sixth}" == delete ]]; then
                    CPU_DELETE_RESOURCES+=("${resource_record}"$'\t'"${fifth}")
                elif [[ "${fifth}" == nlb ]]; then
                    CPU_NLB_SERVICES+=("${third}")
                fi
            else
                GPU_RESOURCES+=("${resource_record}")
                if [[ "${second}" == deployment ]]; then
                    GPU_DEPLOYMENTS+=("${third}")
                    if [[ "${fifth}" == producer ]]; then
                        GPU_PRODUCER_DEPLOYMENTS+=("${third}")
                    elif [[ "${fifth}" == executor ]]; then
                        GPU_EXECUTOR_DEPLOYMENTS+=("${third}")
                    fi
                elif [[ "${second}" == daemonset ]]; then
                    GPU_DAEMONSETS+=("${third}")
                fi
                if [[ "${sixth}" == delete ]]; then
                    GPU_DELETE_RESOURCES+=("${resource_record}"$'\t'"${fifth}")
                fi
            fi
            ;;
        DBPREF)
            CPU_DATABASE_POD_PREFERENCE+=("${first}")
            ;;
        ANNOTATION)
            GPU_NODE_ANNOTATIONS+=("${first}")
            ;;
        LABEL)
            GPU_NODE_LABELS+=("${first}")
            ;;
        *)
            die "unexpected config parser record: ${record}"
            ;;
    esac
done <<<"${CONFIG_OUTPUT}"

[[ -n "${NAMESPACE}" && -n "${CPU_KUBECONFIG}" ]] ||
    die "regional release config did not produce CPU settings"
(( ${#CLUSTER_IDS[@]} > 0 )) ||
    die "regional release config did not select a GPU cluster"

PLAN_STEP=0
plan_step() {
    # Numbered in one place so the scope-specific branches stay contiguous.
    local line
    PLAN_STEP=$((PLAN_STEP + 1))
    printf '%3d. %s\n' "${PLAN_STEP}" "$1"
    shift
    for line in "$@"; do
        printf '     %s\n' "${line}"
    done
}

print_plan() {
    printf 'DRY RUN: no Kubernetes object or node service will be changed.\n'
    printf 'Config: %s\n' "${CONFIG}"
    printf 'Scope: %s; mode: %s; node mode: %s\n' \
        "${SCOPE}" "${MODE}" "${NODE_MODE}"
    printf 'CPU: kubeconfig=%s namespace=%s\n' \
        "${CPU_KUBECONFIG}" "${NAMESPACE}"
    local index
    for index in "${!CLUSTER_IDS[@]}"; do
        printf 'GPU: cluster_id=%s context=%s\n' \
            "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
    printf 'CPU ingress deployments: %s\n' \
        "${CPU_INGRESS_DEPLOYMENTS[*]}"
    printf 'CPU consumer deployments: %s\n' \
        "${CPU_CONSUMER_DEPLOYMENTS[*]}"
    printf 'GPU producer deployments: %s\n' \
        "${GPU_PRODUCER_DEPLOYMENTS[*]}"
    printf 'GPU executor deployments: %s\n' \
        "${GPU_EXECUTOR_DEPLOYMENTS[*]}"
    printf 'GPU daemonsets: %s\n' "${GPU_DAEMONSETS[*]}"
    printf 'Ordered plan:\n'
    plan_step 'Verify every Kubernetes context and capture the current replica/resource' \
        'state in a mode-0600 JSON file.'
    plan_step 'Fail closed if Aurora contains an active workflow or open remote command.'
    if [[ "${SCOPE}" == all ]]; then
        plan_step 'Publish DRAINING for every GPU cluster in the regional registry: the API' \
            'then refuses new collector events and executor claims while in-flight' \
            'leases may still renew and complete.'
        plan_step 'Stop every GPU producer Deployment and DaemonSet found in the live' \
            'installed-resource registries.'
        plan_step 'With CPU ingress and consumers still running, wait until no live lease,' \
            'processor row or telemetry spool row remains on two consecutive polls;' \
            'a lease still live at the deadline fails the run, other leftovers proceed.'
        plan_step 'Stop registered CPU consumers, fail the workflows and remote commands' \
            'nothing may claim any more (reset mode; other modes fail if any remain),' \
            'then scale registered CPU ingress Deployments to zero.'
        plan_step 'Stop each registered GPU executor, restore any quiesced host services,' \
            'then stop or uninstall all node collectors, Node Agent, certificate timer,' \
            'DCGM exporter, and GPU persistence service.'
        plan_step 'Stop ADOT, suspend Aurora credential refresh, and remove the control-plane' \
            'sysctl DaemonSet.'
    else
        plan_step 'Stop the selected clusters'"'"' GPU producer Deployments and DaemonSets' \
            'found in the live installed-resource registries.'
        plan_step 'Keep the CPU control plane running while selected-cluster processor,' \
            'spool, workflow, and remote-command rows drain.'
        plan_step 'Stop the selected clusters'"'"' registered GPU executor, restore any' \
            'quiesced host services, then stop or uninstall the node components.'
    fi
    if [[ "${MODE}" == clean || "${MODE}" == reset ]]; then
        plan_step 'Delete application Deployments, DaemonSets, PDBs, service accounts and' \
            'RBAC in scope. Preserve Secrets, release ConfigMaps, Aurora, NLB, VPC,' \
            'IAM, and audit records unless reset mode is selected.'
    fi
    if [[ "${MODE}" == reset ]]; then
        plan_step 'Delete the Kubernetes NLB Service, then delete the solution namespace' \
            'in CPU EKS and every selected GPU EKS. Preserve all EKS clusters.'
        plan_step 'Delete the dedicated Aurora cluster and solution-owned NLB security' \
            'group, ACM certificate, private PKI Secret, Load Balancer Controller' \
            'and IAM resources with the explicit commands in manual section 0.3.'
    fi
    printf '\nTo execute, add --execute --state-file /secure/gpu-fault/<name>.json\n'
}

if [[ "${EXECUTE}" != true ]]; then
    print_plan
    exit 0
fi

command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
[[ -f "${CPU_KUBECONFIG}" ]] ||
    die "CPU kubeconfig does not exist: ${CPU_KUBECONFIG}"

install -d -m 0700 "$(dirname "${STATE_FILE}")"
PYTHONDONTWRITEBYTECODE=1 python3 \
    "${CLEANUP_STATE_TOOL}" init \
    --path "${STATE_FILE}" \
    --config "${CONFIG}" \
    --inventory "${EFFECTIVE_INVENTORY}" \
    --scope "${SCOPE}" \
    --mode "${MODE}" \
    --node-mode "${NODE_MODE}" >/dev/null
STATE_INITIALIZED=true

cpu_kubectl() {
    kubectl --kubeconfig "${CPU_KUBECONFIG}" "$@"
}

gpu_kubectl() {
    local context=$1
    shift
    kubectl --context "${context}" "$@"
}

store_tool() {
    # Every Aurora read or write runs inside a control-plane Pod: the helper's
    # source goes in on stdin (``python -``), the subcommand and its arguments
    # as argv, so the Pod needs nothing but its own psycopg and
    # GPU_FAULT_STORE_URL. See deploy/control-plane/tools/clean_redeploy_store.py
    # for the subcommands and their tab-separated output.
    local pod=$1
    shift
    cpu_kubectl -n "${NAMESPACE}" exec -i "${pod}" -- python - "$@" \
        <"${CLEAN_REDEPLOY_STORE_TOOL}"
}

record_state() {
    local scope=$1
    local context=$2
    local kind=$3
    local name=$4
    local previous=$5
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${CLEANUP_STATE_TOOL}" record \
        --path "${STATE_FILE}" \
        --resource-scope "${scope}" \
        --context "${context}" \
        --kind "${kind}" \
        --name "${name}" \
        --previous "${previous}" >/dev/null
}

transition_state() {
    local phase=$1
    local status=$2
    local message=$3
    CURRENT_PHASE=${phase}
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${CLEANUP_STATE_TOOL}" transition \
        --path "${STATE_FILE}" \
        --phase "${phase}" \
        --status "${status}" \
        --message "${message}" >/dev/null
}

capture_cpu_state() {
    local name
    local value
    for name in "${CPU_DEPLOYMENTS[@]}"; do
        if value="$(
            cpu_kubectl -n "${NAMESPACE}" get deployment "${name}" \
                -o jsonpath='{.spec.replicas}' 2>/dev/null
        )"; then
            record_state cpu "${CPU_KUBECONFIG}" deployment \
                "${name}" "${value:-0}"
        fi
    done
    for name in "${CPU_DAEMONSETS[@]}"; do
        if cpu_kubectl -n "${NAMESPACE}" get daemonset \
            "${name}" >/dev/null 2>&1; then
            record_state cpu "${CPU_KUBECONFIG}" daemonset \
                "${name}" present
        fi
    done
    for name in "${CPU_CRONJOBS[@]}"; do
        if value="$(
            cpu_kubectl -n "${NAMESPACE}" get cronjob \
                "${name}" -o jsonpath='{.spec.suspend}' 2>/dev/null
        )"; then
            record_state cpu "${CPU_KUBECONFIG}" cronjob \
                "${name}" "${value:-false}"
        fi
    done
}

capture_gpu_state() {
    local cluster_id=$1
    local context=$2
    local name
    local value
    for name in "${GPU_DEPLOYMENTS[@]}"; do
        if value="$(
            gpu_kubectl "${context}" -n "${NAMESPACE}" \
                get deployment "${name}" \
                -o jsonpath='{.spec.replicas}' 2>/dev/null
        )"; then
            record_state "gpu:${cluster_id}" "${context}" deployment \
                "${name}" "${value:-0}"
        fi
    done
    for name in "${GPU_DAEMONSETS[@]}"; do
        if gpu_kubectl "${context}" -n "${NAMESPACE}" \
            get daemonset "${name}" >/dev/null 2>&1; then
            record_state "gpu:${cluster_id}" "${context}" daemonset \
                "${name}" present
        fi
    done
    value="$(
        gpu_kubectl "${context}" get nodes --no-headers 2>/dev/null |
            wc -l | tr -d ' '
    )"
    record_state "gpu:${cluster_id}" "${context}" nodes \
        selected-for-node-cleanup "${value}"
}

deployment_replicas_cpu() {
    cpu_kubectl -n "${NAMESPACE}" get deployment "$1" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || true
}

find_database_pod() {
    # Ready and not terminating: a Running Pod already being deleted (a roll
    # the previous command started) completes before the exec reaches it.
    local name
    local pod
    for name in "${CPU_DATABASE_POD_PREFERENCE[@]}"; do
        pod="$(
            cpu_kubectl -n "${NAMESPACE}" get pod -l "app=${name}" \
                --field-selector=status.phase=Running -o json 2>/dev/null |
                python3 -c 'import json,sys
for i in json.load(sys.stdin).get("items", []):
    if not i["metadata"].get("deletionTimestamp") and all(c.get("ready") for c in i.get("status", {}).get("containerStatuses", [])):
        print(i["metadata"]["name"]); break' 2>/dev/null || true
        )"
        if [[ -n "${pod}" ]]; then
            printf '%s\n' "${pod}"
            return 0
        fi
    done
    return 1
}

wait_for_cpu_rollouts() { # a role still rolling would hand us terminating Pods
    local deployment
    for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}" "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
        [[ "$(deployment_replicas_cpu "${deployment}")" =~ ^[1-9] ]] || continue
        cpu_kubectl -n "${NAMESPACE}" rollout status "deployment/${deployment}" \
            --timeout="${TIMEOUT_SECONDS}s" >/dev/null ||
            die "control-plane deployment ${deployment} did not settle before cleanup"
    done
}

reuse_previous_fleet_inventory() {
    # An earlier run of this reset already stopped the control plane, so no
    # Pod can export the fleet inventory again -- but that run attached the
    # export to its state file, which the caller moves aside as
    # <state>.failed-<stamp>.json before retrying (live uninstall,
    # 2026-09-12). Reuse the newest such snapshot instead of refusing.
    local candidate
    local snapshot
    [[ -n "${STATE_FILE}" ]] || return 1
    candidate="$(
        find "$(dirname "${STATE_FILE}")" -maxdepth 1 -type f \
            -name "$(basename "${STATE_FILE%.json}").failed-*.json" \
            -printf '%T@ %p\n' 2>/dev/null |
            sort -rn | head -n 1 | cut -d' ' -f2- || true
    )"
    [[ -n "${candidate}" ]] || return 1
    snapshot="$(
        PYTHONDONTWRITEBYTECODE=1 python3 "${CLEANUP_STATE_TOOL}" reuse-fleet \
            --path "${STATE_FILE}" --from "${candidate}" 2>/dev/null | head -n 1
    )"
    [[ "${snapshot}" =~ ^[0-9]+$ ]] || return 1
    log "control plane already stopped by an earlier run; reused its fleet inventory (${snapshot} agent record(s)) from ${candidate}"
}

capture_fleet_inventory() {
    local pod=$1
    local missing
    FLEET_INVENTORY_FILE="$(mktemp)"
    store_tool "${pod}" fleet-agents "${CLUSTER_IDS[@]}" >"${FLEET_INVENTORY_FILE}"
    PYTHONDONTWRITEBYTECODE=1 python3 \
        "${CLEANUP_STATE_TOOL}" attach-fleet \
        --path "${STATE_FILE}" \
        --input "${FLEET_INVENTORY_FILE}" >/dev/null
    missing="$(
        python3 - "${FLEET_INVENTORY_FILE}" <<'PY'
import json
import sys

agents = json.load(open(sys.argv[1], encoding="utf-8"))
missing = []
for agent in agents:
    if agent.get("lifecycle_state") != "ACTIVE":
        continue
    inventory = agent.get("installed_unit_inventory")
    if not inventory or not inventory.get("units"):
        missing.append(
            f'{agent.get("cluster_id")}/{agent.get("node_id")}'
        )
print(",".join(missing))
PY
    )"
    if [[ "${MODE}" == reset && -n "${missing}" ]]; then
        die "ACTIVE agents lack installed unit inventory: ${missing}"
    fi
}

database_snapshot() {
    # active_workflows open_commands processor_rows spool_rows leased_live
    # leased_details, tab-separated; --scope gpu restricts to CLUSTER_IDS.
    local pod=$1
    local query_scope=$2
    store_tool "${pod}" snapshot --scope "${query_scope}" "${CLUSTER_IDS[@]}"
}

assert_no_active_work() {
    local pod=$1
    local snapshot
    local active_workflows
    local open_commands
    local processor_rows
    local spool_rows
    local leased_live
    local leased_details
    snapshot="$(database_snapshot "${pod}" "${SCOPE}")"
    IFS=$'\t' read -r active_workflows open_commands \
        processor_rows spool_rows leased_live leased_details <<<"${snapshot}"
    [[ "${active_workflows}" =~ ^[0-9]+$ &&
        "${open_commands}" =~ ^[0-9]+$ ]] ||
        die "could not parse Aurora safety snapshot: ${snapshot}"
    if ((active_workflows != 0 || open_commands != 0)); then
        die "active workflows=${active_workflows}, open remote commands=${open_commands}; finish or block them before cleanup"
    fi
    log "safety snapshot: workflows=${active_workflows} remote_commands=${open_commands} processor_rows=${processor_rows} spool_rows=${spool_rows} live_leases=${leased_live}"
}

wait_for_shared_queues() {
    # --scope gpu: the CPU control plane keeps running, so every row of the
    # selected clusters can still complete; all four counters must reach zero.
    local pod=$1
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local snapshot
    local active_workflows
    local open_commands
    local processor_rows
    local spool_rows
    local leased_live
    local leased_details
    while ((SECONDS < deadline)); do
        snapshot="$(database_snapshot "${pod}" "${SCOPE}")"
        IFS=$'\t' read -r active_workflows open_commands \
            processor_rows spool_rows leased_live leased_details <<<"${snapshot}"
        if [[ "${active_workflows}" == 0 && "${open_commands}" == 0 &&
            "${processor_rows}" == 0 && "${spool_rows}" == 0 ]]; then
            log "Aurora queues are drained"
            return 0
        fi
        log "waiting: workflows=${active_workflows} remote_commands=${open_commands} processor_rows=${processor_rows} spool_rows=${spool_rows} live_leases=${leased_live}"
        sleep "${DRAIN_POLL_SECONDS}"
    done
    die "Aurora queues did not drain within ${TIMEOUT_SECONDS}s"
}

wait_for_live_work_to_finish() {
    # --scope all, with ingress and the consumers still running: a live lease
    # can renew and finish, the processor and the spool can drain. Every
    # cluster is DRAINING, so PENDING rows can never be claimed and are not
    # waited for (abandon_unclaimable_work fails them once the consumers are
    # gone). Converged: no live lease, processor row or spool row on two
    # consecutive polls. A lease still live at the deadline is a node action
    # in flight: fail with ingress and the consumers untouched so it can
    # finish and a rerun starts coherent.
    local pod=$1
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local converged=0
    local snapshot
    local active_workflows
    local open_commands
    local processor_rows
    local spool_rows
    local leased_live
    local leased_details
    while :; do
        snapshot="$(database_snapshot "${pod}" "${SCOPE}")"
        IFS=$'\t' read -r active_workflows open_commands \
            processor_rows spool_rows leased_live leased_details <<<"${snapshot}"
        [[ "${leased_live}" =~ ^[0-9]+$ && "${processor_rows}" =~ ^[0-9]+$ &&
            "${spool_rows}" =~ ^[0-9]+$ ]] ||
            die "could not parse Aurora drain snapshot: ${snapshot}"
        if ((leased_live == 0 && processor_rows == 0 && spool_rows == 0)); then
            converged=$((converged + 1))
            if ((converged >= 2)); then
                log "Aurora drained: no live lease, processor row or spool row on two consecutive polls (workflows=${active_workflows} remote_commands=${open_commands} remain unclaimable under DRAINING)"
                return 0
            fi
        else
            converged=0
        fi
        ((SECONDS < deadline)) || break
        log "waiting: live_leases=${leased_live} processor_rows=${processor_rows} spool_rows=${spool_rows} (workflows=${active_workflows} remote_commands=${open_commands})"
        sleep "${DRAIN_POLL_SECONDS}"
    done
    if ((leased_live > 0)); then
        die "${leased_live} remote command lease(s) still live after ${TIMEOUT_SECONDS}s: a node action is running; ingress and the consumers stay up so it can finish, rerun afterwards (key|operation|nodes): ${leased_details}"
    fi
    log "drain window of ${TIMEOUT_SECONDS}s elapsed with no live lease; ${open_commands} remote command(s) and ${active_workflows} workflow(s) cannot execute under DRAINING and will be abandoned; processor_rows=${processor_rows} spool_rows=${spool_rows} are left behind"
}

scale_cpu_deployment_zero() {
    local name=$1
    local replicas
    replicas="$(deployment_replicas_cpu "${name}")"
    [[ -n "${replicas}" ]] || return 0
    cpu_kubectl -n "${NAMESPACE}" scale "deployment/${name}" --replicas=0
    if ((replicas > 0)); then
        cpu_kubectl -n "${NAMESPACE}" wait --for=delete pod \
            -l "app=${name}" --timeout="${TIMEOUT_SECONDS}s"
    fi
}

scale_gpu_deployment_zero() {
    local context=$1
    local name=$2
    local replicas
    replicas="$(
        gpu_kubectl "${context}" -n "${NAMESPACE}" \
            get deployment "${name}" \
            -o jsonpath='{.spec.replicas}' 2>/dev/null || true
    )"
    [[ -n "${replicas}" ]] || return 0
    gpu_kubectl "${context}" -n "${NAMESPACE}" \
        scale "deployment/${name}" --replicas=0
    if ((replicas > 0)); then
        gpu_kubectl "${context}" -n "${NAMESPACE}" \
            wait --for=delete pod -l "app=${name}" \
            --timeout="${TIMEOUT_SECONDS}s"
    fi
}

fail_orphaned_remote_commands() {
    # LEASED commands whose lease has lapsed after every executor was scaled
    # to zero have no claimant left to complete, renew or acknowledge a
    # cancellation; the drain would wait on them forever. Reset is a wipe, so
    # they are failed here with an audit source, and the consumers then fail
    # their workflow steps in the ordinary way.
    local pod=$1
    local failed
    failed="$(store_tool "${pod}" fail-orphaned-leases "${CLUSTER_IDS[@]}")"
    [[ "${failed}" =~ ^[0-9]+$ ]] ||
        die "could not fail orphaned remote commands: ${failed}"
    if ((failed > 0)); then
        log "failed ${failed} orphaned LEASED remote command(s) with no executor left to complete them"
    fi
}

drain_registry_clusters() {
    # Under DRAINING the API refuses new collector events (no incident can be
    # born) and executor claims (nothing PENDING can start) while heartbeats,
    # lease renewals and results still pass, so a running command finishes.
    # Same rollout entry point as remove-cluster, itself fail-closed on
    # non-idle remote commands; re-publishing a DRAINING cluster is a new
    # revision with the same content, so a rerun is harmless. One invocation
    # for every cluster: a revision publishes the clusters it does not name as
    # ACTIVE, so draining them one revision at a time would flip the earlier
    # ones back and leave only the last one DRAINING.
    local arguments=()
    local cluster_id
    for cluster_id in "${CLUSTER_IDS[@]}"; do
        log "${cluster_id}: publishing DRAINING in the regional registry"
        arguments+=(--cluster-id "${cluster_id}")
    done
    "${SCRIPT_DIR}/rollout-regional-release.sh" drain-cluster \
        "${arguments[@]}" --config "${CONFIG}" ||
        die "could not publish DRAINING for ${CLUSTER_IDS[*]} in the regional registry"
}

ABANDONED_SUMMARY=
abandon_unclaimable_work() {
    # Reset only, between CONTROL_CONSUMERS_STOPPED and INGRESS_STOPPED: no
    # executor may claim what is left (DRAINING) and no consumer can turn a
    # failed step into a successor workflow any more. Workflows first, then
    # commands, one transaction (clean_redeploy_store.py abandon-unclaimable).
    local pod=$1
    local summary
    local workflows
    local commands
    local workflow_keys
    local command_keys
    summary="$(store_tool "${pod}" abandon-unclaimable)"
    IFS=$'\t' read -r workflows commands workflow_keys command_keys <<<"${summary}"
    [[ "${workflows}" =~ ^[0-9]+$ && "${commands}" =~ ^[0-9]+$ ]] ||
        die "could not abandon the unclaimable work: ${summary}"
    ABANDONED_SUMMARY="abandoned ${workflows} workflow(s) and ${commands} remote command(s) nothing may claim under DRAINING (status FAILED, source clean-redeploy-drain)"
    if ((workflows > 0)); then
        ABANDONED_SUMMARY+="; workflows: ${workflow_keys}"
    fi
    if ((commands > 0)); then
        ABANDONED_SUMMARY+="; commands: ${command_keys}"
    fi
    log "${ABANDONED_SUMMARY}"
}

run_node_cleanup() {
    local cluster_id=$1
    local context=$2
    local cleanup_name=gpu-fault-clean-redeploy-node-cleanup
    local host_script
    local host_script_b64
    host_script="$(
        cat <<'HOST_SCRIPT'
set -eu
mode=$1

if [ -s /opt/gpu-fault/installed-units.txt ]; then
    units="$(cat /opt/gpu-fault/installed-units.txt)"
else
    units="$(
        find /etc/systemd/system -maxdepth 1 -type f \
            \( -name 'gpu-fault-*.service' \
            -o -name 'gpu-fault-*.timer' \) \
            -printf '%f\n' |
            sort -u
    )"
fi

if [ "${mode}" = uninstall ] && [ -x /opt/gpu-fault/uninstall ]; then
    /opt/gpu-fault/uninstall
else
    for state_file in /var/lib/gpu-fault/quiesce/quiesce-*.json; do
        [ -e "${state_file}" ] || continue
        restore=/opt/gpu-fault/current/venv/bin/gpu-fault-restore-gpu-services
        if [ ! -x "${restore}" ]; then
            restore=/opt/gpu-fault/venv/bin/gpu-fault-restore-gpu-services
        fi
        if [ ! -x "${restore}" ]; then
            echo "quiesce state exists but restore command is unavailable" >&2
            exit 1
        fi
        "${restore}" --state-file "${state_file}"
    done
    if [ "${mode}" = uninstall ]; then
        echo "/opt/gpu-fault/uninstall was absent; discovered units were stopped only" >&2
    fi
fi

for unit in ${units}; do
    case "${unit}" in
        gpu-fault-*.service | gpu-fault-*.timer) ;;
        *)
            echo "invalid installed unit name: ${unit}" >&2
            exit 1
            ;;
    esac
    systemctl disable --now "${unit}" >/dev/null 2>&1 || true
    if [ "${mode}" = uninstall ]; then
        rm -f "/etc/systemd/system/${unit}"
    fi
    if systemctl is-active --quiet "${unit}"; then
        echo "unit remains active: ${unit}" >&2
        exit 1
    fi
done
if [ "${mode}" = uninstall ]; then
    systemctl daemon-reload
fi
HOST_SCRIPT
    )"
    host_script_b64="$(
        printf '%s\n' "${host_script}" | base64 | tr -d '\n'
    )"

    log "${cluster_id}: applying node cleanup DaemonSet (${NODE_MODE})"
    gpu_kubectl "${context}" -n "${NAMESPACE}" delete daemonset \
        "${cleanup_name}" --ignore-not-found >/dev/null
    gpu_kubectl "${context}" apply -f - <<YAML
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ${cleanup_name}
  namespace: ${NAMESPACE}
  labels:
    app: ${cleanup_name}
spec:
  selector:
    matchLabels:
      app: ${cleanup_name}
  template:
    metadata:
      labels:
        app: ${cleanup_name}
    spec:
      hostPID: true
      tolerations:
        - operator: Exists
      containers:
        - name: cleanup
          image: ${NODE_CLEANUP_IMAGE}
          securityContext:
            privileged: true
          command: ["/bin/sh", "-c"]
          args:
            - |-
              printf '%s' '${host_script_b64}' | base64 -d |
                chroot /host /bin/sh -s -- '${NODE_MODE}'
              touch /tmp/cleanup-complete
              sleep 86400
          readinessProbe:
            exec:
              command: ["/bin/sh", "-c", "test -f /tmp/cleanup-complete"]
            periodSeconds: 2
          volumeMounts:
            - name: host
              mountPath: /host
              mountPropagation: HostToContainer
      volumes:
        - name: host
          hostPath:
            path: /
            type: Directory
YAML
    local desired
    desired="$(
        gpu_kubectl "${context}" -n "${NAMESPACE}" get daemonset \
            "${cleanup_name}" \
            -o jsonpath='{.status.desiredNumberScheduled}'
    )"
    [[ "${desired}" =~ ^[1-9][0-9]*$ ]] ||
        die "${cluster_id}: node cleanup scheduled on no nodes"
    gpu_kubectl "${context}" -n "${NAMESPACE}" rollout status \
        "daemonset/${cleanup_name}" --timeout="${TIMEOUT_SECONDS}s"
    gpu_kubectl "${context}" -n "${NAMESPACE}" delete daemonset \
        "${cleanup_name}" --wait=true --timeout="${TIMEOUT_SECONDS}s"
}

stop_gpu_producers() {
    local cluster_id=$1
    local context=$2
    local name
    for name in "${GPU_PRODUCER_DEPLOYMENTS[@]}"; do
        scale_gpu_deployment_zero "${context}" "${name}"
    done
    for name in "${GPU_DAEMONSETS[@]}"; do
        gpu_kubectl "${context}" -n "${NAMESPACE}" delete daemonset \
            "${name}" --ignore-not-found
    done
    # Node components stop later, inside GPU_EXECUTORS_STOPPED, after the
    # drain and the executors: uninstalling agents while the control plane
    # could still turn their disappearance into incidents left commands
    # LEASED with no claimant once ingress went down (live uninstall,
    # 2026-09-12). Every cluster is DRAINING by now, so the API refuses the
    # events a stopping producer might still emit, and ingress stays up until
    # the drain has settled and the leftovers are failed.
}

assert_no_quarantined_nodes() {
    local cluster_id=$1
    local context=$2
    local count
    count="$(
        gpu_kubectl "${context}" get nodes -o json |
            python3 -c '
import json
import sys

document = json.load(sys.stdin)
print(
    sum(
        1
        for item in document.get("items", [])
        if any(
            taint.get("key") == "gpu-fault.io/quarantined"
            for taint in item.get("spec", {}).get("taints", [])
        )
    )
)
'
    )"
    [[ "${count}" == 0 ]] ||
        die "${cluster_id}: ${count} node(s) retain gpu-fault.io/quarantined; resolve their health and restore scheduling before reset"
}

release_parked_spares() {
    # A declared warm spare is the spare label plus a cordon this deployment
    # placed. Stripping the label alone leaves an anonymous cordon behind, and
    # the next bootstrap's node barrier refuses the node as operator-cordoned
    # (live 2026-09-12). Restore the recorded baseline: uncordon unless the
    # node was already unschedulable before the declaration.
    local context=$1
    local node
    while IFS= read -r node; do
        [[ -n "${node}" ]] || continue
        log "releasing warm spare cordon on ${node}"
        gpu_kubectl "${context}" uncordon "${node}"
    done < <(
        gpu_kubectl "${context}" get nodes -l gpu-fault.io/spare=true -o json |
            python3 -c '
import json, sys
for item in json.load(sys.stdin).get("items", []):
    meta = item.get("metadata", {})
    before = (meta.get("annotations") or {}).get("gpu-fault.io/previous-unschedulable")
    if item.get("spec", {}).get("unschedulable") and str(before).lower() != "true":
        print(meta.get("name", ""))
'
    )
}

clear_node_metadata() {
    local context=$1
    local annotation
    local label
    local arguments=()
    local nodes=()
    # `--all` walks every node the context can reach and would strip metadata
    # from nodes owned by other tenants of a shared cluster, not just this
    # deployment's. A node is ours iff it still carries one of the gpu-fault.io
    # labels or annotations this deployment set, so resolve that exact set from
    # the target cluster's API and touch nothing outside it. This whole step
    # already runs only under `--mode reset --execute`, which is gated by
    # `--confirm-reset RESET_GPU_FAULT_INSTALLATION`.
    mapfile -t nodes < <(
        gpu_kubectl "${context}" get nodes -o json |
            GPU_FAULT_METADATA_KEYS="$(
                printf '%s\n' "${GPU_NODE_ANNOTATIONS[@]}" "${GPU_NODE_LABELS[@]}"
            )" python3 -c '
import json
import os
import sys

keys = {k for k in os.environ.get("GPU_FAULT_METADATA_KEYS", "").splitlines() if k}
document = json.load(sys.stdin)
for item in document.get("items", []):
    meta = item.get("metadata", {})
    present = set(meta.get("labels") or {}) | set(meta.get("annotations") or {})
    if present & keys:
        print(meta.get("name", ""))
'
    )
    ((${#nodes[@]})) || return 0
    release_parked_spares "${context}"
    for annotation in "${GPU_NODE_ANNOTATIONS[@]}"; do
        arguments+=("${annotation}-")
    done
    gpu_kubectl "${context}" annotate nodes "${nodes[@]}" --overwrite \
        "${arguments[@]}"
    arguments=()
    for label in "${GPU_NODE_LABELS[@]}"; do
        arguments+=("${label}-")
    done
    gpu_kubectl "${context}" label nodes "${nodes[@]}" --overwrite \
        "${arguments[@]}"
}

delete_control_plane_nlb_service() {
    local name
    for name in "${CPU_NLB_SERVICES[@]}"; do
        if cpu_kubectl -n "${NAMESPACE}" get service "${name}" \
            >/dev/null 2>&1; then
            cpu_kubectl -n "${NAMESPACE}" delete service \
                "${name}" --wait=true \
                --timeout="${TIMEOUT_SECONDS}s"
        fi
    done
}

verify_gpu_stopped() {
    local context=$1
    local name
    local replicas
    for name in "${GPU_DEPLOYMENTS[@]}"; do
        replicas="$(
            gpu_kubectl "${context}" -n "${NAMESPACE}" \
                get deployment "${name}" \
                -o jsonpath='{.spec.replicas}' 2>/dev/null || true
        )"
        [[ -z "${replicas}" || "${replicas}" == 0 ]] ||
            die "${context}: deployment ${name} still has ${replicas} replicas"
    done
    for name in \
        "${GPU_DAEMONSETS[@]}" \
        gpu-fault-clean-redeploy-node-cleanup; do
        if gpu_kubectl "${context}" -n "${NAMESPACE}" get daemonset \
            "${name}" >/dev/null 2>&1; then
            die "${context}: daemonset ${name} remains"
        fi
    done
}

log "validating Kubernetes contexts"
cpu_kubectl get --raw=/readyz >/dev/null
for index in "${!CLUSTER_IDS[@]}"; do
    gpu_kubectl "${CLUSTER_CONTEXTS[index]}" get --raw=/readyz >/dev/null
done
if [[ "${MODE}" == reset ]]; then
    for index in "${!CLUSTER_IDS[@]}"; do
        assert_no_quarantined_nodes \
            "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
fi

if [[ "${SCOPE}" == all ]]; then
    capture_cpu_state
fi
for index in "${!CLUSTER_IDS[@]}"; do
    capture_gpu_state "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
done
log "captured pre-clean state in ${STATE_FILE}"

if [[ "${SCOPE}" == all ]]; then wait_for_cpu_rollouts; fi
DATABASE_POD="$(find_database_pod || true)"
# PRESENT: the Deployments exist (a previous run may have scaled them to
# zero). RUNNING: at least one still has replicas, so a Pod can exist and
# work can still be created. INGRESS_RUNNING: the API itself has replicas --
# the registry publish and the abandonment both need an ingress Pod, and an
# earlier run may have stopped ingress while the consumers still ran.
CPU_RUNTIME_PRESENT=false
CPU_RUNTIME_RUNNING=false
CPU_INGRESS_RUNNING=false
for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}"; do
    replicas="$(deployment_replicas_cpu "${deployment}")"
    [[ -n "${replicas}" ]] || continue
    CPU_RUNTIME_PRESENT=true
    if ((replicas > 0)); then
        CPU_RUNTIME_RUNNING=true
        CPU_INGRESS_RUNNING=true
    fi
done
for deployment in "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
    replicas="$(deployment_replicas_cpu "${deployment}")"
    [[ -n "${replicas}" ]] || continue
    CPU_RUNTIME_PRESENT=true
    if ((replicas > 0)); then
        CPU_RUNTIME_RUNNING=true
    fi
done
if [[ -n "${DATABASE_POD}" ]]; then
    capture_fleet_inventory "${DATABASE_POD}"
elif [[ "${MODE}" == reset ]]; then
    if [[ "${CPU_RUNTIME_RUNNING}" == true ]] ||
        ! reuse_previous_fleet_inventory; then
        die "reset requires a running control-plane pod to export fleet inventory (or an earlier run's ${STATE_FILE%.json}.failed-*.json to reuse)"
    fi
fi
if [[ -n "${DATABASE_POD}" ]]; then
    if [[ "${MODE}" == reset && "${EXECUTE}" == true ]]; then
        fail_orphaned_remote_commands "${DATABASE_POD}"
    fi
    assert_no_active_work "${DATABASE_POD}"
elif [[ "${SCOPE}" == gpu || "${CPU_RUNTIME_RUNNING}" == true ]]; then
    die "no running CPU control-plane pod is available for the Aurora safety check"
else
    log "CPU runtime is already stopped; nothing can enqueue work, Aurora runtime checks are skipped"
fi
transition_state \
    PREFLIGHT COMPLETED \
    "contexts, quarantine state, inventory, and active work verified"

if [[ "${SCOPE}" == all && "${CPU_RUNTIME_PRESENT}" == true ]]; then
    # Before anything stops: from here on no incident can be born through
    # the API and no claim can start, while in-flight leases still finish.
    if [[ "${CPU_INGRESS_RUNNING}" == true ]]; then
        transition_state \
            CLUSTERS_DRAINING IN_PROGRESS \
            "publishing DRAINING for every GPU cluster in the regional registry"
        drain_registry_clusters
        transition_state \
            CLUSTERS_DRAINING COMPLETED \
            "regional registry publishes DRAINING for ${CLUSTER_IDS[*]}: no new incident or claim may start, in-flight leases may still finish"
    elif [[ "${CPU_RUNTIME_RUNNING}" == true ]]; then
        transition_state \
            CLUSTERS_DRAINING COMPLETED \
            "CPU ingress already stopped by an earlier run; the API admits no event or claim, so no DRAINING publish is possible or needed"
    else
        transition_state \
            CLUSTERS_DRAINING COMPLETED \
            "control plane already stopped by an earlier run; nothing can enqueue work"
    fi
fi

transition_state \
    GPU_DATA_PLANE_SOURCES_STOPPED IN_PROGRESS \
    "stopping GPU producers"
for index in "${!CLUSTER_IDS[@]}"; do
    stop_gpu_producers \
        "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
done
transition_state \
    GPU_DATA_PLANE_SOURCES_STOPPED COMPLETED \
    "GPU producers stopped"

if [[ "${SCOPE}" == all && "${CPU_RUNTIME_PRESENT}" == true ]]; then
    if [[ "${CPU_RUNTIME_RUNNING}" == true ]]; then
        DATABASE_POD="$(find_database_pod || true)"
        [[ -n "${DATABASE_POD}" ]] ||
            die "no control-plane pod remains to verify the Aurora drain"
        transition_state \
            QUEUES_DRAINED IN_PROGRESS \
            "waiting, with ingress and consumers running, until no live lease, processor row or spool row remains"
        wait_for_live_work_to_finish "${DATABASE_POD}"
        transition_state \
            QUEUES_DRAINED COMPLETED \
            "no live lease, processor row or spool row remains; unclaimable leftovers are failed after the consumers stop"
    else
        transition_state \
            QUEUES_DRAINED COMPLETED \
            "control plane already stopped by an earlier run; nothing can enqueue work"
    fi
    # The consumers go first: once they are gone nothing can turn a failed
    # step into a successor workflow, so the leftovers can be failed safely.
    transition_state \
        CONTROL_CONSUMERS_STOPPED IN_PROGRESS \
        "stopping CPU consumers"
    for deployment in "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    transition_state \
        CONTROL_CONSUMERS_STOPPED COMPLETED \
        "CPU consumers stopped"
    transition_state \
        UNCLAIMABLE_WORK_ABANDONED IN_PROGRESS \
        "failing the workflows and remote commands nothing may claim under DRAINING"
    if [[ "${CPU_RUNTIME_RUNNING}" != true ]]; then
        transition_state \
            UNCLAIMABLE_WORK_ABANDONED COMPLETED \
            "control plane already stopped by an earlier run; nothing can enqueue work"
    else
        # Re-picked: the worker is gone, so this is an ingress Pod now.
        DATABASE_POD="$(find_database_pod || true)"
        if [[ -z "${DATABASE_POD}" ]]; then
            [[ "${CPU_INGRESS_RUNNING}" != true ]] ||
                die "no CPU ingress pod remains to fail the unclaimable work"
            transition_state \
                UNCLAIMABLE_WORK_ABANDONED COMPLETED \
                "CPU ingress already stopped by an earlier run and the consumers are gone; no Pod reaches Aurora and nothing can enqueue work"
        elif [[ "${MODE}" == reset ]]; then
            abandon_unclaimable_work "${DATABASE_POD}"
            transition_state \
                UNCLAIMABLE_WORK_ABANDONED COMPLETED \
                "${ABANDONED_SUMMARY}"
        else
            assert_no_active_work "${DATABASE_POD}"
            transition_state \
                UNCLAIMABLE_WORK_ABANDONED COMPLETED \
                "no workflow or remote command remained after the drain; mode ${MODE} abandons nothing"
        fi
    fi
    transition_state \
        INGRESS_STOPPED IN_PROGRESS \
        "stopping CPU ingress"
    for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    transition_state \
        INGRESS_STOPPED COMPLETED \
        "CPU ingress stopped"
elif [[ "${SCOPE}" == gpu ]]; then
    transition_state \
        QUEUES_DRAINED IN_PROGRESS \
        "waiting for selected GPU cluster queues to drain"
    wait_for_shared_queues "${DATABASE_POD}"
    transition_state \
        QUEUES_DRAINED COMPLETED \
        "selected GPU cluster queues drained"
fi

transition_state \
    GPU_EXECUTORS_STOPPED IN_PROGRESS \
    "stopping GPU executors and node components"
for context in "${CLUSTER_CONTEXTS[@]}"; do
    for deployment in "${GPU_EXECUTOR_DEPLOYMENTS[@]}"; do
        scale_gpu_deployment_zero "${context}" "${deployment}"
    done
done
# No orphan pass here: for --scope all the work nothing may claim was failed
# in UNCLAIMABLE_WORK_ABANDONED, after the consumers stopped and before
# ingress did, and no CPU Pod can reach the store any more; for --scope gpu
# the drain above waited for every command of the selected clusters.
if [[ "${NODE_MODE}" != skip ]]; then
    for index in "${!CLUSTER_IDS[@]}"; do
        run_node_cleanup "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
fi
transition_state \
    GPU_EXECUTORS_STOPPED COMPLETED \
    "GPU executors and node components stopped"

if [[ "${SCOPE}" == all ]]; then
    transition_state \
        CPU_AUXILIARIES_STOPPED IN_PROGRESS \
        "stopping CPU auxiliary components"
    for deployment in "${CPU_AUX_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    for cronjob in "${CPU_CRONJOBS[@]}"; do
        if cpu_kubectl -n "${NAMESPACE}" get cronjob \
            "${cronjob}" >/dev/null 2>&1; then
            cpu_kubectl -n "${NAMESPACE}" patch cronjob \
                "${cronjob}" --type=merge \
                -p '{"spec":{"suspend":true}}'
        fi
    done
    for daemonset in "${CPU_DAEMONSETS[@]}"; do
        cpu_kubectl -n "${NAMESPACE}" delete daemonset \
            "${daemonset}" --ignore-not-found
    done
    transition_state \
        CPU_AUXILIARIES_STOPPED COMPLETED \
        "CPU auxiliary components stopped"
fi

if [[ "${MODE}" == clean || "${MODE}" == reset ]]; then
    transition_state \
        APPLICATION_OBJECTS_DELETED IN_PROGRESS \
        "deleting registered application objects"
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        clean_plane_objects "${context}" GPU_DELETE_RESOURCES
    done
    if [[ "${SCOPE}" == all ]]; then
        clean_plane_objects "" CPU_DELETE_RESOURCES
    fi
    transition_state \
        APPLICATION_OBJECTS_DELETED COMPLETED \
        "registered application objects deleted"
fi

if [[ "${MODE}" == reset ]]; then
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        clear_node_metadata "${context}"
    done
    transition_state \
        NAMESPACES_DELETED IN_PROGRESS \
        "deleting NLB Service and solution namespaces"
    delete_control_plane_nlb_service
    delete_solution_namespaces
    transition_state \
        NAMESPACES_DELETED COMPLETED \
        "NLB Service and solution namespaces deleted"
fi

for context in "${CLUSTER_CONTEXTS[@]}"; do
    verify_gpu_stopped "${context}"
done
if [[ "${SCOPE}" == all ]]; then
    for deployment in "${CPU_DEPLOYMENTS[@]}"; do
        replicas="$(deployment_replicas_cpu "${deployment}")"
        [[ -z "${replicas}" || "${replicas}" == 0 ]] ||
            die "CPU deployment ${deployment} still has ${replicas} replicas"
    done
fi

transition_state \
    CLEANUP_COMPLETED COMPLETED \
    "Kubernetes and node cleanup completed"
STATE_COMPLETE=true

log "cleanup completed"
if [[ "${MODE}" == reset ]]; then
    log "preserved CPU EKS and all GPU EKS clusters"
    log "deleted solution namespaces and the Kubernetes NLB Service"
    log "delete NLB/IAM attachments, then mark READY_TO_DELETE_AURORA using manual section 0.3"
else
    log "preserved Aurora, NLB, VPC, IAM, Secrets, release ConfigMaps, and audit records"
fi
log "state file: ${STATE_FILE}"
