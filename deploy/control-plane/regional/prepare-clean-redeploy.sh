#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RESOURCE_INVENTORY="${SCRIPT_DIR}/cleanup-inventory.json"
CLEANUP_STATE_TOOL="$(
    printf '%s' \
        "${REPO_ROOT}/deploy/control-plane/tools/cleanup_state.py"
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
    python3 - "${CONFIG}" "${EFFECTIVE_INVENTORY}" \
        "${SELECTED_CLUSTER_IDS[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
selected = sys.argv[3:]
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid regional release config: {exc}")
try:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid cleanup resource inventory: {exc}")
unregistered = inventory.get("unregistered_resources") or []
if unregistered:
    details = ", ".join(
        f"{item['context']}:{item.get('namespace') or '_cluster'}:"
        f"{item['kind']}/{item['name']}"
        for item in unregistered
    )
    raise SystemExit(
        "unregistered live gpu-fault resources require "
        "classification before cleanup: " + details
    )

cpu_kubeconfig = value.get("cpu_kubeconfig")
namespace = value.get("namespace", "gpu-fault-system")
clusters = value.get("clusters")
if not isinstance(cpu_kubeconfig, str) or not cpu_kubeconfig:
    raise SystemExit("cpu_kubeconfig must be a non-empty string")
if not isinstance(namespace, str) or not re.fullmatch(
    r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace
):
    raise SystemExit("namespace is not a valid DNS label")
if not isinstance(clusters, list) or not clusters:
    raise SystemExit("clusters must be a non-empty list")

by_id = {}
for item in clusters:
    if not isinstance(item, dict):
        raise SystemExit("each clusters entry must be an object")
    cluster_id = item.get("cluster_id")
    context = item.get("context")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise SystemExit("each cluster_id must be a non-empty string")
    if not isinstance(context, str) or not context:
        raise SystemExit(f"cluster {cluster_id!r} has no context")
    if any(character in cluster_id + context for character in "\t\r\n"):
        raise SystemExit("cluster_id and context may not contain tabs or newlines")
    if cluster_id in by_id:
        raise SystemExit(f"duplicate cluster_id: {cluster_id}")
    by_id[cluster_id] = context

if selected:
    if len(selected) != len(set(selected)):
        raise SystemExit("duplicate --cluster-id")
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise SystemExit("unknown cluster_id: " + ", ".join(missing))
    cluster_ids = selected
else:
    cluster_ids = list(by_id)

print(f"META\t{namespace}\t{cpu_kubeconfig}")
for cluster_id in cluster_ids:
    print(f"CLUSTER\t{cluster_id}\t{by_id[cluster_id]}")

if inventory.get("schema_version") != 1:
    raise SystemExit("cleanup resource inventory schema_version must be 1")
valid_scopes = {"namespaced", "cluster"}
valid_clean = {"delete", "namespace", "reset"}
valid_phases = {
    "ingress",
    "consumer",
    "auxiliary",
    "producer",
    "executor",
    "support",
    "nlb",
}
for plane in ("cpu", "gpu"):
    section = inventory.get(plane)
    if not isinstance(section, dict):
        raise SystemExit(f"cleanup resource inventory lacks {plane}")
    resources = section.get("resources")
    if not isinstance(resources, list) or not resources:
        raise SystemExit(f"cleanup resource inventory {plane}.resources is empty")
    seen = set()
    for resource in resources:
        if not isinstance(resource, dict):
            raise SystemExit(f"{plane} resource entry must be an object")
        kind = resource.get("kind")
        name = resource.get("name")
        scope = resource.get("scope")
        phase = resource.get("phase")
        clean = resource.get("clean")
        fields = (kind, name, scope, phase, clean)
        if not all(isinstance(field, str) and field for field in fields):
            raise SystemExit(f"{plane} resource entry has an empty field")
        if any(character in "".join(fields) for character in "\t\r\n"):
            raise SystemExit(f"{plane} resource fields may not contain whitespace controls")
        if not re.fullmatch(r"[a-z][a-z0-9]*", kind):
            raise SystemExit(f"invalid kubectl resource kind: {kind}")
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name):
            raise SystemExit(f"invalid resource name: {name}")
        if scope not in valid_scopes:
            raise SystemExit(f"invalid resource scope: {scope}")
        if phase not in valid_phases:
            raise SystemExit(f"invalid cleanup phase: {phase}")
        if clean not in valid_clean:
            raise SystemExit(f"invalid cleanup action: {clean}")
        identity = (kind, name)
        if identity in seen:
            raise SystemExit(f"duplicate {plane} resource: {kind}/{name}")
        seen.add(identity)
        print(
            "RESOURCE",
            plane,
            kind,
            name,
            scope,
            phase,
            clean,
            sep="\t",
        )
    if plane == "cpu":
        preferences = section.get("database_pod_preference")
        if not isinstance(preferences, list) or not preferences:
            raise SystemExit("cpu.database_pod_preference is empty")
        for name in preferences:
            if ("deployment", name) not in seen:
                raise SystemExit(
                    f"database pod preference is not a CPU deployment: {name}"
                )
            print("DBPREF", name, sep="\t")
    else:
        annotations = section.get("node_annotations")
        if not isinstance(annotations, list) or not annotations:
            raise SystemExit("gpu.node_annotations is empty")
        for annotation in annotations:
            if not isinstance(annotation, str) or not annotation.startswith(
                "gpu-fault.io/"
            ):
                raise SystemExit(f"invalid node annotation: {annotation!r}")
            print("ANNOTATION", annotation, sep="\t")
PY
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
                    CPU_DELETE_RESOURCES+=("${resource_record}")
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
                    GPU_DELETE_RESOURCES+=("${resource_record}")
                fi
            fi
            ;;
        DBPREF)
            CPU_DATABASE_POD_PREFERENCE+=("${first}")
            ;;
        ANNOTATION)
            GPU_NODE_ANNOTATIONS+=("${first}")
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
    cat <<'EOF'
Ordered plan:
  1. Verify every Kubernetes context and capture the current replica/resource
     state in a mode-0600 TSV file.
  2. Fail closed if Aurora contains an active workflow or open remote command.
  3. Stop every GPU producer Deployment and DaemonSet found in the live
     installed-resource registries.
  4. Restore any quiesced host services, then stop or uninstall all node
     collectors, Node Agent, certificate timer, DCGM exporter, and GPU
     persistence service.
EOF
    if [[ "${SCOPE}" == all ]]; then
        cat <<'EOF'
  5. Scale registered CPU ingress Deployments to zero, wait for processor and
     telemetry spool rows to drain, then stop registered consumers.
  6. Stop each registered GPU executor only after remote commands drain.
  7. Stop ADOT, suspend Aurora credential refresh, and remove the control-plane
     sysctl DaemonSet.
EOF
    else
        cat <<'EOF'
  5. Keep the CPU control plane running while selected-cluster processor,
     spool, workflow, and remote-command rows drain.
  6. Stop the selected cluster's registered GPU executor.
EOF
    fi
    if [[ "${MODE}" == clean || "${MODE}" == reset ]]; then
        cat <<'EOF'
  8. Delete application Deployments, DaemonSets, PDBs, service accounts and
     RBAC in scope. Preserve Secrets, release ConfigMaps, Aurora, NLB, VPC,
     IAM, and audit records unless reset mode is selected.
EOF
    fi
    if [[ "${MODE}" == reset ]]; then
        cat <<'EOF'
  9. Delete the Kubernetes NLB Service, then delete the solution namespace
     in CPU EKS and every selected GPU EKS. Preserve all EKS clusters.
 10. Delete the dedicated Aurora cluster and solution-owned NLB security
     group, ACM certificate, private PKI Secret, Load Balancer Controller
     and IAM resources with the explicit commands in manual section 0.3.
EOF
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
    local name
    local pod
    for name in "${CPU_DATABASE_POD_PREFERENCE[@]}"; do
        pod="$(
            cpu_kubectl -n "${NAMESPACE}" get pod \
                -l "app=${name}" \
                --field-selector=status.phase=Running \
                -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
        )"
        if [[ -n "${pod}" ]]; then
            printf '%s\n' "${pod}"
            return 0
        fi
    done
    return 1
}

capture_fleet_inventory() {
    local pod=$1
    local missing
    FLEET_INVENTORY_FILE="$(mktemp)"
    cpu_kubectl -n "${NAMESPACE}" exec "${pod}" -- \
        python -c '
import json
import os
import sys

import psycopg

cluster_ids = set(sys.argv[1:])
with psycopg.connect(
    os.environ["GPU_FAULT_STORE_URL"],
    connect_timeout=10,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT payload
            FROM gpu_fault_objects
            WHERE kind='\''agent'\''
            ORDER BY key
            """
        )
        agents = [
            row[0]
            for row in cursor.fetchall()
            if row[0].get("cluster_id") in cluster_ids
        ]
print(json.dumps(agents, sort_keys=True))
' "${CLUSTER_IDS[@]}" >"${FLEET_INVENTORY_FILE}"
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
    local pod=$1
    local query_scope=$2
    shift 2
    cpu_kubectl -n "${NAMESPACE}" exec "${pod}" -- \
        env GPU_FAULT_CLEAN_QUERY_SCOPE="${query_scope}" \
        python -c '
import os
import sys

import psycopg

cluster_ids = sys.argv[1:]
scoped = os.environ["GPU_FAULT_CLEAN_QUERY_SCOPE"] == "gpu"
if scoped and not cluster_ids:
    raise SystemExit("scoped cleanup requires cluster ids")

def count(cursor, base, cluster_clause="", cluster_params=()):
    statement = base
    params = ()
    if scoped:
        statement += cluster_clause
        params = cluster_params
    try:
        cursor.execute(statement, params)
    except psycopg.errors.UndefinedTable:
        return 0
    return int(cursor.fetchone()[0])

with psycopg.connect(
    os.environ["GPU_FAULT_STORE_URL"],
    connect_timeout=10,
) as connection:
    with connection.cursor() as cursor:
        active_workflows = count(
            cursor,
            """
            SELECT count(*)
            FROM gpu_fault_objects AS workflow
            JOIN gpu_fault_objects AS incident
              ON incident.kind = '\''incident'\''
             AND incident.key = workflow.payload->>'\''incident_id'\''
            WHERE workflow.kind = '\''workflow'\''
              AND workflow.payload->>'\''status'\''
                  IN ('\''PENDING'\'', '\''RUNNING'\'', '\''SAFETY_PENDING'\'')
            """,
            " AND incident.payload->>'\''cluster_id'\'' = ANY(%s)",
            (cluster_ids,),
        )
        open_commands = count(
            cursor,
            """
            SELECT count(*)
            FROM gpu_fault_objects
            WHERE kind = '\''remote_command'\''
              AND payload->>'\''status'\''
                  IN ('\''PENDING'\'', '\''WAITING'\'', '\''LEASED'\'')
            """,
            " AND payload->>'\''cluster_id'\'' = ANY(%s)",
            (cluster_ids,),
        )
        processor_rows = count(
            cursor,
            """
            SELECT count(*)
            FROM gpu_fault_processor_queue
            WHERE status IN ('\''PENDING'\'', '\''LEASED'\'')
            """,
            " AND cluster_id = ANY(%s)",
            (cluster_ids,),
        )
        spool_rows = count(
            cursor,
            "SELECT count(*) FROM gpu_fault_telemetry_spool WHERE true",
            " AND cluster_id = ANY(%s)",
            (cluster_ids,),
        )
print(
    active_workflows,
    open_commands,
    processor_rows,
    spool_rows,
    sep="\t",
)
' "${CLUSTER_IDS[@]}"
}

assert_no_active_work() {
    local pod=$1
    local snapshot
    local active_workflows
    local open_commands
    local processor_rows
    local spool_rows
    snapshot="$(database_snapshot "${pod}" "${SCOPE}")"
    IFS=$'\t' read -r active_workflows open_commands \
        processor_rows spool_rows <<<"${snapshot}"
    [[ "${active_workflows}" =~ ^[0-9]+$ &&
        "${open_commands}" =~ ^[0-9]+$ ]] ||
        die "could not parse Aurora safety snapshot: ${snapshot}"
    if ((active_workflows != 0 || open_commands != 0)); then
        die "active workflows=${active_workflows}, open remote commands=${open_commands}; finish or block them before cleanup"
    fi
    log "safety snapshot: workflows=${active_workflows} remote_commands=${open_commands} processor_rows=${processor_rows} spool_rows=${spool_rows}"
}

wait_for_shared_queues() {
    local pod=$1
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local snapshot
    local active_workflows
    local open_commands
    local processor_rows
    local spool_rows
    while ((SECONDS < deadline)); do
        snapshot="$(database_snapshot "${pod}" "${SCOPE}")"
        IFS=$'\t' read -r active_workflows open_commands \
            processor_rows spool_rows <<<"${snapshot}"
        if [[ "${active_workflows}" == 0 && "${open_commands}" == 0 &&
            "${processor_rows}" == 0 && "${spool_rows}" == 0 ]]; then
            log "Aurora queues are drained"
            return 0
        fi
        log "waiting: workflows=${active_workflows} remote_commands=${open_commands} processor_rows=${processor_rows} spool_rows=${spool_rows}"
        sleep 5
    done
    die "Aurora queues did not drain within ${TIMEOUT_SECONDS}s"
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
        restore=/opt/gpu-fault/venv/bin/gpu-fault-restore-gpu-services
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
    if [[ "${NODE_MODE}" != skip ]]; then
        run_node_cleanup "${cluster_id}" "${context}"
    fi
}

clean_gpu_objects() {
    local context=$1
    local entry
    local kind
    local name
    local resource_scope
    for entry in "${GPU_DELETE_RESOURCES[@]}"; do
        IFS=$'\t' read -r kind name resource_scope <<<"${entry}"
        if [[ "${resource_scope}" == cluster ]]; then
            gpu_kubectl "${context}" delete "${kind}" "${name}" \
                --ignore-not-found
        else
            gpu_kubectl "${context}" -n "${NAMESPACE}" \
                delete "${kind}" "${name}" --ignore-not-found
        fi
    done
}

clean_cpu_objects() {
    local entry
    local kind
    local name
    local resource_scope
    for entry in "${CPU_DELETE_RESOURCES[@]}"; do
        IFS=$'\t' read -r kind name resource_scope <<<"${entry}"
        if [[ "${resource_scope}" == cluster ]]; then
            cpu_kubectl delete "${kind}" "${name}" --ignore-not-found
        else
            cpu_kubectl -n "${NAMESPACE}" \
                delete "${kind}" "${name}" --ignore-not-found
        fi
    done
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

clear_installer_annotations() {
    local context=$1
    local annotation
    local arguments=()
    for annotation in "${GPU_NODE_ANNOTATIONS[@]}"; do
        arguments+=("${annotation}-")
    done
    gpu_kubectl "${context}" annotate nodes --all --overwrite \
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

delete_solution_namespaces() {
    local context
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        gpu_kubectl "${context}" delete namespace "${NAMESPACE}" \
            --ignore-not-found --wait=true \
            --timeout="${TIMEOUT_SECONDS}s"
    done
    cpu_kubectl delete namespace "${NAMESPACE}" \
        --ignore-not-found --wait=true \
        --timeout="${TIMEOUT_SECONDS}s"
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

DATABASE_POD="$(find_database_pod || true)"
CPU_RUNTIME_PRESENT=false
for deployment in \
    "${CPU_INGRESS_DEPLOYMENTS[@]}" \
    "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
    if [[ -n "$(deployment_replicas_cpu "${deployment}")" ]]; then
        CPU_RUNTIME_PRESENT=true
        break
    fi
done
if [[ -n "${DATABASE_POD}" ]]; then
    capture_fleet_inventory "${DATABASE_POD}"
elif [[ "${MODE}" == reset ]]; then
    die "reset requires a running control-plane pod to export fleet inventory"
fi
if [[ -n "${DATABASE_POD}" ]]; then
    assert_no_active_work "${DATABASE_POD}"
elif [[ "${SCOPE}" == gpu || "${CPU_RUNTIME_PRESENT}" == true ]]; then
    die "no running CPU control-plane pod is available for the Aurora safety check"
else
    log "CPU runtime is already absent; Aurora runtime checks are not available"
fi
transition_state \
    PREFLIGHT COMPLETED \
    "contexts, quarantine state, inventory, and active work verified"

transition_state \
    GPU_DATA_PLANE_SOURCES_STOPPED IN_PROGRESS \
    "stopping GPU producers and node components"
for index in "${!CLUSTER_IDS[@]}"; do
    stop_gpu_producers \
        "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
done
transition_state \
    GPU_DATA_PLANE_SOURCES_STOPPED COMPLETED \
    "GPU producers and node components stopped"

if [[ "${SCOPE}" == all && "${CPU_RUNTIME_PRESENT}" == true ]]; then
    transition_state \
        INGRESS_STOPPED IN_PROGRESS \
        "stopping CPU ingress"
    for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    transition_state \
        INGRESS_STOPPED COMPLETED \
        "CPU ingress stopped"
    DATABASE_POD="$(find_database_pod || true)"
    [[ -n "${DATABASE_POD}" ]] ||
        die "no worker pod remains to verify the Aurora drain"
    transition_state \
        QUEUES_DRAINED IN_PROGRESS \
        "waiting for processor, spool, workflow, and remote command drain"
    wait_for_shared_queues "${DATABASE_POD}"
    transition_state \
        QUEUES_DRAINED COMPLETED \
        "processor, spool, workflow, and remote command state drained"
    transition_state \
        CONTROL_CONSUMERS_STOPPED IN_PROGRESS \
        "stopping CPU consumers"
    for deployment in "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    transition_state \
        CONTROL_CONSUMERS_STOPPED COMPLETED \
        "CPU consumers stopped"
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
    "stopping GPU executors"
for context in "${CLUSTER_CONTEXTS[@]}"; do
    for deployment in "${GPU_EXECUTOR_DEPLOYMENTS[@]}"; do
        scale_gpu_deployment_zero "${context}" "${deployment}"
    done
done
transition_state \
    GPU_EXECUTORS_STOPPED COMPLETED \
    "GPU executors stopped"

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
        clean_gpu_objects "${context}"
    done
    if [[ "${SCOPE}" == all ]]; then
        clean_cpu_objects
    fi
    transition_state \
        APPLICATION_OBJECTS_DELETED COMPLETED \
        "registered application objects deleted"
fi

if [[ "${MODE}" == reset ]]; then
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        clear_installer_annotations "${context}"
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
