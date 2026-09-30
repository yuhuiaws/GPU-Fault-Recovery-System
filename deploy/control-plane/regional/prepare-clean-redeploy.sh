#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESOURCE_INVENTORY="${SCRIPT_DIR}/cleanup-inventory.json"
CLEANUP_KUBERNETES_TOOL="${REPO_ROOT}/deploy/control-plane/tools/cleanup_kubernetes.py"
CLEANUP_PROBE="${REPO_ROOT}/deploy/control-plane/tools/cleanup_activity.py"
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
  --state-file FILE         Mode-0600 JSON audit file. Reuse only to resume
                            the same cleanup. Required with --execute.
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
RESUMING=false
RESUME_OUTPUT=
declare -A COMPLETED_PHASES=()
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
    if [[ "${EXECUTE}" == true && -e "${STATE_FILE}" ]]; then
        resume_arguments=(
            resume --path "${STATE_FILE}" --config "${CONFIG}"
            --scope "${SCOPE}" --mode "${MODE}" --node-mode "${NODE_MODE}"
            --inventory-output "${RUNTIME_INVENTORY_FILE}"
        )
        for cluster_id in "${SELECTED_CLUSTER_IDS[@]}"; do
            resume_arguments+=(--cluster-id "${cluster_id}")
        done
        RESUME_OUTPUT="$(python3 "${CLEANUP_STATE_TOOL}" "${resume_arguments[@]}")"
        RESUMING=true
        while read -r phase; do
            [[ -n "${phase}" ]] && COMPLETED_PHASES["${phase}"]=true
        done < <(python3 -c \
            'import json,sys; print(*json.load(sys.stdin)["completed_phases"], sep="\n")' \
            <<<"${RESUME_OUTPUT}")
        CURRENT_PHASE="$(python3 -c \
            'import json,sys; print(json.load(sys.stdin)["phase"])' <<<"${RESUME_OUTPUT}")"
        if [[ "${COMPLETED_PHASES[CLEANUP_COMPLETED]:-false}" == true ]]; then
            STATE_COMPLETE=true
        fi
    else
        # Collecting a cleanup plan must not rewrite registries outside the
        # selected mutation scope (or resurrect them during a retry).
        PYTHONDONTWRITEBYTECODE=1 python3 \
            "${REPO_ROOT}/deploy/control-plane/tools/collect_installed_resource_registry.py" \
            --config "${CONFIG}" --inventory "${RESOURCE_INVENTORY}" \
            >"${RUNTIME_INVENTORY_FILE}"
    fi
    EFFECTIVE_INVENTORY="${RUNTIME_INVENTORY_FILE}"
fi

CONFIG_OUTPUT="$(
    python3 - "${CONFIG}" "${EFFECTIVE_INVENTORY}" "${SCOPE}" \
        "${SELECTED_CLUSTER_IDS[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
scope = sys.argv[3]
selected = sys.argv[4:]
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid regional release config: {exc}")
try:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid cleanup resource inventory: {exc}")
cpu_kubeconfig = value.get("cpu_kubeconfig")
namespace = value.get("namespace", "gpu-fault-system")
clusters = value.get("clusters")
if not isinstance(cpu_kubeconfig, str) or not cpu_kubeconfig:
    raise SystemExit("cpu_kubeconfig must be a non-empty string")
if not isinstance(namespace, str) or not re.fullmatch(
    r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace
):
    raise SystemExit("namespace is not a valid DNS label")
if not isinstance(clusters, list):
    raise SystemExit("clusters must be a list")

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

contexts = {by_id[key] for key in cluster_ids}
if (
    len(by_id) > 1
    and str(inventory.get("generated_by", "")).endswith("collect_installed_resource_registry.py")
    and not contexts.issubset((inventory.get("gpu") or {}).get("by_context", {}))
):
    raise SystemExit("live cleanup inventory lacks per-context GPU resource ownership")
unregistered = [
    item for item in inventory.get("unregistered_resources") or []
    if scope == "all" or item.get("plane") == "gpu" and item.get("context") in contexts
]
if unregistered:
    details = ", ".join(
        f"{item['context']}:{item.get('namespace') or '_cluster'}:"
        f"{item['kind']}/{item['name']}"
        for item in unregistered
    )
    raise SystemExit(
        "unregistered live gpu-fault resources require classification before cleanup: " + details
    )

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
    if not isinstance(resources, list):
        raise SystemExit(f"cleanup resource inventory {plane}.resources is invalid")
    seen = set()
    for resource in resources:
        if not isinstance(resource, dict):
            raise SystemExit(f"{plane} resource entry must be an object")
        kind = resource.get("kind")
        name = resource.get("name")
        scope = resource.get("scope")
        phase = resource.get("phase")
        clean = resource.get("clean")
        guarded_delete = resource.get("guarded_delete", "")
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
        if (
            not isinstance(guarded_delete, str)
            or guarded_delete not in {"", "workload-rbac"}
            or guarded_delete and (
                plane != "gpu" or kind not in {"role", "rolebinding"}
                or scope != "namespaced" or phase != "support" or clean != "delete"
            )
        ):
            raise SystemExit("invalid guarded cleanup resource")
        resource_namespace = resource.get("namespace") or namespace
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", resource_namespace):
            raise SystemExit("resource namespace is invalid")
        identity = (scope, resource_namespace if scope == "namespaced" else "", kind, name)
        if identity in seen:
            raise SystemExit(f"duplicate {plane} resource: {kind}/{name}")
        seen.add(identity)
        resource_contexts = ["cpu"] if plane == "cpu" else [by_id[key] for key in cluster_ids]
        for resource_context in resource_contexts:
            per_context = section.get("by_context", {}).get(resource_context)
            if per_context is not None and not any(
                (item["scope"], item.get("namespace") or namespace, item["kind"], item["name"])
                == (scope, resource_namespace, kind, name)
                for item in per_context["resources"]
            ):
                continue
            if guarded_delete and (
                not isinstance(per_context, dict)
                or not isinstance(per_context.get("workload_rbac"), dict)
            ):
                raise SystemExit("guarded workload RBAC lacks context-specific proof")
            if per_context is not None and any(
                (item["scope"], item.get("namespace") or namespace, item["kind"], item["name"])
                == (scope, resource_namespace, kind, name)
                and item.get("guarded_delete", "") != guarded_delete
                for item in per_context["resources"]
            ):
                raise SystemExit("context-specific cleanup guard differs from inventory")
            print(
                "RESOURCE", plane, kind, name, scope, phase, clean,
                resource_namespace, resource_context, guarded_delete, sep="\t",
            )
    if plane == "cpu":
        preferences = section.get("database_pod_preference")
        if not isinstance(preferences, list):
            raise SystemExit("cpu.database_pod_preference is invalid")
        for name in preferences:
            if ("namespaced", namespace, "deployment", name) not in seen:
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
        labels = section.get("node_labels")
        if not isinstance(labels, list) or not labels:
            raise SystemExit("gpu.node_labels is empty")
        for label in labels:
            if not isinstance(label, str) or not label.startswith("gpu-fault.io/"):
                raise SystemExit(f"invalid node label: {label!r}")
            print("LABEL", label, sep="\t")
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
GPU_NODE_LABELS=()
while IFS=$'\t' read -r record first second third fourth fifth sixth seventh eighth ninth; do
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
            resource_record="${second}"$'\t'"${third}"$'\t'"${fourth}"$'\t'"${seventh}"$'\t'"${eighth}"
            resource_target="${third}"$'\t'"${seventh}"$'\t'"${eighth}"
            if [[ "${first}" == cpu ]]; then
                CPU_RESOURCES+=("${resource_record}")
                if [[ "${second}" == deployment ]]; then
                    CPU_DEPLOYMENTS+=("${resource_target}")
                    case "${fifth}" in
                        ingress)
                            CPU_INGRESS_DEPLOYMENTS+=("${resource_target}")
                            ;;
                        consumer)
                            CPU_CONSUMER_DEPLOYMENTS+=("${resource_target}")
                            ;;
                        auxiliary)
                            CPU_AUX_DEPLOYMENTS+=("${resource_target}")
                            ;;
                    esac
                elif [[ "${second}" == daemonset ]]; then
                    CPU_DAEMONSETS+=("${resource_target}")
                elif [[ "${second}" == cronjob ]]; then
                    CPU_CRONJOBS+=("${resource_target}")
                fi
                if [[ "${sixth}" == delete && -z "${ninth}" ]]; then
                    CPU_DELETE_RESOURCES+=("${resource_record}")
                elif [[ "${fifth}" == nlb ]]; then
                    CPU_NLB_SERVICES+=("${resource_target}")
                fi
            else
                GPU_RESOURCES+=("${resource_record}")
                if [[ "${second}" == deployment ]]; then
                    GPU_DEPLOYMENTS+=("${resource_target}")
                    if [[ "${fifth}" == producer ]]; then
                        GPU_PRODUCER_DEPLOYMENTS+=("${resource_target}")
                    elif [[ "${fifth}" == executor ]]; then
                        GPU_EXECUTOR_DEPLOYMENTS+=("${resource_target}")
                    fi
                elif [[ "${second}" == daemonset ]]; then
                    GPU_DAEMONSETS+=("${resource_target}")
                fi
                if [[ "${sixth}" == delete && -z "${ninth}" ]]; then
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
     state in a mode-0600 JSON file bound to the cleanup phase order.
  2. Fail closed if Aurora contains an active workflow or open remote command.
EOF
    if [[ "${SCOPE}" == all ]]; then
        printf '     Publish DRAINING for all GPU clusters in one fleet-ACKed registry revision.\n'
    fi
    cat <<'EOF'
  3. Stop every GPU producer Deployment and DaemonSet found in the live
     installed-resource registries.
EOF
    if [[ "${SCOPE}" == all ]]; then
        cat <<'EOF'
  4. Keep CPU ingress and consumers running until workflow, remote-command,
     processor and telemetry spool rows are all drained; leftovers block cleanup.
  5. Stop registered CPU consumers, then scale CPU ingress Deployments to zero.
  6. Stop GPU Executors, then restore quiesced host services and stop or uninstall
     node collectors, Node Agent, certificate timer, DCGM exporter and persistence.
  7. Stop ADOT, suspend Aurora credential refresh, and remove the control-plane
     sysctl DaemonSet.
EOF
    else
        cat <<'EOF'
  4. Keep the CPU control plane running while selected-cluster processor,
     spool, workflow, and remote-command rows drain.
  5. Stop the selected clusters' GPU Executors, then restore quiesced host services
     and stop or uninstall their node components.
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
if [[ "${RESUMING}" != true ]]; then
    init_arguments=(
        init --path "${STATE_FILE}" --config "${CONFIG}"
        --inventory "${EFFECTIVE_INVENTORY}" --scope "${SCOPE}"
        --mode "${MODE}" --node-mode "${NODE_MODE}"
    )
    for cluster_id in "${SELECTED_CLUSTER_IDS[@]}"; do
        init_arguments+=(--cluster-id "${cluster_id}")
    done
    python3 "${CLEANUP_STATE_TOOL}" "${init_arguments[@]}" >/dev/null
fi
STATE_INITIALIZED=true

cpu_kubectl() {
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --context cpu \
        --timeout-seconds "${KUBECTL_COMMAND_TIMEOUT:-${TIMEOUT_SECONDS}}" \
        command -- "$@"
}

gpu_kubectl() {
    local context=$1
    shift
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --context "gpu:${context}" \
        --timeout-seconds "${KUBECTL_COMMAND_TIMEOUT:-${TIMEOUT_SECONDS}}" \
        command -- "$@"
}

parse_target() {
    IFS=$'\t' read -r TARGET_NAME TARGET_NAMESPACE TARGET_CONTEXT <<<"$1"
}

phase_complete() {
    [[ "${COMPLETED_PHASES[$1]:-false}" == true ]]
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
    if [[ "${status}" == COMPLETED ]]; then
        COMPLETED_PHASES["${phase}"]=true
    fi
}

capture_cpu_state() {
    local name
    local value
    for name in "${CPU_DEPLOYMENTS[@]}"; do
        parse_target "${name}"
        value="$(cpu_kubectl -n "${TARGET_NAMESPACE}" get deployment "${TARGET_NAME}" \
            --ignore-not-found -o jsonpath='{.spec.replicas}')"
        if [[ -n "${value}" ]]; then
            record_state cpu "${CPU_KUBECONFIG}" deployment \
                "${TARGET_NAMESPACE}/${TARGET_NAME}" "${value}"
        fi
    done
    for name in "${CPU_DAEMONSETS[@]}"; do
        parse_target "${name}"
        value="$(cpu_kubectl -n "${TARGET_NAMESPACE}" get daemonset \
            "${TARGET_NAME}" --ignore-not-found -o name)"
        if [[ -n "${value}" ]]; then
            record_state cpu "${CPU_KUBECONFIG}" daemonset \
                "${TARGET_NAMESPACE}/${TARGET_NAME}" present
        fi
    done
    for name in "${CPU_CRONJOBS[@]}"; do
        parse_target "${name}"
        value="$(cpu_kubectl -n "${TARGET_NAMESPACE}" get cronjob \
            "${TARGET_NAME}" --ignore-not-found -o jsonpath='{.spec.suspend}')"
        if [[ -n "${value}" ]]; then
            record_state cpu "${CPU_KUBECONFIG}" cronjob \
                "${TARGET_NAMESPACE}/${TARGET_NAME}" "${value}"
        fi
    done
}

capture_gpu_state() {
    local cluster_id=$1
    local context=$2
    local name
    local value
    for name in "${GPU_DEPLOYMENTS[@]}"; do
        parse_target "${name}"
        [[ "${TARGET_CONTEXT}" == "${context}" ]] || continue
        value="$(gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
            get deployment "${TARGET_NAME}" --ignore-not-found -o jsonpath='{.spec.replicas}')"
        if [[ -n "${value}" ]]; then
            record_state "gpu:${cluster_id}" "${context}" deployment \
                "${TARGET_NAMESPACE}/${TARGET_NAME}" "${value}"
        fi
    done
    for name in "${GPU_DAEMONSETS[@]}"; do
        parse_target "${name}"
        [[ "${TARGET_CONTEXT}" == "${context}" ]] || continue
        value="$(gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
            get daemonset "${TARGET_NAME}" --ignore-not-found -o name)"
        if [[ -n "${value}" ]]; then
            record_state "gpu:${cluster_id}" "${context}" daemonset \
                "${TARGET_NAMESPACE}/${TARGET_NAME}" present
        fi
    done
}

deployment_replicas_cpu() {
    parse_target "$1"
    cpu_kubectl -n "${TARGET_NAMESPACE}" get deployment "${TARGET_NAME}" \
        --ignore-not-found -o jsonpath='{.spec.replicas}'
}

find_database_pod() {
    # Ready and not terminating: a Running Pod already being deleted (a roll
    # the previous command started) completes before the exec reaches it.
    local name
    local pod
    for name in "${CPU_DATABASE_POD_PREFERENCE[@]}"; do
        pod="$(
            cpu_kubectl -n "${NAMESPACE}" get pod -l "app=${name}" \
                --field-selector=status.phase=Running -o json |
                python3 -c '
import json
import sys

document = json.load(sys.stdin)
if not isinstance(document, dict) or not isinstance(document.get("items"), list):
    raise SystemExit("invalid control-plane Pod list")
for item in document["items"]:
    metadata = item.get("metadata")
    status = item.get("status")
    if not isinstance(metadata, dict) or not metadata.get("name") or not isinstance(status, dict):
        raise SystemExit("invalid control-plane Pod identity or status")
    containers = status.get("containerStatuses")
    if (
        not metadata.get("deletionTimestamp")
        and status.get("phase") == "Running"
        and isinstance(containers, list)
        and containers
        and all(isinstance(member, dict) and member.get("ready") is True for member in containers)
    ):
        print(metadata["name"])
        break
'
        )" || return
        if [[ -n "${pod}" ]]; then
            printf '%s\n' "${pod}"
            return 0
        fi
    done
    return 0
}

wait_for_cpu_rollouts() { # a role still rolling would hand us terminating Pods
    local deployment
    local replicas
    for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}" "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
        parse_target "${deployment}"
        [[ "${TARGET_CONTEXT}" == cpu ]] || die "invalid CPU rollout context"
        replicas="$(deployment_replicas_cpu "${deployment}")" || return
        [[ -n "${replicas}" ]] || continue
        [[ "${replicas}" =~ ^[0-9]+$ ]] ||
            die "invalid replicas for control-plane deployment ${TARGET_NAME}"
        ((replicas > 0)) || continue
        cpu_kubectl -n "${TARGET_NAMESPACE}" rollout status "deployment/${TARGET_NAME}" \
            --timeout="${TIMEOUT_SECONDS}s" >/dev/null ||
            die "control-plane deployment ${TARGET_NAME} did not settle before cleanup"
    done
}

capture_fleet_inventory() {
    local pod=$1
    local missing
    FLEET_INVENTORY_FILE="$(mktemp)"
    cpu_kubectl -n "${NAMESPACE}" exec "${pod}" -- \
        python -c "$(cat "${CLEANUP_PROBE}")" fleet "${SCOPE}" \
        "${CLUSTER_IDS[@]}" >"${FLEET_INVENTORY_FILE}"
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
        python -c "$(cat "${CLEANUP_PROBE}")" counts "${query_scope}" "${CLUSTER_IDS[@]}"
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
        "${open_commands}" =~ ^[0-9]+$ &&
        "${processor_rows}" =~ ^[0-9]+$ &&
        "${spool_rows}" =~ ^[0-9]+$ ]] ||
        die "could not parse Aurora safety snapshot"
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
        snapshot="$(KUBECTL_COMMAND_TIMEOUT=$((deadline - SECONDS)) \
            database_snapshot "${pod}" "${SCOPE}")"
        ((SECONDS < deadline)) || break
        IFS=$'\t' read -r active_workflows open_commands \
            processor_rows spool_rows <<<"${snapshot}"
        [[ "${active_workflows}" =~ ^[0-9]+$ &&
            "${open_commands}" =~ ^[0-9]+$ &&
            "${processor_rows}" =~ ^[0-9]+$ &&
            "${spool_rows}" =~ ^[0-9]+$ ]] ||
            die "could not parse Aurora drain snapshot"
        if [[ "${active_workflows}" == 0 && "${open_commands}" == 0 &&
            "${processor_rows}" == 0 && "${spool_rows}" == 0 ]]; then
            log "Aurora queues are drained"
            return 0
        fi
        log "waiting: workflows=${active_workflows} remote_commands=${open_commands} processor_rows=${processor_rows} spool_rows=${spool_rows}"
        sleep "$((deadline - SECONDS < 5 ? deadline - SECONDS : 5))"
    done
    die "Aurora queues did not drain within ${TIMEOUT_SECONDS}s"
}

scale_cpu_deployment_zero() {
    local name=$1
    local replicas
    replicas="$(deployment_replicas_cpu "${name}")"
    [[ -n "${replicas}" ]] || return 0
    parse_target "${name}"
    [[ "${replicas}" =~ ^[0-9]+$ ]] || die "invalid CPU deployment replica count"
    cpu_kubectl -n "${TARGET_NAMESPACE}" scale "deployment/${TARGET_NAME}" --replicas=0
    cpu_kubectl -n "${TARGET_NAMESPACE}" wait --for=delete pod \
        -l "app=${TARGET_NAME}" --timeout="${TIMEOUT_SECONDS}s"
}

scale_gpu_deployment_zero() {
    local context=$1
    local name=$2
    local replicas
    parse_target "${name}"
    [[ "${TARGET_CONTEXT}" == "${context}" ]] || return 0
    replicas="$(
        gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
            get deployment "${TARGET_NAME}" --ignore-not-found \
            -o jsonpath='{.spec.replicas}'
    )"
    [[ -n "${replicas}" ]] || return 0
    [[ "${replicas}" =~ ^[0-9]+$ ]] || die "invalid GPU deployment replica count"
    gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
        scale "deployment/${TARGET_NAME}" --replicas=0
    gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
        wait --for=delete pod -l "app=${TARGET_NAME}" \
        --timeout="${TIMEOUT_SECONDS}s"
}

run_node_cleanup() {
    local cluster_id=$1
    local context=$2
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

for unit in ${units}; do
    case "${unit}" in
        *[!A-Za-z0-9_.@-]*)
            echo "invalid installed unit name" >&2
            exit 1
            ;;
        gpu-fault-*.service | gpu-fault-*.timer) ;;
        *)
            echo "invalid installed unit name" >&2
            exit 1
            ;;
    esac
done

if [ "${mode}" = uninstall ] && [ -x /opt/gpu-fault/uninstall ]; then
    /opt/gpu-fault/uninstall
else
    if [ "${mode}" = uninstall ]; then
        for path in /opt/gpu-fault /etc/gpu-fault /var/lib/gpu-fault; do
            if [ -e "${path}" ] || [ -L "${path}" ]; then
                echo "node runtime remains but its uninstaller is unavailable" >&2
                exit 1
            fi
        done
    fi
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
    active_status=0
    systemctl is-active --quiet "${unit}" || active_status=$?
    if [ "${active_status}" -eq 0 ]; then
        echo "unit remains active: ${unit}" >&2
        exit 1
    elif [ "${active_status}" -ne 3 ] && [ "${active_status}" -ne 4 ]; then
        echo "cannot verify stopped unit: ${unit}" >&2
        exit 1
    fi
done
if [ "${mode}" = uninstall ]; then
    systemctl daemon-reload
    for path in /opt/gpu-fault /etc/gpu-fault /var/lib/gpu-fault; do
        if [ -e "${path}" ] || [ -L "${path}" ]; then
            echo "node runtime files remain after uninstall" >&2
            exit 1
        fi
    done
fi
HOST_SCRIPT
    )"
    host_script_b64="$(
        printf '%s\n' "${host_script}" | base64 | tr -d '\n'
    )"

    log "${cluster_id}: running owned node cleanup DaemonSet (${NODE_MODE})"
    GPU_FAULT_CLEANUP_HOST_SCRIPT_B64="${host_script_b64}" \
        python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --context "gpu:${context}" --cluster-id "${cluster_id}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" --image "${NODE_CLEANUP_IMAGE}" \
        --node-mode "${NODE_MODE}" node-cleanup
}

stop_gpu_producers() {
    local cluster_id=$1
    local context=$2
    local name
    log "${cluster_id}: stopping GPU producer workloads"
    for name in "${GPU_PRODUCER_DEPLOYMENTS[@]}"; do
        scale_gpu_deployment_zero "${context}" "${name}"
    done
    for name in "${GPU_DAEMONSETS[@]}"; do
        parse_target "${name}"
        [[ "${TARGET_CONTEXT}" == "${context}" ]] || continue
        gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" delete daemonset \
            "${TARGET_NAME}" --ignore-not-found --wait=true --timeout="${TIMEOUT_SECONDS}s"
    done
}

clean_gpu_objects() {
    local context=$1
    local entry
    local kind
    local name
    local resource_scope
    local resource_namespace
    local resource_context
    for entry in "${GPU_DELETE_RESOURCES[@]}"; do
        IFS=$'\t' read -r kind name resource_scope resource_namespace resource_context <<<"${entry}"
        [[ "${resource_context}" == "${context}" ]] || continue
        if [[ "${resource_scope}" == cluster ]]; then
            gpu_kubectl "${context}" delete "${kind}" "${name}" \
                --ignore-not-found --wait=true --timeout="${TIMEOUT_SECONDS}s"
        else
            gpu_kubectl "${context}" -n "${resource_namespace}" \
                delete "${kind}" "${name}" --ignore-not-found \
                --wait=true --timeout="${TIMEOUT_SECONDS}s"
        fi
    done
}

clean_cpu_objects() {
    local entry
    local kind
    local name
    local resource_scope
    local resource_namespace
    local resource_context
    for entry in "${CPU_DELETE_RESOURCES[@]}"; do
        IFS=$'\t' read -r kind name resource_scope resource_namespace resource_context <<<"${entry}"
        [[ "${resource_context}" == cpu ]] || die "invalid CPU resource context"
        if [[ "${resource_scope}" == cluster ]]; then
            cpu_kubectl delete "${kind}" "${name}" --ignore-not-found \
                --wait=true --timeout="${TIMEOUT_SECONDS}s"
        else
            cpu_kubectl -n "${resource_namespace}" \
                delete "${kind}" "${name}" --ignore-not-found \
                --wait=true --timeout="${TIMEOUT_SECONDS}s"
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

clear_node_metadata() {
    local context=$1
    local cluster_id=$2
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --context "gpu:${context}" --cluster-id "${cluster_id}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" clear-node-metadata
}

delete_control_plane_nlb_service() {
    local name
    for name in "${CPU_NLB_SERVICES[@]}"; do
        parse_target "${name}"
        cpu_kubectl -n "${TARGET_NAMESPACE}" delete service \
            "${TARGET_NAME}" --ignore-not-found --wait=true \
            --timeout="${TIMEOUT_SECONDS}s"
    done
}

delete_solution_namespaces() {
    local context
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        python3 "${CLEANUP_KUBERNETES_TOOL}" \
            --config "${CONFIG}" --state-file "${STATE_FILE}" \
            --context "gpu:${context}" --timeout-seconds "${TIMEOUT_SECONDS}" \
            delete-namespace
    done
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" delete-namespace
}

verify_gpu_stopped() {
    local context=$1
    local name
    local replicas
    for name in "${GPU_DEPLOYMENTS[@]}"; do
        parse_target "${name}"
        [[ "${TARGET_CONTEXT}" == "${context}" ]] || continue
        replicas="$(
            gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" \
                get deployment "${TARGET_NAME}" --ignore-not-found \
                -o jsonpath='{.spec.replicas}'
        )"
        [[ -z "${replicas}" || "${replicas}" == 0 ]] ||
            die "${context}: deployment ${name} still has ${replicas} replicas"
    done
    for name in "${GPU_DAEMONSETS[@]}"; do
        parse_target "${name}"
        [[ "${TARGET_CONTEXT}" == "${context}" ]] || continue
        replicas="$(gpu_kubectl "${context}" -n "${TARGET_NAMESPACE}" get daemonset \
            "${TARGET_NAME}" --ignore-not-found -o name)"
        if [[ -n "${replicas}" ]]; then
            die "${context}: daemonset ${TARGET_NAME} remains"
        fi
    done
}

run_phase() {
    local phase=$1
    shift
    if phase_complete "${phase}"; then
        return 0
    fi
    transition_state "${phase}" IN_PROGRESS "running ${phase}"
    "$@"
    transition_state "${phase}" COMPLETED "completed ${phase}"
}

require_database_pod() {
    DATABASE_POD="$(find_database_pod)"
    [[ -n "${DATABASE_POD}" ]] ||
        die "no running CPU pod is available for the Aurora safety check"
}

capture_preflight() {
    local index
    if [[ "${SCOPE}" == all ]]; then
        wait_for_cpu_rollouts
        capture_cpu_state
    fi
    for index in "${!CLUSTER_IDS[@]}"; do
        capture_gpu_state "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
    require_database_pod
    capture_fleet_inventory "${DATABASE_POD}"
    assert_no_active_work "${DATABASE_POD}"
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" capture
    for index in "${!CLUSTER_IDS[@]}"; do
        python3 "${CLEANUP_KUBERNETES_TOOL}" \
            --config "${CONFIG}" --state-file "${STATE_FILE}" \
            --context "gpu:${CLUSTER_CONTEXTS[index]}" \
            --cluster-id "${CLUSTER_IDS[index]}" \
            --timeout-seconds "${TIMEOUT_SECONDS}" capture
    done
    log_node_capture_exceptions
}

log_node_capture_exceptions() {
    # Nodes the capture proved safe to leave out of the fleet targets: agents
    # whose spot instance HyperPod no longer lists, and replacement nodes that
    # carry only unfinished installer annotations. Both are journaled in the
    # state file; this only surfaces them in the run output.
    local line
    local lines
    lines="$(
        python3 - "${STATE_FILE}" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
for entry in document.get("departed_fleet_nodes") or []:
    print(
        f"{entry.get('cluster_id')}: departed fleet node {entry.get('node_id')} "
        f"(instance {entry.get('node_instance_id')} left HyperPod); "
        "not a node cleanup target"
    )
for context, nodes in sorted((document.get("orphaned_installer_nodes") or {}).items()):
    for name in sorted(nodes):
        print(
            f"{context}: node {name} carries only unfinished installer "
            "annotations and no fleet agent; annotations are removed at reset"
        )
PY
    )"
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then
            log "${line}"
        fi
    done <<<"${lines}"
}

stop_all_gpu_producers() {
    local index
    for index in "${!CLUSTER_IDS[@]}"; do
        stop_gpu_producers "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
}

drain_registry_clusters() {
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" drain-registry
}

stop_ingress() {
    local deployment
    for deployment in "${CPU_INGRESS_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
}

drain_queues() {
    require_database_pod
    wait_for_shared_queues "${DATABASE_POD}"
}

stop_node_runtimes() {
    local index
    [[ "${NODE_MODE}" != skip ]] || return 0
    for index in "${!CLUSTER_IDS[@]}"; do
        if [[ "${SCOPE}" == gpu ]]; then
            drain_queues
        fi
        run_node_cleanup "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    done
}

stop_consumers() {
    local deployment
    for deployment in "${CPU_CONSUMER_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
}

stop_executors() {
    local context
    local deployment
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        for deployment in "${GPU_EXECUTOR_DEPLOYMENTS[@]}"; do
            scale_gpu_deployment_zero "${context}" "${deployment}"
        done
    done
}

stop_auxiliaries() {
    local deployment cronjob daemonset present
    for deployment in "${CPU_AUX_DEPLOYMENTS[@]}"; do
        scale_cpu_deployment_zero "${deployment}"
    done
    for cronjob in "${CPU_CRONJOBS[@]}"; do
        parse_target "${cronjob}"
        present="$(cpu_kubectl -n "${TARGET_NAMESPACE}" get cronjob \
            "${TARGET_NAME}" --ignore-not-found -o name)"
        if [[ -n "${present}" ]]; then
            cpu_kubectl -n "${TARGET_NAMESPACE}" patch cronjob \
                "${TARGET_NAME}" --type=merge \
                -p '{"spec":{"suspend":true}}'
        fi
    done
    for daemonset in "${CPU_DAEMONSETS[@]}"; do
        parse_target "${daemonset}"
        cpu_kubectl -n "${TARGET_NAMESPACE}" delete daemonset \
            "${TARGET_NAME}" --ignore-not-found --wait=true --timeout="${TIMEOUT_SECONDS}s"
    done
}

delete_application_objects() {
    local context index
    for index in "${!CLUSTER_IDS[@]}"; do
        python3 "${CLEANUP_KUBERNETES_TOOL}" \
            --config "${CONFIG}" --state-file "${STATE_FILE}" \
            --context "gpu:${CLUSTER_CONTEXTS[index]}" \
            --cluster-id "${CLUSTER_IDS[index]}" \
            --timeout-seconds "${TIMEOUT_SECONDS}" delete-workload-rbac
    done
    for context in "${CLUSTER_CONTEXTS[@]}"; do
        clean_gpu_objects "${context}"
    done
    if [[ "${SCOPE}" == all ]]; then
        clean_cpu_objects
    fi
}

reset_namespaces() {
    local index
    for index in "${!CLUSTER_IDS[@]}"; do
        clear_node_metadata "${CLUSTER_CONTEXTS[index]}" "${CLUSTER_IDS[index]}"
    done
    delete_control_plane_nlb_service
    delete_solution_namespaces
}

log "validating Kubernetes contexts"
cpu_kubectl get --raw=/readyz >/dev/null
for index in "${!CLUSTER_IDS[@]}"; do
    gpu_kubectl "${CLUSTER_CONTEXTS[index]}" get --raw=/readyz >/dev/null
    if [[ "${MODE}" == reset ]]; then
        assert_no_quarantined_nodes "${CLUSTER_IDS[index]}" "${CLUSTER_CONTEXTS[index]}"
    fi
done

if phase_complete PREFLIGHT; then
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --timeout-seconds "${TIMEOUT_SECONDS}" verify-targets
fi
if phase_complete CLEANUP_COMPLETED; then
    STATE_COMPLETE=true
else
    python3 "${CLEANUP_KUBERNETES_TOOL}" \
        --config "${CONFIG}" --state-file "${STATE_FILE}" \
        --timeout-seconds 90 cleanup-owned
    if [[ "${RESUMING}" == true && "${SCOPE}" == all ]] &&
        phase_complete CLUSTERS_DRAINING &&
        ! phase_complete CONTROL_CONSUMERS_STOPPED; then
        # A completed checkpoint does not prove admission or queue state
        # stayed unchanged while the cleanup process was absent.
        drain_registry_clusters
        if phase_complete QUEUES_DRAINED; then
            drain_queues
        fi
    fi
    run_phase PREFLIGHT capture_preflight
    if [[ "${SCOPE}" == all ]]; then
        run_phase CLUSTERS_DRAINING drain_registry_clusters
    fi
    run_phase GPU_DATA_PLANE_SOURCES_STOPPED stop_all_gpu_producers
    run_phase QUEUES_DRAINED drain_queues
    if [[ "${SCOPE}" == all ]]; then
        run_phase CONTROL_CONSUMERS_STOPPED stop_consumers
        run_phase INGRESS_STOPPED stop_ingress
    fi
    run_phase GPU_EXECUTORS_STOPPED stop_executors
    run_phase NODE_RUNTIMES_STOPPED stop_node_runtimes
    if [[ "${SCOPE}" == all ]]; then
        run_phase CPU_AUXILIARIES_STOPPED stop_auxiliaries
    fi
    if [[ "${MODE}" == clean || "${MODE}" == reset ]]; then
        run_phase APPLICATION_OBJECTS_DELETED delete_application_objects
    fi
    if [[ "${MODE}" == reset ]]; then
        run_phase NAMESPACES_DELETED reset_namespaces
    fi
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

if [[ "${STATE_COMPLETE}" != true ]]; then
    transition_state \
        CLEANUP_COMPLETED COMPLETED \
        "Kubernetes and node cleanup completed"
fi
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
