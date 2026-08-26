#!/usr/bin/env bash
# Reclaim accumulated wheel ConfigMaps.
#
# deploy/hyperpod/deploy.sh names each wheel ConfigMap
# gpu-fault-control-plane-wheel-<VERSION_TAG>-<sha256[:12]> and creates a new
# one whenever the wheel content changes, but never deletes the old ones. A
# development namespace therefore grows one ~440 KiB etcd object per build.
#
# Deleting a wheel ConfigMap that is still referenced breaks things in two
# non-obvious ways, so both are treated as references here:
#   - a workload template still naming it cannot start a new Pod, even at 0
#     replicas (gpu-fault-cluster-executor is deliberately kept at 0 on the
#     CPU control plane while the executor runs on the GPU cluster);
#   - an older ReplicaSet still naming it makes `kubectl rollout undo`
#     unusable, because the rolled-back Pod cannot mount its wheel.
#
# Safety rules:
#   - dry run unless GPU_FAULT_PRUNE_APPLY=true;
#   - only names matching the generated <tag>-<12 hex> shape are ever
#     considered, so hand-pinned experiment wheels such as
#     gpu-fault-control-plane-wheel-reboot-email-reason-0723 are untouchable;
#   - the KEEP_RECENT newest survivors are retained regardless of references,
#     so a rollback target always exists.
set -euo pipefail

NAMESPACE="${NAMESPACE:-gpu-fault-system}"
KEEP_RECENT="${GPU_FAULT_PRUNE_KEEP_RECENT:-3}"
APPLY="${GPU_FAULT_PRUNE_APPLY:-false}"
# Comma-separated names to keep regardless of references. deploy.sh passes the
# wheel it just built, which no workload names yet at prune time.
PROTECT="${GPU_FAULT_PRUNE_PROTECT:-}"

KUBECTL=(kubectl)
if [[ -n "${KUBECONFIG_PATH:-}" ]]; then
    KUBECTL+=(--kubeconfig "${KUBECONFIG_PATH}")
fi
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
    KUBECTL+=(--context "${KUBE_CONTEXT}")
fi

for command in kubectl python3.12; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done

if ! [[ "${KEEP_RECENT}" =~ ^[0-9]+$ ]] || (( KEEP_RECENT < 1 )); then
    printf 'ERROR: GPU_FAULT_PRUNE_KEEP_RECENT must be >= 1\n' >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

"${KUBECTL[@]}" -n "${NAMESPACE}" get configmap -o json \
    >"${WORK_DIR}/configmaps.json"
# Every object kind that can pin a wheel ConfigMap. replicaset is included for
# rollback safety; pod covers bare Pods and anything already running.
"${KUBECTL[@]}" -n "${NAMESPACE}" get \
    pod,replicaset,deployment,statefulset,daemonset,job,cronjob \
    -o json >"${WORK_DIR}/workloads.json"

PLAN_PATH="${WORK_DIR}/plan.txt"
PLAN="${PLAN_PATH}" \
KEEP_RECENT="${KEEP_RECENT}" \
PROTECT="${PROTECT}" \
WORK_DIR="${WORK_DIR}" \
python3.12 <<'PY'
import json
import os
import re

work_dir = os.environ["WORK_DIR"]
keep_recent = int(os.environ["KEEP_RECENT"])
protected_explicit = {
    item.strip()
    for item in os.environ.get("PROTECT", "").split(",")
    if item.strip()
}

# Only the shape deploy.sh generates. Anything else was pinned by hand.
generated = re.compile(
    r"^gpu-fault-(?:control-plane|executor)-wheel-[0-9a-z]+-[0-9a-f]{12}$"
)

with open(f"{work_dir}/configmaps.json") as handle:
    configmaps = json.load(handle)["items"]
with open(f"{work_dir}/workloads.json") as handle:
    workloads = json.load(handle)["items"]


def volume_sources(spec):
    for volume in (spec or {}).get("volumes") or []:
        config_map = volume.get("configMap")
        if config_map and config_map.get("name"):
            yield config_map["name"]
        projected = volume.get("projected") or {}
        for source in projected.get("sources") or []:
            config_map = source.get("configMap")
            if config_map and config_map.get("name"):
                yield config_map["name"]
    for container in (
        ((spec or {}).get("containers") or [])
        + ((spec or {}).get("initContainers") or [])
    ):
        for entry in container.get("envFrom") or []:
            config_map = entry.get("configMapRef")
            if config_map and config_map.get("name"):
                yield config_map["name"]


referenced = {}
for workload in workloads:
    kind = workload["kind"]
    name = workload["metadata"]["name"]
    spec = workload.get("spec") or {}
    specs = [spec if kind == "Pod" else None]
    template = spec.get("template") or {}
    specs.append(template.get("spec"))
    job_template = (spec.get("jobTemplate") or {}).get("spec") or {}
    specs.append((job_template.get("template") or {}).get("spec"))
    for pod_spec in specs:
        for config_map_name in volume_sources(pod_spec):
            referenced.setdefault(config_map_name, set()).add(
                f"{kind}/{name}"
            )

candidates = []
protected_manual = []
for config_map in configmaps:
    name = config_map["metadata"]["name"]
    if not any(
        marker in name
        for marker in ("control-plane-wheel", "executor-wheel")
    ):
        continue
    if not generated.match(name):
        protected_manual.append(name)
        continue
    candidates.append(
        (config_map["metadata"]["creationTimestamp"], name)
    )

candidates.sort()
recent = {name for _, name in candidates[-keep_recent:]}

lines = []
kept = []
for created, name in candidates:
    holders = sorted(referenced.get(name, ()))
    if name in protected_explicit:
        kept.append((name, created, "explicitly protected"))
    elif holders:
        kept.append((name, created, "referenced by " + ", ".join(holders)))
    elif name in recent:
        kept.append((name, created, f"within newest {keep_recent}"))
    else:
        lines.append(name)

print(f"namespace wheel ConfigMaps: {len(candidates)} generated, "
      f"{len(protected_manual)} hand-pinned (never pruned)")
print(f"keep: {len(kept)}")
for name, created, reason in kept:
    print(f"  KEEP   {name}  {created}  ({reason})")
print(f"prune: {len(lines)}")
for created, name in candidates:
    if name in lines:
        print(f"  PRUNE  {name}  {created}")

with open(os.environ["PLAN"], "w") as handle:
    handle.write("\n".join(lines))
    if lines:
        handle.write("\n")
PY

if [[ ! -s "${PLAN_PATH}" ]]; then
    printf '\nNothing to prune.\n'
    exit 0
fi

if [[ "${APPLY}" != "true" ]]; then
    printf '\nDry run. Re-run with GPU_FAULT_PRUNE_APPLY=true to delete.\n'
    exit 0
fi

while read -r name; do
    [[ -n "${name}" ]] || continue
    "${KUBECTL[@]}" -n "${NAMESPACE}" delete configmap "${name}"
done <"${PLAN_PATH}"
