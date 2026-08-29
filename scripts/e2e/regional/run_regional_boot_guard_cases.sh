#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIXTURE_DIR="${ROOT}/scripts/e2e/regional/boot_guard"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:=gpu-fault-system}"
: "${RUN_DIR:?}"
: "${CPU_HYPERPOD_CLUSTER:?}"
: "${AWS_REGION:?}"
: "${BOOT_GUARD_START_CASE:=1}"

if [[ "${BOOT_GUARD_START_CASE}" != "1" &&
      "${BOOT_GUARD_START_CASE}" != "7" &&
      "${BOOT_GUARD_START_CASE}" != "8" ]]; then
  echo "BOOT_GUARD_START_CASE must be 1, 7, or 8" >&2
  exit 2
fi

PROBE="gpu-fault-api-guard-probe"
BASE="${GUARD_PROBE_BASE:-/tmp/guard-probe-base.json}"
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CASE_DIR="${RUN_DIR}/cases"

install -d -m 0700 "${CASE_DIR}"
export CPU_KUBECONFIG NAMESPACE PROBE

if (( BOOT_GUARD_START_CASE > 1 )); then
  for case_number in $(seq 1 $((BOOT_GUARD_START_CASE - 1))); do
    printf -v case_id 'GF-REGIONAL-BOOT-%03d' "${case_number}"
    evidence="${CASE_DIR}/${case_id}.txt"
    if [[ ! -f "${evidence}" ]] || ! grep -qx "PASS" "${evidence}"; then
      echo "BOOT-007 resume requires prior PASS evidence: ${evidence}" >&2
      exit 2
    fi
  done
fi

latest_ready_api_pod() {
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l app=gpu-fault-api-ha -o json |
    jq -r '
      [
        .items[]
        | select(.metadata.deletionTimestamp == null)
        | select(any(
            .status.conditions[]?;
            .type == "Ready" and .status == "True"
          ))
      ]
      | sort_by(.metadata.creationTimestamp)
      | last
      | .metadata.name
    '
}

reset_probe() {
  "${FIXTURE_DIR}/reset.sh"
}

assert_probe() {
  "${FIXTURE_DIR}/assert.sh" "$1" 180
}

apply_mutation() {
  "${PYTHON}" "${FIXTURE_DIR}/mutate.py" "$@" |
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
}

probe_registry() {
  reset_probe
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    create secret generic gpu-fault-regional-clusters-probe \
    --from-literal=clusters.json="$(
      "${PYTHON}" "${FIXTURE_DIR}/registry.py" "$@"
    )" \
    --dry-run=client -o yaml |
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
  apply_mutation \
    sref GPU_FAULT_REGIONAL_CLUSTERS_JSON \
    gpu-fault-regional-clusters-probe clusters.json
}

cleanup_all() {
  set +e
  "${FIXTURE_DIR}/cleanup.sh" --drop-database \
    >"${CASE_DIR}/GF-REGIONAL-BOOT-001-010-cleanup.txt" 2>&1
  shred -u "${BASE}" 2>/dev/null || true
}
trap cleanup_all EXIT

baseline="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get deployment gpu-fault-api-ha \
    -o jsonpath='{.metadata.generation} {.status.readyReplicas}'
)"
printf 'baseline=%s\n' "${baseline}" |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-baseline.txt"

api_pod="$(latest_ready_api_pod)"
database_output="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    exec -i "${api_pod}" -- python - <<'PY'
import os
import urllib.parse

import psycopg

from gpu_fault.store import PostgresStore

parts = urllib.parse.urlsplit(os.environ["GPU_FAULT_STORE_URL"])
assert parts.path == "/gpu_fault", parts.path
admin = urllib.parse.urlunsplit(parts._replace(path="/postgres"))
with psycopg.connect(admin, autocommit=True) as connection:
    exists = connection.execute(
        "SELECT 1 FROM pg_database WHERE datname='gpu_fault_guardprobe'"
    ).fetchone()
    if not exists:
        connection.execute("CREATE DATABASE gpu_fault_guardprobe")
probe = urllib.parse.urlunsplit(
    parts._replace(path="/gpu_fault_guardprobe")
)
store = PostgresStore(
    probe,
    pool_min_size=0,
    pool_max_size=1,
    pool_timeout_seconds=10,
    initialize_schema=True,
    hot_state_mode="legacy",
)
store.close()
print("guardprobe schema initialized")
PY
)"
printf '%s\n' "${database_output}" |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-database.txt"
unset database_output

secret_manifest="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    exec -i "${api_pod}" -- python - <<'PY'
import base64
import json
import os
import urllib.parse

parts = urllib.parse.urlsplit(os.environ["GPU_FAULT_STORE_URL"])
assert parts.path == "/gpu_fault", parts.path
probe = urllib.parse.urlunsplit(
    parts._replace(path="/gpu_fault_guardprobe")
)
print(
    json.dumps(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "gpu-fault-aurora-guardprobe"},
            "type": "Opaque",
            "data": {
                "postgres-url": base64.b64encode(
                    probe.encode()
                ).decode()
            },
        }
    )
)
PY
)"
printf '%s\n' "${secret_manifest}" |
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
unset secret_manifest

"${FIXTURE_DIR}/derive.sh" "${BASE}"
export GUARD_PROBE_BASE="${BASE}"

if (( BOOT_GUARD_START_CASE <= 1 )); then
  reset_probe
  apply_mutation del GPU_FAULT_REGIONAL_CLUSTERS_JSON
  assert_probe \
    "regional mode requires GPU_FAULT_REGIONAL_CLUSTERS_JSON" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-001.txt"
  endpoints="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get endpoints gpu-fault-api \
      -o jsonpath='{range .subsets[*].addresses[*]}{.targetRef.name}{"\n"}{end}'
  )"
  printf '%s\n' "${endpoints}" \
    >>"${CASE_DIR}/GF-REGIONAL-BOOT-001.txt"
  if grep -q "gpu-fault-api-guard-probe" <<<"${endpoints}"; then
    echo "guard probe entered the production Service endpoints" >&2
    exit 1
  fi
fi

if (( BOOT_GUARD_START_CASE <= 2 )); then
  : >"${CASE_DIR}/GF-REGIONAL-BOOT-002.txt"
  for payload in '{"cluster_id":' '{}' '[1]'; do
    reset_probe
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      create secret generic gpu-fault-regional-clusters-bad \
      --from-literal=clusters.json="${payload}" \
      --dry-run=client -o yaml |
      kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
    apply_mutation \
      sref GPU_FAULT_REGIONAL_CLUSTERS_JSON \
      gpu-fault-regional-clusters-bad clusters.json
    if [[ "${payload}" == '{"cluster_id":' ]]; then
      expected="GPU_FAULT_REGIONAL_CLUSTERS_JSON is invalid"
    elif [[ "${payload}" == '{}' ]]; then
      expected="regional cluster registry must be a list"
    else
      expected="regional cluster registry entries must be objects"
    fi
    printf 'payload=%s\n' "${payload}" |
      tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-002.txt"
    assert_probe "${expected}" |
      tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-002.txt"
  done

  reset_probe
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    create secret generic gpu-fault-regional-clusters-bad \
    --from-literal=clusters.json='[]' \
    --dry-run=client -o yaml |
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
  apply_mutation \
    sref GPU_FAULT_REGIONAL_CLUSTERS_JSON \
    gpu-fault-regional-clusters-bad clusters.json
  probe_pod="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get pod -l "app=${PROBE}" -o jsonpath='{.items[0].metadata.name}'
  )"
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    wait --for=condition=ready "pod/${probe_pod}" --timeout=600s
  empty_registry_output="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      exec -i "${probe_pod}" -- python - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/regional/clusters",
    headers={
        "X-GPU-Fault-Execution-Token": os.environ[
            "GPU_FAULT_EXECUTION_TOKEN"
        ]
    },
)
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
    print("status", response.status, "clusters", payload)
    assert response.status == 200
    assert payload == []
PY
  )"
  printf 'payload=[]\n%s\n' "${empty_registry_output}" |
    tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-002.txt"
fi

if (( BOOT_GUARD_START_CASE <= 3 )); then
  reset_probe
  apply_mutation \
    set GPU_FAULT_HYPERPOD_CLUSTER "${CPU_HYPERPOD_CLUSTER}"
  assert_probe \
    "regional mode uses the cluster registry; GPU_FAULT_HYPERPOD_CLUSTER must be unset" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-003.txt"
fi

if (( BOOT_GUARD_START_CASE <= 4 )); then
  reset_probe
  apply_mutation set GPU_FAULT_ENABLE_KUBERNETES_ADAPTER true
  assert_probe \
    "regional control plane must not enable the in-cluster KubernetesWorkflowAdapter" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-004.txt"
fi

if (( BOOT_GUARD_START_CASE <= 5 )); then
  reset_probe
  apply_mutation set GPU_FAULT_ENABLE_HYPERPOD_ADAPTER true
  assert_probe \
    "regional control plane must delegate HyperPod mutations to the target cluster executor" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-005.txt"
  # shellcheck disable=SC2016 # The expression is AWS CLI JMESPath.
  mutations="$(
    aws cloudtrail lookup-events \
    --region "${AWS_REGION}" \
    --start-time "${RUN_STARTED_AT}" \
    --lookup-attributes \
      AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com \
    --query \
      'Events[?EventName==`RebootClusterNodes` || EventName==`BatchDeleteClusterNodes` || EventName==`BatchReplaceClusterNodes` || EventName==`UpdateClusterSoftware`].[EventTime,EventName,Username]' \
    --output json
  )"
  printf '%s\n' "${mutations}" |
    tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-005.txt"
  jq -e 'length == 0' <<<"${mutations}" >/dev/null
fi

if (( BOOT_GUARD_START_CASE <= 6 )); then
  reset_probe
  apply_mutation set GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS true
  assert_probe \
    "regional control plane cannot run in-cluster quick diagnostics against its own EKS" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-006.txt"
fi

if (( BOOT_GUARD_START_CASE <= 7 )); then
  probe_registry 31
  assert_probe "cluster token must contain at least 32 characters" |
    tee "${CASE_DIR}/GF-REGIONAL-BOOT-007.txt"

  probe_registry 32
  probe_pod="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get pod -l "app=${PROBE}" -o jsonpath='{.items[0].metadata.name}'
  )"
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    wait --for=condition=ready "pod/${probe_pod}" --timeout=600s
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod "${probe_pod}" \
    -o custom-columns=READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount \
    --no-headers |
    tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-007.txt"
fi

probe_registry 64 "" eks_cluster_arn
assert_probe "validation error for RegionalClusterRegistration" |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-008.txt"

probe_registry 64 "" "" alowed_namespaces
assert_probe "Extra inputs are not permitted" |
  tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-008.txt"

probe_registry 64 disabled
probe_pod="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l "app=${PROBE}" -o jsonpath='{.items[0].metadata.name}'
)"
kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
  wait --for=condition=ready "pod/${probe_pod}" --timeout=600s
disabled_output="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    exec -i "${probe_pod}" -- python - <<'PY'
import json
import urllib.error
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/regional/executors/claim",
    data=json.dumps(
        {
            "executor_id": "boot008-disabled-probe",
            "executor_protocol_version": 2,
            "execution_owners": ["gpu-fault-kubernetes-adapter"],
            "max_commands": 1,
            "lease_seconds": 60,
        }
    ).encode(),
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": "guardprobe-fake-cluster",
        "Authorization": "Bearer " + "g" * 64,
    },
    method="POST",
)
try:
    urllib.request.urlopen(request, timeout=10)
except urllib.error.HTTPError as exc:
    body = exc.read().decode()
    print("status", exc.code, body)
    assert exc.code == 403
    assert "regional cluster authentication failed" in body
else:
    raise SystemExit("disabled cluster authenticated")
PY
)"
printf '%s\n' "${disabled_output}" |
  tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-008.txt"
unset disabled_output

reset_probe
apply_mutation set GPU_FAULT_ENABLE_AGENT_REGISTRY false
assert_probe \
  "HyperPod managed recovery observer requires GPU_FAULT_ENABLE_AGENT_REGISTRY=true" |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-009.txt"

reset_probe
"${FIXTURE_DIR}/cleanup.sh" --drop-database |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-cleanup.txt"
trap - EXIT
shred -u "${BASE}" 2>/dev/null || true

current="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get deployment gpu-fault-api-ha \
    -o jsonpath='{.metadata.generation} {.status.readyReplicas}'
)"
printf 'current=%s\n' "${current}" |
  tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-baseline.txt"
[[ "${current}" == "${baseline}" ]]

: >"${CASE_DIR}/GF-REGIONAL-BOOT-010.txt"
baseline_env=""
for pod in $(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l app=gpu-fault-api-ha \
    -o jsonpath='{.items[*].metadata.name}'
); do
  output="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      exec "${pod}" -- printenv |
      grep -E \
        '^GPU_FAULT_(DEPLOYMENT_MODE|ENABLE_KUBERNETES_ADAPTER|ENABLE_NODE_ACTION_ADAPTER|ENABLE_HYPERPOD_ADAPTER|ENABLE_HYPERPOD_SPARE_FAILOVER|ENABLE_HYPERPOD_MANAGED_OBSERVER|ENABLE_AGENT_REGISTRY|ENABLE_QUICK_DIAGNOSTICS|HYPERPOD_CLUSTER|PROCESSOR_MODE)=' |
      sort
  )"
  printf '== %s\n%s\n' "${pod}" "${output}" |
    tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-010.txt"
  if [[ -z "${baseline_env}" ]]; then
    baseline_env="${output}"
  else
    [[ "${output}" == "${baseline_env}" ]]
  fi
done
grep -q '^GPU_FAULT_DEPLOYMENT_MODE=regional$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_KUBERNETES_ADAPTER=false$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER=false$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_HYPERPOD_ADAPTER=false$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER=true$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_AGENT_REGISTRY=true$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS=false$' <<<"${baseline_env}"
grep -q '^GPU_FAULT_PROCESSOR_MODE=active-active$' <<<"${baseline_env}"
if grep -q '^GPU_FAULT_HYPERPOD_CLUSTER=' <<<"${baseline_env}"; then
  echo "GPU_FAULT_HYPERPOD_CLUSTER must be absent in regional mode" >&2
  exit 1
fi

kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
  get pod -l app=gpu-fault-api-ha \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName |
  tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-010.txt"
node_count="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l app=gpu-fault-api-ha -o json |
    jq '[.items[].spec.nodeName] | unique | length'
)"
[[ "${node_count}" == "3" ]]

echo "GF-REGIONAL-BOOT-001..010 PASS"
