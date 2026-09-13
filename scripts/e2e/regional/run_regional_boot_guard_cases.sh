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
# Optional: the IAM role name the control-plane Pods run as. When set, the
# BOOT-005 CloudTrail read is filtered to that principal; when unset, every
# HyperPod mutation in the window is reported.
: "${CONTROL_PLANE_ROLE_NAME:=}"

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

# ---------------------------------------------------------------------------
# Verdict contract. Every case file ends in exactly one line
#   VERDICT PASS | VERDICT FAIL
# written by pass_case / fail_case / the ERR trap. Anything else in the file
# (assert.sh's own "PASS", tee'd kubectl output) is evidence, not a verdict:
# the old resume gate grepped a bare "PASS" line that assert.sh writes *before*
# the later checks of the same case run, so a case could resume as passed after
# a later check had failed.
# ---------------------------------------------------------------------------
CURRENT_EVIDENCE=""

begin_case() {
  printf -v case_id 'GF-REGIONAL-BOOT-%03d' "$1"
  CURRENT_EVIDENCE="${CASE_DIR}/${case_id}.txt"
  : >"${CURRENT_EVIDENCE}"
  echo "== ${case_id}"
}

pass_case() {
  echo "VERDICT PASS" | tee -a "${CURRENT_EVIDENCE}"
  CURRENT_EVIDENCE=""
}

fail_case() {
  echo "FAIL: $*" >&2
  if [[ -n "${CURRENT_EVIDENCE}" ]]; then
    printf 'FAIL: %s\nVERDICT FAIL\n' "$*" >>"${CURRENT_EVIDENCE}"
  fi
  exit 1
}

on_error() {
  local status=$?
  if [[ -n "${CURRENT_EVIDENCE}" ]]; then
    printf 'FAIL: command exited %s\nVERDICT FAIL\n' "${status}" \
      >>"${CURRENT_EVIDENCE}"
  fi
}
trap on_error ERR

case_passed() {
  # The gate reads the last line only: a "VERDICT PASS" followed by anything
  # is not a finished case.
  if [[ -f "$1" && "$(tail -n 1 "$1")" == "VERDICT PASS" ]]; then
    return 0
  fi
  return 1
}

if (( BOOT_GUARD_START_CASE > 1 )); then
  for case_number in $(seq 1 $((BOOT_GUARD_START_CASE - 1))); do
    # BOOT-006 is retired: the in-cluster quick-diagnostics guard it drove no
    # longer exists, so no run of this script produces its evidence. The
    # numbering of the surviving cases is unchanged.
    if (( case_number == 6 )); then
      continue
    fi
    printf -v case_id 'GF-REGIONAL-BOOT-%03d' "${case_number}"
    evidence="${CASE_DIR}/${case_id}.txt"
    if ! case_passed "${evidence}"; then
      echo "BOOT-007 resume requires prior VERDICT PASS evidence: ${evidence}" >&2
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

# The Pod name of the single probe replica, read only after the new rollout has
# genuinely finished. `.items[0]` straight after `kubectl apply` races the
# ReplicaSet: the list is empty, or still holds the previous round's terminating
# Pod.
#
# `wait --for=condition=Available` is NOT enough here: deleting and re-applying
# the Deployment under the same name races the controller, so `wait` matches a
# stale `Available=True` from the prior generation and returns in ~1s -- before
# the new Pod's app has bound :8080. The positive callers then `exec` straight
# into a Pod whose uvicorn is still starting and get `Connection refused`
# (BOOT-002's empty-registry `[]` check failed exactly this way on 2026-09-13).
# `rollout status` gates on observedGeneration + availableReplicas, so it only
# returns once a Pod is Ready (readiness is GET /healthz on :8080, i.e. the app
# is actually serving).
probe_pod_name() {
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    rollout status deployment "${PROBE}" --timeout=600s >/dev/null
  local pods
  pods="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get pod -l "app=${PROBE}" -o json |
      jq -r '[.items[] | select(.metadata.deletionTimestamp == null) | .metadata.name] | .[]'
  )"
  if [[ "$(wc -l <<<"${pods}")" != "1" || -z "${pods}" ]]; then
    fail_case "expected exactly one probe Pod, got: ${pods//$'\n'/ }"
  fi
  printf '%s\n' "${pods}"
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
# One readiness period of the probe: after the guard text is seen, Ready is
# re-read this much later so a Pod that flips Ready on the next probe tick is
# not recorded as refused.
readiness_period="$(
  jq -r '.spec.template.spec.containers[0].readinessProbe.periodSeconds // 10' \
    "${BASE}"
)"

if (( BOOT_GUARD_START_CASE <= 1 )); then
  begin_case 1
  reset_probe
  apply_mutation del GPU_FAULT_REGIONAL_CLUSTERS_JSON
  assert_probe \
    "regional mode requires GPU_FAULT_REGIONAL_CLUSTERS_JSON" |
    tee -a "${CURRENT_EVIDENCE}"
  endpoints="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get endpoints gpu-fault-api \
      -o jsonpath='{range .subsets[*].addresses[*]}{.targetRef.name}{"\n"}{end}'
  )"
  printf '%s\n' "${endpoints}" >>"${CURRENT_EVIDENCE}"
  if grep -q "gpu-fault-api-guard-probe" <<<"${endpoints}"; then
    fail_case "guard probe entered the production Service endpoints"
  fi
  # assert.sh saw "not Ready" at the moment the text matched; one readiness
  # period later it must still not be Ready, or the guard only delayed startup.
  sleep "${readiness_period}"
  probe_pod="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get pod -l "app=${PROBE}" -o jsonpath='{.items[0].metadata.name}'
  )"
  ready_again="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      get pod "${probe_pod}" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
  )"
  printf 'ready_after_%ss=%s\n' "${readiness_period}" "${ready_again}" |
    tee -a "${CURRENT_EVIDENCE}"
  if [[ "${ready_again}" == "True" ]]; then
    fail_case "probe became Ready one readiness period after the guard fired"
  fi
  pass_case
fi

if (( BOOT_GUARD_START_CASE <= 2 )); then
  begin_case 2
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
      tee -a "${CURRENT_EVIDENCE}"
    assert_probe "${expected}" |
      tee -a "${CURRENT_EVIDENCE}"
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
  probe_pod="$(probe_pod_name)"
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
    if response.status != 200 or payload != []:
        raise SystemExit("empty registry did not answer 200 []")
PY
  )"
  printf 'payload=[]\n%s\n' "${empty_registry_output}" |
    tee -a "${CURRENT_EVIDENCE}"
  pass_case
fi

if (( BOOT_GUARD_START_CASE <= 3 )); then
  begin_case 3
  reset_probe
  apply_mutation \
    set GPU_FAULT_HYPERPOD_CLUSTER "${CPU_HYPERPOD_CLUSTER}"
  assert_probe \
    "regional mode uses the cluster registry; GPU_FAULT_HYPERPOD_CLUSTER must be unset" |
    tee -a "${CURRENT_EVIDENCE}"
  pass_case
fi

if (( BOOT_GUARD_START_CASE <= 4 )); then
  begin_case 4
  reset_probe
  apply_mutation set GPU_FAULT_ENABLE_KUBERNETES_ADAPTER true
  assert_probe \
    "regional control plane must not enable the in-cluster KubernetesWorkflowAdapter" |
    tee -a "${CURRENT_EVIDENCE}"
  pass_case
fi

if (( BOOT_GUARD_START_CASE <= 5 )); then
  begin_case 5
  reset_probe
  apply_mutation set GPU_FAULT_ENABLE_HYPERPOD_ADAPTER true
  assert_probe \
    "regional control plane must delegate HyperPod mutations to the target cluster executor" |
    tee -a "${CURRENT_EVIDENCE}"
  # CloudTrail is eventually consistent (delivery within 15 minutes), so a
  # lookup seconds after the action cannot prove absence. The positive
  # evidence for this case is the guard exit above; the CloudTrail read is
  # recorded as provisional, filtered to the control-plane role when known.
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
  if [[ -n "${CONTROL_PLANE_ROLE_NAME}" ]]; then
    mutations="$(
      jq --arg role "${CONTROL_PLANE_ROLE_NAME}" \
        '[.[] | select((.[2] // "") | contains($role))]' <<<"${mutations}"
    )"
    printf 'cloudtrail_filter=role:%s\n' "${CONTROL_PLANE_ROLE_NAME}" |
      tee -a "${CURRENT_EVIDENCE}"
  else
    printf 'cloudtrail_filter=none\n' | tee -a "${CURRENT_EVIDENCE}"
  fi
  printf 'cloudtrail_provisional=true (lookup at %s, CloudTrail delivers within 15 min)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" |
    tee -a "${CURRENT_EVIDENCE}"
  printf '%s\n' "${mutations}" |
    tee -a "${CURRENT_EVIDENCE}"
  if ! jq -e 'length == 0' <<<"${mutations}" >/dev/null; then
    fail_case "CloudTrail shows a HyperPod mutation during the guard window"
  fi
  pass_case
fi

if (( BOOT_GUARD_START_CASE <= 7 )); then
  begin_case 7
  probe_registry 31
  assert_probe "cluster token must contain at least 32 characters" |
    tee -a "${CURRENT_EVIDENCE}"

  probe_registry 32
  probe_pod="$(probe_pod_name)"
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod "${probe_pod}" \
    -o custom-columns=READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount \
    --no-headers |
    tee -a "${CURRENT_EVIDENCE}"
  # Ready alone is the kubelet's view; the positive branch also has to answer
  # its own health endpoint from inside the Pod.
  healthz_output="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      exec -i "${probe_pod}" -- python - <<'PY'
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=10) as response:
    print("healthz", response.status)
    if response.status != 200:
        raise SystemExit("healthz did not answer 200")
PY
  )"
  printf '%s\n' "${healthz_output}" | tee -a "${CURRENT_EVIDENCE}"
  pass_case
fi

begin_case 8
probe_registry 64 "" eks_cluster_arn
assert_probe "validation error for RegionalClusterRegistration" |
  tee -a "${CURRENT_EVIDENCE}"

probe_registry 64 "" "" alowed_namespaces
assert_probe "Extra inputs are not permitted" |
  tee -a "${CURRENT_EVIDENCE}"

probe_registry 64 disabled
probe_pod="$(probe_pod_name)"
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
    if exc.code != 403 or "regional cluster authentication failed" not in body:
        raise SystemExit("disabled cluster was not refused with 403")
else:
    raise SystemExit("disabled cluster authenticated")
PY
)"
printf '%s\n' "${disabled_output}" |
  tee -a "${CURRENT_EVIDENCE}"
unset disabled_output
pass_case

begin_case 9
reset_probe
apply_mutation set GPU_FAULT_ENABLE_AGENT_REGISTRY false
assert_probe \
  "HyperPod managed recovery observer requires GPU_FAULT_ENABLE_AGENT_REGISTRY=true" |
  tee -a "${CURRENT_EVIDENCE}"
pass_case

reset_probe
"${FIXTURE_DIR}/cleanup.sh" --drop-database |
  tee "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-cleanup.txt"
trap - EXIT
shred -u "${BASE}" 2>/dev/null || true

# BOOT-010 is the positive gate: production is back at its baseline and the
# three replicas run in regional mode with the dangerous switches off. Every
# comparison names what it found; a silent `[[ ]]` under set -e exits with no
# line in the evidence saying which one failed.
begin_case 10
current="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get deployment gpu-fault-api-ha \
    -o jsonpath='{.metadata.generation} {.status.readyReplicas}'
)"
printf 'current=%s\n' "${current}" |
  tee -a "${CASE_DIR}/GF-REGIONAL-BOOT-001-010-baseline.txt" "${CURRENT_EVIDENCE}"
if [[ "${current}" != "${baseline}" ]]; then
  fail_case "production deployment left its baseline: ${baseline} -> ${current}"
fi

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
        '^GPU_FAULT_(DEPLOYMENT_MODE|ENABLE_KUBERNETES_ADAPTER|ENABLE_NODE_ACTION_ADAPTER|ENABLE_HYPERPOD_ADAPTER|ENABLE_HYPERPOD_SPARE_FAILOVER|ENABLE_HYPERPOD_MANAGED_OBSERVER|ENABLE_AGENT_REGISTRY|HYPERPOD_CLUSTER|PROCESSOR_MODE)=' |
      sort
  )"
  printf '== %s\n%s\n' "${pod}" "${output}" |
    tee -a "${CURRENT_EVIDENCE}"
  if [[ -z "${baseline_env}" ]]; then
    baseline_env="${output}"
  elif [[ "${output}" != "${baseline_env}" ]]; then
    fail_case "replica ${pod} environment differs from the first replica"
  fi
done

expect_env() {
  if ! grep -q "^$1\$" <<<"${baseline_env}"; then
    fail_case "expected '$1' in every replica environment"
  fi
}
expect_env 'GPU_FAULT_DEPLOYMENT_MODE=regional'
expect_env 'GPU_FAULT_ENABLE_KUBERNETES_ADAPTER=false'
expect_env 'GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER=false'
expect_env 'GPU_FAULT_ENABLE_HYPERPOD_ADAPTER=false'
expect_env 'GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false'
expect_env 'GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER=true'
expect_env 'GPU_FAULT_ENABLE_AGENT_REGISTRY=true'
expect_env 'GPU_FAULT_PROCESSOR_MODE=active-active'
if grep -q '^GPU_FAULT_HYPERPOD_CLUSTER=' <<<"${baseline_env}"; then
  fail_case "GPU_FAULT_HYPERPOD_CLUSTER must be absent in regional mode"
fi

# The registry is loaded: every registered cluster answers the read-only
# collector-status side channel with 200 (not 404/500). Read from inside one
# replica so the execution token never leaves the Pod.
cluster_ids="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get secret gpu-fault-regional-clusters -o jsonpath='{.data.clusters\.json}' |
    base64 -d | jq -r '.[].cluster_id'
)"
if [[ -z "${cluster_ids}" ]]; then
  fail_case "production registry names no clusters"
fi
api_pod="$(latest_ready_api_pod)"
collector_status_output="$(
  CLUSTER_IDS="${cluster_ids}" kubectl --kubeconfig "${CPU_KUBECONFIG}" \
    -n "${NAMESPACE}" exec -i "${api_pod}" -- \
    env "PROBE_CLUSTER_IDS=${cluster_ids}" python - <<'PY'
import os
import urllib.error
import urllib.request

failures = []
for cluster_id in os.environ["PROBE_CLUSTER_IDS"].split():
    request = urllib.request.Request(
        f"http://127.0.0.1:8080/v1/collector-status/{cluster_id}",
        headers={
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"]
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    print(f"collector-status {cluster_id} {status}")
    if status != 200:
        failures.append(cluster_id)
if failures:
    raise SystemExit("collector-status not 200 for: " + ", ".join(failures))
PY
)"
printf '%s\n' "${collector_status_output}" | tee -a "${CURRENT_EVIDENCE}"

kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
  get pod -l app=gpu-fault-api-ha \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName |
  tee -a "${CURRENT_EVIDENCE}"
node_count="$(
  kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l app=gpu-fault-api-ha -o json |
    jq '[.items[].spec.nodeName] | unique | length'
)"
if [[ "${node_count}" != "3" ]]; then
  fail_case "api replicas span ${node_count} nodes, expected 3"
fi
pass_case

echo "GF-REGIONAL-BOOT-001..010 PASS"
