from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)

# Same value as identity_acceptance_common.ACCEPTANCE_PROBE_OWNER (asserted by
# a unit test). The store filters claim candidates by step.execution_owner, so
# a claim advertising this owner authenticates exactly like the executor's own
# and never leases a real command; the previous payload advertised the real
# adapter owner and every 200 in the matrix took a 60 s lease on production
# work.
ACCEPTANCE_PROBE_OWNER = "gpu-fault-acceptance-probe"

# Which matrix entries make up which catalogued case. AUTH-001..006/008/009/011
# had no live evidence path: the matrix printed to stdout and nothing landed
# under cases/<case>/<case>.json, so AUTH-007's predecessor (AUTH-006) could
# never be satisfied by a real run.
CASE_ENTRIES: dict[str, tuple[str, ...]] = {
    "GF-REGIONAL-AUTH-001": ("AUTH-001",),
    "GF-REGIONAL-AUTH-002": (
        "AUTH-002-no-auth",
        "AUTH-002-basic",
        "AUTH-002-empty-bearer",
    ),
    "GF-REGIONAL-AUTH-003": ("AUTH-003",),
    "GF-REGIONAL-AUTH-004": ("AUTH-004-zero", "AUTH-004-near"),
    "GF-REGIONAL-AUTH-005": ("AUTH-005",),
    "GF-REGIONAL-AUTH-006": ("AUTH-006",),
    "GF-REGIONAL-AUTH-008": (
        "AUTH-008-A-normal",
        "AUTH-008-A-fake-executor",
        "AUTH-008-B-header-A-token",
        "AUTH-008-A-header-B-token",
    ),
    "GF-REGIONAL-AUTH-009": ("AUTH-009 ",),
    "GF-REGIONAL-AUTH-011": (
        "AUTH-011-health",
        "AUTH-011-metrics",
        "AUTH-011-clusters-anon",
        "AUTH-011-clusters-cluster-token",
    ),
}
# Cases whose denial must also leave the store untouched.
STORE_NEGATIVE_CASES = (
    "GF-REGIONAL-AUTH-005",
    "GF-REGIONAL-AUTH-006",
    "GF-REGIONAL-AUTH-008",
    "GF-REGIONAL-AUTH-009",
)

# Runs inside a control-plane API Pod. Counts only; no record content, no
# lease tokens.
STORE_NEGATIVE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
cluster_ids = sys.argv[1:]
store = ApplicationContext.from_environment().store
result = {"clusters": {}, "commands": {}}
for cluster_id in cluster_ids:
    agents = store.list_agents(cluster_id)
    result["clusters"][cluster_id] = {
        "agent_generations": sorted(
            f"{item.node_id}:{item.generation}" for item in agents
        ),
        "attempt_observations": len(store.list_attempt_observations(cluster_id)),
    }
for item in store.list_remote_commands():
    if item.status.value in {"PENDING", "WAITING", "LEASED"}:
        result["commands"][item.command_id] = {
            "cluster_id": item.cluster_id,
            "status": item.status.value,
            "lease_owner": item.lease_owner,
        }
print(json.dumps(result, sort_keys=True))
"""


def redact_body(body: Any) -> Any:
    """Drop lease tokens from any command a claim body might carry."""

    if isinstance(body, dict):
        return {
            key: redact_body(value)
            for key, value in body.items()
            if key != "lease_token"
        }
    if isinstance(body, list):
        return [redact_body(item) for item in body]
    return body


def post(
    base_url: str,
    ca_file: Path,
    path: str,
    *,
    cluster_id: str | None,
    token: str | None,
    payload: dict[str, Any],
    authorization: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if cluster_id is not None:
        headers["X-GPU-Fault-Cluster-ID"] = cluster_id
    if authorization is not None:
        headers["Authorization"] = authorization
    elif token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    context = ssl.create_default_context(cafile=str(ca_file))
    try:
        with urllib.request.urlopen(request, context=context, timeout=20) as response:
            raw = response.read()
            return {
                "status": response.status,
                "body": redact_body(json.loads(raw or b"{}")),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": redact_body(json.loads(exc.read() or b"{}")),
        }


def get(
    base_url: str,
    ca_file: Path,
    path: str,
    *,
    cluster_id: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    headers = {}
    if cluster_id is not None:
        headers["X-GPU-Fault-Cluster-ID"] = cluster_id
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers=headers,
        method="GET",
    )
    context = ssl.create_default_context(cafile=str(ca_file))
    try:
        with urllib.request.urlopen(request, context=context, timeout=20) as response:
            raw = response.read()
            return {
                "status": response.status,
                "body": redact_body(json.loads(raw or b"{}")),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": redact_body(json.loads(exc.read() or b"{}")),
        }


def claim_payload(
    *,
    executor_id: str,
    artifact_sha256: str,
    compatibility_digest: str,
) -> dict[str, Any]:
    return {
        "executor_id": executor_id,
        "executor_protocol_version": 2,
        "executor_artifact_sha256": artifact_sha256,
        "executor_compatibility_digest": compatibility_digest,
        "execution_owners": [ACCEPTANCE_PROBE_OWNER],
        "max_commands": 1,
        "lease_seconds": 60,
    }


def expected_statuses(results: dict[str, Any]) -> dict[str, int]:
    expected = {
        "AUTH-001": 401,
        "AUTH-002-no-auth": 401,
        "AUTH-002-basic": 401,
        "AUTH-002-empty-bearer": 403,
        "AUTH-003": 403,
        "AUTH-004-zero": 403,
        "AUTH-004-near": 403,
        "AUTH-005": 403,
        "AUTH-006": 403,
        "AUTH-008-A-normal": 200,
        "AUTH-008-A-fake-executor": 200,
        "AUTH-008-B-header-A-token": 403,
        "AUTH-008-A-header-B-token": 403,
        "AUTH-011-health": 200,
        "AUTH-011-metrics": 403,
        "AUTH-011-clusters-anon": 403,
        "AUTH-011-clusters-cluster-token": 403,
    }
    expected.update({name: 403 for name in results if name.startswith("AUTH-009 ")})
    return expected


def matrix_errors(results: dict[str, Any], *, cluster_a: str) -> dict[str, list[str]]:
    """Per matrix entry, why its answer is not the contract's."""

    errors: dict[str, list[str]] = {}
    for name, status in expected_statuses(results).items():
        actual = results.get(name, {}).get("status")
        if actual != status:
            errors.setdefault(name, []).append(f"status {actual}, expected {status}")
    expected_detail = "regional cluster authentication failed"
    for name in ("AUTH-004-zero", "AUTH-004-near"):
        if results.get(name, {}).get("body", {}).get("detail") != expected_detail:
            errors.setdefault(name, []).append("detail is not the generic denial")
    for name in ("AUTH-008-A-normal", "AUTH-008-A-fake-executor"):
        commands = results.get(name, {}).get("body", {}).get("commands") or []
        if any(command.get("cluster_id") != cluster_a for command in commands):
            errors.setdefault(name, []).append("a claimed command is foreign")
    return errors


def probe_claims_leased_nothing(results: dict[str, Any]) -> bool:
    """The probe owner has no commands; a non-empty claim body leased real work."""

    return not any(
        results.get(name, {}).get("body", {}).get("commands")
        for name in ("AUTH-008-A-normal", "AUTH-008-A-fake-executor")
    )


def validate_matrix(results: dict[str, Any], *, cluster_a: str) -> None:
    errors = matrix_errors(results, cluster_a=cluster_a)
    assert not errors, errors


def store_negative_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    cluster_a: str,
    cluster_b: str,
) -> dict[str, list[str]]:
    """What the denied requests must not have done to the store.

    AUTH-005 posts a workload observation for B with A's token; AUTH-006 a
    heartbeat for B; AUTH-009 every collector/event route for B; AUTH-008
    claims with a fake executor id and with crossed token/header pairs. None
    may add a record for B, and no command open before the matrix may have
    changed status or lease owner.
    """

    errors: dict[str, list[str]] = {}
    b_before = before.get("clusters", {}).get(cluster_b, {})
    b_after = after.get("clusters", {}).get(cluster_b, {})
    if b_after.get("attempt_observations") != b_before.get("attempt_observations"):
        errors.setdefault("GF-REGIONAL-AUTH-005", []).append(
            "cluster B attempt observations changed"
        )
        errors.setdefault("GF-REGIONAL-AUTH-009", []).append(
            "cluster B attempt observations changed"
        )
    if b_after.get("agent_generations") != b_before.get("agent_generations"):
        errors.setdefault("GF-REGIONAL-AUTH-006", []).append("cluster B agents changed")
        errors.setdefault("GF-REGIONAL-AUTH-009", []).append("cluster B agents changed")
    for command_id, state in before.get("commands", {}).items():
        current = after.get("commands", {}).get(command_id)
        if current is None:
            if state.get("status") == "PENDING":
                errors.setdefault("GF-REGIONAL-AUTH-008", []).append(
                    f"command {command_id} left PENDING during the matrix"
                )
            continue
        if state.get("status") == "PENDING" and (
            current.get("status") != "PENDING"
            or current.get("lease_owner") != state.get("lease_owner")
        ):
            errors.setdefault("GF-REGIONAL-AUTH-008", []).append(
                f"command {command_id} was leased or changed by the matrix"
            )
    return errors


def case_documents(
    results: dict[str, Any],
    *,
    cluster_a: str,
    store_errors: dict[str, list[str]] | None,
    identity: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """One evidence document per audited case, verdict included."""

    entry_errors = matrix_errors(results, cluster_a=cluster_a)
    executed_at = datetime.now(timezone.utc).isoformat()
    documents = {}
    for case_id, prefixes in CASE_ENTRIES.items():
        names = sorted(
            name
            for name in results
            if any(
                name == prefix or (prefix.endswith(" ") and name.startswith(prefix))
                for prefix in prefixes
            )
        )
        errors = {name: entry_errors[name] for name in names if name in entry_errors}
        store_checked = case_id in STORE_NEGATIVE_CASES
        store_case_errors = (
            (store_errors or {}).get(case_id, []) if store_errors is not None else []
        )
        not_evaluated = {}
        if store_checked and store_errors is None:
            not_evaluated["store_unchanged"] = (
                "no --cpu-kubeconfig: the store was not read before and after"
            )
        checks: dict[str, bool] = {
            "matrix_entries_match_contract": bool(names) and not errors,
        }
        if store_checked and store_errors is not None:
            checks["store_unchanged"] = not store_case_errors
        if case_id == "GF-REGIONAL-AUTH-008":
            checks["probe_claims_leased_nothing"] = probe_claims_leased_nothing(results)
        verdict = "PASS" if all(checks.values()) and not not_evaluated else "FAIL"
        documents[case_id] = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": case_id,
            "verdict": verdict,
            "executed_at": executed_at,
            **identity,
            "checks": checks,
            "not_evaluated": not_evaluated,
            "entries": {name: results[name] for name in names},
            "entry_errors": errors,
            "store_errors": store_case_errors,
        }
    return documents


def kubectl_api_pod(kubeconfig: Path, namespace: str) -> str:
    completed = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            namespace,
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "-o",
            "json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    )
    for item in json.loads(completed.stdout).get("items", []):
        status = item.get("status") or {}
        if status.get("phase") == "Running" and all(
            bool(entry.get("ready")) for entry in status.get("containerStatuses") or []
        ):
            return str(item["metadata"]["name"])
    raise RuntimeError("no Ready control-plane API Pod")


def store_snapshot(
    kubeconfig: Path, namespace: str, pod: str, cluster_ids: list[str]
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            namespace,
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-",
            *cluster_ids,
        ],
        input=STORE_NEGATIVE_PROBE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=180,
    )
    value = json.loads(completed.stdout.splitlines()[-1])
    if not isinstance(value, dict):
        raise RuntimeError("store probe did not return an object")
    return value


def release_id(kubeconfig: Path, namespace: str) -> str:
    completed = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            namespace,
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    )
    state = json.loads(json.loads(completed.stdout)["data"]["state.json"])
    return str(state.get("release_id") or "")


def run_matrix(arguments: argparse.Namespace) -> dict[str, Any]:
    token_a = arguments.token_a_file.read_text(encoding="utf-8").strip()
    token_b = arguments.token_b_file.read_text(encoding="utf-8").strip()
    claim = claim_payload(
        executor_id="auth-probe",
        artifact_sha256=arguments.executor_artifact_sha256,
        compatibility_digest=arguments.executor_compatibility_digest,
    )
    near_token = token_a[:-1] + ("a" if token_a[-1] != "a" else "b")
    results = {
        "AUTH-001": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=None,
            token=token_a,
            payload=claim,
        ),
        "AUTH-002-no-auth": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=None,
            payload=claim,
        ),
        "AUTH-002-basic": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=None,
            authorization="Basic YWJjOmRlZg==",
            payload=claim,
        ),
        "AUTH-002-empty-bearer": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=None,
            authorization="Bearer ",
            payload=claim,
        ),
        "AUTH-003": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id="cluster-not-registered",
            token=token_a,
            payload=claim,
        ),
        "AUTH-004-zero": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token="0" * 32,
            payload=claim,
        ),
        "AUTH-004-near": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=near_token,
            payload=claim,
        ),
        "AUTH-005": post(
            arguments.url,
            arguments.ca_file,
            "/v1/workload-observations",
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload={"cluster_id": arguments.cluster_b},
        ),
        "AUTH-006": post(
            arguments.url,
            arguments.ca_file,
            "/v1/fleet/agents/heartbeat",
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload={
                "heartbeat": {
                    "cluster_id": arguments.cluster_b,
                    "node_id": "auth-probe-node",
                    "agent_version": "0.10.0",
                },
                "signature": "invalid",
            },
        ),
        "AUTH-008-A-normal": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload=claim,
        ),
        "AUTH-008-A-fake-executor": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload={**claim, "executor_id": "cluster-b/executor"},
        ),
        "AUTH-008-B-header-A-token": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_b,
            token=token_a,
            payload=claim,
        ),
        "AUTH-008-A-header-B-token": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=token_b,
            payload=claim,
        ),
    }
    for path in (
        "/v1/collector-events/nvidia-kernel",
        "/v1/collector-events/fabric-manager",
        "/v1/collector-events/gpu-metrics",
        "/v1/collector-events/host-telemetry",
        "/v1/collector-events/node-logs",
        "/v1/gpu-events/xid",
        "/v1/gpu-events/sxid",
        "/v1/provider-events/hyperpod-hma/node",
        "/v1/attempts/terminal",
        "/v1/training-progress",
        "/v1/triage-results",
    ):
        results[f"AUTH-009 {path}"] = post(
            arguments.url,
            arguments.ca_file,
            path,
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload={"cluster_id": arguments.cluster_b},
        )
    results.update(
        {
            "AUTH-011-health": get(
                arguments.url,
                arguments.ca_file,
                "/healthz",
            ),
            "AUTH-011-metrics": get(
                arguments.url,
                arguments.ca_file,
                "/metrics",
            ),
            "AUTH-011-clusters-anon": get(
                arguments.url,
                arguments.ca_file,
                "/v1/regional/clusters",
            ),
            "AUTH-011-clusters-cluster-token": get(
                arguments.url,
                arguments.ca_file,
                "/v1/regional/clusters",
                cluster_id=arguments.cluster_a,
                token=token_a,
            ),
        }
    )
    return results


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Live AUTH-001..006/008/009/011 boundary matrix. Tokens are read "
            "from files and stay in memory; with --run-dir each case gets its "
            "own evidence document, with --cpu-kubeconfig the store is read "
            "before and after so the denials are proven to have written nothing."
        )
    )
    # The former "probe-clusters" mode (one enabled-claim per cluster) is what
    # run_identity_acceptance --case GF-REGIONAL-AUTH-007 does with a restore;
    # it is gone from here.
    value.add_argument("mode", choices=("matrix",))
    value.add_argument("--url", required=True)
    value.add_argument("--ca-file", required=True, type=Path)
    value.add_argument("--cluster-a", required=True)
    value.add_argument("--token-a-file", required=True, type=Path)
    value.add_argument("--cluster-b", required=True)
    value.add_argument("--token-b-file", required=True, type=Path)
    value.add_argument("--executor-artifact-sha256", required=True)
    value.add_argument("--executor-compatibility-digest", required=True)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cpu-kubeconfig", type=Path)
    value.add_argument("--namespace", default="gpu-fault-system")
    return value


def main() -> int:
    arguments = parser().parse_args()
    identity: dict[str, str] = {"cluster_id": arguments.cluster_a}
    store_before: dict[str, Any] | None = None
    api_pod = ""
    if arguments.cpu_kubeconfig is not None:
        identity["release_id"] = release_id(
            arguments.cpu_kubeconfig, arguments.namespace
        )
        api_pod = kubectl_api_pod(arguments.cpu_kubeconfig, arguments.namespace)
        store_before = store_snapshot(
            arguments.cpu_kubeconfig,
            arguments.namespace,
            api_pod,
            [arguments.cluster_a, arguments.cluster_b],
        )
    results = run_matrix(arguments)
    store_errors: dict[str, list[str]] | None = None
    if store_before is not None:
        store_after = store_snapshot(
            arguments.cpu_kubeconfig,
            arguments.namespace,
            api_pod,
            [arguments.cluster_a, arguments.cluster_b],
        )
        store_errors = store_negative_errors(
            store_before,
            store_after,
            cluster_a=arguments.cluster_a,
            cluster_b=arguments.cluster_b,
        )
    documents = case_documents(
        results,
        cluster_a=arguments.cluster_a,
        store_errors=store_errors,
        identity=identity,
    )
    if arguments.run_dir is not None:
        for case_id, document in documents.items():
            path = arguments.run_dir / "cases" / case_id / f"{case_id}.json"
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_json_atomic(path, document)
    print(
        json.dumps(
            {
                "results": results,
                "verdicts": {
                    case_id: document["verdict"]
                    for case_id, document in documents.items()
                },
                "store_errors": store_errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    failed = [case for case, doc in documents.items() if doc["verdict"] != "PASS"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
