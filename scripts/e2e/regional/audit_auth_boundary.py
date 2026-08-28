from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


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
                "body": json.loads(raw or b"{}"),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": json.loads(exc.read() or b"{}"),
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
                "body": json.loads(raw or b"{}"),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": json.loads(exc.read() or b"{}"),
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
        "execution_owners": ["gpu-fault-kubernetes-adapter"],
        "max_commands": 1,
        "lease_seconds": 60,
    }


def validate_matrix(results: dict[str, Any], *, cluster_a: str) -> None:
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
    for name, status in expected.items():
        assert results[name]["status"] == status, (name, results[name])
    expected_detail = "regional cluster authentication failed"
    for name in ("AUTH-004-zero", "AUTH-004-near"):
        assert results[name]["body"].get("detail") == expected_detail, (
            name,
            results[name],
        )
    for name in ("AUTH-008-A-normal", "AUTH-008-A-fake-executor"):
        commands = results[name]["body"].get("commands") or []
        assert all(command.get("cluster_id") == cluster_a for command in commands), (
            name,
            commands,
        )


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
    validate_matrix(results, cluster_a=arguments.cluster_a)
    return results


def probe_clusters(arguments: argparse.Namespace) -> dict[str, Any]:
    token_a = arguments.token_a_file.read_text(encoding="utf-8").strip()
    token_b = arguments.token_b_file.read_text(encoding="utf-8").strip()
    claim = claim_payload(
        executor_id="auth-enable-probe",
        artifact_sha256=arguments.executor_artifact_sha256,
        compatibility_digest=arguments.executor_compatibility_digest,
    )
    return {
        "A": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_a,
            token=token_a,
            payload=claim,
        ),
        "B": post(
            arguments.url,
            arguments.ca_file,
            "/v1/regional/executors/claim",
            cluster_id=arguments.cluster_b,
            token=token_b,
            payload=claim,
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("matrix", "probe-clusters"))
    value.add_argument("--url", required=True)
    value.add_argument("--ca-file", required=True, type=Path)
    value.add_argument("--cluster-a", required=True)
    value.add_argument("--token-a-file", required=True, type=Path)
    value.add_argument("--cluster-b", required=True)
    value.add_argument("--token-b-file", required=True, type=Path)
    value.add_argument("--executor-artifact-sha256", required=True)
    value.add_argument("--executor-compatibility-digest", required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    result = (
        run_matrix(arguments)
        if arguments.mode == "matrix"
        else probe_clusters(arguments)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
