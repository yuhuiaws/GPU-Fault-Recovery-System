from __future__ import annotations

import subprocess

import pytest

from gpu_fault.fleet_cli import _deployment_path, _nodes, parser, run_deployment


def test_nodes_accept_csv_and_file(tmp_path) -> None:
    inventory = tmp_path / "nodes.txt"
    inventory.write_text("node-a\nnode-b\n", encoding="utf-8")

    assert _nodes("node-a,node-b") == ["node-a", "node-b"]
    assert _nodes(str(inventory)) == ["node-a", "node-b"]

    with pytest.raises(Exception, match="unique"):
        _nodes("node-a,node-a")


def test_create_deployment_cli_parses_rollout_contract() -> None:
    args = parser().parse_args(
        [
            "--control-plane-url",
            "http://control-plane",
            "create-deployment",
            "--execution-token",
            "token",
            "--cluster-id",
            "cluster-a",
            "--nodes",
            "node-a,node-b",
            "--agent-version",
            "0.9.0",
            "--artifact-sha256",
            "a" * 64,
            "--policy-version",
            "catalog-a",
            "--runtime-profile-version",
            "profile-a",
            "--config-digest",
            "c" * 64,
            "--max-unavailable",
            "1",
        ]
    )

    assert args.nodes == ["node-a", "node-b"]
    assert args.max_unavailable == 1


def test_run_deployment_waits_for_heartbeat_readiness() -> None:
    calls = []
    transport_calls = []
    deployment = {
        "deployment_id": "deployment/a",
        "cluster_id": "cluster-a",
        "desired_agent_version": "0.9.0",
        "desired_artifact_sha256": "a" * 64,
        "status": "PLANNED",
        "nodes": [{"node_id": "node-a", "status": "PENDING"}],
    }
    reconciled = {
        **deployment,
        "status": "SUCCEEDED",
        "nodes": [{"node_id": "node-a", "status": "READY"}],
    }
    deployment_reads = iter([deployment, reconciled])

    def requester(base_url, path, *, payload=None, token=None):
        calls.append((path, payload, token))
        if path == "/v1/fleet/readiness":
            return {"ready": True, "nodes": []}
        if path.endswith("/next-wave"):
            return {"node_ids": ["node-a"]}
        return next(deployment_reads)

    def runner(command, **kwargs):
        transport_calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    result = run_deployment(
        "http://control-plane",
        "token",
        "deployment/a",
        "deploy --node {node_id} --version {agent_version}",
        poll_interval_seconds=1,
        wave_timeout_seconds=30,
        requester=requester,
        runner=runner,
        sleep=lambda _: None,
        monotonic=lambda: 0,
    )

    assert _deployment_path("deployment/a").endswith("deployment%2Fa")
    assert transport_calls[0][0] == ["deploy", "--node", "node-a", "--version", "0.9.0"]
    assert transport_calls[0][1]["GPU_FAULT_FLEET_NODE_ID"] == "node-a"
    assert result["readiness"]["ready"] is True
    assert calls[-1][0] == "/v1/fleet/readiness"


def test_run_deployment_marks_transport_failure() -> None:
    updates = []
    deployment = {
        "deployment_id": "deployment-a",
        "cluster_id": "cluster-a",
        "desired_agent_version": "0.9.0",
        "desired_artifact_sha256": "a" * 64,
        "status": "PLANNED",
        "nodes": [{"node_id": "node-a", "status": "PENDING"}],
    }

    def requester(base_url, path, *, payload=None, token=None):
        if path.endswith("/next-wave"):
            return {"node_ids": ["node-a"]}
        if "/nodes/" in path:
            updates.append(payload)
            return {}
        return deployment

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, "", "installation failed")

    with pytest.raises(SystemExit, match="transport failed"):
        run_deployment(
            "http://control-plane",
            "token",
            "deployment-a",
            "deploy {node_id}",
            poll_interval_seconds=1,
            wave_timeout_seconds=30,
            requester=requester,
            runner=runner,
        )

    assert updates == [
        {
            "status": "FAILED",
            "reason": ("transport exited with status 7: installation failed"),
        }
    ]
