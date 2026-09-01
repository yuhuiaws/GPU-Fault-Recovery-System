from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from scripts.perf import benchmark_regional_action_executor as benchmark
from scripts.perf import regional_action_capacity_suite as suite
from tests._builders import build_store
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/perf/seed_regional_action_workflows.py"
MODULE = lazy_script_module("seed_regional_action_workflows", PATH)
NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)


def release_agent_identity() -> dict[str, object]:
    return {
        "agent_protocol_version": 3,
        "node_action_key_version": 2,
        "agent_version": "0.10.0",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "installer_bundle_sha256": "c" * 64,
        "policy_version": "catalog-v1",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "d" * 64,
        "allowed_operations": [
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS.value,
            WorkflowOperation.RESET_GPU.value,
        ],
    }


def test_action_capacity_workflow_is_multinode_dag() -> None:
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    steps = MODULE.workflow_steps(
        nodes=nodes, workload_id="training/PyTorchJob/test", run_id="action-test"
    )

    assert len(steps) == 10
    assert all(step.node_ids == nodes for step in steps)
    assert steps[3].branch_id == "diagnostics"
    assert steps[4].branch_id == "mutation"
    assert steps[5].depends_on_step_indexes == [3, 4]
    assert [step.operation for step in steps] == [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.CHECKPOINT_WORKLOADS,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]


def test_action_capacity_seed_clones_current_agent_identity() -> None:
    template = AgentRecord(
        cluster_id="production",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=3,
        node_action_key_version=2,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        compatibility_digest="b" * 64,
        policy_version="catalog-v1",
        runtime_profile_version="hyperpod-v1",
        config_digest="c" * 64,
        allowed_operations=[WorkflowOperation.RESET_GPU],
        first_seen_at=NOW - timedelta(minutes=1),
        last_seen_at=NOW,
        lease_expires_at=NOW + timedelta(minutes=1),
    )

    agent = MODULE.synthetic_agent(
        template,
        cluster_id="perf-cap-000",
        node_id="synthetic-node",
        run_id="run-a",
        now=NOW,
    )

    assert agent.cluster_id == "perf-cap-000"
    assert agent.node_id == "synthetic-node"
    assert agent.endpoint == "http://127.0.0.1:9"
    assert agent.identity == template.identity
    assert agent.lease_expires_at == NOW + timedelta(minutes=30)


def test_action_capacity_seed_builds_release_bound_agent_identity() -> None:
    agent = MODULE.release_bound_synthetic_agent(
        release_agent_identity(),
        cluster_id="perf-cap-000",
        node_id="synthetic-node",
        run_id="run-a",
        now=NOW,
    )

    assert agent.cluster_id == "perf-cap-000"
    assert agent.node_id == "synthetic-node"
    assert agent.agent_protocol_version == 3
    assert agent.node_action_key_version == 2
    assert agent.artifact_sha256 == "a" * 64
    assert agent.compatibility_digest == "b" * 64
    assert agent.installer_bundle_sha256 == "c" * 64
    assert agent.allowed_operations == [
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
    ]
    assert agent.lease_expires_at == NOW + timedelta(minutes=30)


def test_integrated_agent_heartbeat_refresh_preserves_identity() -> None:
    context = ApplicationContext(store=build_store())
    node_id = MODULE.integrated_node_id("run-a", 0, 0)
    agent = MODULE.release_bound_synthetic_agent(
        release_agent_identity(),
        cluster_id="perf-cap-000",
        node_id=node_id,
        run_id="run-a",
        now=NOW,
    )
    context.store.save_agent(agent)
    refreshed_at = NOW + timedelta(minutes=5)

    count = MODULE.refresh_integrated_agent_heartbeats(
        context,
        run_id="run-a",
        clusters=1,
        workflows_per_cluster=1,
        now=refreshed_at,
        lease_seconds=600,
    )
    refreshed = context.store.list_agents("perf-cap-000")[0]

    assert count == 1
    assert refreshed.identity == agent.identity
    assert refreshed.first_seen_at == agent.first_seen_at
    assert refreshed.last_seen_at == refreshed_at
    assert refreshed.lease_expires_at == refreshed_at + timedelta(seconds=600)
    latest = context.gpu_metrics.latest("perf-cap-000", node_id)
    assert {item.sample.canonical_name for item in latest} == {
        "gpu_temperature_c",
        "nvlink_crc_aggregate_error_total",
        "nvlink_recovery_aggregate_error_total",
        "nvlink_replay_aggregate_error_total",
    }
    host = context.store.list_telemetry_metrics_latest("perf-cap-000", node_id)
    assert {item.name: item.value for item in host} == {
        "load1_per_cpu": 0.1,
        "memory_used_percent": 10.0,
        "filesystem_used_percent": 20.0,
        "network_link_up": 1.0,
        "rdma_link_down": 0.0,
        "rdma_errors_delta": 0.0,
    }
    assert all(item.observed_at == refreshed_at for item in host), (
        "synthetic host metrics did not cross the post-action freshness barrier"
    )
    statuses = context.store.list_collector_statuses("perf-cap-000", node_id)
    assert {item.collector.value for item in statuses} == {
        "FABRIC_MANAGER_LOG",
        "GPU_INVENTORY",
        "GPU_METRICS",
        "HOST_TELEMETRY",
        "NVIDIA_KERNEL",
    }
    assert all(item.last_success_at == refreshed_at for item in statuses), (
        "synthetic collector refresh left a stale success timestamp"
    )


def test_action_capacity_seed_rejects_unknown_release_identity_fields() -> None:
    identity = release_agent_identity()
    identity["unexpected"] = "value"

    with pytest.raises(ValueError, match="identity fields are invalid"):
        MODULE.release_bound_synthetic_agent(
            identity,
            cluster_id="perf-cap-000",
            node_id="synthetic-node",
            run_id="run-a",
            now=NOW,
        )


def test_isolated_executor_identity_does_not_read_live_deployment(monkeypatch) -> None:
    state = {
        "executor_protocol_version": 2,
        "executor_wheel_sha256": "a" * 64,
        "component_digests": {"executor": "b" * 64},
    }
    environment = {
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
    }
    monkeypatch.setattr(suite, "_release_state", lambda: state)
    monkeypatch.setattr(suite, "_control_environment", lambda _keys: environment)
    monkeypatch.setattr(
        suite,
        "dataplane_identity",
        lambda *_args, **_kwargs: pytest.fail(
            "isolated identity read a live Executor Deployment"
        ),
    )

    assert suite.executor_identity(require_dataplane_deployment=False) == {
        "executor_protocol_version": 2,
        "executor_artifact_sha256": "a" * 64,
        "executor_compatibility_digest": "b" * 64,
    }


def test_live_executor_identity_reads_production_deployment(monkeypatch) -> None:
    state = {
        "executor_protocol_version": 2,
        "executor_wheel_sha256": "a" * 64,
        "component_digests": {"executor": "b" * 64},
    }
    environment = {
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
    }
    calls = []
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
                                    "value": "a" * 64,
                                },
                                {
                                    "name": ("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"),
                                    "value": "b" * 64,
                                },
                            ]
                        }
                    ]
                }
            }
        }
    }
    monkeypatch.setattr(suite, "_release_state", lambda: state)
    monkeypatch.setattr(suite, "_control_environment", lambda _keys: environment)
    monkeypatch.setattr(
        suite,
        "dataplane_identity",
        lambda *args, **_kwargs: calls.append(args) or json.dumps(deployment),
    )

    assert suite.executor_identity(require_dataplane_deployment=True) == {
        "executor_protocol_version": 2,
        "executor_artifact_sha256": "a" * 64,
        "executor_compatibility_digest": "b" * 64,
    }
    assert calls == [("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")]


def test_release_agent_identity_fails_closed_on_control_pin_drift(monkeypatch) -> None:
    state = {
        "agent_protocol_version": 3,
        "node_wheel_sha256": "a" * 64,
        "bundle_sha256": "c" * 64,
        "runtime_profile_version": "hyperpod-v1",
        "agent_config_digest": "d" * 64,
        "component_digests": {"node_runtime": "b" * 64},
    }
    environment = {
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "3",
        "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "2",
        "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.10.0",
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "e" * 64,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": "b" * 64,
        "GPU_FAULT_REQUIRED_POLICY_VERSION": "catalog-v1",
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "hyperpod-v1",
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "d" * 64,
        "GPU_FAULT_ALLOWED_OPERATIONS": "VERIFY_NO_GPU_CLIENTS,RESET_GPU",
    }
    monkeypatch.setattr(suite, "_release_state", lambda: state)
    monkeypatch.setattr(suite, "_control_environment", lambda _keys: environment)

    with pytest.raises(RuntimeError, match="differ from release state"):
        suite.release_agent_identity()


def test_action_capacity_claim_carries_live_executor_pins(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTOR_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("EXECUTOR_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("EXECUTOR_COMPATIBILITY_DIGEST", "b" * 64)
    monkeypatch.setenv("ACTION_LEASE_SECONDS", "10")

    payload = benchmark.claim_payload("executor-a", 5)

    assert payload["executor_protocol_version"] == 2
    assert payload["executor_artifact_sha256"] == "a" * 64
    assert payload["executor_compatibility_digest"] == "b" * 64
    assert payload["max_commands"] == 5
    assert payload["lease_seconds"] == 10


def test_action_capacity_classifies_claim_errors_without_messages() -> None:
    http_error = benchmark.error.HTTPError(
        "https://control.example", 503, "unavailable", {}, None
    )
    reset_error = benchmark.error.URLError(ConnectionResetError("private detail"))

    assert benchmark.request_error_category(http_error) == "HTTPError:503"
    assert (
        benchmark.request_error_category(reset_error) == "URLError:ConnectionResetError"
    )


def test_action_capacity_job_injects_live_executor_pins() -> None:
    manifest = suite.executor_job(
        clusters=1,
        expected_commands=10,
        workers=5,
        delay_scale=1,
        lease_seconds=10,
        inject_renew_failure_once=True,
        executor_protocol_version=2,
        executor_artifact_sha256="a" * 64,
        executor_compatibility_digest="b" * 64,
        connection_secret="isolated-regional-connection",
    )
    environment = {
        item["name"]: item["value"]
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }

    assert environment["EXECUTOR_PROTOCOL_VERSION"] == "2"
    assert environment["EXECUTOR_ARTIFACT_SHA256"] == "a" * 64
    assert environment["EXECUTOR_COMPATIBILITY_DIGEST"] == "b" * 64
    assert environment["ACTION_LEASE_SECONDS"] == "10"
    assert environment["ACTION_INJECT_RENEW_FAILURE_ONCE"] == "true"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    control_plane = next(
        item
        for item in container["env"]
        if item["name"] == "GPU_FAULT_CONTROL_PLANE_URL"
    )
    assert (
        control_plane["valueFrom"]["secretKeyRef"]["name"]
        == "isolated-regional-connection"
    )


def test_action_capacity_summary_keeps_renewal_and_concurrency_evidence() -> None:
    summary = suite.aggregate_executor_documents(
        [
            {
                "wall_seconds": 10.0,
                "claim_errors": 1,
                "claim_error_counts": {"URLError:ConnectionResetError": 1},
                "claim_error_samples": [
                    {
                        "category": "URLError:ConnectionResetError",
                        "elapsed_seconds": 2.5,
                    }
                ],
                "renewals": 8,
                "renewal_errors": 1,
                "injected_renewal_failures": 1,
                "long_commands": 3,
                "max_concurrent_commands": 4,
            },
            {
                "wall_seconds": 12.0,
                "claim_errors": 2,
                "claim_error_counts": {"HTTPError:503": 2},
                "claim_error_samples": [
                    {"category": "HTTPError:503", "elapsed_seconds": 1.5}
                ],
                "renewals": 9,
                "renewal_errors": 0,
                "injected_renewal_failures": 0,
                "long_commands": 2,
                "max_concurrent_commands": 5,
            },
        ]
    )

    assert summary["claim_errors"] == 3
    assert summary["claim_error_counts"] == {
        "HTTPError:503": 2,
        "URLError:ConnectionResetError": 1,
    }
    assert summary["claim_error_samples"] == [
        {"category": "HTTPError:503", "elapsed_seconds": 1.5},
        {"category": "URLError:ConnectionResetError", "elapsed_seconds": 2.5},
    ]
    assert summary["renewals"] == 17
    assert summary["renewal_errors"] == 1
    assert summary["injected_renewal_failures"] == 1
    assert summary["long_commands"] == 5
    assert summary["max_concurrent_commands"] == 5
