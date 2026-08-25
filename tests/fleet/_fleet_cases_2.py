from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import (
    AgentTransitionRequest,
    BarrierCoordinator,
    BarrierState,
    FleetDeploymentRequest,
    FleetReadinessRequest,
    sign_agent_heartbeat,
)
from gpu_fault.models import WorkflowExecutionRequest, WorkflowOperation, WorkflowStatus
from gpu_fault.node_agent import AgentHeartbeatReporter, NodeActionStatus
from gpu_fault.runtime_adapters import NodeActionWorkflowAdapter
from gpu_fault.store import SqliteStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import (
    active_workflow_executor,
    asgi_client,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
    workflow_step,
)
from tests.fleet._support import (
    ARTIFACT,
    CONFIG,
    NOW,
    SECRET,
    heartbeat,
    quiesce_then_full_reset_workflow,
    registry,
    signed,
    workflow_state,
)


def test_quiesced_barrier_rejects_expired_maintenance_window() -> None:
    current = [NOW]
    store = build_store()
    fleet = registry(store, now=lambda: current[0])
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: current[0])
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"failsafe_seconds": 420},
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = quiesce_then_full_reset_workflow(store)
    executor = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        },
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    current[0] = NOW + timedelta(seconds=420)
    failed = executor.execute(workflow.request_id, request)

    assert prepared.status is WorkflowStatus.RUNNING
    assert failed.status is WorkflowStatus.FAILED
    assert "maintenance window expired" in failed.error
    assert (
        len(
            [
                item
                for item in sent
                if item.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            ]
        )
        == 0
    )


def test_quiesced_barrier_rejects_changed_agent_generation() -> None:
    current = [NOW]
    store = build_store()
    fleet = registry(store, now=lambda: current[0])
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: current[0])
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"failsafe_seconds": 420},
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = quiesce_then_full_reset_workflow(store)
    executor = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        },
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    current[0] = NOW + timedelta(seconds=10)
    fleet.register(
        signed(heartbeat("node-a", observed_at=current[0], boot_id="boot-b"))
    )
    failed = executor.execute(workflow.request_id, request)

    assert prepared.status is WorkflowStatus.RUNNING
    assert failed.status is WorkflowStatus.FAILED
    assert "agent generation changed from 1 to 2" in failed.error
    assert (
        len(
            [
                item
                for item in sent
                if item.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            ]
        )
        == 0
    )


def test_unquiesced_reset_rejects_stale_agent_heartbeat() -> None:
    current = [NOW]
    store = build_store()
    fleet = registry(store, now=lambda: current[0])
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    current[0] = NOW + timedelta(seconds=138)
    sent = []

    adapter = NodeActionWorkflowAdapter(
        {},
        SECRET,
        registry=fleet,
        barriers=BarrierCoordinator(store, now=lambda: current[0]),
        sender=lambda _, envelope: sent.append(envelope.command),
    )
    workflow = workflow_state(store)
    executor = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})

    failed = execute_workflow(executor, workflow.request_id)

    assert failed.status is WorkflowStatus.FAILED
    assert "agent lease is expired" in failed.error
    assert sent == []


def test_prepare_failure_aborts_without_any_reset() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        failed = envelope.command.node_id == "node-b"
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            NodeActionStatus.FAILED if failed else NodeActionStatus.SUCCEEDED,
            error="client active" if failed else None,
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = workflow_state(store)
    executor = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})

    result = execute_workflow(executor, workflow.request_id)
    barrier = store.get_barrier("workflow-a/0/RESET_GPU")

    assert result.status is WorkflowStatus.FAILED
    assert barrier.state is BarrierState.ABORTED
    assert not any(item.operation is WorkflowOperation.RESET_GPU for item in sent)


def test_commit_client_conflict_retries_only_failed_node() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    sent = []
    node_b_reset_attempts = 0

    def sender(_, envelope):
        nonlocal node_b_reset_attempts
        command = envelope.command
        sent.append(command)
        failed = False
        if (
            command.operation is WorkflowOperation.RESET_GPU
            and command.node_id == "node-b"
        ):
            node_b_reset_attempts += 1
            failed = node_b_reset_attempts == 1
        return node_action_result(
            command.command_id,
            command.operation,
            NodeActionStatus.FAILED if failed else NodeActionStatus.SUCCEEDED,
            error="RuntimeError: GPU device clients are still active"
            if failed
            else None,
            details={} if failed else {"reset_gpu_uuids": command.gpu_uuids},
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = workflow_state(store)
    executor = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    waiting = executor.execute(workflow.request_id, request)
    completed = executor.execute(workflow.request_id, request)

    assert prepared.status is WorkflowStatus.RUNNING
    assert waiting.status is WorkflowStatus.RUNNING
    assert completed.status is WorkflowStatus.SUCCEEDED
    reset_commands = [
        item for item in sent if item.operation is WorkflowOperation.RESET_GPU
    ]
    assert [item.node_id for item in reset_commands] == ["node-a", "node-b", "node-b"]
    assert reset_commands[1].command_id.endswith("node-b/attempt-1/agent-1")
    assert reset_commands[2].command_id.endswith("node-b/attempt-2/agent-1")


def test_verify_gpu_clients_waits_with_fresh_command_id() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    workflow = workflow_state(store)
    step = workflow_step(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        "gpu-fault-node-agent",
        gpu_uuids=["GPU-a"],
        parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}},
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        active = len(sent) == 1
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            NodeActionStatus.FAILED if active else NodeActionStatus.SUCCEEDED,
            error="RuntimeError: GPU device clients are still active"
            if active
            else None,
            details={} if active else {"verified_no_gpu_clients": True},
        )

    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=fleet, sender=sender)
    executor = active_workflow_executor(
        store, [adapter], {WorkflowOperation.VERIFY_NO_GPU_CLIENTS}
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    waiting = executor.execute(workflow.request_id, request)
    completed = executor.execute(workflow.request_id, request)

    assert waiting.status is WorkflowStatus.RUNNING
    assert waiting.waiting_step_index == 0
    assert completed.status is WorkflowStatus.SUCCEEDED
    assert sent[0].command_id.endswith("node-a/attempt-1/agent-1")
    assert sent[1].command_id.endswith("node-a/attempt-2/agent-1")


def test_agent_generation_change_aborts_prepared_barrier() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        return node_action_result(
            envelope.command.command_id, envelope.command.operation
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = workflow_state(store)
    executor = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    rebooted = heartbeat(
        "node-a", observed_at=NOW + timedelta(seconds=10), boot_id="boot-b"
    )
    fleet.register(signed(rebooted))
    aborted = executor.execute(workflow.request_id, request)
    barrier = store.get_barrier("workflow-a/0/RESET_GPU")

    assert prepared.status is WorkflowStatus.RUNNING
    assert aborted.status is WorkflowStatus.FAILED
    assert barrier.state is BarrierState.ABORTED
    assert "generation changed" in aborted.error
    assert not any(item.operation is WorkflowOperation.RESET_GPU for item in sent)


def test_sqlite_persists_agent_deployment_and_barrier(tmp_path) -> None:
    path = tmp_path / "fleet.db"
    first = SqliteStore(str(path))
    fleet = registry(first)
    record = fleet.register(signed(heartbeat("node-a")))
    deployment = fleet.create_deployment(
        FleetDeploymentRequest(
            cluster_id="cluster-a",
            node_ids=["node-a"],
            desired_agent_version="0.9.0",
            desired_artifact_sha256=ARTIFACT,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="profile-a",
            desired_config_digest=CONFIG,
        )
    )
    barrier = BarrierCoordinator(first, now=lambda: NOW).create(
        barrier_id="barrier-a",
        cluster_id="cluster-a",
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        generations={"node-a": record.generation},
    )
    transition = AgentTransitionRequest(
        expected_generation=record.generation,
        transition_id="workflow-a/replace-node",
        reason="planned replacement",
    )
    fleet.drain_agent("cluster-a", "node-a", transition)
    revoked = fleet.revoke_agent("cluster-a", "node-a", transition)

    second = SqliteStore(str(path))

    assert second.get_agent("cluster-a", "node-a") == revoked
    assert second.get_fleet_deployment(deployment.deployment_id) == deployment
    assert second.get_barrier(barrier.barrier_id) == barrier


def test_barrier_id_cannot_be_reused_for_another_contract() -> None:
    store = build_store()
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    coordinator.create(
        barrier_id="barrier-a",
        cluster_id="cluster-a",
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        generations={"node-a": 1},
    )

    try:
        coordinator.create(
            barrier_id="barrier-a",
            cluster_id="cluster-a",
            workflow_request_id="workflow-a",
            incident_id="incident-a",
            fencing_token=2,
            operation=WorkflowOperation.RESET_GPU,
            generations={"node-a": 1},
        )
    except ValueError as exc:
        assert "different contract" in str(exc)
    else:
        raise AssertionError("barrier id was rebound")


def test_agent_heartbeat_reporter_signs_payload() -> None:
    captured = []
    generations = []

    def sender(endpoint, envelope):
        captured.append((endpoint, envelope))
        return {"generation": 7}

    reporter = AgentHeartbeatReporter(
        control_plane_url="http://control-plane",
        secret=SECRET,
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_version="0.9.0",
        artifact_sha256=ARTIFACT,
        policy_version="catalog-a",
        runtime_profile_version="profile-a",
        config_digest=CONFIG,
        allowed_operations=[WorkflowOperation.RESET_GPU],
        collector_status_provider=lambda: {
            "gpu-fault-kernel-collector": {"active": "active", "enabled": "enabled"}
        },
        boot_id="boot-a",
        generation_sink=generations.append,
        sender=sender,
        now=lambda: NOW,
    )

    reporter.report_once()

    assert captured[0][0] == "http://control-plane"
    envelope = captured[0][1]
    assert envelope.signature == sign_agent_heartbeat(envelope.heartbeat, SECRET)
    assert envelope.heartbeat.agent_protocol_version == 3
    service = envelope.heartbeat.collector_services["gpu-fault-kernel-collector"]
    assert service.active.value == "active"
    assert service.enabled.value == "enabled"
    assert generations == [7]


def test_fleet_api_registers_and_reports_readiness() -> None:
    store = build_store()
    fleet = registry(store)
    context = ApplicationContext(
        store=store,
        execution_token="e" * 32,
        fleet_registry=fleet,
        barrier_coordinator=BarrierCoordinator(store),
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            registered = await client.post(
                "/v1/fleet/agents/heartbeat",
                json=signed(heartbeat("node-a")).model_dump(mode="json"),
            )
            assert registered.status_code == 200
            readiness = await client.post(
                "/v1/fleet/readiness",
                json=FleetReadinessRequest(
                    cluster_id="cluster-a", node_ids=["node-a"]
                ).model_dump(mode="json"),
            )
            assert readiness.status_code == 200
            assert readiness.json()["ready"] is True
            collector_readiness = await client.get(
                "/v1/collector-readiness/cluster-a",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
            )
            assert collector_readiness.status_code == 200
            assert collector_readiness.json()["ready"] is False
            assert collector_readiness.json()["nodes"][0]["node_id"] == "node-a"

            denied = await client.post(
                "/v1/fleet/deployments",
                json={
                    "cluster_id": "cluster-a",
                    "node_ids": ["node-a"],
                    "desired_agent_version": "0.9.0",
                    "desired_artifact_sha256": ARTIFACT,
                    "desired_policy_version": "catalog-a",
                    "desired_runtime_profile_version": "profile-a",
                    "desired_config_digest": CONFIG,
                },
            )
            assert denied.status_code == 403
            created = await client.post(
                "/v1/fleet/deployments",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
                json={
                    "cluster_id": "cluster-a",
                    "node_ids": ["node-a", "node-b"],
                    "desired_agent_version": "0.9.0",
                    "desired_artifact_sha256": ARTIFACT,
                    "desired_policy_version": "catalog-a",
                    "desired_runtime_profile_version": "profile-a",
                    "desired_config_digest": CONFIG,
                },
            )
            assert created.status_code == 200
            deployment_id = created.json()["deployment_id"]
            wave = await client.post(
                f"/v1/fleet/deployments/{deployment_id}/next-wave",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
            )
            assert wave.status_code == 200
            assert wave.json()["node_ids"] == ["node-b"]

            transition = {
                "expected_generation": 1,
                "transition_id": "workflow-a/restart-node",
                "reason": "planned reboot",
            }
            denied_drain = await client.post(
                "/v1/fleet/agents/cluster-a/node-a/drain", json=transition
            )
            assert denied_drain.status_code == 403
            drained = await client.post(
                "/v1/fleet/agents/cluster-a/node-a/drain",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
                json=transition,
            )
            assert drained.status_code == 200
            assert drained.json()["lifecycle_state"] == "DRAINING"
            revoked = await client.post(
                "/v1/fleet/agents/cluster-a/node-a/revoke",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
                json=transition,
            )
            assert revoked.status_code == 200
            assert revoked.json()["lifecycle_state"] == "REVOKED"

            reactivation = {
                **transition,
                "expected_generation": revoked.json()["generation"],
                "reason": "operator verified original node",
            }
            denied_reactivation = await client.post(
                "/v1/fleet/agents/cluster-a/node-a/reactivate", json=reactivation
            )
            assert denied_reactivation.status_code == 403
            reactivated = await client.post(
                "/v1/fleet/agents/cluster-a/node-a/reactivate",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
                json=reactivation,
            )
            assert reactivated.status_code == 200
            assert reactivated.json()["lifecycle_state"] == "ACTIVE"

    asyncio.run(scenario())


def test_collector_readiness_accepts_reported_service_state() -> None:
    store = build_store()
    context = ApplicationContext(
        store=store,
        execution_token="e" * 32,
        fleet_registry=registry(store),
        barrier_coordinator=BarrierCoordinator(store),
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            registered = await client.post(
                "/v1/fleet/agents/heartbeat",
                json=signed(
                    heartbeat(
                        "node-a",
                        collector_services={
                            "gpu-fault-metrics-collector": {
                                "active": "active",
                                "enabled": "enabled",
                            }
                        },
                    )
                ).model_dump(mode="json"),
            )
            assert registered.status_code == 200

            fresh = datetime.now(timezone.utc)
            for kind in {
                CollectorKind.GPU_INVENTORY,
                CollectorKind.GPU_METRICS,
                CollectorKind.HOST_TELEMETRY,
                CollectorKind.NVIDIA_KERNEL,
                CollectorKind.FABRIC_MANAGER_LOG,
            }:
                store.save_collector_status(
                    CollectorStatus(
                        cluster_id="cluster-a",
                        node_id="node-a",
                        collector=kind,
                        observed_at=fresh,
                        ingested_at=fresh,
                        last_success_at=fresh,
                    )
                )

            response = await client.get(
                "/v1/collector-readiness/cluster-a",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        collectors = payload["nodes"][0]["collectors"]
        assert collectors["GPU_METRICS"]["unit_state"] == "active"
        assert collectors["GPU_METRICS"]["unit_enabled"] == "enabled"
        assert collectors["NVIDIA_KERNEL"]["unit_state"] == "unknown"
        assert collectors["NVIDIA_KERNEL"]["unit_enabled"] == "unknown"

    asyncio.run(scenario())
