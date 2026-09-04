from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthPolicy,
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.models import (
    EfaTrafficAdminAction,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import SqliteStore
from tests._builders import (
    asgi_client,
    build_context,
    build_store,
    copy_model,
    host_telemetry_batch,
    workflow_step_execution,
)
from tests.host_health._support import (
    NOW,
    RecordingNotifier,
    efa_telemetry,
    observe_running_attempt,
    telemetry,
)

ADMIN_TOKEN = "a" * 32


def test_efa_inventory_transition_rearms_after_recovery() -> None:
    policy = NodeHealthPolicy(build_store())

    def batch(
        batch_id: str, value: float, failure_mode: str, observed_at: datetime
    ) -> HostTelemetryBatch:
        return host_telemetry_batch(
            batch_id,
            observed_at,
            [
                HostMetricSample(
                    name="efa_inventory_mismatch",
                    value=value,
                    labels={
                        "failure_mode": failure_mode,
                        "expected_count": "16",
                        "observed_count": "15" if value else "16",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )

    first = policy.evaluate_metrics(batch("efa-first", 1, "DRIVER_UNBOUND", NOW))
    recovered = policy.evaluate_metrics(
        batch("efa-recovered", 0, "HEALTHY", NOW + timedelta(seconds=15))
    )
    second = policy.evaluate_metrics(
        batch("efa-second", 1, "DRIVER_UNBOUND", NOW + timedelta(seconds=30))
    )

    assert len(first) == 1
    assert recovered == []
    assert len(second) == 1
    assert first[0].event_id != second[0].event_id


def test_kubernetes_efa_allocatable_loss_restarts_device_plugin() -> None:
    policy = NodeHealthPolicy(build_store())
    finding = policy.evaluate_metrics(
        host_telemetry_batch(
            "efa-kubernetes-resource-loss",
            NOW,
            [
                HostMetricSample(
                    name="efa_kubernetes_allocatable_mismatch",
                    value=1,
                    labels={
                        "failure_mode": "KUBERNETES_RESOURCE_MISSING",
                        "expected_count": "16",
                        "observed_count": "0",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )
    )[0]

    assert finding.recommended_action is RecoveryAction.RESTART_EFA_DEVICE_PLUGIN


def test_kubernetes_gpu_allocatable_loss_restarts_device_plugin() -> None:
    policy = NodeHealthPolicy(build_store())
    finding = policy.evaluate_metrics(
        host_telemetry_batch(
            "gpu-kubernetes-resource-loss",
            NOW,
            [
                HostMetricSample(
                    name="gpu_kubernetes_allocatable_mismatch",
                    value=1,
                    labels={
                        "failure_mode": "KUBERNETES_RESOURCE_MISSING",
                        "expected_count": "8",
                        "observed_count": "0",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )
    )[0]

    assert finding.recommended_action is RecoveryAction.RESTART_GPU_DEVICE_PLUGIN


@pytest.mark.parametrize(
    ("failure_mode", "metric_name", "expected_operation"),
    [
        (
            "DRIVER_UNBOUND",
            "efa_inventory_mismatch",
            WorkflowOperation.REMEDIATE_EFA_DRIVER,
        ),
        (
            "KUBERNETES_RESOURCE_MISSING",
            "efa_kubernetes_allocatable_mismatch",
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        ),
        (
            "KUBERNETES_RESOURCE_MISSING",
            "gpu_kubernetes_allocatable_mismatch",
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        ),
    ],
)
def test_efa_recovery_workflow_uses_targeted_remediation(
    failure_mode, metric_name, expected_operation
) -> None:
    context = build_context()
    observe_running_attempt(context.store, NOW)
    batch = host_telemetry_batch(
        f"efa-targeted-{failure_mode.lower()}",
        NOW,
        [
            HostMetricSample(
                name=metric_name,
                value=1,
                labels={
                    "resource": "EFA",
                    "failure_mode": failure_mode,
                    "expected_count": "16",
                    "expected_efa_device_count": "16",
                    "observed_count": "15",
                    "discovered_count": "16",
                    "driver_bound_count": "15"
                    if failure_mode == "DRIVER_UNBOUND"
                    else "16",
                },
            )
        ],
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        runtime_profile_version="simulated-v1",
    )

    findings = NodeHealthPolicy(context.store).evaluate_metrics(batch)
    incident, workflow = context.orchestrator.ingest_node_health(findings[0])

    assert workflow.status is WorkflowStatus.PENDING
    operations = [step.operation for step in workflow.official_steps]
    assert expected_operation in operations
    assert WorkflowOperation.RESTART_NODE not in operations
    if expected_operation is WorkflowOperation.REMEDIATE_EFA_DRIVER:
        assert WorkflowOperation.STOP_WORKLOADS in operations
        assert WorkflowOperation.RESTART_WORKLOAD in operations
    else:
        assert WorkflowOperation.STOP_WORKLOADS not in operations
        assert WorkflowOperation.RESTART_WORKLOAD not in operations
        assert WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT in operations
    targeted_step = next(
        step for step in workflow.official_steps if step.operation is expected_operation
    )
    assert targeted_step.parameters["expected_count"] == 16
    assert targeted_step.parameters["failure_escalation_action"] == "REBOOT_NODE"
    assert incident.effective_action is findings[0].recommended_action


def test_failed_targeted_efa_recovery_escalates_to_reboot() -> None:
    context = build_context()
    finding = NodeHealthPolicy(context.store).evaluate_metrics(
        host_telemetry_batch(
            "efa-driver-unbound",
            NOW,
            [
                HostMetricSample(
                    name="efa_inventory_mismatch",
                    value=1,
                    labels={
                        "failure_mode": "DRIVER_UNBOUND",
                        "expected_count": "16",
                        "observed_count": "15",
                        "discovered_count": "16",
                        "driver_bound_count": "15",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )
    )[0]
    _, workflow = context.orchestrator.ingest_node_health(finding)
    index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.REMEDIATE_EFA_DRIVER
    )
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                index,
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowStepStatus.FAILED,
                error="bind failed",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalation = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalation is not None
    incident, reboot = escalation
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in reboot.official_steps
    }


def test_inactive_efa_link_diagnoses_then_escalates_to_reboot() -> None:
    context = build_context()
    finding = NodeHealthPolicy(context.store).evaluate_metrics(
        host_telemetry_batch(
            "efa-link-inactive",
            NOW,
            [
                HostMetricSample(
                    name="efa_inventory_mismatch",
                    value=1,
                    labels={
                        "failure_mode": "LINK_INACTIVE",
                        "expected_count": "16",
                        "observed_count": "15",
                        "discovered_count": "16",
                        "driver_bound_count": "16",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )
    )[0]
    _, workflow = context.orchestrator.ingest_node_health(finding)
    operations = [step.operation for step in workflow.official_steps]
    assert operations == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.VALIDATE_FABRIC,
    ]
    validation_index = 1
    assert (
        workflow.official_steps[validation_index].parameters[
            "failure_escalation_action"
        ]
        == "REBOOT_NODE"
    )
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                validation_index,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowStepStatus.FAILED,
                error="EFA link remains inactive",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalation = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalation is not None
    assert escalation[0].effective_action is RecoveryAction.REBOOT_NODE


def test_efa_traffic_anomaly_sends_fixed_email(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    observe_running_attempt(context.store, NOW)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            baseline = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry("efa-email-baseline", 1_000, NOW).model_dump(
                    mode="json"
                ),
            )
            spike = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry(
                    "efa-email-spike", 3_000, NOW + timedelta(seconds=15)
                ).model_dump(mode="json"),
            )

        assert baseline.json()["notification_ids"] == []
        assert len(spike.json()["notification_ids"]) == 1

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    body = notifier.notifications[0].body_text
    assert "事件类型：EFA_TRAFFIC_ANOMALY" in body
    assert "流量状态：SPIKE" in body
    assert "流量 baseline：1000.0" in body
    assert "Job ID：job-a" in body
    assert "Attempt ID：attempt-a" in body


def test_admin_can_acknowledge_transient_efa_spike(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")
    context = ApplicationContext(execution_token=ADMIN_TOKEN)
    observe_running_attempt(context.store, NOW)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry("efa-ack-baseline", 1_000, NOW).model_dump(
                    mode="json"
                ),
            )
            spike = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry(
                    "efa-ack-spike", 3_000, NOW + timedelta(seconds=15)
                ).model_dump(mode="json"),
            )
            event_id = spike.json()["findings"][0]["event_id"]
            payload = {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "job_id": "job-a",
                "attempt_id": "attempt-a",
                "event_id": event_id,
                "action": "ACKNOWLEDGE_TRANSIENT",
                "operator": "admin@example.com",
                "reason": "Expected checkpoint traffic burst",
            }
            unauthorized = await client.post(
                "/v1/efa-traffic/admin-actions", json=payload
            )
            first = await client.post(
                "/v1/efa-traffic/admin-actions",
                json=payload,
                headers={"X-GPU-Fault-Execution-Token": ADMIN_TOKEN},
            )
            duplicate = await client.post(
                "/v1/efa-traffic/admin-actions",
                json=payload,
                headers={"X-GPU-Fault-Execution-Token": ADMIN_TOKEN},
            )

        assert unauthorized.status_code == 403
        assert first.status_code == 200
        assert duplicate.status_code == 200
        assert first.json() == duplicate.json()
        assert first.json()["resulting_signal"] == "SPIKE"
        assert first.json()["resulting_baseline_bytes_per_second"] == 1_000

    asyncio.run(scenario())
    state = context.store.get_efa_traffic_state("cluster-a/node-a/job-a/attempt-a")
    assert state.signal.value == "SPIKE"
    assert state.spike_acknowledged_by == "admin@example.com"
    assert state.spike_acknowledgement_reason == "Expected checkpoint traffic burst"
    evidence = context.store.list_raw_evidence(
        "cluster-a", node_id="node-a", attempt_id="attempt-a"
    )
    assert any(item.kind.value == "ADMIN_ACTION" for item in evidence)


def test_admin_can_accept_new_efa_baseline_and_stale_event_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")
    context = ApplicationContext(execution_token=ADMIN_TOKEN)
    observe_running_attempt(context.store, NOW)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry("efa-rebase-baseline", 1_000, NOW).model_dump(
                    mode="json"
                ),
            )
            first_spike = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry(
                    "efa-rebase-spike", 3_000, NOW + timedelta(seconds=15)
                ).model_dump(mode="json"),
            )
            old_event_id = first_spike.json()["findings"][0]["event_id"]
            common = {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "job_id": "job-a",
                "attempt_id": "attempt-a",
                "operator": "admin@example.com",
                "reason": "Approved new communication phase",
            }
            accepted = await client.post(
                "/v1/efa-traffic/admin-actions",
                json={
                    **common,
                    "event_id": old_event_id,
                    "action": "ACCEPT_NEW_BASELINE",
                },
                headers={"X-GPU-Fault-Execution-Token": ADMIN_TOKEN},
            )
            normal = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry(
                    "efa-rebase-normal", 3_300, NOW + timedelta(seconds=30)
                ).model_dump(mode="json"),
            )
            second_spike = await client.post(
                "/v1/collector-events/host-telemetry",
                json=efa_telemetry(
                    "efa-rebase-second-spike", 7_000, NOW + timedelta(seconds=45)
                ).model_dump(mode="json"),
            )
            stale = await client.post(
                "/v1/efa-traffic/admin-actions",
                json={
                    **common,
                    "event_id": old_event_id,
                    "action": "ACKNOWLEDGE_TRANSIENT",
                },
                headers={"X-GPU-Fault-Execution-Token": ADMIN_TOKEN},
            )

        assert accepted.status_code == 200
        assert accepted.json()["resulting_signal"] == "NORMAL"
        assert accepted.json()["resulting_baseline_bytes_per_second"] == 3_000
        assert normal.json()["findings"] == []
        assert second_spike.json()["findings"][0]["event_id"].endswith(
            "-efa-traffic-spike"
        )
        assert stale.status_code == 409
        assert "active SPIKE transition" in stale.json()["detail"]

    asyncio.run(scenario())
    state = context.store.get_efa_traffic_state("cluster-a/node-a/job-a/attempt-a")
    assert state.signal.value == "SPIKE"
    assert state.baseline_bytes_per_second == 3_060
    assert state.active_spike_event_id == "efa-rebase-second-spike-efa-traffic-spike"


def test_efa_rebaseline_is_durable_and_idempotent_in_sqlite(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")
    path = tmp_path / "efa-admin.db"
    first = SqliteStore(str(path))
    policy = NodeHealthPolicy(first)
    observe_running_attempt(first, NOW)
    policy.evaluate_metrics(efa_telemetry("sqlite-efa-baseline", 1_000, NOW))
    finding = policy.evaluate_metrics(
        efa_telemetry("sqlite-efa-spike", 3_000, NOW + timedelta(seconds=15))
    )[0]
    decision = first.apply_efa_traffic_admin_action(
        state_key="cluster-a/node-a/job-a/attempt-a",
        event_id=finding.event_id,
        action=EfaTrafficAdminAction.ACCEPT_NEW_BASELINE,
        operator="admin@example.com",
        reason="Approved sustained communication phase",
        decided_at=NOW + timedelta(seconds=20),
    )
    first.close()

    second = SqliteStore(str(path))
    try:
        state = second.get_efa_traffic_state("cluster-a/node-a/job-a/attempt-a")
        duplicate = second.apply_efa_traffic_admin_action(
            state_key="cluster-a/node-a/job-a/attempt-a",
            event_id=finding.event_id,
            action=EfaTrafficAdminAction.ACCEPT_NEW_BASELINE,
            operator="admin@example.com",
            reason="Approved sustained communication phase",
            decided_at=NOW + timedelta(seconds=30),
        )
    finally:
        second.close()

    assert state.signal.value == "NORMAL"
    assert state.baseline_bytes_per_second == 3_000
    assert duplicate == decision


def test_cpu_finding_requests_host_validation() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry("cpu-saturation", "cpu_usage_percent", 99).model_dump(
                    mode="json"
                ),
            )

        incident = context.store.get_incident(response.json()["incident_ids"][0])
        workflow = context.store.get_workflow(incident.workflow_request_id)
        assert [item.operation.value for item in workflow.official_steps] == [
            "FREEZE_EVIDENCE",
            "VALIDATE_HOST",
        ]

    asyncio.run(scenario())


def test_host_telemetry_endpoint_builds_quarantine_workflow() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry("host-critical", "smart_health_failed", 1).model_dump(
                    mode="json"
                ),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["findings"][0]["category"] == "STORAGE"
        workflow = context.store.get_workflow(body["workflow_request_ids"][0])
        assert [item.operation.value for item in workflow.official_steps] == [
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "QUARANTINE",
        ]

    asyncio.run(scenario())


def test_node_log_endpoint_classifies_mce_and_nccl() -> None:
    context = build_context()
    batch = NodeLogBatch(
        batch_id="logs-1",
        cluster_id="cluster-a",
        node_id="node-a",
        collected_at=NOW,
        runtime_profile_version="simulated-v1",
        entries=[
            NodeLogEntry(
                entry_id="journal-1",
                source="dmesg",
                observed_at=NOW,
                message="mce: Hardware Error: Machine check",
            ),
            NodeLogEntry(
                entry_id="training-1",
                source="training-log",
                observed_at=NOW,
                message="NCCL WARN collective timeout error",
            ),
        ],
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/node-logs", json=batch.model_dump(mode="json")
            )
        assert response.status_code == 200
        body = response.json()
        assert {item["category"] for item in body["findings"]} == {"MCE", "NCCL"}
        assert {item["recommended_action"] for item in body["findings"]} == {
            "QUARANTINE",
            "RUN_DIAGNOSTICS",
        }
        nccl = next(item for item in body["findings"] if item["category"] == "NCCL")
        incident = context.store.get_incident_by_event(nccl["event_id"])
        assert incident is not None
        workflow = context.store.get_workflow(incident.workflow_request_id)
        assert [item.operation.value for item in workflow.official_steps] == [
            "FREEZE_EVIDENCE",
            "VALIDATE_FABRIC",
        ]

    asyncio.run(scenario())


def test_a_tcp_retransmit_raises_no_finding_and_ships_no_batch() -> None:
    """Retransmits are congestion control, not an error counter.

    Every other METRIC_RULES entry is an error counter where a delta of 1 is
    genuinely abnormal, and the collector reports this one as a delta of
    /proc/net/snmp RetransSegs. A threshold of 1.0 therefore fired on every
    collection cycle on every node -- measured at ~155 RUN_DIAGNOSTICS
    workflows an hour across a 4-node p5en fleet, each with its own incident.
    """

    assert "tcp_retransmits_delta" not in NodeHealthPolicy.METRIC_RULES
    policy = NodeHealthPolicy(build_store())

    findings = policy.evaluate_metrics(
        host_telemetry_batch(
            "tcp-retransmit-only",
            NOW,
            [HostMetricSample(name="tcp_retransmits_delta", value=4200)],
        )
    )

    assert findings == [], "a retransmit delta must not mint a RUN_DIAGNOSTICS workflow"


def test_genuine_network_error_counters_keep_their_finding() -> None:
    """The removal is one metric, not the network category."""

    for name in ("network_errors_delta", "network_drops_delta", "rdma_errors_delta"):
        policy = NodeHealthPolicy(build_store())
        findings = policy.evaluate_metrics(
            host_telemetry_batch(
                f"{name}-guard",
                NOW,
                [HostMetricSample(name=name, value=1, device="eth0")],
            )
        )
        assert [item.metric_name for item in findings] == [name], (name, findings)
        assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS
