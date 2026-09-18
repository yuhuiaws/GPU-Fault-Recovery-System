from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import IncidentState, RecoveryAction, Severity, WorkloadState
from tests._builders import build_context, node_health_finding
from tests.app_services._cov95_runtime_ingest import NOW
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def finding(identifier: str, metric: str | None, **values: Any) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{identifier}",
        f"event-{identifier}",
        observed_at=NOW,
        category=values.pop("category", NodeHealthCategory.GPU),
        severity=Severity.WARNING,
        reason="synthetic unit observation",
        recommended_action=values.pop(
            "recommended_action", RecoveryAction.COLLECT_EVIDENCE
        ),
        metric_name=metric,
        runtime_profile_version="simulated-v1",
        **values,
    )


@pytest.mark.parametrize(
    ("metric", "category", "notification_count"),
    [
        ("gpu_inventory_mismatch", NodeHealthCategory.GPU, 1),
        ("efa_inventory_mismatch", NodeHealthCategory.RDMA, 1),
        ("unparsed_xid_line", NodeHealthCategory.GPU, 1),
        ("rdma_link_down", NodeHealthCategory.RDMA, 1),
        (None, NodeHealthCategory.RDMA, 0),
        ("ordinary_observation", NodeHealthCategory.GPU, 0),
    ],
)
def test_health_finding_marker_precedes_ingestion_and_binds_the_actual_incident(
    monkeypatch: pytest.MonkeyPatch,
    metric: str | None,
    category: NodeHealthCategory,
    notification_count: int,
) -> None:
    context = build_context()
    markers = []
    notified = []
    add_marker = context.completion.add_marker

    def add(value: Any) -> None:
        markers.append(value)
        add_marker(value)

    monkeypatch.setattr(context.completion, "add_marker", add)
    monkeypatch.setattr(context.advisory_notifications, "send", notified.append)
    source = finding("unit", metric, category=category)
    result = NodeHealthIngestionService(context).ingest("unit-batch", [source])
    assert [value.incident_id for value in markers] == ["", result.incident_ids[0]]
    assert result.marker_ids == [source.marker().marker_id]
    assert len(result.notification_ids) == notification_count
    assert notified == result.notification_ids
    assert context.store.get_incident(result.incident_ids[0]).node_ids == ["node-a"]
    assert len(result.workflow_request_ids) == 1
    workflow = context.store.get_workflow(result.workflow_request_ids[0])
    assert [step.operation.value for step in workflow.official_steps] == [
        "FREEZE_EVIDENCE"
    ]


@pytest.mark.parametrize("with_devices", [False, True])
def test_host_resource_notifications_aggregate_devices_and_do_not_rearm_unsettled_incidents(
    monkeypatch: pytest.MonkeyPatch, with_devices: bool
) -> None:
    context = build_context()
    sent = []
    monkeypatch.setattr(context.advisory_notifications, "send", sent.append)
    service = NodeHealthIngestionService(context)
    findings = [
        finding(
            f"first-host_gpu_utilization_percent-{index}-low_gpu_utilization",
            "host_gpu_utilization_percent",
            policy_source="SITE_HOST_RESOURCE_HEALTH",
            device=f"GPU-{index}" if with_devices else None,
            diagnostic_parameters={"signal": "LOW_GPU_UTILIZATION"},
            recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/pytorchjob/unit"],
        )
        for index in range(2)
    ]
    first = service.ingest("first", findings)
    assert all(
        context.store.get_incident(identifier).state is IncidentState.ACTION_PENDING
        for identifier in first.incident_ids
    ), "the duplicate-suppression fixture must have an unsettled diagnostic"
    second = service.ingest(
        "second",
        [
            value.model_copy(
                update={
                    "finding_id": f"finding-second-host_gpu_utilization_percent-{index}-low_gpu_utilization",
                    "event_id": f"event-second-host_gpu_utilization_percent-{index}-low_gpu_utilization",
                }
            )
            for index, value in enumerate(findings)
        ],
    )
    assert len(first.notification_ids) == 1
    assert second.notification_ids == []
    assert sent == first.notification_ids
    note = context.store.get_notification(first.notification_ids[0])
    assert note.incident_id == first.incident_ids[0]
    if with_devices:
        assert "GPU-0" in note.body_text and "GPU-1" in note.body_text
    assert all(
        context.store.get_incident(identifier).node_ids == ["node-a"]
        for identifier in second.incident_ids
    ), "repeated findings must retain the original node ownership"


def test_failed_ingestion_leaves_an_observational_marker_without_a_phantom_incident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_context()

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic incident unavailable")

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", failed)
    with pytest.raises(RuntimeError, match="incident unavailable"):
        NodeHealthIngestionService(context).ingest(
            "unit", [finding("unit", "unparsed_xid_line")]
        )
    (marker,) = context.store.list_markers()
    assert marker.incident_id == ""
    assert context.store.list_incidents_by_state("cluster-a", set(IncidentState)) == []
    assert context.store.list_notifications() == []
