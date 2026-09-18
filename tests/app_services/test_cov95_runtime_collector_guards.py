from __future__ import annotations

import pytest

from gpu_fault.channel_registry import COLLECTOR_HEALTH_PATH, NODE_LOG_PATH
from gpu_fault.host_health import NodeLogBatch
from gpu_fault.models import EfaTrafficAdminRequest
from gpu_fault.telemetry import CollectorHealthSummary, CollectorKind
from tests._builders import build_context, fault_incident
from tests.app_services._cov95_runtime_api import full_app
from tests.app_services._cov95_runtime_ingest import NOW
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

TOKEN = "synthetic-runtime-operator-token"


@pytest.mark.parametrize(
    "defect",
    [
        "event-type",
        "policy",
        "cluster",
        "node",
        "missing",
        "credential",
        "unconfigured",
    ],
)
def test_efa_operator_decision_refuses_an_unbound_incident_before_state_or_evidence_changes(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    context = build_context(execution_token="" if defect == "unconfigured" else TOKEN)
    incident = fault_incident(
        "unit-incident",
        "unit-event",
        event_type="XID" if defect == "event-type" else "NODE_HEALTH",
        policy_source="OTHER" if defect == "policy" else "SITE_EFA_TRAFFIC",
        cluster_id="cluster-b" if defect == "cluster" else "cluster-a",
        node_ids=["node-b"] if defect == "node" else ["node-a"],
    )
    if defect != "missing":
        context.store.save_incident(incident)
    decisions = []
    monkeypatch.setattr(
        context.store,
        "apply_efa_traffic_admin_action",
        lambda **kwargs: decisions.append(kwargs),
    )
    request = EfaTrafficAdminRequest(
        cluster_id="cluster-a",
        node_id="node-a",
        job_id="job-a",
        attempt_id="attempt-a",
        event_id="unit-event",
        action="ACKNOWLEDGE_TRANSIENT",
        operator="unit-operator",
        reason="owned unit evidence review",
    )
    with full_app(context) as (_app, client):
        response = client.post(
            "/v1/efa-traffic/admin-actions",
            json=request.model_dump(mode="json"),
            headers={
                "X-GPU-Fault-Execution-Token": "wrong"
                if defect == "credential"
                else TOKEN
            },
        )
    assert response.status_code == (
        403
        if defect in {"credential", "unconfigured"}
        else 404
        if defect == "missing"
        else 409
    )
    assert decisions == []
    assert context.store.list_raw_evidence("cluster-a") == []
    assert context.store.list_notifications() == []
    if defect != "missing":
        assert context.store.get_incident(incident.incident_id) == incident


@pytest.mark.parametrize("failed_collection", [False, True])
def test_empty_logs_preserve_collection_failure_without_fabricating_a_fault(
    failed_collection: bool,
) -> None:
    context = build_context()
    batch = NodeLogBatch(
        batch_id="unit-empty-log",
        cluster_id="cluster-a",
        node_id="node-a",
        collected_at=NOW,
        entries=[],
        collection_errors=["synthetic read failed"] if failed_collection else [],
    )
    with full_app(context) as (_app, client):
        response = client.post(NODE_LOG_PATH, json=batch.model_dump(mode="json"))
    assert response.status_code == 200
    assert response.json()["findings"] == []
    (status,) = context.store.list_collector_statuses("cluster-a", "node-a")
    assert status.collector is CollectorKind.NODE_LOGS
    assert (status.last_error_at is not None) is failed_collection
    assert (status.last_success_at is not None) is (not failed_collection)
    assert len(context.store.list_raw_evidence("cluster-a")) == int(failed_collection)
    assert context.store.list_markers() == []


def test_periodic_collector_cannot_publish_an_event_driven_health_summary() -> None:
    context = build_context()
    summary = CollectorHealthSummary(
        summary_id="unit-summary",
        cluster_id="cluster-a",
        node_id="node-a",
        collector=CollectorKind.GPU_METRICS,
        observed_at=NOW,
    )
    with full_app(context) as (_app, client):
        response = client.post(
            COLLECTOR_HEALTH_PATH, json=summary.model_dump(mode="json")
        )
    assert response.status_code == 422
    assert "event-driven collectors" in response.json()["detail"]
    assert context.store.list_collector_statuses("cluster-a") == []
