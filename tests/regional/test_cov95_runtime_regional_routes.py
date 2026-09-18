from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded
from gpu_fault.fleet import AgentTransitionRequest
from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.regional import (
    RegionalExecutorReadinessRequest,
    RemoteCommandClaimRequest,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.fleet._support import NOW
from tests.regional._cov95_runtime_routes import HEADERS, Routes
from tests.regional._cov95_runtime_routes import routes_fixture as routes_fixture
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("rotation", [False, True])
def test_cluster_listing_redacts_current_and_retiring_credentials(
    routes: Routes, rotation: bool
) -> None:
    routes.context.store.save_regional_cluster(
        CLUSTER_A.model_copy(
            update={
                "retiring_token_sha256": "b" * 64,
                "token_rotation_expires_at": NOW + timedelta(minutes=10),
            }
            if rotation
            else {}
        )
    )
    response = routes.client.get("/v1/regional/clusters", headers=HEADERS)
    assert response.status_code == 200
    (record,) = response.json()
    assert "token_sha256" not in record
    assert "retiring_token_sha256" not in record
    assert record["token_sha256_present"] is True
    assert record["token_sha256_length"] == 64
    assert record["retiring_token_sha256_present"] is rotation
    assert record["retiring_token_sha256_length"] == (64 if rotation else 0)


@pytest.mark.parametrize("token", [None, "wrong"])
def test_cluster_listing_refuses_missing_or_wrong_execution_credentials(
    routes: Routes, token: str | None
) -> None:
    response = routes.client.get(
        "/v1/regional/clusters",
        headers={} if token is None else {"X-GPU-Fault-Execution-Token": token},
    )
    assert response.status_code == 403
    assert routes.io.calls == []


@pytest.mark.parametrize("endpoint", ["claim", "readiness"])
def test_executor_routes_require_a_bound_cluster_before_any_store_call(
    routes: Routes, endpoint: str
) -> None:
    payload = (
        RemoteCommandClaimRequest(executor_id="unit")
        if endpoint == "claim"
        else RegionalExecutorReadinessRequest(executor_id="unit")
    )
    response = routes.client.post(
        f"/v1/regional/executors/{endpoint}", json=payload.model_dump(mode="json")
    )
    assert response.status_code == 401
    assert routes.io.calls == []


@pytest.mark.parametrize(
    "defect",
    [
        "healthy",
        "no-owner",
        "unsupported-owner",
        "old-claim",
        "old-backlog",
        "protocol",
    ],
)
def test_executor_readiness_reports_the_concrete_admission_failure(
    monkeypatch: pytest.MonkeyPatch, routes: Routes, defect: str
) -> None:
    health = {
        "pending_owner_counts": {
            "foreign-owner" if defect == "unsupported-owner" else "node-owner": 1
        },
        "oldest_unclaimed_age_seconds": 31.0 if defect == "old-backlog" else 0.0,
        "open_total": 1,
        "pending_total": 1,
    }
    monkeypatch.setattr(
        routes.context.store, "remote_command_cluster_health", lambda cluster: health
    )
    probe = RegionalExecutorReadinessRequest(
        executor_id="unit",
        execution_owners=[] if defect == "no-owner" else ["node-owner"],
        last_successful_claim_age_seconds=31.0 if defect == "old-claim" else 0.0,
    )
    if defect == "protocol":
        probe = probe.model_copy(update={"executor_protocol_version": 99})
    response = routes.client.post(
        "/v1/regional/executors/readiness",
        json=probe.model_dump(mode="json"),
        headers=HEADERS,
    )
    assert response.status_code == (200 if defect == "healthy" else 503), response.text
    body = response.json()
    assert body["ready"] is (defect == "healthy")
    expected = {
        "no-owner": "advertised no execution owners",
        "unsupported-owner": "does not advertise",
        "old-claim": "last successful claim",
        "old-backlog": "has been unclaimed",
        "protocol": "protocol version mismatch",
    }
    if defect != "healthy":
        assert any(expected[defect] in reason for reason in body["reasons"]), body
    assert body["cluster_id"] == "cluster-a"


@pytest.mark.parametrize(
    "error",
    [
        StoreIoCapacityExceeded("unit capacity"),
        RequestDeadlineExceeded("unit deadline"),
    ],
)
def test_executor_readiness_preserves_capacity_vs_deadline_failure(
    routes: Routes, error: Exception
) -> None:
    routes.io.error = error
    response = routes.client.post(
        "/v1/regional/executors/readiness",
        json=RegionalExecutorReadinessRequest(executor_id="unit").model_dump(
            mode="json"
        ),
        headers=HEADERS,
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["detail"] == (
        "request deadline exceeded"
        if isinstance(error, RequestDeadlineExceeded)
        else "store I/O capacity exceeded"
    )


@pytest.mark.parametrize(
    "state",
    [
        "missing",
        "no-workflow",
        "missing-workflow",
        "foreign",
        "failed",
        "quarantined",
        "restored",
        "spare",
    ],
)
def test_incident_ownership_never_invents_terminal_or_released_custody(
    routes: Routes, state: str
) -> None:
    if state != "missing":
        incident = fault_incident(
            "unit-incident",
            "unit-event",
            cluster_id="cluster-b" if state == "foreign" else "cluster-a",
            workflow_request_id=None if state == "no-workflow" else "unit-workflow",
        )
        routes.context.store.save_incident(incident)
    known = state in {"failed", "quarantined", "restored", "spare"}
    if known:
        completed = (
            [WorkflowOperation.QUARANTINE]
            if state in {"quarantined", "restored"}
            else []
        )
        if state == "restored":
            completed.append(WorkflowOperation.RESTORE_SCHEDULING)
        routes.context.store.save_workflow(
            workflow_request(
                "unit-workflow",
                "unit-incident",
                status=WorkflowStatus.FAILED,
                completed_operations=completed,
                official_steps=[workflow_step(operation) for operation in completed],
                step_executions=(
                    [
                        workflow_step_execution(
                            0,
                            WorkflowOperation.REPLACE_NODE,
                            details={"action": "SPARE_FAILOVER"},
                        )
                    ]
                    if state == "spare"
                    else [
                        workflow_step_execution(index, operation, phase="official")
                        for index, operation in enumerate(completed)
                    ]
                ),
            )
        )
    response = routes.client.get(
        "/v1/regional/executors/incident-ownership",
        params={"incident_id": "unit-incident"},
        headers=HEADERS,
    )
    if state == "foreign":
        assert response.status_code == 403
        return
    assert response.status_code == 200
    body = response.json()
    assert body["known"] is known
    assert body["terminal"] is known
    assert body["quarantine_hold"] is (state in {"quarantined", "spare"})
    if state == "missing-workflow":
        assert body["workflow_request_id"] == "unit-workflow"


def test_hyperpod_outcome_requires_the_same_reserved_request_identity(
    routes: Routes,
) -> None:
    record = HyperPodSubmissionRecord(
        cluster_name=CLUSTER_A.hyperpod_cluster_name,
        idempotency_key="unit/workflow",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )
    payload = {"cluster_id": "cluster-a", "record": record.model_dump(mode="json")}
    endpoint = "/v1/regional/executors/hyperpod-submissions"
    missing = routes.client.get(
        endpoint,
        params={
            "cluster_name": record.cluster_name,
            "idempotency_key": record.idempotency_key,
        },
        headers=HEADERS,
    )
    assert missing.status_code == 200 and missing.json() is None
    assert (
        routes.client.post(
            endpoint + "/outcome", json=payload, headers=HEADERS
        ).status_code
        == 409
    )
    first = routes.client.post(endpoint + "/reserve", json=payload, headers=HEADERS)
    second = routes.client.post(endpoint + "/reserve", json=payload, headers=HEADERS)
    assert first.json()["reserved"] is True
    assert second.json()["reserved"] is False
    changed = record.model_copy(update={"requested_node_identifiers": ["node-b"]})
    conflict = routes.client.post(
        endpoint + "/outcome",
        json={"cluster_id": "cluster-a", "record": changed.model_dump(mode="json")},
        headers=HEADERS,
    )
    assert conflict.status_code == 409
    accepted = routes.client.post(endpoint + "/outcome", json=payload, headers=HEADERS)
    assert accepted.status_code == 200
    assert accepted.json()["reserved"] is False
    assert (
        routes.context.store.get_hyperpod_submission(
            record.cluster_name, record.idempotency_key
        )
        == record
    )


@pytest.mark.parametrize("defect", ["header", "provider-cluster"])
@pytest.mark.parametrize("operation", ["reserve", "outcome"])
def test_provider_submission_scope_mismatch_is_rejected_before_store_io(
    routes: Routes, defect: str, operation: str
) -> None:
    record = HyperPodSubmissionRecord(
        cluster_name="foreign"
        if defect == "provider-cluster"
        else CLUSTER_A.hyperpod_cluster_name,
        idempotency_key="unit",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )
    response = routes.client.post(
        "/v1/regional/executors/hyperpod-submissions/" + operation,
        json={
            "cluster_id": "cluster-b" if defect == "header" else "cluster-a",
            "record": record.model_dump(mode="json"),
        },
        headers=HEADERS,
    )
    assert response.status_code == 403
    assert routes.io.calls == []


@pytest.mark.parametrize("defect", ["disabled", "missing", "ambiguous", "draining"])
def test_spare_agent_health_does_not_accept_unavailable_or_ambiguous_membership(
    routes: Routes, defect: str
) -> None:
    aliases = ["node-a"]
    if defect == "disabled":
        routes.context.fleet_registry = None
    elif defect == "missing":
        aliases = ["unknown"]
    elif defect == "ambiguous":
        aliases = ["node-a", "node-b"]
    else:
        routes.context.fleet_registry.drain_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=1, transition_id="unit", reason="unit"
            ),
        )
    response = routes.client.post(
        "/v1/regional/executors/spares/health",
        json={"cluster_id": "cluster-a", "node_aliases": aliases},
        headers=HEADERS,
    )
    assert response.status_code == (503 if defect == "disabled" else 200)
    if defect != "disabled":
        assert response.json()["ready"] is False
        assert len(response.json()["reasons"]) == 1
