from __future__ import annotations

import base64

import pytest

from gpu_fault.async_store import StoreIoCapacityExceeded
from gpu_fault.installation_resources import InstallationResource
from gpu_fault.models import Environment, RuntimeProfile
from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.telemetry import EvidenceKind
from tests._builders import processor_request
from tests.app_services._cov95_runtime_api import Api
from tests.app_services._cov95_runtime_api import api_fixture as api_fixture
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.completion.test_training_health import NOW, heartbeat, observation
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("limit", [0, 1, 1000, 1001])
def test_raw_evidence_limit_is_checked_before_reading_the_store(
    api: Api, limit: int
) -> None:
    response = api.client.get("/v1/evidence/cluster-a", params={"limit": limit})
    valid = 1 <= limit <= 1000
    assert response.status_code == (200 if valid else 422)
    assert api.io.calls == (["list_raw_evidence"] if valid else [])
    if valid:
        assert response.json() == []
    else:
        assert response.json()["detail"] == "limit must be between 1 and 1000"


@pytest.mark.parametrize("mode", ["nonregional", "registered", "unregistered"])
def test_runtime_profile_registration_requires_the_regional_cluster_to_exist(
    api: Api, mode: str
) -> None:
    api.context.regional_mode = mode != "nonregional"
    if mode == "registered":
        api.context.store.save_regional_cluster(CLUSTER_A)
    profile = RuntimeProfile(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        claims=[],
        observed=[],
        profile_version="unit-empty-profile",
    )
    response = api.client.post(
        "/v1/runtime-profiles", json=profile.model_dump(mode="json")
    )
    if mode == "unregistered":
        assert response.status_code == 403
        with pytest.raises(KeyError):
            api.context.store.get_profile(profile.profile_version)
    else:
        assert response.status_code == 200
        saved = api.context.store.get_profile(profile.profile_version)
        assert saved.cluster_id == "cluster-a"
        assert saved.capabilities == []


@pytest.mark.parametrize("mismatch", ["site", "key", "none"])
def test_installation_record_update_keeps_path_and_body_identity_bound(
    api: Api, mismatch: str
) -> None:
    resource = InstallationResource(
        site_id="unit-site",
        resource_key="unit/component",
        resource_type="cpu_eks",
        resource_id="unit-resource",
        ownership="EXTERNAL",
        delete_policy="PRESERVE",
    )
    path = (
        "/v1/installation-resources/"
        + ("foreign-site" if mismatch == "site" else resource.site_id)
        + "/"
        + ("foreign/component" if mismatch == "key" else resource.resource_key)
    )
    response = api.client.put(path, json=resource.model_dump(mode="json"))
    if mismatch == "none":
        assert response.status_code == 200
        assert response.json()["resource_key"] == resource.resource_key
        assert api.io.calls == ["save_installation_resource"]
    else:
        assert response.status_code == 409
        assert api.context.store.list_installation_resources() == []
        assert api.io.calls == []


@pytest.mark.parametrize("credential", ["missing", "wrong", "unconfigured"])
def test_processor_status_handler_refuses_credentials_before_queue_diagnostics(
    api: Api, credential: str
) -> None:
    api.client.headers.pop("X-GPU-Fault-Execution-Token")
    if credential == "unconfigured":
        api.context.execution_token = ""
    response = api.client.get(
        "/v1/processor/status",
        headers=(
            {"X-GPU-Fault-Execution-Token": "unit-wrong"}
            if credential != "missing"
            else {}
        ),
    )
    assert response.status_code == 403
    assert api.io.calls == []


@pytest.mark.parametrize("mode", ["pending", "complete", "untyped", "foreign"])
def test_receipt_does_not_leak_another_cluster_or_invent_a_success_status(
    api: Api, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    api.context.regional_mode = True
    item = processor_request("/v1/collector-events/gpu-metrics")
    current = item.model_copy(
        update={
            "cluster_id": "cluster-b" if mode == "foreign" else "cluster-a",
            "status": (
                ProcessorRequestStatus.PENDING
                if mode == "pending"
                else ProcessorRequestStatus.COMPLETED
            ),
            "response_status": None if mode == "untyped" else 201,
            "response_content_type": None if mode == "untyped" else "application/json",
            "response_body_base64": base64.b64encode(b'{"owned":true}').decode(),
        }
    )
    monkeypatch.setattr(
        api.context.store, "get_processor_request", lambda request_id: current
    )
    response = api.client.get(f"/v1/processor/requests/{item.request_id}")
    assert (
        response.status_code
        == {"pending": 202, "complete": 201, "untyped": 500, "foreign": 403}[mode]
    )
    if mode == "foreign":
        assert "owned" not in response.text
    elif mode == "pending":
        assert response.json()["processor_request_id"] == item.request_id
        assert response.json()["status"] == "PENDING"
    else:
        assert response.json() == {"owned": True}
        assert response.headers["X-GPU-Fault-Processor-Status"] == "COMPLETED"
        assert ("content-type" in response.headers) is (mode == "complete")


@pytest.mark.parametrize(
    "path",
    [
        "/v1/evidence/cluster-a",
        "/v1/installation-resources",
        "/v1/processor/status",
        "/v1/processor/requests/unit-request",
    ],
)
def test_route_store_capacity_is_an_explicit_retryable_response(
    api: Api, path: str
) -> None:
    api.io.error = StoreIoCapacityExceeded("synthetic capacity")
    response = api.client.get(path)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json()["detail"] == "store I/O capacity exceeded"


def test_training_progress_node_mismatch_cannot_become_evidence_or_a_notification(
    api: Api,
) -> None:
    api.context.store.save_attempt_observation(observation())
    wrong = heartbeat(0).model_copy(update={"node_id": "node-foreign"})
    response = api.client.post(
        "/v1/training-progress", json=wrong.model_dump(mode="json")
    )
    assert response.status_code == 409
    assert "node does not match allocation" in response.json()["detail"]
    assert api.context.store.list_training_progress("cluster-a", "attempt-a") == []
    assert api.context.store.list_raw_evidence("cluster-a") == []
    assert api.context.store.list_notifications() == []


@pytest.mark.parametrize("has_observation", [False, True])
def test_training_scan_latches_only_findings_delivered_to_incidents(
    api: Api, has_observation: bool
) -> None:
    if has_observation:
        api.context.store.save_attempt_observation(observation())
        api.context.training_health.ingest(heartbeat(0, observed_at=NOW))
        api.context.training_health.ingest(heartbeat(1, observed_at=NOW))
    response = api.client.post("/v1/training-health/cluster-a/scan")
    assert response.status_code == 200
    findings = response.json()["findings"]
    assert len(findings) == (2 if has_observation else 0)
    for finding in findings:
        assert api.context.store.get_incident_by_event(finding["event_id"]) is not None
    second = api.client.post("/v1/training-health/cluster-a/scan")
    assert second.status_code == 200
    assert second.json()["findings"] == []
    assert (
        api.context.store.list_raw_evidence(
            "cluster-a", kind=EvidenceKind.TRAINING_PROGRESS
        )
        == []
    )
