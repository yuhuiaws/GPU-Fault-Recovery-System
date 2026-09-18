from __future__ import annotations

import io
import json
import ssl
from datetime import timedelta
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
    RegionalHyperPodSubmissionStore,
    RegionalIncidentOwnershipProvider,
)
from gpu_fault.fleet import AgentLifecycleState, AgentTransitionRequest
from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.regional import (
    RemoteCommandResult,
    RemoteCommandStatus,
    RemoteEvidenceCaptureRequest,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from tests.fleet._support import NOW, heartbeat, registry, signed
from tests.regional._cov95_runtime_client import client
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime
from tests.regional._regional_support import enqueue_remote_command, spare_alert


def leased_command() -> Any:
    return enqueue_remote_command(InMemoryStore(), "remote-" + "a" * 24).model_copy(
        update={"status": RemoteCommandStatus.LEASED, "lease_token": "unit-lease"}
    )


@pytest.mark.parametrize("field", ["base_url", "cluster_id", "token"])
@pytest.mark.parametrize("value", [" ", "unit\nunsafe", "unit\x7funsafe"])
def test_client_rejects_unsafe_connection_identity_without_transport(
    field: str, value: str
) -> None:
    values = {
        "base_url": "https://unit.invalid",
        "cluster_id": "cluster-a",
        "token": "a" * 32,
    }
    values[field] = value
    with pytest.raises(ClusterExecutorError, match="empty|control characters"):
        RegionalExecutorClient(**values)


@pytest.mark.parametrize(
    "error",
    [
        URLError("unit disconnected"),
        TimeoutError("unit timeout"),
        ssl.SSLError("unit TLS rejection"),
        ConnectionResetError("unit reset"),
        IncompleteRead(b"unit partial"),
    ],
)
def test_no_transport_verdict_is_not_misclassified_as_an_authentication_denial(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    api, wire = client(monkeypatch, error)
    with pytest.raises(ClusterExecutorError) as failed:
        api.readiness(
            "unit-executor",
            execution_owners=["unit-owner"],
            last_successful_claim_age_seconds=1,
        )
    assert failed.value.status_code is None
    assert type(error).__name__ in str(failed.value)
    assert len(wire.calls) == 1


@pytest.mark.parametrize("status", [403, 404, 503])
def test_agent_lookup_translates_only_exact_404_to_not_found(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    failure = HTTPError(
        "https://unit.invalid",
        status,
        "unit rejection",
        {},
        io.BytesIO(b"upstream mentions (404)"),
    )
    api, wire = client(monkeypatch, failure)
    fleet = RegionalFleetRegistry(api)
    if status == 404:
        with pytest.raises(KeyError) as error:
            fleet.get_agent("cluster-a", "node-a")
        assert error.value.args == (("cluster-a", "node-a"),)
    else:
        with pytest.raises(ClusterExecutorError) as error:
            fleet.get_agent("cluster-a", "node-a")
        assert error.value.status_code == status
    assert len(wire.calls) == 1


@pytest.mark.parametrize("wait", [0, 7])
def test_claim_timeout_and_wire_body_preserve_long_poll_compatibility(
    monkeypatch: pytest.MonkeyPatch, wait: int
) -> None:
    api, wire = client(monkeypatch, {"commands": []})
    assert (
        api.claim("unit-executor", max_commands=1, lease_seconds=30, wait_seconds=wait)
        == []
    )
    request, options = wire.calls[0]
    payload = json.loads(request.data)
    assert ("wait_seconds" in payload) is bool(wait)
    assert options["timeout"] == api.timeout_seconds + wait
    assert request.get_header("X-gpu-fault-cluster-id") == "cluster-a"
    assert request.get_header("Authorization") == "Bearer " + "a" * 32
    assert payload["executor_compatibility_digest"] == "b" * 64


@pytest.mark.parametrize("defect", ["no-id", "no-lease", "report-failed", "reported"])
def test_poison_command_never_drops_a_valid_sibling_lease(
    monkeypatch: pytest.MonkeyPatch, defect: str, caplog: pytest.LogCaptureFixture
) -> None:
    valid = leased_command()
    malformed = {
        "command_id": "unit/poison",
        "lease_token": "unit-poison-lease",
        "status": "NOT_A_STATUS",
    }
    if defect == "no-id":
        malformed.pop("command_id")
    elif defect == "no-lease":
        malformed.pop("lease_token")

    def respond(request: Any) -> Any:
        if request.full_url.endswith("/claim"):
            return {"commands": [malformed, valid.model_dump(mode="json")]}
        return URLError("unit report failed") if defect == "report-failed" else {}

    api, wire = client(monkeypatch, respond)
    assert api.claim("unit-executor", max_commands=2, lease_seconds=30) == [valid]
    assert len(wire.calls) == (1 if defect in {"no-id", "no-lease"} else 2)
    if len(wire.calls) == 2:
        request, _options = wire.calls[-1]
        assert request.full_url.endswith("/unit%2Fpoison/result"), request.full_url
        body = json.loads(request.data)
        assert (
            body["status"] == "FAILED" and body["status_source"] == "executor-rejected"
        )
        assert body["lease_token"] == "unit-poison-lease"
        assert len(body["error"]) <= 500
    assert "unit-poison-lease" not in caplog.text


@pytest.mark.parametrize("method", ["complete", "progress", "renew"])
def test_result_protocol_uses_real_models_and_preserves_command_identity(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    command = leased_command()
    api, wire = client(monkeypatch, command.model_dump(mode="json"))
    if method == "complete":
        result = api.complete(
            command,
            RemoteCommandResult(
                lease_token="unit-lease", status=RemoteCommandStatus.SUCCEEDED
            ),
        )
    elif method == "progress":
        result = api.progress(
            command,
            "unit-executor",
            {"0": {"status": "SUCCEEDED", "details": {"observed": True}}},
        )
    else:
        result = api.renew(command, "unit-executor", 30)
    assert result == command
    body = json.loads(wire.calls[0][0].data)
    assert body["lease_token"] == "unit-lease"
    assert wire.calls[0][0].full_url.endswith(
        "/" + command.command_id + "/" + ("result" if method == "complete" else method)
    ), wire.calls[0][0].full_url


@pytest.mark.parametrize("method", ["progress", "renew"])
def test_lease_less_command_cannot_send_progress_or_renewal(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    api, wire = client(monkeypatch, {})
    command = leased_command().model_copy(update={"lease_token": None})
    with pytest.raises(ClusterExecutorError, match="without a lease token"):
        if method == "progress":
            api.progress(command, "unit-executor", {"0": {"status": "SUCCEEDED"}})
        else:
            api.renew(command, "unit-executor", 30)
    assert wire.calls == []


@pytest.mark.parametrize(
    "action", ["fence", "list", "get", "drain", "revoke", "evidence"]
)
def test_fleet_proxy_rejects_foreign_cluster_before_http(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    api, wire = client(monkeypatch)
    fleet = RegionalFleetRegistry(api)
    transition = AgentTransitionRequest(
        expected_generation=1, transition_id="unit-transition", reason="unit"
    )
    with pytest.raises(ClusterExecutorError, match="another cluster"):
        if action == "fence":
            fleet.fleet_rollout_fence_deployments("foreign")
        elif action == "list":
            fleet.list_agents("foreign")
        elif action == "get":
            fleet.get_agent("foreign", "node-a")
        elif action == "drain":
            fleet.drain_agent("foreign", "node-a", transition)
        elif action == "revoke":
            fleet.revoke_agent("foreign", "node-a", transition)
        else:
            fleet.capture_evidence(
                RemoteEvidenceCaptureRequest(
                    cluster_id="foreign",
                    record_id="unit",
                    node_id="node-a",
                    kind=EvidenceKind.NVIDIA_KERNEL,
                    observed_at=NOW,
                    payload={},
                )
            )
    assert wire.calls == []


@pytest.mark.parametrize("defect", ["none", "generation", "lifecycle", "endpoint"])
def test_maintenance_endpoint_preserves_generation_and_lifecycle_fences(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    record = registry().register(signed(heartbeat("node-a")))
    if defect == "generation":
        record = record.model_copy(update={"generation": 2})
    elif defect == "lifecycle":
        record = record.model_copy(
            update={"lifecycle_state": AgentLifecycleState.REVOKED}
        )
    elif defect == "endpoint":
        record = record.model_copy(update={"endpoint": ""})
    api, _wire = client(monkeypatch, record.model_dump(mode="json"))
    fleet = RegionalFleetRegistry(api)
    if defect == "none":
        assert fleet.maintenance_endpoint("cluster-a", "node-a", 1) == record.endpoint
    else:
        with pytest.raises(
            ValueError, match="generation changed|lifecycle state|no endpoint"
        ):
            fleet.maintenance_endpoint("cluster-a", "node-a", 1)


def test_fleet_proxy_keeps_positive_readiness_transition_notification_and_evidence_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = registry()
    record = local.register(signed(heartbeat("node-a")))
    ready = local.readiness("cluster-a", ["node-a"])
    notice = spare_alert("cluster-a")
    evidence = RawEvidenceRecord(
        record_id="unit-evidence",
        cluster_id="cluster-a",
        node_id="node-a",
        kind=EvidenceKind.NVIDIA_KERNEL,
        observed_at=NOW,
        ingested_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        payload={},
    )
    api, wire = client(monkeypatch)
    fleet = RegionalFleetRegistry(api, now=lambda: NOW)
    wire.response = {
        "cluster_id": "cluster-a",
        "fencing_deployment_ids": ["deployment-a"],
    }
    assert fleet.fleet_rollout_fence_deployments("cluster-a") == ["deployment-a"]
    wire.response = [record.model_dump(mode="json")]
    assert fleet.list_agents() == [record]
    wire.response = ready.model_dump(mode="json")
    assert fleet.endpoint("cluster-a", "node-a") == (record.endpoint, record.generation)
    wire.response = record.model_dump(mode="json")
    transition = AgentTransitionRequest(
        expected_generation=1, transition_id="unit-transition", reason="unit"
    )
    assert fleet.drain_agent("cluster-a", "node-a", transition) == record
    assert fleet.revoke_agent("cluster-a", "node-a", transition) == record
    wire.response = {"ready": False, "reasons": ["HMA_HEALTH_UNKNOWN"]}
    assert fleet.spare_health_reasons(
        cluster_id="cluster-a",
        node_aliases=["node-a"],
        incident_id="unit-incident",
        observed_after=NOW,
    ) == ["HMA_HEALTH_UNKNOWN"]
    wire.response = notice.model_dump(mode="json")
    assert fleet.save_notification_if_absent(notice) == notice
    wire.response = evidence.model_dump(mode="json")
    assert (
        fleet.capture_evidence(
            RemoteEvidenceCaptureRequest(
                cluster_id="cluster-a",
                record_id="unit-evidence",
                node_id="node-a",
                kind=EvidenceKind.NVIDIA_KERNEL,
                observed_at=NOW,
                payload={},
            )
        )
        == evidence
    )
    assert all(
        request.get_header("X-gpu-fault-cluster-id") == "cluster-a"
        for request, _ in wire.calls
    ), "a fleet proxy request lost its local cluster binding"


def test_unready_fleet_proxy_endpoint_cannot_be_used_for_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = registry()
    report = local.readiness("cluster-a", ["node-a"])
    api, _wire = client(monkeypatch, report.model_dump(mode="json"))
    with pytest.raises(ValueError, match="not fleet-ready"):
        RegionalFleetRegistry(api).endpoint("cluster-a", "node-a")


@pytest.mark.parametrize("reserved", [False, True])
def test_provider_submission_proxy_distinguishes_reservation_from_a_read(
    monkeypatch: pytest.MonkeyPatch, reserved: bool
) -> None:
    record = HyperPodSubmissionRecord(
        cluster_name="unit-hyperpod",
        idempotency_key="workflow/step",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )
    api, wire = client(
        monkeypatch, {"record": record.model_dump(mode="json"), "reserved": reserved}
    )
    store = RegionalHyperPodSubmissionStore(api)
    assert store.reserve_hyperpod_submission(record) == (record, reserved)
    wire.response = {}
    store.save_hyperpod_submission(record)
    assert wire.calls[-1][0].full_url.endswith("/hyperpod-submissions/outcome"), (
        wire.calls[-1][0].full_url
    )
    wire.response = record.model_dump(mode="json")
    assert store.get_hyperpod_submission("unit-hyperpod", "workflow/step") == record
    request = wire.calls[-1][0]
    assert request.get_method() == "GET"
    assert parse_qs(urlsplit(request.full_url).query) == {
        "cluster_name": ["unit-hyperpod"],
        "idempotency_key": ["workflow/step"],
    }
    wire.response = None
    with pytest.raises(KeyError):
        store.get_hyperpod_submission("unit-hyperpod", "absent")


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_incident_ownership_requires_known_and_terminal_before_takeover(
    monkeypatch: pytest.MonkeyPatch, known: bool, terminal: bool
) -> None:
    api, wire = client(
        monkeypatch,
        {"incident_id": "unit", "known": known, "terminal": terminal}
        if known
        else None,
    )
    ownership = RegionalIncidentOwnershipProvider(api)
    assert ownership.incident_workflow_is_terminal("unit") is (known and terminal)
    assert parse_qs(urlsplit(wire.calls[0][0].full_url).query) == {
        "incident_id": ["unit"]
    }
