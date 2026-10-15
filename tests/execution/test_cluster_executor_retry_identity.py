"""Which executor posts carry a retry identity, and the readiness probe's repeat.

The transport repeats a post whose reused connection failed after the write
only when the post carries a retry identity. The executor gives one to every
post a repeat cannot harm -- the lease-fenced result, renewal and progress
(``X-GPU-Fault-Command-ID``) and the read-only readiness probe
(``X-GPU-Fault-Request-ID``) -- and none to a claim, which would lease more
commands the second time. The probe runs in a fresh process, so its first
connection is new and a connection closed before any status line is asked
once more by the client itself.
"""

from __future__ import annotations

import json
from http.client import RemoteDisconnected
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from gpu_fault.cluster_executor import regional_client
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalFleetRegistry,
    RegionalHyperPodSubmissionStore,
    command_identity_headers,
)
from gpu_fault.fleet import AgentTransitionRequest
from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.regional import (
    RemoteCommandResult,
    RemoteCommandStatus,
    RemoteEvidenceCaptureRequest,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import EvidenceKind
from gpu_fault.transport.http_client import COMMAND_ID_HEADER, REQUEST_ID_HEADER
from tests.fleet._support import NOW
from tests.regional._cov95_runtime_client import client
from tests.regional._regional_support import enqueue_remote_command, spare_alert

COMMAND_ID = "remote-" + "e" * 24
READY = {"ready": True, "reasons": []}


def leased() -> Any:
    return enqueue_remote_command(InMemoryStore(), COMMAND_ID).model_copy(
        update={"status": RemoteCommandStatus.LEASED, "lease_token": "unit-lease"}
    )


def identities(request: Any) -> dict[str, str]:
    """The retry identities a request carries (urllib capitalises names)."""

    headers = {key.lower(): value for key, value in request.header_items()}
    return {
        name: headers[name.lower()]
        for name in (COMMAND_ID_HEADER, REQUEST_ID_HEADER, "Idempotency-Key")
        if name.lower() in headers
    }


@pytest.mark.parametrize("post", ["complete", "renew", "progress"])
def test_a_lease_fenced_post_names_its_command(
    monkeypatch: pytest.MonkeyPatch, post: str
) -> None:
    command = leased()
    api, wire = client(monkeypatch, command.model_dump(mode="json"))
    if post == "complete":
        api.complete(
            command,
            RemoteCommandResult(
                lease_token="unit-lease", status=RemoteCommandStatus.SUCCEEDED
            ),
        )
    elif post == "renew":
        api.renew(command, "unit-executor", 30)
    else:
        api.progress(command, "unit-executor", {"0": {"status": "SUCCEEDED"}})

    [(request, _options)] = wire.calls
    assert identities(request) == {COMMAND_ID_HEADER: COMMAND_ID}


def test_a_claim_carries_no_retry_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    api, wire = client(monkeypatch, {"commands": []})
    assert api.claim("unit-executor", max_commands=1, lease_seconds=30) == []
    [(request, _options)] = wire.calls
    assert identities(request) == {}, "a repeated claim would lease more commands"


def test_the_failure_report_for_an_unparseable_command_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poison = {"command_id": COMMAND_ID, "lease_token": "unit-lease", "status": "?"}

    def respond(request: Any) -> Any:
        return {"commands": [poison]} if request.full_url.endswith("/claim") else {}

    api, wire = client(monkeypatch, respond)
    assert api.claim("unit-executor", max_commands=1, lease_seconds=30) == []
    claim, report = (request for request, _options in wire.calls)
    assert identities(claim) == {}
    assert identities(report) == {COMMAND_ID_HEADER: COMMAND_ID}
    assert json.loads(report.data)["status_source"] == "executor-rejected"


@pytest.mark.parametrize(
    "command_id", ["remote-with space", "remote-é", "remote-\nsplit", ""]
)
def test_an_id_a_header_cannot_carry_is_sent_without_one(command_id: str) -> None:
    assert command_identity_headers(command_id) == {}
    assert command_identity_headers(COMMAND_ID) == {COMMAND_ID_HEADER: COMMAND_ID}


def closed() -> URLError:
    return URLError(RemoteDisconnected("Remote end closed connection without response"))


def readiness(api: Any) -> dict[str, Any]:
    return api.readiness(
        "unit-executor",
        execution_owners=["gpu-fault-kubernetes-adapter"],
        last_successful_claim_age_seconds=None,
    )


def test_readiness_carries_a_request_id_and_asks_once_more_after_a_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[Any] = [closed(), READY]
    api, wire = client(monkeypatch, lambda _request: answers.pop(0))
    pauses: list[float] = []
    monkeypatch.setattr(regional_client.time, "sleep", pauses.append)

    assert readiness(api) == READY

    first, second = (request for request, _options in wire.calls)
    assert set(identities(first)) == {REQUEST_ID_HEADER}
    assert identities(second) == identities(first), "one probe, one request id"
    assert first.data == second.data
    assert pauses == [regional_client.CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS]


@pytest.mark.parametrize("failure", ["verdict", "timeout", "closed-twice"])
def test_readiness_is_not_asked_again_for_a_verdict_or_a_slow_answer(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    answers: list[Any] = {
        "verdict": [
            HTTPError("https://unit.invalid", 503, "not ready", {}, None)  # type: ignore[arg-type]
        ],
        "timeout": [TimeoutError("unit read timeout")],
        "closed-twice": [closed(), closed()],
    }[failure]
    api, wire = client(monkeypatch, lambda _request: answers.pop(0))
    monkeypatch.setattr(regional_client.time, "sleep", lambda _seconds: None)

    with pytest.raises(ClusterExecutorError):
        readiness(api)

    assert len(wire.calls) == (2 if failure == "closed-twice" else 1)


class Recorded(Exception):
    """What the wire raises once it has seen the request; no answer is needed."""


def other_posts(api: Any) -> dict[str, Any]:
    fleet = RegionalFleetRegistry(api, now=lambda: NOW)
    submissions = RegionalHyperPodSubmissionStore(api)
    transition = AgentTransitionRequest(
        expected_generation=1, transition_id="unit-transition", reason="unit"
    )
    record = HyperPodSubmissionRecord(
        cluster_name="unit-hyperpod",
        idempotency_key="workflow/step",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )
    evidence = RemoteEvidenceCaptureRequest(
        cluster_id="cluster-a",
        record_id="unit-evidence",
        node_id="node-a",
        kind=EvidenceKind.NVIDIA_KERNEL,
        observed_at=NOW,
        payload={},
    )
    return {
        "node-inventory": lambda: api.report_node_inventory(
            {"cluster_id": "cluster-a", "node_names": ["node-a"]}
        ),
        "node-action-keys": lambda: api.node_action_keys(["node-a"]),
        "fleet-readiness": lambda: fleet.readiness("cluster-a", ["node-a"]),
        "drain": lambda: fleet.drain_agent("cluster-a", "node-a", transition),
        "revoke": lambda: fleet.revoke_agent("cluster-a", "node-a", transition),
        "spare-health": lambda: fleet.spare_health_reasons(
            cluster_id="cluster-a",
            node_aliases=["node-a"],
            incident_id="unit-incident",
            observed_after=NOW,
        ),
        "advisory": lambda: fleet.save_notification_if_absent(spare_alert("cluster-a")),
        "evidence": lambda: fleet.capture_evidence(evidence),
        "submission-outcome": lambda: submissions.save_hyperpod_submission(record),
        "reactivate": lambda: fleet.reactivate_agent("cluster-a", "node-a", transition),
        "submission-reserve": lambda: submissions.reserve_hyperpod_submission(record),
    }


#: Posts a repeat would answer differently: a reactivation 409s, a second
#: reservation answers ``reserved=False`` (and the submission fails closed).
UNREPEATABLE = {"reactivate", "submission-reserve"}


@pytest.mark.parametrize("post", sorted(other_posts(None)))
def test_only_posts_whose_repeat_is_answered_alike_carry_a_request_identity(
    monkeypatch: pytest.MonkeyPatch, post: str
) -> None:
    api, wire = client(monkeypatch, Recorded())

    with pytest.raises(Recorded):
        other_posts(api)[post]()

    [(request, _options)] = wire.calls
    assert request.get_method() == "POST"
    expected = set() if post in UNREPEATABLE else {REQUEST_ID_HEADER}
    assert set(identities(request)) == expected
