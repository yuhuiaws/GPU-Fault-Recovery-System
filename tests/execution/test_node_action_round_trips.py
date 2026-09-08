"""How many control-plane and TLS round trips one node action costs.

cluster-executor review F6/F7. A single-node RESET_GPU that is still pending
costs the regional executor a full cycle every two seconds, and every cycle
used to pay for the same agent record twice -- once in ``_secret_for_node`` to
learn the key version, once in ``_ssl_context`` to learn the pinned
certificate -- and to build a fresh ``SSLContext`` for each send. Because
``transport/http_client.py`` keys its connection pool on ``id(ssl_context)``,
a new context per send means a new TLS handshake per send and a dead pooled
connection left behind (evicted at eight).

The record is read once per node per cycle and handed to both readers, and the
context is cached per ``(node_id, sha256(certificate))`` so a rotated
certificate still gets a new context.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
)
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.execution._support import StubFleetRegistry

SECRET = "s" * 32
FENCING_TOKEN = 3

# Two real self-signed certificates: ``ssl.create_default_context(cadata=...)``
# parses what it is given, so the cache key can only be exercised with PEM the
# stdlib accepts. Generated for this test file and used nowhere else.
CERTIFICATE_A = """-----BEGIN CERTIFICATE-----
MIIDITCCAgmgAwIBAgIUV8NaIpvA37QbOuZKPUmDK7VPNsIwDQYJKoZIhvcNAQEL
BQAwHzEdMBsGA1UEAwwUZ3B1LWZhdWx0LXRlc3QtYWdlbnQwIBcNMjYwOTA4MTAz
MTQzWhgPMjEyNjA4MTUxMDMxNDNaMB8xHTAbBgNVBAMMFGdwdS1mYXVsdC10ZXN0
LWFnZW50MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvn13ibollVTm
AezARd88Rcz57KRIRQqC9KW7ow6YTPZKgUw/iKPGrCAAsZ5jDLNEWe8Ro/5AQ7sa
J4S/PNEfp8rF0Gt7vTBnt821a7hW+Ik9eb4LyCP3oIRSAviTEbXi2hvnv/JhcX17
0fEGTsMGakyyrC8JOYLPe8J3syLXAm6J0+ExOKJWK4D1tOxfkY5LmEL1C8zd8/A6
NeB9RaGDKfBcNOmBEoktNGI8gbgcgS2eONnFsci16jyUuyoX5SnSPguKryi4F/G3
IgvxIu9USCGCNooIbBuoD7pxSVyU23WvmuE5LShFYfhXA15yx+ClPw0E5vPJPA5I
PceN0fpeBQIDAQABo1MwUTAdBgNVHQ4EFgQUp1jn1TppO1+R+RvwkVMXe6Us48ow
HwYDVR0jBBgwFoAUp1jn1TppO1+R+RvwkVMXe6Us48owDwYDVR0TAQH/BAUwAwEB
/zANBgkqhkiG9w0BAQsFAAOCAQEAa/cgBsel/wT7AowLqDx5tp99G91WEGiBoMto
ciNWObRGmicOEmLKqjzZkdPlZA3U37tiWEkzr5M3ynivRfiyk9ypPwIrCnSgdMZt
ocpJSc4VdNFkb1Jfi3ZmPLtUvdOPrDvRqHS8fdDGy+pMjq0AK/XJA3/Wgx2XiuC1
oETFTUmLVL4XcWwIXxVkzx2XbW44IoQZIRoSEBjbqrDGHDcp9oqqtP9mC/GQuEKj
jdaU0/XDmDjyJCADpexYUIYOjeRe7YKU6A05FiCMk15RFaeLXlDMBQ9qV1zVWo7D
GuGxoF5vcuDHjJ1jWL6oimPW13WnYhn4fTDQW1D5+IK4okpccw==
-----END CERTIFICATE-----
"""

CERTIFICATE_B = """-----BEGIN CERTIFICATE-----
MIIDMTCCAhmgAwIBAgIUGfrt9Ko1iQvwGlUGQQ1v92kkP9gwDQYJKoZIhvcNAQEL
BQAwJzElMCMGA1UEAwwcZ3B1LWZhdWx0LXRlc3QtYWdlbnQtcm90YXRlZDAgFw0y
NjA5MDgxMDMxNTFaGA8yMTI2MDgxNTEwMzE1MVowJzElMCMGA1UEAwwcZ3B1LWZh
dWx0LXRlc3QtYWdlbnQtcm90YXRlZDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBALWaR9v7moIoYHl/ZXj7//cYH0+qaF63a1uuo6v5GEr96DLM7CQeRuyT
mpIC2dCnDktL7LOSaTSzVnfahv+t8hCZ7o/+meHJl91rdEdz+pRlxWcSuKWsvEQ5
4tJnczQ1RXHQSU2MnZ+PX2gKdJcFoEXHn8wtrJC2mw52Ie8z4yfZ7NXfQMCO436K
TvKzeGQc0/tnahd6ppkt1dnlneePHX19fB2EC/2QtgGTgfrJHcb75ndxwVX57RZi
DUB4JJjQTJhYCBjKnuLpLoVAsrzUofCHHxQL+m8b6cJIz2JJM8qxkf2JgJPm97U7
xCAZDerQwB3oaVbqwn9Nc6Pl3//QNpkCAwEAAaNTMFEwHQYDVR0OBBYEFJuSxGoC
8Wlrr6wum0g1XCTNTQ9aMB8GA1UdIwQYMBaAFJuSxGoC8Wlrr6wum0g1XCTNTQ9a
MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAFkuslOgh/zk7APp
FnkZUI20lerRwkthNulPUT++AScKnR+JHG4TF/bzYVJTSDDJSS/EKP5OMe84qkX3
0i3yH9+/zB+cetqMUjV9yUnnQt7N4gr9W8xr7Ov9+Y512YOr8RRl2Ksuse1YH79p
EIgppGlC56WrVaQC/RG3FAHe5GAZXKpbVaiNyYj3hXuQLdtdierJ8010gtUcUXGK
uGpYZ01oTw5CvnwP0vZVD5AjNMN3+gI+0ebCQbJrxn4/3zld8TKuuBtQX3hlVFbu
RN1hP1Tmg1yUfMuYqmO3B4/MxacpjEEhkEgBlYXPz1qdhfpwuXpTcI7WLNW5m2xt
PB7PtdY=
-----END CERTIFICATE-----
"""


class CountingFleetRegistry(StubFleetRegistry):
    """Records every ``get_agent`` and pins a TLS certificate per node.

    ``StubFleetRegistry`` models the addressing contract; what this subclass
    adds is the two things the round-trip count depends on -- a call log and a
    certificate, so ``_ssl_context`` takes its real branch instead of the
    plaintext short circuit.
    """

    def __init__(self, endpoints: dict[str, str], **kwargs: Any) -> None:
        super().__init__(endpoints, **kwargs)
        self.get_agent_calls: list[tuple[str, str]] = []
        self.certificates = {node_id: CERTIFICATE_A for node_id in endpoints}

    def get_agent(self, cluster_id: str, node_id: str) -> Any:
        self.get_agent_calls.append((cluster_id, node_id))
        record = super().get_agent(cluster_id, node_id)
        return SimpleNamespace(
            **{**vars(record), "tls_certificate_pem": self.certificates.get(node_id)}
        )


def step_context(
    adapter: NodeActionWorkflowAdapter,
    *,
    operation: WorkflowOperation = WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    node_ids: list[str] | None = None,
    idempotency_key: str = "workflow/step",
) -> WorkflowStepContext:
    step = workflow_step(
        operation, adapter.owner, node_ids=node_ids or ["node-a"], gpu_uuids=["GPU-a"]
    )
    return WorkflowStepContext(
        workflow=workflow_request(
            "workflow-a",
            "incident-a",
            fencing_token=FENCING_TOKEN,
            official_steps=[step],
        ),
        incident=fault_incident("incident-a", "event-a", fencing_token=FENCING_TOKEN),
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=FENCING_TOKEN),
        idempotency_key=idempotency_key,
    )


def test_one_get_agent_per_node_per_cycle() -> None:
    """The key version and the pinned certificate come from one read.

    Two reads per node per cycle is two control-plane round trips and two store
    reads for one record that cannot change between them; at a two-second poll
    and two replicas that is the executor's largest source of avoidable
    control-plane load while a node action waits.
    """

    registry = CountingFleetRegistry(
        {"node-a": "https://node-a:9099", "node-b": "https://node-b:9099"}
    )

    def sender(_endpoint: str, envelope: Any) -> NodeActionResult:
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=envelope.command.operation,
            status=NodeActionStatus.SUCCEEDED,
        )

    adapter = NodeActionWorkflowAdapter({}, SECRET, sender=sender, registry=registry)

    outcome = adapter.execute(step_context(adapter, node_ids=["node-a", "node-b"]))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert sorted(registry.get_agent_calls) == [
        ("cluster-a", "node-a"),
        ("cluster-a", "node-b"),
    ], (
        "the agent record must be read once per node per cycle, not once per "
        f"reader: {registry.get_agent_calls}"
    )


class RecordingWire:
    """Answers the ledger poll SUCCEEDED and records every ssl_context."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def __call__(
        self, request: Any, timeout: float | None = None, *, ssl_context: Any = None
    ) -> Any:
        self.contexts.append(ssl_context)
        command_id = request.full_url.split("command_id=")[1].split("&")[0]
        body = (
            NodeActionSubmission(
                command_id=command_id,
                state=NodeActionExecutionState.SUCCEEDED,
                result=NodeActionResult(
                    command_id=command_id,
                    operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
                    status=NodeActionStatus.SUCCEEDED,
                ),
            )
            .model_dump_json()
            .encode()
        )

        class Response:
            def read(self) -> bytes:
                return body

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

        return Response()


def wire(monkeypatch: pytest.MonkeyPatch) -> RecordingWire:
    fake = RecordingWire()
    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", fake)
    return fake


def test_ssl_context_is_reused_across_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    """One context per certificate, so the pooled connection survives.

    ``transport/http_client.py`` pools by ``id(ssl_context)``: a fresh context
    per send means a full TLS handshake per send and a pooled connection that
    can never be reused.
    """

    registry = CountingFleetRegistry({"node-a": "https://node-a:9099"})
    fake = wire(monkeypatch)
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=registry)

    for attempt in range(2):
        outcome = adapter.execute(
            step_context(adapter, idempotency_key=f"workflow/step-{attempt}")
        )
        assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome

    assert len(fake.contexts) == 2, fake.contexts
    assert fake.contexts[0] is fake.contexts[1], (
        "each send built its own SSLContext, so every send pays a handshake"
    )


def test_a_rotated_agent_certificate_gets_a_new_ssl_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is keyed by certificate digest, so rotation is not sticky.

    Caching by node alone would keep trusting the retired certificate for the
    life of the executor process -- every send to a re-registered agent failing
    the handshake with no way to recover short of a restart.
    """

    registry = CountingFleetRegistry({"node-a": "https://node-a:9099"})
    fake = wire(monkeypatch)
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=registry)

    adapter.execute(step_context(adapter, idempotency_key="workflow/step-0"))
    registry.certificates["node-a"] = CERTIFICATE_B
    adapter.execute(step_context(adapter, idempotency_key="workflow/step-1"))
    adapter.execute(step_context(adapter, idempotency_key="workflow/step-2"))

    assert fake.contexts[0] is not fake.contexts[1], (
        "a rotated certificate must not reuse the context built for the old one"
    )
    assert fake.contexts[1] is fake.contexts[2], (
        "the rotated certificate's own context must then be reused"
    )
