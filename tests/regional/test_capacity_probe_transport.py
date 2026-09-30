"""CAP probes: fleet executor pins and the supervised local port-forward."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_base as base
from scripts.e2e.regional import capacity_acceptance_traffic as traffic
from scripts.e2e.regional import capacity_probe_transport as transport
from scripts.e2e.regional import capacity_wire as wire
from scripts.e2e.regional.capacity_acceptance_retry import run_claim_retry_proof
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture

ARTIFACT = "e" * 64
DIGEST = "f" * 64


def test_release_metadata_pins_follow_the_fleet_pin_mapping() -> None:
    pins = transport.executor_pins_from_release_metadata(
        {
            "data": {
                "required-regional-executor-artifact-sha256": ARTIFACT.upper(),
                "required-regional-executor-compatibility-digest": "",
            }
        }
    )
    assert pins == {
        "executor_artifact_sha256": ARTIFACT,
        "executor_compatibility_digest": None,
    }, "an empty key declares no pin; digests are normalised to lowercase"
    assert transport.executor_pins_from_release_metadata({"data": {}}) == {
        "executor_artifact_sha256": None,
        "executor_compatibility_digest": None,
    }, "a ConfigMap without executor pins leaves both unset"


@pytest.mark.parametrize(
    "configmap",
    [
        None,
        {"data": []},
        {"data": {"required-regional-executor-artifact-sha256": "not-a-digest"}},
        {"data": {"required-regional-executor-compatibility-digest": 7}},
    ],
)
def test_malformed_release_metadata_pins_are_refused(configmap: Any) -> None:
    with pytest.raises(base.CapError, match="release metadata"):
        transport.executor_pins_from_release_metadata(configmap)


def test_wire_client_presents_the_fleet_pins_like_the_live_executor() -> None:
    client = wire.CapacityWireClient(
        "http://127.0.0.1:18080",
        "token",
        cluster_id="cap-cluster-000",
        executor_pins={
            "executor_artifact_sha256": ARTIFACT,
            "executor_compatibility_digest": DIGEST,
        },
    )
    assert (client.executor_artifact_sha256, client.executor_compatibility_digest) == (
        ARTIFACT,
        DIGEST,
    ), "both pins must reach the production client unchanged"
    bare = wire.CapacityWireClient(
        "http://127.0.0.1:18080",
        "token",
        cluster_id="cap-cluster-000",
        executor_pins={"executor_artifact_sha256": None},
    )
    assert bare.executor_artifact_sha256 is None, "an undeclared pin stays absent"
    with pytest.raises(wire.CapacityWireError, match="unknown executor pin"):
        wire.executor_pin_arguments({"artifact": ARTIFACT})


def test_retry_proof_claims_carry_the_fleet_pins(monkeypatch, tmp_path) -> None:
    bodies: list[dict[str, Any]] = []

    def urlopen(request, **_kwargs):
        from io import BytesIO
        from urllib.response import addinfourl

        bodies.append(json.loads(request.data))
        return addinfourl(BytesIO(b'{"commands": []}'), {}, request.full_url, 200)

    monkeypatch.setattr("gpu_fault.cluster_executor.regional_client.urlopen", urlopen)
    run_claim_retry_proof(
        "http://127.0.0.1:18623",
        "fixture",
        tmp_path,
        lambda: None,
        poll_seconds=0.001,
        executor_pins={
            "executor_artifact_sha256": ARTIFACT,
            "executor_compatibility_digest": DIGEST,
        },
    )
    assert bodies and all(
        body["executor_artifact_sha256"] == ARTIFACT
        and body["executor_compatibility_digest"] == DIGEST
        for body in bodies
    ), "every claim of the production loop must present the release's pins"


class Forward:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.returncode = None if alive else 1
        self.events: list[str] = []

    def poll(self) -> int | None:
        return None if self.alive else self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        self.alive = False
        self.returncode = -15

    def wait(self, *, timeout: float) -> int:
        return 0


def pod_document(*, uid: str = "probe-uid", restarts: int = 0) -> dict[str, Any]:
    return {
        "metadata": {"name": "probe-pod", "uid": uid},
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "probe",
                    "ready": True,
                    "restartCount": restarts,
                    "state": {"running": {}},
                    "lastState": {"terminated": {"reason": "Error"}}
                    if restarts
                    else {},
                }
            ],
        },
    }


def install(harness, api, monkeypatch, tmp_path, *, alive: bool, pod=None):
    """Scripted kubectl reads for the Pod report and a fake kubectl process."""

    started: list[list[str]] = []
    forward = Forward(alive=alive)
    log = tmp_path / "cap001-port-forward.log"
    log.write_text("Forwarding from 127.0.0.1:12345 -> 18080\nlost connection to pod\n")
    probe = base.Probe(
        case="CAP001",
        deployment="gpu-fault-cap-cap001",
        service="gpu-fault-cap-cap001",
        database="gpu_fault_cap_cap001",
        pod="probe-pod",
        local_port=12345,
        url="http://127.0.0.1:12345",
        port_forward=forward,  # type: ignore[arg-type]
        pod_uid="probe-uid",
        restart_count=0,
        forward_log=log,
    )
    harness.active_probe = probe

    def kubectl(*args: str, check: bool = True, **_options: Any):
        if args[:2] == ("get", "pod"):
            output = json.dumps(pod) if pod is not None else ""
        elif args[:2] == ("get", "events"):
            output = json.dumps(
                {
                    "items": [
                        {
                            "lastTimestamp": "2026-01-01T00:00:02Z",
                            "type": "Warning",
                            "reason": "BackOff",
                            "message": "Back-off restarting failed container",
                        },
                        {
                            "lastTimestamp": "2026-01-01T00:00:01Z",
                            "type": "Normal",
                            "reason": "Pulled",
                            "message": "image already present",
                        },
                    ]
                }
            )
        elif args[:1] == ("logs",):
            output = "line 1\nline 2\n" + ("crash\n" if "--previous" in args else "")
        else:
            raise AssertionError(f"unexpected kubectl call {args}")
        return subprocess.CompletedProcess(args, 0, output, "")

    def popen(arguments: list[str], **options: Any) -> Forward:
        started.append(list(arguments))
        assert options["stdout"].name == str(log) and options["stdout"].mode == "a", (
            "kubectl output must be appended to the probe's forward log, not a pipe"
        )
        return Forward(alive=True)

    monkeypatch.setattr(harness, "kubectl", kubectl)
    monkeypatch.setattr(base.subprocess, "Popen", popen)
    monkeypatch.setattr(
        base.httpx,
        "get",
        lambda url, **_kw: SimpleNamespace(
            status_code=200, json=lambda: {"status": "ok"}
        ),
    )
    return probe, forward, started


def test_live_forward_is_left_alone(capacity, monkeypatch, tmp_path) -> None:
    harness, api = capacity
    probe, forward, started = install(
        harness, api, monkeypatch, tmp_path, alive=True, pod=pod_document()
    )
    assert harness.ensure_probe_transport(probe, reason="unit") is None, (
        "a live forward means the caller's error was not the tunnel"
    )
    assert started == [] and harness.transport_incidents == [], (
        "nothing is restarted or recorded while the forward is alive"
    )


def test_dead_forward_is_reestablished_with_diagnostics_before_cleanup(
    capacity, monkeypatch, tmp_path
) -> None:
    harness, api = capacity
    probe, forward, started = install(
        harness, api, monkeypatch, tmp_path, alive=False, pod=pod_document()
    )
    incident = harness.ensure_probe_transport(probe, reason="metrics: refused")
    assert incident is not None and incident["recovered"] is True, (
        "a dead forward in front of the same Pod is re-established"
    )
    assert len(started) == 1 and started[0][-2:] == [
        "service/gpu-fault-cap-cap001",
        "12345:18080",
    ], "the forward is restarted on the same local port so probe.url stays valid"
    assert probe.port_forward is not forward and probe.port_forward.poll() is None, (
        "the probe now owns the new forward"
    )
    evidence = json.loads(
        (harness.run_dir / "cap001-transport-incidents.json").read_text()
    )
    assert [item["reason"] for item in evidence] == ["metrics: refused"], (
        "every incident is written to the case directory"
    )
    record = evidence[0]
    assert record["port_forward"]["returncode"] == 1 and record["port_forward"][
        "log_tail"
    ] == ["Forwarding from 127.0.0.1:12345 -> 18080", "lost connection to pod"], (
        "kubectl's exit code and log tail explain why the tunnel died"
    )
    assert record["pod"]["restart_count"] == 0 and record["logs"] == [
        "line 1",
        "line 2",
    ], "the Pod's restart count and log tail are captured"
    assert [event["reason"] for event in record["events"]] == ["Pulled", "BackOff"], (
        "events are kept in time order"
    )
    assert record["previous_logs"] is None, "no previous container to read"


def test_restarted_pod_is_a_distinct_failure(capacity, monkeypatch, tmp_path) -> None:
    harness, api = capacity
    probe, _forward, started = install(
        harness, api, monkeypatch, tmp_path, alive=False, pod=pod_document(restarts=1)
    )
    with pytest.raises(transport.ProbeTransportError, match="restarted or vanished"):
        harness.ensure_probe_transport(probe, reason="storm send: refused")
    assert started == [], "a restarted Pod is not papered over with a new tunnel"
    record = json.loads(
        (harness.run_dir / "cap001-transport-incidents.json").read_text()
    )[0]
    assert record["pod"]["restart_count"] == 1 and record["previous_logs"] == [
        "line 1",
        "line 2",
        "crash",
    ], "the crashed container's logs are captured before cleanup"
    assert record["pod"]["containers"][0]["last_state"] == {
        "terminated": {"reason": "Error"}
    }, "the last container state names the crash"


def test_vanished_pod_is_a_distinct_failure(capacity, monkeypatch, tmp_path) -> None:
    harness, api = capacity
    probe, _forward, _started = install(
        harness, api, monkeypatch, tmp_path, alive=False, pod=None
    )
    with pytest.raises(transport.ProbeTransportError, match="vanished"):
        harness.ensure_probe_transport(probe, reason="unit")


def test_metrics_recover_through_the_supervised_forward(
    capacity, monkeypatch, tmp_path
) -> None:
    harness, api = capacity
    probe, _forward, started = install(
        harness, api, monkeypatch, tmp_path, alive=False, pod=pod_document()
    )
    attempts: list[str] = []

    def get(url: str, **_kw: Any):
        attempts.append(url)
        if url.startswith("http://127.0.0.1:1/") or (
            url.endswith("/metrics") and len(attempts) == 1
        ):
            raise httpx.ConnectError("refused", request=httpx.Request("GET", url))
        if url.endswith("/healthz"):
            return SimpleNamespace(status_code=200, json=lambda: {"status": "ok"})
        return httpx.Response(200, text="queue 4\n", request=httpx.Request("GET", url))

    monkeypatch.setattr(base.httpx, "get", get)
    assert harness.metrics(probe.url) == [("queue", {}, 4.0)], (
        "one recovered read replaces the refused one"
    )
    assert len(started) == 1, "the dead forward was restarted exactly once"
    with pytest.raises(httpx.ConnectError):
        harness.metrics("http://127.0.0.1:1/")
    assert len(started) == 1, "a URL that is not the probe's is not recovered"


def test_storm_sender_recovers_the_forward_before_retrying(monkeypatch) -> None:
    ensured: list[str] = []
    requests: list[httpx.Request] = []

    def send(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(202)

    probe = SimpleNamespace(url="http://probe.invalid")
    harness = SimpleNamespace(
        tokens=["a", "b"],
        cluster_headers=base.CapHarnessBase.cluster_headers,
        active_probe=probe,
        ensure_probe_transport=lambda current, *, reason: ensured.append(reason),
    )
    monkeypatch.setattr(traffic.time, "sleep", lambda _seconds: None)
    with httpx.Client(
        base_url="http://probe.invalid", transport=httpx.MockTransport(send)
    ) as client:
        result = traffic.cap001_send(harness, client, "storm", 0, 3, 0)
    assert result["status"] == 202 and result["transport_retries"] == 1, (
        "the retry after recovery succeeds and is counted"
    )
    assert len(ensured) == 1 and ensured[0].startswith("storm send: ConnectError"), (
        "only a refused connection asks for the tunnel to be checked"
    )
