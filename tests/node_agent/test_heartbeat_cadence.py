"""The node Agent heartbeat on the pooled transport with a jittered interval.

The heartbeat is the fleet's most frequent post (every 30 s from every node);
it used to open a connection per post through ``urllib`` and beat at exactly
the configured interval. Now it reuses the process's pooled connection,
carries a request id so a connection the control plane closed meanwhile costs
one transparent retry instead of a missed beat, and waits ``+-10 %`` between
beats - the first beat, the registration, is never delayed.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from threading import Event
from typing import Any
from unittest import mock

from gpu_fault.collectors.cadence import CadenceJitter
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import heartbeat as heartbeat_module
from gpu_fault.node_agent.heartbeat import AgentHeartbeatReporter


def _reporter(**overrides: Any) -> AgentHeartbeatReporter:
    settings: dict[str, Any] = dict(
        control_plane_url="https://control.example",
        secret="s" * 32,
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="https://node-a:9099",
        agent_version="1.0.0",
        artifact_sha256="a" * 64,
        policy_version="1",
        runtime_profile_version="hyperpod-v1",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        boot_id="boot-a",
        interval_seconds=30,
        sender=lambda _url, _envelope: {},
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    settings.update(overrides)
    return AgentHeartbeatReporter(**settings)


def test_the_wait_between_heartbeats_is_jittered_but_the_first_beat_is_not() -> None:
    reporter = _reporter()
    reporter.cadence_jitter = CadenceJitter(ratio=0.1, startup_seconds=20.0)
    beats: list[str] = []
    reporter.report_once = lambda: beats.append("beat")  # type: ignore[method-assign]
    waits: list[float] = []
    stop = Event()

    def wait(timeout: float | None = None) -> bool:
        waits.append(float(timeout or 0))
        if len(waits) >= 50:
            stop.set()
        return stop.is_set()

    stop.wait = wait  # type: ignore[method-assign]
    reporter.run(stop)
    assert len(beats) == 50 and beats[0] == "beat", "the registration goes out at once"
    assert all(27.0 <= wait <= 33.0 for wait in waits), waits
    assert len(set(waits)) > 1, (
        "the waits move; a fleet restarted together does not beat in step"
    )


def test_the_heartbeat_posts_through_the_pooled_client_with_a_request_id() -> None:
    seen: list[Any] = []

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

    def urlopen(request: Any, timeout: float | None = None) -> Response:
        seen.append((request, timeout))
        return Response(json.dumps({"generation": 7}).encode())

    with mock.patch.object(heartbeat_module, "urlopen", urlopen):
        result = AgentHeartbeatReporter._send(
            "https://control.example",
            mock.Mock(model_dump_json=lambda: "{}"),
            cluster_id="cluster-a",
            bearer_token="token-a",
        )
    assert result == {"generation": 7}
    request, timeout = seen[0]
    assert timeout == 10
    assert request.full_url == "https://control.example/v1/fleet/agents/heartbeat"
    assert request.get_method() == "POST"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["x-gpu-fault-cluster-id"] == "cluster-a"
    assert headers["authorization"] == "Bearer token-a"
    assert len(headers["x-gpu-fault-request-id"]) == 32, "a fresh id per heartbeat"
    second: list[Any] = []
    with mock.patch.object(
        heartbeat_module,
        "urlopen",
        lambda r, timeout=None: second.append(r) or Response(b""),
    ):
        assert (
            AgentHeartbeatReporter._send(
                "https://control.example",
                mock.Mock(model_dump_json=lambda: "{}"),
                cluster_id="cluster-a",
                bearer_token=None,
            )
            == {}
        )
    first_id = headers["x-gpu-fault-request-id"]
    second_id = dict(second[0].header_items())["X-gpu-fault-request-id"]
    assert first_id != second_id
