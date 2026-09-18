from __future__ import annotations

import json
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.response import addinfourl

import httpx

from scripts.e2e.regional.capacity_acceptance_cases import CapacityAcceptanceCases


class InlinePool:
    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False

    def __enter__(self) -> InlinePool:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.closed = True

    def map(self, operation: Any, values: Any) -> Any:
        return map(operation, values)

    def submit(self, operation: Any, *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(operation(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.sleep_scale = 1.0

    def perf_counter(self) -> float:
        self.elapsed += 0.001
        return self.elapsed

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return 2_000_000_000 + self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds * self.sleep_scale


class Client:
    def __init__(self, harness: CapacityModel, **kwargs: Any) -> None:
        self.harness = harness
        self.options = kwargs
        self.closed = False

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        self.harness.requests.append((path, kwargs))
        if self.harness.failure == "request":
            raise httpx.ReadTimeout("synthetic claim read failure")
        if self.harness.case == "CAP002":
            status = 503 if "behavior" in self.harness.holds else 200
            if self.harness.failure == "behavior":
                status = 403
            return httpx.Response(status, headers={"Retry-After": "2"})
        return httpx.Response(
            503 if self.harness.failure == "capacity" else 200,
            json={"commands": []} if self.harness.failure != "claim-shape" else {},
        )


class CapacityModel(CapacityAcceptanceCases):
    """Real case methods with recording HTTP, probe and metric boundaries."""

    def __init__(self, root: Path, clock: Clock, *, failure: str = "") -> None:
        self.run_dir = root
        self.run_id = "caplocaltest"
        self.tokens = [f"synthetic-{index:03d}" for index in range(20)]
        self.b_latency_factor = 2.0
        self.clock = clock
        self.failure = failure
        self.events: list[Any] = []
        self.requests: list[Any] = []
        self.clients: list[Client] = []
        self.wire_requests: list[dict[str, Any]] = []
        self.wire_responses: list[addinfourl] = []
        self.wire_commands: list[dict[str, Any]] = []
        self.holds: set[str] = set()
        self.releases: dict[str, int] = {}
        self.behavior_release_phases: list[str] = []
        self.retry_release_pending = False
        self.retry_cleanup_pending = False
        self.case = ""
        self.cleanup_started = False
        self.cpu_reads = 0
        self.queue_samples = [1.0, 0.0, 1.0, 0.0, 0.0]
        self.claim_request_budget = 15.0
        self.claim_wait_maximum = 30.0
        self.claim_elapsed_override: float | None = None

    def client(self, **kwargs: Any) -> Client:
        client = Client(self, **kwargs)
        self.clients.append(client)
        return client

    def urlopen(
        self, request: Request, *, timeout: float, ssl_context: Any
    ) -> addinfourl:
        target = urlsplit(request.full_url)
        assert target.scheme == "http" and target.netloc == "127.0.0.1:18623", (
            "the capacity client must target only the isolated fixture"
        )
        assert request.get_method() == "POST", "claim must use the production POST"
        assert target.path == "/v1/regional/executors/claim", (
            "the fixture only serves the production Executor claim endpoint"
        )
        assert ssl_context is None, "the loopback HTTP fixture does not use TLS"
        payload = json.loads(request.data or b"{}")
        headers = {key.lower(): value for key, value in request.header_items()}
        cluster_id = headers["x-gpu-fault-cluster-id"]
        index = int(cluster_id.removeprefix("cap-cluster-"))
        assert headers["authorization"] == f"Bearer {self.tokens[index]}", (
            "the serialized request must retain its cluster credential binding"
        )
        self.wire_requests.append(
            {
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
                "path": target.path,
            }
        )
        if self.case == "CAP002":
            self.retry_cleanup_pending = True
            status = (
                503
                if "behavior" in self.holds or self.failure == "retry-exhaustion"
                else 200
            )
            if self.failure == "behavior":
                status = 403
            if len(self.wire_requests) == 1 and status == 503:
                self.retry_release_pending = True
        else:
            assert self.case == "CAP003", "no other case may use the claim fixture"
            effective = self.wait_configuration(payload["wait_seconds"])[
                "effective_wait_seconds"
            ]
            self.clock.sleep(
                effective
                if self.claim_elapsed_override is None
                else self.claim_elapsed_override
            )
            status = 200
        if status != 200:
            raise HTTPError(
                request.full_url,
                status,
                "synthetic claim response",
                {"Retry-After": "2"},
                BytesIO(b"{}"),
            )
        response = addinfourl(
            BytesIO(json.dumps({"commands": self.wire_commands}).encode()),
            {},
            request.full_url,
            status,
        )
        self.wire_responses.append(response)
        return response

    def deploy_probe(self, name: str, environment: dict[str, str]) -> Any:
        self.case = name
        self.events.append(("deploy", name, environment))
        return SimpleNamespace(
            url="http://127.0.0.1:18623",
            pod="cap-probe",
            database=f"gpu_fault_{self.run_id}_{name.lower()}",
        )

    def cleanup_probe(self, probe: Any) -> dict[str, Any]:
        self.events.append(("cleanup", probe.pod))
        if self.failure == "cleanup":
            raise OSError("synthetic probe cleanup failure")
        return {"database_dropped": True, "residual_probe_pods": []}

    def probe_control(
        self, _probe: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        tag = payload["tag"]
        self.events.append((path, tag))
        if path == "/__cap__/claim-wakeups":
            return {
                "listener_connected": True,
                "listener_alive": True,
                "disconnects_total": 0,
                **(
                    self.wait_configuration(payload["wait_seconds"])
                    if "wait_seconds" in payload
                    else {}
                ),
            }
        if path.endswith("hold"):
            self.holds.add(tag)
        else:
            assert path == "/__cap__/release", "unknown probe control request"
            self.releases[tag] = self.releases.get(tag, 0) + 1
            if tag == "behavior":
                # The retry callback and its finally release precede case cleanup.
                if self.retry_release_pending:
                    self.retry_release_pending = False
                    phase = "retry"
                elif self.retry_cleanup_pending:
                    self.retry_cleanup_pending = False
                    phase = "proof-finally"
                else:
                    self.cleanup_started = True
                    phase = "cleanup"
                self.behavior_release_phases.append(phase)
            self.holds.discard(tag)
            if self.failure == "release" and self.cleanup_started:
                raise OSError("synthetic hold release failure")
        return {"tag": tag}

    def wait_configuration(self, requested: float) -> dict[str, float]:
        budget = self.claim_request_budget
        return {
            "requested_wait_seconds": requested,
            "request_budget_seconds": budget,
            "claim_wait_reserve_seconds": 3.0,
            "server_max_wait_seconds": self.claim_wait_maximum,
            "effective_wait_seconds": min(
                requested,
                self.claim_wait_maximum,
                max(0.0, budget - 3.0) if budget > 0 else requested,
            ),
        }

    def metrics(self, _url: str) -> list[Any]:
        queue = self.queue_samples.pop(0) if self.queue_samples else 0.0
        if self.failure == "drain":
            queue = 1.0
        return [
            ("gpu_fault_processor_queue_depth", {}, queue),
            ("gpu_fault_store_io_in_flight", {}, 4.0),
            (
                "gpu_fault_store_io_max_in_flight",
                {},
                0.0 if self.failure == "saturation" else 4.0,
            ),
            ("gpu_fault_store_io_rejections_total", {"reason": "capacity"}, 3.0),
            ("gpu_fault_store_io_admission_wait_seconds_max", {}, 0.0),
        ]

    def amp_request(self, method: str, path: str, params: Any) -> dict[str, Any]:
        self.events.append(("amp", method, path, params))
        return {
            "status": "success",
            "data": {
                "result": []
                if self.failure == "alert"
                else [{"value": [self.clock.time(), "1"]}]
            },
        }

    def alert_states(self, _name: str) -> list[str]:
        if self.failure == "preactive":
            return ["firing"]
        if self.cleanup_started and self.failure == "resolution-read":
            raise OSError("synthetic alert read failure")
        if self.cleanup_started and self.failure == "unresolved":
            return ["firing"]
        return ["firing"] if "alert" in self.holds else []

    def pod_cpu_usage_usec(self, _pod: str) -> int:
        self.cpu_reads += 1
        return self.cpu_reads * (-1 if self.failure == "cpu" else 1_000_000)

    def isolated_db_connections(self, _pod: str) -> int:
        return 0 if self.failure == "connections" else 2

    def connection_budget(self) -> dict[str, Any]:
        self.events.append(("connection-budget",))
        return {"budget_ratio": 0.5}

    def cloudwatch_window(self, started: Any, ended: Any) -> dict[str, Any]:
        self.events.append(("cloudwatch", started, ended))
        return {"database_connections": 2}
