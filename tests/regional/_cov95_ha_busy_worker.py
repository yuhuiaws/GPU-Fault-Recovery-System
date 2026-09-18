"""Bounded worker-lease simulation, reusable with a parent-supplied PostgreSQL Store."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from typing import Any

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorRequest,
    ProcessorSpoolSettings,
    coordinator,
)


class StoreTransport:
    def __init__(self, store: Any, *, failed: Event, original: bool) -> None:
        self.store, self.failed, self.original = store, failed, original
        self.late_fenced = Event()

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.store, name)
        if name in {
            "renew_active_processor_request",
            "release_active_processor_request",
        }:

            def unavailable(*args: Any, **kwargs: Any) -> Any:
                if self.original and self.failed.is_set():
                    raise OSError("unit original CPU Store transport lost")
                return method(*args, **kwargs)

            return unavailable
        if name == "complete_active_processor_request":

            def complete(*args: Any, **kwargs: Any) -> Any:
                try:
                    return method(*args, **kwargs)
                except ValueError:
                    if self.original and self.failed.is_set():
                        self.late_fenced.set()
                    raise

            return complete
        if name == "complete_telemetry_spool":

            def complete_spool(*args: Any, **kwargs: Any) -> Any:
                completed = method(*args, **kwargs)
                if self.original and self.failed.is_set() and completed == 0:
                    self.late_fenced.set()
                return completed

            return complete_spool
        return method


class Response:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


def exercise_busy_worker_takeover(
    store: Any, monkeypatch: pytest.MonkeyPatch, *, role: str
) -> dict[str, Any]:
    """Keep work in flight, lose its owner, reclaim, then fence the late completion."""
    entered, release, replacement_entered, failed = Event(), Event(), Event(), Event()
    lock = Lock()
    attempts: list[str] = []
    transports = [
        StoreTransport(store, failed=failed, original=True),
        StoreTransport(store, failed=failed, original=False),
    ]
    processors = [
        ProcessorCoordinator(
            transport,
            owner_id=owner,
            internal_token="REPLACE_WITH_UNIT_REPLAY_" + "r" * 32,
            execution_token="REPLACE_WITH_UNIT_EXECUTION_" + "x" * 32,
            local_url="http://127.0.0.1:1",
            active_consumers=True,
            lease=ProcessorLeaseSettings(
                poll_seconds=0.01, request_lease_seconds=1, request_renew_seconds=0.05
            ),
            spool=ProcessorSpoolSettings(
                telemetry_spool_enabled=role == "spool",
                telemetry_spool_workers=1,
                telemetry_spool_lease_seconds=1,
                telemetry_spool_notification_fallback_seconds=2,
            ),
        )
        for transport, owner in zip(
            transports, ["unit-first", "unit-second"], strict=True
        )
    ]
    path = (
        "/v1/collector-events/host-telemetry"
        if role == "spool"
        else "/v1/terminal-events"
    )
    payload = {
        "batch_id": "unit-batch",
        "cluster_id": "unit-cluster",
        "node_id": "unit-node",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "edge_filter_reasons": ["health-summary"],
        "samples": [],
        "collection_errors": [],
    }
    request = ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=json.dumps(payload).encode(),
        content_type="application/json",
        cluster_id="unit-cluster",
    )
    if role == "spool":
        store.try_spool_telemetry_requests(
            [request], max_depth=10, max_cluster_depth=10
        )
    else:
        store.enqueue_processor_request(request)

    def work(batch: dict[str, Any] | None = None) -> dict[str, Any]:
        with lock:
            original = not attempts
            owner = "first" if original else "second"
            attempts.append(owner)
        if original:
            entered.set()
            if not release.wait(timeout=4):
                raise RuntimeError("bounded unit worker barrier expired")
        else:
            replacement_entered.set()
        if batch is not None:
            return {
                "results": [
                    {
                        "request_id": item["request_id"],
                        "status": 200,
                        "body": {"owner": owner},
                    }
                    for item in batch["items"]
                ]
            }
        return {"owner": owner}

    def urlopen(message: Any, **kwargs: Any) -> Response:
        document = json.loads(message.data)
        return Response(work(document if "items" in document else None))

    for processor in processors:
        processor.telemetry_spool_replay_handler = work
    target = "run_telemetry_spool" if role == "spool" else "run_processor"
    threads = [
        Thread(
            target=getattr(processor, target), name=f"unit-{role}-{index}", daemon=True
        )
        for index, processor in enumerate(processors)
    ]
    with monkeypatch.context() as local:
        local.setattr(coordinator, "urlopen", urlopen)
        threads[0].start()
        try:
            assert entered.wait(timeout=2), (
                "first CPU worker must hold real in-flight work"
            )
            if role == "spool":
                early = store.claim_telemetry_spool(
                    "unit-early-contender",
                    now=datetime.now(timezone.utc),
                    lease_duration=timedelta(seconds=1),
                    limit=1,
                    path=path,
                )
                assert processors[0].spool_consumer_running is True
            else:
                early = store.claim_active_processor_requests(
                    "unit-early-contender",
                    now=datetime.now(timezone.utc),
                    lease_duration=timedelta(seconds=1),
                    limit=1,
                )
                assert processors[0].processor_consumer_running is True
            assert early == [], (
                "busy work must remain exclusive while its lease is live"
            )
            failed.set()
            if role == "spool":
                processors[0].stop()
            else:
                abandoned = processors[0].abandon_in_flight()
                assert abandoned == {"in_flight": 1, "released": 0, "failed": 1}
            threads[1].start()
            assert replacement_entered.wait(timeout=3), (
                "replacement must reclaim after lease expiry"
            )
            deadline = time.monotonic() + 2
            completed = False
            while time.monotonic() < deadline:
                if role == "spool":
                    completed = store.telemetry_spool_stats()["depth"] == 0
                else:
                    completed = (
                        store.get_processor_request(request.request_id).status.value
                        == "COMPLETED"
                    )
                if completed:
                    break
                time.sleep(0.01)
            assert completed, (
                "replacement must commit its work within the bounded window"
            )
            release.set()
            assert transports[0].late_fenced.wait(timeout=2), (
                "the original late result must be fenced"
            )
        finally:
            release.set()
            for processor in processors:
                processor.stop()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(timeout=3)
            assert not any(thread.is_alive() for thread in threads), (
                "all unit CPU workers must stop"
            )
    assert attempts == ["first", "second"], (
        "only the original and one replacement may execute"
    )
    if role == "processor":
        final = store.get_processor_request(request.request_id)
        assert json.loads(base64.b64decode(final.response_body_base64)) == {
            "owner": "second"
        }
        depth = store.processor_queue_stats()["depth"]
    else:
        depth = store.telemetry_spool_stats()["depth"]
    assert depth == 0, "the owned queue must drain after takeover"
    return {
        "validation_scope": "local-worker-lease-simulation",
        "role": role,
        "handler_attempts": len(attempts),
        "late_result_fenced": True,
        "queue_depth": depth,
        "cpu_saturation_tested": False,
    }
