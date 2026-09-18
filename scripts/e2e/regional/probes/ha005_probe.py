#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlsplit

import gpu_fault.collectors.sinks as sinks
import gpu_fault.cluster_executor.regional_client as regional_client

from gpu_fault.cluster_executor import (
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.collectors import CollectorError, HttpEventSink
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus


STATE = Path("/state")
READY = STATE / "ready.json"
STATS = STATE / "stats.json"
STOP = STATE / "stop"
OUTBOX = STATE / "outbox.ndjson"
HOST_PATH = "/v1/collector-events/host-telemetry"


class ProbeEvidence:
    def __init__(self) -> None:
        self.lock = Lock()
        self.admissions: list[dict] = []
        self.responses: list[dict] = []

    @contextmanager
    def observe_http(self):
        collector_open, executor_open = sinks.urlopen, regional_client.urlopen

        def observe(open_request):
            def wrapped(request, *args, **kwargs):
                entry = {
                    "path": urlsplit(request.full_url).path,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    response = open_request(request, *args, **kwargs)
                except HTTPError as exc:
                    entry.update(
                        status=exc.code, retry_after=exc.headers.get("Retry-After")
                    )
                    raise
                except Exception as exc:
                    entry.update(status=None, transport_error=type(exc).__name__)
                    raise
                else:
                    entry.update(
                        status=response.status,
                        retry_after=response.headers.get("Retry-After"),
                    )
                    return response
                finally:
                    with self.lock:
                        self.responses.append(entry)

            return wrapped

        sinks.urlopen, regional_client.urlopen = (
            observe(collector_open),
            observe(executor_open),
        )
        try:
            yield
        finally:
            sinks.urlopen, regional_client.urlopen = collector_open, executor_open

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "admissions": [dict(item) for item in self.admissions],
                "wire_responses": [dict(item) for item in self.responses],
                "accepted_request_ids": sorted(
                    {
                        item["request_id"]
                        for item in self.admissions
                        if isinstance(item.get("request_id"), str)
                        and item["request_id"]
                    }
                ),
            }


class ObservedSink(HttpEventSink):
    def __init__(self, *args, evidence: ProbeEvidence, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evidence = evidence

    def _post_with_retry(self, path, payload, *, buffer_failure, **kwargs):
        result = super()._post_with_retry(
            path, payload, buffer_failure=buffer_failure, **kwargs
        )
        with self.evidence.lock:
            self.evidence.admissions.append(
                {
                    "batch_id": payload.get("batch_id"),
                    "replay": not buffer_failure,
                    "accepted": result.get("accepted"),
                    "spooled": result.get("spooled", False),
                    "coalesced": result.get("coalesced"),
                    "request_id": result.get("processor_request_id"),
                }
            )
        return result


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def outbox_status() -> dict:
    if not OUTBOX.is_file():
        return {"records": 0, "replayable": 0}
    records = []
    for line in OUTBOX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return {
        "records": len(records),
        "replayable": sum(bool(item.get("replayable")) for item in records),
    }


def main() -> None:
    evidence = ProbeEvidence()
    with evidence.observe_http():
        run_probe(evidence)


def run_probe(evidence: ProbeEvidence) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    registrations = json.loads(Path("/tokens/clusters.json").read_text())
    registration = registrations[0]
    run_id = os.environ["RUN_ID"]
    client = RegionalExecutorClient(
        os.environ["CONTROL_PLANE_URL"],
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=15,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    sink = ObservedSink(
        os.environ["CONTROL_PLANE_URL"],
        bearer_token=registration["token"],
        timeout_seconds=5,
        max_attempts=1,
        outbox_path=str(OUTBOX),
        outbox_max_records=1000,
        outbox_replay_batch_size=10,
        outbox_replay_budget_seconds=5,
        evidence=evidence,
    )
    atomic_json(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "run_id": run_id,
            "node_id": f"ha005-node-{run_id}",
        },
    )
    counters: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    attempted_batch_ids: list[str] = []
    attempted_events: list[dict] = []
    recent_errors: list[dict] = []
    sequence = 0
    next_claim = 0.0
    next_event = 0.0
    started = time.monotonic()
    while not STOP.exists():
        elapsed = time.monotonic() - started
        if elapsed >= next_claim:
            counters["claim_attempts"] += 1
            try:
                commands = client.claim(
                    f"ha005-probe/{run_id}",
                    execution_owners=["gpu-fault-ha005-noop"],
                    max_commands=1,
                    lease_seconds=60,
                )
                counters["claim_success"] += 1
                counters["commands_claimed"] += len(commands)
                for command in commands:
                    delay = float(os.environ.get("COMMAND_DELAY_SECONDS", "0"))
                    if delay > 0:
                        time.sleep(delay)
                    client.complete(
                        command,
                        RemoteCommandResult(
                            lease_token=str(command.lease_token),
                            status=RemoteCommandStatus.SUCCEEDED,
                            details={"simulated": True, "run_id": run_id},
                        ),
                    )
                    counters["commands_completed"] += 1
            except Exception as exc:
                counters["claim_failures"] += 1
                key = (
                    f"http-{exc.status_code}"
                    if isinstance(exc, ClusterExecutorError)
                    and exc.status_code is not None
                    else type(exc).__name__
                )
                error_types[key] += 1
                recent_errors.append(
                    {
                        "phase": "claim",
                        "type": key,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            next_claim = elapsed + 2
        if elapsed >= next_event:
            batch_id = f"ha005-{run_id}-{sequence:05d}"
            attempted_batch_ids.append(batch_id)
            counters["event_attempts"] += 1
            payload = {
                "batch_id": batch_id,
                "cluster_id": registration["cluster_id"],
                "node_id": f"ha005-node-{run_id}",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "samples": [
                    {"name": "cpu_usage_percent", "value": 1.0, "unit": "percent"},
                    {"name": "load1_per_cpu", "value": 0.01},
                    {
                        "name": "memory_used_percent",
                        "value": 1.0,
                        "unit": "percent",
                    },
                    {
                        "name": "filesystem_used_percent",
                        "value": 1.0,
                        "unit": "percent",
                        "device": "/",
                    },
                    {"name": "network_link_up", "value": 1.0},
                    {"name": "network_errors_delta", "value": 0.0},
                    {"name": "network_drops_delta", "value": 0.0},
                    {"name": "rdma_link_down", "value": 0.0},
                    {"name": "rdma_errors_delta", "value": 0.0},
                ],
                "collection_errors": [],
                "runtime_profile_version": "hyperpod-v1",
                "workload_state": "IDLE",
                "affected_workload_ids": [],
                "edge_filter_reasons": ["health-summary"],
                "context_history": [],
            }
            attempted_events.append(
                {"batch_id": batch_id, "observed_at": payload["observed_at"]}
            )
            try:
                sink.post(HOST_PATH, payload)
                counters["event_accepted"] += 1
                statuses["202"] += 1
            except CollectorError as exc:
                counters["event_failures"] += 1
                key = (
                    f"http-{exc.status_code}"
                    if exc.status_code is not None
                    else type(exc).__name__
                )
                error_types[key] += 1
                statuses[str(exc.status_code or 0)] += 1
                recent_errors.append(
                    {
                        "phase": "event",
                        "type": key,
                        "buffered": exc.buffered,
                        "replayable": exc.replayable,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if exc.buffered:
                    counters["event_buffered"] += 1
            sequence += 1
            next_event = elapsed + 5
        atomic_json(
            STATS,
            {
                "run_id": run_id,
                "cluster_id": registration["cluster_id"],
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "counters": dict(counters),
                "statuses": dict(statuses),
                "error_types": dict(error_types),
                **evidence.snapshot(),
                "attempted_batch_ids": attempted_batch_ids,
                "attempted_events": attempted_events,
                "payload_contract": {
                    "path": HOST_PATH,
                    "node_id": f"ha005-node-{run_id}",
                    "edge_filter_reasons": ["health-summary"],
                    "collection_errors": [],
                    "sample_count": 9,
                },
                "recent_errors": recent_errors[-30:],
                "outbox": outbox_status(),
            },
        )
        time.sleep(0.2)
    replay_stopped = sink.wait_for_outbox_replay(10)
    value = json.loads(STATS.read_text(encoding="utf-8"))
    value.update(evidence.snapshot())
    value["outbox"] = outbox_status()
    value["replay_stopped"] = replay_stopped
    value["stopped"] = True
    atomic_json(STATS, value)


if __name__ == "__main__":
    main()
