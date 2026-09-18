"""Identity-bound training progress during the EFA plugin acceptance window."""

from __future__ import annotations

import re
import time
from typing import Any

from scripts.e2e.regional.collector_action_guard import (
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.managed_workload_fixture import heartbeat_healthy
from scripts.e2e.regional.regional_commands import RegionalFixtureError

STEP = re.compile(r"\bstep=(\d+)\b")


def progress_reading(
    snapshot: dict[str, Any], *, attempt_id: str, world_size: int, expected_pods: int
) -> dict[str, dict[str, Any]]:
    pods = snapshot.get("pods") or []
    names = {item.get("name") for item in pods}
    uids = {item.get("uid") for item in pods}
    if (
        len(pods) != expected_pods
        or len(names) != expected_pods
        or len(uids) != expected_pods
        or not all(names)
        or not all(uids)
    ):
        raise RegionalFixtureError("training progress has incomplete Pod identities")
    readings = {}
    for pod in pods:
        name = str(pod["name"])
        if (
            pod.get("phase") != "Running"
            or pod.get("ready") is not True
            or not pod.get("node")
            or pod.get("attempt_id") != attempt_id
        ):
            raise RegionalFixtureError("training progress lost its running attempt")
        lines = [
            line
            for line in (snapshot.get("heartbeat_logs") or {})
            .get(name, "")
            .splitlines()
            if "HEARTBEAT" in line
        ]
        latest = lines[-1] if lines else ""
        match = STEP.search(latest)
        if match is None or not heartbeat_healthy(latest, world_size=world_size):
            raise RegionalFixtureError("training progress lacks a valid NCCL heartbeat")
        readings[name] = {
            "uid": pod["uid"],
            "node": pod["node"],
            "step": int(match.group(1)),
        }
    return readings


def wait_training_progress(
    workload: Any,
    before: dict[str, Any],
    *,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    timeout = finite_seconds(timeout_seconds)
    settings = workload.settings
    options = {
        "attempt_id": settings.attempt_id,
        "world_size": settings.expected_gpu_count,
        "expected_pods": settings.expected_pods,
    }
    baseline = progress_reading(before, **options)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require_action_time()
        current = workload.snapshot()
        reading = progress_reading(current, **options)
        if set(reading) != set(baseline) or any(
            reading[name][field] != baseline[name][field]
            for name in reading
            for field in ("uid", "node")
        ):
            raise RegionalFixtureError("training progress changed the source Pod set")
        if any(reading[name]["step"] < baseline[name]["step"] for name in reading):
            raise RegionalFixtureError("training heartbeat step regressed")
        if all(reading[name]["step"] > baseline[name]["step"] for name in reading):
            if time.monotonic() >= deadline:
                break
            return {
                **current,
                "progress_evidence": {"before": baseline, "after": reading},
            }
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise RegionalFixtureError("managed training did not advance on every source Pod")


def incident_attempt_errors(
    state: dict[str, Any],
    *,
    cluster_id: str,
    node: str,
    job_id: str,
    attempt_id: str,
    workflow_id: str,
) -> list[str]:
    incident = state.get("incident") or {}
    workflow = state.get("workflow") or {}
    if (
        not workflow_id
        or workflow.get("request_id") != workflow_id
        or workflow.get("incident_id") != incident.get("incident_id")
        or incident.get("cluster_id") != cluster_id
        or incident.get("job_id") != job_id
        or incident.get("attempt_id") != attempt_id
        or node not in (incident.get("node_ids") or [])
    ):
        return ["EFA plugin incident is not bound to the managed training attempt"]
    return []
