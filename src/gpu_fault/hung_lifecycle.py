"""A live stalled attempt and an attempt already owned by exit handling differ."""

from __future__ import annotations

from typing import Any


def cleanup_reason(observation: Any, nodes: list[str] | None = None) -> str | None:
    if observation.termination_initiator_incident_id:
        return "ATTEMPT_CLEANUP_IN_PROGRESS"
    if observation.workload_phase.value in {"FAILED", "SUCCEEDED", "STOPPED"}:
        return "ATTEMPT_TERMINAL"
    if any(
        item.critical
        and item.terminated
        and item.exit_code not in {None, 0}
        and not item.deletion_requested
        for item in observation.containers
    ):
        return "FAILURE_OWNED_BY_WATCHER"
    targets = set(nodes or [])
    containers = [
        item
        for item in observation.containers
        if not targets or item.node_id in targets
    ]
    if (
        containers
        and (not targets or targets <= {item.node_id for item in containers})
        and all(item.terminated or item.deletion_requested for item in containers)
    ):
        return "NO_LIVE_TRAINING_CONTAINERS"
    return None
