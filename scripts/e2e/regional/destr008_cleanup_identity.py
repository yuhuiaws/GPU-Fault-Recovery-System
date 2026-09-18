"""Prove that restoration targets this drill's root or actual escalation chain."""

from __future__ import annotations

from typing import Any

from gpu_fault.orchestration.escalation import escalation_origin
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture


def require_cleanup_node(
    warm: WarmSpareLiveFixture, *, node: str, uid: str, owner: str
) -> None:
    current = warm.node_snapshot(node)
    if (
        not isinstance(current, dict)
        or current.get("uid") != uid
        or not isinstance(current.get("annotations"), dict)
        or str(current["annotations"].get("gpu-fault.io/incident-id") or "") != owner
    ):
        raise RegionalFixtureError("cleanup quarantine owner or Node UID changed")


def require_cleanup_family(
    warm: WarmSpareLiveFixture,
    *,
    incident_id: str,
    owner: str,
    cluster_id: str,
    node: str,
    profile_version: str,
) -> None:
    root = warm.incident_by_id(incident_id)
    if (
        not isinstance(root, dict)
        or root.get("incident_id") != incident_id
        or root.get("cluster_id") != cluster_id
        or root.get("node_ids") != [node]
        or not isinstance(root.get("job_id"), str)
        or not root["job_id"]
        or not isinstance(root.get("attempt_id"), str)
        or not root["attempt_id"]
    ):
        raise RegionalFixtureError("cleanup root incident identity is unproven")
    current_id = owner or incident_id
    seen: set[str] = set()
    for _ in range(8):
        if current_id == incident_id:
            return
        if current_id in seen:
            raise RegionalFixtureError("cleanup escalation ancestry is cyclic")
        seen.add(current_id)
        current: dict[str, Any] = warm.incident_by_id(current_id)
        if (
            not isinstance(current, dict)
            or current.get("incident_id") != current_id
            or any(
                current.get(field) != root.get(field)
                for field in ("cluster_id", "job_id", "attempt_id", "node_ids")
            )
            or not isinstance(current.get("event_id"), str)
            or current_id != "inc-" + current["event_id"]
        ):
            raise RegionalFixtureError(
                "cleanup quarantine owner belongs to another run"
            )
        origin = escalation_origin(current["event_id"])
        if origin is None:
            raise RegionalFixtureError("cleanup quarantine owner is not an escalation")
        workflow = warm.wait_workflow_id(origin[1], timeout_seconds=1)
        parent_id = workflow.get("incident_id") if isinstance(workflow, dict) else None
        if (
            not isinstance(workflow, dict)
            or workflow.get("request_id") != origin[1]
            or workflow.get("runtime_profile_version") != profile_version
            or workflow.get("status") != "FAILED"
            or not isinstance(parent_id, str)
            or not parent_id
        ):
            raise RegionalFixtureError("cleanup escalation source is unproven")
        current_id = parent_id
    raise RegionalFixtureError("cleanup escalation ancestry exceeds its bound")
