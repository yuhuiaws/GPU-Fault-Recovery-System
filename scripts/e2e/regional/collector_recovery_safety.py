"""Refuse cleanup while a workflow or physical action remains unsettled."""

from __future__ import annotations

from typing import Any

from gpu_fault.recovery_safety import (
    recovery_safety_errors,
    unresolved_details as unresolved_details,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def require_bound_refresh(
    previous: dict[str, Any], current: dict[str, Any], marker: str
) -> None:
    if current.get("seed_marker") != marker:
        raise RegionalFixtureError("collector recovery marker identity changed")
    for field, identity in (("incidents", "incident_id"), ("workflows", "request_id")):
        before = {
            item[identity]
            for item in previous.get(field) or []
            if isinstance(item, dict) and isinstance(item.get(identity), str)
        }
        after = {
            item[identity]
            for item in current.get(field) or []
            if isinstance(item, dict) and isinstance(item.get(identity), str)
        }
        if not before <= after:
            raise RegionalFixtureError(
                "collector recovery refresh lost a tracked identity"
            )


def require_settled_recovery(state: dict[str, Any]) -> None:
    errors = recovery_safety_errors(state.get("workflows"), state.get("commands"))
    if errors:
        raise RegionalFixtureError(errors[0])
    incidents = state.get("incidents", [])
    if not isinstance(incidents, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("incident_id"), str)
        or not item["incident_id"]
        for item in incidents
    ):
        raise RegionalFixtureError("recovery has no exact incident identity")
