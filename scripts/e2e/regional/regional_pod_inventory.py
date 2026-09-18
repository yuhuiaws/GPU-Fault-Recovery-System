"""Validate complete Pod readiness without treating missing data as health."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError


def ready_pod_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise RegionalFixtureError("Pod inventory is missing or malformed")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for item in value["items"]:
        if not isinstance(item, dict):
            raise RegionalFixtureError("Pod inventory entry is malformed")
        metadata = item.get("metadata")
        spec = item.get("spec")
        status = item.get("status")
        if not all(isinstance(part, dict) for part in (metadata, spec, status)):
            raise RegionalFixtureError("Pod identity or state is missing")
        name, uid = metadata.get("name"), metadata.get("uid")
        if not isinstance(name, str) or not name or not isinstance(uid, str) or not uid:
            raise RegionalFixtureError("Pod identity is missing")
        if (name, uid) in identities:
            raise RegionalFixtureError("Pod inventory repeats an identity")
        identities.add((name, uid))
        if metadata.get("deletionTimestamp") or status.get("phase") != "Running":
            continue
        conditions = status.get("conditions")
        containers = spec.get("containers")
        statuses = status.get("containerStatuses")
        if not all(
            isinstance(part, list) for part in (conditions, containers, statuses)
        ):
            continue
        ready = [
            condition.get("status")
            for condition in conditions
            if isinstance(condition, dict) and condition.get("type") == "Ready"
        ]
        if ready != ["True"]:
            continue
        names = [
            container.get("name")
            for container in containers
            if isinstance(container, dict)
        ]
        reported = [
            container.get("name")
            for container in statuses
            if isinstance(container, dict) and container.get("ready") is True
        ]
        if (
            not names
            or len(names) != len(containers)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or len(reported) != len(statuses)
            or len(reported) != len(names)
            or set(reported) != set(names)
        ):
            continue
        result.append({"name": name, "uid": uid, "node": spec.get("nodeName")})
    return sorted(result, key=lambda item: str(item["name"]))
