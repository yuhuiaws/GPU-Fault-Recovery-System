"""Identity-bound, conditional changes shared by temporary deployment windows."""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError


def deployment_snapshot(
    value: Any,
    *,
    plane: str,
    deployment: str,
    container: str,
    variables: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegionalFixtureError("deployment window inventory is malformed")
    metadata, spec, status = (
        value.get("metadata"),
        value.get("spec"),
        value.get("status", {}),
    )
    if (
        not isinstance(metadata, dict)
        or not isinstance(spec, dict)
        or not isinstance(status, dict)
    ):
        raise RegionalFixtureError("deployment window identity is missing")
    uid, version, generation = (
        metadata.get("uid"),
        metadata.get("resourceVersion"),
        metadata.get("generation"),
    )
    replicas = spec.get("replicas", 1)
    if (
        not isinstance(uid, str)
        or not uid
        or not isinstance(version, str)
        or not version
        or type(generation) is not int
        or generation < 1
        or type(replicas) is not int
        or replicas < 1
        or metadata.get("deletionTimestamp")
    ):
        raise RegionalFixtureError(
            "deployment window identity or replica target is invalid"
        )
    containers = spec.get("template", {}).get("spec", {}).get("containers")
    if not isinstance(containers, list):
        raise RegionalFixtureError("deployment window containers are missing")
    matches = [
        (index, item)
        for index, item in enumerate(containers)
        if isinstance(item, dict) and item.get("name") == container
    ]
    if len(matches) != 1:
        raise RegionalFixtureError(
            f"deployment window container {container} is missing or duplicated"
        )
    index, selected = matches[0]
    entries = selected.get("env", [])
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) for item in entries
    ):
        raise RegionalFixtureError("deployment window environment is malformed")
    observed: dict[str, dict[str, Any]] = {}
    for name in variables:
        found = [item for item in entries if item.get("name") == name]
        if len(found) > 1 or any("valueFrom" in item for item in found):
            raise RegionalFixtureError(f"{name} is duplicated or set from a reference")
        literal = found[0].get("value", "") if found else None
        if found and not isinstance(literal, str):
            raise RegionalFixtureError(f"{name} is not a literal string")
        observed[name] = {"present": bool(found), "value": literal}
    observed_generation = status.get("observedGeneration")
    ready = (
        type(observed_generation) is int
        and observed_generation >= generation
        and all(
            type(status.get(field)) is int and status[field] == replicas
            for field in (
                "replicas",
                "updatedReplicas",
                "readyReplicas",
                "availableReplicas",
            )
        )
    )
    return {
        "plane": plane,
        "deployment": deployment,
        "container": container,
        "uid": uid,
        "generation": generation,
        "resource_version": version,
        "replicas": replicas,
        "rollout_ready": ready,
        "container_index": index,
        "variables": observed,
    }


def window_scope(
    regional: Any, *, plane: str, deployment: str, container: str
) -> dict[str, Any]:
    identity = regional.evidence_identity()
    if not all(
        isinstance(identity.get(key), str) and identity[key]
        for key in ("release_id", "cluster_id")
    ):
        raise RegionalFixtureError("deployment window release identity is unavailable")
    return {
        "plane": plane,
        "deployment": deployment,
        "container": container,
        "environment": regional.settings.environment(),
        "identity": identity,
    }


def require_window_record(
    record: dict[str, Any],
    scope: dict[str, Any],
    live: dict[str, Any],
    allowed: Iterable[str],
) -> None:
    baseline = record.get("baseline")
    assignments = record.get("assignments")
    if (
        record.get("schema_version") != 2
        or record.get("scope") != scope
        or not isinstance(baseline, dict)
        or not isinstance(assignments, dict)
        or not assignments
        or not set(assignments).issubset(allowed)
        or any(not isinstance(value, str) for value in assignments.values())
    ):
        raise RegionalFixtureError("deployment window baseline is unbound or invalid")
    if not baseline.get("uid") or baseline["uid"] != live["uid"]:
        raise RegionalFixtureError("deployment window target UID changed")
    variables = baseline.get("variables")
    if not isinstance(variables, dict) or not set(assignments).issubset(variables):
        raise RegionalFixtureError("deployment window baseline variables are missing")
    for name in assignments:
        entry = variables[name]
        if (
            not isinstance(entry, dict)
            or type(entry.get("present")) is not bool
            or (entry["present"] and not isinstance(entry.get("value"), str))
            or (not entry["present"] and entry.get("value") is not None)
        ):
            raise RegionalFixtureError(
                "deployment window baseline variables are malformed"
            )


def retire_foreign_closed_record(
    path: Path, record: dict[str, Any] | None, scope: dict[str, Any]
) -> dict[str, Any] | None:
    """Archive a CLOSED window record that another site/release identity wrote.

    Records are kept per case directory, so a re-run after a deploy meets the
    previous attempt's CLOSED record under the old release identity; requiring
    that record to match the new scope refused HA-004's re-run outright
    ("deployment window baseline is unbound or invalid"). A closed record owns
    nothing on the cluster: move it aside and start from no record. Anything
    not CLOSED keeps the strict scope check -- an open window from another
    identity is exactly what must never be silently replaced.
    """

    if (
        record is None
        or record.get("state") != "CLOSED"
        or not record.get("closed_at")
        or record.get("scope") == scope
    ):
        return record
    stamp = re.sub(r"[^0-9A-Za-z]", "", str(record["closed_at"]))[:16] or "unknown"
    archive = path.with_name(f"{path.stem}.closed-{stamp}{path.suffix}")
    if archive.exists():
        archive = path.with_name(
            f"{path.stem}.closed-{stamp}-{os.getpid()}{path.suffix}"
        )
    path.replace(archive)
    return None


def managed_variables_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], names: Iterable[str]
) -> bool:
    return all(
        actual["variables"].get(name) == expected["variables"].get(name)
        for name in names
    )


def complete_population(
    snapshot: Mapping[str, Any], replicas: list[dict[str, Any]]
) -> bool:
    names = [item.get("pod") for item in replicas]
    return (
        snapshot.get("rollout_ready") is True
        and len(names) == snapshot["replicas"]
        and all(isinstance(name, str) and name for name in names)
        and len(set(names)) == len(names)
    )


def apply_window_variables(
    regional: Any,
    *,
    plane: str,
    deployment: str,
    container: str,
    expected: Mapping[str, Any],
    desired: Mapping[str, str | None],
) -> None:
    value = json.loads(
        regional.kubectl(plane, "get", "deployment", deployment, "-o", "json")
    )
    current = deployment_snapshot(
        value,
        plane=plane,
        deployment=deployment,
        container=container,
        variables=desired,
    )
    if current["uid"] != expected.get("uid"):
        raise RegionalFixtureError("deployment window target UID changed")
    for name in desired:
        if current["variables"][name] != expected["variables"].get(name):
            raise RegionalFixtureError(
                "deployment window variables changed before mutation"
            )
    index = current["container_index"]
    selected = value["spec"]["template"]["spec"]["containers"][index]
    entries = copy.deepcopy(selected.get("env", []))
    replacement = []
    seen: set[str] = set()
    for entry in entries:
        name = entry.get("name")
        if name not in desired:
            replacement.append(entry)
            continue
        seen.add(name)
        if desired[name] is not None:
            replacement.append({**entry, "value": desired[name]})
    replacement.extend(
        {"name": name, "value": literal}
        for name, literal in desired.items()
        if name not in seen and literal is not None
    )
    if entries == replacement:
        return
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": current["uid"]},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": current["resource_version"],
        },
        {
            "op": "add",
            "path": f"/spec/template/spec/containers/{index}/env",
            "value": replacement,
        },
    ]
    # The full environment stays in private stdin; it can contain unrelated
    # credential literals that must never be copied into the process argv.
    regional.kubectl(
        plane,
        "patch",
        "deployment",
        deployment,
        "--type=json",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(patch),
    )
