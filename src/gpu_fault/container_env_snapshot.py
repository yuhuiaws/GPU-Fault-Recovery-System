"""Shape and equality rules for the rollback container-environment snapshot.

An automatic rollback restores the previous release's control-plane role
Deployments with their container ``env``/``envFrom`` lists verbatim, because
the previous image fails fast on ``GPU_FAULT_*`` names it does not know and
the current template is the only one the engine can render. Three tools read
that snapshot -- the renderer substitutes it into the rendered Deployments,
``gpu_fault.config_cli validate`` checks the render against it, and the
post-deploy verifier checks the live Deployments against it -- and all three
must agree on what a well-formed snapshot is and on when two environments are
"the same". This module is that single definition. It is pure: no kubectl, no
environment reads, no knowledge of which variable names the file.

Snapshot shape (what ``cpu_role_container_env`` writes)::

    {"<deployment>": {"<container>": {"env": [...], "envFrom": [...]}}}

Equality is judged after :func:`normalised_container_env`: ``env`` entries are
sorted by ``name`` (kubelet resolves the list by name, so two lists that differ
only in order build the same process environment; a duplicate name is kept
and therefore still shows up as a difference), while ``envFrom`` keeps its
order because a later source overrides an earlier one. Entries themselves are
compared whole, so a changed ``value`` or a changed ``valueFrom`` reference is a
difference, not just a changed name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ContainerEnv = dict[str, list[dict[str, Any]]]
ContainerEnvSnapshot = dict[str, dict[str, ContainerEnv]]

# The three control-plane role Deployments the snapshot covers. The renderer
# and the verifier name the same three; a snapshot naming anything else was
# not captured from a role split this repository renders.
ROLE_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
SENSITIVE_ENV = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)")


class ContainerEnvSnapshotError(ValueError):
    """The snapshot cannot be applied or compared verbatim; fail closed."""


def validate_container_env_snapshot(snapshot: object) -> ContainerEnvSnapshot:
    """Return ``snapshot`` once it has exactly the captured shape.

    Every rule here is one the renderer relied on when it copied the lists
    into the Deployments; a snapshot that breaks one would have stopped the
    render, so a comparison tool meeting it can only be reading a file the
    renderer never saw.
    """

    if not isinstance(snapshot, dict) or not snapshot:
        raise ContainerEnvSnapshotError("expected a non-empty Deployment mapping")
    for deployment_name, containers in snapshot.items():
        if not isinstance(deployment_name, str) or not isinstance(containers, dict):
            raise ContainerEnvSnapshotError(
                "Deployment entries must map container names"
            )
        if not containers:
            raise ContainerEnvSnapshotError(f"{deployment_name} lists no containers")
        for container_name, spec in containers.items():
            _validate_container_spec(deployment_name, container_name, spec)
    return snapshot


def _validate_container_spec(
    deployment_name: str, container_name: object, spec: object
) -> None:
    if (
        not isinstance(container_name, str)
        or not isinstance(spec, dict)
        or set(spec) != {"env", "envFrom"}
        or not isinstance(spec["env"], list)
        or not isinstance(spec["envFrom"], list)
    ):
        raise ContainerEnvSnapshotError(
            f"{deployment_name}/{container_name} must carry exactly "
            "env and envFrom lists"
        )
    for item in spec["env"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ContainerEnvSnapshotError(
                f"{deployment_name}/{container_name} has an env entry without a name"
            )
        if ("value" in item) == ("valueFrom" in item):
            raise ContainerEnvSnapshotError(
                f"{deployment_name}/{container_name} env {item['name']} "
                "must define exactly one of value or valueFrom"
            )
        if "value" in item and SENSITIVE_ENV.search(item["name"]):
            raise ContainerEnvSnapshotError(
                f"{deployment_name}/{container_name} sensitive env "
                f"{item['name']} carries a literal value"
            )
    for source in spec["envFrom"]:
        if not isinstance(source, dict):
            raise ContainerEnvSnapshotError(
                f"{deployment_name}/{container_name} has a malformed envFrom entry"
            )


def load_container_env_snapshot(path: str | Path) -> ContainerEnvSnapshot:
    """Read and validate the snapshot file; any failure is a snapshot error."""

    try:
        snapshot: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContainerEnvSnapshotError(f"cannot read {path}: {exc}") from exc
    return validate_container_env_snapshot(snapshot)


def pod_container_env(deployment: dict[str, Any]) -> dict[str, ContainerEnv]:
    """Every container's ``env``/``envFrom`` from a Deployment document.

    Init containers are included, matching the capture. Absent lists read as
    empty: a container without ``env`` and one with ``env: []`` build the same
    process environment, so they must compare equal.
    """

    pod_spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
    result: dict[str, ContainerEnv] = {}
    for container in [
        *pod_spec.get("initContainers", []),
        *pod_spec.get("containers", []),
    ]:
        result[str(container["name"])] = {
            "env": list(container.get("env") or []),
            "envFrom": list(container.get("envFrom") or []),
        }
    return result


def normalised_container_env(spec: ContainerEnv) -> ContainerEnv:
    """The comparison form: ``env`` sorted by name, ``envFrom`` in order."""

    return {
        "env": sorted(
            spec.get("env") or [],
            key=lambda item: (str(item.get("name")), json.dumps(item, sort_keys=True)),
        ),
        "envFrom": list(spec.get("envFrom") or []),
    }


def container_env_differences(
    snapshot: ContainerEnvSnapshot,
    actual: dict[str, dict[str, ContainerEnv]],
    *,
    actual_label: str,
) -> list[str]:
    """Where ``actual`` departs from the snapshot, one line per departure.

    Coverage is checked in both directions at both levels: a Deployment or
    container present on one side only is a difference. Within a container the
    first differing ``env`` name (by sorted order) and the first differing
    ``envFrom`` position are reported, so the line names what to look at.
    ``actual_label`` is the word for the compared side ("rendered", "live").
    """

    problems: list[str] = []
    for name in sorted(set(snapshot) - set(actual)):
        problems.append(
            f"snapshot names Deployment {name}, which is not {actual_label}"
        )
    for name in sorted(set(actual) - set(snapshot)):
        problems.append(f"{actual_label} Deployment {name} is not in the snapshot")
    for deployment_name in sorted(set(snapshot) & set(actual)):
        expected_containers = snapshot[deployment_name]
        actual_containers = actual[deployment_name]
        for container_name in sorted(set(expected_containers) - set(actual_containers)):
            problems.append(
                f"{deployment_name} has no {actual_label} container named "
                f"{container_name}"
            )
        for container_name in sorted(set(actual_containers) - set(expected_containers)):
            problems.append(
                f"{deployment_name} {actual_label} container {container_name} "
                "is not in the snapshot"
            )
        for container_name in sorted(set(expected_containers) & set(actual_containers)):
            problems.extend(
                _container_differences(
                    f"{deployment_name}/{container_name}",
                    normalised_container_env(expected_containers[container_name]),
                    normalised_container_env(actual_containers[container_name]),
                    actual_label,
                )
            )
    return problems


def _container_differences(
    label: str,
    expected: ContainerEnv,
    actual: ContainerEnv,
    actual_label: str,
) -> list[str]:
    problems: list[str] = []
    # Merge-walk the two name-sorted lists: the first position where they part
    # names either the entry one side lacks or the entry whose body changed.
    expected_env, actual_env = expected["env"], actual["env"]
    index = 0
    while index < max(len(expected_env), len(actual_env)):
        source = expected_env[index] if index < len(expected_env) else None
        live = actual_env[index] if index < len(actual_env) else None
        if source == live:
            index += 1
            continue
        source_name = None if source is None else str(source.get("name"))
        live_name = None if live is None else str(live.get("name"))
        if live_name is None or (source_name is not None and source_name < live_name):
            problems.append(
                f"{label} env {source_name} is in the snapshot but not {actual_label}"
            )
        elif source_name is None or live_name < source_name:
            problems.append(
                f"{label} env {live_name} is {actual_label} but not in the snapshot"
            )
        else:
            problems.append(f"{label} env {source_name} differs from the snapshot")
        break
    expected_from, actual_from = expected["envFrom"], actual["envFrom"]
    for position in range(max(len(expected_from), len(actual_from))):
        source = expected_from[position] if position < len(expected_from) else None
        live = actual_from[position] if position < len(actual_from) else None
        if source != live:
            problems.append(
                f"{label} envFrom[{position}] differs from the snapshot: "
                f"{actual_label} {json.dumps(live, sort_keys=True)}, "
                f"snapshot {json.dumps(source, sort_keys=True)}"
            )
            break
    return problems
