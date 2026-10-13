"""Where a capacity run's measured components ran: Aurora writer and CPU nodes.

Every statement the control plane sends crosses from its node's Availability
Zone to the Aurora writer's, and the writer moves on every failover, so a run's
latencies cannot be compared without both. The burst suite used to read
CloudWatch for ``GPU_FAULT_AURORA_INSTANCE`` and fall back to a default
instance name that was not the site's writer at all, and no run recorded the
writer. Each run now resolves the writer from the Aurora cluster it names and
records the writer instance, its AZ and the control-plane Pods' node AZs in
``run.json`` at the start (``start_placement``) and again in ``summary.json``
at the end (``placement_section``).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
else:
    import regional_capacity_registry as registry

CONTROL_POD_PREFIXES = (
    "gpu-fault-api-ha-",
    "gpu-fault-control-worker-",
    "gpu-fault-telemetry-spool-worker-",
)
ZONE_LABEL = "topology.kubernetes.io/zone"
AURORA_CLUSTER_ENV = "GPU_FAULT_AURORA_CLUSTER_ID"
# What a best-effort read survives: a failed or timed-out command, unparsable
# output, or a description without the expected fields.
READ_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    LookupError,
    TypeError,
    subprocess.SubprocessError,
)


def _first(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        raise RuntimeError(f"{what} returned nothing")
    return value[0]


def aurora_writer(
    aws_json: Callable[..., dict[str, Any]], *, cluster_id: str, aws_region: str
) -> dict[str, Any]:
    """The cluster's writer instance and every member's Availability Zone."""

    cluster = _first(
        aws_json(
            "rds",
            "describe-db-clusters",
            "--region",
            aws_region,
            "--db-cluster-identifier",
            cluster_id,
        ).get("DBClusters"),
        f"Aurora cluster {cluster_id}",
    )
    members = [item for item in cluster.get("DBClusterMembers") or [] if item]
    writers = [
        str(item["DBInstanceIdentifier"])
        for item in members
        if item.get("IsClusterWriter")
    ]
    if len(writers) != 1:
        raise RuntimeError(
            f"Aurora cluster {cluster_id} names {len(writers)} writer instances"
        )
    instances = {}
    for member in members:
        instance_id = str(member["DBInstanceIdentifier"])
        instance = _first(
            aws_json(
                "rds",
                "describe-db-instances",
                "--region",
                aws_region,
                "--db-instance-identifier",
                instance_id,
            ).get("DBInstances"),
            f"Aurora instance {instance_id}",
        )
        instances[instance_id] = {
            "role": "writer" if member.get("IsClusterWriter") else "reader",
            "availability_zone": instance.get("AvailabilityZone"),
        }
    return {
        "cluster_id": cluster_id,
        "writer_instance_id": writers[0],
        "writer_availability_zone": instances[writers[0]]["availability_zone"],
        "instances": dict(sorted(instances.items())),
    }


def control_plane_zones(
    control_fn: Callable[..., str], *, prefixes: tuple[str, ...]
) -> dict[str, Any]:
    """Best effort: each control-plane Pod's node and that node's zone label."""

    try:
        pods = json.loads(
            control_fn("get", "pod", "-o", "json", check=False, timeout=60)
        )
        nodes = json.loads(
            control_fn("get", "nodes", "-o", "json", check=False, timeout=60)
        )
    except READ_ERRORS as exc:
        return {"recorded": False, "error": type(exc).__name__}
    zones = {
        str((item.get("metadata") or {}).get("name")): (
            (item.get("metadata") or {}).get("labels") or {}
        ).get(ZONE_LABEL)
        for item in nodes.get("items") or []
    }
    placed: dict[str, dict[str, Any]] = {}
    for pod in pods.get("items") or []:
        metadata = pod.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name.startswith(prefixes):
            continue
        node = (pod.get("spec") or {}).get("nodeName")
        placed[name] = {
            "role": (metadata.get("labels") or {}).get("app"),
            "node": node,
            "zone": zones.get(str(node)),
        }
    by_zone: dict[str, dict[str, int]] = {}
    for item in placed.values():
        roles = by_zone.setdefault(str(item["zone"]), {})
        roles[str(item["role"])] = roles.get(str(item["role"]), 0) + 1
    return {
        "recorded": bool(placed) and any(item["zone"] for item in placed.values()),
        "pods": dict(sorted(placed.items())),
        "zones": {
            zone: dict(sorted(roles.items())) for zone, roles in sorted(by_zone.items())
        },
    }


def cli_aws_json(*arguments: str) -> dict[str, Any]:
    """One read-only ``aws`` call through the registry's checked runner."""

    completed = registry.run(["aws", *arguments, "--output", "json"], timeout=120)
    document: dict[str, Any] = json.loads(completed.stdout)
    return document


def run_placement(aurora_cluster_id: str) -> dict[str, Any]:
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "aurora": aurora_writer(
            cli_aws_json, cluster_id=aurora_cluster_id, aws_region=registry.AWS_REGION
        ),
        "control_plane": control_plane_zones(
            registry.control, prefixes=CONTROL_POD_PREFIXES
        ),
    }


def start_placement(aurora_cluster_id: str, artifacts: Path) -> dict[str, Any]:
    """Resolve the run's placement before its load and add it to run.json.

    An unresolvable writer fails the run here, before any load: its Aurora
    evidence would otherwise describe some other instance.
    """

    placement = run_placement(aurora_cluster_id)
    path = artifacts / "run.json"
    document = json.loads(path.read_text())
    document["placement"] = placement
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return placement


def placement_section(start: dict[str, Any], aurora_cluster_id: str) -> dict[str, Any]:
    """summary.json's placement: start, end and whether the writer moved."""

    try:
        end = run_placement(aurora_cluster_id)
    except READ_ERRORS as exc:
        end = {"error": f"{type(exc).__name__}: {exc}"}
    writer = (end.get("aurora") or {}).get("writer_instance_id")
    return {
        "start": start,
        "end": end,
        "aurora_writer_changed": (
            None if writer is None else writer != start["aurora"]["writer_instance_id"]
        ),
    }


def optional_start_placement(artifacts: Path) -> dict[str, Any] | None:
    """``start_placement`` when the round names its Aurora cluster, else ``None``.

    For a suite that reads no Aurora metrics of its own (the action suite): the
    placement is evidence there, so a round without the cluster still runs and
    its summary says the placement was not recorded.
    """

    cluster_id = os.getenv(AURORA_CLUSTER_ENV, "")
    return start_placement(cluster_id, artifacts) if cluster_id else None


def optional_placement_section(start: dict[str, Any] | None) -> dict[str, Any]:
    if start is None:
        return {"recorded": False, "reason": f"{AURORA_CLUSTER_ENV} is not set"}
    return placement_section(start, str(start["aurora"]["cluster_id"]))
