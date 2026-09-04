from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from gpu_fault.admin.config import AdminConfigError, AuroraCapacityConfig

CAPACITY_SETTLE_STABLE_POLLS = 3
"""Consecutive settled observations required after a capacity change.

``modify-db-cluster --apply-immediately`` answers before RDS moves the cluster:
the response still reads ``available`` and already reports the requested window,
so the very first poll can satisfy every readiness condition while the change has
not started. Scaling up happens to be shielded by the observed-ACU check, but
scaling down is not -- ``actual_acu`` is trivially above the new floor -- so a
single poll would hand a cluster that is about to flip to ``modifying`` on to the
release that follows. That is how a deploy once failed its own ``aurora``
preflight check a few hundred lines after scaling the cluster itself.
"""


def _aws_json(
    aws_region: str,
    *arguments: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    command = [
        "aws",
        *arguments,
        "--region",
        aws_region,
        "--output",
        "json",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        operation = " ".join(arguments[:2])
        raise AdminConfigError(f"Aurora capacity command failed: aws {operation}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AdminConfigError("Aurora capacity command returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise AdminConfigError("Aurora capacity command returned a non-object")
    return value


def _first(items: object, description: str) -> dict[str, Any]:
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise AdminConfigError(f"{description} must resolve to exactly one object")
    return items[0]


def _latest_capacity(
    aws_region: str,
    instance_id: str,
) -> tuple[float, str] | None:
    end = datetime.now(UTC)
    document = _aws_json(
        aws_region,
        "cloudwatch",
        "get-metric-statistics",
        "--namespace",
        "AWS/RDS",
        "--metric-name",
        "ServerlessDatabaseCapacity",
        "--dimensions",
        f"Name=DBInstanceIdentifier,Value={instance_id}",
        "--start-time",
        (end - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time",
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--period",
        "60",
        "--statistics",
        "Maximum",
    )
    points = document.get("Datapoints") or []
    if not isinstance(points, list):
        raise AdminConfigError("Aurora capacity metric datapoints are invalid")
    valid = []
    for item in points:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("Timestamp"), str)
            or not isinstance(item.get("Maximum"), (int, float))
        ):
            continue
        try:
            timestamp = datetime.fromisoformat(
                str(item["Timestamp"]).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            continue
        valid.append((timestamp, float(item["Maximum"])))
    if not valid:
        return None
    timestamp, maximum = max(valid, key=lambda item: item[0])
    return maximum, timestamp.astimezone(UTC).isoformat()


def observe_aurora_capacity(
    *,
    aws_region: str,
    cluster_id: str,
    include_observed_capacity: bool,
) -> dict[str, Any]:
    cluster = _first(
        _aws_json(
            aws_region,
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            cluster_id,
        ).get("DBClusters"),
        "Aurora cluster lookup",
    )
    scaling = cluster.get("ServerlessV2ScalingConfiguration") or {}
    if not isinstance(scaling, dict):
        raise AdminConfigError("Aurora Serverless v2 scaling configuration is invalid")
    members = cluster.get("DBClusterMembers") or []
    if not isinstance(members, list) or len(members) != 2:
        raise AdminConfigError(
            "managed Aurora must contain exactly one writer and one reader"
        )
    instances = []
    for member in members:
        if not isinstance(member, dict):
            raise AdminConfigError("Aurora cluster member is invalid")
        instance_id = member.get("DBInstanceIdentifier")
        if not isinstance(instance_id, str) or not instance_id:
            raise AdminConfigError("Aurora cluster member has no instance identifier")
        instance = _first(
            _aws_json(
                aws_region,
                "rds",
                "describe-db-instances",
                "--db-instance-identifier",
                instance_id,
            ).get("DBInstances"),
            "Aurora instance lookup",
        )
        sample = (
            _latest_capacity(aws_region, instance_id)
            if include_observed_capacity
            else None
        )
        instances.append(
            {
                "role": "writer" if member.get("IsClusterWriter") else "reader",
                "status": instance.get("DBInstanceStatus"),
                "class": instance.get("DBInstanceClass"),
                "actual_acu": sample[0] if sample is not None else None,
                "actual_acu_observed_at": sample[1] if sample is not None else None,
            }
        )
    instances.sort(key=lambda item: str(item["role"]))
    if [item["role"] for item in instances] != ["reader", "writer"]:
        raise AdminConfigError(
            "managed Aurora must contain exactly one writer and one reader"
        )
    return {
        "cluster_status": cluster.get("Status"),
        "min_acu": scaling.get("MinCapacity"),
        "max_acu": scaling.get("MaxCapacity"),
        "instances": instances,
    }


def _configured_capacity(state: dict[str, Any]) -> tuple[float, float]:
    try:
        return float(state["min_acu"]), float(state["max_acu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdminConfigError("Aurora scaling configuration is incomplete") from exc


def _ready(
    state: dict[str, Any],
    desired: AuroraCapacityConfig,
    *,
    require_observed_capacity: bool,
) -> bool:
    instances = state.get("instances")
    if not isinstance(instances, list) or len(instances) != 2:
        return False
    if state.get("cluster_status") != "available" or _configured_capacity(state) != (
        desired.min_acu,
        desired.max_acu,
    ):
        return False
    for instance in instances:
        if (
            not isinstance(instance, dict)
            or instance.get("status") != "available"
            or instance.get("class") != "db.serverless"
        ):
            return False
        if require_observed_capacity and float(instance.get("actual_acu") or 0) < (
            desired.min_acu
        ):
            return False
        if require_observed_capacity:
            observed_at = instance.get("actual_acu_observed_at")
            if not isinstance(observed_at, str):
                return False
            try:
                timestamp = datetime.fromisoformat(observed_at)
            except ValueError:
                return False
            age = datetime.now(UTC) - timestamp.astimezone(UTC)
            if age < timedelta(minutes=-1) or age > timedelta(minutes=3):
                return False
    return True


def reconcile_aurora_capacity(
    *,
    aws_region: str,
    cluster_id: str,
    expected: AuroraCapacityConfig,
    desired: AuroraCapacityConfig,
    timeout_seconds: int = 1800,
    poll_seconds: float = 10.0,
) -> dict[str, Any]:
    expected.validate()
    desired.validate()
    require_observed_capacity = desired.min_acu > expected.min_acu
    initial = observe_aurora_capacity(
        aws_region=aws_region,
        cluster_id=cluster_id,
        include_observed_capacity=False,
    )
    live_capacity = _configured_capacity(initial)
    accepted = {
        (expected.min_acu, expected.max_acu),
        (desired.min_acu, desired.max_acu),
    }
    if live_capacity not in accepted:
        raise AdminConfigError(
            "live Aurora capacity differs from both the reviewed current and "
            "desired administrator configuration"
        )
    modified = live_capacity != (desired.min_acu, desired.max_acu)
    if modified:
        _aws_json(
            aws_region,
            "rds",
            "modify-db-cluster",
            "--db-cluster-identifier",
            cluster_id,
            "--serverless-v2-scaling-configuration",
            f"MinCapacity={desired.min_acu:g},MaxCapacity={desired.max_acu:g}",
            "--apply-immediately",
        )
    # A cluster nobody touched is already settled, so one confirming observation
    # answers the question; after a modify it cannot.
    required_stable = CAPACITY_SETTLE_STABLE_POLLS if modified else 1
    deadline = time.monotonic() + timeout_seconds
    last = initial
    stable = 0
    while time.monotonic() < deadline:
        last = observe_aurora_capacity(
            aws_region=aws_region,
            cluster_id=cluster_id,
            include_observed_capacity=require_observed_capacity,
        )
        stable = (
            stable + 1
            if _ready(
                last,
                desired,
                require_observed_capacity=require_observed_capacity,
            )
            else 0
        )
        if stable >= required_stable:
            return {
                "before": initial,
                "after": last,
                "modified": modified,
            }
        time.sleep(poll_seconds)
    raise AdminConfigError(
        "Aurora capacity did not converge before the administrator timeout: "
        + json.dumps(last, sort_keys=True)
    )


def aurora_capacity_changed(
    current: AuroraCapacityConfig,
    desired: AuroraCapacityConfig,
) -> bool:
    return current != desired
