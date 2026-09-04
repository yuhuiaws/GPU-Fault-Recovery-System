from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.config import AuroraCapacityConfig, load_desired_admin_config

CAPACITY_SETTLE_POLL_SECONDS = 10.0
CAPACITY_SETTLE_STABLE_POLLS = 3
CAPACITY_SETTLE_TIMEOUT_SECONDS = 1800.0


def bootstrap_aurora_capacity(state_dir: Path) -> AuroraCapacityConfig:
    return load_desired_admin_config(
        state_dir,
        migrate_legacy=True,
    ).aurora


def scaling_configuration(capacity: AuroraCapacityConfig) -> str:
    capacity.validate()
    return f"MinCapacity={capacity.min_acu:g},MaxCapacity={capacity.max_acu:g}"


def reconcile_existing_capacity(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    cluster: dict[str, Any],
    capacity: AuroraCapacityConfig,
) -> None:
    scaling = cluster.get("ServerlessV2ScalingConfiguration") or {}
    live_capacity = (
        float(scaling.get("MinCapacity") or 0),
        float(scaling.get("MaxCapacity") or 0),
    )
    if live_capacity == (capacity.min_acu, capacity.max_acu):
        return
    runner.run(
        [
            "aws",
            "rds",
            "modify-db-cluster",
            "--region",
            aws_region,
            "--db-cluster-identifier",
            cluster_id,
            "--serverless-v2-scaling-configuration",
            scaling_configuration(capacity),
            "--apply-immediately",
        ],
        mutate=True,
        capture=False,
    )
    if runner.dry_run:
        return
    wait_for_capacity_to_settle(
        runner,
        aws_region=aws_region,
        cluster_id=cluster_id,
    )


def wait_for_capacity_to_settle(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    poll_seconds: float = CAPACITY_SETTLE_POLL_SECONDS,
    timeout_seconds: float = CAPACITY_SETTLE_TIMEOUT_SECONDS,
) -> None:
    """Block until the cluster and every member instance read ``available``.

    ``modify-db-cluster --apply-immediately`` answers before RDS moves the
    cluster: the response still says ``available`` and already reports the new
    scaling configuration. So neither ``rds wait db-cluster-available`` nor a
    configuration comparison can tell "settled" from "not started yet" -- both
    return instantly and the cluster flips to ``modifying`` seconds later, which
    is how a capacity change made the very same deployment fail its own
    ``aurora`` preflight check. Require several consecutive quiet polls instead,
    so a late flip is still caught.
    """

    deadline = time.monotonic() + timeout_seconds
    stable = 0
    while True:
        quiet = _capacity_is_quiet(
            runner,
            aws_region=aws_region,
            cluster_id=cluster_id,
        )
        stable = stable + 1 if quiet else 0
        if stable >= CAPACITY_SETTLE_STABLE_POLLS:
            return
        if time.monotonic() >= deadline:
            raise BootstrapError(
                f"Aurora cluster {cluster_id} did not settle within "
                f"{timeout_seconds:g}s of the capacity change"
            )
        time.sleep(poll_seconds)


def _capacity_is_quiet(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
) -> bool:
    cluster = runner.aws_json(
        aws_region,
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        cluster_id,
    )["DBClusters"][0]
    if str(cluster.get("Status") or "") != "available":
        return False
    instances = (
        runner.aws_json(
            aws_region,
            "rds",
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster_id}",
        ).get("DBInstances")
        or []
    )
    return all(
        str(instance.get("DBInstanceStatus") or "") == "available"
        for instance in instances
    )


def ensure_serverless_instances(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    availability_zones: list[str],
    safe_name: Callable[..., str],
) -> list[str]:
    instance_ids = [
        safe_name(f"{cluster_id}-{suffix}", maximum=63)
        for suffix in ("writer", "reader")
    ]
    for instance_id, availability_zone in zip(
        instance_ids,
        availability_zones,
        strict=True,
    ):
        exists = (
            subprocess.run(
                [
                    "aws",
                    "rds",
                    "describe-db-instances",
                    "--region",
                    aws_region,
                    "--db-instance-identifier",
                    instance_id,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if exists:
            continue
        runner.run(
            [
                "aws",
                "rds",
                "create-db-instance",
                "--region",
                aws_region,
                "--db-instance-identifier",
                instance_id,
                "--db-cluster-identifier",
                cluster_id,
                "--engine",
                "aurora-postgresql",
                "--db-instance-class",
                "db.serverless",
                "--availability-zone",
                availability_zone,
                "--promotion-tier",
                "0",
            ],
            mutate=True,
            capture=False,
        )
    for instance_id in instance_ids:
        runner.run(
            [
                "aws",
                "rds",
                "wait",
                "db-instance-available",
                "--region",
                aws_region,
                "--db-instance-identifier",
                instance_id,
            ],
            mutate=True,
            capture=False,
        )
    return instance_ids
