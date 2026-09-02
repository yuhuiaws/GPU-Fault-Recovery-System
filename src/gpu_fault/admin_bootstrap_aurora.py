from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin_bootstrap_common import CommandRunner
from gpu_fault.admin_config import AuroraCapacityConfig, load_desired_admin_config


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
