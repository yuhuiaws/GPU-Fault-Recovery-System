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

# What the control plane needs the database to record about itself (store
# review 2026-09-07, item K). The store retries deadlocks (40P01) silently and
# the default Aurora parameter group logs neither lock waits nor slow
# statements, so a locking defect is invisible in production; pg_stat_statements
# is the only per-statement view of where the ACU budget goes. Three of these
# are dynamic and take effect as soon as the group is attached; the preload
# library is static and waits for a writer reboot the operator schedules
# (`管理员日常运维.md` §4.6) -- until then the cluster reads ``pending-reboot``,
# which is expected, not a defect.
DIAGNOSTIC_PARAMETERS: tuple[tuple[str, str, str], ...] = (
    ("log_lock_waits", "1", "immediate"),
    ("log_min_duration_statement", "1000", "immediate"),
    ("pg_stat_statements.track", "all", "immediate"),
    ("shared_preload_libraries", "pg_stat_statements", "pending-reboot"),
)
POSTGRESQL_LOG_EXPORT = "postgresql"


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


def cluster_parameter_group_name(
    cluster_id: str, *, safe_name: Callable[..., str]
) -> str:
    # A group per cluster: the default group cannot be modified, and sharing
    # one between sites would let one site's change reach another's database.
    return safe_name(f"{cluster_id}-pg", maximum=255)


def ensure_cluster_parameter_group(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    engine_version: str,
    safe_name: Callable[..., str],
) -> str:
    """Create the cluster parameter group when missing and hold its parameters
    at the diagnostic values; returns the group name.

    Read-compare-write per parameter so a routine deploy against a settled
    group issues no modification at all (``modify-db-cluster-parameter-group``
    is a mutation even when nothing changes), and so an operator's deliberate
    change to any other parameter in the group is left alone.
    """

    group = cluster_parameter_group_name(cluster_id, safe_name=safe_name)
    exists = (
        subprocess.run(
            [
                "aws",
                "rds",
                "describe-db-cluster-parameter-groups",
                "--region",
                aws_region,
                "--db-cluster-parameter-group-name",
                group,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not exists:
        versions = (
            runner.aws_json(
                aws_region,
                "rds",
                "describe-db-engine-versions",
                "--engine",
                "aurora-postgresql",
                "--engine-version",
                engine_version,
            ).get("DBEngineVersions")
            or []
        )
        family = str(
            (versions[0] if versions else {}).get("DBParameterGroupFamily") or ""
        )
        if not family:
            raise BootstrapError(
                f"cannot resolve the parameter group family for aurora-postgresql "
                f"{engine_version}"
            )
        runner.run(
            [
                "aws",
                "rds",
                "create-db-cluster-parameter-group",
                "--region",
                aws_region,
                "--db-cluster-parameter-group-name",
                group,
                "--db-parameter-group-family",
                family,
                "--description",
                f"gpu-fault control plane diagnostics for {cluster_id}",
            ],
            mutate=True,
            capture=False,
        )
        current: dict[str, str] = {}
    else:
        names = ", ".join(
            f"'{name}'" for name, _value, _method in DIAGNOSTIC_PARAMETERS
        )
        current = {
            str(item.get("ParameterName")): str(item.get("ParameterValue") or "")
            for item in runner.aws_json(
                aws_region,
                "rds",
                "describe-db-cluster-parameters",
                "--db-cluster-parameter-group-name",
                group,
                "--query",
                f"Parameters[?contains([{names}], ParameterName)]",
            ).get("Parameters")
            or []
        }
    if runner.dry_run and not exists:
        return group
    drifted = [
        f"ParameterName={name},ParameterValue={value},ApplyMethod={method}"
        for name, value, method in DIAGNOSTIC_PARAMETERS
        if current.get(name) != value
    ]
    if drifted:
        runner.run(
            [
                "aws",
                "rds",
                "modify-db-cluster-parameter-group",
                "--region",
                aws_region,
                "--db-cluster-parameter-group-name",
                group,
                "--parameters",
                *drifted,
            ],
            mutate=True,
            capture=False,
        )
    return group


def reconcile_cluster_diagnostics(
    runner: CommandRunner,
    *,
    aws_region: str,
    cluster_id: str,
    cluster: dict[str, Any],
    parameter_group: str,
) -> bool:
    """Attach the parameter group and the PostgreSQL log export to an existing
    cluster when either is missing; returns whether anything was changed.

    Both are online changes: no instance restarts, no connection drops. The
    static ``shared_preload_libraries`` becomes ``pending-reboot`` and stays so
    until the operator's reboot window.
    """

    arguments: list[str] = []
    if str(cluster.get("DBClusterParameterGroup") or "") != parameter_group:
        arguments.extend(["--db-cluster-parameter-group-name", parameter_group])
    exports = [str(item) for item in cluster.get("EnabledCloudwatchLogsExports") or []]
    if POSTGRESQL_LOG_EXPORT not in exports:
        arguments.extend(
            [
                "--cloudwatch-logs-export-configuration",
                f'{{"EnableLogTypes":["{POSTGRESQL_LOG_EXPORT}"]}}',
            ]
        )
    if not arguments:
        return False
    runner.run(
        [
            "aws",
            "rds",
            "modify-db-cluster",
            "--region",
            aws_region,
            "--db-cluster-identifier",
            cluster_id,
            *arguments,
            "--apply-immediately",
        ],
        mutate=True,
        capture=False,
    )
    if not runner.dry_run:
        # The modify answers before the cluster flips to ``modifying``; the
        # same settle wait the capacity change needs (see above).
        wait_for_capacity_to_settle(
            runner, aws_region=aws_region, cluster_id=cluster_id
        )
    return True
