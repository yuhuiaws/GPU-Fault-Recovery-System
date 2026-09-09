"""The one place that writes an Aurora Serverless v2 capacity window.

Two RDS calls set the ACU window of the control-plane database: the
``create-db-cluster`` that brings the cluster up and the ``modify-db-cluster``
that moves an existing one. Both live here, and only here. Bootstrap, the
legacy ``deploy/hyperpod/deploy.sh`` (through ``python -m`` below), the perf
harness and the administrator's ``config apply`` all call in; none of them
spells the flag itself. The repository test
``test_the_scaling_flag_is_written_from_exactly_one_module`` holds that line.

The reason is an incident, not tidiness: a rollback once wrote 0.5/8 ACU through
a path the deploy did not know about, and the deploy could not correct it because
its own path compared against a different record. Five writers in three trees and
two languages is how that happens.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, Sequence

from gpu_fault.admin.config import AdminConfigError, AuroraCapacityConfig

AURORA_ENGINE_VERSION = "16.8"
"""Engine version every new control-plane cluster is created on.

One constant, read by bootstrap and by the legacy deploy script's ``create``
entry; the parameter group family is derived from it, so the two cannot drift
apart.
"""

CONTROL_PLANE_APPLICATION_TAG = "Key=Application,Value=gpu-fault-control-plane"
"""The console-filter tag deploy.sh always wrote; bootstrap adds the site tag."""

SITE_TAG_KEY = "gpu-fault:site-id"

SERVERLESS_V2_SCALING_FLAG = "--serverless-v2-scaling-configuration"

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


class AwsJsonCaller(Protocol):
    """``aws <arguments> --region R --output json`` as a parsed object.

    ``mutate`` marks the calls that change AWS state so a dry-run runner can hold
    them back; read-only callers ignore it. ``CommandRunner.aws_json`` satisfies
    this directly.
    """

    def __call__(
        self,
        region: str,
        /,
        *arguments: str,
        mutate: bool = False,
    ) -> dict[str, Any]: ...


def _aws_json(
    aws_region: str,
    /,
    *arguments: str,
    mutate: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    del mutate  # every call here is live; the flag is for injected runners
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
        detail = completed.stderr.strip()
        raise AdminConfigError(
            f"Aurora capacity command failed: aws {operation}"
            + (f": {detail}" if detail else "")
        )
    try:
        value = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AdminConfigError("Aurora capacity command returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise AdminConfigError("Aurora capacity command returned a non-object")
    return value


def _caller(aws_json: AwsJsonCaller | None) -> AwsJsonCaller:
    # Resolved at call time so a test can swap the module-level ``_aws_json``.
    return aws_json if aws_json is not None else _aws_json


def scaling_configuration(capacity: AuroraCapacityConfig) -> str:
    capacity.validate()
    return f"MinCapacity={capacity.min_acu:g},MaxCapacity={capacity.max_acu:g}"


@dataclass(frozen=True)
class AuroraClusterSpec:
    """Everything ``create-db-cluster`` needs that differs between clusters.

    The rest -- engine, encryption, deletion protection, IAM auth, log export,
    seven-day backups, the managed master password -- is the same for every
    control-plane database and is not a parameter.
    """

    cluster_id: str
    subnet_group: str
    security_group_ids: tuple[str, ...]
    parameter_group: str
    capacity: AuroraCapacityConfig
    site_id: str | None = None
    engine_version: str = AURORA_ENGINE_VERSION
    database_name: str = "gpu_fault"
    master_username: str = "gpu_fault_admin"
    backup_retention_days: int = 7


def create_db_cluster_arguments(spec: AuroraClusterSpec) -> list[str]:
    """``aws rds create-db-cluster`` arguments, without ``aws``/region/output.

    Feed the list to ``CommandRunner.aws_json(region, *arguments, mutate=True)``
    or to ``_aws_json``; both append the region and output format.
    """

    if not spec.security_group_ids:
        raise AdminConfigError("an Aurora cluster needs at least one security group")
    tags = [CONTROL_PLANE_APPLICATION_TAG]
    if spec.site_id:
        tags.insert(0, f"Key={SITE_TAG_KEY},Value={spec.site_id}")
    return [
        "rds",
        "create-db-cluster",
        "--db-cluster-identifier",
        spec.cluster_id,
        "--engine",
        "aurora-postgresql",
        "--engine-version",
        spec.engine_version,
        "--engine-mode",
        "provisioned",
        "--database-name",
        spec.database_name,
        "--master-username",
        spec.master_username,
        "--manage-master-user-password",
        SERVERLESS_V2_SCALING_FLAG,
        scaling_configuration(spec.capacity),
        "--db-subnet-group-name",
        spec.subnet_group,
        "--vpc-security-group-ids",
        *spec.security_group_ids,
        "--storage-encrypted",
        "--backup-retention-period",
        str(spec.backup_retention_days),
        "--deletion-protection",
        "--copy-tags-to-snapshot",
        "--enable-iam-database-authentication",
        "--db-cluster-parameter-group-name",
        spec.parameter_group,
        "--enable-cloudwatch-logs-exports",
        "postgresql",
        "--tags",
        *tags,
    ]


def create_aurora_cluster(
    spec: AuroraClusterSpec,
    *,
    aws_region: str,
    aws_json: AwsJsonCaller | None = None,
) -> dict[str, Any]:
    """Issue the create and return the identifier and status RDS answered with."""

    document = _caller(aws_json)(
        aws_region,
        *create_db_cluster_arguments(spec),
        mutate=True,
    )
    cluster = document.get("DBCluster") or {}
    return {
        "cluster_id": str(cluster.get("DBClusterIdentifier") or spec.cluster_id),
        "status": cluster.get("Status"),
        "engine_version": str(cluster.get("EngineVersion") or spec.engine_version),
        "min_acu": spec.capacity.min_acu,
        "max_acu": spec.capacity.max_acu,
    }


def _first(items: object, description: str) -> dict[str, Any]:
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise AdminConfigError(f"{description} must resolve to exactly one object")
    return items[0]


def _latest_capacity(
    aws_json: AwsJsonCaller,
    aws_region: str,
    instance_id: str,
) -> tuple[float, str] | None:
    end = datetime.now(UTC)
    document = aws_json(
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
    aws_json: AwsJsonCaller | None = None,
) -> dict[str, Any]:
    caller = _caller(aws_json)
    cluster = _first(
        caller(
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
            caller(
                aws_region,
                "rds",
                "describe-db-instances",
                "--db-instance-identifier",
                instance_id,
            ).get("DBInstances"),
            "Aurora instance lookup",
        )
        sample = (
            _latest_capacity(caller, aws_region, instance_id)
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


def _await_ready(
    caller: AwsJsonCaller,
    *,
    aws_region: str,
    cluster_id: str,
    desired: AuroraCapacityConfig,
    require_observed_capacity: bool,
    required_stable: int,
    timeout_seconds: int,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    stable = 0
    while time.monotonic() < deadline:
        last = observe_aurora_capacity(
            aws_region=aws_region,
            cluster_id=cluster_id,
            include_observed_capacity=require_observed_capacity,
            aws_json=caller,
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
            return last
        time.sleep(poll_seconds)
    raise AdminConfigError(
        "Aurora capacity did not converge before the administrator timeout: "
        + json.dumps(last, sort_keys=True)
    )


def _modify_window(
    caller: AwsJsonCaller,
    *,
    aws_region: str,
    cluster_id: str,
    desired: AuroraCapacityConfig,
    expected: AuroraCapacityConfig | None,
) -> tuple[dict[str, Any], bool, bool]:
    """Observe, refuse drift, issue the modify; ``(before, modified, scale_up)``.

    ``expected`` is the reviewed current window of an apply: the live cluster
    must match it or ``desired`` (a resumed apply), anything else is somebody
    else's change and is refused. Callers without such a record -- bootstrap,
    the legacy deploy, the perf harness -- leave it ``None`` and reconcile
    whatever is live.
    """

    desired.validate()
    if expected is not None:
        expected.validate()
    initial = observe_aurora_capacity(
        aws_region=aws_region,
        cluster_id=cluster_id,
        include_observed_capacity=False,
        aws_json=caller,
    )
    live_capacity = _configured_capacity(initial)
    if expected is not None and live_capacity not in {
        (expected.min_acu, expected.max_acu),
        (desired.min_acu, desired.max_acu),
    }:
        raise AdminConfigError(
            "live Aurora capacity differs from both the reviewed current and "
            "desired administrator configuration"
        )
    # A scale-up is only done when the instances actually run at the new floor;
    # the reviewed record is the baseline when there is one (a resumed apply
    # still proves the ACU it asked for), the live window otherwise.
    baseline = expected.min_acu if expected is not None else live_capacity[0]
    modified = live_capacity != (desired.min_acu, desired.max_acu)
    if modified:
        caller(
            aws_region,
            "rds",
            "modify-db-cluster",
            "--db-cluster-identifier",
            cluster_id,
            SERVERLESS_V2_SCALING_FLAG,
            scaling_configuration(desired),
            "--apply-immediately",
            mutate=True,
        )
    return initial, modified, desired.min_acu > baseline


def request_aurora_capacity(
    *,
    aws_region: str,
    cluster_id: str,
    desired: AuroraCapacityConfig,
    expected: AuroraCapacityConfig | None = None,
    timeout_seconds: int = 1800,
    poll_seconds: float = 10.0,
    aws_json: AwsJsonCaller | None = None,
) -> dict[str, Any]:
    """Move the window to ``desired`` and wait until RDS reads ``available`` on it.

    The first half of a capacity change for a caller with work to do in
    between: the ``modify-db-cluster`` and the wait for the cluster and both
    instances to be ``available`` with the new window
    (``CAPACITY_SETTLE_STABLE_POLLS`` times after a modify). It does not wait
    for a scale-up's instances to actually run at the new floor; that is
    ``await_aurora_capacity``, which the administrator command runs after the
    control-plane roles have rolled, because the two are independent and the
    ACU ramp is the slow part. The structural wait has to happen here, though:
    the release's own preflight refuses an Aurora cluster that is not
    ``available``. The result's ``scale_up`` says whether
    ``await_aurora_capacity`` still has work to do.
    """

    caller = _caller(aws_json)
    initial, modified, scale_up = _modify_window(
        caller,
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        expected=expected,
    )
    # A cluster nobody touched is already settled, so one confirming observation
    # answers the question; after a modify it cannot.
    after = _await_ready(
        caller,
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        require_observed_capacity=False,
        required_stable=CAPACITY_SETTLE_STABLE_POLLS if modified else 1,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    return {
        "before": initial,
        "after": after,
        "modified": modified,
        "scale_up": scale_up,
    }


def await_aurora_capacity(
    *,
    aws_region: str,
    cluster_id: str,
    desired: AuroraCapacityConfig,
    modified: bool,
    timeout_seconds: int = 1800,
    poll_seconds: float = 10.0,
    aws_json: AwsJsonCaller | None = None,
) -> dict[str, Any]:
    """Wait until both instances actually run at ``desired.min_acu``.

    The second half of a scale-up, after ``request_aurora_capacity``: the
    cluster already reads ``available`` on the new window, this proves the
    ``ServerlessDatabaseCapacity`` metric reached the floor on the writer and
    the reader. ``modified`` says whether this apply issued the change, in
    which case the same consecutive-poll rule applies.
    """

    desired.validate()
    return _await_ready(
        _caller(aws_json),
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        require_observed_capacity=True,
        required_stable=CAPACITY_SETTLE_STABLE_POLLS if modified else 1,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def reconcile_aurora_capacity(
    *,
    aws_region: str,
    cluster_id: str,
    desired: AuroraCapacityConfig,
    expected: AuroraCapacityConfig | None = None,
    timeout_seconds: int = 1800,
    poll_seconds: float = 10.0,
    aws_json: AwsJsonCaller | None = None,
) -> dict[str, Any]:
    """Move the cluster to ``desired`` and wait until RDS has really done it.

    The whole change in one settle loop (structure and, for a scale-up, the
    observed ACU together), for callers with nothing to do in between:
    bootstrap, the legacy deploy, the perf harness and the administrator
    command's Aurora rollback. See ``_modify_window`` for ``expected``.
    """

    caller = _caller(aws_json)
    initial, modified, scale_up = _modify_window(
        caller,
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        expected=expected,
    )
    after = _await_ready(
        caller,
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        require_observed_capacity=scale_up,
        required_stable=CAPACITY_SETTLE_STABLE_POLLS if modified else 1,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    return {"before": initial, "after": after, "modified": modified}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fault.admin.aurora_capacity",
        description=(
            "Create or resize the control-plane Aurora Serverless v2 cluster. "
            "This is the only writer of the ACU window; shell callers use it "
            "instead of spelling the RDS flags themselves."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("create", "create the cluster (the caller has checked it is absent)"),
        ("reconcile", "move an existing cluster to the window and wait to settle"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--region", required=True)
        command.add_argument("--cluster-id", required=True)
        command.add_argument("--min-acu", type=float, required=True)
        command.add_argument("--max-acu", type=float, required=True)
        if name == "create":
            command.add_argument("--subnet-group", required=True)
            command.add_argument(
                "--security-group",
                action="append",
                required=True,
                dest="security_groups",
            )
            command.add_argument("--parameter-group", required=True)
            command.add_argument("--site-id", default=None)
            command.add_argument("--engine-version", default=AURORA_ENGINE_VERSION)
        else:
            command.add_argument("--timeout-seconds", type=int, default=1800)
            command.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    capacity = AuroraCapacityConfig(
        min_acu=float(arguments.min_acu), max_acu=float(arguments.max_acu)
    )
    try:
        if arguments.command == "create":
            result = create_aurora_cluster(
                AuroraClusterSpec(
                    cluster_id=arguments.cluster_id,
                    subnet_group=arguments.subnet_group,
                    security_group_ids=tuple(arguments.security_groups),
                    parameter_group=arguments.parameter_group,
                    capacity=capacity,
                    site_id=arguments.site_id,
                    engine_version=arguments.engine_version,
                ),
                aws_region=arguments.region,
            )
        else:
            result = reconcile_aurora_capacity(
                aws_region=arguments.region,
                cluster_id=arguments.cluster_id,
                desired=capacity,
                timeout_seconds=arguments.timeout_seconds,
                poll_seconds=arguments.poll_seconds,
            )
    except AdminConfigError as exc:
        print(f"aurora_capacity {arguments.command}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
